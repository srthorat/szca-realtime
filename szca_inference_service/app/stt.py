"""Speech-to-Text via Parakeet TDT 0.6B v3 (int8 ONNX).

Real inference chain, no stubs:

    waveform (16 kHz mono f32)
      -> nemo128.onnx          (log-mel feature frontend, exported from NeMo)
      -> encoder.int8.onnx     (Conformer acoustic encoder, 8x subsampling)
      -> decoder_joint.int8    (RNN-T/TDT prediction + joint network)
      -> TDT greedy decode     (token + duration jump)
      -> SentencePiece vocab   (detokenize, '_' -> space)

The frontend is a real ONNX graph, so NeMo's exact mel features are reproduced
bit-for-bit rather than hand-ported (the usual "runs but transcribes garbage"
trap). The decoder is a Token-and-Duration Transducer: each joint step emits a
token logit block AND a duration logit block; the duration tells us how many
encoder frames to skip, which is what makes TDT fast.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import onnxruntime as ort


# TDT/vocab geometry, derived from the model + vocab.txt:
#   vocab.txt has 8193 entries, last is <blk> (id 8192).
#   decoder_joint output width is 8198 = 8193 vocab logits + 5 duration logits.
VOCAB_SIZE = 8193
BLANK_ID = 8192
NUM_DURATIONS = 5            # TDT duration bins: [0,1,2,3,4] frames
PRED_STATE_DIM = 640         # LSTM hidden width of the prediction network
PRED_STATE_LAYERS = 2        # stacked LSTM layers -> state shape [2, 1, 640]
# Guardrails so a pathological clip cannot loop forever.
MAX_SYMBOLS_PER_STEP = 10    # emitted tokens before we force a frame advance


def _load_vocab(path: str) -> List[str]:
    """Parse `token id` lines into an index-ordered list."""
    table = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            # Format is "<token><space><id>"; split from the right so tokens
            # that themselves contain spaces are preserved.
            piece, _, idx = line.rpartition(" ")
            table[int(idx)] = piece
    return [table.get(i, "") for i in range(max(table) + 1)]


class SttEngine:
    """Loads the three Parakeet ONNX graphs and runs TDT greedy decoding."""

    def __init__(self, model_dir: str, intra_op_threads: int = 0):
        so = ort.SessionOptions()
        if intra_op_threads:
            so.intra_op_num_threads = intra_op_threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = ["CPUExecutionProvider"]
        self.mel = ort.InferenceSession(
            os.path.join(model_dir, "parakeet_nemo128.onnx"), so, providers=providers
        )
        self.encoder = ort.InferenceSession(
            os.path.join(model_dir, "parakeet_encoder.int8.onnx"), so, providers=providers
        )
        self.decoder = ort.InferenceSession(
            os.path.join(model_dir, "parakeet_decoder_joint.int8.onnx"), so, providers=providers
        )
        self.vocab = _load_vocab(os.path.join(model_dir, "parakeet_vocab.txt"))

    def _zero_state(self):
        shape = (PRED_STATE_LAYERS, 1, PRED_STATE_DIM)
        return (
            np.zeros(shape, dtype=np.float32),
            np.zeros(shape, dtype=np.float32),
        )

    def _decode_step(self, enc_frame: np.ndarray, target: int, state1, state2):
        """Run one joint step for `target` at a single encoder frame.

        enc_frame: [1, 1024, 1] single encoder timestep.
        Returns (token_logits[8193], duration_logits[5], new_state1, new_state2).
        """
        targets = np.array([[target]], dtype=np.int32)
        target_length = np.array([1], dtype=np.int32)
        outputs = self.decoder.run(
            None,
            {
                "encoder_outputs": enc_frame,
                "targets": targets,
                "target_length": target_length,
                "input_states_1": state1,
                "input_states_2": state2,
            },
        )
        # decoder_joint outputs: [logits, prednet_lengths, out_state_1, out_state_2]
        logits = outputs[0].reshape(-1)  # width 8198
        new_state1, new_state2 = outputs[2], outputs[3]
        token_logits = logits[:VOCAB_SIZE]
        duration_logits = logits[VOCAB_SIZE:VOCAB_SIZE + NUM_DURATIONS]
        return token_logits, duration_logits, new_state1, new_state2

    def transcribe(self, waveform: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe a 16 kHz mono float waveform to text."""
        if sample_rate != 16000:
            raise ValueError(f"STT expects 16 kHz audio, got {sample_rate}")
        wav = np.asarray(waveform, dtype=np.float32).reshape(1, -1)
        wav_lens = np.array([wav.shape[1]], dtype=np.int64)

        # 1. Log-mel features [1, 128, T].
        feats, feat_lens = self.mel.run(
            None, {"waveforms": wav, "waveforms_lens": wav_lens}
        )

        # 2. Conformer encoder -> [1, 1024, T'] (T' ~= T/8).
        enc_out, enc_lens = self.encoder.run(
            None, {"audio_signal": feats, "length": feat_lens.astype(np.int64)}
        )
        num_frames = int(enc_lens[0])

        # 3. TDT greedy decode over encoder frames.
        state1, state2 = self._zero_state()
        last_token = BLANK_ID  # prime the prediction net with blank
        hyp: List[int] = []

        t = 0
        while t < num_frames:
            enc_frame = enc_out[:, :, t:t + 1]  # [1,1024,1]
            emitted = 0
            advanced = False
            while emitted < MAX_SYMBOLS_PER_STEP:
                tok_logits, dur_logits, s1, s2 = self._decode_step(
                    enc_frame, last_token, state1, state2
                )
                token = int(np.argmax(tok_logits))
                duration = int(np.argmax(dur_logits))

                if token == BLANK_ID:
                    # Blank: advance time by the predicted duration (>=1) and
                    # do NOT update the prediction-net state.
                    t += max(duration, 1)
                    advanced = True
                    break

                # Real token: commit it, update state, jump `duration` frames.
                hyp.append(token)
                last_token = token
                state1, state2 = s1, s2
                emitted += 1
                if duration > 0:
                    t += duration
                    advanced = True
                    break
            if not advanced:
                # Emitted MAX symbols at duration 0 -> force progress.
                t += 1

        return self._detokenize(hyp)

    def _detokenize(self, ids: List[int]) -> str:
        pieces = []
        for i in ids:
            if 0 <= i < len(self.vocab):
                pieces.append(self.vocab[i])
        # SentencePiece: '▁' (U+2581) marks a leading space.
        text = "".join(pieces).replace("▁", " ").strip()
        return text
