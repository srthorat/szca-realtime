"""Text-to-Speech via Kokoro-82M v1.0 (quantized ONNX) + Misaki G2P.

Real inference chain, no stubs:

    text
      -> Misaki G2P            (Kokoro's OFFICIAL grapheme->phoneme; IPA out)
      -> Kokoro tokenizer      (IPA char -> input_id, per the model's vocab)
      -> kokoro.onnx           (input_ids + style[1,256] + speed -> waveform)
      -> 24 kHz mono float waveform

Using Misaki (not espeak directly) matters: Kokoro was trained on Misaki's
phoneme alphabet, so this reproduces the exact tokens the model expects.
The voice is a style vector picked from the voice pack by phoneme-length index
(Kokoro voice packs are [510, 1, 256]; row = len(tokens)).
"""

from __future__ import annotations

import json
import os
from typing import List

import numpy as np
import onnxruntime as ort


class TtsEngine:
    def __init__(self, model_dir: str, voice: str = "af_heart", intra_op_threads: int = 0):
        so = ort.SessionOptions()
        if intra_op_threads:
            so.intra_op_num_threads = intra_op_threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            os.path.join(model_dir, "kokoro_v1.0_quantized.onnx"),
            so,
            providers=["CPUExecutionProvider"],
        )
        self.sample_rate = 24000

        # Phoneme -> id map from the Kokoro tokenizer.json vocab.
        with open(os.path.join(model_dir, "kokoro_tokenizer.json"), "r", encoding="utf-8") as fh:
            tok = json.load(fh)
        self.vocab = tok["model"]["vocab"]

        # Voice pack: [510, 1, 256] float32 style vectors, indexed by token len.
        voice_path = os.path.join(model_dir, "kokoro_voices", f"{voice}.bin")
        self.voice_pack = np.fromfile(voice_path, dtype=np.float32).reshape(-1, 1, 256)

        # Lazy G2P init (spacy load is ~1s); created on first synth.
        self._g2p = None

    def _g2p_engine(self):
        if self._g2p is None:
            from misaki import en

            self._g2p = en.G2P(trf=False, british=False, fallback=None)
        return self._g2p

    def _phonemes_to_ids(self, phonemes: str) -> List[int]:
        ids = [0]  # leading pad/BOS ($ == id 0 in Kokoro vocab)
        for ch in phonemes:
            if ch in self.vocab:
                ids.append(self.vocab[ch])
            # Unknown symbols are skipped rather than mapped to <unk>, matching
            # Kokoro's reference behavior (its normalizer strips them).
        ids.append(0)  # trailing pad
        return ids

    def synthesize(self, text: str, speed: float = 1.0) -> np.ndarray:
        """Return a 24 kHz mono float32 waveform for `text`."""
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32)

        phonemes, _ = self._g2p_engine()(text)
        input_ids = self._phonemes_to_ids(phonemes)

        # Kokoro caps at 510 phoneme tokens; the style row is chosen by length.
        n = min(len(input_ids), self.voice_pack.shape[0] - 1)
        style = self.voice_pack[n]  # [1, 256]

        ids = np.array([input_ids], dtype=np.int64)
        speed_arr = np.array([speed], dtype=np.float32)

        waveform = self.session.run(
            None,
            {"input_ids": ids, "style": style, "speed": speed_arr},
        )[0]
        return np.asarray(waveform, dtype=np.float32).reshape(-1)
