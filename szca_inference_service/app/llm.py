"""LLM via an ONNX causal LM with KV-cache autoregressive decode.

Model-agnostic: geometry (layers / KV-heads / head-dim / EOS) is read from the
model's config.json, and the chat template is chosen by architecture family, so
the SAME loop serves Llama-3.2 and Qwen2.5 (and future swaps) without code edits.

    messages -> family chat template -> token ids
      -> ONNX forward (input_ids + attention_mask + position_ids + past_kv)
      -> greedy argmax  [or temperature / top-p / top-k / repetition-penalty]
      -> feed present.* back as past_key_values.*, append token, repeat
      -> detokenize (until EOS / max_new_tokens)

Config is discovered from files in the model dir:
  * an ONNX file (LLM_ONNX_FILE, else the first *.onnx),
  * a tokenizer.json (LLM_TOKENIZER_FILE, else the first *tokenizer*.json),
  * an optional config.json / generation_config.json for geometry + EOS.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


def _first(model_dir: str, env: str, *patterns: str) -> Optional[str]:
    """Resolve a file: explicit env var wins, else first glob match."""
    override = os.environ.get(env)
    if override:
        path = override if os.path.isabs(override) else os.path.join(model_dir, override)
        return path if os.path.exists(path) else None
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(model_dir, pat)))
        if hits:
            return hits[0]
    return None


class LlmEngine:
    def __init__(self, model_dir: str, intra_op_threads: int = 0):
        onnx_path = _first(model_dir, "LLM_ONNX_FILE", "*.onnx")
        tok_path = _first(model_dir, "LLM_TOKENIZER_FILE", "*tokenizer*.json")
        if not onnx_path or not tok_path:
            raise FileNotFoundError(
                f"LLM model/tokenizer not found in {model_dir} "
                f"(onnx={onnx_path}, tokenizer={tok_path})"
            )

        so = ort.SessionOptions()
        if intra_op_threads:
            so.intra_op_num_threads = intra_op_threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(tok_path)
        self._out_names = [o.name for o in self.session.get_outputs()]

        cfg = self._load_json(model_dir, "config.json")
        gen_cfg = self._load_json(model_dir, "generation_config.json")

        # KV-cache geometry. head_dim falls back to hidden/num_heads.
        self.n_layers = int(cfg.get("num_hidden_layers", 0)) or self._infer_layers()
        self.n_kv_heads = int(cfg.get("num_key_value_heads", cfg.get("num_attention_heads", 0)))
        head_dim = cfg.get("head_dim")
        if head_dim is None and cfg.get("hidden_size") and cfg.get("num_attention_heads"):
            head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]
        self.head_dim = int(head_dim or 64)

        eos = gen_cfg.get("eos_token_id", cfg.get("eos_token_id"))
        if isinstance(eos, int):
            eos = [eos]
        self.eos_ids = set(eos or [])

        self.family = str(cfg.get("model_type", "")).lower()
        # Sane sampling defaults taken from the model's own generation_config.
        self.default_top_k = int(gen_cfg.get("top_k", 0) or 0)
        self.default_rep_penalty = float(gen_cfg.get("repetition_penalty", 1.0) or 1.0)

    @staticmethod
    def _load_json(model_dir: str, name: str) -> dict:
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return {}

    def _infer_layers(self) -> int:
        """Count layers from the present.N.* output names if config is absent."""
        idxs = {int(n.split(".")[1]) for n in self._out_names if n.startswith("present.")}
        return (max(idxs) + 1) if idxs else 0

    def _empty_kv(self) -> Dict[str, np.ndarray]:
        return {
            f"past_key_values.{i}.{kv}": np.zeros(
                (1, self.n_kv_heads, 0, self.head_dim), dtype=np.float32
            )
            for i in range(self.n_layers)
            for kv in ("key", "value")
        }

    # ---- chat templates ----------------------------------------------------

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        if self.family == "qwen2":
            return self._qwen_prompt(messages)
        return self._llama_prompt(messages)

    def _qwen_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Qwen2.5 ChatML template. A default system message is prepended when
        the caller omits one (matches Qwen's reference tokenizer behavior)."""
        msgs = list(messages)
        if not msgs or msgs[0].get("role") != "system":
            msgs = [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}] + msgs
        parts = []
        for m in msgs:
            parts.append(f"<|im_start|>{m.get('role','user')}\n{m.get('content','')}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def _llama_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Llama-3.2 template. It ALWAYS emits a system block (with the date
        preamble) even when no system message is given; omitting it corrupts
        turn boundaries."""
        system_content = ""
        rest = messages
        if messages and messages[0].get("role") == "system":
            system_content = messages[0].get("content", "").strip()
            rest = messages[1:]
        system_body = "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n"
        if system_content:
            system_body += system_content
        parts = [
            "<|begin_of_text|>",
            f"<|start_header_id|>system<|end_header_id|>\n\n{system_body}<|eot_id|>",
        ]
        for m in rest:
            parts.append(
                f"<|start_header_id|>{m.get('role','user')}<|end_header_id|>\n\n"
                f"{m.get('content','')}<|eot_id|>"
            )
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(parts)

    # ---- sampling ----------------------------------------------------------

    def _sample(self, logits, temperature, top_p, top_k, generated, rep_penalty):
        logits = logits.astype(np.float64)
        # Repetition penalty over already-generated tokens.
        if rep_penalty and rep_penalty != 1.0 and generated:
            uniq = list(set(generated))
            pos = logits[uniq] > 0
            logits[uniq] = np.where(pos, logits[uniq] / rep_penalty, logits[uniq] * rep_penalty)

        if temperature <= 0.0:
            return int(np.argmax(logits))

        logits = logits / max(temperature, 1e-5)
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()

        order = np.argsort(probs)[::-1]
        if top_k and top_k > 0:
            order = order[:top_k]
        cumulative = np.cumsum(probs[order])
        keep = cumulative <= top_p
        keep[0] = True
        kept = order[keep]
        p = probs[kept]
        p /= p.sum()
        return int(np.random.choice(kept, p=p))

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
    ) -> Tuple[str, int]:
        prompt = self._build_prompt(messages)
        cur = self.tokenizer.encode(prompt, add_special_tokens=False).ids

        top_k = self.default_top_k if top_k is None else top_k
        rep_penalty = self.default_rep_penalty if repetition_penalty is None else repetition_penalty

        past = self._empty_kv()
        generated: List[int] = []

        for step in range(max_new_tokens):
            inp = np.array([cur if step == 0 else [cur[-1]]], dtype=np.int64)
            past_len = past["past_key_values.0.key"].shape[2]
            attn = np.ones((1, past_len + inp.shape[1]), dtype=np.int64)
            pos = np.arange(past_len, past_len + inp.shape[1], dtype=np.int64).reshape(1, -1)

            feeds = {"input_ids": inp, "attention_mask": attn, "position_ids": pos, **past}
            out = self.session.run(None, feeds)

            logits = out[0][0, -1]
            next_id = self._sample(logits, temperature, top_p, top_k, generated, rep_penalty)
            if next_id in self.eos_ids:
                break

            generated.append(next_id)
            cur.append(next_id)
            past = {
                f"past_key_values.{n.split('.')[1]}.{n.split('.')[2]}": out[i]
                for i, n in enumerate(self._out_names)
                if n.startswith("present.")
            }

        text = self.tokenizer.decode(generated)
        return text, len(generated)
