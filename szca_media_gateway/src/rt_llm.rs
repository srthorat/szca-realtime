/// Real in-process LLM stage: an ONNX causal LM driven by a KV-cache
/// autoregressive decode loop, implementing [`crate::rt_pipeline::LlmStage`].
///
/// Model-family agnostic. Verified checkpoints: Qwen2.5-1.5B-Instruct (int8,
/// ChatML) and Hermes-3-Llama-3.2-3B (FP32 + external data, ChatML on a Llama
/// base). The prompt template and stop tokens are read from the checkpoint, not
/// hard-coded — see [`detect_chat_template`] and [`parse_eos`].
///
/// This is a faithful Rust port of the correctness oracle in
/// `szca_inference_service/app/llm.py`:
///
/// ```text
/// messages → ChatML template → token ids
///   → ONNX forward (input_ids + attention_mask + position_ids + past_kv)
///   → greedy argmax (with repetition penalty)
///   → feed present.* back as past_key_values.*, append token, repeat
///   → detokenize (until EOS / max_new_tokens)
/// ```
///
/// Geometry (layers / KV-heads / head-dim / EOS) is read from the model's
/// `config.json` + `generation_config.json` + `tokenizer_config.json`, so the
/// loop is not hard-wired to a single checkpoint.
///
/// Streaming: each newly generated token is decoded and the *incremental*
/// suffix is pushed through the `on_token` callback, so the WebSocket client
/// sees `response.output_text.delta` events as the reply forms. A trailing
/// Unicode replacement char (an as-yet-incomplete multi-byte sequence) is held
/// back until the following token completes it, so partial UTF-8 never leaks.
///
/// Cancellation (barge-in) is honored cooperatively: the loop checks the shared
/// `cancel` flag before every forward pass and returns what it has so far.

use std::collections::HashSet;
use std::sync::atomic::AtomicBool;

use ndarray::{Array2, Array4};
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::onnx::init_ort;
use crate::rt_pipeline::LlmStage;
use crate::stage_pool::Replica;

/// Default cap on generated tokens per turn.
const DEFAULT_MAX_NEW_TOKENS: usize = 256;

/// Default LLM model root. Holds ONE SUBDIRECTORY PER MODEL, because
/// `config.json` / `generation_config.json` / `tokenizer.json` are per-model and
/// would collide if two checkpoints shared a directory.
const DEFAULT_MODEL_DIR: &str = "./models/llm";

/// Chat prompt template, detected from the model's own Jinja `chat_template`
/// (see [`detect_chat_template`]) or forced via `LLM_CHAT_TEMPLATE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChatTemplate {
    /// `<|im_start|>role\n…<|im_end|>` — Qwen2/3 and ChatML fine-tunes of other
    /// bases (e.g. Hermes-3-Llama-3.2-3B).
    ChatML,
    /// `<|start_header_id|>role<|end_header_id|>\n\n…<|eot_id|>` — stock
    /// Llama-3.x Instruct.
    Llama,
}

/// Default system message when the caller supplies no `instructions`.
///
/// Deliberately model-agnostic: the same text is used for both templates so a
/// dev/prod backend swap can't silently change the assistant's persona. (An
/// earlier version hard-coded the Qwen persona here, which then leaked into
/// Hermes-3 replies.)
fn default_system(_template: ChatTemplate) -> &'static str {
    "You are a helpful, respectful and honest assistant. Keep spoken replies \
     brief and conversational."
}

/// ChatML special tokens (Qwen2/3, Hermes-3)
const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";

/// Llama-3.x Instruct chat template tokens
const LLAMA_BOS: &str = "<|begin_of_text|>";
const LLAMA_SYS_START: &str = "<|start_header_id|>system<|end_header_id|>\n\n";
const LLAMA_USER_START: &str = "<|start_header_id|>user<|end_header_id|>\n\n";
const LLAMA_ASST_START: &str = "<|start_header_id|>assistant<|end_header_id|>\n";
const LLAMA_EOT: &str = "<|eot_id|>";

/// Loaded ONNX causal-LM with the geometry needed to drive the KV cache.
/// Supports both Qwen2/3 (ChatML) and Llama-family (Hermes-3, Llama-3.x).
pub struct QwenLlm {
    session: Session,
    tokenizer: Tokenizer,
    /// Chat template detected from the checkpoint (or forced by env).
    chat_template: ChatTemplate,
    /// Number of transformer layers (KV-cache tensor pairs).
    n_layers: usize,
    /// Number of key/value heads (grouped-query attention).
    n_kv_heads: usize,
    /// Per-head dimension.
    head_dim: usize,
    /// End-of-sequence token ids; generation stops on any of them.
    eos_ids: HashSet<u32>,
    /// Repetition penalty applied to already-generated tokens (>1 discourages
    /// repeats). Taken from `generation_config.json`.
    rep_penalty: f32,
    /// Maximum tokens to generate per turn.
    max_new_tokens: usize,
}

/// Where to find the model on disk. Resolved from the environment by
/// [`QwenLlm::from_env`].
struct LlmPaths {
    onnx: String,
    tokenizer: String,
    config: Option<String>,
    gen_config: Option<String>,
    /// `tokenizer_config.json`: source of truth for the chat template + eos token.
    tok_config: Option<String>,
    chat_template: ChatTemplate,
}

impl QwenLlm {
    /// Load the LLM using environment configuration:
    ///   * `LLM_MODEL_DIR`      — model directory (default `./models/llm`)
    ///   * `LLM_ONNX_FILE`      — ONNX filename (default: first `*.onnx` in dir)
    ///   * `LLM_TOKENIZER_FILE` — tokenizer.json (default: first `*tokenizer*.json`)
    ///   * `LLM_MAX_NEW_TOKENS` — per-turn token cap (default 256)
    ///
    /// Returns an error (rather than panicking) when the model, tokenizer, or
    /// ONNX Runtime dylib is absent, so the caller can fall back to a stub.
    pub fn from_env() -> Result<Self, String> {
        let dir = std::env::var("LLM_MODEL_DIR")
            .ok()
            .filter(|d| !d.is_empty())
            .unwrap_or_else(|| DEFAULT_MODEL_DIR.to_string());
        let dir = descend_to_single_model(&dir);
        let paths = resolve_paths(&dir)?;
        let max_new = std::env::var("LLM_MAX_NEW_TOKENS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|n| *n > 0)
            .unwrap_or(DEFAULT_MAX_NEW_TOKENS);
        Self::load(&paths, max_new)
    }

    fn load(paths: &LlmPaths, max_new_tokens: usize) -> Result<Self, String> {
        init_ort()?;

        let session = Session::builder()
            .map_err(|e| format!("session builder: {e}"))?
            .commit_from_file(&paths.onnx)
            .map_err(|e| format!("load {}: {e}", paths.onnx))?;

        let tokenizer = Tokenizer::from_file(&paths.tokenizer)
            .map_err(|e| format!("load tokenizer {}: {e}", paths.tokenizer))?;

        // ---- geometry from config.json / generation_config.json ----
        let cfg = load_json(paths.config.as_deref());
        let gen_cfg = load_json(paths.gen_config.as_deref());

        let n_layers = cfg
            .get("num_hidden_layers")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .or_else(|| infer_layers(&session))
            .ok_or_else(|| "could not determine num_hidden_layers".to_string())?;

        let n_kv_heads = cfg
            .get("num_key_value_heads")
            .or_else(|| cfg.get("num_attention_heads"))
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .ok_or_else(|| "could not determine num_key_value_heads".to_string())?;

        let head_dim = cfg
            .get("head_dim")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .or_else(|| {
                let hidden = cfg.get("hidden_size").and_then(|v| v.as_u64())?;
                let heads = cfg.get("num_attention_heads").and_then(|v| v.as_u64())?;
                if heads == 0 {
                    None
                } else {
                    Some((hidden / heads) as usize)
                }
            })
            .unwrap_or(64);

        let mut eos_ids = parse_eos(&gen_cfg, &cfg);
        if let Some(id) = tokenizer_eos_id(&tokenizer, paths.tok_config.as_deref()) {
            eos_ids.insert(id);
        }
        if eos_ids.is_empty() {
            return Err("no eos_token_id in config/generation_config".to_string());
        }

        let rep_penalty = gen_cfg
            .get("repetition_penalty")
            .and_then(|v| v.as_f64())
            .filter(|p| *p > 0.0)
            .unwrap_or(1.0) as f32;

        let mut eos_sorted: Vec<u32> = eos_ids.iter().copied().collect();
        eos_sorted.sort_unstable();
        tracing::info!(
            n_layers,
            n_kv_heads,
            head_dim,
            max_new_tokens,
            rep_penalty,
            chat_template = ?paths.chat_template,
            eos_ids = ?eos_sorted,
            "ONNX causal-LM loaded"
        );

        Ok(Self {
            session,
            tokenizer,
            chat_template: paths.chat_template,
            n_layers,
            n_kv_heads,
            head_dim,
            eos_ids,
            rep_penalty,
            max_new_tokens,
        })
    }

    /// Build the chat prompt for a single user turn.
    ///
    /// A system message is always present: the caller's `instructions` when
    /// given, else the default for the detected model family.
    /// Template is auto-detected from config.json model_type (ChatML vs Llama).
    fn build_prompt(&self, user: &str, instructions: Option<&str>) -> String {
        let system = instructions
            .filter(|s| !s.trim().is_empty())
            .unwrap_or(default_system(self.chat_template));
        render_prompt(self.chat_template, system, user)
    }

    /// Build the four fixed (non-KV) input tensors for one decode step.
    ///
    /// `tokens` is the full input for this step (whole prompt on step 0, else
    /// the single previous token); `past_len` is the KV-cache length so far.
    #[allow(clippy::type_complexity)]
    fn step_inputs(
        tokens: &[i64],
        past_len: usize,
    ) -> Result<(Tensor<i64>, Tensor<i64>, Tensor<i64>), String> {
        let seq = tokens.len();
        let input_ids = Array2::from_shape_vec((1, seq), tokens.to_vec())
            .map_err(|e| format!("input_ids shape: {e}"))?;
        let attn = Array2::<i64>::ones((1, past_len + seq));
        let pos: Vec<i64> = (past_len..past_len + seq).map(|p| p as i64).collect();
        let position_ids =
            Array2::from_shape_vec((1, seq), pos).map_err(|e| format!("position_ids shape: {e}"))?;

        Ok((
            Tensor::from_array(input_ids).map_err(|e| format!("input_ids tensor: {e}"))?,
            Tensor::from_array(attn).map_err(|e| format!("attention_mask tensor: {e}"))?,
            Tensor::from_array(position_ids).map_err(|e| format!("position_ids tensor: {e}"))?,
        ))
    }

}

/// Greedy next-token selection with repetition penalty over `generated`,
/// mirroring the oracle's `_sample` in its `temperature <= 0` (argmax) path.
fn argmax_with_penalty(logits: &[f32], generated: &HashSet<u32>, rep_penalty: f32) -> u32 {
    let mut best_id = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    let penalize = rep_penalty != 1.0;
    for (id, &raw) in logits.iter().enumerate() {
        let mut v = raw;
        if penalize && generated.contains(&(id as u32)) {
            v = if v > 0.0 {
                v / rep_penalty
            } else {
                v * rep_penalty
            };
        }
        if v > best_val {
            best_val = v;
            best_id = id as u32;
        }
    }
    best_id
}

impl LlmStage for QwenLlm {
    fn generate(
        &mut self,
        prompt: &str,
        instructions: Option<&str>,
        cancel: &std::sync::atomic::AtomicBool,
        on_token: &mut dyn FnMut(&str),
    ) -> String {
        use std::sync::atomic::Ordering;

        let text = self.build_prompt(prompt, instructions);
        let encoded = match self.tokenizer.encode(text, false) {
            Ok(e) => e,
            Err(e) => {
                tracing::error!(error = %e, "LLM tokenizer encode failed");
                return String::new();
            }
        };
        // Full running token sequence (prompt + generated), i64 for the model.
        let mut cur: Vec<i64> = encoded.get_ids().iter().map(|&id| id as i64).collect();

        // KV cache: flat f32 buffers per layer, plus the shared cache length.
        let n = self.n_layers;
        let mut past_keys: Vec<Vec<f32>> = vec![Vec::new(); n];
        let mut past_values: Vec<Vec<f32>> = vec![Vec::new(); n];
        let mut past_len: usize = 0;

        let mut generated: Vec<u32> = Vec::new();
        let mut generated_set: HashSet<u32> = HashSet::new();
        let mut emitted = String::new();

        for step in 0..self.max_new_tokens {
            if cancel.load(Ordering::Relaxed) {
                break;
            }

            // Step 0 feeds the whole prompt; later steps feed only the last token.
            let tokens: &[i64] = if step == 0 {
                &cur
            } else {
                &cur[cur.len() - 1..]
            };
            let seq = tokens.len();

            let (input_ids, attn, position_ids) = match Self::step_inputs(tokens, past_len) {
                Ok(t) => t,
                Err(e) => {
                    tracing::error!(error = %e, "LLM input build failed");
                    break;
                }
            };

            // Assemble the dynamic feed list: 3 fixed + 2*n_layers KV tensors.
            let mut feeds: Vec<(String, SessionInputValue)> =
                Vec::with_capacity(3 + 2 * n);
            feeds.push(("input_ids".to_string(), Tensor::into(input_ids)));
            feeds.push(("attention_mask".to_string(), Tensor::into(attn)));
            feeds.push(("position_ids".to_string(), Tensor::into(position_ids)));

            let mut build_ok = true;
            for i in 0..n {
                match kv_tensor(&past_keys[i], self.n_kv_heads, past_len, self.head_dim) {
                    Ok(t) => feeds.push((format!("past_key_values.{i}.key"), t.into())),
                    Err(e) => {
                        tracing::error!(error = %e, layer = i, "LLM past key build failed");
                        build_ok = false;
                        break;
                    }
                }
                match kv_tensor(&past_values[i], self.n_kv_heads, past_len, self.head_dim) {
                    Ok(t) => feeds.push((format!("past_key_values.{i}.value"), t.into())),
                    Err(e) => {
                        tracing::error!(error = %e, layer = i, "LLM past value build failed");
                        build_ok = false;
                        break;
                    }
                }
            }
            if !build_ok {
                break;
            }

            let outputs = match self.session.run(feeds) {
                Ok(o) => o,
                Err(e) => {
                    tracing::error!(error = %e, "LLM forward failed");
                    break;
                }
            };

            // logits: [1, seq, vocab]; extract + copy the final position's row
            // inside a block so the borrow on `outputs` ends before KV harvest.
            let logits_owned: Vec<f32> = {
                let (logits_shape, logits) = match outputs[0].try_extract_tensor::<f32>() {
                    Ok(v) => v,
                    Err(e) => {
                        tracing::error!(error = %e, "LLM logits extract failed");
                        break;
                    }
                };
                let vocab = *logits_shape.last().unwrap_or(&0) as usize;
                if vocab == 0 || logits.len() < seq * vocab {
                    tracing::error!("LLM logits shape unexpected");
                    break;
                }
                logits[(seq - 1) * vocab..seq * vocab].to_vec()
            };

            // Harvest the new KV cache (present.*) while outputs is still live.
            let mut harvested: Vec<(Vec<f32>, Vec<f32>)> = Vec::with_capacity(n);
            let mut harvest_ok = true;
            for i in 0..n {
                let k = outputs
                    .get(format!("present.{i}.key").as_str())
                    .and_then(|v| v.try_extract_tensor::<f32>().ok());
                let val = outputs
                    .get(format!("present.{i}.value").as_str())
                    .and_then(|v| v.try_extract_tensor::<f32>().ok());
                match (k, val) {
                    (Some((_, kd)), Some((_, vd))) => {
                        harvested.push((kd.to_vec(), vd.to_vec()));
                    }
                    _ => {
                        tracing::error!(layer = i, "LLM present.* missing");
                        harvest_ok = false;
                        break;
                    }
                }
            }
            drop(outputs);
            if !harvest_ok {
                break;
            }
            for (i, (k, v)) in harvested.into_iter().enumerate() {
                past_keys[i] = k;
                past_values[i] = v;
            }
            past_len += seq;

            let next_id = argmax_with_penalty(&logits_owned, &generated_set, self.rep_penalty);

            if self.eos_ids.contains(&next_id) {
                break;
            }
            // Commit the token and stream its decoded delta.
            generated.push(next_id);
            generated_set.insert(next_id);
            cur.push(next_id as i64);

            if let Ok(full) = self.tokenizer.decode(&generated, true) {
                if full.len() > emitted.len() && full.starts_with(&emitted) {
                    // Hold back a trailing replacement char (incomplete UTF-8).
                    let mut end = full.len();
                    if full.ends_with('\u{FFFD}') {
                        end -= '\u{FFFD}'.len_utf8();
                    }
                    if end > emitted.len() {
                        let delta = full[emitted.len()..end].to_string();
                        on_token(&delta);
                        emitted.push_str(&delta);
                    }
                }
            }
        }

        // Flush any held-back tail so the returned text == everything emitted.
        if let Ok(full) = self.tokenizer.decode(&generated, true) {
            if full.len() > emitted.len() && full.starts_with(&emitted) {
                let tail = full[emitted.len()..].to_string();
                on_token(&tail);
                emitted.push_str(&tail);
            }
            return full;
        }
        emitted
    }
}

/// Render a single-turn chat prompt for `template`.
///
/// Free function (not a method) so the template contract is unit-testable
/// without loading model weights.
fn render_prompt(template: ChatTemplate, system: &str, user: &str) -> String {
    let mut p = String::with_capacity(system.len() + user.len() + 96);
    match template {
        ChatTemplate::ChatML => {
            p.push_str(IM_START);
            p.push_str("system\n");
            p.push_str(system);
            p.push_str(IM_END);
            p.push('\n');
            p.push_str(IM_START);
            p.push_str("user\n");
            p.push_str(user);
            p.push_str(IM_END);
            p.push('\n');
            p.push_str(IM_START);
            p.push_str("assistant\n");
        }
        ChatTemplate::Llama => {
            p.push_str(LLAMA_BOS);
            p.push_str(LLAMA_SYS_START);
            p.push_str(system);
            p.push_str(LLAMA_EOT);
            p.push_str(LLAMA_USER_START);
            p.push_str(user);
            p.push_str(LLAMA_EOT);
            p.push_str(LLAMA_ASST_START);
            p.push('\n');
        }
    }
    p
}

// ---------------------------------------------------------------------------
// StagePool Replica implementation
// ---------------------------------------------------------------------------

/// Input for the LLM pool: a user prompt plus optional system instructions.
pub struct LlmInput {
    pub prompt: String,
    pub instructions: Option<String>,
}

impl Replica for QwenLlm {
    type Input = LlmInput;
    type Delta = String;
    type Output = String;

    fn process(
        &mut self,
        input: Self::Input,
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(String),
    ) -> Self::Output {
        self.generate(
            &input.prompt,
            input.instructions.as_deref(),
            cancel,
            &mut |t: &str| emit(t.to_string()),
        )
    }
}

/// Build a KV-cache tensor of shape `[1, n_kv_heads, past_len, head_dim]` from a
/// flat buffer. On step 0 `past_len` is 0 and `data` is empty.
fn kv_tensor(
    data: &[f32],
    n_kv_heads: usize,
    past_len: usize,
    head_dim: usize,
) -> Result<Tensor<f32>, String> {
    let arr = Array4::from_shape_vec((1, n_kv_heads, past_len, head_dim), data.to_vec())
        .map_err(|e| format!("kv reshape: {e}"))?;
    Tensor::from_array(arr).map_err(|e| format!("kv tensor: {e}"))
}

/// If `dir` holds no `*.onnx` of its own but contains exactly ONE subdirectory
/// that does, return that child; otherwise return `dir` unchanged.
///
/// This makes `./models/llm` (the per-model root) work directly when only one
/// model is installed, which is the common case. With several installed there is
/// no defensible guess — picking one alphabetically would silently run a model
/// nobody asked for — so `dir` is returned as-is and the caller reports a clear
/// "no onnx model found" error naming the directory.
fn descend_to_single_model(dir: &str) -> String {
    let has_onnx = |p: &std::path::Path| {
        std::fs::read_dir(p).is_ok_and(|rd| {
            rd.filter_map(|e| e.ok())
                .any(|e| e.file_name().to_string_lossy().ends_with(".onnx"))
        })
    };
    let root = std::path::Path::new(dir);
    if has_onnx(root) {
        return dir.to_string();
    }
    let Ok(entries) = std::fs::read_dir(root) else {
        return dir.to_string();
    };
    let mut children: Vec<std::path::PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir() && has_onnx(p))
        .collect();
    if children.len() == 1 {
        let child = children.remove(0).to_string_lossy().into_owned();
        tracing::info!(dir, resolved = %child, "LLM_MODEL_DIR held one model; using it");
        return child;
    }
    if children.len() > 1 {
        children.sort();
        let names: Vec<String> = children
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();
        tracing::warn!(
            dir,
            candidates = ?names,
            "LLM_MODEL_DIR holds several models; set LLM_MODEL_DIR to one of them"
        );
    }
    dir.to_string()
}

/// Resolve the model files inside `dir`, honoring the `LLM_ONNX_FILE` /
/// `LLM_TOKENIZER_FILE` overrides (relative to `dir` unless absolute).
fn resolve_paths(dir: &str) -> Result<LlmPaths, String> {
    let onnx = resolve_file(dir, "LLM_ONNX_FILE", &["*.onnx"], "onnx model")?;
    let tokenizer = resolve_file(
        dir,
        "LLM_TOKENIZER_FILE",
        &["*tokenizer*.json"],
        "tokenizer.json",
    )?;
    let config = existing(&format!("{dir}/config.json"));
    let gen_config = existing(&format!("{dir}/generation_config.json"));
    let tok_config = existing(&format!("{dir}/tokenizer_config.json"));

    // Chat template: read the model's OWN Jinja `chat_template` from
    // tokenizer_config.json and detect which family it renders. Deriving the
    // template from `config.json`'s `model_type` is WRONG — Hermes-3-Llama-3.2-3B
    // reports `model_type: "llama"` but was fine-tuned on ChatML
    // (`<|im_start|>`/`<|im_end|>`, eos `<|im_end|>`), so the architecture says
    // nothing about the prompt format. `LLM_CHAT_TEMPLATE=chatml|llama` forces it.
    let chat_template = env_chat_template()
        .or_else(|| detect_chat_template(tok_config.as_deref()))
        .unwrap_or(ChatTemplate::ChatML);
    tracing::info!(?chat_template, "LLM chat template selected");

    Ok(LlmPaths {
        onnx,
        tokenizer,
        config,
        gen_config,
        tok_config,
        chat_template,
    })
}

/// Explicit `LLM_CHAT_TEMPLATE` override (`chatml` | `llama`), if set & valid.
fn env_chat_template() -> Option<ChatTemplate> {
    let raw = std::env::var("LLM_CHAT_TEMPLATE").ok()?;
    match raw.trim().to_lowercase().as_str() {
        "chatml" | "qwen" | "hermes" => Some(ChatTemplate::ChatML),
        "llama" | "llama3" => Some(ChatTemplate::Llama),
        "" => None,
        other => {
            tracing::warn!(value = other, "unknown LLM_CHAT_TEMPLATE; auto-detecting");
            None
        }
    }
}

/// Detect the chat family from the Jinja `chat_template` in `tokenizer_config.json`.
///
/// We only need to know which special tokens the template emits, so a substring
/// probe is sufficient (and far cheaper than a Jinja engine). ChatML is checked
/// first because ChatML-tuned Llama derivatives (Hermes-3) carry Llama's
/// `<|begin_of_text|>` in the tokenizer while templating with `<|im_start|>`.
fn detect_chat_template(tok_config: Option<&str>) -> Option<ChatTemplate> {
    let raw = std::fs::read_to_string(tok_config?).ok()?;
    let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let tmpl = v.get("chat_template")?.as_str()?;
    if tmpl.contains(IM_START) {
        Some(ChatTemplate::ChatML)
    } else if tmpl.contains("start_header_id") {
        Some(ChatTemplate::Llama)
    } else {
        None
    }
}

/// Env override wins; else first filename in `dir` matching any glob pattern.
fn resolve_file(dir: &str, env: &str, patterns: &[&str], what: &str) -> Result<String, String> {
    if let Some(v) = std::env::var(env).ok().filter(|v| !v.is_empty()) {
        let p = if std::path::Path::new(&v).is_absolute() {
            v
        } else {
            format!("{dir}/{v}")
        };
        return existing(&p).ok_or_else(|| format!("{env}={p} does not exist"));
    }
    let entries = std::fs::read_dir(dir).map_err(|e| format!("read_dir {dir}: {e}"))?;
    let mut names: Vec<String> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().into_string().ok())
        .collect();
    names.sort();
    for name in &names {
        if patterns.iter().any(|pat| glob_match(pat, name)) {
            return Ok(format!("{dir}/{name}"));
        }
    }
    Err(format!("no {what} found in {dir}"))
}

fn existing(path: &str) -> Option<String> {
    if std::path::Path::new(path).exists() {
        Some(path.to_string())
    } else {
        None
    }
}

/// Minimal glob: supports a single `*` wildcard anywhere in `pat`, plus a bare
/// `*.ext` / `*substr*` shape. Sufficient for the fixed patterns above.
fn glob_match(pat: &str, name: &str) -> bool {
    let parts: Vec<&str> = pat.split('*').collect();
    if parts.len() == 1 {
        return pat == name;
    }
    let mut pos = 0usize;
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        if i == 0 {
            if !name.starts_with(part) {
                return false;
            }
            pos = part.len();
        } else if i == parts.len() - 1 {
            return name[pos..].ends_with(part) && name.len() - pos >= part.len();
        } else {
            match name[pos..].find(part) {
                Some(idx) => pos += idx + part.len(),
                None => return false,
            }
        }
    }
    true
}

fn load_json(path: Option<&str>) -> serde_json::Value {
    path.and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or(serde_json::Value::Null)
}

/// EOS ids: the UNION of `generation_config.eos_token_id` and
/// `config.eos_token_id`. Accepts either a scalar or an array in each.
///
/// The union (not "gen_config wins") matters for ChatML-tuned Llama derivatives.
/// Hermes-3-Llama-3.2-3B ships an inherited-from-base
/// `generation_config.eos_token_id = [128001, 128008, 128009]` while
/// `config.eos_token_id = 128039` (`<|im_end|>`) is the token the fine-tune
/// actually emits to end a turn. Preferring gen_config alone would drop 128039,
/// so generation would never stop on its own and every reply would run to the
/// `LLM_MAX_NEW_TOKENS` cap. Extra ids are harmless: a model simply never emits
/// the ones it doesn't use.
fn parse_eos(gen_cfg: &serde_json::Value, cfg: &serde_json::Value) -> HashSet<u32> {
    let mut out = HashSet::new();
    for src in [gen_cfg, cfg] {
        match src.get("eos_token_id") {
            Some(serde_json::Value::Number(n)) => {
                if let Some(id) = n.as_u64() {
                    out.insert(id as u32);
                }
            }
            Some(serde_json::Value::Array(arr)) => {
                for v in arr {
                    if let Some(id) = v.as_u64() {
                        out.insert(id as u32);
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// Extra stop ids from the tokenizer's `eos_token` string, resolved to an id.
///
/// Belt-and-braces for exports whose `config.json` predates the fine-tune's
/// ChatML retarget: `tokenizer_config.json`'s `eos_token` is authoritative about
/// what the chat format ends with.
fn tokenizer_eos_id(tokenizer: &Tokenizer, tok_config: Option<&str>) -> Option<u32> {
    let raw = std::fs::read_to_string(tok_config?).ok()?;
    let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
    // `eos_token` is either a plain string or an AddedToken object.
    let tok = v.get("eos_token").and_then(|e| {
        e.as_str()
            .map(str::to_string)
            .or_else(|| e.get("content")?.as_str().map(str::to_string))
    })?;
    tokenizer.token_to_id(&tok)
}

/// Count layers from `present.N.*` output names when config is unavailable.
fn infer_layers(session: &Session) -> Option<usize> {
    let mut max_idx: Option<usize> = None;
    for out in &session.outputs {
        if let Some(rest) = out.name.strip_prefix("present.") {
            if let Some(idx_str) = rest.split('.').next() {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    max_idx = Some(max_idx.map_or(idx, |m| m.max(idx)));
                }
            }
        }
    }
    max_idx.map(|m| m + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_matches_extension_and_substring() {
        assert!(glob_match("*.onnx", "qwen2.5-1.5b-instruct.int8.onnx"));
        assert!(!glob_match("*.onnx", "config.json"));
        assert!(glob_match("*tokenizer*.json", "qwen_tokenizer.json"));
        assert!(!glob_match("*tokenizer*.json", "config.json"));
        assert!(glob_match("exact", "exact"));
    }

    #[test]
    fn parse_eos_scalar_and_array() {
        let gen = serde_json::json!({ "eos_token_id": [151645, 151643] });
        let cfg = serde_json::json!({ "eos_token_id": 151645 });
        let ids = parse_eos(&gen, &cfg);
        assert!(ids.contains(&151645) && ids.contains(&151643));

        // Falls back to config when gen_config lacks the key.
        let empty = serde_json::Value::Null;
        let ids2 = parse_eos(&empty, &cfg);
        assert_eq!(ids2.len(), 1);
        assert!(ids2.contains(&151645));
    }
    #[test]
    fn chatml_prompt_has_all_three_turns_and_open_assistant() {
        let p = render_prompt(ChatTemplate::ChatML, "SYS", "USER");
        assert_eq!(p.matches(IM_START).count(), 3, "prompt: {p:?}");
        assert_eq!(p.matches(IM_END).count(), 2, "assistant turn stays open: {p:?}");
        assert!(p.contains("system\nSYS"));
        assert!(p.contains("user\nUSER"));
        assert!(p.ends_with("assistant\n"), "prompt: {p:?}");
        // No Llama tokens leak into the ChatML template.
        assert!(!p.contains(LLAMA_BOS));
        assert!(!p.contains(LLAMA_EOT));
    }

    #[test]
    fn llama_prompt_uses_header_tokens_and_ends_open() {
        let p = render_prompt(ChatTemplate::Llama, "SYS", "USER");
        assert!(p.starts_with(LLAMA_BOS), "prompt: {p:?}");
        assert_eq!(p.matches(LLAMA_BOS).count(), 1, "exactly one BOS: {p:?}");
        assert_eq!(p.matches(LLAMA_EOT).count(), 2, "system + user closed: {p:?}");
        assert!(p.contains(LLAMA_SYS_START));
        assert!(p.contains(LLAMA_USER_START));
        // Assistant header is last and left open for generation.
        let asst = p.find(LLAMA_ASST_START).expect("assistant header present");
        assert!(asst > p.rfind(LLAMA_EOT).unwrap(), "assistant header must be last");
        assert!(p.contains("SYS") && p.contains("USER"));
        // No ChatML tokens leak into the Llama template.
        assert!(!p.contains(IM_START));
        assert!(!p.contains(IM_END));
    }

    #[test]
    fn default_system_is_model_agnostic() {
        let chatml = default_system(ChatTemplate::ChatML);
        let llama = default_system(ChatTemplate::Llama);
        // Same persona regardless of template: a dev(ONNX)/prod(vLLM) backend
        // swap must not change how the assistant behaves.
        assert_eq!(chatml, llama);
        assert!(!chatml.trim().is_empty());
        // No vendor persona leaks in (the old Qwen default did).
        assert!(!chatml.contains("Qwen"));
        assert!(!chatml.contains("Alibaba"));
    }

    #[test]
    fn chat_template_detected_from_the_models_own_jinja_template() {
        let dir = std::env::temp_dir().join("szca_tmpl_detect_test");
        std::fs::create_dir_all(&dir).expect("tmpdir");
        let write = |name: &str, body: &str| {
            let p = dir.join(name);
            std::fs::write(&p, body).expect("write");
            p.to_string_lossy().to_string()
        };

        // Hermes-3-Llama-3.2-3B: `model_type` is "llama" but the template is
        // ChatML. Detection must follow the template, not the architecture.
        let hermes = write(
            "hermes_tokenizer_config.json",
            r#"{"chat_template":"{% for m in messages %}{{'<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>'}}{% endfor %}","eos_token":"<|im_end|>"}"#,
        );
        assert_eq!(
            detect_chat_template(Some(&hermes)),
            Some(ChatTemplate::ChatML),
            "ChatML-tuned Llama derivative must resolve to ChatML"
        );

        // Stock Llama-3 Instruct uses the header-id template.
        let llama3 = write(
            "llama3_tokenizer_config.json",
            r#"{"chat_template":"{{ '<|start_header_id|>' + role + '<|end_header_id|>\n\n' + content + '<|eot_id|>' }}"}"#,
        );
        assert_eq!(
            detect_chat_template(Some(&llama3)),
            Some(ChatTemplate::Llama)
        );

        // No template / missing file / unrecognized tokens → no opinion, so the
        // caller falls back to its default rather than guessing wrong.
        let bare = write("bare_tokenizer_config.json", r#"{"eos_token":"</s>"}"#);
        assert_eq!(detect_chat_template(Some(&bare)), None);
        assert_eq!(detect_chat_template(None), None);
        assert_eq!(
            detect_chat_template(Some("/nonexistent/tokenizer_config.json")),
            None
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_root_descends_to_a_lone_model_but_never_guesses_between_two() {
        let root = std::env::temp_dir().join("szca_descend_test");
        let _ = std::fs::remove_dir_all(&root);
        let a = root.join("Hermes-3-Llama-3.2-3B");
        std::fs::create_dir_all(&a).expect("tmpdir");
        std::fs::write(a.join("model.onnx"), b"x").expect("write");
        let root_s = root.to_string_lossy().to_string();

        // One model installed under the root: resolve into it, so the default
        // `./models/llm` works with no LLM_MODEL_DIR set at all.
        assert_eq!(
            descend_to_single_model(&root_s),
            a.to_string_lossy().to_string()
        );

        // A dir that holds the graph itself is already correct — don't descend.
        let a_s = a.to_string_lossy().to_string();
        assert_eq!(descend_to_single_model(&a_s), a_s);

        // Two models: there is no right answer, so return the root unchanged and
        // let the caller fail loudly. Silently picking one would run a model the
        // operator never selected, with that model's KV geometry and stop tokens.
        let b = root.join("Qwen2.5-1.5B-Instruct");
        std::fs::create_dir_all(&b).expect("tmpdir");
        std::fs::write(b.join("model.onnx"), b"x").expect("write");
        assert_eq!(
            descend_to_single_model(&root_s),
            root_s,
            "ambiguous root must not resolve to an arbitrary model"
        );
        assert!(
            resolve_paths(&root_s).is_err(),
            "ambiguous root must surface an error, not a silent stub-quality load"
        );

        // A nonexistent path is returned as-is; resolve_paths reports it.
        assert_eq!(descend_to_single_model("/nonexistent/llm"), "/nonexistent/llm");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn eos_ids_union_both_configs() {
        // Hermes-3-Llama-3.2-3B's real values: generation_config carries the
        // inherited Llama ids, config.json carries 128039 (`<|im_end|>`) — the
        // token the ChatML fine-tune actually ends turns with. Dropping it would
        // make every reply run to the max-token cap.
        let gen = serde_json::json!({ "eos_token_id": [128001, 128008, 128009] });
        let cfg = serde_json::json!({ "eos_token_id": 128039 });
        let ids = parse_eos(&gen, &cfg);
        assert_eq!(ids.len(), 4, "expected the union, got {ids:?}");
        for id in [128001u32, 128008, 128009, 128039] {
            assert!(ids.contains(&id), "missing {id} in {ids:?}");
        }
    }
}
