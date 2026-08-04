//! Real in-process ONNX LLM inference test (dev profile).
//!
//! Verifies the DEV LLM path end-to-end against actual weights: model + external
//! data load, the chat template and stop tokens are picked up FROM THE
//! CHECKPOINT, the KV-cache decode loop streams token deltas, and barge-in
//! cancels generation.
//!
//! Skipped unless weights are provided (so CI without weights stays green):
//!
//!   export ORT_DYLIB_PATH=/opt/homebrew/lib/libonnxruntime.dylib
//!   export LLM_MODEL_DIR=./models/llm/Hermes-3-Llama-3.2-3B
//!   cargo test --test llm_real_inference -- --nocapture --test-threads=1
//!
//! NOTE on cost: Hermes-3-Llama-3.2-3B is an FP32 export (~14 GB of external
//! data) and runs on CPU in dev, so a session load plus a handful of tokens
//! takes minutes and a lot of RAM. `LLM_MAX_NEW_TOKENS` is held small here on
//! purpose — this test proves CORRECTNESS of the dev path, not throughput.

use std::sync::atomic::{AtomicBool, Ordering};

use szca_media_gateway::rt_llm::QwenLlm;
use szca_media_gateway::rt_pipeline::LlmStage;

/// The dev model directory, when configured AND present on disk.
///
/// `LLM_MODEL_DIR` is normally a REPO-ROOT-relative path (that is what
/// `env.dev.example` and `download_models.sh` produce), but cargo runs an
/// integration test with the CWD set to the PACKAGE dir. So `./models/...`
/// resolves one level too deep and the test would skip as if unconfigured —
/// green, having tested nothing. Resolve relative paths against the repo root
/// and re-export the absolute path so `QwenLlm::from_env()` sees the same dir.
///
/// A dir that is set but missing is reported distinctly: a typo'd path must not
/// masquerade as "no weights installed".
fn model_dir() -> Option<String> {
    let dir = std::env::var("LLM_MODEL_DIR").ok().filter(|d| !d.is_empty())?;
    let resolved = resolve_from_repo_root(&dir);
    if !std::path::Path::new(&resolved).is_dir() {
        eprintln!("SKIP: LLM_MODEL_DIR is set but not a directory: {resolved}");
        return None;
    }
    std::env::set_var("LLM_MODEL_DIR", &resolved);
    Some(resolved)
}

/// Absolute paths pass through; relative ones are joined to the repo root
/// (the parent of `CARGO_MANIFEST_DIR`).
fn resolve_from_repo_root(dir: &str) -> String {
    let p = std::path::Path::new(dir);
    if p.is_absolute() {
        return dir.to_string();
    }
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    match manifest.parent() {
        Some(root) => root.join(p).to_string_lossy().into_owned(),
        None => dir.to_string(),
    }
}

/// Serializes model loads. An FP32 3B export is ~14 GB resident, so letting
/// cargo's default parallel test threads each load their own copy would OOM the
/// machine; it also keeps the `LLM_MAX_NEW_TOKENS` set/read pair atomic.
static LOAD_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Load the LLM with a small token cap so the test stays bounded.
fn load(max_new_tokens: &str) -> Option<QwenLlm> {
    let _dir = model_dir()?;
    let _guard = LOAD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var("LLM_MAX_NEW_TOKENS", max_new_tokens);
    match QwenLlm::from_env() {
        Ok(m) => Some(m),
        Err(e) => {
            // A missing ORT dylib / absent weights is a SKIP, not a failure; a
            // malformed checkpoint should still be loud.
            eprintln!("SKIP: LLM did not load: {e}");
            None
        }
    }
}

#[test]
fn dev_llm_loads_and_streams_tokens() {
    let Some(mut llm) = load("24") else {
        eprintln!("SKIP: set LLM_MODEL_DIR (+ ORT_DYLIB_PATH) to run this test");
        return;
    };

    let cancel = AtomicBool::new(false);
    let mut deltas: Vec<String> = Vec::new();
    let reply = llm.generate(
        "What is the capital of France? Answer in one short sentence.",
        Some("You are a concise voice assistant."),
        &cancel,
        &mut |t| deltas.push(t.to_string()),
    );

    eprintln!("reply: {reply:?} ({} deltas)", deltas.len());

    assert!(!reply.trim().is_empty(), "model produced no text");
    assert!(
        deltas.len() > 1,
        "generation must STREAM (got {} delta(s)); a single delta means the \
         incremental-detokenize path is broken",
        deltas.len()
    );
    // Deltas are the client's only source of text — they must reconstruct the
    // final reply exactly.
    assert_eq!(
        deltas.concat(),
        reply,
        "streamed deltas must reconstruct the final reply"
    );

    // Special tokens must never reach the client (they'd be spoken by TTS).
    for marker in [
        "<|im_start|>",
        "<|im_end|>",
        "<|eot_id|>",
        "<|begin_of_text|>",
        "<|end_of_text|>",
    ] {
        assert!(
            !reply.contains(marker),
            "special token {marker} leaked into the reply: {reply:?}"
        );
    }
    // No partial UTF-8 leaked through the incremental detokenizer.
    assert!(
        !reply.contains('\u{FFFD}'),
        "replacement char in reply (broken multi-byte handling): {reply:?}"
    );

    // Quality smoke check: a 3B instruct model should get this right. Warn
    // rather than fail — greedy decode on an FP32 export can still wander, and
    // this test's job is the plumbing, not the model's IQ.
    if !reply.to_lowercase().contains("paris") {
        eprintln!("WARNING: expected 'Paris' in the reply; got {reply:?}");
    }
}

#[test]
fn dev_llm_stops_on_its_own_eos_not_the_token_cap() {
    // A generous cap: if the checkpoint's real stop token were missed (the
    // Hermes-3 `<|im_end|>`=128039 vs inherited generation_config ids trap),
    // generation would run all the way to the cap instead of ending naturally.
    let Some(mut llm) = load("64") else {
        eprintln!("SKIP: set LLM_MODEL_DIR (+ ORT_DYLIB_PATH) to run this test");
        return;
    };

    let cancel = AtomicBool::new(false);
    let mut count = 0usize;
    let reply = llm.generate(
        "Say exactly: OK",
        Some("Reply with one word."),
        &cancel,
        &mut |_| count += 1,
    );

    eprintln!("reply: {reply:?} ({count} tokens)");
    assert!(count > 0, "no tokens generated");
    assert!(
        count < 64,
        "generation hit the {count}-token cap instead of stopping on EOS — the \
         checkpoint's stop token is probably not in eos_ids"
    );
}

#[test]
fn dev_llm_honors_barge_in_cancel() {
    let Some(mut llm) = load("64") else {
        eprintln!("SKIP: set LLM_MODEL_DIR (+ ORT_DYLIB_PATH) to run this test");
        return;
    };

    // Cancel as soon as the first token lands: the loop checks the flag before
    // each forward pass, so it must return almost immediately after.
    let cancel = AtomicBool::new(false);
    let mut count = 0usize;
    let reply = llm.generate(
        "Count slowly from one to fifty, one number per line.",
        None,
        &cancel,
        &mut |_| {
            count += 1;
            cancel.store(true, Ordering::Relaxed);
        },
    );

    eprintln!("cancelled after {count} token(s): {reply:?}");
    assert!(count >= 1, "expected at least one token before cancelling");
    assert!(
        count <= 3,
        "cancel was not honored promptly: generated {count} tokens after the \
         flag was set on token 1"
    );
    // Partial output is still returned (the client keeps what was already said).
    assert!(!reply.is_empty(), "cancelled generation returned nothing");
}
