//! Real streaming-STT inference test: Parakeet EOU 120M, cache-aware encoder.
//!
//! Skipped (green) when weights are absent, like the other real-weights tests:
//!
//! ```bash
//! ORT_DYLIB_PATH=/opt/homebrew/lib/libonnxruntime.dylib \
//! STT_EOU_MODEL_DIR=$PWD/models/stt_eou \
//! cargo test --release --manifest-path szca_media_gateway/Cargo.toml \
//!   --test stt_eou_real_inference -- --nocapture --test-threads=1
//! ```
//!
//! ## Why this test exists
//!
//! Every failure mode found while building this stage was SILENT — the model
//! loads, inference runs, no error is raised, and the transcript is simply empty
//! or subtly wrong:
//!
//! | Mistake | Symptom |
//! |---|---|
//! | Any mel normalization | `""` — zero tokens, no error |
//! | Log guard `1e-5` instead of `2^-24` | `"hello"` — truncated |
//! | Reading joint logit slot 0, not the last | `"he wor worww"` — plausible garbage |
//! | Pair-format vocab parser | `""` — empty vocab, no error |
//! | Dropping the encoder caches between chunks | each chunk decoded as a fresh utterance |
//!
//! A test that only asserted "it ran without error" would pass in all five
//! cases. So this asserts on the DECODED TEXT.

use szca_media_gateway::rt_pipeline::SttStage;
use szca_media_gateway::rt_stt_eou::ParakeetEouStt;
use szca_media_gateway::stage_pool::StagePool;
use szca_media_gateway::stage_pools::{SttBackend, SttPoolAdapter};

/// Resolve a possibly-relative path against the REPO ROOT.
///
/// Cargo runs integration tests with CWD = the package dir
/// (`szca_media_gateway/`), so `./models/...` from `env.dev.example` would
/// otherwise resolve one level too deep.
fn resolve_from_repo_root(p: &str) -> Option<String> {
    let path = std::path::Path::new(p);
    if path.is_absolute() {
        return Some(p.to_string());
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent()?;
    Some(root.join(path).to_string_lossy().into_owned())
}

/// Model dir, when explicitly configured AND present on disk.
///
/// Gated on `STT_EOU_MODEL_DIR` being SET, matching `llm_real_inference.rs`.
/// Defaulting to `./models/stt_eou` would make a plain `cargo test` load a
/// 230 MB encoder on any machine that has run `download_models.sh` — real-weights
/// tests are opt-in here by convention.
///
/// A dir that is set but missing is reported distinctly, so a typo'd path cannot
/// masquerade as "no weights installed" and skip green having tested nothing.
fn model_dir() -> Option<String> {
    let raw = std::env::var("STT_EOU_MODEL_DIR").ok().filter(|d| !d.is_empty())?;
    let dir = resolve_from_repo_root(&raw)?;
    if !std::path::Path::new(&format!("{dir}/encoder.onnx")).is_file() {
        eprintln!("SKIP: STT_EOU_MODEL_DIR is set but has no encoder.onnx: {dir}");
        return None;
    }
    Some(dir)
}

/// Minimal PCM16 mono WAV reader: returns (samples as f32 in [-1,1], sample_rate).
///
/// Hand-rolled rather than pulling in a `wav` crate for one test: it walks the
/// RIFF chunk list instead of assuming a fixed 44-byte header, because
/// TTS-generated files often carry a `LIST` chunk before `data`.
fn read_wav_mono(path: &str) -> Option<(Vec<f32>, u32)> {
    let b = std::fs::read(path).ok()?;
    if b.len() < 12 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return None;
    }
    let rd32 = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) as usize;
    let rd16 = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]) as usize;

    let mut pos = 12;
    let (mut rate, mut channels, mut bits) = (0u32, 0usize, 0usize);
    let mut samples: Vec<f32> = Vec::new();

    while pos + 8 <= b.len() {
        let id = &b[pos..pos + 4];
        let size = rd32(pos + 4);
        let body = pos + 8;
        if body + size > b.len() {
            break;
        }
        if id == b"fmt " && size >= 16 {
            channels = rd16(body + 2);
            rate = rd32(body + 4) as u32;
            bits = rd16(body + 14);
        } else if id == b"data" {
            if bits != 16 || channels == 0 {
                return None;
            }
            let step = 2 * channels; // take channel 0 only
            samples = b[body..body + size]
                .chunks_exact(step)
                .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
                .collect();
        }
        pos = body + size + (size & 1); // chunks are word-aligned
    }
    if samples.is_empty() || rate == 0 {
        None
    } else {
        Some((samples, rate))
    }
}

/// Linear resample to 16 kHz. Good enough for a decode assertion; the engine's
/// hot path uses SoXR.
fn to_16k(samples: &[f32], rate: u32) -> Vec<f32> {
    if rate == 16_000 || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = rate as f64 / 16_000.0;
    let n = (samples.len() as f64 / ratio).floor() as usize;
    (0..n)
        .map(|i| {
            let src = i as f64 * ratio;
            let i0 = src.floor() as usize;
            let i1 = (i0 + 1).min(samples.len() - 1);
            let frac = (src - i0 as f64) as f32;
            samples[i0] * (1.0 - frac) + samples[i1] * frac
        })
        .collect()
}

/// The spoken-"hello world" fixture, resampled to 16 kHz, as PCM16 LE bytes.
///
/// `STT_EOU_TEST_WAV` overrides the path; the default is the Kokoro sample that
/// ships next to the TTS weights. Both are gitignored model artifacts, so this
/// returns `None` (→ skip) on a checkout without `./download_models.sh`.
fn hello_world_pcm() -> Option<Vec<u8>> {
    let raw = std::env::var("STT_EOU_TEST_WAV")
        .ok()
        .filter(|p| !p.is_empty())
        .unwrap_or_else(|| "./models/tts/kokoro_hello_sample.wav".to_string());
    let path = resolve_from_repo_root(&raw)?;
    let (samples, rate) = read_wav_mono(&path)?;
    let s16 = to_16k(&samples, rate);
    Some(
        s16.iter()
            .flat_map(|&v| {
                let q = (v.clamp(-1.0, 1.0) * 32767.0) as i16;
                q.to_le_bytes()
            })
            .collect(),
    )
}

#[test]
fn eou_model_loads_and_reports_vocab() {
    let Some(dir) = model_dir() else {
        eprintln!("SKIP: no encoder.onnx under STT_EOU_MODEL_DIR");
        return;
    };
    let stt = ParakeetEouStt::load(&dir).expect("EOU model should load on ORT 1.22");
    // Loading at all is the assertion that matters: the INT8 export of this same
    // model fails here with `NOT_IMPLEMENTED: ConvInteger(10)` on ORT < 1.24.
    assert_eq!(stt.transcript(), "", "fresh engine must have an empty transcript");
    eprintln!("EOU streaming STT loaded from {dir}");
}

/// The headline assertion: real audio in, `"hello world"` out.
#[test]
fn eou_streaming_decodes_hello_world() {
    let Some(dir) = model_dir() else {
        eprintln!("SKIP: no encoder.onnx under STT_EOU_MODEL_DIR");
        return;
    };
    let Some(pcm) = hello_world_pcm() else {
        eprintln!("SKIP: no 'hello world' fixture (set STT_EOU_TEST_WAV)");
        return;
    };

    let mut stt = ParakeetEouStt::load(&dir).expect("model should load");

    let mut partials = Vec::new();
    let text = stt.transcribe(&pcm, &mut |p| partials.push(p.to_string()));
    eprintln!("EOU transcript: {text:?}  partials={partials:?}");

    let lower = text.to_lowercase();
    assert!(
        lower.contains("hello"),
        "expected 'hello' in transcript, got {text:?} — check mel normalization (any \
         normalization yields zero tokens) and the log guard (1e-5 truncates to 'hello')"
    );
    assert!(
        lower.contains("world"),
        "expected 'world' in transcript, got {text:?} — check the joint logit slot: \
         reading slot 0 instead of the LAST slot decodes 'he wor worww'"
    );

    // Streaming means partials arrive as chunks complete, not one batch result.
    assert!(!partials.is_empty(), "no partial results emitted");
}

/// Feeding the same audio in tiny frames must give the same answer as one push:
/// this is what proves the sample/frame carry-over buffering is correct.
#[test]
fn eou_chunked_feed_matches_single_push() {
    let Some(dir) = model_dir() else {
        eprintln!("SKIP: no encoder.onnx under STT_EOU_MODEL_DIR");
        return;
    };
    let Some(pcm) = hello_world_pcm() else {
        eprintln!("SKIP: no 'hello world' fixture (set STT_EOU_TEST_WAV)");
        return;
    };

    let mut stt = ParakeetEouStt::load(&dir).expect("model should load");

    let one = stt.transcribe(&pcm, &mut |_| {});

    // 20 ms frames = 640 bytes, i.e. what a WebSocket client actually sends.
    stt.reset();
    for frame in pcm.chunks(640) {
        stt.push_pcm(frame);
    }
    stt.flush();
    let many = stt.transcript().trim().to_string();

    assert_eq!(
        one, many,
        "20 ms frame-by-frame feed diverged from a single push — the sample carry \
         buffer or the mel pending buffer is dropping/duplicating audio"
    );
    eprintln!("chunked == single push: {one:?}");
}

/// `reset()` must clear the encoder caches. Without it, turn N+1 inherits turn
/// N's acoustic context and the second transcript drifts.
#[test]
fn eou_reset_gives_a_repeatable_transcript() {
    let Some(dir) = model_dir() else {
        eprintln!("SKIP: no encoder.onnx under STT_EOU_MODEL_DIR");
        return;
    };
    let Some(pcm) = hello_world_pcm() else {
        eprintln!("SKIP: no 'hello world' fixture (set STT_EOU_TEST_WAV)");
        return;
    };

    let mut stt = ParakeetEouStt::load(&dir).expect("model should load");
    let first = stt.transcribe(&pcm, &mut |_| {});
    let second = stt.transcribe(&pcm, &mut |_| {});
    assert_eq!(
        first, second,
        "same audio decoded differently on the second turn — reset() is not \
         clearing the encoder cache"
    );
}

/// A streaming encoder is only useful if a chunk decodes faster than it plays.
/// At RTF > 1 the buffer grows without bound and latency diverges under load, so
/// this asserts the invariant rather than just printing a number.
///
/// The bar is deliberately loose (RTF > 2×, i.e. half the chunk budget) because
/// this runs on developer laptops and in CI alongside other tests. The measured
/// figure on an M-series CPU is ~21×; a regression to 2× would mean something
/// structural broke, not that the machine was busy.
#[test]
fn eou_decodes_faster_than_realtime() {
    let Some(dir) = model_dir() else {
        eprintln!("SKIP: no encoder.onnx under STT_EOU_MODEL_DIR");
        return;
    };
    let Some(pcm) = hello_world_pcm() else {
        eprintln!("SKIP: no 'hello world' fixture (set STT_EOU_TEST_WAV)");
        return;
    };

    let mut stt = ParakeetEouStt::load(&dir).expect("model should load");
    // Warm up: the first ORT run pays one-off arena allocation and kernel setup.
    let _ = stt.transcribe(&pcm, &mut |_| {});

    let audio_secs = (pcm.len() / 2) as f64 / 16_000.0;
    let t0 = std::time::Instant::now();
    let text = stt.transcribe(&pcm, &mut |_| {});
    let elapsed = t0.elapsed().as_secs_f64();
    let rtf = audio_secs / elapsed;

    eprintln!(
        "EOU streaming: {audio_secs:.2}s audio in {:.3}s => {rtf:.1}x realtime ({text:?})",
        elapsed
    );
    assert!(
        rtf > 2.0,
        "{rtf:.2}x realtime is too slow for streaming — the 1.28s chunk budget \
         would be exceeded under any concurrency"
    );
}

/// The production path: through `SttBackend::Streaming` in a real `StagePool`,
/// via `SttPoolAdapter`. The direct-struct tests above would still pass if the
/// enum dispatch, the pool worker, or the adapter's delta plumbing were wrong.
#[test]
fn eou_streams_through_the_stage_pool() {
    let Some(dir) = model_dir() else {
        eprintln!("SKIP: no encoder.onnx under STT_EOU_MODEL_DIR");
        return;
    };
    let Some(pcm) = hello_world_pcm() else {
        eprintln!("SKIP: no 'hello world' fixture (set STT_EOU_TEST_WAV)");
        return;
    };

    let pool = StagePool::build("stt-eou-test", 1, |_| {
        ParakeetEouStt::load(&dir).map(SttBackend::Streaming)
    })
    .expect("pool should build");

    let mut adapter = SttPoolAdapter::new(&pool);
    let mut partials = Vec::new();
    let text = adapter.transcribe(&pcm, &mut |p| partials.push(p.to_string()));

    assert!(
        text.to_lowercase().contains("hello world"),
        "pool path returned {text:?}"
    );
    assert!(!partials.is_empty(), "adapter forwarded no partial deltas");

    // A pooled replica is reused across jobs. If `process()` did not reset the
    // encoder cache, the second submission would inherit the first turn's
    // context — the single most likely way this stage breaks in production.
    let second = adapter.transcribe(&pcm, &mut |_| {});
    assert_eq!(text, second, "replica reuse changed the transcript");

    eprintln!("pool path: {text:?} partials={partials:?}");
}

/// Silence must not hallucinate words.
#[test]
fn eou_silence_produces_no_words() {
    let Some(dir) = model_dir() else {
        eprintln!("SKIP: no encoder.onnx under STT_EOU_MODEL_DIR");
        return;
    };
    let mut stt = ParakeetEouStt::load(&dir).expect("model should load");
    let silence = vec![0u8; 16_000 * 2 * 2]; // 2 s of digital silence
    let text = stt.transcribe(&silence, &mut |_| {});
    assert!(
        text.chars().filter(|c| c.is_alphabetic()).count() <= 2,
        "silence decoded as {text:?}"
    );
    eprintln!("silence -> {text:?}");
}
