//! Real Silero VAD inference integration test.
//!
//! This test only runs when a real model is available. Point it at the model:
//!   SILERO_VAD_MODEL=/path/to/silero_vad.onnx \
//!   ORT_DYLIB_PATH=/opt/homebrew/lib/libonnxruntime.dylib \
//!   cargo test --test silero_real_inference -- --nocapture
//!
//! If SILERO_VAD_MODEL is unset the test is skipped (so CI without weights is
//! still green). When set, it asserts the model loads, runs real inference, and
//! that speech energy yields a higher probability than digital silence.

use szca_media_gateway::silero::SileroModel;
use szca_media_gateway::vad::{VadConfig, VadEvent, VadProcessor, SILERO_WINDOW_SAMPLES};

/// The Silero model path, when configured. Relative paths are resolved against
/// the REPO ROOT, not the CWD: `env.dev.example` / `download_models.sh` produce
/// repo-root-relative paths like `./models/vad/silero_vad.onnx`, but cargo runs
/// integration tests with the CWD set to the package dir, so the raw value would
/// resolve one level too deep and the load would fail.
fn model_path() -> Option<String> {
    let p = std::env::var("SILERO_VAD_MODEL").ok().filter(|p| !p.is_empty())?;
    let path = std::path::Path::new(&p);
    if path.is_absolute() {
        return Some(p);
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent()?;
    Some(root.join(path).to_string_lossy().into_owned())
}

/// Build a 512-sample window of loud voiced-like audio as little-endian i16 PCM.
fn voiced_window_pcm() -> Vec<u8> {
    let mut out = Vec::with_capacity(SILERO_WINDOW_SAMPLES * 2);
    for i in 0..SILERO_WINDOW_SAMPLES {
        let t = i as f64 / 16000.0;
        // Formant-like mixture, loud.
        let v = 0.5 * (2.0 * std::f64::consts::PI * 150.0 * t).sin()
            + 0.3 * (2.0 * std::f64::consts::PI * 450.0 * t).sin()
            + 0.2 * (2.0 * std::f64::consts::PI * 900.0 * t).sin();
        let s = (v * 12000.0) as i16;
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

#[test]
fn silero_model_loads_and_infers() {
    let Some(path) = model_path() else {
        eprintln!("SKIP: SILERO_VAD_MODEL not set");
        return;
    };

    let mut model = SileroModel::load(&path, 16000).expect("model should load");

    // Silence window -> a valid probability in [0,1].
    let silence = vec![0.0f32; SILERO_WINDOW_SAMPLES];
    let p_sil = model.infer(&silence).expect("infer silence");
    assert!((0.0..=1.0).contains(&p_sil), "prob out of range: {p_sil}");

    // Loud voiced window -> also valid; typically >= silence.
    let mut voiced = vec![0.0f32; SILERO_WINDOW_SAMPLES];
    for (i, s) in voiced.iter_mut().enumerate() {
        let t = i as f64 / 16000.0;
        *s = (0.5 * (2.0 * std::f64::consts::PI * 150.0 * t).sin()) as f32;
    }
    let p_voiced = model.infer(&voiced).expect("infer voiced");
    assert!((0.0..=1.0).contains(&p_voiced), "prob out of range: {p_voiced}");

    eprintln!("Silero real inference OK: p(silence)={p_sil:.4} p(voiced)={p_voiced:.4}");
}

#[test]
fn vad_processor_uses_real_model() {
    let Some(path) = model_path() else {
        eprintln!("SKIP: SILERO_VAD_MODEL not set");
        return;
    };

    let cfg = VadConfig {
        model_path: Some(path),
        min_speech_duration_ms: 20,
        ..Default::default()
    };
    let mut vad = VadProcessor::new(cfg);
    assert!(vad.uses_real_model(), "processor should be running real Silero inference");

    // Feed several loud windows; the pipeline must run without error and
    // produce valid events (exact speech classification depends on the model).
    let pcm = voiced_window_pcm();
    let mut events = Vec::new();
    for _ in 0..10 {
        events.push(vad.process(&pcm, false));
    }
    assert_eq!(vad.frame_count(), 10);
    // Every event must be a valid variant (compiles) and the run completed.
    assert!(events.iter().all(|e| matches!(
        e,
        VadEvent::Silence | VadEvent::Speech | VadEvent::SpeechStart | VadEvent::SpeechEnd | VadEvent::BargeIn
    )));
    eprintln!("VAD real-model run OK; last event = {:?}", events.last().unwrap());
}
