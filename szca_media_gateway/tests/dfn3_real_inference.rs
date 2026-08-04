//! Real DeepFilterNet3 three-stage inference integration test.
//!
//! Runs the genuine enc -> erb_dec + df_dec network chain. Requires the three
//! ONNX stages; skipped if DFN3_MODEL_DIR is unset.
//!
//!   DFN3_MODEL_DIR=/path/to/models \
//!   ORT_DYLIB_PATH=/opt/homebrew/lib/libonnxruntime.dylib \
//!   cargo test --test dfn3_real_inference -- --nocapture

use szca_media_gateway::dfn3::{Dfn3Model, Dfn3Paths, DFN3_NB_DF, DFN3_NB_ERB};
use szca_media_gateway::dfn3_dsp::Dfn3Dsp;

/// The DFN3 stage directory, when configured. Relative paths are resolved
/// against the REPO ROOT, not the CWD: `download_models.sh` writes
/// `./models/dfn3`, but cargo runs integration tests with the CWD set to the
/// package dir, so the raw value would resolve one level too deep.
fn model_dir() -> Option<String> {
    let d = std::env::var("DFN3_MODEL_DIR").ok().filter(|p| !p.is_empty())?;
    let path = std::path::Path::new(&d);
    if path.is_absolute() {
        return Some(d);
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent()?;
    Some(root.join(path).to_string_lossy().into_owned())
}

#[test]
fn dfn3_three_stage_chain_runs() {
    let Some(dir) = model_dir() else {
        eprintln!("SKIP: DFN3_MODEL_DIR not set");
        return;
    };

    let paths = Dfn3Paths::in_dir(&dir);
    let mut model = Dfn3Model::load(&paths).expect("all three DFN3 stages should load");

    // Two time frames of deterministic non-zero features.
    let s = 2usize;
    let erb: Vec<f32> = (0..s * DFN3_NB_ERB)
        .map(|i| 0.01 * (i % 7) as f32)
        .collect();
    let spec: Vec<f32> = (0..2 * s * DFN3_NB_DF)
        .map(|i| 0.005 * ((i % 11) as f32 - 5.0))
        .collect();

    let out = model.run_flat(s, &erb, &spec).expect("network should run");

    assert!(!out.mask.is_empty(), "ERB mask should be non-empty");
    assert!(!out.coefs.is_empty(), "deep-filter coefs should be non-empty");
    assert!(!out.lsnr.is_empty(), "lsnr should be non-empty");

    // Mask is a sigmoid output -> must be within [0,1] and finite.
    assert!(
        out.mask.iter().all(|&v| v.is_finite() && (0.0..=1.0).contains(&v)),
        "ERB mask values must be finite and in [0,1]"
    );
    assert!(out.coefs.iter().all(|v| v.is_finite()), "coefs must be finite");
    assert!(out.lsnr.iter().all(|v| v.is_finite()), "lsnr must be finite");

    eprintln!(
        "DFN3 real 3-stage inference OK: mask_len={} coefs_len={} lsnr={:?}",
        out.mask.len(),
        out.coefs.len(),
        out.lsnr
    );
}

#[test]
fn dfn3_dsp_pipeline_runs() {
    let Some(dir) = model_dir() else {
        eprintln!("SKIP: DFN3_MODEL_DIR not set");
        return;
    };

    let paths = Dfn3Paths::in_dir(&dir);
    let model = match Dfn3Model::load(&paths) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("SKIP: DFN3 model not available ({e})");
            return;
        }
    };

    let mut dsp = Dfn3Dsp::load(model);

    // Generate 80ms of PCM16 sine wave at 16 kHz (1280 samples = 2560 bytes).
    let sample_rate = 16000;
    let duration_ms = 80;
    let num_samples = sample_rate * duration_ms / 1000;
    let mut pcm: Vec<i16> = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let t = i as f64 / sample_rate as f64;
        let v = (2.0 * std::f64::consts::PI * 440.0 * t).sin() as f32;
        pcm.push((v * 8000.0) as i16);
    }

    // Process through the DSP pipeline. The first call may not produce output
    // (Dfn3Dsp buffers until it has a full block).
    let out = dsp.process(&pcm);
    eprintln!(
        "DFN3 DSP pipeline: in={} i16 samples, out={} i16 samples",
        num_samples,
        out.len(),
    );

    // DSP should produce some output (even if just zeros, it shouldn't crash).
    assert!(!out.is_empty(), "DSP should produce non-empty output");

    // The output should be in valid i16 range.
    assert!(
        out.iter().any(|&s| s != 0),
        "DSP output should not be all-zeros"
    );
}
