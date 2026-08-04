//! Real Sherpa Zipformer inference test.
//!
//! ```bash
//! ORT_DYLIB_PATH=/opt/homebrew/lib/libonnxruntime.dylib \
//! SHERPA_MODEL_DIR=$PWD/models/sherpa_zipformer \
//! cargo test --release --manifest-path szca_media_gateway/Cargo.toml \
//!   --test stt_zipformer_real_inference -- --nocapture --test-threads=1
//! ```

use szca_media_gateway::rt_pipeline::SttStage;
use szca_media_gateway::rt_stt_zipformer::SherpaZipformer;
use szca_media_gateway::stage_pool::StagePool;
use szca_media_gateway::stage_pools::{SttBackend, SttPoolAdapter};

fn resolve_from_repo_root(p: &str) -> Option<String> {
    let path = std::path::Path::new(p);
    if path.is_absolute() {
        return Some(p.to_string());
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent()?;
    Some(root.join(path).to_string_lossy().into_owned())
}

fn model_dir() -> Option<String> {
    let raw = std::env::var("SHERPA_MODEL_DIR").ok().filter(|d| !d.is_empty())?;
    let dir = resolve_from_repo_root(&raw)?;
    if !std::path::Path::new(&format!("{dir}/encoder.onnx")).is_file() {
        eprintln!("SKIP: SHERPA_MODEL_DIR is set but has no encoder.onnx: {dir}");
        return None;
    }
    Some(dir)
}

fn read_wav_mono(path: &str) -> Option<(Vec<f32>, u32)> {
    let b = std::fs::read(path).ok()?;
    if b.len() < 12 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return None;
    }
    let rd32 = |o: usize| u32::from_le_bytes([b[o], b[o+1], b[o+2], b[o+3]]) as usize;
    let rd16 = |o: usize| u16::from_le_bytes([b[o], b[o+1]]) as usize;
    let mut pos = 12;
    let (mut rate, mut channels) = (0u32, 0usize);
    let mut samples: Vec<f32> = Vec::new();
    while pos + 8 <= b.len() {
        let id = &b[pos..pos+4];
        let size = rd32(pos+4);
        let body = pos+8;
        if body+size > b.len() { break; }
        if id == b"fmt " && size >= 16 {
            channels = rd16(body+2);
            rate = rd32(body+4) as u32;
        } else if id == b"data" {
            let step = 2*channels;
            samples = b[body..body+size].chunks_exact(step)
                .map(|c| i16::from_le_bytes([c[0],c[1]]) as f32 / 32768.0).collect();
        }
        pos = body + size + (size & 1);
    }
    if samples.is_empty() || rate == 0 { None } else { Some((samples, rate)) }
}

fn to_16k(samples: &[f32], rate: u32) -> Vec<f32> {
    if rate == 16000 || samples.is_empty() { return samples.to_vec(); }
    let ratio = rate as f64 / 16000.0;
    let n = (samples.len() as f64 / ratio).floor() as usize;
    (0..n).map(|i| {
        let src = i as f64 * ratio;
        let i0 = src.floor() as usize;
        let i1 = (i0+1).min(samples.len()-1);
        let frac = (src - i0 as f64) as f32;
        samples[i0] * (1.0-frac) + samples[i1] * frac
    }).collect()
}

fn hello_world_pcm() -> Option<Vec<u8>> {
    let raw = std::env::var("STT_EOU_TEST_WAV")
        .ok().filter(|p| !p.is_empty())
        .unwrap_or_else(|| "./models/tts/kokoro_hello_sample.wav".to_string());
    let path = resolve_from_repo_root(&raw)?;
    let (samples, rate) = read_wav_mono(&path)?;
    let s16 = to_16k(&samples, rate);
    Some(s16.iter().flat_map(|&v| {
        let q = (v.clamp(-1.0,1.0) * 32767.0) as i16;
        q.to_le_bytes()
    }).collect())
}

#[test]
fn zipformer_model_loads() {
    let Some(dir) = model_dir() else {
        eprintln!("SKIP: SHERPA_MODEL_DIR not set");
        return;
    };
    let stt = SherpaZipformer::load(&dir).expect("model should load");
    assert_eq!(stt.transcript(), "");
    eprintln!("Zipformer loaded from {dir}");
}

#[test]
fn zipformer_decodes_hello_world() {
    let Some(dir) = model_dir() else {
        eprintln!("SKIP: SHERPA_MODEL_DIR not set");
        return;
    };
    let Some(pcm) = hello_world_pcm() else {
        eprintln!("SKIP: no hello-world fixture");
        return;
    };
    let mut stt = SherpaZipformer::load(&dir).expect("model should load");
    let mut partials = Vec::new();
    let text = stt.transcribe(&pcm, &mut |p| partials.push(p.to_string()));
    eprintln!("Zipformer transcript: {text:?} partials={partials:?}");
    assert!(!text.is_empty(), "transcript should not be empty");
    assert!(text.to_lowercase().contains("hello"), "expected 'hello' in transcript, got {text:?}");
    assert!(text.to_lowercase().contains("world"), "expected 'world' in transcript, got {text:?}");
}

#[test]
fn zipformer_decodes_faster_than_realtime() {
    let Some(dir) = model_dir() else {
        eprintln!("SKIP: SHERPA_MODEL_DIR not set");
        return;
    };
    let Some(pcm) = hello_world_pcm() else {
        eprintln!("SKIP: no hello-world fixture");
        return;
    };
    let mut stt = SherpaZipformer::load(&dir).expect("model should load");
    let _ = stt.transcribe(&pcm, &mut |_| {}); // warmup
    let audio_secs = (pcm.len() / 2) as f64 / 16000.0;
    let t0 = std::time::Instant::now();
    let text = stt.transcribe(&pcm, &mut |_| {});
    let elapsed = t0.elapsed().as_secs_f64();
    let rtf = audio_secs / elapsed;
    eprintln!("Zipformer: {audio_secs:.2}s audio in {elapsed:.3}s => {rtf:.1}x realtime ({text:?})");
    assert!(rtf > 1.0, "RTF {rtf:.1} must be > 1.0 for streaming viability");
}

#[test]
fn zipformer_silence_produces_no_words() {
    let Some(dir) = model_dir() else {
        eprintln!("SKIP: SHERPA_MODEL_DIR not set");
        return;
    };
    let mut stt = SherpaZipformer::load(&dir).expect("model should load");
    let silence = vec![0u8; 16000 * 2 * 3]; // 3 s
    let text = stt.transcribe(&silence, &mut |_| {});
    assert!(text.chars().filter(|c| c.is_alphabetic()).count() <= 2, "silence decoded as {text:?}");
    eprintln!("silence -> {text:?}");
}

#[test]
fn zipformer_through_stage_pool() {
    let Some(dir) = model_dir() else {
        eprintln!("SKIP: SHERPA_MODEL_DIR not set");
        return;
    };
    let Some(pcm) = hello_world_pcm() else {
        eprintln!("SKIP: no hello-world fixture");
        return;
    };
    let pool = StagePool::build("zipf-test", 1, |_| {
        SherpaZipformer::load(&dir).map(SttBackend::Zipformer)
    }).expect("pool should build");
    let mut adapter = SttPoolAdapter::new(&pool);
    let mut partials = Vec::new();
    let text = adapter.transcribe(&pcm, &mut |p| partials.push(p.to_string()));
    eprintln!("pool path: {text:?} partials={partials:?}");
    assert!(!text.is_empty(), "pool path returned empty");
}

#[test]
fn zipformer_chunked_feed_works() {
    let Some(dir) = model_dir() else {
        eprintln!("SKIP: SHERPA_MODEL_DIR not set");
        return;
    };
    let Some(pcm) = hello_world_pcm() else {
        eprintln!("SKIP: no hello-world fixture");
        return;
    };
    let mut stt = SherpaZipformer::load(&dir).expect("model should load");
    // Feed 20ms PCM frames (640 bytes) like a WebSocket client would.
    for frame in pcm.chunks(640) {
        stt.push_pcm(frame);
    }
    stt.flush();
    let text = stt.transcript().trim().to_string();
    assert!(!text.is_empty(), "20ms-frame feed produced empty transcript");
    assert!(text.to_lowercase().contains("hello"), "expected 'hello', got {text:?}");
    eprintln!("20ms-frame feed: {text:?}");
}
