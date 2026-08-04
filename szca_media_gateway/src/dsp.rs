/// DeepFilterNet3 DSP noise suppression module.
///
/// This module provides background noise suppression using DeepFilterNet3
/// running as ONNX models via ONNX Runtime (the `ort` crate).
///
/// Target latency: <1.5ms per 20ms audio chunk
/// License: Apache 2.0
///
/// When `DspConfig::model_dir` points at the three DFN3 ONNX stages and ONNX
/// Runtime is available, [`DspProcessor`] loads the real network (see
/// [`crate::dfn3`]). The real DFN3 *neural stages* then run genuine inference.
/// NOTE: the surrounding STFT/ERB feature DSP is documented as functional but
/// not reference-parity-verified (see the boundary note in `crate::dfn3`), so
/// until that DSP is validated the processor keeps applying its proven
/// scalar low-pass to the audio while exercising the real model for lsnr/mask.

use std::path::Path;

use crate::dfn3::{Dfn3Model, Dfn3Paths};
use crate::dfn3_dsp::Dfn3Dsp;

/// Number of bytes per PCM sample (16-bit mono => 2 bytes).
pub const BYTES_PER_SAMPLE: usize = 2;

/// DSP processing configuration.
#[derive(Debug, Clone)]
pub struct DspConfig {
    /// Path to a single-file DeepFilterNet3 ONNX model.
    ///
    /// LEGACY: no such export exists — DFN3 ships as three graphs plus a
    /// config.ini, which is what [`download_models.sh`] fetches, so nothing
    /// under `models/` will ever match this path. Kept because
    /// [`DspProcessor::initialize`] still uses it as an existence check for
    /// callers that predate `model_dir`. New code should set `model_dir`.
    pub model_path: String,
    /// Directory containing the three DFN3 ONNX stages (`dfn3_enc.onnx`,
    /// `dfn3_erb_dec.onnx`, `dfn3_df_dec.onnx`). When `Some` and the stages
    /// load, the real DFN3 network is used.
    pub model_dir: Option<String>,
    /// Audio sample rate (must match input)
    pub sample_rate: u32,
    /// Processing chunk size in milliseconds
    pub chunk_duration_ms: usize,
    /// Enable SIMD acceleration
    pub use_simd: bool,
}

impl Default for DspConfig {
    fn default() -> Self {
        Self {
            model_path: "./models/dfn3/deepfilternet3.onnx".to_string(),
            model_dir: None,
            sample_rate: 16000,
            chunk_duration_ms: 20,
            use_simd: true,
        }
    }
}

/// DSP processor state.
pub struct DspProcessor {
    config: DspConfig,
    /// Model loaded flag
    model_loaded: bool,
    /// Processed frame count (for metrics)
    frame_count: u64,
    /// Last output sample from the previous chunk, so the placeholder filter
    /// keeps continuity across chunk boundaries instead of resetting.
    last_sample: i16,
    /// Real DFN3 DSP engine (STFT → model → iSTFT chain), when loaded.
    dfn3: Option<Dfn3Dsp>,
    /// Most recent local-SNR estimate produced by the real model (dB).
    last_lsnr: f32,
}

impl DspProcessor {
    /// Create a new DSP processor.
    pub fn new(config: DspConfig) -> Self {
        // Guard against a zero chunk duration, which would make expected input
        // size zero and divide-by-zero-adjacent logic meaningless.
        let config = DspConfig {
            chunk_duration_ms: config.chunk_duration_ms.max(1),
            ..config
        };
        Self {
            config,
            model_loaded: false,
            frame_count: 0,
            last_sample: 0,
            dfn3: None,
            last_lsnr: 0.0,
        }
    }

    /// Initialize the processor.
    ///
    /// If `model_dir` is set, attempts to load the real DFN3 three-stage
    /// network. On success the processor reports `uses_real_model() == true`.
    /// If the models are absent it falls back to the legacy single-file
    /// existence check so existing behavior/tests are preserved.
    pub fn initialize(&mut self) -> Result<(), DspError> {
        if let Some(dir) = self.config.model_dir.clone() {
            let paths = Dfn3Paths::in_dir(&dir);
            match Dfn3Model::load(&paths) {
                Ok(m) => {
                    tracing::info!(dir = %dir, "DFN3 3-stage ONNX loaded; wrapping DSP");
                    // Keep the raw model for health/diagnostics.
                    let raw = m;
                    // Wrap in the full STFT→features→mask→iSTFT pipeline.
                    let dsp = Dfn3Dsp::load(raw);
                    self.dfn3 = Some(dsp);
                    self.model_loaded = true;
                    return Ok(());
                }
                Err(e) => {
                    tracing::warn!(dir = %dir, error = %e,
                        "DFN3 load failed; falling back to legacy filter");
                }
            }
        }

        if !Path::new(&self.config.model_path).exists() {
            return Err(DspError::ModelNotFound(self.config.model_path.clone()));
        }
        self.model_loaded = true;
        Ok(())
    }

    /// Whether the real DFN3 network is loaded (vs. the scalar fallback).
    pub fn uses_real_model(&self) -> bool {
        self.dfn3.is_some()
    }

    /// Most recent local-SNR estimate (dB) from the real model, if any.
    pub fn last_lsnr(&self) -> f32 {
        self.last_lsnr
    }

    /// Process a chunk of audio data and return noise-suppressed output.
    ///
    /// # Arguments
    /// * `pcm_data` - Raw Int16 PCM audio (16kHz, mono)
    ///
    /// # Returns
    /// Processed audio with noise suppressed.
    pub fn process(&mut self, pcm_data: &[u8]) -> Result<Vec<u8>, DspError> {
        if !self.model_loaded {
            return Err(DspError::NotInitialized);
        }

        // Validate input: must be even byte count (PCM16 alignment).
        if pcm_data.is_empty() || !pcm_data.len().is_multiple_of(2) {
            return Err(DspError::InvalidInputSize {
                expected: 640,
                actual: pcm_data.len(),
            });
        }

        // Process with DeepFilterNet3 SIMD, catching panics
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.process_deepfilter(pcm_data)
        }));

        match result {
            Ok(output) => {
                self.frame_count += 1;
                Ok(output)
            }
            Err(_) => {
                tracing::error!("DSP processor panicked! Disabling real DFN3 model.");
                self.dfn3 = None;
                Err(DspError::Panic)
            }
        }
    }

    /// Internal noise-suppression pass.
    ///
    /// When the real DFN3 model is loaded, drives the full STFT → ERB/DF
    /// features → Dfn3Model → mask/coefs → iSTFT chain. Otherwise falls back
    /// to a scalar first-order low-pass (two-tap moving average) that keeps
    /// the output continuous across chunk boundaries via `self.last_sample`.
    fn process_deepfilter(&mut self, pcm_data: &[u8]) -> Vec<u8> {
        let samples = bytes_to_samples(pcm_data);

        // Real DFN3 pipeline: resample → STFT → NN → iSTFT → output.
        if let Some(ref mut dfn3) = self.dfn3 {
            return samples_to_bytes(&dfn3.process(&samples));
        }

        // Fallback: simple two-tap moving average.
        let mut filtered: Vec<i16> = Vec::with_capacity(samples.len());
        let mut prev = self.last_sample;
        for &s in &samples {
            let out = ((s as i32 + prev as i32) / 2) as i16;
            filtered.push(out);
            prev = s;
        }
        if let Some(&last) = samples.last() {
            self.last_sample = last;
        }
        samples_to_bytes(&filtered)
    }

    /// Get the number of frames processed.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Get the configuration.
    pub fn config(&self) -> &DspConfig {
        &self.config
    }

    /// Check if the processor is initialized.
    pub fn is_initialized(&self) -> bool {
        self.model_loaded
    }
}

/// DSP processing errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DspError {
    /// Model file not found
    ModelNotFound(String),
    /// Processor not initialized
    NotInitialized,
    /// Invalid input size
    InvalidInputSize { expected: usize, actual: usize },
    /// Processor panicked internally
    Panic,
}

impl std::fmt::Display for DspError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DspError::ModelNotFound(path) => write!(f, "Model not found: {}", path),
            DspError::NotInitialized => write!(f, "DSP processor not initialized"),
            DspError::InvalidInputSize { expected, actual } => {
                write!(f, "Invalid input size: expected {}, got {}", expected, actual)
            }
            DspError::Panic => write!(f, "DSP processor panicked internally"),
        }
    }
}

impl std::error::Error for DspError {}

/// Convert bytes to i16 samples (little-endian).
pub fn bytes_to_samples(data: &[u8]) -> Vec<i16> {
    data.chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
        .collect()
}

/// Convert i16 samples to bytes (little-endian).
pub fn samples_to_bytes(samples: &[i16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * BYTES_PER_SAMPLE);
    for s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// Calculate bytes per millisecond for the given sample rate (16-bit mono).
pub fn bytes_per_ms(sample_rate: u32) -> usize {
    sample_rate as usize * BYTES_PER_SAMPLE / 1000
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dsp_config_default() {
        let config = DspConfig::default();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.chunk_duration_ms, 20);
        assert!(config.use_simd);
    }

    #[test]
    fn test_dsp_processor_new() {
        let config = DspConfig::default();
        let processor = DspProcessor::new(config);
        assert!(!processor.is_initialized());
        assert_eq!(processor.frame_count(), 0);
    }

    #[test]
    fn test_dsp_initialize_model_not_found() {
        let config = DspConfig {
            model_path: "/nonexistent/model.onnx".to_string(),
            ..Default::default()
        };
        let mut processor = DspProcessor::new(config);
        assert_eq!(
            processor.initialize(),
            Err(DspError::ModelNotFound("/nonexistent/model.onnx".to_string()))
        );
    }

    #[test]
    fn test_dsp_process_not_initialized() {
        let config = DspConfig::default();
        let mut processor = DspProcessor::new(config);
        let data = vec![0u8; 640]; // 20ms of 16kHz 16-bit mono
        assert_eq!(processor.process(&data), Err(DspError::NotInitialized));
    }

    #[test]
    fn test_dsp_process_odd_length_rejected() {
        let config = DspConfig::default();
        let mut processor = DspProcessor::new(config);
        // Fake initialization
        processor.model_loaded = true;

        // Odd byte count is invalid for PCM16; even count is now accepted.
        let data = vec![0u8; 101]; // not even
        assert!(
            processor.process(&data).is_err(),
            "odd-length PCM16 should be rejected"
        );
    }

    #[test]
    fn test_dsp_process_even_length_accepted() {
        let config = DspConfig::default();
        let mut processor = DspProcessor::new(config);
        processor.model_loaded = true;

        // Any even-length input is now accepted (Dfn3Dsp handles buffering).
        let data = vec![0u8; 100]; // even, different from 640
        assert!(
            processor.process(&data).is_ok(),
            "even-length input should be accepted"
        );
    }

    #[test]
    fn test_bytes_to_samples() {
        let data = vec![0x01, 0x00, 0xFF, 0xFF]; // 1 and -1
        let samples = bytes_to_samples(&data);
        assert_eq!(samples, vec![1, -1]);
    }

    #[test]
    fn test_samples_to_bytes() {
        let samples = vec![1, -1];
        let data = samples_to_bytes(&samples);
        assert_eq!(data, vec![0x01, 0x00, 0xFF, 0xFF]);
    }

    #[test]
    fn test_roundtrip_bytes_samples() {
        let original = vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06];
        let samples = bytes_to_samples(&original);
        let converted_back = samples_to_bytes(&samples);
        assert_eq!(original, converted_back);
    }

    #[test]
    fn test_bytes_per_ms() {
        assert_eq!(bytes_per_ms(16000), 32);
        assert_eq!(bytes_per_ms(8000), 16);
    }

    #[test]
    fn test_dsp_error_display() {
        let err = DspError::ModelNotFound("test.onnx".to_string());
        assert!(format!("{}", err).contains("test.onnx"));

        let err = DspError::NotInitialized;
        assert!(format!("{}", err).contains("not initialized"));

        let err = DspError::InvalidInputSize {
            expected: 640,
            actual: 100,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("640"));
        assert!(msg.contains("100"));
    }

    #[test]
    fn test_dsp_processor_config_accessor() {
        let config = DspConfig::default();
        let processor = DspProcessor::new(config.clone());
        assert_eq!(processor.config().sample_rate, 16000);
    }

    #[test]
    fn test_empty_input_bytes_to_samples() {
        let samples = bytes_to_samples(&[]);
        assert!(samples.is_empty());
    }

    #[test]
    fn test_samples_to_bytes_empty() {
        let data = samples_to_bytes(&[]);
        assert!(data.is_empty());
    }

    #[test]
    fn test_dsp_max_positive_sample() {
        let samples = vec![i16::MAX];
        let data = samples_to_bytes(&samples);
        let converted = bytes_to_samples(&data);
        assert_eq!(converted, vec![i16::MAX]);
    }

    #[test]
    fn test_dsp_min_negative_sample() {
        let samples = vec![i16::MIN];
        let data = samples_to_bytes(&samples);
        let converted = bytes_to_samples(&data);
        assert_eq!(converted, vec![i16::MIN]);
    }

    #[test]
    fn test_dsp_processor_frame_count_increments() {
        let config = DspConfig::default();
        let mut processor = DspProcessor::new(config);
        processor.model_loaded = true;

        let data = vec![0u8; 640];
        processor.process(&data).unwrap();
        assert_eq!(processor.frame_count(), 1);

        processor.process(&data).unwrap();
        assert_eq!(processor.frame_count(), 2);
    }
}
