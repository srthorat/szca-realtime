/// Real DeepFilterNet3 (DFN3) noise-suppression inference.
///
/// DFN3 is a streaming speech-enhancement network shipped as THREE ONNX graphs
/// plus a `config.ini` (see the bitsydarel/upstream mirror):
///
///   enc.onnx      feat_erb[1,1,S,32], feat_spec[1,2,S,96]
///                   -> e0,e1,e2,e3 (skip features), emb, c0, lsnr
///   erb_dec.onnx  emb, e3,e2,e1,e0            -> m   (ERB gain mask, sigmoid)
///   df_dec.onnx   emb, c0                     -> coefs (deep-filter coefs), lsnr2
///
/// This module loads and runs those three graphs, **correctly chained**, using
/// the `ort` crate. That proves the real DFN3 network executes inference inside
/// the gateway.
///
/// ---------------------------------------------------------------------------
/// CORRECTNESS BOUNDARY (read before trusting the enhanced audio):
///
/// The neural stages here are the genuine trained model. However, DFN3's
/// front/back-end DSP — STFT (fft 960 / hop 480 @48 kHz), the exact 32-band ERB
/// filterbank, the 96-bin complex "DF" features, `norm_tau` normalization, the
/// 5-tap deep-filter application with 2-frame lookahead, iSTFT, and 16<->48 kHz
/// resampling — is reproduced from the DFN3 config/paper, NOT ported 1:1 from
/// the upstream `libDF` reference. The plumbing runs and produces finite audio,
/// but bit-exact / audio-quality PARITY with reference libDF is NOT validated
/// here (no reference output was available to A/B against). Treat the enhanced
/// output as functional, not reference-verified, until such an A/B test exists.
/// ---------------------------------------------------------------------------

use ndarray::{Array2, Array3, Array4};
use ort::session::Session;
use ort::value::Tensor;

use crate::onnx::init_ort;

/// DFN3 fixed model geometry (from config.ini `[df]`).
pub const DFN3_SR: u32 = 48_000;
pub const DFN3_FFT: usize = 960;
pub const DFN3_HOP: usize = 480;
pub const DFN3_FREQ_BINS: usize = DFN3_FFT / 2 + 1; // 481
pub const DFN3_NB_ERB: usize = 32;
pub const DFN3_NB_DF: usize = 96;

/// Paths to the three DFN3 ONNX stages.
#[derive(Debug, Clone)]
pub struct Dfn3Paths {
    pub enc: String,
    pub erb_dec: String,
    pub df_dec: String,
}

impl Dfn3Paths {
    /// Derive the three stage paths from a directory containing the standard
    /// `dfn3_enc.onnx` / `dfn3_erb_dec.onnx` / `dfn3_df_dec.onnx` files.
    pub fn in_dir(dir: &str) -> Self {
        let d = dir.trim_end_matches('/');
        Self {
            enc: format!("{d}/dfn3_enc.onnx"),
            erb_dec: format!("{d}/dfn3_erb_dec.onnx"),
            df_dec: format!("{d}/dfn3_df_dec.onnx"),
        }
    }
}

/// Loaded DFN3 network (three chained ONNX sessions).
pub struct Dfn3Model {
    enc: Session,
    erb_dec: Session,
    df_dec: Session,
}

/// Result of running one DFN3 frame block through the network.
#[derive(Debug, Clone)]
pub struct Dfn3Output {
    /// ERB gain mask `m`, flattened. Length = S * nb_erb (approx; last dim is
    /// model-defined). Values in [0,1].
    pub mask: Vec<f32>,
    /// Deep-filter coefficients `coefs`, flattened.
    pub coefs: Vec<f32>,
    /// Per-frame local SNR estimate (lsnr), one value per time step.
    pub lsnr: Vec<f32>,
}

impl Dfn3Model {
    /// Load all three DFN3 stages. Fails loudly if any stage cannot be loaded.
    pub fn load(paths: &Dfn3Paths) -> Result<Self, String> {
        init_ort()?;
        let build = |p: &str| -> Result<Session, String> {
            Session::builder()
                .map_err(|e| format!("session builder: {e}"))?
                .commit_from_file(p)
                .map_err(|e| format!("load {p}: {e}"))
        };
        Ok(Self {
            enc: build(&paths.enc)?,
            erb_dec: build(&paths.erb_dec)?,
            df_dec: build(&paths.df_dec)?,
        })
    }

    /// Run the full three-stage network for `S` time frames.
    ///
    /// * `feat_erb`  — shape [1,1,S,32] ERB band features (log-power, normalized)
    /// * `feat_spec` — shape [1,2,S,96] complex DF-band features (real, imag)
    ///
    /// Returns the ERB gain mask, deep-filter coefficients, and lsnr. This is
    /// where the real trained weights execute; feature construction and mask/
    /// coef application live in the (documented, not-parity-verified) DSP layer.
    pub fn run(
        &mut self,
        feat_erb: Array4<f32>,
        feat_spec: Array4<f32>,
    ) -> Result<Dfn3Output, String> {
        // ---- Stage 1: encoder ----
        let erb_t = Tensor::from_array(feat_erb).map_err(|e| format!("feat_erb: {e}"))?;
        let spec_t = Tensor::from_array(feat_spec).map_err(|e| format!("feat_spec: {e}"))?;
        let enc_out = self
            .enc
            .run(ort::inputs!["feat_erb" => erb_t, "feat_spec" => spec_t])
            .map_err(|e| format!("enc run: {e}"))?;

        // Extract encoder outputs we need to forward. We must copy them out
        // (owned) before building the next session's input tensors.
        let take = |name: &str, out: &ort::session::SessionOutputs| -> Result<(Vec<usize>, Vec<f32>), String> {
            let (shape, data) = out[name]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("extract {name}: {e}"))?;
            Ok((shape.iter().map(|&d| d as usize).collect(), data.to_vec()))
        };

        let (emb_shape, emb) = take("emb", &enc_out)?;
        let (e0_shape, e0) = take("e0", &enc_out)?;
        let (e1_shape, e1) = take("e1", &enc_out)?;
        let (e2_shape, e2) = take("e2", &enc_out)?;
        let (e3_shape, e3) = take("e3", &enc_out)?;
        let (c0_shape, c0) = take("c0", &enc_out)?;
        let (_lsnr_shape, lsnr) = take("lsnr", &enc_out)?;
        drop(enc_out);

        let mk4 = |shape: &[usize], data: Vec<f32>, name: &str| -> Result<Tensor<f32>, String> {
            let arr = Array4::from_shape_vec((shape[0], shape[1], shape[2], shape[3]), data)
                .map_err(|e| format!("{name} reshape: {e}"))?;
            Tensor::from_array(arr).map_err(|e| format!("{name} tensor: {e}"))
        };
        let mk3 = |shape: &[usize], data: Vec<f32>, name: &str| -> Result<Tensor<f32>, String> {
            let arr = Array3::from_shape_vec((shape[0], shape[1], shape[2]), data)
                .map_err(|e| format!("{name} reshape: {e}"))?;
            Tensor::from_array(arr).map_err(|e| format!("{name} tensor: {e}"))
        };

        // ---- Stage 2: ERB decoder -> mask m ----
        let erb_out = self
            .erb_dec
            .run(ort::inputs![
                "emb" => mk3(&emb_shape, emb.clone(), "emb")?,
                "e3" => mk4(&e3_shape, e3, "e3")?,
                "e2" => mk4(&e2_shape, e2, "e2")?,
                "e1" => mk4(&e1_shape, e1, "e1")?,
                "e0" => mk4(&e0_shape, e0, "e0")?,
            ])
            .map_err(|e| format!("erb_dec run: {e}"))?;
        let (_m_shape, mask) = {
            let (s, d) = erb_out["m"]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("extract m: {e}"))?;
            (s.to_vec(), d.to_vec())
        };
        drop(erb_out);

        // ---- Stage 3: DF decoder -> coefs ----
        let df_out = self
            .df_dec
            .run(ort::inputs![
                "emb" => mk3(&emb_shape, emb, "emb")?,
                "c0" => mk4(&c0_shape, c0, "c0")?,
            ])
            .map_err(|e| format!("df_dec run: {e}"))?;
        let coefs = {
            let (_s, d) = df_out["coefs"]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("extract coefs: {e}"))?;
            d.to_vec()
        };
        drop(df_out);

        Ok(Dfn3Output { mask, coefs, lsnr })
    }

    /// Convenience: run the network for `S` frames given raw feature vectors.
    /// `erb` must be S*32 long, `spec` must be 2*S*96 long (channel-major:
    /// [real block (S*96), imag block (S*96)]).
    pub fn run_flat(&mut self, s: usize, erb: &[f32], spec: &[f32]) -> Result<Dfn3Output, String> {
        if erb.len() != s * DFN3_NB_ERB {
            return Err(format!("erb len {} != S*{}", erb.len(), DFN3_NB_ERB));
        }
        if spec.len() != 2 * s * DFN3_NB_DF {
            return Err(format!("spec len {} != 2*S*{}", spec.len(), DFN3_NB_DF));
        }
        let feat_erb = Array4::from_shape_vec((1, 1, s, DFN3_NB_ERB), erb.to_vec())
            .map_err(|e| format!("feat_erb build: {e}"))?;
        let feat_spec = Array4::from_shape_vec((1, 2, s, DFN3_NB_DF), spec.to_vec())
            .map_err(|e| format!("feat_spec build: {e}"))?;
        self.run(feat_erb, feat_spec)
    }
}

/// Build a zero-initialized ERB feature block for `s` frames (test/util helper).
pub fn zero_erb(s: usize) -> Array4<f32> {
    Array4::<f32>::zeros((1, 1, s, DFN3_NB_ERB))
}

/// Build a zero-initialized complex-spec feature block for `s` frames.
pub fn zero_spec(s: usize) -> Array4<f32> {
    Array4::<f32>::zeros((1, 2, s, DFN3_NB_DF))
}

/// Placeholder for the 2D feature helper (kept for symmetry with callers).
pub fn erb_frame(_bins: &[f32]) -> Array2<f32> {
    Array2::<f32>::zeros((1, DFN3_NB_ERB))
}
