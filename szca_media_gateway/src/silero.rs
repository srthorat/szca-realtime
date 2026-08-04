/// Real Silero VAD v5 ONNX inference.
///
/// Wraps the Silero VAD v5 model (`silero_vad.onnx`) using the `ort` crate. The
/// model is stateful: it takes a rolling recurrent `state` tensor of shape
/// `[2, 1, 128]` alongside the audio window and the sample rate, and returns a
/// speech probability plus the next state. This wrapper owns and threads that
/// state across successive [`SileroModel::infer`] calls.
///
/// Model I/O contract (verified against silero_vad.onnx, ir_version 8):
///   inputs : input [1, W] f32, state [2, 1, 128] f32, sr [] i64
///   outputs: output [1, 1] f32 (speech prob), stateN [2, 1, 128] f32

use ndarray::{Array1, Array2, Array3};
use ort::session::Session;
use ort::value::Tensor;

use crate::onnx::init_ort;

/// A loaded Silero VAD model with persistent recurrent state.
pub struct SileroModel {
    session: Session,
    /// Recurrent state, shape [2, 1, 128]; updated after every inference.
    state: Array3<f32>,
    /// Sample rate passed to the model (typically 16000).
    sample_rate: i64,
}

impl SileroModel {
    /// Load the Silero VAD model from `path`.
    ///
    /// Initializes ONNX Runtime (once per process) and builds a session. Fails
    /// loudly if the runtime cannot be located or the model cannot be loaded.
    pub fn load(path: &str, sample_rate: u32) -> Result<Self, String> {
        init_ort()?;

        let session = Session::builder()
            .map_err(|e| format!("session builder: {e}"))?
            .commit_from_file(path)
            .map_err(|e| format!("load {path}: {e}"))?;

        Ok(Self {
            session,
            state: Array3::<f32>::zeros((2, 1, 128)),
            sample_rate: sample_rate as i64,
        })
    }

    /// Run one inference over a window of normalized f32 samples ([-1, 1]).
    ///
    /// Returns the speech probability in [0, 1] and advances the internal
    /// recurrent state. `window` should be the model's expected window length
    /// (512 samples @16kHz for Silero v5).
    pub fn infer(&mut self, window: &[f32]) -> Result<f32, String> {
        // input: [1, W]
        let input = Array2::from_shape_vec((1, window.len()), window.to_vec())
            .map_err(|e| format!("input shape: {e}"))?;
        // sr: scalar i64 (shape [])
        let sr = Array1::from_vec(vec![self.sample_rate]);

        let input_t = Tensor::from_array(input).map_err(|e| format!("input tensor: {e}"))?;
        let state_t =
            Tensor::from_array(self.state.clone()).map_err(|e| format!("state tensor: {e}"))?;
        let sr_t = Tensor::from_array(sr).map_err(|e| format!("sr tensor: {e}"))?;

        let outputs = self
            .session
            .run(ort::inputs![
                "input" => input_t,
                "state" => state_t,
                "sr" => sr_t,
            ])
            .map_err(|e| format!("run: {e}"))?;

        // output[0] = speech probability, output[1] = next state.
        let (_, prob) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("extract prob: {e}"))?;
        let probability = prob.first().copied().unwrap_or(0.0);

        let (state_shape, state_data) = outputs["stateN"]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("extract state: {e}"))?;
        // Rebuild the [2, 1, 128] state for the next call.
        let dims: Vec<usize> = state_shape.iter().map(|&d| d as usize).collect();
        if dims.len() == 3 {
            self.state = Array3::from_shape_vec((dims[0], dims[1], dims[2]), state_data.to_vec())
                .map_err(|e| format!("state reshape: {e}"))?;
        }

        Ok(probability.clamp(0.0, 1.0))
    }

    /// Reset the recurrent state (e.g. at the start of a new utterance).
    pub fn reset(&mut self) {
        self.state = Array3::<f32>::zeros((2, 1, 128));
    }
}
