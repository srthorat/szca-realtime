/// Shared ONNX Runtime plumbing for the gateway.
///
/// The gateway historically shipped energy-heuristic placeholders for VAD and
/// noise suppression. This module wires in the real `ort` (ONNX Runtime) crate
/// so those paths can run genuine model inference.
///
/// ORT is used in `load-dynamic` mode: the runtime shared library is located at
/// startup rather than linked at build time. Point the loader at your installed
/// library via the `ORT_DYLIB_PATH` environment variable, e.g. on macOS/Homebrew:
///   export ORT_DYLIB_PATH=/opt/homebrew/lib/libonnxruntime.dylib
/// If unset, `ort` searches the standard system library paths.

use std::sync::Once;

static INIT: Once = Once::new();

/// Initialize the ONNX Runtime environment exactly once for the process.
///
/// Safe to call repeatedly; only the first call has any effect. Returns Ok even
/// if already initialized. Any real initialization error is surfaced to the
/// caller so a misconfigured `ORT_DYLIB_PATH` fails loudly instead of silently
/// falling back to a stub.
pub fn init_ort() -> Result<(), String> {
    let mut result = Ok(());
    INIT.call_once(|| {
        // ort lazily initializes on first session build; we just validate that
        // the dynamic library can be resolved by constructing the environment.
        if let Err(e) = ort::init().with_name("szca_gateway").commit() {
            result = Err(format!("failed to initialize ONNX Runtime: {e}"));
        }
    });
    result
}
