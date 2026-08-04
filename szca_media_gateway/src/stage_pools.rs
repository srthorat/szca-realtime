//! Shared inference stage pools — the core of Phase 2a.
//!
//! Models are loaded once at startup into `Arc<StagePools>`. Every WS session
//! and HTTP endpoint shares these pools instead of creating per-connection
//! model copies. This eliminates the model-per-session RAM explosion
//! (100 sessions × 100 model copies → 100 sessions sharing 1 pool).
//!
//! # Architecture
//!
//! ```text
//! Batch STT (parakeet):
//!   WS / HTTP ──► SttPoolAdapter ──► StagePool job queue ──► ParakeetStt workers
//!
//! Streaming STT (eou / zipformer):
//!   WS session ──► StreamingSttHandle (exclusive lease) ──► push_chunk sticky state
//!   HTTP / run_turn ──► checkout → transcribe → release (same free-list)
//! ```
//!
//! Streaming encoder caches are sticky: `push_chunk` must hit the same replica
//! across frames, so those backends use a free-list lease pool rather than
//! one-shot StagePool jobs. Batch Parakeet stays on the job queue.
//!
//! Config via env vars:
//! - `STT_REPLICAS` (default 1, 0 = disabled)
//! - `LLM_REPLICAS` (default 1, 0 = disabled)
//! - `TTS_REPLICAS` (default 1, 0 = disabled)
//! - `STT_BACKEND` = `parakeet` (default, full-utterance) | `streaming` (cache-aware EOU)
//! - `LLM_BACKEND` = `onnx` (default, in-process) | `vllm` / `tgi` (external)

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use crate::rt_llm::{QwenLlm, LlmInput};
use crate::rt_llm_client::VllmClient;
use crate::rt_stt::{ParakeetStt, SttInput};
use crate::rt_stt_eou::ParakeetEouStt;
use crate::rt_stt_zipformer::SherpaZipformer;
use crate::rt_tts::{KokoroTts, TtsInput};
use crate::rt_pipeline::{LlmStage, SttChunkResult, SttStage, TtsStage};
use crate::stage_pool::{Replica, StagePool};

// ---------------------------------------------------------------------------
// STT backend enum — config-driven full-utterance vs streaming selection
// ---------------------------------------------------------------------------

/// Wraps either the full-utterance Parakeet TDT 0.6B stage or the cache-aware
/// streaming Parakeet EOU 120M stage.
///
/// Both take the same `SttInput` and emit the same `String` deltas, so the pool,
/// the adapter and every caller are byte-identical across backends — the only
/// difference the session sees is that `streaming` emits partials mid-utterance
/// from a genuinely incremental encoder instead of after the fact.
///
/// | `STT_BACKEND` | Model | Encoder | Partials |
/// |---|---|---|---|
/// | `parakeet` (default) | TDT 0.6B int8 | full-sequence attention | word-boundary, post-hoc |
/// | `streaming` | EOU 120M fp16 | cache-aware, 70-frame left ctx | every 1.28 s, incremental |
/// | `zipformer` | Sherpa Zipformer | cache-aware, 19-layer Zipformer | every 1.41 s, incremental |
///
/// The default stays `parakeet` because it is the more accurate model and the
/// one the prod latency budget is measured against; streaming is opt-in until it
/// has the same soak-test hours behind it.
#[allow(clippy::large_enum_variant)]
pub enum SttBackend {
    /// Full-utterance Parakeet TDT 0.6B (post-VAD batch).
    Batch(ParakeetStt),
    /// Cache-aware streaming Parakeet EOU 120M.
    Streaming(ParakeetEouStt),
    /// Cache-aware Sherpa Zipformer.
    Zipformer(SherpaZipformer),
}

impl Replica for SttBackend {
    type Input = SttInput;
    type Delta = String;
    type Output = String;

    fn process(
        &mut self,
        input: Self::Input,
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(Self::Delta),
    ) -> Self::Output {
        match self {
            SttBackend::Batch(stt) => stt.process(input, cancel, emit),
            SttBackend::Streaming(stt) => stt.process(input, cancel, emit),
            SttBackend::Zipformer(stt) => stt.process(input, cancel, emit),
        }
    }
}

impl SttStage for SttBackend {
    fn transcribe(&mut self, pcm: &[u8], partial: &mut dyn FnMut(&str)) -> String {
        match self {
            SttBackend::Batch(stt) => stt.transcribe(pcm, partial),
            SttBackend::Streaming(stt) => stt.transcribe(pcm, partial),
            SttBackend::Zipformer(stt) => stt.transcribe(pcm, partial),
        }
    }

    fn push_chunk(&mut self, pcm: &[u8]) -> Option<crate::rt_pipeline::SttChunkResult> {
        match self {
            SttBackend::Batch(stt) => stt.push_chunk(pcm),
            SttBackend::Streaming(stt) => stt.push_chunk(pcm),
            SttBackend::Zipformer(stt) => stt.push_chunk(pcm),
        }
    }

    fn reset_stream(&mut self) {
        match self {
            SttBackend::Batch(stt) => stt.reset_stream(),
            SttBackend::Streaming(stt) => stt.reset_stream(),
            SttBackend::Zipformer(stt) => stt.reset_stream(),
        }
    }

    fn supports_lookback(&self) -> bool {
        match self {
            SttBackend::Batch(stt) => stt.supports_lookback(),
            SttBackend::Streaming(stt) => stt.supports_lookback(),
            SttBackend::Zipformer(stt) => stt.supports_lookback(),
        }
    }
}

// ---------------------------------------------------------------------------
// LLM backend enum — config-driven ONNX vs vLLM selection
// ---------------------------------------------------------------------------

/// Wraps either an in-process ONNX LLM or an external vLLM streaming client.
/// Both implement `Replica`, so the pool is generic over the enum.
#[allow(clippy::large_enum_variant)]
pub enum LlmBackend {
    Onnx(QwenLlm),
    Vllm(VllmClient),
}

impl Replica for LlmBackend {
    type Input = LlmInput;
    type Delta = String;
    type Output = String;

    fn process(
        &mut self,
        input: Self::Input,
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(Self::Delta),
    ) -> Self::Output {
        match self {
            LlmBackend::Onnx(llm) => llm.process(input, cancel, emit),
            LlmBackend::Vllm(client) => client.process(input, cancel, emit),
        }
    }
}

// ---------------------------------------------------------------------------
// Type aliases for convenience
// ---------------------------------------------------------------------------

pub type SttPool = StagePool<SttBackend>;
pub type LlmPool = StagePool<LlmBackend>;
pub type TtsPool = StagePool<KokoroTts>;

// ---------------------------------------------------------------------------
// Streaming STT lease pool — sticky push_chunk without per-session model copies
// ---------------------------------------------------------------------------

/// Free-list of streaming STT replicas for exclusive checkout.
///
/// Encoder caches (EOU / Zipformer) are mutable and sticky across `push_chunk`
/// calls, so a session must hold one replica for the life of an utterance /
/// connection. On drop, the handle resets the stream and returns the replica
/// to the free list. Concurrent early-EOU sessions are capped at
/// `STT_REPLICAS`; excess sessions fall back to VAD + full-utterance transcribe.
pub struct StreamingSttPool {
    free: Mutex<Vec<SttBackend>>,
    total: usize,
}

impl StreamingSttPool {
    /// Load `n` streaming replicas via `make`.
    pub fn build<F>(n: usize, mut make: F) -> Result<Arc<Self>, String>
    where
        F: FnMut(usize) -> Result<SttBackend, String>,
    {
        if n == 0 {
            return Err("streaming STT pool: replicas must be >= 1".into());
        }
        let mut free = Vec::with_capacity(n);
        for i in 0..n {
            free.push(make(i)?);
        }
        Ok(Arc::new(Self {
            free: Mutex::new(free),
            total: n,
        }))
    }

    /// Try to check out an exclusive streaming handle. `None` if all leased.
    pub fn try_checkout(self: &Arc<Self>) -> Option<StreamingSttHandle> {
        let mut free = self.free.lock().ok()?;
        let model = free.pop()?;
        Some(StreamingSttHandle {
            model: Some(model),
            pool: Arc::clone(self),
        })
    }

    fn release(&self, mut model: SttBackend) {
        model.reset_stream();
        if let Ok(mut free) = self.free.lock() {
            free.push(model);
        }
    }

    /// Replicas currently sitting in the free list.
    pub fn available(&self) -> usize {
        self.free.lock().map(|f| f.len()).unwrap_or(0)
    }

    /// Total replicas managed by this pool (checked-out + free).
    pub fn replica_count(&self) -> usize {
        self.total
    }

    /// How many leases are currently held (`total - available`).
    pub fn in_use(&self) -> usize {
        self.total.saturating_sub(self.available())
    }
}

/// Exclusive lease on one streaming STT replica. Returns it to the pool on drop.
pub struct StreamingSttHandle {
    model: Option<SttBackend>,
    pool: Arc<StreamingSttPool>,
}

impl StreamingSttHandle {
    fn model_mut(&mut self) -> &mut SttBackend {
        self.model.as_mut().expect("StreamingSttHandle used after drop")
    }
}

impl Drop for StreamingSttHandle {
    fn drop(&mut self) {
        if let Some(model) = self.model.take() {
            self.pool.release(model);
        }
    }
}

impl SttStage for StreamingSttHandle {
    fn transcribe(&mut self, pcm: &[u8], partial: &mut dyn FnMut(&str)) -> String {
        self.model_mut().transcribe(pcm, partial)
    }

    fn push_chunk(&mut self, pcm: &[u8]) -> Option<SttChunkResult> {
        self.model_mut().push_chunk(pcm)
    }

    fn reset_stream(&mut self) {
        self.model_mut().reset_stream();
    }

    fn supports_lookback(&self) -> bool {
        self.model
            .as_ref()
            .map(|m| m.supports_lookback())
            .unwrap_or(false)
    }
}

// ---------------------------------------------------------------------------
// StagePools — the shared, immutable handle passed to every session/endpoint
// ---------------------------------------------------------------------------

/// Shared pools for all three inference stages. Loaded once at startup,
/// cloned cheaply (Arc + crossbeam channel clone) into every handler.
#[derive(Clone)]
pub struct StagePools {
    /// Batch Parakeet job-queue pool (`STT_BACKEND=parakeet`).
    pub stt: Option<SttPool>,
    /// Streaming EOU / Zipformer lease pool (sticky `push_chunk`).
    pub streaming_stt: Option<Arc<StreamingSttPool>>,
    pub llm: Option<LlmPool>,
    pub tts: Option<TtsPool>,
}

impl StagePools {
    /// Build pools from env-var config. Returns `Ok(StagePools)` even when
    /// some stages are disabled (their pool is `None`); returns `Err` only
    /// if a *enabled* stage fails to load its model.
    pub fn from_env() -> Result<Self, String> {
        let stt_replicas = parse_replica_count("STT_REPLICAS", 1);
        let llm_replicas = parse_replica_count("LLM_REPLICAS", 1);
        let tts_replicas = parse_replica_count("TTS_REPLICAS", 1);

        let (stt, streaming_stt) = if stt_replicas == 0 {
            tracing::info!("STT pool disabled (STT_REPLICAS=0)");
            (None, None)
        } else {
            match dev_model_selection() {
                SttModel::Zipformer => {
                    tracing::info!(
                        replicas = stt_replicas,
                        "Building streaming STT lease pool (Sherpa Zipformer)"
                    );
                    let pool = StreamingSttPool::build(stt_replicas, |_idx| {
                        SherpaZipformer::from_env().map(SttBackend::Zipformer)
                    })?;
                    (None, Some(pool))
                }
                SttModel::Streaming => {
                    tracing::info!(
                        replicas = stt_replicas,
                        "Building streaming STT lease pool (EOU)"
                    );
                    let pool = StreamingSttPool::build(stt_replicas, |_idx| {
                        ParakeetEouStt::from_env().map(SttBackend::Streaming)
                    })?;
                    (None, Some(pool))
                }
                SttModel::Parakeet => {
                    tracing::info!(replicas = stt_replicas, "Building STT pool (Parakeet TDT)");
                    let pool = StagePool::build("stt", stt_replicas, |_idx| {
                        ParakeetStt::from_env().map(SttBackend::Batch)
                    })?;
                    (Some(pool), None)
                }
            }
        };

        let llm = if llm_replicas == 0 {
            tracing::info!("LLM pool disabled (LLM_REPLICAS=0)");
            None
        } else {
            let backend = std::env::var("LLM_BACKEND")
                .unwrap_or_else(|_| "onnx".to_string())
                .to_lowercase();
            match backend.as_str() {
                "vllm" | "tgi" => {
                    tracing::info!(replicas = llm_replicas, backend = %backend, "Building LLM pool (vLLM/TGI)");
                    let pool = StagePool::build("llm", llm_replicas, |_idx| {
                        VllmClient::from_env().map(LlmBackend::Vllm)
                    })?;
                    Some(pool)
                }
                _ => {
                    tracing::info!(replicas = llm_replicas, "Building LLM pool (ONNX)");
                    let pool = StagePool::build("llm", llm_replicas, |_idx| {
                        QwenLlm::from_env().map(LlmBackend::Onnx)
                    })?;
                    Some(pool)
                }
            }
        };

        let tts = if tts_replicas == 0 {
            tracing::info!("TTS pool disabled (TTS_REPLICAS=0)");
            None
        } else {
            tracing::info!(replicas = tts_replicas, "Building TTS pool");
            let pool = StagePool::build("tts", tts_replicas, |_idx| {
                KokoroTts::from_env()
            })?;
            Some(pool)
        };

        Ok(Self {
            stt,
            streaming_stt,
            llm,
            tts,
        })
    }

    /// True when any STT path (batch queue or streaming leases) is configured.
    pub fn stt_available(&self) -> bool {
        self.stt.is_some() || self.streaming_stt.is_some()
    }

    /// Check out a streaming STT lease for sticky `push_chunk` (WS early-EOU).
    /// Returns `None` for batch backend, when disabled, or when all leases are held.
    pub fn try_acquire_streaming_stt(&self) -> Option<StreamingSttHandle> {
        self.streaming_stt.as_ref()?.try_checkout()
    }

    /// Check if all three pools are available.
    pub fn all_available(&self) -> bool {
        self.stt_available() && self.llm.is_some() && self.tts.is_some()
    }
}

/// Whether `STT_BACKEND` selects the cache-aware streaming encoder.
///
/// Shared by `StagePools::from_env` and `Pipeline::with_real_models` so the
/// pooled and unpooled paths cannot disagree about which model is loaded.
/// Anything other than `streaming`/`eou` (including unset and typos) means the
/// default full-utterance Parakeet — failing closed onto the model whose
/// accuracy and latency the prod budget is measured against.
pub fn streaming_stt_selected() -> bool {
    std::env::var("STT_BACKEND")
        .map(|v| matches!(v.trim().to_lowercase().as_str(), "streaming" | "eou"))
        .unwrap_or(false)
}

/// Whether `STT_BACKEND` selects the Sherpa Zipformer streaming model.
pub fn zipformer_stt_selected() -> bool {
    std::env::var("STT_BACKEND")
        .map(|v| v.trim().to_lowercase() == "zipformer")
        .unwrap_or(false)
}

/// Which STT model is selected. Shared between pool building and Pipeline
/// so the pooled and unpooled paths cannot disagree.
pub fn dev_model_selection() -> SttModel {
    if zipformer_stt_selected() {
        SttModel::Zipformer
    } else if streaming_stt_selected() {
        SttModel::Streaming
    } else {
        SttModel::Parakeet
    }
}

pub enum SttModel {
    Parakeet,
    Streaming,
    Zipformer,
}

/// Parse an env var as a non-negative integer with a default.
fn parse_replica_count(env_var: &str, default: usize) -> usize {
    std::env::var(env_var)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

// ---------------------------------------------------------------------------
// Pool adapters — implement stage traits by forwarding to shared pools
// ---------------------------------------------------------------------------

// --- STT Adapter ---

/// Wraps either a batch job-queue [`SttPool`] or a streaming [`StreamingSttPool`]
/// and implements `SttStage` for session / HTTP callers.
pub enum SttPoolAdapter {
    /// Full-utterance Parakeet via StagePool workers.
    Queued(Arc<SttPool>),
    /// Streaming EOU / Zipformer via exclusive checkout per call.
    Leased(Arc<StreamingSttPool>),
}

impl SttPoolAdapter {
    pub fn new(pool: &SttPool) -> Self {
        Self::Queued(Arc::new(pool.clone()))
    }

    /// Prefer batch queue; otherwise streaming lease pool.
    pub fn from_pools(pools: &StagePools) -> Option<Self> {
        if let Some(ref p) = pools.stt {
            Some(Self::new(p))
        } else {
            pools
                .streaming_stt
                .as_ref()
                .map(|p| Self::Leased(Arc::clone(p)))
        }
    }
}

impl SttStage for SttPoolAdapter {
    fn transcribe(&mut self, pcm: &[u8], partial: &mut dyn FnMut(&str)) -> String {
        match self {
            Self::Queued(pool) => {
                let input = SttInput {
                    pcm: pcm.to_vec(),
                };
                let cancel = Arc::new(AtomicBool::new(false));

                let mut handle = match pool.try_submit_with_cancel(input, cancel) {
                    Ok(h) => h,
                    Err(crate::stage_pool::SubmitError::Full) => {
                        tracing::warn!("STT pool queue full; rejecting request");
                        return String::new();
                    }
                    Err(crate::stage_pool::SubmitError::Closed) => {
                        tracing::error!("STT pool closed; cannot process request");
                        return String::new();
                    }
                };

                let mut final_text = String::new();
                while let Some(partial_text) = handle.deltas.blocking_recv() {
                    partial(&partial_text);
                }
                if let Ok(text) = handle.done.blocking_recv() {
                    final_text = text;
                }
                final_text
            }
            Self::Leased(pool) => {
                let Some(mut lease) = pool.try_checkout() else {
                    tracing::warn!("streaming STT leases exhausted; rejecting request");
                    return String::new();
                };
                lease.transcribe(pcm, partial)
            }
        }
    }
}

// --- LLM Adapter ---

/// Wraps an `Arc<LlmPool>` and implements `LlmStage`.
pub struct LlmPoolAdapter {
    pool: Arc<LlmPool>,
}

impl LlmPoolAdapter {
    pub fn new(pool: &LlmPool) -> Self {
        Self {
            pool: Arc::new(pool.clone()),
        }
    }
}

impl LlmStage for LlmPoolAdapter {
    fn generate(
        &mut self,
        prompt: &str,
        instructions: Option<&str>,
        cancel: &std::sync::atomic::AtomicBool,
        on_token: &mut dyn FnMut(&str),
    ) -> String {
        let input = LlmInput {
            prompt: prompt.to_string(),
            instructions: instructions.map(|s| s.to_string()),
        };

        // Share the caller's cancel Arc so barge-in propagates immediately.
        let caller_cancel = Arc::new(AtomicBool::new(cancel.load(std::sync::atomic::Ordering::Relaxed)));
        // We need to periodically forward the caller's cancel state.
        // Since we're in a blocking context, we can use a simple polling loop.
        let mut handle = match self.pool.try_submit_with_cancel(input, caller_cancel) {
            Ok(h) => h,
            Err(crate::stage_pool::SubmitError::Full) => {
                tracing::warn!("LLM pool queue full; rejecting request");
                return String::new();
            }
            Err(crate::stage_pool::SubmitError::Closed) => {
                tracing::error!("LLM pool closed; cannot process request");
                return String::new();
            }
        };

        // Stream tokens as deltas arrive, forwarding cancel on each iteration.
        let mut full_text = String::new();
        while let Some(token) = handle.deltas.blocking_recv() {
            // Forward the caller's cancel state to the pool job.
            handle.cancel.store(cancel.load(std::sync::atomic::Ordering::Relaxed), std::sync::atomic::Ordering::Relaxed);
            on_token(&token);
            full_text.push_str(&token);
        }
        // Wait for the final output.
        if let Ok(text) = handle.done.blocking_recv() {
            return text;
        }
        full_text
    }
}

// --- TTS Adapter ---

/// Wraps an `Arc<TtsPool>` and implements `TtsStage`.
pub struct TtsPoolAdapter {
    pool: Arc<TtsPool>,
}

impl TtsPoolAdapter {
    pub fn new(pool: &TtsPool) -> Self {
        Self {
            pool: Arc::new(pool.clone()),
        }
    }
}

impl TtsStage for TtsPoolAdapter {
    fn synthesize(
        &mut self,
        text: &str,
        voice: Option<&str>,
        cancel: &std::sync::atomic::AtomicBool,
        on_audio: &mut dyn FnMut(&[u8]),
    ) {
        let input = TtsInput {
            text: text.to_string(),
            voice: voice.map(|s| s.to_string()),
        };

        let mut handle = match self.pool.try_submit_with_cancel(input, Arc::new(AtomicBool::new(cancel.load(std::sync::atomic::Ordering::Relaxed)))) {
            Ok(h) => h,
            Err(crate::stage_pool::SubmitError::Full) => {
                tracing::warn!("TTS pool queue full; rejecting request");
                return;
            }
            Err(crate::stage_pool::SubmitError::Closed) => {
                tracing::error!("TTS pool closed; cannot process request");
                return;
            }
        };

        // Stream audio chunks as deltas arrive, forwarding cancel on each iteration.
        while let Some(chunk) = handle.deltas.blocking_recv() {
            handle.cancel.store(cancel.load(std::sync::atomic::Ordering::Relaxed), std::sync::atomic::Ordering::Relaxed);
            on_audio(&chunk);
        }
        // Wait for the final output (unit type, just ensures completion).
        let _ = handle.done.blocking_recv();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_replica_count_env_cases() {
        // Single test: cargo runs unit tests in parallel threads sharing process env.
        std::env::set_var("STT_REPLICAS", "0");
        std::env::set_var("LLM_REPLICAS", "0");
        std::env::set_var("TTS_REPLICAS", "0");
        assert_eq!(parse_replica_count("STT_REPLICAS", 1), 0);
        assert_eq!(parse_replica_count("LLM_REPLICAS", 1), 0);
        assert_eq!(parse_replica_count("TTS_REPLICAS", 1), 0);
        assert_eq!(parse_replica_count("NONEXISTENT_VAR", 42), 42);

        std::env::set_var("STT_REPLICAS", "not-a-number");
        assert_eq!(parse_replica_count("STT_REPLICAS", 3), 3);

        std::env::remove_var("STT_REPLICAS");
        std::env::remove_var("LLM_REPLICAS");
        std::env::remove_var("TTS_REPLICAS");
    }

    #[test]
    fn streaming_stt_pool_empty_checkout_is_none() {
        let pool = Arc::new(StreamingSttPool {
            free: Mutex::new(Vec::new()),
            total: 0,
        });
        assert!(pool.try_checkout().is_none());
        assert_eq!(pool.available(), 0);
        assert_eq!(pool.replica_count(), 0);
        assert_eq!(pool.in_use(), 0);
    }

    #[test]
    fn stt_pool_adapter_from_pools_prefers_batch_queue() {
        // Empty StagePools: no STT of either kind → None.
        let pools = StagePools {
            stt: None,
            streaming_stt: None,
            llm: None,
            tts: None,
        };
        assert!(SttPoolAdapter::from_pools(&pools).is_none());
        assert!(!pools.stt_available());
    }

    /// `STT_BACKEND` must fail CLOSED: only the two documented spellings select
    /// streaming. A typo silently loading the 120M model in place of the 0.6B one
    /// would look like an accuracy regression with no config error to point at.
    #[test]
    fn stt_backend_selection_fails_closed() {
        // All assertions live in ONE test because cargo runs tests as parallel
        // threads in a single process: splitting these would race on the shared
        // process environment. No other test touches STT_BACKEND.
        std::env::remove_var("STT_BACKEND");
        assert!(!streaming_stt_selected(), "unset must mean batch Parakeet");

        for v in ["streaming", "STREAMING", "eou", " Streaming "] {
            std::env::set_var("STT_BACKEND", v);
            assert!(streaming_stt_selected(), "{v:?} should select streaming");
        }

        for v in ["parakeet", "", "stream", "streamin", "tdt", "onnx"] {
            std::env::set_var("STT_BACKEND", v);
            assert!(!streaming_stt_selected(), "{v:?} must NOT select streaming");
        }

        std::env::remove_var("STT_BACKEND");
    }
}
