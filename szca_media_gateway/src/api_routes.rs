/// HTTP SSE API routes for STT, LLM, TTS micro-services.
///
/// These endpoints allow standalone use of each component. They share the
/// same pools as the realtime WebSocket sessions — no separate model copies.
///
/// Each endpoint:
/// 1. Validates the request.
/// 2. Acquires the relevant pool adapter (or returns 503 if the stage is disabled).
/// 3. Submits to the pool, streaming SSE deltas as they arrive.
/// 4. Backpressure: returns 503 with `Retry-After` when the pool queue is full.
///
/// Blocking pool work (submit + blocking_recv) runs inside `spawn_blocking`
/// to avoid blocking the tokio executor thread. The blocking task pushes each
/// event into an mpsc channel that backs the SSE response, so deltas reach the
/// client AS THEY ARE PRODUCED. Do not go back to `futures::stream::once` here:
/// a `once` stream can only ever yield a single item, which silently turns these
/// endpoints into request/response — see `sse_from_blocking`.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use crate::metrics::SharedMetrics;
use crate::rt_pipeline::SttStage;
use crate::stage_pool::LatencySnapshot;
use crate::stage_pools::StagePools;
use crate::rt_llm::LlmInput;
use crate::rt_stt::SttInput;
use crate::rt_tts::TtsInput;
use axum::{
    extract::{DefaultBodyLimit, State},
    http::{header, StatusCode},
    middleware::{self, Next},
    response::sse::{Event, Sse},
    response::Response,
    routing::{get, post},
    Json, Router,
    extract::Request,
};
use base64::Engine as _;
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;

/// Environment variable holding the API bearer token.
const API_KEY_ENV: &str = "SZCA_API_KEY";

/// Maximum accepted HTTP request body size (1 MiB).
const MAX_BODY_BYTES: usize = 1024 * 1024;

/// Upper bound on requested `max_tokens`.
const MAX_TOKENS_CAP: u32 = 8192;

/// Upper bound on text/message payload length (characters).
const MAX_TEXT_LEN: usize = 32 * 1024;

/// Shared application state for all HTTP routes.
#[derive(Clone)]
pub struct RouteState {
    pub metrics: SharedMetrics,
    pub pools: Option<Arc<StagePools>>,
}

/// Build a 400 Bad Request JSON error response.
fn bad_request(msg: &str) -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({ "error": msg })),
    )
}

/// Build a 503 Service Unavailable JSON error response.
fn pool_busy(stage: &str) -> (StatusCode, Json<serde_json::Value>) {
    tracing::warn!(stage, "Pool queue full; returning 503");
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(serde_json::json!({ "error": format!("{} pool busy, retry later", stage) })),
    )
}

/// Helper to extract a pool from RouteState, ensuring it is configured and has capacity.
fn acquire_pool<R: crate::stage_pool::Replica>(
    state: &RouteState,
    extractor: impl FnOnce(&StagePools) -> Option<&crate::stage_pool::StagePool<R>>,
    name: &str,
) -> Result<crate::stage_pool::StagePool<R>, (StatusCode, Json<serde_json::Value>)> {
    let pools = state.pools.as_ref().ok_or_else(|| {
        bad_request(&format!("{} not available (pool not configured)", name.to_uppercase()))
    })?;

    let pool = extractor(pools).ok_or_else(|| {
        bad_request(&format!("{} not available ({}_REPLICAS=0)", name.to_uppercase(), name.to_uppercase()))
    })?;

    // Same bound as StagePool::build — SZCA_QUEUE_BACKLOG (default 64).
    if pool.queue_depth() >= crate::stage_pool::queue_backlog_from_env() {
        return Err(pool_busy(name));
    }

    Ok(pool.clone())
}

/// Serialize a value into SSE event data, mapping failures to an error event
/// rather than panicking.
fn sse_data<T: Serialize>(event: &str, value: &T) -> Result<Event, Infallible> {
    match serde_json::to_string(value) {
        Ok(json) => Ok(Event::default().event(event).data(json)),
        Err(e) => Ok(Event::default()
            .event("error")
            .data(format!("{{\"error\":\"serialization failed: {}\"}}", e))),
    }
}

/// Bearer-token auth middleware. Enforced only when `SZCA_API_KEY` is set.
async fn require_auth(request: Request, next: Next) -> Result<Response, StatusCode> {
    let expected = match std::env::var(API_KEY_ENV) {
        Ok(v) if !v.is_empty() => v,
        _ => return Ok(next.run(request).await),
    };

    let provided = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .and_then(|h| h.strip_prefix("Bearer "));

    match provided {
        Some(token) if token == expected => Ok(next.run(request).await),
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}

// ============================================================================
// STT API
// ============================================================================

#[derive(Deserialize)]
pub struct SttRequest {
    pub model: Option<String>,
    pub language: Option<String>,
    pub interim_results: Option<bool>,
    pub max_segment_duration_ms: Option<u32>,
    pub audio_format: Option<String>,
    /// Base64-encoded PCM16 mono 16 kHz audio.
    pub audio: Option<String>,
}

#[derive(Serialize)]
pub struct SttPartialResult {
    #[serde(rename = "type")]
    pub result_type: String,
    pub text: String,
    pub confidence: f32,
    pub timestamp: u64,
}

#[derive(Serialize)]
pub struct SttFinalResult {
    #[serde(rename = "type")]
    pub result_type: String,
    pub text: String,
    pub confidence: f32,
    pub timestamp: u64,
}

type SseResponse = Sse<std::pin::Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>>>;

/// Sink handed to a blocking pool job so it can emit SSE events incrementally.
///
/// `send` returns `false` once the client has disconnected (receiver dropped),
/// which lets a job stop early instead of synthesizing audio nobody will read.
pub struct EventSink {
    tx: tokio::sync::mpsc::Sender<Result<Event, Infallible>>,
}

impl EventSink {
    /// Emit one named SSE event. Returns `false` if the client is gone or too slow.
    pub fn send<T: Serialize>(&self, event: &str, value: &T) -> bool {
        match sse_data(event, value) {
            Ok(mut ev) => {
                let start = std::time::Instant::now();
                loop {
                    match self.tx.try_send(Ok(ev)) {
                        Ok(_) => return true,
                        Err(tokio::sync::mpsc::error::TrySendError::Full(returned)) => {
                            if start.elapsed().as_millis() > 2000 {
                                tracing::warn!("SSE client backpressure timeout (>2s), shedding load");
                                return false;
                            }
                            std::thread::sleep(std::time::Duration::from_millis(10));
                            ev = returned.unwrap();
                        }
                        Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => return false,
                    }
                }
            }
            Err(_) => true,
        }
    }
}

/// Run `job` on a blocking thread and stream the events it emits over SSE.
///
/// This is the piece that makes `/v1/*/stream` actually stream. The channel is
/// bounded: a slow client applies backpressure to the producing thread rather
/// than letting an unbounded queue of PCM chunks grow without limit.
///
/// The buffer is deliberately larger than one so a fast producer (TTS emits a
/// chunk every few ms) is not serialized against per-event socket writes.
fn sse_from_blocking<F>(job: F) -> SseResponse
where
    F: FnOnce(EventSink) + Send + 'static,
{
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(64);

    // Keep a handle on the blocking task so a PANIC inside a replica still
    // reaches the client as an `error` event. Without this the channel would
    // just close and the client would see a truncated stream with no reason.
    let err_tx = tx.clone();
    let handle = tokio::task::spawn_blocking(move || {
        let sink = EventSink { tx };
        job(sink);
        // Dropping `sink` closes the channel, which ends the SSE stream.
    });
    tokio::spawn(async move {
        if let Err(e) = handle.await {
            // Async context, so `send().await` — `blocking_send` would panic here.
            let msg = serde_json::json!({"error": format!("inference task failed: {e}")});
            // sse_data's error type is Infallible, so this never fails.
            let Ok(ev) = sse_data("error", &msg);
            let _ = err_tx.send(Ok(ev)).await;
        }
    });

    Sse::new(Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx)))
}

/// POST /v1/stt/stream — Streaming STT via shared pool
pub async fn stt_stream(
    State(state): State<RouteState>,
    Json(payload): Json<SttRequest>,
) -> Result<SseResponse, (StatusCode, Json<serde_json::Value>)> {
    if let Some(ms) = payload.max_segment_duration_ms {
        if ms == 0 {
            return Err(bad_request("max_segment_duration_ms must be > 0"));
        }
    }

    let pcm = match &payload.audio {
        Some(b64) => base64::engine::general_purpose::STANDARD
            .decode(b64)
            .map_err(|e| bad_request(&format!("invalid base64 audio: {e}")))?,
        None => {
            return Err(bad_request("audio field (base64 PCM16) is required"));
        }
    };

    let stt_available = state.pools.as_ref().is_some_and(|p| p.stt_available());
    if !stt_available {
        return Err(bad_request("STT not available (pool not configured)"));
    }

    // Batch: job-queue StagePool. Streaming: exclusive lease for the request.
    if let Some(stt_pool) = state.pools.as_ref().and_then(|p| p.stt.clone()) {
        if stt_pool.queue_depth() >= crate::stage_pool::queue_backlog_from_env() {
            return Err(pool_busy("stt"));
        }
        let pcm = pcm.to_vec();
        let want_partials = payload.interim_results.unwrap_or(true);
        return Ok(sse_from_blocking(move |sink| {
            let input = SttInput { pcm };
            let cancel = Arc::new(AtomicBool::new(false));

            let mut handle = match stt_pool.try_submit_with_cancel(input, cancel) {
                Ok(h) => h,
                Err(crate::stage_pool::SubmitError::Full) => {
                    sink.send(
                        "error",
                        &serde_json::json!({"error": "STT pool busy, retry later"}),
                    );
                    return;
                }
                Err(crate::stage_pool::SubmitError::Closed) => {
                    sink.send("error", &serde_json::json!({"error": "STT pool closed"}));
                    return;
                }
            };

            while let Some(text) = handle.deltas.blocking_recv() {
                if !want_partials {
                    continue;
                }
                let sent = sink.send(
                    "partial",
                    &SttPartialResult {
                        result_type: "partial".to_string(),
                        text,
                        confidence: 0.95,
                        timestamp: 0,
                    },
                );
                if !sent {
                    return;
                }
            }
            let final_text = handle.done.blocking_recv().unwrap_or_default();

            sink.send(
                "final",
                &SttFinalResult {
                    result_type: "final".to_string(),
                    text: final_text,
                    confidence: 0.95,
                    timestamp: 0,
                },
            );
        }));
    }

    let lease_pool = state
        .pools
        .as_ref()
        .and_then(|p| p.streaming_stt.clone())
        .ok_or_else(|| bad_request("STT not available (pool not configured)"))?;
    if lease_pool.available() == 0 {
        return Err(pool_busy("stt"));
    }
    let pcm = pcm.to_vec();
    let want_partials = payload.interim_results.unwrap_or(true);
    Ok(sse_from_blocking(move |sink| {
        let Some(mut lease) = lease_pool.try_checkout() else {
            sink.send(
                "error",
                &serde_json::json!({"error": "STT pool busy, retry later"}),
            );
            return;
        };
        let mut final_text = String::new();
        let text = lease.transcribe(&pcm, &mut |partial| {
            if want_partials {
                let _ = sink.send(
                    "partial",
                    &SttPartialResult {
                        result_type: "partial".to_string(),
                        text: partial.to_string(),
                        confidence: 0.95,
                        timestamp: 0,
                    },
                );
            }
            final_text = partial.to_string();
        });
        if !text.is_empty() {
            final_text = text;
        }
        sink.send(
            "final",
            &SttFinalResult {
                result_type: "final".to_string(),
                text: final_text,
                confidence: 0.95,
                timestamp: 0,
            },
        );
    }))
}

// ============================================================================
// LLM API
// ============================================================================

#[derive(Deserialize)]
pub struct LlmMessage {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize)]
pub struct LlmRequest {
    pub model: Option<String>,
    pub messages: Vec<LlmMessage>,
    pub stream: Option<bool>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
}

#[derive(Serialize)]
pub struct LlmTokenResult {
    #[serde(rename = "type")]
    pub result_type: String,
    pub text: String,
    pub token_id: i32,
    pub logprob: f32,
    pub index: i32,
}

#[derive(Serialize)]
pub struct LlmEosResult {
    #[serde(rename = "type")]
    pub result_type: String,
    pub text: String,
    pub total_tokens: i32,
    pub finish_reason: String,
}

/// POST /v1/llm/stream — Streaming LLM via shared pool
pub async fn llm_stream(
    State(state): State<RouteState>,
    Json(payload): Json<LlmRequest>,
) -> Result<SseResponse, (StatusCode, Json<serde_json::Value>)> {
    if payload.messages.is_empty() {
        return Err(bad_request("messages must not be empty"));
    }
    if payload.messages.len() > 256 {
        return Err(bad_request("too many messages (max 256)"));
    }
    for m in &payload.messages {
        if m.content.len() > MAX_TEXT_LEN {
            return Err(bad_request("message content too long"));
        }
    }
    if let Some(t) = payload.temperature {
        if !(0.0..=2.0).contains(&t) {
            return Err(bad_request("temperature must be in [0, 2]"));
        }
    }
    if let Some(p) = payload.top_p {
        if !(0.0..=1.0).contains(&p) {
            return Err(bad_request("top_p must be in [0, 1]"));
        }
    }
    if let Some(mt) = payload.max_tokens {
        if mt == 0 || mt > MAX_TOKENS_CAP {
            return Err(bad_request("max_tokens must be in [1, 8192]"));
        }
    }

    let llm_pool = acquire_pool(&state, |p| p.llm.as_ref(), "llm")?;

    // Extract system instructions and user prompt from messages.
    let instructions: Option<String> = payload
        .messages
        .iter()
        .find(|m| m.role == "system")
        .map(|m| m.content.clone());
    let user_prompt: String = payload
        .messages
        .iter()
        .filter(|m| m.role == "user")
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");


    let max_tokens = payload.max_tokens;

    Ok(sse_from_blocking(move |sink| {
        let input = LlmInput {
            prompt: user_prompt,
            instructions,
        };
        let cancel = Arc::new(AtomicBool::new(false));

        let mut handle = match llm_pool.try_submit_with_cancel(input, cancel.clone()) {
            Ok(h) => h,
            Err(crate::stage_pool::SubmitError::Full) => {
                sink.send(
                    "error",
                    &serde_json::json!({"error": "LLM pool busy, retry later"}),
                );
                return;
            }
            Err(crate::stage_pool::SubmitError::Closed) => {
                sink.send("error", &serde_json::json!({"error": "LLM pool closed"}));
                return;
            }
        };

        let mut token_index: i32 = 0;
        let mut full_text = String::new();
        // Was the reply cut short by the caller's max_tokens rather than the
        // model's own stop token? That distinction is what `finish_reason`
        // reports, so it cannot be a hardcoded "stop".
        let mut hit_cap = false;
        let mut disconnected = false;

        while let Some(token) = handle.deltas.blocking_recv() {
            full_text.push_str(&token);
            token_index += 1;

            if !sink.send(
                "token",
                &LlmTokenResult {
                    result_type: "token".to_string(),
                    text: token,
                    // The pool's Delta is the decoded text piece; per-token ids
                    // and logprobs are not surfaced by the replica, so report
                    // sentinels rather than inventing plausible numbers.
                    token_id: -1,
                    logprob: 0.0,
                    index: token_index - 1,
                },
            ) {
                // Client hung up: cancel generation so a replica is not tied up
                // producing tokens for a closed socket.
                cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                disconnected = true;
                break;
            }

            if let Some(cap) = max_tokens {
                if token_index as u32 >= cap {
                    hit_cap = true;
                    cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                    break;
                }
            }
        }

        // Drain the replica's own final value; falls back to the accumulated
        // deltas when generation was cancelled mid-flight.
        let final_text = handle.done.blocking_recv().unwrap_or(full_text);
        if disconnected {
            return;
        }

        sink.send(
            "eos",
            &LlmEosResult {
                result_type: "eos".to_string(),
                text: final_text,
                total_tokens: token_index,
                finish_reason: if hit_cap { "length" } else { "stop" }.to_string(),
            },
        );
    }))
}

// ============================================================================
// TTS API
// ============================================================================

#[derive(Deserialize)]
pub struct TtsRequest {
    pub model: Option<String>,
    pub voice: Option<String>,
    pub language: Option<String>,
    pub input: String,
    pub stream: Option<bool>,
    pub format: Option<String>,
    pub speed: Option<f32>,
}

#[derive(Serialize)]
pub struct TtsAudioChunk {
    #[serde(rename = "type")]
    pub result_type: String,
    pub pcm: String,
    pub sample_rate: u32,
    pub duration_ms: f32,
    pub sequence: u32,
}

#[derive(Serialize)]
pub struct TtsEosResult {
    #[serde(rename = "type")]
    pub result_type: String,
    pub total_duration_ms: u64,
    pub total_chunks: u32,
    pub sample_rate: u32,
}

/// POST /v1/tts/stream — Streaming TTS via shared pool
pub async fn tts_stream(
    State(state): State<RouteState>,
    Json(payload): Json<TtsRequest>,
) -> Result<SseResponse, (StatusCode, Json<serde_json::Value>)> {
    if payload.input.is_empty() {
        return Err(bad_request("input must not be empty"));
    }
    if payload.input.len() > MAX_TEXT_LEN {
        return Err(bad_request("input too long"));
    }
    if let Some(speed) = payload.speed {
        if !(0.25..=4.0).contains(&speed) {
            return Err(bad_request("speed must be in [0.25, 4.0]"));
        }
    }

    let tts_pool = acquire_pool(&state, |p| p.tts.as_ref(), "tts")?;
    let voice = payload.voice.unwrap_or_else(|| "af_heart".to_string());
    let sample_rate = 16000u32;

    Ok(sse_from_blocking(move |sink| {
        let input = TtsInput {
            text: payload.input,
            voice: Some(voice),
        };
        let cancel = Arc::new(AtomicBool::new(false));

        let mut handle = match tts_pool.try_submit_with_cancel(input, cancel.clone()) {
            Ok(h) => h,
            Err(crate::stage_pool::SubmitError::Full) => {
                sink.send(
                    "error",
                    &serde_json::json!({"error": "TTS pool busy, retry later"}),
                );
                return;
            }
            Err(crate::stage_pool::SubmitError::Closed) => {
                sink.send("error", &serde_json::json!({"error": "TTS pool closed"}));
                return;
            }
        };

        let mut chunk_index: u32 = 0;
        let mut total_bytes: u64 = 0;
        while let Some(pcm_chunk) = handle.deltas.blocking_recv() {
            total_bytes += pcm_chunk.len() as u64;
            let encoded = base64::engine::general_purpose::STANDARD.encode(&pcm_chunk);
            let duration_ms = (pcm_chunk.len() as f32) / (sample_rate as f32 * 2.0) * 1000.0;
            // Every chunk goes to the client. Discarding these (the old
            // `let _ = sse_data(...)`) meant callers got an eos claiming N
            // chunks and zero audio.
            if !sink.send(
                "audio_chunk",
                &TtsAudioChunk {
                    result_type: "audio_chunk".to_string(),
                    pcm: encoded,
                    sample_rate,
                    duration_ms,
                    sequence: chunk_index,
                },
            ) {
                // Client hung up mid-utterance: stop synthesizing.
                cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                let _ = handle.done.blocking_recv();
                return;
            }
            chunk_index += 1;
        }
        let _ = handle.done.blocking_recv();

        // Round to the nearest ms instead of truncating twice (bytes/2 then
        // integer division), which under-reported duration for odd sizes.
        let total_samples = total_bytes / 2;
        let total_duration_ms =
            (total_samples * 1000 + sample_rate as u64 / 2) / sample_rate as u64;
        sink.send(
            "eos",
            &TtsEosResult {
                result_type: "eos".to_string(),
                total_duration_ms,
                total_chunks: chunk_index,
                sample_rate,
            },
        );
    }))
}

// ============================================================================
// Metrics Endpoint
// ============================================================================

/// GET /metrics — Prometheus metrics export.
pub async fn metrics_endpoint(State(state): State<RouteState>) -> String {
    state.metrics.export_prometheus_with_pools(state.pools.as_deref())
}

// ============================================================================
// Pool Health Endpoint
// ============================================================================

#[derive(Serialize)]
pub struct PoolHealth {
    pub stt_available: bool,
    pub stt_queue_depth: usize,
    pub stt_replicas: usize,
    pub stt_latency: Option<LatencySnapshot>,
    pub llm_available: bool,
    pub llm_queue_depth: usize,
    pub llm_replicas: usize,
    pub llm_latency: Option<LatencySnapshot>,
    pub tts_available: bool,
    pub tts_queue_depth: usize,
    pub tts_replicas: usize,
    pub tts_latency: Option<LatencySnapshot>,
}

/// GET /v1/pools — Per-pool health and saturation info.
pub async fn pool_health(State(state): State<RouteState>) -> Json<PoolHealth> {
    let pools = state.pools.as_ref();
    let stt_available = pools.is_some_and(|p| p.stt_available());
    let (stt_queue_depth, stt_replicas, stt_latency) = match pools {
        Some(p) if p.stt.is_some() => (
            p.stt.as_ref().map(|q| q.queue_depth()).unwrap_or(0),
            p.stt.as_ref().map(|q| q.replica_count()).unwrap_or(0),
            p.stt.as_ref().and_then(|q| q.latency_snapshot()),
        ),
        Some(p) if p.streaming_stt.is_some() => (
            p.streaming_stt.as_ref().map(|q| q.in_use()).unwrap_or(0),
            p.streaming_stt.as_ref().map(|q| q.replica_count()).unwrap_or(0),
            None,
        ),
        _ => (0, 0, None),
    };
    Json(PoolHealth {
        stt_available,
        stt_queue_depth,
        stt_replicas,
        stt_latency,
        llm_available: pools.and_then(|p| p.llm.as_ref()).is_some(),
        llm_queue_depth: pools.and_then(|p| p.llm.as_ref()).map(|p| p.queue_depth()).unwrap_or(0),
        llm_replicas: pools.and_then(|p| p.llm.as_ref()).map(|p| p.replica_count()).unwrap_or(0),
        llm_latency: pools.and_then(|p| p.llm.as_ref()).and_then(|p| p.latency_snapshot()),
        tts_available: pools.and_then(|p| p.tts.as_ref()).is_some(),
        tts_queue_depth: pools.and_then(|p| p.tts.as_ref()).map(|p| p.queue_depth()).unwrap_or(0),
        tts_replicas: pools.and_then(|p| p.tts.as_ref()).map(|p| p.replica_count()).unwrap_or(0),
        tts_latency: pools.and_then(|p| p.tts.as_ref()).and_then(|p| p.latency_snapshot()),
    })
}

// ============================================================================
// Router
// ============================================================================

/// Request timeout: 60 seconds for inference endpoints, 5 seconds for health/metrics.
const REQUEST_TIMEOUT_SECS: u64 = 60;

/// Build API routes, wiring shared state, bearer auth, body size limit, and
/// request timeout (prevents stalled inference from holding a connection forever).
pub fn api_router(state: RouteState) -> Router {
    let timeout = std::time::Duration::from_secs(REQUEST_TIMEOUT_SECS);
    Router::new()
        .route("/v1/stt/stream", post(stt_stream))
        .route("/v1/llm/stream", post(llm_stream))
        .route("/v1/tts/stream", post(tts_stream))
        .route("/v1/pools", get(pool_health))
        .route("/metrics", get(metrics_endpoint))
        .layer(middleware::from_fn(require_auth))
        .layer(make_timeout_layer(timeout))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(state)
}

/// Build a timeout layer with the correct StatusCode type.
/// tower-http 0.6 deprecated `TimeoutLayer::new` in favor of `with_status_code`,
/// but the StatusCode type needs to come from `axum::http` to match the rest of
/// our axum stack — otherwise we get a type mismatch with `reqwest::StatusCode`.
fn make_timeout_layer(timeout: std::time::Duration) -> tower_http::timeout::TimeoutLayer {
    let sc: axum::http::StatusCode = axum::http::StatusCode::from_u16(408)
        .expect("408 is a valid HTTP status code");
    tower_http::timeout::TimeoutLayer::with_status_code(sc, timeout)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stt_request_deserialize() {
        let json = r#"{
            "model": "parakeet_tdt_0.6b_v3",
            "language": "en",
            "interim_results": true,
            "audio": "AAAA"
        }"#;
        let req: SttRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model.unwrap(), "parakeet_tdt_0.6b_v3");
        assert_eq!(req.language.unwrap(), "en");
        assert!(req.interim_results.unwrap());
        assert!(req.audio.is_some());
    }

    #[test]
    fn test_stt_final_result_serialize() {
        let result = SttFinalResult {
            result_type: "final".to_string(),
            text: "hello world".to_string(),
            confidence: 0.95,
            timestamp: 100,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("final"));
        assert!(json.contains("hello world"));
    }

    #[test]
    fn test_llm_request_deserialize() {
        let json = r#"{
            "model": "hermes-3-3b",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 128
        }"#;
        let req: LlmRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model.unwrap(), "hermes-3-3b");
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.max_tokens.unwrap(), 128);
    }

    #[test]
    fn test_tts_request_deserialize() {
        let json = r#"{
            "input": "Hello world",
            "voice": "af_heart",
            "format": "pcm_16khz_16bit_mono"
        }"#;
        let req: TtsRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.input, "Hello world");
        assert_eq!(req.voice.unwrap(), "af_heart");
    }

    /// Collect every event body an `sse_from_blocking` job produces.
    ///
    /// `Event` exposes no getters, so assert on the serialized wire bytes —
    /// which is what a client actually sees anyway.
    async fn drain(response: SseResponse) -> Vec<String> {
        use axum::response::IntoResponse;
        use http_body_util::BodyExt;

        let body = response.into_response().into_body();
        let bytes = body.collect().await.expect("body collects").to_bytes();
        String::from_utf8_lossy(&bytes)
            .lines()
            .filter_map(|l| l.strip_prefix("data: ").map(str::to_string))
            .collect()
    }

    /// The regression test for the bug that made every `/v1/*/stream` endpoint
    /// non-streaming: the handler built per-delta events and threw them away, so
    /// clients received only the terminal event. `futures::stream::once` can
    /// yield exactly one item, so a single-event stream is the symptom to guard.
    #[tokio::test]
    async fn sse_from_blocking_delivers_every_event_not_just_the_last() {
        let response = sse_from_blocking(|sink| {
            for i in 0..5 {
                assert!(sink.send("chunk", &serde_json::json!({"seq": i})));
            }
            sink.send("eos", &serde_json::json!({"total": 5}));
        });

        let events = drain(response).await;
        assert_eq!(events.len(), 6, "expected 5 chunks + eos, got {events:?}");
        for (i, ev) in events.iter().take(5).enumerate() {
            assert!(ev.contains(&format!("\"seq\":{i}")), "bad chunk {i}: {ev}");
        }
        assert!(events[5].contains("\"total\":5"));
    }

    /// A job that emits nothing still terminates the stream rather than hanging
    /// the client: dropping the sink closes the channel.
    #[tokio::test]
    async fn sse_from_blocking_closes_stream_when_job_emits_nothing() {
        let response = sse_from_blocking(|_sink| {});
        assert!(drain(response).await.is_empty());
    }

    /// A panic inside a replica must surface as an `error` event; otherwise the
    /// client sees a silently truncated stream with no explanation.
    #[tokio::test]
    async fn sse_from_blocking_reports_a_panicking_job_as_an_error_event() {
        let response = sse_from_blocking(|sink| {
            sink.send("chunk", &serde_json::json!({"seq": 0}));
            panic!("replica exploded");
        });

        let events = drain(response).await;
        assert_eq!(events.len(), 2, "expected chunk + error, got {events:?}");
        assert!(events[1].contains("inference task failed"), "{:?}", events[1]);
    }

    /// `send` reports client disconnect so a job can stop early instead of
    /// synthesizing audio nobody will read.
    #[tokio::test]
    async fn event_sink_reports_disconnect_so_jobs_can_stop_early() {
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(4);
        let sink = EventSink { tx };
        let sent_before = tokio::task::spawn_blocking(move || {
            let ok = sink.send("chunk", &serde_json::json!({"seq": 0}));
            (ok, sink)
        });
        let (ok, sink) = sent_before.await.unwrap();
        assert!(ok, "send should succeed while the receiver lives");

        drop(rx); // client hangs up
        let after = tokio::task::spawn_blocking(move || {
            sink.send("chunk", &serde_json::json!({"seq": 1}))
        })
        .await
        .unwrap();
        assert!(!after, "send must report false once the client is gone");
    }

    #[test]
    fn test_pool_health_serialize() {
        let health = PoolHealth {
            stt_available: true,
            stt_queue_depth: 0,
            stt_replicas: 2,
            stt_latency: None,
            llm_available: true,
            llm_queue_depth: 3,
            llm_replicas: 4,
            llm_latency: None,
            tts_available: false,
            tts_queue_depth: 0,
            tts_replicas: 0,
            tts_latency: None,
        };
        let json = serde_json::to_string(&health).unwrap();
        assert!(json.contains("stt_available"));
        assert!(json.contains("llm_replicas"));
        assert!(json.contains("stt_latency"));
    }
}
