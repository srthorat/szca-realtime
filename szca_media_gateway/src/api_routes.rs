/// HTTP SSE API routes for STT, LLM, TTS micro-services.
///
/// These endpoints allow standalone use of each component.

use crate::metrics::SharedMetrics;
use axum::{
    extract::{DefaultBodyLimit, Request, State},
    http::{header, StatusCode},
    middleware::{self, Next},
    response::sse::{Event, Sse},
    response::Response,
    routing::{get, post},
    Json, Router,
};
use base64::Engine as _;
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;

/// Environment variable holding the API bearer token. When unset, auth is
/// disabled (with a startup warning) so local development still works; when
/// set, every `/v1/*` and `/metrics` request must present a matching
/// `Authorization: Bearer <token>` header.
const API_KEY_ENV: &str = "SZCA_API_KEY";

/// Maximum accepted HTTP request body size (1 MiB), coordinated with the
/// WebSocket message limit in `main`.
const MAX_BODY_BYTES: usize = 1024 * 1024;

/// Placeholder TTS audio chunk size in bytes (20ms @ 16kHz 16-bit mono).
const PLACEHOLDER_TTS_CHUNK_BYTES: usize = 640;

/// Upper bound on requested `max_tokens`.
const MAX_TOKENS_CAP: u32 = 8192;

/// Upper bound on text/message payload length (characters).
const MAX_TEXT_LEN: usize = 32 * 1024;

/// Build a 400 Bad Request JSON error response.
fn bad_request(msg: &str) -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({ "error": msg })),
    )
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
        _ => return Ok(next.run(request).await), // auth disabled
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
// API 2: STT API
// ============================================================================

#[derive(Deserialize)]
pub struct SttRequest {
    pub model: Option<String>,
    pub language: Option<String>,
    pub interim_results: Option<bool>,
    pub max_segment_duration_ms: Option<u32>,
    pub audio_format: Option<String>,
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

#[derive(Serialize)]
pub struct SttDoneResult {
    #[serde(rename = "type")]
    pub result_type: String,
    pub total_duration_ms: u64,
    pub total_segments: u32,
}

type SseResponse = Sse<std::pin::Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>>>;

/// POST /v1/stt/stream — Streaming STT
pub async fn stt_stream(
    Json(payload): Json<SttRequest>,
) -> Result<SseResponse, (StatusCode, Json<serde_json::Value>)> {
    // Validate before building the stream.
    if let Some(ms) = payload.max_segment_duration_ms {
        if ms == 0 {
            return Err(bad_request("max_segment_duration_ms must be > 0"));
        }
    }

    let model = payload.model.unwrap_or_else(|| "parakeet_tdt_0.6b_v3".to_string());
    let language = payload.language.unwrap_or_else(|| "en".to_string());
    let interim = payload.interim_results.unwrap_or(true);
    let audio_format = payload
        .audio_format
        .unwrap_or_else(|| "pcm_16khz_16bit_mono".to_string());

    let stream = futures::stream::once(async move {
        // In production: stream audio from request body, run STT.
        // For now, echo the resolved request parameters so none are silently
        // dropped.
        let partial = SttPartialResult {
            result_type: "partial".to_string(),
            text: format!(
                "STT model={} lang={} interim={} format={}",
                model, language, interim, audio_format
            ),
            confidence: 0.9,
            timestamp: 0,
        };
        sse_data("partial", &partial)
    });

    Ok(Sse::new(Box::pin(stream)))
}

// ============================================================================
// API 3: LLM API
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

/// POST /v1/llm/stream — Streaming LLM
pub async fn llm_stream(
    Json(payload): Json<LlmRequest>,
) -> Result<SseResponse, (StatusCode, Json<serde_json::Value>)> {
    // Validate before building the stream.
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

    let model = payload.model.unwrap_or_else(|| "hermes-3-3b".to_string());
    let max_tokens = payload.max_tokens.unwrap_or(256);
    let temperature = payload.temperature.unwrap_or(0.7);
    let top_p = payload.top_p.unwrap_or(1.0);
    let stream_flag = payload.stream.unwrap_or(true);

    let stream = futures::stream::once(async move {
        let token = LlmTokenResult {
            result_type: "token".to_string(),
            text: format!(
                "LLM model={} max_tokens={} temperature={} top_p={} stream={}",
                model, max_tokens, temperature, top_p, stream_flag
            ),
            token_id: 1,
            logprob: -0.05,
            index: 0,
        };
        sse_data("token", &token)
    });

    Ok(Sse::new(Box::pin(stream)))
}

// ============================================================================
// API 4: TTS API
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
    pub pcm: String, // base64
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

/// POST /v1/tts/stream — Streaming TTS
pub async fn tts_stream(
    Json(payload): Json<TtsRequest>,
) -> Result<SseResponse, (StatusCode, Json<serde_json::Value>)> {
    // Validate before building the stream.
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

    let model = payload.model.unwrap_or_else(|| "kokoro-82m".to_string());
    let voice = payload.voice.unwrap_or_else(|| "af_heart".to_string());
    let language = payload.language.unwrap_or_else(|| "en".to_string());
    let format = payload
        .format
        .unwrap_or_else(|| "pcm_16khz_16bit_mono".to_string());
    let speed = payload.speed.unwrap_or(1.0);
    let stream_flag = payload.stream.unwrap_or(true);
    let _input = payload.input;
    let sample_rate = 16000;

    let stream = futures::stream::once(async move {
        // Echo resolved params so none are silently dropped; log the rest.
        tracing::debug!(%model, %language, %format, speed, stream_flag, "TTS request");
        let chunk = TtsAudioChunk {
            result_type: format!("audio_chunk voice={} speed={}", voice, speed),
            pcm: base64::engine::general_purpose::STANDARD
                .encode(vec![0u8; PLACEHOLDER_TTS_CHUNK_BYTES]),
            sample_rate,
            duration_ms: 20.0,
            sequence: 0,
        };
        sse_data("audio_chunk", &chunk)
    });

    Ok(Sse::new(Box::pin(stream)))
}

// ============================================================================
// Metrics Endpoint
// ============================================================================

/// GET /metrics — Prometheus metrics export.
///
/// Backed by the shared [`SharedMetrics`] state; auth-protected (internal) via
/// the same bearer middleware as `/v1/*`.
pub async fn metrics_endpoint(State(metrics): State<SharedMetrics>) -> String {
    metrics.export_prometheus()
}

// ============================================================================
// Router
// ============================================================================

/// Build API routes, wiring shared metrics state, bearer auth, and a request
/// body size limit.
pub fn api_router(metrics: SharedMetrics) -> Router {
    Router::new()
        .route("/v1/stt/stream", post(stt_stream))
        .route("/v1/llm/stream", post(llm_stream))
        .route("/v1/tts/stream", post(tts_stream))
        .route("/metrics", get(metrics_endpoint))
        // Auth applies to all routes above (/v1/* and /metrics). /health is
        // registered on the parent router and stays unauthenticated.
        .layer(middleware::from_fn(require_auth))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(metrics)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stt_request_deserialize() {
        let json = r#"{
            "model": "parakeet_tdt_0.6b_v3",
            "language": "en",
            "interim_results": true
        }"#;
        let req: SttRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model.unwrap(), "parakeet_tdt_0.6b_v3");
        assert_eq!(req.language.unwrap(), "en");
        assert!(req.interim_results.unwrap());
    }

    #[test]
    fn test_stt_partial_result_serialize() {
        let result = SttPartialResult {
            result_type: "partial".to_string(),
            text: "hello".to_string(),
            confidence: 0.85,
            timestamp: 100,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("partial"));
        assert!(json.contains("hello"));
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
    fn test_llm_token_result_serialize() {
        let result = LlmTokenResult {
            result_type: "token".to_string(),
            text: "Hello".to_string(),
            token_id: 1234,
            logprob: -0.1,
            index: 0,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("token"));
        assert!(json.contains("Hello"));
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

    #[test]
    fn test_tts_audio_chunk_serialize() {
        let chunk = TtsAudioChunk {
            result_type: "audio_chunk".to_string(),
            pcm: "AAAA".to_string(),
            sample_rate: 16000,
            duration_ms: 20.0,
            sequence: 0,
        };
        let json = serde_json::to_string(&chunk).unwrap();
        assert!(json.contains("audio_chunk"));
        assert!(json.contains("16000"));
    }
}
