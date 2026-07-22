/// SZCA Media Gateway — Entry Point
///
/// Real-time voice streaming gateway with:
/// - Binary WebSocket protocol (16kHz PCM)
/// - DeepFilterNet3 noise suppression
/// - Silero VAD speech detection
/// - Atomic barge-in interrupts
/// - Zero-copy IPC to inference engine
/// - HTTP SSE APIs (STT, LLM, TTS)
/// - Health check endpoint
/// - Graceful shutdown

use szca_media_gateway::{api_routes, dsp, gateway, metrics, protocol, session, vad};

use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    http::StatusCode,
    response::IntoResponse,
    routing::get,
    Json, Router,
};
use futures_util::{SinkExt, StreamExt};
use gateway::GatewayConfig;
use metrics::SharedMetrics;
use protocol::AudioConfig;
use session::{Session, SessionConfig, SessionManager};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::time::{timeout, Duration};

/// Embedded model weights (compiled into binary)
static SILERO_VAD_MODEL: &[u8] = include_bytes!("../models/silero_vad.onnx");
static DEEPFILTER_MODEL: &[u8] = include_bytes!("../models/deepfilternet3.onnx");

/// Global uptime counter
static UPTIME_SECS: AtomicU64 = AtomicU64::new(0);

/// Per-message idle timeout in seconds. Resets on every received message so a
/// long-but-active session is never cut off; only true idleness terminates it.
const IDLE_TIMEOUT_SECS: u64 = 30;

/// Channel capacity for egress
const EGRESS_CHANNEL_CAPACITY: usize = 100;

/// Maximum accepted inbound WebSocket binary message size (1 MiB), coordinated
/// with the HTTP body limit in `api_routes`.
const MAX_WS_MESSAGE_BYTES: usize = 1024 * 1024;

/// RAII guard that unregisters a session from the [`SessionManager`] on drop,
/// so admission-control slots are always released even on early return/panic.
struct SessionGuard(Arc<SessionManager>);

impl Drop for SessionGuard {
    fn drop(&mut self) {
        self.0.unregister();
    }
}

/// WebSocket upgrade handler with admission control.
///
/// Rejects the upgrade with 503 when the session limit is reached; otherwise
/// reserves a slot (released via [`SessionGuard`]) and proceeds.
async fn realtime_handler(
    ws: WebSocketUpgrade,
    sessions: Arc<SessionManager>,
    metrics: SharedMetrics,
) -> axum::response::Response {
    if sessions.register().is_err() {
        tracing::warn!("Session limit reached; rejecting new connection with 503");
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "session limit reached",
        )
            .into_response();
    }

    let guard = SessionGuard(sessions);
    metrics.record_session_start();
    let metrics_for_session = metrics;

    ws.on_upgrade(move |socket| handle_voice_session(socket, guard, metrics_for_session))
        .into_response()
}

/// Health check endpoint.
async fn health_check() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "version": "5.0.0",
        "engine": "szca-media-gateway",
        "uptime_secs": UPTIME_SECS.load(Ordering::Relaxed),
        "models": {
            "silero_vad": SILERO_VAD_MODEL.len() / 1024,
            "deepfilternet": DEEPFILTER_MODEL.len() / 1024
        }
    }))
}

/// Handle a single voice streaming session.
///
/// `_guard` releases the admission-control slot on drop; `metrics` records the
/// session end.
async fn handle_voice_session(
    socket: WebSocket,
    _guard: SessionGuard,
    metrics: SharedMetrics,
) {
    let (mut ws_sender, mut ws_receiver) = socket.split();

    // Create session
    let session_config = SessionConfig::default();
    let mut session = Session::new(session_config);

    tracing::info!(session_id = %session.id(), "New session started");

    // Create egress channel (engine → client)
    let (tx_pcm_out, mut rx_pcm_out) = mpsc::channel::<Vec<u8>>(EGRESS_CHANNEL_CAPACITY);

    // Egress task: forward audio from engine to client
    let egress = tokio::spawn(async move {
        while let Some(pcm_16khz) = rx_pcm_out.recv().await {
            if ws_sender.send(Message::Binary(pcm_16khz)).await.is_err() {
                tracing::warn!("Client disconnected during egress");
                break;
            }
        }
    });

    // Ingress: process incoming audio.
    let audio_config = AudioConfig::default();
    let mut dsp_processor = dsp::DspProcessor::new(dsp::DspConfig::default());
    let mut vad_processor = vad::VadProcessor::new(vad::VadConfig::default());

    process_ingress(
        &mut ws_receiver,
        &mut session,
        &mut dsp_processor,
        &mut vad_processor,
        &audio_config,
        &tx_pcm_out,
    )
    .await;

    session.end();
    metrics.record_session_end();
    egress.abort();
}

/// Process incoming WebSocket messages.
///
/// Applies a per-message idle timeout (resets on every received message).
/// Barge-in is handled as a one-shot flush: on a barge-in event we signal the
/// client and immediately reset the cancel flag so subsequent audio is still
/// processed. It is NOT used as a permanent gate that drops messages.
async fn process_ingress(
    ws_receiver: &mut futures_util::stream::SplitStream<WebSocket>,
    session: &mut Session,
    dsp_processor: &mut dsp::DspProcessor,
    vad_processor: &mut vad::VadProcessor,
    audio_config: &AudioConfig,
    tx_pcm_out: &mpsc::Sender<Vec<u8>>,
) {
    loop {
        // Per-message idle timeout: reset on each message, terminate on idle.
        let next = match timeout(
            Duration::from_secs(IDLE_TIMEOUT_SECS),
            ws_receiver.next(),
        )
        .await
        {
            Ok(Some(Ok(msg))) => msg,
            Ok(Some(Err(_))) | Ok(None) => {
                tracing::info!(session_id = %session.id(), "Session ended normally");
                break;
            }
            Err(_) => {
                tracing::warn!(
                    session_id = %session.id(),
                    "Session idle for {}s, closing",
                    IDLE_TIMEOUT_SECS
                );
                break;
            }
        };

        match next {
            Message::Binary(data) => {
                // M10: reject oversize binary frames before decoding.
                if data.len() > MAX_WS_MESSAGE_BYTES {
                    tracing::warn!(
                        len = data.len(),
                        max = MAX_WS_MESSAGE_BYTES,
                        "Dropping oversize WebSocket message"
                    );
                    continue;
                }

                let frames = gateway::process_incoming_message(
                    &data,
                    session,
                    audio_config,
                );

                match frames {
                    Ok(frames) => {
                        for frame in frames {
                            if let protocol::Frame::Audio(pcm_data) = frame {
                                // Step 1: Noise suppression
                                match dsp_processor.process(&pcm_data) {
                                    Ok(clean_pcm) => {
                                        // Step 2: VAD
                                        let is_tts_playing = session.is_tts_playing();
                                        let event = vad_processor.process(&clean_pcm, is_tts_playing);

                                        match event {
                                            vad::VadEvent::BargeIn => {
                                                // One-shot barge-in: flush egress
                                                // (signal client) then reset so
                                                // subsequent audio keeps flowing.
                                                session.barge_in();
                                                if tx_pcm_out
                                                    .send(protocol::encode_control_frame(
                                                        protocol::Opcode::Interrupt,
                                                    ))
                                                    .await
                                                    .is_err()
                                                {
                                                    tracing::warn!("Failed to send barge-in signal — receiver dropped");
                                                }
                                                session.reset_barge_in();
                                            }
                                            vad::VadEvent::Speech
                                            | vad::VadEvent::SpeechStart => {
                                                tracing::debug!("Speech detected, forwarding processed PCM to egress");
                                                // Forward processed PCM downstream.
                                                if tx_pcm_out.send(clean_pcm).await.is_err() {
                                                    tracing::warn!("Egress receiver dropped; stopping ingress");
                                                    return;
                                                }
                                            }
                                            _ => {}
                                        }
                                    }
                                    Err(e) => {
                                        tracing::error!(error = %e, "DSP processing failed");
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!(error = %e, "Gateway message processing failed");
                    }
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }
}

/// Print startup banner.
fn print_banner() {
    println!("┌─────────────────────────────────────────────────────┐");
    println!("│  SZCA Media Gateway v5.0.0                          │");
    println!("│  SRAM-Mesh Zero-Copy Architecture                   │");
    println!("├─────────────────────────────────────────────────────┤");
    println!("│  Models: Silero VAD ({}KB), DeepFilter ({}KB)    │",
             SILERO_VAD_MODEL.len() / 1024,
             DEEPFILTER_MODEL.len() / 1024);
    println!("│  Audio:  16kHz PCM 16-bit Mono                      │");
    println!("│  APIs:   /v1/realtime (WS), /v1/stt, /v1/llm, /v1/tts │");
    println!("│  Health: /health                                    │");
    println!("└─────────────────────────────────────────────────────┘");
}

#[tokio::main]
async fn main() {
    // Initialize structured logging
    tracing_subscriber::fmt::init();

    print_banner();

    if std::env::var("SZCA_API_KEY").map(|v| v.is_empty()).unwrap_or(true) {
        tracing::warn!(
            "SZCA_API_KEY is not set — API authentication is DISABLED. Set it to enforce bearer-token auth on /v1/* and /metrics."
        );
    }

    let config = GatewayConfig::default();
    let addr = format!("{}:{}", config.listen_addr, config.port);

    tracing::info!(addr = %addr, "Starting SZCA Media Gateway");

    // Shared admission control + metrics.
    let sessions = Arc::new(SessionManager::new(config.max_sessions));
    let metrics = metrics::create_shared_metrics();

    let realtime_route = {
        let sessions = Arc::clone(&sessions);
        let metrics = Arc::clone(&metrics);
        get(move |ws: WebSocketUpgrade| {
            realtime_handler(ws, Arc::clone(&sessions), Arc::clone(&metrics))
        })
    };

    let app = Router::new()
        .route("/v1/realtime", realtime_route)
        .route("/health", get(health_check))
        .merge(api_routes::api_router(Arc::clone(&metrics)));

    // Bind with graceful error handling (no unwrap)
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => {
            tracing::info!(addr = %addr, "Successfully bound to address");
            l
        }
        Err(e) => {
            tracing::error!(error = %e, addr = %addr, "Failed to bind to address");
            std::process::exit(1);
        }
    };

    tracing::info!(
        endpoints = "POST /v1/stt/stream, POST /v1/llm/stream, POST /v1/tts/stream, GET /health",
        "Server ready"
    );

    // Graceful shutdown on SIGTERM/SIGINT
    let shutdown_signal = async {
        let _ = tokio::signal::ctrl_c().await;
        tracing::info!("Received shutdown signal, stopping gracefully...");
    };

    if let Err(e) = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal)
        .await
    {
        tracing::error!(error = %e, "Server error");
        std::process::exit(1);
    }

    tracing::info!("Server stopped");
}
