/// SZCA Media Gateway — Entry Point
///
/// Real-time voice streaming gateway with:
/// - Binary WebSocket protocol (16kHz PCM)
/// - DeepFilterNet3 noise suppression
/// - Silero VAD speech detection
/// - Atomic barge-in interrupts
/// - In-process StagePool inference (STT / LLM / TTS)
/// - HTTP SSE APIs (STT, LLM, TTS)
/// - Health check endpoint
/// - Graceful shutdown

use szca_media_gateway::{
    admission::SessionManager,
    api_routes,
    config::GatewayConfig,
    dfn3,
    dsp,
    metrics,
    rt_protocol::DialectKind,
    rt_session::{self, RealtimeConfig},
    stage_pools::{dev_model_selection, SttModel, StagePools},
    vad,
};

use axum::{
    extract::{
        ws::{WebSocket, WebSocketUpgrade},
        RawQuery,
    },
    http::StatusCode,
    response::IntoResponse,
    routing::get,
    Json, Router,
};
use metrics::SharedMetrics;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use uuid::Uuid;

/// Default on-disk locations for the two DSP models, relative to the CWD.
///
/// These were previously `include_bytes!`-embedded, which baked 2.3 MB into the
/// binary purely to report a size in `/health` — the inference paths always
/// loaded from disk. Worse, it made the model files a BUILD dependency: a fresh
/// clone could not compile until `download_models.sh` had run, and an empty
/// placeholder file compiled fine while reporting `0 KB`. Resolving at runtime
/// means the health endpoint reports what is actually loadable.
const DEFAULT_SILERO_PATH: &str = "./models/vad/silero_vad.onnx";
const DEFAULT_DFN3_DIR: &str = "./models/dfn3";

/// Global uptime counter
static UPTIME_SECS: AtomicU64 = AtomicU64::new(0);

/// RAII guard that unregisters a session from the [`SessionManager`] on drop,
/// so admission-control slots are always released even on early return/panic.
struct SessionGuard(Arc<SessionManager>);

impl Drop for SessionGuard {
    fn drop(&mut self) {
        self.0.unregister();
    }
}

/// Build the default VAD config for a realtime session.
///
/// Real Silero VAD is enabled by pointing at an on-disk model: `SILERO_VAD_MODEL`
/// if set, else [`DEFAULT_SILERO_PATH`]. When the model (or ONNX Runtime via
/// `ORT_DYLIB_PATH`) is unavailable, [`vad::VadProcessor`] logs and falls back to
/// the RMS-energy heuristic, so the gateway still runs without model weights.
fn default_vad_config() -> vad::VadConfig {
    let model_path = std::env::var("SILERO_VAD_MODEL")
        .ok()
        .filter(|p| !p.is_empty())
        .or_else(|| {
            std::path::Path::new(DEFAULT_SILERO_PATH)
                .exists()
                .then(|| DEFAULT_SILERO_PATH.to_string())
        });
    vad::VadConfig {
        model_path,
        ..Default::default()
    }
}

/// WebSocket upgrade handler with admission control.
///
/// Rejects the upgrade with 503 when the session limit is reached; otherwise
/// reserves a slot (released via [`SessionGuard`]) and proceeds. The wire
/// dialect (OpenAI Realtime / Gemini Live) is chosen from the `?dialect=` query.
async fn realtime_handler(
    ws: WebSocketUpgrade,
    RawQuery(query): RawQuery,
    sessions: Arc<SessionManager>,
    metrics: SharedMetrics,
    pools: Option<Arc<StagePools>>,
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
    let dialect = DialectKind::from_query(query.as_deref().unwrap_or(""));

    ws.on_upgrade(move |socket| {
        handle_voice_session(socket, guard, metrics_for_session, dialect, pools)
    })
    .into_response()
}

/// Size of `path` in KiB, or 0 when it is absent/unreadable.
///
/// 0 therefore means "this DSP model is NOT available" — the gateway still
/// serves (VAD degrades to the RMS heuristic), so the number is the only signal
/// that a deployment is missing weights.
fn model_kb(path: &str) -> u64 {
    std::fs::metadata(path).map(|m| m.len() / 1024).unwrap_or(0)
}

/// Health check endpoint.
async fn health_check() -> Json<serde_json::Value> {
    let dfn3_dir = std::env::var("DFN3_MODEL_DIR")
        .ok()
        .filter(|p| !p.is_empty())
        .unwrap_or_else(|| DEFAULT_DFN3_DIR.to_string());
    let silero = std::env::var("SILERO_VAD_MODEL")
        .ok()
        .filter(|p| !p.is_empty())
        .unwrap_or_else(|| DEFAULT_SILERO_PATH.to_string());
    // Which STT backend is live. `STT_BACKEND` falls back to `parakeet` on any
    // unrecognized value, so echoing the RESOLVED choice (not the raw env var) is
    // what makes a typo visible: a misspelt `streming` shows up here as
    // `parakeet`, which is the only signal that the requested model isn't loaded.
    let stt_backend = match dev_model_selection() {
        SttModel::Zipformer => "sherpa-zipformer",
        SttModel::Streaming => "streaming-eou",
        SttModel::Parakeet => "parakeet-tdt",
    };
    // Check LLM, STT, TTS model directories for completeness reporting.
    let llm_dir = std::env::var("LLM_MODEL_DIR")
        .ok()
        .filter(|p| !p.is_empty())
        .unwrap_or_else(|| "./models/llm".to_string());
    let stt_dir = std::env::var("STT_MODEL_DIR")
        .ok()
        .filter(|p| !p.is_empty())
        .unwrap_or_else(|| "./models/stt".to_string());
    let tts_dir = std::env::var("TTS_MODEL_DIR")
        .ok()
        .filter(|p| !p.is_empty())
        .unwrap_or_else(|| "./models/tts".to_string());
    // Total KiB for all .onnx files in each model directory (0 if empty/missing).
    let dir_kb = |dir: &str| -> u64 {
        std::fs::read_dir(dir).ok()
            .map(|entries| entries.filter_map(|e| e.ok()).map(|e| e.path())
                .filter(|p| p.extension().map(|ext| ext == "onnx" || ext == "bin" || ext == "data").unwrap_or(false))
                .filter_map(|p| std::fs::metadata(p).ok())
                .map(|m| m.len())
                .sum::<u64>() / 1024)
            .unwrap_or(0)
    };
    Json(serde_json::json!({
        "status": "ok",
        "version": "5.0.0",
        "engine": "szca-media-gateway",
        "uptime_secs": UPTIME_SECS.load(Ordering::Relaxed),
        "stt_backend": stt_backend,
        // KiB on disk; 0 = model absent (see `model_kb`).
        "models": {
            "silero_vad": model_kb(&silero),
            "deepfilternet": model_kb(&dfn3::Dfn3Paths::in_dir(&dfn3_dir).enc),
            "stt": dir_kb(&stt_dir),
            "llm": dir_kb(&llm_dir),
            "tts": dir_kb(&tts_dir)
        }
    }))
}

/// Handle a single voice streaming session.
///
/// `_guard` releases the admission-control slot on drop; `metrics` records the
/// session end. Delegates the full bidirectional realtime loop (dialect decode,
/// VAD turn detection, STT->LLM->TTS pipeline, barge-in) to [`rt_session`].
///
async fn handle_voice_session(
    socket: WebSocket,
    _guard: SessionGuard,
    metrics: SharedMetrics,
    dialect: DialectKind,
    pools: Option<Arc<StagePools>>,
) {
    let session_id = Uuid::new_v4().to_string();
    tracing::info!(session_id = %session_id, ?dialect, "New realtime session started");

    // Attempt to load the DFN3 noise-cancellation DSP. Fails silently if
    // models aren't configured — audio flows mic -> VAD -> pipeline without
    // denoising.
    let dsp = {
        let dfn3_dir = std::env::var("DFN3_MODEL_DIR")
            .ok()
            .filter(|p| !p.is_empty())
            .unwrap_or_else(|| DEFAULT_DFN3_DIR.to_string());
        let mut proc = dsp::DspProcessor::new(dsp::DspConfig {
            model_dir: Some(dfn3_dir),
            ..Default::default()
        });
        match proc.initialize() {
            Ok(()) => {
                tracing::info!(
                    "DFN3 noise cancellation active (real model: {})",
                    proc.uses_real_model()
                );
                Some(Box::new(proc))
            }
            Err(e) => {
                tracing::warn!(error = %e, "DFN3 noise cancellation unavailable");
                None
            }
        }
    };

    let config = RealtimeConfig {
        dialect,
        vad: default_vad_config(),
        pools,
        dsp,
    };
    rt_session::run_session(socket, session_id, config).await;

    metrics.record_session_end();
}

/// Print startup banner via tracing (not println).
fn print_banner() {
    tracing::info!("┌─────────────────────────────────────────────────────┐");
    tracing::info!("│  SZCA Media Gateway v5.0.0                          │");
    tracing::info!("│  SRAM-Mesh Zero-Copy Architecture                   │");
    tracing::info!("├─────────────────────────────────────────────────────┤");
    tracing::info!(
        "│  Models: Silero VAD ({}KB), DeepFilter ({}KB)    │",
        model_kb(DEFAULT_SILERO_PATH),
        model_kb(&dfn3::Dfn3Paths::in_dir(DEFAULT_DFN3_DIR).enc)
    );
    tracing::info!("│  Audio:  16kHz PCM 16-bit Mono                      │");
    tracing::info!("│  APIs:   /v1/realtime (WS), /v1/stt, /v1/llm, /v1/tts │");
    tracing::info!("│  Health: /health                                    │");
    tracing::info!("└─────────────────────────────────────────────────────┘");
}

fn main() {
    // Initialize structured logging
    tracing_subscriber::fmt::init();

    // Build the tokio runtime with env-configurable thread pools so the gateway
    // can scale to 1k+ concurrent sessions without starving the async or
    // blocking thread pools. Tunable via SZCA_WORKER_THREADS and
    // SZCA_BLOCKING_THREADS (see env.prod.example).
    let worker_threads = parse_env_u32("SZCA_WORKER_THREADS", num_cpus::get() as u32);
    let max_blocking = parse_env_u32("SZCA_BLOCKING_THREADS", 512);

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(worker_threads as usize)
        .max_blocking_threads(max_blocking as usize)
        .enable_all()
        .build()
        .expect("tokio runtime build failed");

    rt.block_on(async {
        inner_main().await;
    });
}

async fn inner_main() {

    print_banner();

    if std::env::var("SZCA_API_KEY").map(|v| v.is_empty()).unwrap_or(true) {
        tracing::warn!(
            "SZCA_API_KEY is not set — API authentication is DISABLED. Set it to enforce bearer-token auth on /v1/* and /metrics."
        );
    }

    let config = GatewayConfig::from_env();
    let addr = format!("{}:{}", config.listen_addr, config.port);

    tracing::info!(
        addr = %addr,
        max_sessions = config.max_sessions,
        "Starting SZCA Media Gateway"
    );

    // Shared admission control + metrics.
    let sessions = Arc::new(SessionManager::new(config.max_sessions));
    let metrics = metrics::create_shared_metrics();

    // Build shared inference pools — models loaded once, shared across all sessions.
    let pools = match StagePools::from_env() {
        Ok(p) => {
            tracing::info!("All configured inference pools ready");
            Some(Arc::new(p))
        }
        Err(e) => {
            tracing::error!(error = %e, "Failed to build inference pools; sessions will use stubs");
            None
        }
    };

    let realtime_route = {
        let sessions = Arc::clone(&sessions);
        let metrics = Arc::clone(&metrics);
        let pools = pools.clone();
        get(move |ws: WebSocketUpgrade, query: RawQuery| {
            realtime_handler(ws, query, Arc::clone(&sessions), Arc::clone(&metrics), pools.clone())
        })
    };

    let route_state = api_routes::RouteState {
        metrics: Arc::clone(&metrics),
        pools: pools.clone(),
    };

    let app = Router::new()
        .route("/v1/realtime", realtime_route)
        .route("/health", get(health_check))
        .merge(api_routes::api_router(route_state));

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

    // Graceful shutdown on SIGTERM/SIGINT:
    // 1. Stop accepting new connections immediately (axum drops the accept loop).
    // 2. Wait up to 30s for in-flight WS sessions to finish their current response.
    // 3. Then exit cleanly.
    let shutdown_signal = async {
        let _ = tokio::signal::ctrl_c().await;
        tracing::info!("Received shutdown signal; draining in-flight sessions (30s timeout)...");
        tokio::time::sleep(std::time::Duration::from_secs(30)).await;
        tracing::info!("Drain timeout reached; exiting.");
    };

    if let Err(e) = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal)
        .await
    {
        tracing::error!(error = %e, "Server error");
        std::process::exit(1);
    }

    tracing::info!("Server stopped cleanly.");
}

/// Parse an env var as a u32, returning `default` if unset or invalid.
fn parse_env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(default)
}
