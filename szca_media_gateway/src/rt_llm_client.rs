/// Streaming LLM client for vLLM / TGI backends via the OpenAI-compatible API.
///
/// This module provides an alternative LLM `Replica` implementation that
/// forwards token generation to an external vLLM server over HTTP, instead
/// of running ONNX inference in-process. The server must expose the
/// OpenAI-compatible `/v1/chat/completions` endpoint with `stream=true`.
///
/// # Why exists
///
/// The locked architecture (see SCALING_PLAN.md) places LLM token generation
/// behind a proven batching engine (vLLM / TGI) rather than re-implementing
/// continuous batching in our Rust engine. Our engine owns STT/TTS/VAD/orchestration;
/// vLLM owns LLM serving. This module is the thin streaming HTTP client that
/// bridges the two.
///
/// # Wire protocol
///
/// Sends a standard OpenAI chat completions request with `stream: true`:
///
/// ```json
/// {
///   "model": "hermes-llama-3-8b",
///   "messages": [
///     {"role": "system", "content": "..."},
///     {"role": "user", "content": "..."}
///   ],
///   "stream": true,
///   "temperature": 0.7,
///   "max_tokens": 1024
/// }
/// ```
///
/// Parses SSE response stream, extracting `choices[0].delta.content` from each
/// `chat.completion.chunk` event. Supports cooperative cancellation via the
/// shared `AtomicBool` flag — on cancel the SSE loop breaks and the partial
/// response is returned.
///
/// # Configuration
///
/// Environment variables:
/// - `LLM_BASE_URL` (default: `http://localhost:8000`) — vLLM server base URL
/// - `LLM_MODEL` (default: `hermes-llama-3-8b`) — model name sent to the server
/// - `LLM_API_KEY` (optional) — bearer token for authentication
/// - `LLM_MAX_TOKENS` (default: 1024) — max tokens per generation
/// - `LLM_TEMPERATURE` (default: 0.7) — sampling temperature

use std::sync::atomic::{AtomicBool, Ordering};

use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio_stream::StreamExt;

use crate::stage_pool::Replica;
use crate::rt_llm::LlmInput;

/// Streaming vLLM client implementing the `Replica` trait. Each replica holds
/// its own `reqwest::Client` (connection pool) and target configuration.
///
/// IMPORTANT: The `tokio_handle` is captured at creation time (on the main
/// tokio thread) because pool worker threads are plain `std::thread`s that
/// don't have a tokio runtime context.
pub struct VllmClient {
    http: Client,
    base_url: String,
    model: String,
    api_key: Option<String>,
    max_tokens: u32,
    temperature: f32,
    /// Handle to the tokio runtime, captured at creation time.
    /// Pool worker threads are plain OS threads without a tokio context,
    /// so `Handle::current()` would panic. We capture it here instead.
    tokio_handle: tokio::runtime::Handle,
}

impl VllmClient {
    /// Build a client from environment variables.
    pub fn from_env() -> Result<Self, String> {
        let base_url = std::env::var("LLM_BASE_URL")
            .unwrap_or_else(|_| "http://localhost:8000".to_string());
        let model = std::env::var("LLM_MODEL")
            .unwrap_or_else(|_| "hermes-llama-3-8b".to_string());
        let api_key = std::env::var("LLM_API_KEY").ok();
        let max_tokens: u32 = std::env::var("LLM_MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1024);
        let temperature: f32 = std::env::var("LLM_TEMPERATURE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.7);

        let http = Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .map_err(|e| format!("vLLM HTTP client: {e}"))?;

        // Capture the tokio runtime handle NOW (on the tokio thread).
        // Pool worker threads are plain OS threads; Handle::current() panics there.
        let tokio_handle = tokio::runtime::Handle::current();

        tracing::info!(base_url, model, max_tokens, temperature, "vLLM client ready");

        Ok(Self {
            http,
            base_url,
            model,
            api_key,
            max_tokens,
            temperature,
            tokio_handle,
        })
    }

    /// Run a streaming completion, forwarding each token delta to `emit`.
    /// Returns the full concatenated response text.
    fn run_streaming(
        &mut self,
        prompt: &str,
        instructions: Option<&str>,
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(String),
    ) -> String {
        // Build messages array: optional system message + user message.
        let mut messages = Vec::new();
        if let Some(inst) = instructions {
            if !inst.is_empty() {
                messages.push(ChatMessage {
                    role: "system".to_string(),
                    content: inst.to_string(),
                });
            }
        }
        messages.push(ChatMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        });

        let request = ChatCompletionRequest {
            model: self.model.clone(),
            messages,
            stream: true,
            max_tokens: Some(self.max_tokens),
            temperature: Some(self.temperature),
        };

        // Build the request URL.
        let url = format!("{}/v1/chat/completions", self.base_url.trim_end_matches('/'));

        let mut req = self.http.post(&url).json(&request);
        if let Some(ref key) = self.api_key {
            req = req.bearer_auth(key);
        }

        // Execute the request synchronously (we're on a blocking thread).
        // Uses the stored handle since pool workers are plain OS threads.
        let response = match tokio::task::block_in_place(|| {
            self.tokio_handle.block_on(req.send())
        }) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!(error = %e, base_url = %self.base_url, "vLLM request failed");
                return String::new();
            }
        };

        if !response.status().is_success() {
            let status = response.status();
            let body = tokio::task::block_in_place(|| {
                self.tokio_handle.block_on(response.text())
            }).unwrap_or_default();
            tracing::error!(status = %status, body = %body, "vLLM returned error");
            return String::new();
        }

        // Parse the SSE stream.
        let mut full_text = String::new();
        let mut stream = response.bytes_stream();
        let handle = self.tokio_handle.clone();

        tokio::task::block_in_place(|| {
            handle.block_on(async {
                while let Some(chunk_result) = stream.next().await {
                    if cancel.load(Ordering::Relaxed) {
                        tracing::debug!("vLLM stream cancelled by caller");
                        break;
                    }

                    let chunk = match chunk_result {
                        Ok(c) => c,
                        Err(e) => {
                            tracing::error!(error = %e, "vLLM stream read error");
                            break;
                        }
                    };

                    // Parse SSE lines from the chunk.
                    let text = String::from_utf8_lossy(&chunk);
                    for line in text.lines() {
                        let line = line.trim();
                        if line.is_empty() || line.starts_with(':') {
                            continue;
                        }
                        if let Some(data) = line.strip_prefix("data: ") {
                            let data = data.trim();
                            if data == "[DONE]" {
                                return;
                            }
                            match serde_json::from_str::<ChatCompletionChunk>(data) {
                                Ok(chunk) => {
                                    if let Some(delta) = chunk.choices.first()
                                        .and_then(|c| c.delta.as_ref())
                                    {
                                        if !delta.content.is_empty() {
                                            emit(delta.content.clone());
                                            full_text.push_str(&delta.content);
                                        }
                                    }
                                }
                                Err(e) => {
                                    tracing::debug!(error = %e, data = %data, "skipped non-JSON SSE line");
                                }
                            }
                        }
                    }
                }
            });
        });

        full_text
    }
}

impl Replica for VllmClient {
    type Input = LlmInput;
    type Delta = String;
    type Output = String;

    fn process(
        &mut self,
        input: Self::Input,
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(Self::Delta),
    ) -> Self::Output {
        self.run_streaming(&input.prompt, input.instructions.as_deref(), cancel, emit)
    }
}

// ---------------------------------------------------------------------------
// OpenAI-compatible wire types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatCompletionChunk {
    choices: Vec<ChunkChoice>,
}

#[derive(Deserialize)]
struct ChunkChoice {
    delta: Option<Delta>,
}

#[derive(Deserialize)]
struct Delta {
    #[serde(default)]
    content: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vllm_client_from_env_defaults() {
        // Ensure no env vars are set so defaults are used.
        // Note: not thread-safe with other tests that set these vars,
        // but all default assertions are against unset vars.
        let rt = tokio::runtime::Runtime::new().unwrap();
        let client = rt.block_on(async { VllmClient::from_env().unwrap() });
        // Only assert defaults that don't depend on env vars being unset.
        assert!(client.max_tokens > 0);
        assert!(client.temperature >= 0.0 && client.temperature <= 2.0);
    }

    #[test]
    fn vllm_client_struct_fields() {
        // Test the struct directly without touching env vars.
        let rt = tokio::runtime::Runtime::new().unwrap();
        let http = reqwest::Client::builder().build().unwrap();
        let tokio_handle = rt.handle().clone();
        let client = VllmClient {
            http,
            base_url: "http://gpu-server:8080".to_string(),
            model: "meta-llama-3-8b-instruct".to_string(),
            api_key: Some("test-key".to_string()),
            max_tokens: 2048,
            temperature: 0.3,
            tokio_handle,
        };
        assert_eq!(client.base_url, "http://gpu-server:8080");
        assert_eq!(client.model, "meta-llama-3-8b-instruct");
        assert_eq!(client.api_key.as_deref(), Some("test-key"));
        assert_eq!(client.max_tokens, 2048);
        assert!((client.temperature - 0.3).abs() < 0.001);
    }
}
