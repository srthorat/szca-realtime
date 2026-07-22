# SZCA Test Plan v5.0.0

## Test Suite Overview

| Suite | Tests | Coverage | Status |
|---|---|---|---|
| Rust Gateway Unit | 119 | 100% | Passing |
| C++ Engine Unit | 74 | 100% | Implemented |
| Integration | 15 | Pipeline | Implemented |
| E2E | 12 | User Journeys | Implemented |
| Performance | 10 | Benchmarks | Implemented |
| Security | 18 | Attack Surface | Implemented |
| Metrics | 20 | Quality KPIs | Implemented |
| **Total** | **268** | | |

---

## 1. Unit Tests (193)

### Rust Gateway (119 tests)

| Module | Tests | What is Tested |
|---|---|---|
| ring_buffer.rs | 15 | Push, pop, FIFO, wraparound, full/empty, stress |
| protocol.rs | 21 | Opcode decode, audio encode/decode, alignment |
| dsp.rs | 16 | Init, process, error handling, byte conversion |
| vad.rs | 14 | Silence, speech, barge-in, reset, ratio |
| ipc.rs | 16 | Connect, write, read, buffer full/empty, wraparound |
| session.rs | 22 | State machine, barge-in, stats, manager |
| gateway.rs | 14 | Handshake, audio, interrupt, upgrade, formatting |

### C++ Engine (74 tests)

| Module | Tests | What is Tested |
|---|---|---|
| stt.cpp | 12 | Init, process, silence, speech, final, null |
| llm.cpp | 10 | Init, generate, complete, reset, no messages |
| tts.cpp | 9 | Init, synthesize, voices, empty, not initialized |
| resampler.cpp | 10 | Init, process, ratio, amplitude, null |
| ipc.cpp | 12 | Init, write, read, full, empty, wraparound |
| session.cpp | 12 | State machine, activate, pause, cancel, stats |
| engine.cpp | 9 | Init, process_audio, session, reset, config |

---

## 2. Integration Tests (15)

| Test | Description |
|---|---|
| test_pipeline_silence_in_silence_out | Silence produces no output |
| test_pipeline_speech_in_audio_out | Speech produces audio output |
| test_pipeline_16khz_in_16khz_out | Sample rate preserved |
| test_pipeline_latency_budget | Total latency < 60ms |
| test_pipeline_barge_in_cancels_tts | Barge-in stops TTS |
| test_pipeline_multiple_chunks_streaming | Streaming multiple chunks works |
| test_pipeline_stt_partial_and_final | STT produces both partial and final |
| test_pipeline_llm_generates_tokens | LLM generates tokens |
| test_pipeline_tts_generates_audio | TTS generates audio |
| test_pipeline_empty_input | Empty input handled gracefully |
| test_pipeline_oversized_input | Large input handled gracefully |
| test_pipeline_zero_amplitude | Zero amplitude handled |
| test_pipeline_max_amplitude | Max amplitude handled |
| test_pipeline_negative_amplitude | Negative amplitude handled |
| test_pipeline_concurrent_sessions | 10 concurrent sessions work |

---

## 3. E2E Tests (12)

| Test | Description |
|---|---|
| test_e2e_greeting_conversation | Full greeting flow |
| test_e2e_multi_turn_conversation | Multi-turn dialog |
| test_e2e_barge_in_during_response | Interruption handling |
| test_e2e_long_utterance | 5-second speech |
| test_e2e_silence_timeout | Silence handling |
| test_e2e_session_lifecycle | Connect to hangup |
| test_e2e_ttft_latency | First token latency |
| test_e2e_tpot_latency | Per-token latency |
| test_e2e_stt_latency | STT processing time |
| test_e2e_tts_latency | TTS generation time |
| test_e2e_100_concurrent_sessions | 100 concurrent users |
| test_e2e_sustained_load | 1000 consecutive chunks |

---

## 4. Performance Tests (10)

| Test | Description |
|---|---|
| bench_pipeline_throughput | Max chunks/sec |
| bench_stt_throughput | STT chunks/sec |
| bench_llm_throughput | LLM requests/sec |
| bench_tts_throughput | TTS chunks/sec |
| bench_pipeline_latency_p50 | p50/p95/p99 latencies |
| bench_stt_latency_p50 | STT latency percentiles |
| bench_tts_latency_p50 | TTS latency percentiles |
| bench_concurrent_500_sessions | 500 concurrent sessions |
| bench_concurrent_1000_sessions | 1000 concurrent sessions |
| bench_pipeline_latency_p99 | P99 latency budget |

---

## 5. Security Tests (18)

| Test | Description |
|---|---|
| test_auth_reject_no_token | No token rejected |
| test_auth_reject_invalid_token | Invalid token rejected |
| test_auth_accept_valid_token | Valid token accepted |
| test_auth_reject_expired_token | Expired token rejected |
| test_validate_audio_format | Audio format validation |
| test_validate_audio_chunk_size | Chunk size validation |
| test_validate_text_input | Text input validation |
| test_validate_model_name | Model name validation |
| test_sql_injection_prevention | SQL injection blocked |
| test_xss_prevention | XSS sanitized |
| test_path_traversal_prevention | Path traversal blocked |
| test_buffer_overflow_prevention | Oversized input blocked |
| test_command_injection_prevention | Command injection blocked |
| test_rate_limit_enforced | Rate limiting works |
| test_rate_limit_reset | Rate limit resets |
| test_session_id_uniqueness | Session IDs unique |
| test_session_timeout | Session timeout works |
| test_audio_not_persisted | Audio not stored |

---

## 6. Metrics Tests (20)

### STT Metrics

| Metric | Target | Test |
|---|---|---|
| WER | < 3% | test_stt_wer_accuracy |
| Latency (p50) | < 22ms | test_stt_latency_per_chunk |
| Streaming | Partial + Final | test_stt_streaming_partial_results |
| Final on silence | Yes | test_stt_final_on_silence |

### LLM Metrics

| Metric | Target | Test |
|---|---|---|
| TTFT | < 20ms | test_llm_ttft_latency |
| TPOT | < 5ms | test_llm_tpot_latency |
| Tokens/sec | > 100 | test_llm_tokens_per_second |
| Correctness | Coherent | test_llm_correctness |
| Max tokens | Respected | test_llm_max_tokens_respected |

### TTS Metrics

| Metric | Target | Test |
|---|---|---|
| SNR | > 20 dB | test_tts_audio_quality |
| First chunk latency | < 12ms | test_tts_latency_first_chunk |
| Sample rate | 24kHz | test_tts_sample_rate |
| Multilingual | 5+ langs | test_tts_multilingual |

### E2E Metrics

| Metric | Target | Test |
|---|---|---|
| Glass-to-glass | < 60ms | test_e2e_glass_to_glass_latency |
| 500 concurrent p95 | < 100ms | test_e2e_concurrent_500_latency |

---

## Running Tests

```bash
# Run all tests
chmod +x run_tests.sh
./run_tests.sh

# Run specific suite
cd szca_media_gateway && cargo test
cd szca_tests && cargo test integration
cd szca_tests && cargo test e2e
cd szca_tests && cargo test benchmark
cd szca_tests && cargo test security
cd szca_tests && cargo test metrics
```

---

*SZCA Test Plan v5.0.0 — 268 tests, 100% coverage target*
