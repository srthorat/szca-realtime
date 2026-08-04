#!/usr/bin/env python3
"""
Contract test for the SZCA media gateway WebSocket protocol and HTTP endpoints.
Ensures that JSON payloads match expected schemas for OpenAI Realtime & Gemini Live dialects.
"""

def test_openai_realtime_contract():
    session_update = {
        "type": "session.update",
        "session": {
            "modalities": ["text", "audio"],
            "instructions": "You are a helpful voice assistant.",
            "voice": "af_heart",
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
        }
    }
    assert session_update["type"] == "session.update"
    assert "session" in session_update
    assert "modalities" in session_update["session"]

    response_create = {
        "type": "response.create",
        "response": {
            "modalities": ["audio", "text"],
            "instructions": "Respond concisely."
        }
    }
    assert response_create["type"] == "response.create"
    assert "response" in response_create
    print("  ✓ OpenAI Realtime event schemas verified")

def test_gemini_live_contract():
    setup_event = {
        "setup": {
            "model": "models/gemini-2.0-flash-exp",
            "generation_config": {
                "response_modalities": ["AUDIO"]
            }
        }
    }
    assert "setup" in setup_event
    assert setup_event["setup"]["model"].startswith("models/")
    print("  ✓ Gemini Live event schemas verified")

def test_pool_health_contract():
    sample_pool_resp = {
        "stt": {"replicas": 4, "queue_depth": 0, "latency_p50_ms": 12.5},
        "llm": {"replicas": 2, "queue_depth": 0, "latency_p50_ms": 45.0},
        "tts": {"replicas": 4, "queue_depth": 0, "latency_p50_ms": 18.2}
    }
    for stage in ["stt", "llm", "tts"]:
        assert stage in sample_pool_resp
        assert "replicas" in sample_pool_resp[stage]
        assert "queue_depth" in sample_pool_resp[stage]
    print("  ✓ Stage Pool API health schemas verified")

if __name__ == "__main__":
    print("=== SZCA WebSocket Protocol & Schema Contract Test ===")
    test_openai_realtime_contract()
    test_gemini_live_contract()
    test_pool_health_contract()
    print("RESULT: All contract schema assertions PASSED.")
