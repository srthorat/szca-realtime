import json
import time
import struct
import gevent
from locust import HttpUser, task, between, events
import websocket

# 20ms of 16kHz Mono 16-bit PCM audio = 320 samples = 640 bytes
CHUNK_SIZE_BYTES = 640
DUMMY_PCM_CHUNK = b'\x00' * CHUNK_SIZE_BYTES

class RealtimeWebSocketClient:
    def __init__(self, host):
        self.host = host
        ws_url = host.replace("http://", "ws://").replace("https://", "wss://") + "/v1/realtime?dialect=openai"
        self.ws = websocket.WebSocket()
        self.ws.connect(ws_url)

    def send_pcm_chunk(self):
        """Send a 20ms PCM 16-bit mono audio frame."""
        self.ws.send_binary(DUMMY_PCM_CHUNK)

    def recv_event(self):
        return self.ws.recv()

    def close(self):
        self.ws.close()

class SZCASpeechUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        try:
            self.ws_client = RealtimeWebSocketClient(self.host)
        except Exception as e:
            events.request.fire(
                request_type="ws_connect", name="connect", response_time=0, exception=e
            )
            self.ws_client = None

    def on_stop(self):
        if hasattr(self, 'ws_client') and self.ws_client:
            self.ws_client.close()

    @task
    def simulate_audio_stream_turn(self):
        if not self.ws_client:
            return

        start_time = time.time()
        try:
            # Stream 1 second of audio (50 chunks of 20ms PCM audio)
            for _ in range(50):
                self.ws_client.send_pcm_chunk()
                gevent.sleep(0.02)  # Simulate real-time 20ms pace

            elapsed = int((time.time() - start_time) * 1000)
            events.request.fire(
                request_type="ws_stream", name="audio_turn", response_time=elapsed, response_length=50 * CHUNK_SIZE_BYTES, exception=None
            )
        except Exception as e:
            elapsed = int((time.time() - start_time) * 1000)
            events.request.fire(
                request_type="ws_stream", name="audio_turn", response_time=elapsed, response_length=0, exception=e
            )
