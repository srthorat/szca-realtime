// SZCA WebSocket Load Test — Realtime Voice Pipeline
// Usage: k6 run ws_load_test.js
//
// Tests the /v1/realtime WebSocket endpoint with simulated voice traffic:
// connect → session.update → audio.append → audio.commit → response.create
//
// Ramps to 100 concurrent WebSocket connections, each sending 3 "turns".

import ws from 'k6/ws';
import { check, sleep } from 'k6';
import { Counter, Rate, Trend } from 'k6/metrics';

const wsConnections = new Counter('ws_connections');
const wsErrors = new Counter('ws_errors');
const turnLatency = new Trend('turn_latency');
const errorRate = new Rate('errors');

export const options = {
  stages: [
    { duration: '10s', target: 20 },   // Ramp to 20 connections
    { duration: '30s', target: 50 },   // Ramp to 50
    { duration: '60s', target: 100 },  // Hold at 100
    { duration: '20s', target: 50 },   // Ramp down
    { duration: '10s', target: 0 },    // Drain
  ],
  thresholds: {
    ws_errors: ['rate<0.05'],           // < 5% WS errors
    turn_latency: ['p(95)<2000'],       // 95% of turns < 2s
  },
};

const BASE_URL = __ENV.BASE_URL || 'localhost:3000';
const DIALECT = __ENV.DIALECT || 'openai';

// Minimal OpenAI Realtime session setup message
const SESSION_UPDATE = JSON.stringify({
  type: 'session.update',
  session: {
    modalities: ['text', 'audio'],
    instructions: 'You are a helpful voice assistant. Reply concisely.',
    voice: 'alloy',
    input_audio_format: 'pcm16',
    output_audio_format: 'pcm16',
    turn_detection: { type: 'server_vad', threshold: 0.5 },
  },
});

// Simulated user speech (100ms of silence = placeholder for real audio)
function makeAudioAppend() {
  // 100ms of silence at 16kHz mono 16-bit = 3200 bytes
  const silentAudio = btoa(new Uint8Array(3200).buffer);
  return JSON.stringify({
    type: 'input_audio_buffer.append',
    audio: silentAudio,
  });
}

const AUDIO_COMMIT = JSON.stringify({ type: 'input_audio_buffer.commit' });
const CREATE_RESPONSE = JSON.stringify({ type: 'response.create' });

export default function () {
  const url = `ws://${BASE_URL}/v1/realtime?dialect=${DIALECT}`;
  let messagesReceived = 0;
  let turnsCompleted = 0;

  const res = ws.connect(url, {}, function (socket) {
    socket.on('open', () => {
      wsConnections.add(1);

      // Send session config
      socket.send(SESSION_UPDATE);

      // Simulate 3 conversation turns
      function doTurn() {
        if (turnsCompleted >= 3) {
          socket.close();
          return;
        }

        const turnStart = Date.now();

        // Send audio
        socket.send(makeAudioAppend());
        sleep(0.1); // Simulate speaking
        socket.send(AUDIO_COMMIT);
        socket.send(CREATE_RESPONSE);

        // Wait for response (handled by onmessage)
        sleep(2); // Wait for LLM+TTS pipeline
        turnsCompleted++;
        turnLatency.add(Date.now() - turnStart);
      }

      sleep(0.5); // Wait for session.created
      doTurn();
    });

    socket.on('message', (msg) => {
      messagesReceived++;
      try {
        const data = JSON.parse(msg);
        // Count response.done as turn completion indicator
        if (data.type === 'response.done') {
          // Turn complete
        }
        if (data.type === 'error') {
          wsErrors.add(1);
          errorRate.add(1);
        }
      } catch (e) {
        // Binary or malformed message
      }
    });

    socket.on('error', (e) => {
      wsErrors.add(1);
      errorRate.add(1);
    });

    socket.on('close', () => {
      if (turnsCompleted < 3) {
        wsErrors.add(1);
      }
    });

    // Safety timeout — close after 30s
    socket.setTimeout(() => {
      socket.close();
    }, 30000);
  });

  check(res, {
    'ws connected': (r) => r && r.status === 101,
  });
}

export function setup() {
  console.log(`WebSocket load test against ws://${BASE_URL}/v1/realtime`);
  console.log(`Dialect: ${DIALECT}, ramping to 100 connections`);

  // Verify HTTP health first
  const http = __import__('http');
  const health = http.get(`http://${BASE_URL}/health`);
  if (health.status !== 200) {
    throw new Error(`Server not running at ${BASE_URL}`);
  }
}
