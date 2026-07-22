// SZCA Load Test Script (k6)
// Usage: k6 run load_test.js

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const latency = new Trend('request_latency');

// Test configuration
export const options = {
  stages: [
    { duration: '10s', target: 10 },   // Ramp up to 10 users
    { duration: '30s', target: 50 },   // Ramp up to 50 users
    { duration: '30s', target: 100 },  // Ramp up to 100 users
    { duration: '30s', target: 50 },   // Ramp down to 50 users
    { duration: '10s', target: 0 },    // Ramp down to 0
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'],  // 95% of requests under 500ms
    http_req_failed: ['rate<0.01'],    // Less than 1% error rate
    errors: ['rate<0.01'],
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8080';

// Test: Health check
export function testHealth() {
  const res = http.get(`${BASE_URL}/health`);
  check(res, {
    'health status is 200': (r) => r.status === 200,
    'health response has status ok': (r) => JSON.parse(r.body).status === 'ok',
  });
  latency.add(res.timings.duration);
  errorRate.add(res.status !== 200);
}

// Test: STT endpoint
export function testSTT() {
  const payload = JSON.stringify({
    model: 'parakeet_tdt_0.6b_v3',
    language: 'en',
    interim_results: true,
  });

  const params = {
    headers: { 'Content-Type': 'application/json' },
  };

  const res = http.post(`${BASE_URL}/v1/stt/stream`, payload, params);
  check(res, {
    'stt status is 200': (r) => r.status === 200,
  });
  latency.add(res.timings.duration);
  errorRate.add(res.status !== 200);
}

// Test: LLM endpoint
export function testLLM() {
  const payload = JSON.stringify({
    model: 'hermes-3-3b',
    messages: [{ role: 'user', content: 'Hello' }],
    stream: true,
    max_tokens: 100,
  });

  const params = {
    headers: { 'Content-Type': 'application/json' },
  };

  const res = http.post(`${BASE_URL}/v1/llm/stream`, payload, params);
  check(res, {
    'llm status is 200': (r) => r.status === 200,
  });
  latency.add(res.timings.duration);
  errorRate.add(res.status !== 200);
}

// Test: TTS endpoint
export function testTTS() {
  const payload = JSON.stringify({
    model: 'kokoro-82m',
    voice: 'af_heart',
    input: 'Hello world',
    stream: true,
  });

  const params = {
    headers: { 'Content-Type': 'application/json' },
  };

  const res = http.post(`${BASE_URL}/v1/tts/stream`, payload, params);
  check(res, {
    'tts status is 200': (r) => r.status === 200,
  });
  latency.add(res.timings.duration);
  errorRate.add(res.status !== 200);
}

// Main test function
export default function () {
  // Run each test with weighted probability
  const rand = Math.random();

  if (rand < 0.3) {
    testHealth();
  } else if (rand < 0.5) {
    testSTT();
  } else if (rand < 0.8) {
    testLLM();
  } else {
    testTTS();
  }

  sleep(0.1); // 100ms between requests
}

// Setup function (runs once before test)
export function setup() {
  console.log(`Running load test against ${BASE_URL}`);
  console.log('Ramping up to 100 concurrent users...');

  // Verify server is running
  const res = http.get(`${BASE_URL}/health`);
  if (res.status !== 200) {
    throw new Error(`Server not running at ${BASE_URL}`);
  }
}

// Teardown function (runs once after test)
export function teardown(data) {
  console.log('Load test complete');
}
