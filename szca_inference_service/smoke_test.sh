#!/usr/bin/env bash
# End-to-end smoke test for the SZCA inference service.
#
# Exercises all four endpoints with REAL audio/text and asserts correct output.
# The strongest check is the TTS<->STT round-trip: we synthesize a known
# sentence, transcribe it back, and require the text to survive.
#
#   BASE=http://localhost:8900 ./smoke_test.sh
set -euo pipefail

BASE="${BASE:-http://localhost:8900}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }

echo "== readiness =="
curl -fsS "$BASE/ready" >/dev/null || fail "/ready not OK"
echo "  ready"

echo "== [1/4] TTS: text -> WAV =="
SENTENCE="The quick brown fox jumps over the lazy dog."
curl -fsS -X POST "$BASE/v1/tts" -H 'Content-Type: application/json' \
  -d "{\"text\":\"$SENTENCE\"}" -o "$TMP/tts.wav"
[ -s "$TMP/tts.wav" ] || fail "TTS produced empty audio"
echo "  synthesized $(wc -c < "$TMP/tts.wav") bytes"

echo "== [2/4] STT: WAV -> text (round-trip) =="
STT_JSON="$(curl -fsS -X POST "$BASE/v1/stt" -F "file=@$TMP/tts.wav")"
echo "  transcript: $STT_JSON"
# Require the round-trip to recover the key content words.
echo "$STT_JSON" | grep -iq "quick brown fox" || fail "round-trip lost 'quick brown fox'"
echo "$STT_JSON" | grep -iq "lazy dog" || fail "round-trip lost 'lazy dog'"
echo "  round-trip OK"

echo "== [3/4] LLM: prompt -> reply =="
LLM_JSON="$(curl -fsS -X POST "$BASE/v1/llm" -H 'Content-Type: application/json' \
  -d '{"prompt":"What is the capital of France? Answer in one word.","max_new_tokens":30,"temperature":0.0}')"
echo "  reply: $LLM_JSON"
echo "$LLM_JSON" | grep -iq "paris" || fail "LLM did not mention Paris"
echo "  LLM correctness OK"

echo "== [4/4] PIPELINE: voice -> voice =="
curl -fsS -X POST "$BASE/v1/tts" -H 'Content-Type: application/json' \
  -d '{"text":"What is the capital of France?"}' -o "$TMP/q.wav"
curl -fsS -X POST "$BASE/v1/pipeline" -F "file=@$TMP/q.wav" \
  -o "$TMP/reply.wav" -D "$TMP/h.txt"
[ -s "$TMP/reply.wav" ] || fail "pipeline produced empty audio"
TRANSCRIPT="$(grep -i '^x-transcript:' "$TMP/h.txt" | awk '{print $2}' | tr -d '\r' | base64 --decode)"
REPLY="$(grep -i '^x-reply:' "$TMP/h.txt" | awk '{print $2}' | tr -d '\r' | base64 --decode)"
echo "  transcript: $TRANSCRIPT"
echo "  reply: $REPLY"
echo "$REPLY" | grep -iq "paris" || fail "pipeline reply did not mention Paris"
echo "  pipeline OK ($(wc -c < "$TMP/reply.wav") bytes of speech)"

echo ""
echo "ALL 4 ENDPOINTS PASSED (real inference)."
