#!/usr/bin/env python3
"""
Full e2e test: send audio file through realtime WebSocket,
capture transcript and synthesized audio output.
"""
import asyncio, base64, json, sys, wave, os
import websockets

GATEWAY_URL = "ws://localhost:3000/v1/realtime?dialect=openai"

async def test_e2e(wav_path):
    audio = wave.open(wav_path, 'rb')
    print(f"Audio: {audio.getnchannels()}ch {audio.getsampwidth()*8}bit {audio.getframerate()}Hz "
          f"{audio.getnframes()/audio.getframerate():.1f}s")

    if audio.getsampwidth() != 2:
        print("ERROR: not 16-bit PCM")
        return
    pcm = audio.readframes(audio.getnframes())
    audio.close()

    # Convert to base64 for WebSocket
    b64_pcm = base64.b64encode(pcm).decode()

    async with websockets.connect(GATEWAY_URL) as ws:
        print("Connected to gateway\n")

        # Wait for session.created
        resp = json.loads(await ws.recv())
        print(f"← {resp.get('type','?')}  session_id={resp.get('session',{}).get('id','?')}")

        # Send audio
        msg = json.dumps({"type": "input_audio_buffer.append", "audio": b64_pcm})
        await ws.send(msg)
        print(f"→ input_audio_buffer.append  ({len(pcm)} bytes PCM)")

        # Commit audio buffer (end user turn)
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        print("→ input_audio_buffer.commit")

        # Request response
        await ws.send(json.dumps({"type": "response.create"}))
        print("→ response.create\n")

        # Collect all events until response.done or response.cancelled
        transcript = ""
        audio_chunks = []
        while True:
            resp = json.loads(await ws.recv())
            ev = resp.get("type", "?")
            data = next((v for k,v in resp.items() if k not in ("type",)), "{}")
            print(f"  ← {ev}")

            if ev == "conversation.item.input_audio_transcription.delta":
                transcript += resp.get("delta", "")
            elif ev == "conversation.item.input_audio_transcription.completed":
                transcript = resp.get("transcript", transcript)
            elif ev == "response.output_audio.delta":
                pcm_chunk = base64.b64decode(resp.get("delta", ""))
                audio_chunks.append(pcm_chunk)
            elif ev == "response.output_text.delta":
                print(f"     text: {resp.get('delta','')!r}")
            elif ev == "response.done":
                print("\n✅ Response complete")
                break
            elif ev == "response.cancelled":
                print("\n⛔ Response cancelled")
                break

        print(f"\n📝 Transcript: {transcript!r}")

        if audio_chunks:
            out_path = wav_path.replace(".wav", "_output.wav")
            all_pcm = b"".join(audio_chunks)
            wf = wave.open(out_path, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(all_pcm)
            wf.close()
            print(f"🔊 Output audio saved: {out_path} ({len(all_pcm)} bytes, "
                  f"{len(all_pcm)/32000:.1f}s)")
        else:
            print("⚠ No audio output received")

if __name__ == "__main__":
    wav = sys.argv[1] if len(sys.argv) > 1 else "/Users/sthorat/Documents/odysse.wav"
    asyncio.run(test_e2e(wav))
