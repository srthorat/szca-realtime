import ws from 'k6/ws';
import { check, sleep } from 'k6';
import { Trend, Counter } from 'k6/metrics';

// Custom metrics to track during the load test
const ttft = new Trend('ttft_ms', true);          // Time To First Token (from send to first response)
const connectionErrors = new Counter('connection_errors');

export const options = {
    // 1000 concurrent sessions for 2 minutes to stress test the SZCA_QUEUE_BACKLOG and GPU
    stages: [
        { duration: '30s', target: 1000 }, // Ramp up to 1000 users over 30 seconds
        { duration: '90s', target: 1000 }, // Hold at 1000 users for 90 seconds
        { duration: '10s', target: 0 },    // Ramp down to 0
    ],
};

export default function () {
    const url = __ENV.WS_URL || 'ws://localhost:3000/v1/realtime';
    const apiKey = __ENV.API_KEY || '';
    
    const params = {
        headers: { 'Authorization': `Bearer ${apiKey}` }
    };

    const res = ws.connect(url, params, function (socket) {
        let sessionStarted = Date.now();
        let requestTime = 0;
        let gotFirstAudio = false;

        socket.on('open', function () {
            // Send a standard initialization event (using OpenAI Realtime dialect for example)
            socket.send(JSON.stringify({
                type: 'session.update',
                session: {
                    modalities: ["text", "audio"],
                    instructions: "You are a helpful assistant."
                }
            }));

            // Simulate the user speaking a short sentence by sending a dummy base64 PCM frame
            // In a real test, this would be actual audio data to force STT to decode.
            // For capacity testing the queue and LLM, we send a client text event.
            requestTime = Date.now();
            socket.send(JSON.stringify({
                type: 'conversation.item.create',
                item: {
                    type: "message",
                    role: "user",
                    content: [{
                        type: "input_text",
                        text: "Tell me a short joke."
                    }]
                }
            }));
            
            socket.send(JSON.stringify({ type: 'response.create' }));
        });

        socket.on('message', function (msg) {
            const event = JSON.parse(msg);
            
            // Track Time To First Token (TTFT) when we get our first text or audio delta
            if (!gotFirstAudio && (event.type === 'response.audio.delta' || event.type === 'response.text.delta')) {
                const latency = Date.now() - requestTime;
                ttft.add(latency);
                gotFirstAudio = true;
            }

            // Close the connection once the turn is complete
            if (event.type === 'response.done') {
                socket.close();
            }
        });

        socket.on('error', function (e) {
            connectionErrors.add(1);
            if (__ENV.DEBUG) {
                console.log('An unexpected error occurred: ', e.error());
            }
        });

        socket.on('close', function () {
            sleep(1);
        });
    });

    check(res, { 'status is 101': (r) => r && r.status === 101 });
}
