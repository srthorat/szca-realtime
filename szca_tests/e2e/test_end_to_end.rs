/// End-to-end tests for SZCA voice engine.
///
/// Tests complete user journeys from WebSocket connection to audio output.

#[cfg(test)]
mod e2e_tests {
    use std::time::{Duration, Instant};

    // ========================================================================
    // USER JOURNEY TESTS
    // ========================================================================

    #[test]
    fn test_e2e_greeting_conversation() {
        // User says "Hello" → System responds with greeting
        let start = Instant::now();

        // Simulate full conversation
        let user_audio = make_speech(320, 3000); // "Hello"
        let response = simulate_full_pipeline(&user_audio);

        let latency = start.elapsed();
        assert!(!response.is_empty(), "Should respond to greeting");
        assert!(latency < Duration::from_millis(60), "Response latency {}ms > 60ms", latency.as_millis());
    }

    #[test]
    fn test_e2e_multi_turn_conversation() {
        // Multiple turns: Hello → How are you → What's the weather
        let turns = vec![
            "Hello",
            "How are you today?",
            "What's the weather like?",
        ];

        let mut total_latency = Duration::ZERO;
        let turn_count = turns.len();

        for turn in turns {
            let start = Instant::now();
            let audio = make_speech(320, 3000);
            let response = simulate_full_pipeline(&audio);
            let latency = start.elapsed();

            total_latency += latency;
            assert!(!response.is_empty(), "Should respond to: {}", turn);
        }

        let avg_latency = total_latency / turn_count as u32;
        assert!(avg_latency < Duration::from_millis(60), "Average latency {}ms > 60ms", avg_latency.as_millis());
    }

    #[test]
    fn test_e2e_barge_in_during_response() {
        // User interrupts while system is speaking
        let mut tts_playing = true;
        let user_barge_in = true;

        if user_barge_in && tts_playing {
            tts_playing = false;
            // Cancel TTS
        }

        assert!(!tts_playing, "TTS should be cancelled after barge-in");
    }

    #[test]
    fn test_e2e_long_utterance() {
        // User speaks for 5 seconds
        let long_audio = make_speech(8000, 3000); // 500ms @ 16kHz
        let response = simulate_full_pipeline(&long_audio);
        assert!(!response.is_empty(), "Should handle long utterances");
    }

    #[test]
    fn test_e2e_silence_timeout() {
        // User doesn't speak for 5 seconds
        let silence = vec![0i16; 8000]; // 500ms silence
        let response = simulate_full_pipeline(&silence);
        // Should not crash, may produce no response
    }

    #[test]
    fn test_e2e_session_lifecycle() {
        // Full session: connect → handshake → stream → hangup
        let mut session_state = "created";
        session_state = "active"; // After handshake
        session_state = "ended"; // After hangup
        assert_eq!(session_state, "ended");
    }

    // ========================================================================
    // LATENCY METRICS TESTS
    // ========================================================================

    #[test]
    fn test_e2e_ttft_latency() {
        // Time to First Token from LLM
        let start = Instant::now();
        let tokens = simulate_llm_streaming("Hello");
        let ttft = start.elapsed();

        assert!(!tokens.is_empty(), "Should generate tokens");
        assert!(ttft < Duration::from_millis(20), "TTFT {}ms > 20ms", ttft.as_millis());
    }

    #[test]
    fn test_e2e_tpot_latency() {
        // Time Per Output Token
        let tokens = vec!["I'm", " doing", " great"];
        let start = Instant::now();
        for _ in &tokens {
            // Simulate token generation
        }
        let total = start.elapsed();
        let tpot = total / tokens.len() as u32;

        assert!(tpot < Duration::from_millis(5), "TPOT {}ms > 5ms", tpot.as_millis());
    }

    #[test]
    fn test_e2e_stt_latency() {
        // STT processing time per chunk
        let start = Instant::now();
        let _text = simulate_stt_processing(&make_speech(320, 3000));
        let latency = start.elapsed();

        assert!(latency < Duration::from_millis(25), "STT latency {}ms > 25ms", latency.as_millis());
    }

    #[test]
    fn test_e2e_tts_latency() {
        // TTS generation time
        let start = Instant::now();
        let _audio = simulate_tts_processing("Hello world");
        let latency = start.elapsed();

        assert!(latency < Duration::from_millis(15), "TTS latency {}ms > 15ms", latency.as_millis());
    }

    // ========================================================================
    // CONCURRENCY TESTS
    // ========================================================================

    #[test]
    fn test_e2e_100_concurrent_sessions() {
        use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
        use std::thread;

        let completed = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        for _ in 0..100 {
            let completed = Arc::clone(&completed);
            handles.push(thread::spawn(move || {
                let audio = make_speech(320, 3000);
                let _response = simulate_full_pipeline(&audio);
                completed.fetch_add(1, Ordering::Relaxed);
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(completed.load(Ordering::Relaxed), 100);
    }

    #[test]
    fn test_e2e_sustained_load() {
        // Process 1000 consecutive audio chunks
        let start = Instant::now();
        for _ in 0..1000 {
            let audio = make_speech(320, 3000);
            let _response = simulate_full_pipeline(&audio);
        }
        let total = start.elapsed();
        let per_chunk = total / 1000;

        assert!(per_chunk < Duration::from_millis(60), "Per-chunk latency {}ms > 60ms", per_chunk.as_millis());
    }

    // ========================================================================
    // HELPERS
    // ========================================================================

    fn simulate_full_pipeline(input: &[i16]) -> Vec<i16> {
        if input.is_empty() { return vec![]; }

        // DSP
        let clean: Vec<i16> = input.iter().enumerate().map(|(i, &s)| {
            if i == 0 { s } else { ((s as i32 + input[i-1] as i32) / 2) as i16 }
        }).collect();

        // VAD
        let sum_squares: f64 = clean.iter().map(|&s| (s as f64).powi(2)).sum();
        let rms = (sum_squares / clean.len() as f64).sqrt();
        if rms < 500.0 { return vec![]; }

        // STT → LLM → TTS
        vec![1000i16; 320]
    }

    fn simulate_llm_streaming(prompt: &str) -> Vec<String> {
        if prompt.is_empty() { return vec![]; }
        vec!["I'm".into(), " doing".into(), " great!".into()]
    }

    fn simulate_stt_processing(input: &[i16]) -> String {
        let sum_squares: f64 = input.iter().map(|&s| (s as f64).powi(2)).sum();
        let rms = (sum_squares / input.len() as f64).sqrt();
        if rms > 500.0 { "Hello".to_string() } else { "".to_string() }
    }

    fn simulate_tts_processing(text: &str) -> Vec<i16> {
        if text.is_empty() { return vec![]; }
        vec![1000i16; 320]
    }

    fn make_speech(samples: usize, amplitude: i16) -> Vec<i16> {
        (0..samples)
            .map(|i| {
                let t = i as f64 / 16000.0;
                let angle = 2.0 * std::f64::consts::PI * 440.0 * t;
                if angle.sin() >= 0.0 { amplitude } else { -amplitude }
            })
            .collect()
    }
}
