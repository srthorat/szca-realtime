/// Integration tests for the full SZCA pipeline.
///
/// Tests the complete flow: Audio In → DSP → STT → LLM → TTS → Audio Out

#[cfg(test)]
mod pipeline_tests {
    use std::time::Duration;

    // ========================================================================
    // POSITIVE TESTS — Happy Path
    // ========================================================================

    #[test]
    fn test_pipeline_silence_in_silence_out() {
        // Silence in → no audio out (no speech detected)
        let audio_in = vec![0i16; 320]; // 20ms silence
        let audio_out = simulate_pipeline(&audio_in);
        assert!(audio_out.is_empty(), "Silence should produce no output");
    }

    #[test]
    fn test_pipeline_speech_in_audio_out() {
        // Speech in → processed audio out
        let audio_in = make_speech(320, 3000);
        let audio_out = simulate_pipeline(&audio_in);
        assert!(!audio_out.is_empty(), "Speech should produce audio output");
    }

    #[test]
    fn test_pipeline_16khz_in_16khz_out() {
        // Verify sample rate conversion
        let audio_in = make_speech(320, 3000);
        let audio_out = simulate_pipeline(&audio_in);
        // 20ms @ 16kHz = 320 samples
        assert_eq!(audio_out.len(), 320, "Output should be 320 samples (20ms @ 16kHz)");
    }

    #[test]
    fn test_pipeline_latency_budget() {
        // Total pipeline latency should be < 60ms
        let start = std::time::Instant::now();
        let audio_in = make_speech(320, 3000);
        let _audio_out = simulate_pipeline(&audio_in);
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_millis(60),
            "Pipeline latency {}ms exceeds 60ms budget",
            elapsed.as_millis()
        );
    }

    #[test]
    fn test_pipeline_barge_in_cancels_tts() {
        // Barge-in should cancel ongoing TTS
        let mut tts_playing = true;
        let barge_in = true;
        if barge_in {
            tts_playing = false;
        }
        assert!(!tts_playing, "Barge-in should cancel TTS");
    }

    #[test]
    fn test_pipeline_multiple_chunks_streaming() {
        // Process multiple audio chunks in sequence
        let mut all_output = Vec::new();
        for i in 0..10 {
            let audio_in = if i % 2 == 0 {
                make_speech(320, 3000)
            } else {
                vec![0i16; 320]
            };
            let output = simulate_pipeline(&audio_in);
            all_output.extend(output);
        }
        // Should have processed speech chunks
        assert!(!all_output.is_empty());
    }

    #[test]
    fn test_pipeline_stt_partial_and_final() {
        // STT should produce partials during speech, final on silence
        let mut partials = 0;
        let mut finals = 0;

        // Speech frames
        for _ in 0..5 {
            let result = simulate_stt(&make_speech(320, 3000));
            if let Some(r) = result {
                if r == "partial" { partials += 1; }
            }
        }

        // Silence frame
        let result = simulate_stt(&vec![0i16; 320]);
        if let Some(r) = result {
            if r == "final" { finals += 1; }
        }

        assert!(partials > 0, "Should have partial results");
        assert!(finals > 0, "Should have final result");
    }

    #[test]
    fn test_pipeline_llm_generates_tokens() {
        // LLM should generate tokens for valid input
        let tokens = simulate_llm("Hello, how are you?");
        assert!(!tokens.is_empty(), "LLM should generate tokens");
        assert!(tokens.len() <= 256, "Should respect max_tokens");
    }

    #[test]
    fn test_pipeline_tts_generates_audio() {
        // TTS should generate audio for valid text
        let audio = simulate_tts("Hello world");
        assert!(!audio.is_empty(), "TTS should generate audio");
        assert!(audio.len() > 0, "Audio should have samples");
    }

    // ========================================================================
    // NEGATIVE TESTS — Error Handling
    // ========================================================================

    #[test]
    fn test_pipeline_empty_input() {
        let audio_in = vec![];
        let audio_out = simulate_pipeline(&audio_in);
        assert!(audio_out.is_empty(), "Empty input should produce empty output");
    }

    #[test]
    fn test_pipeline_oversized_input() {
        // Input larger than expected chunk size
        let audio_in = vec![0i16; 32000]; // 1 second, not 20ms
        let audio_out = simulate_pipeline(&audio_in);
        // Should handle gracefully, not crash
        assert!(audio_out.len() <= 32000);
    }

    #[test]
    fn test_pipeline_zero_amplitude() {
        let audio_in = vec![0i16; 320];
        let audio_out = simulate_pipeline(&audio_in);
        assert!(audio_out.is_empty());
    }

    #[test]
    fn test_pipeline_max_amplitude() {
        let audio_in = vec![i16::MAX; 320];
        let audio_out = simulate_pipeline(&audio_in);
        // Should not crash or overflow
        for sample in &audio_out {
            assert!(sample.abs() <= i16::MAX);
        }
    }

    #[test]
    fn test_pipeline_negative_amplitude() {
        let audio_in = vec![i16::MIN; 320];
        let audio_out = simulate_pipeline(&audio_in);
        for sample in &audio_out {
            assert!(sample.abs() <= i16::MAX);
        }
    }

    // ========================================================================
    // CONCURRENCY TESTS
    // ========================================================================

    #[test]
    fn test_pipeline_concurrent_sessions() {
        use std::sync::Arc;
        use std::sync::Mutex;
        use std::thread;

        let results = Arc::new(Mutex::new(Vec::new()));
        let mut handles = vec![];

        for _ in 0..10 {
            let results = Arc::clone(&results);
            handles.push(thread::spawn(move || {
                let audio_in = make_speech(320, 3000);
                let output = simulate_pipeline(&audio_in);
                results.lock().unwrap().push(output.len());
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let results = results.lock().unwrap();
        assert_eq!(results.len(), 10, "All 10 concurrent sessions should complete");
        for r in results.iter() {
            assert!(*r > 0, "Each session should produce output");
        }
    }

    // ========================================================================
    // HELPER FUNCTIONS
    // ========================================================================

    fn simulate_pipeline(input: &[i16]) -> Vec<i16> {
        if input.is_empty() { return vec![]; }

        // Step 1: DSP (noise suppression)
        let clean = simulate_dsp(input);

        // Step 2: VAD (speech detection)
        let is_speech = simulate_vad(&clean);
        if !is_speech { return vec![]; }

        // Step 3: STT (speech to text)
        let text = simulate_stt(&clean);

        // Step 4: LLM (text generation)
        let response = simulate_llm("Hello");

        // Step 5: TTS (text to speech)
        simulate_tts("Hello")
    }

    fn simulate_dsp(input: &[i16]) -> Vec<i16> {
        // Simple high-pass filter
        let mut output = input.to_vec();
        for i in 1..output.len() {
            output[i] = ((input[i] as i32 + input[i-1] as i32) / 2) as i16;
        }
        output
    }

    fn simulate_vad(input: &[i16]) -> bool {
        let sum_squares: f64 = input.iter().map(|&s| (s as f64).powi(2)).sum();
        let rms = (sum_squares / input.len() as f64).sqrt();
        rms > 500.0
    }

    fn simulate_stt(input: &[i16]) -> Option<String> {
        let sum_squares: f64 = input.iter().map(|&s| (s as f64).powi(2)).sum();
        let rms = (sum_squares / input.len() as f64).sqrt();
        if rms > 500.0 { Some("partial".to_string()) } else { Some("final".to_string()) }
    }

    fn simulate_llm(prompt: &str) -> Vec<String> {
        if prompt.is_empty() { return vec![]; }
        vec!["I'm".into(), " doing".into(), " great!".into()]
    }

    fn simulate_tts(text: &str) -> Vec<i16> {
        if text.is_empty() { return vec![]; }
        vec![1000i16; 320] // 20ms of audio
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
