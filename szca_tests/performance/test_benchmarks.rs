//! Performance benchmark HARNESS tests for SZCA.
//!
//! IMPORTANT — SCOPE OF THIS MODULE:
//! These benchmarks exercise the *test harness / scaffolding* using the
//! lightweight in-test stub functions in the `HELPERS` section
//! (`simulate_pipeline`, `simulate_stt`, `simulate_llm`, `simulate_tts`).
//! They do NOT measure the real STT/LLM/TTS models or the deployed
//! pipeline. The numbers here validate that the benchmark scaffolding
//! runs and stays within sanity bounds; real model performance is
//! measured separately against the production services.

#[cfg(test)]
mod benchmark_tests {
    use std::time::{Duration, Instant};

    // ========================================================================
    // THROUGHPUT TESTS
    // ========================================================================

    #[test]
    fn bench_pipeline_throughput() {
        // Measure max audio chunks per second
        let chunk_size = 320; // 20ms @ 16kHz
        let iterations = 10000;

        let start = Instant::now();
        for _ in 0..iterations {
            let audio = make_speech(chunk_size, 3000);
            let _output = simulate_pipeline(&audio);
        }
        let elapsed = start.elapsed();
        let chunks_per_sec = iterations as f64 / elapsed.as_secs_f64();
        let throughput_ms = elapsed.as_millis() as f64 / iterations as f64;

        println!("Pipeline throughput: {:.0} chunks/sec ({:.2}ms/chunk)", chunks_per_sec, throughput_ms);
        assert!(chunks_per_sec > 100.0, "Should process >100 chunks/sec");
    }

    #[test]
    fn bench_stt_throughput() {
        let iterations = 1000;
        let audio = make_speech(320, 3000);

        let start = Instant::now();
        for _ in 0..iterations {
            let _text = simulate_stt(&audio);
        }
        let elapsed = start.elapsed();
        let chunks_per_sec = iterations as f64 / elapsed.as_secs_f64();

        println!("STT throughput: {:.0} chunks/sec", chunks_per_sec);
        assert!(chunks_per_sec > 50.0, "STT should process >50 chunks/sec");
    }

    #[test]
    fn bench_llm_throughput() {
        let iterations = 100;

        let start = Instant::now();
        for _ in 0..iterations {
            let _tokens = simulate_llm("Hello");
        }
        let elapsed = start.elapsed();
        let requests_per_sec = iterations as f64 / elapsed.as_secs_f64();

        println!("LLM throughput: {:.1} requests/sec", requests_per_sec);
        assert!(requests_per_sec > 10.0, "LLM should handle >10 requests/sec");
    }

    #[test]
    fn bench_tts_throughput() {
        let iterations = 100;

        let start = Instant::now();
        for _ in 0..iterations {
            let _audio = simulate_tts("Hello world");
        }
        let elapsed = start.elapsed();
        let chunks_per_sec = iterations as f64 / elapsed.as_secs_f64();

        println!("TTS throughput: {:.1} chunks/sec", chunks_per_sec);
        assert!(chunks_per_sec > 10.0, "TTS should generate >10 chunks/sec");
    }

    // ========================================================================
    // LATENCY TESTS
    // ========================================================================

    #[test]
    fn bench_pipeline_latency_p50() {
        let iterations = 1000;
        let mut latencies = Vec::new();

        for _ in 0..iterations {
            let audio = make_speech(320, 3000);
            let start = Instant::now();
            let _output = simulate_pipeline(&audio);
            latencies.push(start.elapsed());
        }

        latencies.sort();
        let p50 = latencies[iterations / 2];
        let p95 = latencies[(iterations as f64 * 0.95) as usize];
        let p99 = latencies[(iterations as f64 * 0.99) as usize];

        println!("Pipeline latency: p50={:?}, p95={:?}, p99={:?}", p50, p95, p99);
        assert!(p50 < Duration::from_millis(10), "p50 latency {}ms > 10ms", p50.as_millis());
        assert!(p95 < Duration::from_millis(30), "p95 latency {}ms > 30ms", p95.as_millis());
        assert!(p99 < Duration::from_millis(50), "p99 latency {}ms > 50ms", p99.as_millis());
    }

    #[test]
    fn bench_stt_latency_p50() {
        let iterations = 1000;
        let audio = make_speech(320, 3000);
        let mut latencies = Vec::new();

        for _ in 0..iterations {
            let start = Instant::now();
            let _text = simulate_stt(&audio);
            latencies.push(start.elapsed());
        }

        latencies.sort();
        let p50 = latencies[iterations / 2];

        println!("STT latency p50: {:?}", p50);
        assert!(p50 < Duration::from_millis(25), "STT p50 {}ms > 25ms", p50.as_millis());
    }

    #[test]
    fn bench_tts_latency_p50() {
        let iterations = 1000;
        let mut latencies = Vec::new();

        for _ in 0..iterations {
            let start = Instant::now();
            let _audio = simulate_tts("Hello world");
            latencies.push(start.elapsed());
        }

        latencies.sort();
        let p50 = latencies[iterations / 2];

        println!("TTS latency p50: {:?}", p50);
        assert!(p50 < Duration::from_millis(15), "TTS p50 {}ms > 15ms", p50.as_millis());
    }

    // ========================================================================
    // CONCURRENCY TESTS
    // ========================================================================

    #[test]
    fn bench_concurrent_500_sessions() {
        use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
        use std::thread;

        let completed = Arc::new(AtomicUsize::new(0));
        let start = Instant::now();
        let mut handles = vec![];

        for _ in 0..500 {
            let completed = Arc::clone(&completed);
            handles.push(thread::spawn(move || {
                let audio = make_speech(320, 3000);
                let _output = simulate_pipeline(&audio);
                completed.fetch_add(1, Ordering::Relaxed);
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let elapsed = start.elapsed();
        let throughput = 500.0 / elapsed.as_secs_f64();

        println!("500 concurrent sessions: {:?} ({:.0} sessions/sec)", elapsed, throughput);
        assert_eq!(completed.load(Ordering::Relaxed), 500);
        assert!(throughput > 100.0, "Should handle >100 sessions/sec");
    }

    #[test]
    fn bench_concurrent_1000_sessions() {
        use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
        use std::thread;

        let completed = Arc::new(AtomicUsize::new(0));
        let start = Instant::now();
        let mut handles = vec![];

        for _ in 0..1000 {
            let completed = Arc::clone(&completed);
            handles.push(thread::spawn(move || {
                let audio = make_speech(320, 3200);
                let _output = simulate_pipeline(&audio);
                completed.fetch_add(1, Ordering::Relaxed);
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let elapsed = start.elapsed();
        println!("1000 concurrent sessions: {:?}", elapsed);
        assert_eq!(completed.load(Ordering::Relaxed), 1000);
    }

    // ========================================================================
    // HELPERS
    // ========================================================================

    fn simulate_pipeline(input: &[i16]) -> Vec<i16> {
        if input.is_empty() { return vec![]; }
        let clean: Vec<i16> = input.to_vec();
        let sum_squares: f64 = clean.iter().map(|&s| (s as f64).powi(2)).sum();
        let rms = (sum_squares / clean.len() as f64).sqrt();
        if rms < 500.0 { return vec![]; }
        vec![1000i16; 320]
    }

    fn simulate_stt(input: &[i16]) -> String {
        let sum_squares: f64 = input.iter().map(|&s| (s as f64).powi(2)).sum();
        let rms = (sum_squares / input.len() as f64).sqrt();
        if rms > 500.0 { "Hello".to_string() } else { "".to_string() }
    }

    fn simulate_llm(prompt: &str) -> Vec<String> {
        if prompt.is_empty() { return vec![]; }
        vec!["I'm".into(), " doing".into(), " great!".into()]
    }

    fn simulate_tts(text: &str) -> Vec<i16> {
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
