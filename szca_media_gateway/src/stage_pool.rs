/// Generic inference **stage pool**: the concurrency foundation for serving
/// STT / LLM / TTS as independent, composable, horizontally-scaled services.
///
/// # Why this exists
///
/// ONNX `Session::run` takes `&mut self` — one model instance runs one inference
/// at a time. Concurrency therefore comes from a **pool of `N` model replicas**
/// fed by a shared work queue, NOT from sharing a single model. Each replica
/// owns its own loaded model on a dedicated OS thread and pulls jobs off the
/// queue; `N` concurrent jobs run on `N` replicas.
///
/// Capacity is **fungible across stages**: `N` is configured per stage
/// independently (`STT_REPLICAS`, `LLM_REPLICAS`, `TTS_REPLICAS`), so one
/// hardware budget can be reallocated between full speech-to-speech, STT-only,
/// LLM-only, or TTS-only deployments by config alone.
///
/// # Shape
///
/// ```text
///   submit(input, cancel) ─► [ bounded MPMC queue ] ─► replica thread 0 ─┐
///                                                    ─► replica thread 1 ─┤─► deltas
///                                                    ─► …                 │   stream back
///                                                    ─► replica thread N-1┘   to caller
/// ```
///
/// A [`Replica`] processes one job, emitting incremental **deltas** (transcript
/// text, LLM tokens, or PCM audio chunks) through a callback as it works; the
/// pool forwards those deltas to the caller over a channel and finishes with the
/// replica's final output. Long jobs cooperatively honor a shared cancel flag
/// (barge-in), exactly as the underlying stage impls already do.
///
/// This module is model-agnostic and fully unit-testable with a mock replica;
/// the real STT/LLM/TTS wrappers live in their own modules (Phase 2).

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use hdrhistogram::Histogram;
use serde::Serialize;
use tokio::sync::mpsc;

/// Per-pool latency histogram snapshot. Returned by [`StagePool::latency_snapshot`].
#[derive(Clone, Serialize, Debug)]
#[serde(rename_all = "snake_case")]
pub struct LatencySnapshot {
    pub count: u64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub mean_ms: f64,
    pub p50_ms: f64,
    pub p90_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
}

/// Bound on how many jobs may sit queued (beyond those actively running) before
/// [`StagePool::try_submit`] rejects with [`SubmitError::Full`]. Keeps a
/// saturated stage from growing an unbounded backlog; callers surface this as a
/// `503`/busy signal rather than latency that climbs forever.
///
/// Default 64 works for dev (1-2 sessions). For prod at 1k+ concurrent sessions,
/// set the `SZCA_QUEUE_BACKLOG` env var to 1024 or higher — without it, a wave
/// of concurrent turns floods the queue and returns 503 to every excess caller.
/// Shared by pool construction and the HTTP pre-submit busy check in `api_routes`.
pub const DEFAULT_QUEUE_BACKLOG: usize = 64;

/// Read the queue backlog from the `SZCA_QUEUE_BACKLOG` env var, falling back
/// to [`DEFAULT_QUEUE_BACKLOG`].
///
/// Used by [`StagePool::build`] and by HTTP `acquire_pool` so both reject at
/// the same depth — a hardcoded 64 in one place and env 1024 in the other
/// caused false HTTP 503s while the pool still had capacity.
pub fn queue_backlog_from_env() -> usize {
    parse_queue_backlog(std::env::var("SZCA_QUEUE_BACKLOG").ok().as_deref())
}

/// Parse a backlog value: unset / empty / zero / non-numeric → default.
pub fn parse_queue_backlog(raw: Option<&str>) -> usize {
    raw.and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_QUEUE_BACKLOG)
}

/// A single model instance that processes one job at a time.
///
/// Implementors own their loaded model (e.g. an ONNX `Session`). The pool calls
/// [`Replica::process`] on a dedicated worker thread; `&mut self` is sound
/// because each replica is confined to exactly one thread for its whole life.
///
/// `process` should:
///   * emit incremental results by calling `emit(delta)` as they are produced,
///   * check `cancel` cooperatively and return early when it is set,
///   * return the final aggregate output (which may repeat what was emitted, as
///     the LLM stage does — callers decide whether to use deltas, final, or both).
pub trait Replica: Send + 'static {
    /// Per-job input handed to the replica.
    type Input: Send + 'static;
    /// Incremental result streamed as the job runs.
    type Delta: Send + 'static;
    /// Final aggregate result returned when the job completes.
    type Output: Send + 'static;

    /// Process one job. `emit` streams deltas; `cancel` requests early stop.
    fn process(
        &mut self,
        input: Self::Input,
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(Self::Delta),
    ) -> Self::Output;
}

/// One unit of work plus its live streaming/cancel channels.
struct Job<R: Replica> {
    input: R::Input,
    cancel: Arc<AtomicBool>,
    /// Deltas are forwarded here as the replica emits them.
    delta_tx: mpsc::UnboundedSender<R::Delta>,
    /// The final output (or `None` if the job was dropped/cancelled before a
    /// replica picked it up) is delivered here exactly once.
    done_tx: tokio::sync::oneshot::Sender<R::Output>,
    /// Shared counter decremented when the worker picks up this job.
    _queued: Arc<AtomicUsize>,
}

/// A handle to an in-flight job: a stream of deltas + a future for the final
/// output. Returned by [`StagePool::submit`].
pub struct JobHandle<R: Replica> {
    /// Receiver for streamed deltas; yields `None` once the job finishes.
    pub deltas: mpsc::UnboundedReceiver<R::Delta>,
    /// Resolves to the final output, or `Err` if the worker died mid-job.
    pub done: tokio::sync::oneshot::Receiver<R::Output>,
    /// Shared cancel flag for this job (set to request barge-in/early stop).
    pub cancel: Arc<AtomicBool>,
}

/// Why a non-blocking submit was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitError {
    /// The queue backlog is at capacity; retry later (surface as 503/busy).
    Full,
    /// All replica threads have shut down; the pool is unusable.
    Closed,
}

impl std::fmt::Display for SubmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SubmitError::Full => write!(f, "stage pool queue is full"),
            SubmitError::Closed => write!(f, "stage pool is closed"),
        }
    }
}

impl std::error::Error for SubmitError {}

/// A pool of `N` identical model replicas sharing one bounded work queue.
///
/// Cloning a `StagePool` is cheap (it shares the same queue + replicas); pass
/// clones to every session/handler that needs this stage.
pub struct StagePool<R: Replica> {
    job_tx: crossbeam_channel::Sender<Job<R>>,
    replicas: usize,
    /// Number of jobs currently queued (not yet picked up by a worker).
    queued: Arc<AtomicUsize>,
    /// Sharded latency histograms (milliseconds). Each worker exclusively records its
    /// per-job wall-clock duration into its own slot to avoid Mutex contention.
    latencies: Arc<Vec<Arc<Mutex<Histogram<u64>>>>>,
}

impl<R: Replica> Clone for StagePool<R> {
    fn clone(&self) -> Self {
        Self {
            job_tx: self.job_tx.clone(),
            replicas: self.replicas,
            queued: Arc::clone(&self.queued),
            latencies: Arc::clone(&self.latencies),
        }
    }
}

impl<R: Replica> StagePool<R> {
    /// Build a pool by constructing `replicas` model instances via `make` and
    /// spawning one worker thread for each.
    ///
    /// `make` is called once per replica (so each thread owns its own model);
    /// it returns `Err` if a replica fails to load, aborting pool construction.
    /// `replicas` must be >= 1.
    pub fn build<F>(name: &str, replicas: usize, mut make: F) -> Result<Self, String>
    where
        F: FnMut(usize) -> Result<R, String>,
    {
        let backlog = queue_backlog_from_env();
        Self::build_with_backlog(name, replicas, backlog, &mut make)
    }

    /// Like [`build`](Self::build) with an explicit queue backlog bound.
    pub fn build_with_backlog<F>(
        name: &str,
        replicas: usize,
        backlog: usize,
        make: &mut F,
    ) -> Result<Self, String>
    where
        F: FnMut(usize) -> Result<R, String>,
    {
        if replicas == 0 {
            return Err(format!("stage '{name}': replicas must be >= 1"));
        }
        // Bounded so a saturated stage applies backpressure instead of growing
        // an unbounded backlog. Capacity = in-flight (one per replica) + queued.
        let (job_tx, job_rx) = crossbeam_channel::bounded::<Job<R>>(replicas + backlog);

        // Sharded latency histograms: 1 ms to 60 s, 3 significant figures.
        let mut latencies = Vec::with_capacity(replicas);
        for _ in 0..replicas {
            latencies.push(Arc::new(Mutex::new(
                Histogram::new_with_bounds(1, 60_000, 3)
                    .map_err(|e| format!("stage '{name}': histogram init: {e}"))?,
            )));
        }
        let latencies = Arc::new(latencies);

        for idx in 0..replicas {
            let replica = make(idx).map_err(|e| format!("stage '{name}' replica {idx}: {e}"))?;
            let rx = job_rx.clone();
            let label = format!("{name}-{idx}");
            let h = Arc::clone(&latencies[idx]);
            std::thread::Builder::new()
                .name(label.clone())
                .spawn(move || worker_loop(replica, rx, h))
                .map_err(|e| format!("stage '{name}': spawn replica {idx}: {e}"))?;
        }
        // Drop the local receiver so the channel closes once all workers exit.
        drop(job_rx);

        tracing::info!(stage = name, replicas, backlog, "stage pool ready");
        Ok(Self { job_tx, replicas, queued: Arc::new(AtomicUsize::new(0)), latencies })
    }

    /// Number of replicas (max concurrent jobs) in this pool.
    pub fn replicas(&self) -> usize {
        self.replicas
    }

    /// Number of replicas — alias used by the pool health endpoint.
    pub fn replica_count(&self) -> usize {
        self.replicas
    }

    /// Approximate number of jobs sitting in the queue (not yet picked up
    /// by a worker). Updated atomically on each submit/completion.
    pub fn queue_depth(&self) -> usize {
        self.queued.load(Ordering::Relaxed)
    }

    /// Approximate number of jobs currently queued (not yet picked up).
    /// Useful for saturation metrics / backpressure decisions.
    pub fn queue_len(&self) -> usize {
        self.job_tx.len()
    }

    /// Snapshot of per-job latency statistics from the shared histogram.
    ///
    /// Returns `None` if the pool has processed fewer than 2 jobs (not enough
    /// data for meaningful percentiles) or the mutex is poisoned.
    pub fn latency_snapshot(&self) -> Option<LatencySnapshot> {
        let mut agg = Histogram::<u64>::new_with_bounds(1, 60_000, 3).ok()?;
        
        for h_lock in self.latencies.iter() {
            if let Ok(h) = h_lock.lock() {
                let _ = agg.add(&*h);
            }
        }

        if agg.len() < 2 {
            return None;
        }
        Some(LatencySnapshot {
            count: agg.len(),
            min_ms: agg.min() as f64,
            max_ms: agg.max() as f64,
            mean_ms: agg.mean(),
            p50_ms: agg.value_at_percentile(50.0) as f64,
            p90_ms: agg.value_at_percentile(90.0) as f64,
            p95_ms: agg.value_at_percentile(95.0) as f64,
            p99_ms: agg.value_at_percentile(99.0) as f64,
        })
    }

    /// Submit a job, blocking only if the bounded queue is momentarily full.
    /// Returns a [`JobHandle`] to stream deltas and await the final output.
    ///
    /// Prefer [`try_submit`](Self::try_submit) on request paths where you want
    /// to reject-fast with a busy signal instead of waiting.
    pub fn submit(&self, input: R::Input) -> Result<JobHandle<R>, SubmitError> {
        let (job, handle) = self.make_job(input);
        self.job_tx.send(job).map_err(|_| SubmitError::Closed)?;
        Ok(handle)
    }

    /// Non-blocking submit: reject immediately with [`SubmitError::Full`] when
    /// the queue backlog is at capacity, or [`SubmitError::Closed`] if the pool
    /// has shut down.
    pub fn try_submit(&self, input: R::Input) -> Result<JobHandle<R>, SubmitError> {
        let (job, handle) = self.make_job(input);
        match self.job_tx.try_send(job) {
            Ok(()) => Ok(handle),
            Err(crossbeam_channel::TrySendError::Full(_)) => Err(SubmitError::Full),
            Err(crossbeam_channel::TrySendError::Disconnected(_)) => Err(SubmitError::Closed),
        }
    }

    /// Submit a job with the caller's own cancel flag. The replica will check
    /// **this** `AtomicBool` cooperatively, so setting it to `true` from any
    /// thread immediately requests cancellation — no polling or copying needed.
    pub fn submit_with_cancel(
        &self,
        input: R::Input,
        cancel: Arc<AtomicBool>,
    ) -> Result<JobHandle<R>, SubmitError> {
        let (job, handle) = self.make_job_with_cancel(input, cancel);
        self.job_tx.send(job).map_err(|_| SubmitError::Closed)?;
        Ok(handle)
    }

    /// Non-blocking submit with the caller's own cancel flag.
    pub fn try_submit_with_cancel(
        &self,
        input: R::Input,
        cancel: Arc<AtomicBool>,
    ) -> Result<JobHandle<R>, SubmitError> {
        let (job, handle) = self.make_job_with_cancel(input, cancel);
        match self.job_tx.try_send(job) {
            Ok(()) => Ok(handle),
            Err(crossbeam_channel::TrySendError::Full(_)) => Err(SubmitError::Full),
            Err(crossbeam_channel::TrySendError::Disconnected(_)) => Err(SubmitError::Closed),
        }
    }

    fn make_job(&self, input: R::Input) -> (Job<R>, JobHandle<R>) {
        let cancel = Arc::new(AtomicBool::new(false));
        self.make_job_with_cancel(input, cancel)
    }

    fn make_job_with_cancel(
        &self,
        input: R::Input,
        cancel: Arc<AtomicBool>,
    ) -> (Job<R>, JobHandle<R>) {
        let (delta_tx, deltas) = mpsc::unbounded_channel();
        let (done_tx, done) = tokio::sync::oneshot::channel();
        let queued = Arc::clone(&self.queued);
        let job = Job {
            input,
            cancel: Arc::clone(&cancel),
            delta_tx,
            done_tx,
            _queued: queued,
        };
        let handle = JobHandle {
            deltas,
            done,
            cancel,
        };
        // Increment queue depth counter.
        self.queued.fetch_add(1, Ordering::Relaxed);
        (job, handle)
    }
}

/// The per-replica worker: pull jobs off the shared queue until it closes,
/// running each to completion on this dedicated thread. Records per-job
/// wall-clock latency into the shared histogram for pool-health telemetry.
fn worker_loop<R: Replica>(
    mut replica: R,
    rx: crossbeam_channel::Receiver<Job<R>>,
    latency: Arc<Mutex<Histogram<u64>>>,
) {
    while let Ok(job) = rx.recv() {
        // If the caller already hung up (dropped the handle), skip the work.
        if job.done_tx.is_closed() {
            continue;
        }
        let Job {
            input,
            cancel,
            delta_tx,
            done_tx,
            _queued,
        } = job;

        // Decrement queue depth counter (job is no longer queued, it's running).
        _queued.fetch_sub(1, Ordering::Relaxed);

        let start = Instant::now();
        let mut emit = |delta: R::Delta| {
            // Ignore send errors: the caller may have dropped the delta receiver
            // (e.g. barge-in). The cancel flag is the authoritative stop signal.
            let _ = delta_tx.send(delta);
        };
        let output = replica.process(input, &cancel, &mut emit);
        // Record wall-clock process latency (milliseconds).
        let elapsed_ms = start.elapsed().as_millis() as u64;
        if let Ok(mut h) = latency.lock() {
            let _ = h.record(elapsed_ms);
        }
        // Best-effort: caller may have gone away; that's fine.
        let _ = done_tx.send(output);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    /// A mock replica that echoes its input as N deltas then a final summary,
    /// sleeping a bit per delta so concurrency is observable.
    struct EchoReplica {
        id: usize,
    }

    impl Replica for EchoReplica {
        type Input = (u32, Duration); // (count, per-delta delay)
        type Delta = String;
        type Output = String;

        fn process(
            &mut self,
            input: Self::Input,
            cancel: &AtomicBool,
            emit: &mut dyn FnMut(String),
        ) -> String {
            let (count, delay) = input;
            let mut n = 0;
            for i in 0..count {
                if cancel.load(Ordering::Relaxed) {
                    break;
                }
                std::thread::sleep(delay);
                emit(format!("r{}:d{}", self.id, i));
                n += 1;
            }
            format!("done:{n}")
        }
    }

    #[tokio::test]
    async fn single_replica_streams_deltas_then_final() {
        let pool = StagePool::build("echo", 1, |id| Ok(EchoReplica { id })).unwrap();
        let mut h = pool.submit((3, Duration::from_millis(1))).unwrap();

        let mut deltas = Vec::new();
        while let Some(d) = h.deltas.recv().await {
            deltas.push(d);
        }
        assert_eq!(deltas.len(), 3);
        assert_eq!(h.done.await.unwrap(), "done:3");
    }

    #[tokio::test]
    async fn multiple_replicas_run_concurrently() {
        // 4 replicas, 4 jobs each taking ~40ms serial. If truly concurrent the
        // wall-clock is ~one job, not four.
        let pool = StagePool::build("echo", 4, |id| Ok(EchoReplica { id })).unwrap();
        let start = std::time::Instant::now();

        let handles: Vec<_> = (0..4)
            .map(|_| pool.submit((4, Duration::from_millis(10))).unwrap())
            .collect();

        for mut h in handles {
            while h.deltas.recv().await.is_some() {}
            assert_eq!(h.done.await.unwrap(), "done:4");
        }
        let elapsed = start.elapsed();
        // 4 jobs × 4 deltas × 10ms = 160ms serial; concurrent should be well
        // under 2× a single job (80ms). Generous bound to avoid CI flakiness.
        assert!(
            elapsed < Duration::from_millis(120),
            "expected concurrent execution, took {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn cancel_stops_job_early() {
        let pool = StagePool::build("echo", 1, |id| Ok(EchoReplica { id })).unwrap();
        let mut h = pool.submit((1000, Duration::from_millis(5))).unwrap();

        // Let a couple of deltas through, then cancel.
        let _ = h.deltas.recv().await;
        h.cancel.store(true, Ordering::Relaxed);

        // Drain remaining; job must finish quickly rather than doing all 1000.
        let start = std::time::Instant::now();
        while h.deltas.recv().await.is_some() {}
        let final_out = h.done.await.unwrap();
        assert!(final_out.starts_with("done:"));
        assert!(
            start.elapsed() < Duration::from_millis(200),
            "cancel should stop the job promptly"
        );
    }

    #[tokio::test]
    async fn try_submit_reports_full_then_recovers() {
        // 1 replica, backlog 1 → capacity 2 in the channel. Saturate it with
        // long jobs and assert we eventually get a Full rejection.
        let pool = StagePool::build_with_backlog("echo", 1, 1, &mut |id| Ok(EchoReplica { id }))
            .unwrap();

        let mut saw_full = false;
        let mut handles = Vec::new();
        for _ in 0..10 {
            match pool.try_submit((50, Duration::from_millis(5))) {
                Ok(h) => handles.push(h),
                Err(SubmitError::Full) => {
                    saw_full = true;
                    break;
                }
                Err(SubmitError::Closed) => panic!("pool unexpectedly closed"),
            }
        }
        assert!(saw_full, "expected the bounded queue to reject when full");

        // Drain everything so the pool recovers.
        for mut h in handles {
            while h.deltas.recv().await.is_some() {}
            let _ = h.done.await;
        }
    }

    #[test]
    fn build_rejects_zero_replicas() {
        let r = StagePool::build("echo", 0, |id| Ok(EchoReplica { id }));
        assert!(r.is_err());
    }

    #[test]
    fn parse_queue_backlog_uses_env_or_default() {
        assert_eq!(parse_queue_backlog(None), DEFAULT_QUEUE_BACKLOG);
        assert_eq!(parse_queue_backlog(Some("")), DEFAULT_QUEUE_BACKLOG);
        assert_eq!(parse_queue_backlog(Some("0")), DEFAULT_QUEUE_BACKLOG);
        assert_eq!(parse_queue_backlog(Some("nope")), DEFAULT_QUEUE_BACKLOG);
        assert_eq!(parse_queue_backlog(Some("1024")), 1024);
    }

    #[tokio::test]
    async fn replicas_accessor_reports_count() {
        let pool = StagePool::build("echo", 3, |id| Ok(EchoReplica { id })).unwrap();
        assert_eq!(pool.replicas(), 3);
    }
}
