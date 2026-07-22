/// Lock-free Single Producer Single Consumer (SPSC) ring buffer.
///
/// Design goals:
/// - Zero allocation on hot path
/// - Cache-line aligned to prevent false sharing
/// - Atomic operations only (no mutex, no spinlock)
/// - ~10ns per push/pop operation

use std::sync::atomic::{AtomicUsize, Ordering};

/// Cache-line-aligned atomic counter.
///
/// Aligning to 64 bytes ensures the producer's `head` and the consumer's
/// `tail` live on separate cache lines, preventing false sharing between the
/// two threads that would otherwise thrash the cache line on every update.
#[repr(align(64))]
struct CachePadded(AtomicUsize);

/// Lock-free SPSC ring buffer with pre-allocated storage.
///
/// # SPSC contract
///
/// This buffer is Single-Producer / Single-Consumer. `push` may only ever be
/// called from ONE thread (the producer) and `pop` from ONE thread (the
/// consumer). Calling either from multiple threads concurrently, or sharing a
/// single instance mutably across threads, is undefined behavior. The read-only
/// observer methods (`len`, `is_empty`, `is_full`, `capacity`) may be called
/// from either side but return a point-in-time estimate.
///
/// # Examples
///
/// ```
/// use szca_media_gateway::ring_buffer::SpscRingBuffer;
///
/// let mut buf = SpscRingBuffer::<u8>::new(256);
/// assert!(buf.push(42));
/// assert_eq!(buf.pop(), Some(42));
/// ```
pub struct SpscRingBuffer<T> {
    buffer: Vec<Option<T>>,
    /// Producer index (written only by the producer). Cache-line isolated.
    head: CachePadded,
    /// Consumer index (written only by the consumer). Cache-line isolated.
    tail: CachePadded,
    capacity: usize,
}

impl<T> SpscRingBuffer<T> {
    /// Create a new ring buffer with the given capacity.
    /// Capacity is rounded up to the next power of 2 for efficient masking.
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.next_power_of_two();
        let mut buffer = Vec::with_capacity(capacity);
        buffer.resize_with(capacity, || None);

        Self {
            buffer,
            head: CachePadded(AtomicUsize::new(0)),
            tail: CachePadded(AtomicUsize::new(0)),
            capacity,
        }
    }

    /// Push an item into the buffer.
    /// Returns true if successful, false if buffer is full.
    /// This is the PRODUCER side — must be called from a single thread.
    #[inline]
    pub fn push(&mut self, item: T) -> bool {
        let head = self.head.0.load(Ordering::Relaxed);
        let tail = self.tail.0.load(Ordering::Acquire);
        let next = (head + 1) & (self.capacity - 1);

        if next == tail {
            return false; // Buffer full
        }

        self.buffer[head] = Some(item);
        self.head.0.store(next, Ordering::Release);
        true
    }

    /// Pop an item from the buffer.
    /// Returns Some(item) if available, None if empty.
    /// This is the CONSUMER side — must be called from a single thread.
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        let tail = self.tail.0.load(Ordering::Relaxed);
        let head = self.head.0.load(Ordering::Acquire);

        if tail == head {
            return None; // Buffer empty
        }

        let item = self.buffer[tail].take();
        let next = (tail + 1) & (self.capacity - 1);
        self.tail.0.store(next, Ordering::Release);
        item
    }

    /// Returns the number of items currently in the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        let head = self.head.0.load(Ordering::Acquire);
        let tail = self.tail.0.load(Ordering::Acquire);
        // wrapping_sub then mask: correct for the ring even if head has wrapped
        // below tail in the underlying usize space.
        head.wrapping_sub(tail) & (self.capacity - 1)
    }

    /// Returns true if the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the capacity of the buffer.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns true if the buffer is full.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.len() == self.capacity - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_buffer_has_correct_capacity() {
        let buf = SpscRingBuffer::<i32>::new(100);
        assert_eq!(buf.capacity(), 128); // Rounded to power of 2
    }

    #[test]
    fn test_new_buffer_rounds_to_power_of_two() {
        let buf = SpscRingBuffer::<i32>::new(129);
        assert_eq!(buf.capacity(), 256);
    }

    #[test]
    fn test_new_buffer_is_empty() {
        let buf = SpscRingBuffer::<i32>::new(16);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn test_push_and_pop_single_item() {
        let mut buf = SpscRingBuffer::<i32>::new(16);
        assert!(buf.push(42));
        assert_eq!(buf.pop(), Some(42));
    }

    #[test]
    fn test_push_returns_false_when_full() {
        let mut buf = SpscRingBuffer::<i32>::new(4); // Capacity 4, max 3 items
        assert!(buf.push(1));
        assert!(buf.push(2));
        assert!(buf.push(3));
        assert!(!buf.push(4)); // Full
    }

    #[test]
    fn test_pop_returns_none_when_empty() {
        let mut buf = SpscRingBuffer::<i32>::new(16);
        assert_eq!(buf.pop(), None);
    }

    #[test]
    fn test_fifo_ordering() {
        let mut buf = SpscRingBuffer::<i32>::new(16);
        buf.push(1);
        buf.push(2);
        buf.push(3);

        assert_eq!(buf.pop(), Some(1));
        assert_eq!(buf.pop(), Some(2));
        assert_eq!(buf.pop(), Some(3));
        assert_eq!(buf.pop(), None);
    }

    #[test]
    fn test_len_tracking() {
        let mut buf = SpscRingBuffer::<i32>::new(16);
        assert_eq!(buf.len(), 0);

        buf.push(1);
        assert_eq!(buf.len(), 1);

        buf.push(2);
        assert_eq!(buf.len(), 2);

        buf.pop();
        assert_eq!(buf.len(), 1);

        buf.pop();
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn test_is_full() {
        let mut buf = SpscRingBuffer::<i32>::new(4); // Max 3 items
        assert!(!buf.is_full());

        buf.push(1);
        buf.push(2);
        buf.push(3);
        assert!(buf.is_full());
    }

    #[test]
    fn test_wraparound() {
        let mut buf = SpscRingBuffer::<i32>::new(4); // Capacity 4

        // Fill and drain multiple times to test wraparound
        for _ in 0..10 {
            buf.push(1);
            buf.push(2);
            buf.push(3);
            assert_eq!(buf.pop(), Some(1));
            assert_eq!(buf.pop(), Some(2));
            assert_eq!(buf.pop(), Some(3));
            assert!(buf.is_empty());
        }
    }

    #[test]
    fn test_push_pop_interleaved() {
        let mut buf = SpscRingBuffer::<i32>::new(16);

        for i in 0..100 {
            buf.push(i);
            assert_eq!(buf.pop(), Some(i));
        }
    }

    #[test]
    fn test_buffer_with_zst() {
        let mut buf = SpscRingBuffer::<()>::new(16);
        assert!(buf.push(()));
        assert_eq!(buf.pop(), Some(()));
    }

    #[test]
    fn test_buffer_with_large_struct() {
        let mut buf = SpscRingBuffer::<[u8; 1024]>::new(16);
        let data = [0xABu8; 1024];
        assert!(buf.push(data));
        assert_eq!(buf.pop(), Some(data));
    }

    #[test]
    fn test_pop_after_push_does_not_double_count() {
        let mut buf = SpscRingBuffer::<i32>::new(16);
        buf.push(1);
        buf.push(2);
        buf.pop();
        assert_eq!(buf.len(), 1);
        assert_eq!(buf.pop(), Some(2));
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn test_buffer_stress_push_pop() {
        let mut buf = SpscRingBuffer::<usize>::new(1024);
        let capacity = buf.capacity();

        // Fill completely
        for i in 0..capacity - 1 {
            assert!(buf.push(i));
        }
        assert!(buf.is_full());

        // Drain completely
        for i in 0..capacity - 1 {
            assert_eq!(buf.pop(), Some(i));
        }
        assert!(buf.is_empty());
    }
}
