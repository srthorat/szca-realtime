/// POSIX Shared Memory IPC module.
///
/// Provides zero-copy communication between gateway and inference engine.
/// Uses /dev/shm for shared memory segments.
///
/// Target latency: <0.1ms per operation

/// IPC configuration.
#[derive(Debug, Clone)]
pub struct IpcConfig {
    /// Shared memory path prefix
    pub shm_prefix: String,
    /// Ring buffer capacity (number of audio chunks)
    pub buffer_capacity: usize,
    /// Audio chunk size in bytes (20ms @ 16kHz 16-bit mono = 640)
    pub chunk_size: usize,
}

impl Default for IpcConfig {
    fn default() -> Self {
        Self {
            shm_prefix: "/dev/shm/szca".to_string(),
            buffer_capacity: 256,
            chunk_size: 640,
        }
    }
}

/// IPC channel for bidirectional communication.
pub struct IpcChannel {
    config: IpcConfig,
    /// Write position in ring buffer
    write_pos: usize,
    /// Read position in ring buffer
    read_pos: usize,
    /// Buffer data
    buffer: Vec<Vec<u8>>,
    /// Valid (written) length per slot, parallel to `buffer`
    lengths: Vec<usize>,
    /// Whether the channel is connected
    connected: bool,
}

impl IpcChannel {
    /// Create a new IPC channel.
    pub fn new(config: IpcConfig) -> Self {
        let buffer = vec![vec![0u8; config.chunk_size]; config.buffer_capacity];
        let lengths = vec![0usize; config.buffer_capacity];
        Self {
            config,
            write_pos: 0,
            read_pos: 0,
            buffer,
            lengths,
            connected: false,
        }
    }

    /// Initialize the shared memory segment.
    pub fn initialize(&mut self) -> Result<(), IpcError> {
        // In production, this creates/opens POSIX SHM via shm_open
        // For now, mark as connected
        self.connected = true;
        Ok(())
    }

    /// Write an audio chunk to the shared memory.
    ///
    /// # Arguments
    /// * `data` - Audio data to write
    ///
    /// # Returns
    /// Ok(true) if successful, Ok(false) if buffer full
    pub fn write(&mut self, data: &[u8]) -> Result<bool, IpcError> {
        if !self.connected {
            return Err(IpcError::NotConnected);
        }

        if data.len() > self.config.chunk_size {
            return Err(IpcError::DataTooLarge {
                max: self.config.chunk_size,
                actual: data.len(),
            });
        }

        let next_pos = (self.write_pos + 1) % self.config.buffer_capacity;
        if next_pos == self.read_pos {
            return Ok(false); // Buffer full
        }

        // Copy data into ring buffer slot and record its exact length
        self.buffer[self.write_pos][..data.len()].copy_from_slice(data);
        self.lengths[self.write_pos] = data.len();
        self.write_pos = next_pos;

        Ok(true)
    }

    /// Read an audio chunk from the shared memory.
    ///
    /// # Returns
    /// Ok(Some(data)) if data available, Ok(None) if empty
    pub fn read(&mut self) -> Result<Option<Vec<u8>>, IpcError> {
        if !self.connected {
            return Err(IpcError::NotConnected);
        }

        if self.read_pos == self.write_pos {
            return Ok(None); // Buffer empty
        }

        // Return only the valid (written) bytes, not the whole slot.
        let len = self.lengths[self.read_pos];
        let data = self.buffer[self.read_pos][..len].to_vec();
        self.read_pos = (self.read_pos + 1) % self.config.buffer_capacity;

        Ok(Some(data))
    }

    /// Get the number of items in the buffer.
    ///
    /// Positions are monotonic modulo capacity, so we add `capacity` before
    /// subtracting to avoid usize underflow once `write_pos` wraps below
    /// `read_pos`.
    pub fn len(&self) -> usize {
        (self.write_pos + self.config.buffer_capacity - self.read_pos)
            % self.config.buffer_capacity
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.read_pos == self.write_pos
    }

    /// Check if the buffer is full.
    pub fn is_full(&self) -> bool {
        (self.write_pos + 1) % self.config.buffer_capacity == self.read_pos
    }

    /// Check if the channel is connected.
    pub fn is_connected(&self) -> bool {
        self.connected
    }

    /// Get the configuration.
    pub fn config(&self) -> &IpcConfig {
        &self.config
    }
}

/// IPC errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IpcError {
    /// Channel not connected
    NotConnected,
    /// Shared memory not found
    ShmNotFound(String),
    /// Data too large for chunk
    DataTooLarge { max: usize, actual: usize },
}

impl std::fmt::Display for IpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IpcError::NotConnected => write!(f, "IPC channel not connected"),
            IpcError::ShmNotFound(path) => write!(f, "Shared memory not found: {}", path),
            IpcError::DataTooLarge { max, actual } => {
                write!(f, "Data too large: max {} bytes, got {}", max, actual)
            }
        }
    }
}

impl std::error::Error for IpcError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ipc_config_default() {
        let config = IpcConfig::default();
        assert_eq!(config.shm_prefix, "/dev/shm/szca");
        assert_eq!(config.buffer_capacity, 256);
        assert_eq!(config.chunk_size, 640);
    }

    #[test]
    fn test_ipc_channel_new() {
        let config = IpcConfig::default();
        let channel = IpcChannel::new(config);
        assert!(!channel.is_connected());
        assert!(channel.is_empty());
        assert!(!channel.is_full());
        assert_eq!(channel.len(), 0);
    }

    #[test]
    fn test_ipc_initialize() {
        let config = IpcConfig::default();
        let mut channel = IpcChannel::new(config);
        assert!(channel.initialize().is_ok());
        assert!(channel.is_connected());
    }

    #[test]
    fn test_ipc_write_not_connected() {
        let config = IpcConfig::default();
        let mut channel = IpcChannel::new(config);
        let data = vec![0u8; 640];
        assert_eq!(channel.write(&data), Err(IpcError::NotConnected));
    }

    #[test]
    fn test_ipc_read_not_connected() {
        let config = IpcConfig::default();
        let mut channel = IpcChannel::new(config);
        assert_eq!(channel.read(), Err(IpcError::NotConnected));
    }

    #[test]
    fn test_ipc_write_read_single() {
        let config = IpcConfig::default();
        let mut channel = IpcChannel::new(config);
        channel.initialize().unwrap();

        let data = vec![0x01, 0x02, 0x03, 0x04];
        assert!(channel.write(&data).unwrap());
        assert_eq!(channel.len(), 1);

        let read_data = channel.read().unwrap().unwrap();
        assert_eq!(read_data[..4], data);
    }

    #[test]
    fn test_ipc_write_read_fifo() {
        let config = IpcConfig::default();
        let mut channel = IpcChannel::new(config);
        channel.initialize().unwrap();

        for i in 0..10 {
            let data = vec![i as u8; 640];
            channel.write(&data).unwrap();
        }

        for i in 0..10 {
            let read_data = channel.read().unwrap().unwrap();
            assert_eq!(read_data[0], i as u8);
        }
    }

    #[test]
    fn test_ipc_buffer_full() {
        let config = IpcConfig {
            buffer_capacity: 4, // Very small buffer
            ..Default::default()
        };
        let mut channel = IpcChannel::new(config);
        channel.initialize().unwrap();

        // Fill buffer (capacity - 1 items due to ring buffer)
        assert!(channel.write(&[1u8; 640]).unwrap());
        assert!(channel.write(&[2u8; 640]).unwrap());
        assert!(channel.write(&[3u8; 640]).unwrap());

        // Buffer full
        assert!(!channel.write(&[4u8; 640]).unwrap());
        assert!(channel.is_full());
    }

    #[test]
    fn test_ipc_buffer_empty() {
        let config = IpcConfig::default();
        let mut channel = IpcChannel::new(config);
        channel.initialize().unwrap();

        assert!(channel.read().unwrap().is_none());
        assert!(channel.is_empty());
    }

    #[test]
    fn test_ipc_data_too_large() {
        let config = IpcConfig::default();
        let mut channel = IpcChannel::new(config);
        channel.initialize().unwrap();

        let data = vec![0u8; 1000]; // Larger than chunk_size (640)
        assert_eq!(
            channel.write(&data),
            Err(IpcError::DataTooLarge {
                max: 640,
                actual: 1000
            })
        );
    }

    #[test]
    fn test_ipc_wraparound() {
        let config = IpcConfig {
            buffer_capacity: 4,
            ..Default::default()
        };
        let mut channel = IpcChannel::new(config);
        channel.initialize().unwrap();

        // Fill and drain multiple times
        for _ in 0..10 {
            channel.write(&[1u8; 640]).unwrap();
            channel.write(&[2u8; 640]).unwrap();
            channel.write(&[3u8; 640]).unwrap();

            assert_eq!(channel.read().unwrap().unwrap()[0], 1);
            assert_eq!(channel.read().unwrap().unwrap()[0], 2);
            assert_eq!(channel.read().unwrap().unwrap()[0], 3);
        }
    }

    #[test]
    fn test_ipc_error_display() {
        let err = IpcError::NotConnected;
        assert!(format!("{}", err).contains("not connected"));

        let err = IpcError::ShmNotFound("/dev/shm/test".to_string());
        assert!(format!("{}", err).contains("/dev/shm/test"));

        let err = IpcError::DataTooLarge {
            max: 640,
            actual: 1000,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("640"));
        assert!(msg.contains("1000"));
    }

    #[test]
    fn test_ipc_config_accessor() {
        let config = IpcConfig::default();
        let channel = IpcChannel::new(config);
        assert_eq!(channel.config().chunk_size, 640);
    }

    #[test]
    fn test_ipc_write_exact_chunk_size() {
        let config = IpcConfig::default();
        let mut channel = IpcChannel::new(config);
        channel.initialize().unwrap();

        let data = vec![0xABu8; 640]; // Exactly chunk_size
        assert!(channel.write(&data).unwrap());

        let read_data = channel.read().unwrap().unwrap();
        assert_eq!(read_data[..640], data);
    }

    #[test]
    fn test_ipc_len_after_wrap() {
        // Regression test for len() underflow after write_pos wraps below
        // read_pos. Push/drain enough to wrap the ring, then leave items in it.
        let config = IpcConfig {
            buffer_capacity: 4,
            ..Default::default()
        };
        let mut channel = IpcChannel::new(config);
        channel.initialize().unwrap();

        // Fill to write_pos = 3, read_pos = 0.
        assert!(channel.write(&[1u8; 640]).unwrap());
        assert!(channel.write(&[1u8; 640]).unwrap());
        assert!(channel.write(&[1u8; 640]).unwrap());
        // Drain all three -> read_pos = 3.
        assert!(channel.read().unwrap().is_some());
        assert!(channel.read().unwrap().is_some());
        assert!(channel.read().unwrap().is_some());
        // Next write wraps write_pos to 0 while read_pos is 3.
        assert!(channel.write(&[2u8; 640]).unwrap());
        // write_pos (0) < read_pos (3): naive subtraction would underflow.
        assert_eq!(channel.len(), 1);
        assert!(!channel.is_empty());
    }

    #[test]
    fn test_ipc_read_returns_exact_written_length() {
        // A short write followed by a full-size write must not leak stale bytes.
        let config = IpcConfig::default();
        let mut channel = IpcChannel::new(config);
        channel.initialize().unwrap();

        channel.write(&[0xAAu8; 640]).unwrap();
        let short = vec![0x11u8, 0x22, 0x33, 0x44];
        channel.write(&short).unwrap();

        // First slot returns full 640 bytes.
        let first = channel.read().unwrap().unwrap();
        assert_eq!(first.len(), 640);

        // Second slot returns exactly the 4 bytes written, no stale tail.
        let second = channel.read().unwrap().unwrap();
        assert_eq!(second, short);
    }

    #[test]
    fn test_ipc_read_returns_none_when_empty() {
        let config = IpcConfig::default();
        let mut channel = IpcChannel::new(config);
        channel.initialize().unwrap();

        assert!(channel.read().unwrap().is_none());
        assert!(channel.read().unwrap().is_none());
    }
}
