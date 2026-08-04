/// Gateway boot configuration from environment variables.

/// Gateway listen and admission settings.
#[derive(Debug, Clone)]
pub struct GatewayConfig {
    /// Bind address
    pub listen_addr: String,
    /// Bind port
    pub port: u16,
    /// Maximum concurrent WebSocket sessions
    pub max_sessions: usize,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            listen_addr: "0.0.0.0".to_string(),
            port: 3000,
            max_sessions: 1000,
        }
    }
}

impl GatewayConfig {
    /// Build from the environment, falling back to [`Default`] per field:
    ///   * `SZCA_LISTEN_ADDR`  — bind address (default `0.0.0.0`)
    ///   * `SZCA_PORT`         — bind port (default 3000)
    ///   * `SZCA_MAX_SESSIONS` — admission-control cap (default 1000)
    ///
    /// Unparseable or empty values fall back to the default with a warning
    /// rather than failing the boot — a typo'd port should not take the service
    /// down, but it must not be silent either.
    pub fn from_env() -> Self {
        let d = Self::default();
        Self {
            listen_addr: env_string("SZCA_LISTEN_ADDR").unwrap_or(d.listen_addr),
            port: env_parsed("SZCA_PORT").unwrap_or(d.port),
            max_sessions: env_parsed::<usize>("SZCA_MAX_SESSIONS")
                .filter(|n| *n > 0)
                .unwrap_or(d.max_sessions),
        }
    }
}

/// Non-empty env var as a `String`.
fn env_string(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty())
}

/// Non-empty env var parsed to `T`; warns and yields `None` on a parse failure.
fn env_parsed<T: std::str::FromStr>(key: &str) -> Option<T> {
    let raw = env_string(key)?;
    match raw.trim().parse::<T>() {
        Ok(v) => Some(v),
        Err(_) => {
            tracing::warn!(key, value = %raw, "invalid value; using default");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gateway_config_default() {
        let config = GatewayConfig::default();
        assert_eq!(config.listen_addr, "0.0.0.0");
        assert_eq!(config.port, 3000);
        assert_eq!(config.max_sessions, 1000);
    }

    #[test]
    fn gateway_config_from_env_reads_overrides_and_rejects_junk() {
        // NOTE: env is process-global. This test owns the SZCA_* keys and clears
        // them on the way out; no other test touches them.
        std::env::set_var("SZCA_LISTEN_ADDR", "127.0.0.1");
        std::env::set_var("SZCA_PORT", "8443");
        std::env::set_var("SZCA_MAX_SESSIONS", "300");
        let c = GatewayConfig::from_env();
        assert_eq!(c.listen_addr, "127.0.0.1");
        assert_eq!(c.port, 8443);
        assert_eq!(c.max_sessions, 300);

        // Junk / empty / zero fall back to defaults instead of failing the boot.
        std::env::set_var("SZCA_PORT", "not-a-port");
        std::env::set_var("SZCA_LISTEN_ADDR", "   ");
        std::env::set_var("SZCA_MAX_SESSIONS", "0");
        let d = GatewayConfig::from_env();
        assert_eq!(d.port, 3000);
        assert_eq!(d.listen_addr, "0.0.0.0");
        assert_eq!(d.max_sessions, 1000, "0 sessions would accept nothing");

        std::env::remove_var("SZCA_LISTEN_ADDR");
        std::env::remove_var("SZCA_PORT");
        std::env::remove_var("SZCA_MAX_SESSIONS");
        let e = GatewayConfig::from_env();
        assert_eq!(e.port, 3000);
        assert_eq!(e.listen_addr, "0.0.0.0");
    }
}
