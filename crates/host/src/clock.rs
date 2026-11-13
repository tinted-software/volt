use core::time::Duration;
use std::time::Instant;
use volt_platform::{Clock, Timer};

/// Monotonic host clock backed by [`std::time::Instant`].
#[derive(Debug, Clone)]
pub struct StdClock {
    start: Instant,
}

impl Default for StdClock {
    fn default() -> Self {
        Self::new()
    }
}

impl StdClock {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }
}

impl Clock for StdClock {
    fn now(&self) -> Duration {
        self.start.elapsed()
    }
}

impl Timer for StdClock {
    fn sleep(&self, duration: Duration) {
        if !duration.is_zero() {
            std::thread::sleep(duration);
        }
    }
}
