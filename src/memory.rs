/// Guest-memory errors surfaced by translated loads, stores, and instruction
/// fetches. Implementors choose their own backing store and fault details.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryError {
    /// The requested range is not accessible.
    AccessFault { address: u64, length: usize },
}

/// Memory boundary for the JIT. The trait is intentionally independent of any
/// VM or emulator framework; a caller can adapt RAM, a bus, or a test buffer.
pub trait GuestMemory {
    /// Read exactly `out.len()` bytes from guest physical address `address`.
    fn read(&self, address: u64, out: &mut [u8]) -> Result<(), MemoryError>;

    /// Write all bytes to guest physical address `address`.
    fn write(&mut self, address: u64, bytes: &[u8]) -> Result<(), MemoryError>;
}
