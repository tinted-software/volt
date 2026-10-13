//! SIMD structure loads and stores: `LD1`..`LD4`, `ST1`..`ST4`, their single-lane
//! forms and the replicating `LD1R`..`LD4R`.
//!
//! Decode describes one in a [`StructDesc`]. The emulator copies the descriptor into
//! its vCPU state and runs the access against its own memory, one element at a time.

/// Which elements of the registers a structure access touches.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum Shape {
    /// Whole vectors: `ld1 {v0.4s - v3.4s}` and the interleaved `ld2`..`ld4`.
    #[default]
    Multiple,
    /// One element of each register: `ld1 {v0.s}[1]`. Other lanes are kept.
    Lane,
    /// One element broadcast to every lane: `ld1r`. Loads only.
    Replicate,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct StructDesc {
    pub store: bool,
    pub shape: Shape,
    /// Registers touched, starting at `rt` and wrapping from v31 to v0.
    pub registers: u8,
    /// `ld2`..`ld4`: consecutive memory elements alternate between the registers.
    /// Otherwise (`ld1`) each register is a contiguous block.
    pub interleaved: bool,
    /// Element size in bytes: 1, 2, 4 or 8.
    pub esize: u8,
    /// Element index of a [`Shape::Lane`] access.
    pub lane: u8,
    /// A 128-bit vector (otherwise 64-bit, with the upper half zeroed on loads).
    pub q: bool,
    pub rt: u8,
}

impl StructDesc {
    pub fn vector_bytes(&self) -> u8 {
        if self.q { 16 } else { 8 }
    }
    /// Bytes of memory the access spans; the post-index immediate advances by this.
    pub fn bytes(&self) -> u64 {
        let registers = u64::from(self.registers);
        match self.shape {
            Shape::Multiple => registers * u64::from(self.vector_bytes()),
            Shape::Lane | Shape::Replicate => registers * u64::from(self.esize),
        }
    }
}
