//! Vulcan native code generation, encoding, linking, and disassembly.
extern crate alloc;

#[cfg(test)]
extern crate std;

#[cfg(feature = "aarch64")]
pub mod aarch64;

#[cfg(feature = "x86_64")]
pub mod x86_64;

pub mod native;
pub mod regalloc;

use core::fmt::Debug;

/// Failure to decode a guest instruction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeError<W> {
    /// Original instruction bits, useful for diagnostics and guest exceptions.
    pub word: W,
}

/// Architecture-specific decoder. `W` is the instruction word representation;
/// variable-length ISAs can use a byte slice in their own decoder implementation.
pub trait Decoder {
    /// Encoded instruction word.
    type Word: Copy + Debug;
    /// Decoded architecture-specific instruction.
    type Instruction: Debug;

    /// Decode one instruction or reject the encoding.
    fn decode(&self, word: Self::Word) -> Result<Self::Instruction, DecodeError<Self::Word>>;
}
