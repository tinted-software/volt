//! Instructions the JIT does not compile. It stores the raw word in `Cpu::value`
//! and traps; the machine (or user mode) then runs the word on the host here.
//! Floating point and the wider integer SIMD families live in [`fp`] and [`simd`].

use super::{Cpu, fp, gxf, pauth, simd};
use volt_target::aarch64::decode::{DecodeError, Instruction, decode};

/// Whether [`run`] implements `word`.
pub fn is_supported(word: u32) -> bool {
    fp::is_supported(word)
        || simd::is_supported(word)
        || pauth::is_supported(word)
        || gxf::is_supported(word)
}

/// What the emulator does with one instruction word: the host runs the words [`run`]
/// implements, and the JIT compiles every other instruction the decoder recognises.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dispatch {
    /// A structured instruction for the JIT.
    Guest(Instruction),
    /// A word for [`run`]. It ends a block, the same as a trap does.
    Host(u32),
}

impl Dispatch {
    /// Whether this word ends a translated block; see `Instruction::terminates_with`.
    pub fn terminates_with(self, inline_memory: bool) -> bool {
        match self {
            Self::Guest(instruction) => instruction.terminates_with(inline_memory),
            Self::Host(_) => true,
        }
    }

    /// [`Self::terminates_with`] without inline memory.
    pub fn terminates(self) -> bool {
        self.terminates_with(false)
    }
}

/// Classify `word` for execution. A word [`is_supported`] names is never compiled,
/// even where the decoder has a structured form for it.
pub fn dispatch(word: u32) -> Result<Dispatch, DecodeError> {
    if is_supported(word) {
        Ok(Dispatch::Host(word))
    } else {
        decode(word).map(Dispatch::Guest)
    }
}

/// Run `word` on `cpu`. `word` must satisfy [`is_supported`].
pub fn run(cpu: &mut Cpu, word: u32) {
    if fp::is_supported(word) {
        fp::run(cpu, word)
    } else if simd::is_supported(word) {
        simd::run(cpu, word)
    } else if gxf::is_supported(word) {
        gxf::run(cpu, word)
    } else {
        pauth::run(cpu, word)
    }
}
