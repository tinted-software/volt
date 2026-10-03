//! AArch64 instruction selection: IR lowering surface (stub).
//!
//! Full lowering of `volt-ir` functions to A64 machine words lives here once
//! the `volt-ir` crate exists. Until then this module only defines the shared
//! data vocabulary (`Fixup`, `Reloc`, `Compiled`, …) consumed by the
//! peephole, object, link, and JIT stages, plus stub entry points that fail
//! with [`Error::Unsupported`].

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

/// Instruction selection failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    /// The IR construct has no A64 lowering yet (or no IR crate is linked).
    Unsupported,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Unsupported => write!(f, "unsupported IR construct for aarch64"),
        }
    }
}

impl core::error::Error for Error {}

/// A pending branch fixup: the word at `at` must be patched to reach the
/// block numbered `target` once layout is known.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Fixup {
    pub at: usize,
    pub target: u32,
}

/// What an emitted relocation patches: a `bl` call target (the default), one
/// half of a direct `global_addr` `adrp`+`add` pair, or one half of a
/// GOT-indirect `global_addr` `adrp`+`ldr` pair.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RelocKind {
    Call,
    AdrpPg,
    AddPgOff,
    GotPg,
    GotLo12,
}

/// A relocation: the word index of an instruction whose immediate names a
/// symbol, patched once the symbol's address is known.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Reloc {
    pub offset: usize,
    pub symbol: String,
    pub kind: RelocKind,
}

impl Reloc {
    pub fn call(offset: usize, symbol: &str) -> Reloc {
        Reloc {
            offset,
            symbol: symbol.to_string(),
            kind: RelocKind::Call,
        }
    }
}

/// One row of the source-line table: the byte offset (from the function
/// start) where a new source line's code begins, and that line.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LineEntry {
    pub offset: u32,
    pub line: u32,
}

/// Machine words plus the relocations for the linker to resolve.
#[derive(Clone, Debug)]
pub struct Compiled {
    pub code: Vec<u32>,
    pub relocs: Vec<Reloc>,
    pub lines: Vec<LineEntry>,
}

/// Calling convention for argument passing.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Abi {
    /// Standard AAPCS64 layout (default; every non-JIT caller, object path).
    #[default]
    Aapcs64,
    /// Apple-packed stack-argument layout for JIT on a Darwin host.
    Apple,
}

/// Machine-level tuning knobs for instruction selection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelCaps {
    /// Calling convention for argument passing.
    pub abi: Abi,
    /// Loop-header alignment in bytes (0 disables it).
    pub fetch_align: u16,
    /// Use native half-precision arithmetic (FEAT_FP16) instead of emulation.
    pub fp16: bool,
    /// Fuse a compare / flag-setting arith op into its consumer branch.
    pub fuse_cmp_arith: bool,
    /// Fuse a shift into a following add (`ADD Xd, Xn, Xm, LSL #imm`).
    pub fuse_shift_add: bool,
    /// Fuse an `ADRP`+`ADD` address pair into one recognized macro-op.
    pub fuse_addr_hi_lo: bool,
}

impl Default for ModelCaps {
    fn default() -> ModelCaps {
        ModelCaps {
            abi: Abi::Aapcs64,
            fetch_align: 0,
            fp16: false,
            fuse_cmp_arith: true,
            fuse_shift_add: true,
            fuse_addr_hi_lo: false,
        }
    }
}

/// Select A64 words for an IR function, discarding relocations.
///
/// Pending the `volt-ir` crate: always returns [`Error::Unsupported`].
pub fn select_function() -> Result<Vec<u32>, Error> {
    Err(Error::Unsupported)
}

/// Compile an IR function with `caps` tuning.
///
/// Pending the `volt-ir` crate: always returns [`Error::Unsupported`].
pub fn compile_function(_caps: &ModelCaps) -> Result<Compiled, Error> {
    Err(Error::Unsupported)
}

/// Like [`select_function`], but pads loop-header blocks with nops so they
/// land on a `fetch_align`-byte boundary.
///
/// Pending the `volt-ir` crate: always returns [`Error::Unsupported`].
pub fn select_function_aligned(_fetch_align: u16) -> Result<Vec<u32>, Error> {
    Err(Error::Unsupported)
}

/// Like [`select_function`], but also returns the source-line table for DWARF
/// `.debug_line`.
///
/// Pending the `volt-ir` crate: always returns [`Error::Unsupported`].
pub fn select_function_with_lines() -> Result<(Vec<u32>, Vec<LineEntry>), Error> {
    Err(Error::Unsupported)
}
