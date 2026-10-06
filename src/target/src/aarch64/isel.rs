//! AArch64 Function-based instruction selection and AAPCS64 lowering.
//!
//! Ported from Vulcan's AArch64 selector. Allocation uses the shared Wimmer
//! allocator, including split-boundary and parallel control-flow-edge moves.
//! Ported and modified from Vulcan, Copyright 2026 LilithSemi (Apache-2.0).

use alloc::string::{String, ToString as _};
use alloc::vec::Vec;
use core::fmt;
use volt_ir::function::Function;

#[path = "lower.rs"]
mod lower;

/// Instruction selection failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    /// The IR construct has no A64 lowering.
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

/// Compile a callable native function, resolving external symbols.
pub fn compile(function: &Function) -> Result<Vec<u32>, Error> {
    Ok(lower::compile(function, &ModelCaps::default(), true)?.code)
}

/// Select A64 words for an IR function.
pub fn select_function(function: &Function) -> Result<Vec<u32>, Error> {
    compile(function)
}

/// Compile an IR function for the object/module linker, retaining relocations.
pub fn compile_function(function: &Function, caps: &ModelCaps) -> Result<Compiled, Error> {
    lower::compile(function, caps, false)
}

/// Select code with aligned loop headers.
pub fn select_function_aligned(function: &Function, fetch_align: u16) -> Result<Vec<u32>, Error> {
    Ok(lower::compile(
        function,
        &ModelCaps {
            fetch_align,
            ..ModelCaps::default()
        },
        true,
    )?
    .code)
}

/// Select code and its source-line table.
pub fn select_function_with_lines(
    function: &Function,
) -> Result<(Vec<u32>, Vec<LineEntry>), Error> {
    let compiled = lower::compile(function, &ModelCaps::default(), true)?;
    Ok((compiled.code, compiled.lines))
}

#[cfg(test)]
#[path = "isel_tests.rs"]
mod tests;
