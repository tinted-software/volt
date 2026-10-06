// SPDX-License-Identifier: Apache-2.0
// Copyright LilithSemi (Vulcan/Mirage). Modified for the Volt Rust port.
//! Native System V AMD64 backend, ported from Vulcan's x86-64 target.
pub mod disasm;
pub mod elf;
pub mod encode;
pub mod isel;
pub mod link;
pub mod object;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    Unsupported(&'static str),
    InvalidFunction,
    RelocationOutOfRange,
    UndefinedSymbol(alloc::string::String),
    Object(alloc::string::String),
}
impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Unsupported(s) => write!(f, "unsupported x86-64 lowering: {s}"),
            Self::InvalidFunction => write!(f, "invalid IR function"),
            Self::RelocationOutOfRange => write!(f, "x86-64 relocation out of range"),
            Self::UndefinedSymbol(s) => write!(f, "undefined symbol: {s}"),
            Self::Object(s) => write!(f, "ELF object error: {s}"),
        }
    }
}
impl core::error::Error for Error {}
pub fn compile(function: &volt_ir::function::Function) -> Result<alloc::vec::Vec<u8>, Error> {
    let result = isel::compile_object(function)?;
    if let Some(r) = result.relocs.first() {
        return Err(Error::UndefinedSymbol(r.symbol.clone()));
    }
    Ok(result.code)
}

#[cfg(all(test, target_arch = "x86_64", target_os = "linux"))]
mod tests;
