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
