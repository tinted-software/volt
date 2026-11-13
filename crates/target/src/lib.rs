#![cfg_attr(not(feature = "std"), no_std)]

//! Volt native code generation, encoding, linking, and disassembly.
extern crate alloc;

#[cfg(test)]
extern crate std;

pub use volt_target_core::{DecodeError, Decoder, regalloc};

#[cfg(feature = "aarch64")]
pub mod aarch64;

#[cfg(feature = "x86_64")]
pub use volt_target_core::x86_64;

#[cfg(feature = "std")]
pub mod native;

#[cfg(all(test, feature = "x86_64", target_arch = "x86_64", target_os = "linux"))]
mod x86_64_tests;
