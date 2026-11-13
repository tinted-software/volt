//! AArch64 encoding, Function-based native selection, object linking and JIT.
pub use volt_target_core::aarch64::*;

#[cfg(all(
    feature = "std",
    target_os = "linux",
    any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", feature = "x86_64")
    )
))]
pub mod jit;

#[cfg(all(test, feature = "std"))]
mod isel_tests;
