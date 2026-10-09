//! AArch64 encoding, Function-based native selection, object linking and JIT.
pub mod decode;
pub mod disasm;
pub mod encode;
pub mod isel;
#[cfg(all(
    target_os = "linux",
    any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", feature = "x86_64")
    )
))]
pub mod jit;
pub mod link;
pub mod object;
pub mod peephole;
pub mod simd_struct;
