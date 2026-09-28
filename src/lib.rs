//! Modular dynamic translation infrastructure for guest instruction sets.
//!
//! Architecture decoders and state live in target-specific crates; lowering and
//! native execution engines live behind backend crates. The core facade does
//! not depend on Mirage.

pub use volt_jit_aarch64 as aarch64;
pub use volt_jit_aarch64_llvm as aarch64_llvm;
pub use volt_jit_decoder as decoder;
pub use volt_jit_llvm as llvm;

/// Caller-owned guest-memory boundary shared by architecture/runtime crates.
pub mod memory;
