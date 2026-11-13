extern crate alloc;

pub use volt_emulator_core::{devices, memory};

pub mod aarch64;

#[cfg(all(feature = "std", feature = "system"))]
pub mod system;

#[cfg(all(target_os = "linux", feature = "std", feature = "user"))]
pub mod user;

pub use aarch64::cpu::Cpu;
#[cfg(all(target_os = "linux", feature = "std", feature = "user"))]
pub use user::Error;
