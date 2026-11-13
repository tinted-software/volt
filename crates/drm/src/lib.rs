//! Volt DRM — Linux DRM client substrate (`libdrm` replacement).
//!
//! Only compiled on Linux; on other targets this crate exposes no OS symbols
//! so the workspace builds cleanly everywhere.
#![no_std]
extern crate alloc;

#[cfg(target_os = "linux")]
pub mod amdgpu;
#[cfg(target_os = "linux")]
mod linux;

#[cfg(target_os = "linux")]
pub use amdgpu::*;
#[cfg(target_os = "linux")]
pub use linux::*;
