//! Shared Window System Integration (WSI) for Volt (`volt-gl` and `volt-vulkan`).
//!
//! Encapsulates Linux DRM/GEM/DRI3/X11-MIT-SHM presentation, AMDGPU DRM
//! transport, and portable software presentation fallbacks so non-Linux hosts
//! (such as macOS) build and test the software stack cleanly.

extern crate alloc;

#[cfg(all(feature = "std", feature = "amdgpu", target_os = "linux"))]
pub mod amdgpu_drm;
#[cfg(all(feature = "std", target_os = "linux"))]
pub mod dri3;
#[cfg(all(feature = "std", target_os = "linux"))]
pub mod x11;

#[cfg(target_os = "linux")]
pub use volt_drm as drm;

pub mod sw;
