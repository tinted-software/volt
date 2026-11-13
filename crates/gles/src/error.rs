//! Error type shared by the GLES/EGL layers.
//!
//! Failures are surfaced as typed values which callers translate into
//! EGL/GL error codes at the FFI boundary (`eglGetError` / `glGetError`).

use core::fmt;

/// Errors produced by context/surface creation and present paths.
#[derive(Debug)]
pub enum VantageError {
    /// No surface is configured for the requested operation.
    NoSurface,
    /// Acquiring the presentation target failed.
    SurfaceAcquire(&'static str),
    /// The JIT codegen backend is unavailable (freestanding build).
    JitUnavailable,
}

impl fmt::Display for VantageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VantageError::NoSurface => write!(f, "no surface configured"),
            VantageError::SurfaceAcquire(msg) => {
                write!(f, "surface texture acquire failed: {msg}")
            }
            VantageError::JitUnavailable => write!(f, "no codegen backend available"),
        }
    }
}

impl core::error::Error for VantageError {}
