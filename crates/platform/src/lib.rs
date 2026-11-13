//! Platform abstraction traits for Volt's `no_std` runtime core.
#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use core::time::Duration;

/// An owned, immutable executable memory region produced by [`ExecMemory`].
///
/// Implementations must guarantee that `as_ptr()` points to `len()` readable,
/// executable bytes for the lifetime of `self`, and that W^X is upheld (never
/// simultaneously writable and executable).
pub trait ExecImage: Send + Sync {
    fn as_ptr(&self) -> *const u8;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn code(&self) -> &[u8] {
        // SAFETY: implementations guarantee `as_ptr()` is valid for `len()` readable bytes.
        unsafe { core::slice::from_raw_parts(self.as_ptr(), self.len()) }
    }
}

pub trait ExecMemory: Send + Sync {
    type Image: ExecImage;
    type Error: core::error::Error + Send + Sync + 'static;

    fn map_code(&self, code: &[u8]) -> Result<Self::Image, Self::Error>;

    /// Synchronize the calling thread's instruction fetch pipeline with code
    /// mapped on another thread.
    fn instruction_barrier(&self) {}
}

/// Guest-physical RAM backing buffer.
pub trait PhysMap: Send + Sync {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn as_slice(&self) -> &[u8];
    fn as_mut_slice(&mut self) -> &mut [u8];
    fn as_ptr(&self) -> *const u8 {
        self.as_slice().as_ptr()
    }
    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.as_mut_slice().as_mut_ptr()
    }
}

impl PhysMap for Vec<u8> {
    fn len(&self) -> usize {
        Vec::len(self)
    }
    fn as_slice(&self) -> &[u8] {
        self.as_slice()
    }
    fn as_mut_slice(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

/// Storage backend for VirtIO block devices.
#[allow(clippy::result_unit_err)]
pub trait BlockBackend: Send {
    fn len(&self) -> u64;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<(), ()>;
    fn write_at(&mut self, offset: u64, buf: &[u8]) -> Result<(), ()>;
    fn flush(&mut self) -> Result<(), ()>;
}

impl BlockBackend for Vec<u8> {
    fn len(&self) -> u64 {
        self.len() as u64
    }

    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<(), ()> {
        let start = usize::try_from(offset).map_err(|_| ())?;
        let end = start.checked_add(buf.len()).ok_or(())?;
        if end <= self.len() {
            buf.copy_from_slice(&self[start..end]);
            Ok(())
        } else {
            Err(())
        }
    }

    fn write_at(&mut self, offset: u64, buf: &[u8]) -> Result<(), ()> {
        let start = usize::try_from(offset).map_err(|_| ())?;
        let end = start.checked_add(buf.len()).ok_or(())?;
        if end > self.len() {
            self.resize(end, 0);
        }
        self[start..end].copy_from_slice(buf);
        Ok(())
    }

    fn flush(&mut self) -> Result<(), ()> {
        Ok(())
    }
}

/// Serial / UART output sink.
pub trait Console {
    fn write_byte(&mut self, byte: u8);
    fn write_bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_byte(b);
        }
    }
    fn flush(&mut self) {}
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NullConsole;

impl Console for NullConsole {
    fn write_byte(&mut self, _byte: u8) {}
}

impl Console for Vec<u8> {
    fn write_byte(&mut self, byte: u8) {
        self.push(byte);
    }
    fn write_bytes(&mut self, bytes: &[u8]) {
        self.extend_from_slice(bytes);
    }
}

impl<F: FnMut(u8)> Console for F {
    fn write_byte(&mut self, byte: u8) {
        self(byte);
    }
}

/// Monotonic clock source for emulated timers and timeouts.
pub trait Clock: Send + Sync {
    fn now(&self) -> Duration;
}

/// Sleep/wait interface paired with [`Clock`] for paced WFI idling.
pub trait Timer: Clock {
    fn sleep(&self, duration: Duration);
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NullClock;

impl Clock for NullClock {
    fn now(&self) -> Duration {
        Duration::ZERO
    }
}

impl Timer for NullClock {
    fn sleep(&self, _duration: Duration) {}
}

/// Pixel format for [`Surface`] scanout presentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelFormat {
    B8G8R8A8Unorm,
    B8G8R8X8Unorm,
    R8G8B8A8Unorm,
    R8G8B8X8Unorm,
}

/// Scanout presentation target for VirtIO-GPU and EGL.
pub trait Surface: Send {
    fn present(
        &mut self,
        width: u32,
        height: u32,
        stride_bytes: u32,
        format: PixelFormat,
        pixels: &[u8],
    );
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NullSurface;

impl Surface for NullSurface {
    fn present(
        &mut self,
        _width: u32,
        _height: u32,
        _stride_bytes: u32,
        _format: PixelFormat,
        _pixels: &[u8],
    ) {
    }
}

impl<S: Surface + ?Sized> Surface for alloc::boxed::Box<S> {
    fn present(
        &mut self,
        width: u32,
        height: u32,
        stride_bytes: u32,
        format: PixelFormat,
        pixels: &[u8],
    ) {
        (**self).present(width, height, stride_bytes, format, pixels);
    }
}
