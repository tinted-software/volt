use alloc::vec;
use alloc::vec::Vec;
use volt_platform::{PixelFormat, Surface};

/// Portable in-memory software presentation buffer (used on non-Linux hosts and headless surfaces).
#[derive(Debug, Default)]
pub struct SoftwareSurface {
    pub width: u32,
    pub height: u32,
    pub stride_bytes: u32,
    pub pixels: Vec<u8>,
    pub presents: u64,
}

impl SoftwareSurface {
    pub fn new(width: u32, height: u32) -> Self {
        let stride_bytes = width.saturating_mul(4);
        Self {
            width,
            height,
            stride_bytes,
            pixels: vec![0u8; stride_bytes as usize * height as usize],
            presents: 0,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.stride_bytes = width.saturating_mul(4);
        self.pixels
            .resize(self.stride_bytes as usize * height as usize, 0);
    }
}

impl Surface for SoftwareSurface {
    fn present(
        &mut self,
        width: u32,
        height: u32,
        stride_bytes: u32,
        _format: PixelFormat,
        pixels: &[u8],
    ) {
        self.width = width;
        self.height = height;
        self.stride_bytes = stride_bytes;
        let needed = stride_bytes as usize * height as usize;
        if pixels.len() >= needed {
            self.pixels.resize(needed, 0);
            self.pixels.copy_from_slice(&pixels[..needed]);
        } else {
            self.pixels.clear();
            self.pixels.extend_from_slice(pixels);
        }
        self.presents += 1;
    }
}
