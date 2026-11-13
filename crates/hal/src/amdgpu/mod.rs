//! AMDGPU hardware execution backend (GFX10.3 / RDNA2).
//!
//! Submits PM4 command packets through [`AmdgpuTransport`] for hardware
//! triangle rasterization, blending, and viewport transform. Precompiled shader
//! blobs come from `volt-shader` (`volt-shader-abi` format), keeping this module
//! `no_std` and independent of `pliron` or OS DRM syscalls.
#![allow(dead_code)]

pub mod pm4;
pub mod regs;
pub mod state;

use alloc::boxed::Box;
use alloc::vec::Vec;
use hashbrown::HashMap;
use volt_shader::amdgcn::{self, PsConstants, PsKey};
use volt_shader_abi::ShaderBlob;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmdgpuError {
    NoDevice,
    WrongDriver(&'static str),
    Io(i32),
    Timeout,
    UnsupportedShader,
}

/// Host-visible GPU buffer object managed by the platform DRM transport.
pub trait GpuBo: Send + Sync {
    fn handle(&self) -> u32;
    fn gpu_va(&self) -> u64;
    fn size(&self) -> u64;
    fn cpu_ptr(&self) -> *mut u8;
    fn as_slice_mut(&mut self) -> Option<&mut [u8]>;
}

/// Platform / DRM transport for AMDGPU command submission and GTT allocation.
pub trait AmdgpuTransport: Send + Sync {
    fn ip_discovery_version(&self) -> u32;
    fn device_id(&self) -> u16;
    fn virtual_address_range(&self) -> (u64, u64);
    fn create_context(&self) -> Result<u32, AmdgpuError>;
    fn destroy_context(&self, ctx_id: u32) -> Result<(), AmdgpuError>;
    fn alloc_gtt_bo(&self, size: u64, align: u64, va: u64) -> Result<Box<dyn GpuBo>, AmdgpuError>;
    fn submit_gfx_ib(
        &self,
        ctx_id: u32,
        ib_gpu_va: u64,
        dwords: u32,
        handles: &[u32],
    ) -> Result<u64, AmdgpuError>;
    fn wait_gfx_cs(&self, ctx_id: u32, seq: u64, timeout_ns: u64) -> Result<bool, AmdgpuError>;
}

fn flush_cpu_cache(ptr: *const u8, len: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::x86_64::{_mm_clflush, _mm_mfence};
        let start = (ptr as usize) & !63;
        let end = (ptr as usize).saturating_add(len);
        let mut address = start;
        while address < end {
            unsafe { _mm_clflush(address as *const u8) };
            address += 64;
        }
        unsafe { _mm_mfence() };
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = (ptr, len);
}

/// Hardware execution driver. Owns the transport handle, GTT allocators,
/// precompiled shader blobs, and the active PM4 command batch.
pub struct AmdgpuHardwareDriver {
    pub dev: Box<dyn AmdgpuTransport>,
    pub ctx_id: u32,
    pub processor: &'static str,
    va_cursor: u64,
    va_max: u64,

    // Precompiled fixed-function shaders.
    pub vs: ShaderBlob<'static>,
    pub vs_bo: Box<dyn GpuBo>,
    pub ps_cache: HashMap<PsKey, (ShaderBlob<'static>, Box<dyn GpuBo>)>,

    // Reusable GTT buffers.
    pub upload_arena: Box<dyn GpuBo>,
    pub arena_offset: usize,
    pub ib_bo: Box<dyn GpuBo>,

    // Current in-flight command stream.
    pub pm4: pm4::Pm4,
    pub referenced_handles: Vec<u32>,

    // Render-target BOs allocated for Device images.
    pub image_bos: HashMap<crate::ImageId, Box<dyn GpuBo>>,

    /// Latched on timeout or submit failure: forces CPU fallback thereafter.
    pub broken: bool,
}

impl Drop for AmdgpuHardwareDriver {
    fn drop(&mut self) {
        let _ = self.dev.destroy_context(self.ctx_id);
    }
}

const ARENA_INITIAL_SIZE: u64 = 2 * 1024 * 1024; // 2 MB
const IB_INITIAL_SIZE: u64 = 256 * 1024; // 256 KB
const SHADER_PREFETCH_PADDING: u64 = 1024; // 1 KB safe padding

impl AmdgpuHardwareDriver {
    /// Initialize AMDGPU hardware execution over an OS/DRM [`AmdgpuTransport`].
    pub fn from_transport(dev: Box<dyn AmdgpuTransport>) -> Result<Self, AmdgpuError> {
        let disc = dev.ip_discovery_version();
        let is_gfx103 =
            ((disc >> 16) & 0xff == 10) && ((disc >> 8) & 0xff == 3) && (disc & 0xff <= 7);
        if !is_gfx103 {
            #[cfg(feature = "std")]
            std::eprintln!(
                "volt-amdgpu: device 0x{:04x} discovery version {:#x} is not GFX10.3",
                dev.device_id(),
                disc
            );
            return Err(AmdgpuError::WrongDriver("not GFX10.3"));
        }
        let processor = amdgcn::gfx103_processor(disc);
        let ctx_id = dev.create_context()?;

        let (va_min, va_max) = dev.virtual_address_range();
        let mut va_cursor = va_min.max(0x100_0000);

        let mut alloc_gtt_init = |size: u64| -> Result<Box<dyn GpuBo>, AmdgpuError> {
            let align = 0x10000u64;
            let aligned = (size + align - 1) & !(align - 1);
            if va_cursor.saturating_add(aligned) > va_max {
                return Err(AmdgpuError::Io(12));
            }
            let va = va_cursor;
            va_cursor += aligned;
            dev.alloc_gtt_bo(size, 4096, va)
        };

        let upload_arena = alloc_gtt_init(ARENA_INITIAL_SIZE)?;
        let ib_bo = alloc_gtt_init(IB_INITIAL_SIZE)?;

        // Load prebuilt vertex shader blob and upload to GTT.
        let vs = amdgcn::vertex_shader();
        let mut vs_bo = alloc_gtt_init((vs.code().len() as u64) + SHADER_PREFETCH_PADDING)?;
        if let Some(slice) = vs_bo.as_slice_mut() {
            slice[..vs.code().len()].copy_from_slice(vs.code());
            slice[vs.code().len()..].fill(0);
        }

        #[cfg(feature = "std")]
        std::eprintln!(
            "volt-amdgpu: initialized for {} (device 0x{:04x}, context {})",
            processor,
            dev.device_id(),
            ctx_id
        );

        Ok(Self {
            va_cursor,
            va_max,
            dev,
            ctx_id,
            processor,
            vs,
            vs_bo,
            ps_cache: HashMap::new(),
            upload_arena,
            arena_offset: 0,
            ib_bo,
            pm4: pm4::Pm4::with_capacity(4096),
            referenced_handles: Vec::with_capacity(32),
            image_bos: HashMap::new(),
            broken: false,
        })
    }

    pub fn alloc_va(&mut self, size: u64) -> Result<u64, AmdgpuError> {
        let align = 0x10000u64;
        let aligned = (size + align - 1) & !(align - 1);
        if self.va_cursor.saturating_add(aligned) > self.va_max {
            return Err(AmdgpuError::Io(12));
        }
        let va = self.va_cursor;
        self.va_cursor += aligned;
        Ok(va)
    }

    pub fn alloc_gtt(&mut self, size: u64) -> Result<Box<dyn GpuBo>, AmdgpuError> {
        let va = self.alloc_va(size)?;
        self.dev.alloc_gtt_bo(size, 4096, va)
    }

    fn upload_code(&mut self, code: &[u8]) -> Result<Box<dyn GpuBo>, AmdgpuError> {
        let total = (code.len() as u64) + SHADER_PREFETCH_PADDING;
        let mut bo = self.alloc_gtt(total)?;
        if let Some(slice) = bo.as_slice_mut() {
            slice[..code.len()].copy_from_slice(code);
            slice[code.len()..].fill(0);
        }
        Ok(bo)
    }

    pub fn get_or_upload_ps(&mut self, key: &PsKey) -> Result<u64, AmdgpuError> {
        if let Some((_, bo)) = self.ps_cache.get(key) {
            return Ok(bo.gpu_va());
        }
        let sh = amdgcn::pixel_shader(key).ok_or(AmdgpuError::UnsupportedShader)?;
        let bo = self.upload_code(sh.code())?;
        let va = bo.gpu_va();
        self.ps_cache.insert(*key, (sh, bo));
        Ok(va)
    }

    pub fn can_draw(
        &self,
        color_id: Option<crate::ImageId>,
        pipe: &crate::Pipeline,
        frag_state: &volt_raster::FragState,
    ) -> Option<PsKey> {
        if self.broken {
            return None;
        }
        let cid = color_id?;
        if !self.image_bos.contains_key(&cid) {
            return None;
        }
        if pipe.depth_test || pipe.depth_write {
            return None;
        }
        PsKey::from_state(frag_state)
    }

    pub fn flush(&mut self) -> Result<(), AmdgpuError> {
        if self.pm4.is_empty() || self.broken {
            self.pm4.clear();
            self.referenced_handles.clear();
            self.arena_offset = 0;
            return Ok(());
        }

        const EVENT_FLUSH_AND_INV_CB_DATA_TS: u32 = 45;
        const EVENT_INDEX_EOP: u32 = 5;
        const GCR_CNTL_CB_FLUSH: u32 = 0x70f;
        self.pm4.release_mem(
            EVENT_FLUSH_AND_INV_CB_DATA_TS,
            EVENT_INDEX_EOP,
            GCR_CNTL_CB_FLUSH,
            0,
            0,
            0,
            0,
            0,
        );

        self.pm4.pad_to(32);
        let byte_len = self.pm4.len() * 4;

        if byte_len > self.ib_bo.size() as usize {
            let new_size = (byte_len as u64 * 2).max(IB_INITIAL_SIZE);
            self.ib_bo = self.alloc_gtt(new_size)?;
        }

        if let Some(slice) = self.ib_bo.as_slice_mut() {
            let u8s = bytemuck::cast_slice::<u32, u8>(&self.pm4.dwords);
            slice[..u8s.len()].copy_from_slice(u8s);
        }

        let mut handles = Vec::with_capacity(self.referenced_handles.len() + 4);
        handles.push(self.ib_bo.handle());
        handles.push(self.upload_arena.handle());
        handles.push(self.vs_bo.handle());
        for &h in &self.referenced_handles {
            if !handles.contains(&h) {
                handles.push(h);
            }
        }

        flush_cpu_cache(self.ib_bo.cpu_ptr(), byte_len);
        flush_cpu_cache(self.upload_arena.cpu_ptr(), self.arena_offset);
        flush_cpu_cache(self.vs_bo.cpu_ptr(), self.vs_bo.size() as usize);
        for (_, (_, bo)) in &self.ps_cache {
            flush_cpu_cache(bo.cpu_ptr(), bo.size() as usize);
        }
        for bo in self.image_bos.values() {
            flush_cpu_cache(bo.cpu_ptr(), bo.size() as usize);
        }

        let seq = match self.dev.submit_gfx_ib(
            self.ctx_id,
            self.ib_bo.gpu_va(),
            self.pm4.len() as u32,
            &handles,
        ) {
            Ok(s) => s,
            Err(e) => {
                self.broken = true;
                self.pm4.clear();
                self.referenced_handles.clear();
                self.arena_offset = 0;
                return Err(e);
            }
        };

        match self.dev.wait_gfx_cs(self.ctx_id, seq, 2_000_000_000) {
            Ok(true) => {}
            Ok(false) => {
                self.broken = true;
                return Err(AmdgpuError::Timeout);
            }
            Err(e) => {
                self.broken = true;
                return Err(e);
            }
        }

        for bo in self.image_bos.values() {
            flush_cpu_cache(bo.cpu_ptr(), bo.size() as usize);
        }

        self.pm4.clear();
        self.referenced_handles.clear();
        self.arena_offset = 0;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_draw(
        &mut self,
        color_id: crate::ImageId,
        pipe: &crate::Pipeline,
        vertices: &[crate::Vertex],
        indices: Option<&[u32]>,
        viewport: (i32, i32, u32, u32),
        scissor: Option<(i32, i32, u32, u32)>,
        frag_state: &volt_raster::FragState,
        ps_key: &PsKey,
        fb_w: u32,
        fb_h: u32,
        fb_pitch: u32,
        is_bgra: bool,
    ) -> Result<(), AmdgpuError> {
        let vbytes = bytemuck::cast_slice::<crate::Vertex, u8>(vertices);
        let ibytes = indices.map(bytemuck::cast_slice::<u32, u8>).unwrap_or(&[]);
        let v_aligned = (vbytes.len() + 15) & !15;
        let i_aligned = (ibytes.len() + 15) & !15;
        let needed = v_aligned + i_aligned;

        if self.pm4.len() + 512 > (self.ib_bo.size() as usize / 4)
            || self.arena_offset + needed > (self.upload_arena.size() as usize)
        {
            self.flush()?;
        }

        if needed > self.upload_arena.size() as usize {
            let new_size = (needed as u64 * 2).max(ARENA_INITIAL_SIZE);
            self.upload_arena = self.alloc_gtt(new_size)?;
            self.arena_offset = 0;
        }

        let base_va = self.upload_arena.gpu_va();
        let v_off = self.arena_offset;
        let i_off = v_off + v_aligned;
        if let Some(slice) = self.upload_arena.as_slice_mut() {
            slice[v_off..v_off + vbytes.len()].copy_from_slice(vbytes);
            if !ibytes.is_empty() {
                slice[i_off..i_off + ibytes.len()].copy_from_slice(ibytes);
            }
        }
        self.arena_offset += needed;

        let color_bo = self.image_bos.get(&color_id).unwrap();
        let (color_va, color_handle) = (color_bo.gpu_va(), color_bo.handle());
        self.referenced_handles.push(color_handle);

        if self.pm4.is_empty() {
            state::emit_preamble(&mut self.pm4);
        }

        state::emit_color_target(&mut self.pm4, color_va, fb_pitch, fb_w, fb_h, is_bgra);

        state::emit_raster_and_blend(
            &mut self.pm4,
            pipe,
            viewport,
            scissor,
            fb_w,
            fb_h,
            ps_key.uses_kill(),
        );

        let ps_va = self.get_or_upload_ps(ps_key)?;
        let ps_handle = self.ps_cache.get(ps_key).unwrap().1.handle();
        self.referenced_handles.push(ps_handle);

        let ps_vgprs = self
            .ps_cache
            .get(ps_key)
            .unwrap()
            .0
            .reg(regs::SPI_SHADER_PGM_RSRC1_PS)
            .map(|r| ((r & regs::RSRC1_VGPRS_MASK) + 1) * 8)
            .unwrap_or(16);
        let ps_input_ena = self
            .ps_cache
            .get(ps_key)
            .unwrap()
            .0
            .reg(regs::SPI_PS_INPUT_ENA)
            .unwrap_or(0);

        state::emit_shader_programs(
            &mut self.pm4,
            self.vs_bo.gpu_va(),
            16,
            ps_va,
            ps_vgprs,
            ps_input_ena,
            if ps_key.fog_mode != 0 { 2 } else { 1 },
        );

        state::emit_vs_user_data(&mut self.pm4, base_va + (v_off as u64));
        let ps_constants = PsConstants {
            fog_color: [
                frag_state.fog_color[0],
                frag_state.fog_color[1],
                frag_state.fog_color[2],
            ],
            fog_start: frag_state.fog_start,
            fog_end: frag_state.fog_end,
            fog_density: frag_state.fog_density,
            alpha_ref: frag_state.alpha_ref,
        };
        state::emit_ps_user_data(&mut self.pm4, &ps_constants);

        if let Some(idx) = indices {
            state::emit_draw_indexed(
                &mut self.pm4,
                base_va + (i_off as u64),
                idx.len() as u32,
                false,
            );
        } else {
            state::emit_draw_auto(&mut self.pm4, vertices.len() as u32);
        }

        Ok(())
    }
}
