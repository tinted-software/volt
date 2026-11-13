//! AMDGPU kernel UAPI (`include/uapi/drm/amdgpu_drm.h`) and a thin, safe(ish)
//! device layer on top of it: device queries, GEM buffer objects with GPU VA
//! mappings, execution contexts, command submission and fence waits.
//!
//! Struct layouts mirror the kernel header exactly. Ioctl numbers encode the
//! struct (or union) size; `drm_ioctl` zero-extends/truncates between user
//! and kernel sizes, so newer kernels with grown structs stay compatible.
#![allow(non_camel_case_types)]
use crate::{DrmDevice, Error, MAP_SHARED, PROT_READ, PROT_WRITE, errno, iow, iowr, mmap, munmap};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ffi::{c_int, c_ulong, c_void};
use core::mem::size_of;

// ============================================================================
// Ioctl numbers
// ============================================================================

const DRM_COMMAND_BASE: u64 = 0x40;
const DRM_AMDGPU_GEM_CREATE: u64 = 0x00;
const DRM_AMDGPU_GEM_MMAP: u64 = 0x01;
const DRM_AMDGPU_CTX: u64 = 0x02;
const DRM_AMDGPU_CS: u64 = 0x04;
const DRM_AMDGPU_INFO: u64 = 0x05;
const DRM_AMDGPU_GEM_VA: u64 = 0x08;
const DRM_AMDGPU_WAIT_CS: u64 = 0x09;

/// `DRM_IOWR(.., union drm_amdgpu_gem_create)`: the union is the 32-byte `in`.
pub const DRM_IOCTL_AMDGPU_GEM_CREATE: c_ulong = iowr(
    DRM_COMMAND_BASE + DRM_AMDGPU_GEM_CREATE,
    b'd',
    size_of::<drm_amdgpu_gem_create_in>() as u64,
);
/// `DRM_IOWR(.., union drm_amdgpu_gem_mmap)` (8 bytes).
pub const DRM_IOCTL_AMDGPU_GEM_MMAP: c_ulong = iowr(
    DRM_COMMAND_BASE + DRM_AMDGPU_GEM_MMAP,
    b'd',
    size_of::<drm_amdgpu_gem_mmap_in>() as u64,
);
/// `DRM_IOWR(.., union drm_amdgpu_ctx)` (16 bytes).
pub const DRM_IOCTL_AMDGPU_CTX: c_ulong = iowr(
    DRM_COMMAND_BASE + DRM_AMDGPU_CTX,
    b'd',
    size_of::<drm_amdgpu_ctx_in>() as u64,
);
/// `DRM_IOWR(.., union drm_amdgpu_cs)` (24 bytes).
pub const DRM_IOCTL_AMDGPU_CS: c_ulong = iowr(
    DRM_COMMAND_BASE + DRM_AMDGPU_CS,
    b'd',
    size_of::<drm_amdgpu_cs_in>() as u64,
);
/// `DRM_IOW(.., struct drm_amdgpu_info)` (32 bytes).
pub const DRM_IOCTL_AMDGPU_INFO: c_ulong = iow(
    DRM_COMMAND_BASE + DRM_AMDGPU_INFO,
    b'd',
    size_of::<drm_amdgpu_info>() as u64,
);
/// `DRM_IOW(.., struct drm_amdgpu_gem_va)`.
pub const DRM_IOCTL_AMDGPU_GEM_VA: c_ulong = iow(
    DRM_COMMAND_BASE + DRM_AMDGPU_GEM_VA,
    b'd',
    size_of::<drm_amdgpu_gem_va>() as u64,
);
/// `DRM_IOWR(.., union drm_amdgpu_wait_cs)` (32 bytes).
pub const DRM_IOCTL_AMDGPU_WAIT_CS: c_ulong = iowr(
    DRM_COMMAND_BASE + DRM_AMDGPU_WAIT_CS,
    b'd',
    size_of::<drm_amdgpu_wait_cs_in>() as u64,
);

// ============================================================================
// Constants
// ============================================================================

pub const AMDGPU_GEM_DOMAIN_CPU: u64 = 0x1;
pub const AMDGPU_GEM_DOMAIN_GTT: u64 = 0x2;
pub const AMDGPU_GEM_DOMAIN_VRAM: u64 = 0x4;

pub const AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED: u64 = 1 << 0;
pub const AMDGPU_GEM_CREATE_NO_CPU_ACCESS: u64 = 1 << 1;
pub const AMDGPU_GEM_CREATE_CPU_GTT_USWC: u64 = 1 << 2;
pub const AMDGPU_GEM_CREATE_VRAM_CLEARED: u64 = 1 << 3;

pub const AMDGPU_VA_OP_MAP: u32 = 1;
pub const AMDGPU_VA_OP_UNMAP: u32 = 2;

pub const AMDGPU_VM_PAGE_READABLE: u32 = 1 << 1;
pub const AMDGPU_VM_PAGE_WRITEABLE: u32 = 1 << 2;
pub const AMDGPU_VM_PAGE_EXECUTABLE: u32 = 1 << 3;

pub const AMDGPU_CTX_OP_ALLOC_CTX: u32 = 1;
pub const AMDGPU_CTX_OP_FREE_CTX: u32 = 2;
pub const AMDGPU_CTX_OP_QUERY_STATE2: u32 = 4;

pub const AMDGPU_CTX_QUERY2_FLAGS_RESET: u64 = 1 << 0;
pub const AMDGPU_CTX_QUERY2_FLAGS_VRAMLOST: u64 = 1 << 1;
pub const AMDGPU_CTX_QUERY2_FLAGS_GUILTY: u64 = 1 << 2;

pub const AMDGPU_HW_IP_GFX: u32 = 0;
pub const AMDGPU_HW_IP_COMPUTE: u32 = 1;
pub const AMDGPU_HW_IP_DMA: u32 = 2;

pub const AMDGPU_INFO_HW_IP_INFO: u32 = 0x02;
pub const AMDGPU_INFO_VRAM_GTT: u32 = 0x14;
pub const AMDGPU_INFO_DEV_INFO: u32 = 0x16;

pub const AMDGPU_CHUNK_ID_IB: u32 = 0x01;
pub const AMDGPU_CHUNK_ID_BO_HANDLES: u32 = 0x06;

/// Ask the kernel to invalidate/write back GPU caches before the IB runs.
pub const AMDGPU_IB_FLAG_EMIT_MEM_SYNC: u32 = 1 << 6;

pub const AMDGPU_FAMILY_NV: u32 = 143;
pub const AMDGPU_FAMILY_VGH: u32 = 144;
pub const AMDGPU_FAMILY_YC: u32 = 146;
pub const AMDGPU_FAMILY_GC_10_3_6: u32 = 149;
pub const AMDGPU_FAMILY_GC_10_3_7: u32 = 151;

// ============================================================================
// UAPI structs
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct drm_amdgpu_gem_create_in {
    pub bo_size: u64,
    pub alignment: u64,
    pub domains: u64,
    pub domain_flags: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct drm_amdgpu_gem_mmap_in {
    pub handle: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct drm_amdgpu_ctx_in {
    pub op: u32,
    pub flags: u32,
    pub ctx_id: u32,
    pub priority: i32,
}

/// `drm_amdgpu_ctx_out.state` (query views of the ctx union).
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct drm_amdgpu_ctx_out_state {
    pub flags: u64,
    pub hangs: u32,
    pub reset_status: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct drm_amdgpu_bo_list_in {
    pub operation: u32,
    pub list_handle: u32,
    pub bo_number: u32,
    pub bo_info_size: u32,
    pub bo_info_ptr: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct drm_amdgpu_bo_list_entry {
    pub bo_handle: u32,
    pub bo_priority: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct drm_amdgpu_gem_va {
    pub handle: u32,
    pub _pad: u32,
    pub operation: u32,
    pub flags: u32,
    pub va_address: u64,
    pub offset_in_bo: u64,
    pub map_size: u64,
    pub vm_timeline_point: u64,
    pub vm_timeline_syncobj_out: u32,
    pub num_syncobj_handles: u32,
    pub input_fence_syncobj_handles: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct drm_amdgpu_info {
    pub return_pointer: u64,
    pub return_size: u32,
    pub query: u32,
    /// Query-specific union (largest member: `read_mmr_reg`, 4 x u32).
    pub query_data: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct drm_amdgpu_info_hw_ip {
    pub hw_ip_version_major: u32,
    pub hw_ip_version_minor: u32,
    pub capabilities_flags: u64,
    pub ib_start_alignment: u32,
    pub ib_size_alignment: u32,
    pub available_rings: u32,
    pub ip_discovery_version: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct drm_amdgpu_info_vram_gtt {
    pub vram_size: u64,
    pub vram_cpu_accessible_size: u64,
    pub gtt_size: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct drm_amdgpu_info_device {
    pub device_id: u32,
    pub chip_rev: u32,
    pub external_rev: u32,
    pub pci_rev: u32,
    pub family: u32,
    pub num_shader_engines: u32,
    pub num_shader_arrays_per_engine: u32,
    pub gpu_counter_freq: u32,
    pub max_engine_clock: u64,
    pub max_memory_clock: u64,
    pub cu_active_number: u32,
    pub cu_ao_mask: u32,
    pub cu_bitmap: [[u32; 4]; 4],
    pub enabled_rb_pipes_mask: u32,
    pub num_rb_pipes: u32,
    pub num_hw_gfx_contexts: u32,
    pub pcie_gen: u32,
    pub ids_flags: u64,
    pub virtual_address_offset: u64,
    pub virtual_address_max: u64,
    pub virtual_address_alignment: u32,
    pub pte_fragment_size: u32,
    pub gart_page_size: u32,
    pub ce_ram_size: u32,
    pub vram_type: u32,
    pub vram_bit_width: u32,
    pub vce_harvest_config: u32,
    pub gc_double_offchip_lds_buf: u32,
    pub prim_buf_gpu_addr: u64,
    pub pos_buf_gpu_addr: u64,
    pub cntl_sb_buf_gpu_addr: u64,
    pub param_buf_gpu_addr: u64,
    pub prim_buf_size: u32,
    pub pos_buf_size: u32,
    pub cntl_sb_buf_size: u32,
    pub param_buf_size: u32,
    pub wave_front_size: u32,
    pub num_shader_visible_vgprs: u32,
    pub num_cu_per_sh: u32,
    pub num_tcc_blocks: u32,
    pub gs_vgt_table_depth: u32,
    pub gs_prim_buffer_depth: u32,
    pub max_gs_waves_per_vgt: u32,
    pub pcie_num_lanes: u32,
    pub cu_ao_bitmap: [[u32; 4]; 4],
    pub high_va_offset: u64,
    pub high_va_max: u64,
    pub pa_sc_tile_steering_override: u32,
    pub tcc_disabled_mask: u64,
    pub min_engine_clock: u64,
    pub min_memory_clock: u64,
}

impl Default for drm_amdgpu_info_device {
    fn default() -> Self {
        // SAFETY: plain-old-data, all-zero is a valid bit pattern.
        unsafe { core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct drm_amdgpu_cs_chunk {
    pub chunk_id: u32,
    pub length_dw: u32,
    pub chunk_data: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct drm_amdgpu_cs_in {
    pub ctx_id: u32,
    pub bo_list_handle: u32,
    pub num_chunks: u32,
    pub flags: u32,
    /// Pointer to an array of `u64` pointers to [`drm_amdgpu_cs_chunk`].
    pub chunks: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct drm_amdgpu_cs_chunk_ib {
    pub _pad: u32,
    pub flags: u32,
    pub va_start: u64,
    pub ib_bytes: u32,
    pub ip_type: u32,
    pub ip_instance: u32,
    pub ring: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct drm_amdgpu_wait_cs_in {
    pub handle: u64,
    /// Absolute CLOCK_MONOTONIC timeout in ns.
    pub timeout: u64,
    pub ip_type: u32,
    pub ip_instance: u32,
    pub ring: u32,
    pub ctx_id: u32,
}

const _: () = {
    assert!(size_of::<drm_amdgpu_gem_create_in>() == 32);
    assert!(size_of::<drm_amdgpu_ctx_in>() == 16);
    assert!(size_of::<drm_amdgpu_cs_in>() == 24);
    assert!(size_of::<drm_amdgpu_info>() == 32);
    assert!(size_of::<drm_amdgpu_wait_cs_in>() == 32);
    assert!(size_of::<drm_amdgpu_cs_chunk_ib>() == 32);
    assert!(size_of::<drm_amdgpu_gem_va>() == 64);
};

// ============================================================================
// Monotonic clock (absolute ioctl timeouts)
// ============================================================================

#[repr(C)]
struct Timespec {
    tv_sec: i64,
    tv_nsec: i64,
}

unsafe extern "C" {
    fn clock_gettime(clk: c_int, ts: *mut Timespec) -> c_int;
}

const CLOCK_MONOTONIC: c_int = 1;

fn monotonic_ns() -> u64 {
    let mut ts = Timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe { clock_gettime(CLOCK_MONOTONIC, &mut ts) };
    (ts.tv_sec as u64) * 1_000_000_000 + ts.tv_nsec as u64
}

// ============================================================================
// Device
// ============================================================================

fn ioctl<T>(fd: c_int, req: c_ulong, arg: &mut T) -> Result<(), Error> {
    // EINTR/EAGAIN are restartable for every amdgpu ioctl we issue.
    loop {
        let r = unsafe { crate::ioctl(fd, req, arg as *mut T as *mut c_void) };
        if r == 0 {
            return Ok(());
        }
        let e = errno();
        if e != 4 && e != 11 {
            return Err(Error::Io(e));
        }
    }
}

/// An AMDGPU render node. Shared (`Arc`) by every buffer object allocated
/// from it, so the fd outlives all GEM handles and VA mappings.
pub struct AmdgpuDevice {
    drm: DrmDevice,
    pub info: drm_amdgpu_info_device,
    pub gfx_ip: drm_amdgpu_info_hw_ip,
}

impl AmdgpuDevice {
    /// Wraps an open DRM node and queries device + GFX IP info. Fails if the
    /// node is not driven by `amdgpu`.
    pub fn new(drm: DrmDevice) -> Result<Arc<Self>, Error> {
        let mut dev = Self {
            drm,
            info: Default::default(),
            gfx_ip: Default::default(),
        };
        dev.info = dev.query(AMDGPU_INFO_DEV_INFO, [0; 4])?;
        dev.gfx_ip = dev.query(AMDGPU_INFO_HW_IP_INFO, [AMDGPU_HW_IP_GFX, 0, 0, 0])?;
        Ok(Arc::new(dev))
    }

    pub fn fd(&self) -> c_int {
        self.drm.fd()
    }

    pub fn drm(&self) -> &DrmDevice {
        &self.drm
    }

    fn query<T: Default>(&self, query: u32, query_data: [u32; 4]) -> Result<T, Error> {
        let mut out = T::default();
        let mut req = drm_amdgpu_info {
            return_pointer: &mut out as *mut T as u64,
            return_size: size_of::<T>() as u32,
            query,
            query_data,
        };
        ioctl(self.fd(), DRM_IOCTL_AMDGPU_INFO, &mut req)?;
        Ok(out)
    }

    pub fn query_vram_gtt(&self) -> Result<drm_amdgpu_info_vram_gtt, Error> {
        self.query(AMDGPU_INFO_VRAM_GTT, [0; 4])
    }

    /// Allocate a GPU execution context.
    pub fn create_context(&self) -> Result<u32, Error> {
        let mut req = drm_amdgpu_ctx_in {
            op: AMDGPU_CTX_OP_ALLOC_CTX,
            ..Default::default()
        };
        ioctl(self.fd(), DRM_IOCTL_AMDGPU_CTX, &mut req)?;
        // out.alloc.ctx_id aliases the first u32 of the union.
        Ok(req.op)
    }

    pub fn destroy_context(&self, ctx_id: u32) -> Result<(), Error> {
        let mut req = drm_amdgpu_ctx_in {
            op: AMDGPU_CTX_OP_FREE_CTX,
            ctx_id,
            ..Default::default()
        };
        ioctl(self.fd(), DRM_IOCTL_AMDGPU_CTX, &mut req)
    }

    /// `AMDGPU_CTX_OP_QUERY_STATE2`: reset / guilty / VRAM-lost flags.
    pub fn query_context_state(&self, ctx_id: u32) -> Result<u64, Error> {
        let mut req = drm_amdgpu_ctx_in {
            op: AMDGPU_CTX_OP_QUERY_STATE2,
            ctx_id,
            ..Default::default()
        };
        ioctl(self.fd(), DRM_IOCTL_AMDGPU_CTX, &mut req)?;
        // SAFETY: same 16-byte union, now holding `out.state`.
        let st: drm_amdgpu_ctx_out_state = unsafe { core::mem::transmute(req) };
        Ok(st.flags)
    }

    /// Allocate a buffer object, map it at `va` in the process GPU VM and
    /// (optionally) map it for CPU access.
    pub fn alloc_bo(
        self: &Arc<Self>,
        size: u64,
        alignment: u64,
        domain: u64,
        flags: u64,
        va: u64,
        cpu_map: bool,
    ) -> Result<AmdgpuBo, Error> {
        let align = alignment.max(4096);
        let size = (size.max(1) + 4095) & !4095;
        let mut create = drm_amdgpu_gem_create_in {
            bo_size: size,
            alignment: align,
            domains: domain,
            domain_flags: flags,
        };
        ioctl(self.fd(), DRM_IOCTL_AMDGPU_GEM_CREATE, &mut create)?;
        // out.handle aliases the first u32 of the union.
        let handle = create.bo_size as u32;

        // From here on Drop cleans up whatever has been established.
        let mut bo = AmdgpuBo {
            dev: self.clone(),
            handle,
            size,
            gpu_va: 0,
            cpu_ptr: core::ptr::null_mut(),
        };

        let mut va_req = drm_amdgpu_gem_va {
            handle,
            operation: AMDGPU_VA_OP_MAP,
            flags: AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE | AMDGPU_VM_PAGE_EXECUTABLE,
            va_address: va,
            offset_in_bo: 0,
            map_size: size,
            ..Default::default()
        };
        ioctl(self.fd(), DRM_IOCTL_AMDGPU_GEM_VA, &mut va_req)?;
        bo.gpu_va = va;

        if cpu_map {
            let mut m = drm_amdgpu_gem_mmap_in { handle, _pad: 0 };
            ioctl(self.fd(), DRM_IOCTL_AMDGPU_GEM_MMAP, &mut m)?;
            // out.addr_ptr (u64 fake mmap offset) aliases the whole union.
            let offset: u64 = unsafe { core::mem::transmute(m) };
            let ptr = unsafe {
                mmap(
                    core::ptr::null_mut(),
                    size as usize,
                    PROT_READ | PROT_WRITE,
                    MAP_SHARED,
                    self.fd(),
                    offset as i64,
                )
            };
            if ptr as isize == -1 || ptr.is_null() {
                return Err(Error::Io(errno()));
            }
            bo.cpu_ptr = ptr as *mut u8;
        }
        Ok(bo)
    }

    /// Submit one IB to `ip_type` ring 0 with an explicit BO working set.
    /// Returns the fence sequence number for [`AmdgpuDevice::wait_cs`].
    pub fn submit(
        &self,
        ctx_id: u32,
        ip_type: u32,
        ib_va: u64,
        ib_dwords: u32,
        ib_flags: u32,
        bo_handles: &[u32],
    ) -> Result<u64, Error> {
        let entries: Vec<drm_amdgpu_bo_list_entry> = bo_handles
            .iter()
            .map(|&h| drm_amdgpu_bo_list_entry {
                bo_handle: h,
                bo_priority: 0,
            })
            .collect();
        let bo_list = drm_amdgpu_bo_list_in {
            operation: !0, // unused for the chunk form
            list_handle: !0,
            bo_number: entries.len() as u32,
            bo_info_size: size_of::<drm_amdgpu_bo_list_entry>() as u32,
            bo_info_ptr: entries.as_ptr() as u64,
        };
        let ib = drm_amdgpu_cs_chunk_ib {
            _pad: 0,
            flags: ib_flags,
            va_start: ib_va,
            ib_bytes: ib_dwords * 4,
            ip_type,
            ip_instance: 0,
            ring: 0,
        };
        let chunks = [
            drm_amdgpu_cs_chunk {
                chunk_id: AMDGPU_CHUNK_ID_BO_HANDLES,
                length_dw: (size_of::<drm_amdgpu_bo_list_in>() / 4) as u32,
                chunk_data: &bo_list as *const _ as u64,
            },
            drm_amdgpu_cs_chunk {
                chunk_id: AMDGPU_CHUNK_ID_IB,
                length_dw: (size_of::<drm_amdgpu_cs_chunk_ib>() / 4) as u32,
                chunk_data: &ib as *const _ as u64,
            },
        ];
        let chunk_ptrs = [&chunks[0] as *const _ as u64, &chunks[1] as *const _ as u64];
        let mut cs = drm_amdgpu_cs_in {
            ctx_id,
            bo_list_handle: 0,
            num_chunks: chunk_ptrs.len() as u32,
            flags: 0,
            chunks: chunk_ptrs.as_ptr() as u64,
        };
        ioctl(self.fd(), DRM_IOCTL_AMDGPU_CS, &mut cs)?;
        // out.handle (u64 seq) aliases the start of the union.
        Ok(cs.ctx_id as u64 | ((cs.bo_list_handle as u64) << 32))
    }

    /// Wait up to `timeout_ns` for submission `seq`. `Ok(true)` = signaled,
    /// `Ok(false)` = still busy at the deadline.
    pub fn wait_cs(
        &self,
        ctx_id: u32,
        ip_type: u32,
        seq: u64,
        timeout_ns: u64,
    ) -> Result<bool, Error> {
        let mut req = drm_amdgpu_wait_cs_in {
            handle: seq,
            timeout: monotonic_ns().saturating_add(timeout_ns),
            ip_type,
            ip_instance: 0,
            ring: 0,
            ctx_id,
        };
        ioctl(self.fd(), DRM_IOCTL_AMDGPU_WAIT_CS, &mut req)?;
        // out.status aliases `handle`: 0 = idle, 1 = timed out.
        Ok(req.handle == 0)
    }
}

// ============================================================================
// Buffer object
// ============================================================================

/// A GEM buffer object mapped into the GPU VM (and optionally the CPU).
/// Drop unmaps both views and closes the handle.
pub struct AmdgpuBo {
    dev: Arc<AmdgpuDevice>,
    pub handle: u32,
    pub size: u64,
    pub gpu_va: u64,
    pub cpu_ptr: *mut u8,
}

// The CPU mapping is plain shared memory; synchronization with the GPU is
// the owner's responsibility (fence waits).
unsafe impl Send for AmdgpuBo {}
unsafe impl Sync for AmdgpuBo {}

impl AmdgpuBo {
    pub fn as_slice(&self) -> Option<&[u8]> {
        (!self.cpu_ptr.is_null())
            .then(|| unsafe { core::slice::from_raw_parts(self.cpu_ptr, self.size as usize) })
    }

    pub fn as_slice_mut(&mut self) -> Option<&mut [u8]> {
        (!self.cpu_ptr.is_null())
            .then(|| unsafe { core::slice::from_raw_parts_mut(self.cpu_ptr, self.size as usize) })
    }
}

impl Drop for AmdgpuBo {
    fn drop(&mut self) {
        let fd = self.dev.fd();
        if !self.cpu_ptr.is_null() {
            unsafe { munmap(self.cpu_ptr as *mut c_void, self.size as usize) };
        }
        if self.gpu_va != 0 {
            let mut req = drm_amdgpu_gem_va {
                handle: self.handle,
                operation: AMDGPU_VA_OP_UNMAP,
                va_address: self.gpu_va,
                map_size: self.size,
                ..Default::default()
            };
            let _ = ioctl(fd, DRM_IOCTL_AMDGPU_GEM_VA, &mut req);
        }
        let mut close = crate::DrmGemClose {
            handle: self.handle,
            pad: 0,
        };
        let _ = ioctl(fd, crate::DRM_IOCTL_GEM_CLOSE, &mut close);
    }
}
