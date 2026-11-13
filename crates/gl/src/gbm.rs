//! Vantage GBM — `gbm.h`-compatible Generic Buffer Manager over [`volt_wsi::drm`].
//!
//! Rust replacement for Mesa's `libgbm` backend (`gbm_dri`). Exports the
//! complete public `gbm.h` symbol set as a `cdylib` so compositors and
//! EGL/`platform_gbm` consumers can bind against a mesa-free stack.
//!
//! Backend semantics (single-plane, CPU-visible linear buffers for now):
//! - Allocation: DRM dumb GEM + persistent mmap (what DRI3/PRIME imports).
//! - Modifiers: only `GBM_FORMAT_MOD_LINEAR` (== `DRM_FORMAT_MOD_INVALID`
//!   "driver default" also accepted at creation, resolved to linear).
//! - Import: `GBM_BO_IMPORT_FD` / `..._FD_MODIFIER` re-wrap an external
//!   dmabuf via `PRIME_FD_TO_HANDLE` (rejects multi-plane / non-linear).
//!
//! Object lifetime follows Mesa's: `gbm_device` is refcounted through the
//! exported handles (the fd it is given is *borrowed*, never closed), and
//! BOs/surfaces hold device references.
//!
//! Threading: no `std`; internal state is guarded by a small spin lock.
//! Callers that interleave `gbm_bo_map`/`unmap` on the same BO must
//! serialize themselves, matching Mesa's contract.
#![allow(non_camel_case_types, unsafe_op_in_unsafe_fn)]
use alloc::{boxed::Box, sync::Arc, vec::Vec};

use core::ffi::{c_char, c_int, c_uint, c_ulong, c_void};
use core::mem::ManuallyDrop;
use spin::Mutex;
use volt_wsi::drm::{
    DRM_FORMAT_ARGB8888, DRM_FORMAT_MOD_INVALID, DRM_FORMAT_MOD_LINEAR, DRM_FORMAT_RGB565,
    DRM_FORMAT_XRGB8888, DrmDevice, GemBuffer,
};

// ============================================================================
// Public gbm.h types
// ============================================================================

#[repr(C)]
pub struct gbm_device {
    _opaque: [u8; 0],
}
#[repr(C)]
pub struct gbm_bo {
    _opaque: [u8; 0],
}
#[repr(C)]
pub struct gbm_surface {
    _opaque: [u8; 0],
}

/// `union gbm_bo_handle`
#[repr(C)]
#[derive(Clone, Copy)]
pub union gbm_bo_handle {
    pub ptr: *mut c_void,
    pub s32: i32,
    pub u32: u32,
    pub s64: i64,
    pub u64: u64,
}

/// `struct gbm_format_name_desc`
#[repr(C)]
pub struct gbm_format_name_desc {
    pub name: [c_char; 5],
}

// gbm_bo_flags (gbm.h values)
pub const GBM_BO_USE_SCANOUT: u32 = 1 << 0;
pub const GBM_BO_USE_CURSOR: u32 = 1 << 1;
pub const GBM_BO_USE_RENDERING: u32 = 1 << 2;
pub const GBM_BO_USE_WRITE: u32 = 1 << 3;
pub const GBM_BO_USE_LINEAR: u32 = 1 << 4;
pub const GBM_BO_USE_PROTECTED: u32 = 1 << 5;
pub const GBM_BO_USE_FRONT_RENDERING: u32 = 1 << 6;
/// Compression bits: `GBM_BO_FIXED_COMPRESSION_MASK`
pub const GBM_BO_FIXED_COMPRESSION_MASK: u32 = ((1 << 11) - 1) & !((1 << 7) - 1);

// gbm_bo_import types
pub const GBM_BO_IMPORT_WL_BUFFER: u32 = 0x5501;
pub const GBM_BO_IMPORT_EGL_IMAGE: u32 = 0x5502;
pub const GBM_BO_IMPORT_FD: u32 = 0x5503;
pub const GBM_BO_IMPORT_FD_MODIFIER: u32 = 0x5504;

/// `struct gbm_import_fd_data`
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct gbm_import_fd_data {
    pub fd: c_int,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub format: u32,
}

pub const GBM_MAX_PLANES: usize = 4;

/// `struct gbm_import_fd_modifier_data`
#[repr(C)]
#[derive(Clone, Copy)]
pub struct gbm_import_fd_modifier_data {
    pub width: u32,
    pub height: u32,
    pub format: u32,
    pub num_fds: u32,
    pub fds: [c_int; GBM_MAX_PLANES],
    pub strides: [i32; GBM_MAX_PLANES],
    pub offsets: [i32; GBM_MAX_PLANES],
    pub modifier: u64,
}

/// `enum gbm_bo_transfer_flags`
pub const GBM_BO_TRANSFER_READ: u32 = 1 << 0;
pub const GBM_BO_TRANSFER_WRITE: u32 = 1 << 1;
pub const GBM_BO_TRANSFER_READ_WRITE: u32 = GBM_BO_TRANSFER_READ | GBM_BO_TRANSFER_WRITE;

// ============================================================================
// Internal object model
// ============================================================================

/// A `gbm_device`: wraps a caller-provided DRM fd (never closed here).
struct Device {
    /// `ManuallyDrop` so the borrowed fd survives device destruction.
    drm: ManuallyDrop<DrmDevice>,
}

/// BO storage: either a dumb buffer we own or an imported dmabuf handle.
enum Storage {
    Dumb(GemBuffer),
    Imported {
        handle: u32,
        ptr: *mut u8,
        len: usize,
    },
}

struct BoInner {
    dev: Arc<Device>,
    width: u32,
    height: u32,
    format: u32,
    /// GBM_BO_USE_* as given at creation; consumed by surface re-creation
    /// and future tiling decisions (dumb backend is linear-only today).
    #[allow(dead_code)]
    flags: u32,
    modifier: u64,
    stride: u32,
    size: u64,
    storage: Storage,
    /// `gbm_bo_set_user_data` pair.
    user_data: Mutex<(*mut c_void, Option<extern "C" fn(*mut gbm_bo, *mut c_void)>)>,
}

// Mapped storage makes this shareable behind the surface/display mutexes.
unsafe impl Send for BoInner {}
unsafe impl Sync for BoInner {}

impl Drop for BoInner {
    fn drop(&mut self) {
        if let Storage::Imported { handle, ptr, len } = self.storage {
            if !ptr.is_null() {
                unsafe { munmap_impl(ptr as *mut c_void, len) };
            }
            let _ = self.dev.drm.gem_close(handle);
        }
    }
}

struct SurfaceInner {
    dev: Arc<Device>,
    width: u32,
    height: u32,
    format: u32,
    /// GBM_BO_USE_* as given at creation; consumed by surface re-creation
    /// and future tiling decisions (dumb backend is linear-only today).
    #[allow(dead_code)]
    flags: u32,
    /// BOs released back by the consumer, ready for `lock_front_buffer`.
    free: Mutex<Vec<Arc<BoInner>>>,
    /// BOs handed out as front buffers.
    busy: Mutex<Vec<Arc<BoInner>>>,
}

fn dev_of(d: *mut gbm_device) -> Option<Arc<Device>> {
    if d.is_null() {
        return None;
    }
    // Safety: exported handles were produced from `Arc<Device>::into_raw` and
    // carry at least one strong ref (the handle itself).
    let arc = unsafe { Arc::from_raw(d as *const Device) };
    let clone = arc.clone();
    core::mem::forget(arc);
    Some(clone)
}

// ============================================================================
// gbm.h C ABI
// ============================================================================

/// `gbm_create_device`: wraps a DRM fd (borrowed; GBM never closes it).
/// Validates the node with `DRM_IOCTL_VERSION` like Mesa does.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gbm_create_device(fd: c_int) -> *mut gbm_device {
    // Validate the node via DRM_IOCTL_VERSION like Mesa's gbm_dri does; the
    // probe borrows the fd and is forgotten (never closed).
    let probe = DrmDevice::from_fd(fd);
    let valid = probe.version().is_ok();
    core::mem::forget(probe);
    if !valid {
        return core::ptr::null_mut();
    }
    let dev = Arc::new(Device {
        drm: ManuallyDrop::new(unsafe { DrmDevice::from_fd(fd) }),
    });
    as_gbm(dev) as *mut gbm_device
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gbm_device_get_fd(gbm: *mut gbm_device) -> c_int {
    match dev_of(gbm) {
        Some(d) => d.drm.fd(),
        None => -1,
    }
}

static BACKEND_NAME: &[u8] = b"GBM\0";

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gbm_device_get_backend_name(_gbm: *mut gbm_device) -> *const c_char {
    BACKEND_NAME.as_ptr() as *const c_char
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gbm_device_is_format_supported(
    gbm: *mut gbm_device,
    format: u32,
    _flags: u32,
) -> c_int {
    if dev_of(gbm).is_none() {
        return 0;
    }
    (bytes_per_pixel(format) != 0) as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gbm_device_get_format_modifier_plane_count(
    gbm: *mut gbm_device,
    format: u32,
    modifier: u64,
) -> c_int {
    if dev_of(gbm).is_none()
        || bytes_per_pixel(format) == 0
        || (modifier != DRM_FORMAT_MOD_LINEAR && modifier != DRM_FORMAT_MOD_INVALID)
    {
        return -1;
    }
    1 // single-plane linear
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gbm_device_destroy(gbm: *mut gbm_device) {
    if !gbm.is_null() {
        drop(Arc::from_raw(gbm as *const Device));
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gbm_bo_create(
    gbm: *mut gbm_device,
    width: u32,
    height: u32,
    format: u32,
    #[allow(dead_code)] flags: u32,
) -> *mut gbm_bo {
    let Some(dev) = dev_of(gbm) else {
        return core::ptr::null_mut();
    };
    match create_bo(&dev, width, height, format, flags) {
        Some(bo) => as_gbm(bo) as *mut gbm_bo,
        None => core::ptr::null_mut(),
    }
}

unsafe fn create_bo_with_mods(
    gbm: *mut gbm_device,
    width: u32,
    height: u32,
    format: u32,
    modifiers: *const u64,
    count: c_uint,
    #[allow(dead_code)] flags: u32,
) -> *mut gbm_bo {
    let Some(dev) = dev_of(gbm) else {
        return core::ptr::null_mut();
    };
    if modifiers.is_null() || count == 0 {
        return core::ptr::null_mut();
    }
    // First modifier this backend can honor (linear-only dumb buffers).
    let mut supported = false;
    for i in 0..count as usize {
        let m = *modifiers.add(i);
        if m == DRM_FORMAT_MOD_LINEAR || m == DRM_FORMAT_MOD_INVALID {
            supported = true;
            break;
        }
    }
    if !supported {
        return core::ptr::null_mut();
    }
    match create_bo(&dev, width, height, format, flags) {
        Some(bo) => as_gbm(bo) as *mut gbm_bo,
        None => core::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gbm_bo_create_with_modifiers(
    gbm: *mut gbm_device,
    width: u32,
    height: u32,
    format: u32,
    modifiers: *const u64,
    count: c_uint,
) -> *mut gbm_bo {
    // Legacy entry: use flags inferred from the caller's intent are not
    // available; RENDERING|SCANOUT is Mesa's historical implicit set here.
    create_bo_with_mods(
        gbm,
        width,
        height,
        format,
        modifiers,
        count,
        GBM_BO_USE_RENDERING | GBM_BO_USE_SCANOUT,
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gbm_bo_create_with_modifiers2(
    gbm: *mut gbm_device,
    width: u32,
    height: u32,
    format: u32,
    modifiers: *const u64,
    count: c_uint,
    #[allow(dead_code)] flags: u32,
) -> *mut gbm_bo {
    create_bo_with_mods(gbm, width, height, format, modifiers, count, flags)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gbm_bo_import(
    gbm: *mut gbm_device,
    type_: u32,
    buffer: *mut c_void,
    #[allow(dead_code)] flags: u32,
) -> *mut gbm_bo {
    let Some(dev) = dev_of(gbm) else {
        return core::ptr::null_mut();
    };
    if buffer.is_null() {
        return core::ptr::null_mut();
    }
    match type_ {
        GBM_BO_IMPORT_FD => {
            let d = &*(buffer as *const gbm_import_fd_data);
            match import_fd_bo(
                &dev,
                d.fd,
                d.width,
                d.height,
                d.stride,
                d.format,
                DRM_FORMAT_MOD_INVALID,
                flags,
            ) {
                Some(bo) => as_gbm(bo) as *mut gbm_bo,
                None => core::ptr::null_mut(),
            }
        }
        GBM_BO_IMPORT_FD_MODIFIER => {
            let d = &*(buffer as *const gbm_import_fd_modifier_data);
            if d.num_fds != 1 {
                return core::ptr::null_mut(); // multi-plane unsupported backend
            }
            match import_fd_bo(
                &dev,
                d.fds[0],
                d.width,
                d.height,
                d.strides[0] as u32,
                d.format,
                d.modifier,
                flags,
            ) {
                Some(bo) => as_gbm(bo) as *mut gbm_bo,
                None => core::ptr::null_mut(),
            }
        }
        // GBM_BO_IMPORT_WL_BUFFER / GBM_BO_IMPORT_EGL_IMAGE require the
        // respective server objects: MISSING — Mesa backends only implement
        // these via display-specific hooks.
        _ => core::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gbm_bo_map(
    bo: *mut gbm_bo,
    x: u32,
    y: u32,
    _width: u32,
    _height: u32,
    _flags: u32,
    stride: *mut u32,
    map_data: *mut *mut c_void,
) -> *mut c_void {
    let Some(b) = bo_of(bo) else {
        return core::ptr::null_mut();
    };
    let base: *mut u8 = match &b.storage {
        Storage::Dumb(g) => g.map_ptr(),
        Storage::Imported { ptr, .. } => *ptr,
    };
    if base.is_null() {
        return core::ptr::null_mut();
    }
    let bpp = bytes_per_pixel(b.format);
    if !stride.is_null() {
        *stride = b.stride;
    }
    if !map_data.is_null() {
        // Persistent map: the token is bookkeeping only, so unmap stays a
        // no-op on the mapping itself (the BO owns it until destroy).
        let token = Box::new(bo);
        *map_data = Box::into_raw(token) as *mut c_void;
    }
    base.add((y * b.stride + x * bpp) as usize) as *mut c_void
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gbm_bo_unmap(_bo: *mut gbm_bo, map_data: *mut c_void) {
    if !map_data.is_null() {
        drop(Box::from_raw(map_data as *mut *mut gbm_bo));
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gbm_bo_get_width(bo: *mut gbm_bo) -> u32 {
    bo_of(bo).map(|b| b.width).unwrap_or(0)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gbm_bo_get_height(bo: *mut gbm_bo) -> u32 {
    bo_of(bo).map(|b| b.height).unwrap_or(0)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gbm_bo_get_stride(bo: *mut gbm_bo) -> u32 {
    bo_of(bo).map(|b| b.stride).unwrap_or(0)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gbm_bo_get_stride_for_plane(bo: *mut gbm_bo, plane: c_int) -> u32 {
    if plane != 0 {
        return 0;
    }
    gbm_bo_get_stride(bo)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gbm_bo_get_format(bo: *mut gbm_bo) -> u32 {
    bo_of(bo).map(|b| b.format).unwrap_or(0)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gbm_bo_get_bpp(bo: *mut gbm_bo) -> u32 {
    bo_of(bo)
        .map(|b| bytes_per_pixel(b.format) * 8)
        .unwrap_or(0)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gbm_bo_get_offset(_bo: *mut gbm_bo, plane: c_int) -> u32 {
    // Single plane; other planes have no offset.
    let _ = plane;
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gbm_bo_get_device(bo: *mut gbm_bo) -> *mut gbm_device {
    match bo_of(bo) {
        Some(b) => Arc::as_ptr(&b.dev) as *mut gbm_device,
        None => core::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gbm_bo_get_handle(bo: *mut gbm_bo) -> gbm_bo_handle {
    let h = match bo_of(bo) {
        Some(b) => match &b.storage {
            Storage::Dumb(g) => g.handle,
            Storage::Imported { handle, .. } => *handle,
        },
        None => 0,
    };
    gbm_bo_handle { u32: h }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gbm_bo_get_handle_for_plane(
    bo: *mut gbm_bo,
    plane: c_int,
) -> gbm_bo_handle {
    if plane != 0 {
        return gbm_bo_handle { u64: 0 };
    }
    gbm_bo_get_handle(bo)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gbm_bo_get_fd(bo: *mut gbm_bo) -> c_int {
    let Some(b) = bo_of(bo) else {
        return -1;
    };
    let handle = match &b.storage {
        Storage::Dumb(g) => g.handle,
        Storage::Imported { handle, .. } => *handle,
    };
    b.dev.drm.prime_handle_to_fd(handle).unwrap_or(-1)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gbm_bo_get_fd_for_plane(bo: *mut gbm_bo, plane: c_int) -> c_int {
    if plane != 0 {
        return -1;
    }
    gbm_bo_get_fd(bo)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gbm_bo_get_modifier(bo: *mut gbm_bo) -> u64 {
    bo_of(bo)
        .map(|b| b.modifier)
        .unwrap_or(DRM_FORMAT_MOD_INVALID)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gbm_bo_get_plane_count(bo: *mut gbm_bo) -> c_int {
    if bo_of(bo).is_some() { 1 } else { -1 }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gbm_bo_write(bo: *mut gbm_bo, buf: *const c_void, count: usize) -> c_int {
    let Some(b) = bo_of(bo) else {
        return -1;
    };
    if buf.is_null() || count > b.size as usize {
        return -1;
    }
    let base: *mut u8 = match &b.storage {
        Storage::Dumb(g) => g.map_ptr(),
        Storage::Imported { ptr, .. } => *ptr,
    };
    if base.is_null() {
        return -1;
    }
    core::ptr::copy_nonoverlapping(buf as *const u8, base, count);
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gbm_bo_set_user_data(
    bo: *mut gbm_bo,
    data: *mut c_void,
    destroy_user_data: Option<extern "C" fn(*mut gbm_bo, *mut c_void)>,
) {
    if let Some(b) = bo_of(bo) {
        *b.user_data.lock() = (data, destroy_user_data);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gbm_bo_get_user_data(bo: *mut gbm_bo) -> *mut c_void {
    match bo_of(bo) {
        Some(b) => b.user_data.lock().0,
        None => core::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gbm_bo_destroy(bo: *mut gbm_bo) {
    if !bo.is_null() {
        let arc = Arc::from_raw(bo as *const BoInner);
        let (data, dtor) = *arc.user_data.lock();
        if let Some(f) = dtor {
            f(bo, data);
        }
        drop(arc);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gbm_surface_create(
    gbm: *mut gbm_device,
    width: u32,
    height: u32,
    format: u32,
    #[allow(dead_code)] flags: u32,
) -> *mut gbm_surface {
    let Some(dev) = dev_of(gbm) else {
        return core::ptr::null_mut();
    };
    if bytes_per_pixel(format) == 0 || width == 0 || height == 0 {
        return core::ptr::null_mut();
    }
    let s = Arc::new(SurfaceInner {
        dev,
        width,
        height,
        format,
        flags,
        free: Mutex::new(Vec::new()),
        busy: Mutex::new(Vec::new()),
    });
    as_gbm(s) as *mut gbm_surface
}

unsafe fn surface_create_with_mods(
    gbm: *mut gbm_device,
    width: u32,
    height: u32,
    format: u32,
    modifiers: *const u64,
    count: c_uint,
    #[allow(dead_code)] flags: u32,
) -> *mut gbm_surface {
    // Validate modifier support first (bo path does the same per allocation).
    let probe = gbm_bo_create_with_modifiers2(gbm, width, height, format, modifiers, count, flags);
    if probe.is_null() {
        return core::ptr::null_mut();
    }
    gbm_bo_destroy(probe);
    gbm_surface_create(gbm, width, height, format, flags)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gbm_surface_create_with_modifiers(
    gbm: *mut gbm_device,
    width: u32,
    height: u32,
    format: u32,
    modifiers: *const u64,
    count: c_uint,
) -> *mut gbm_surface {
    surface_create_with_mods(
        gbm,
        width,
        height,
        format,
        modifiers,
        count,
        GBM_BO_USE_RENDERING | GBM_BO_USE_SCANOUT,
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gbm_surface_create_with_modifiers2(
    gbm: *mut gbm_device,
    width: u32,
    height: u32,
    format: u32,
    modifiers: *const u64,
    count: c_uint,
    #[allow(dead_code)] flags: u32,
) -> *mut gbm_surface {
    surface_create_with_mods(gbm, width, height, format, modifiers, count, flags)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gbm_surface_lock_front_buffer(surface: *mut gbm_surface) -> *mut gbm_bo {
    let Some(s) = surf_of(surface) else {
        return core::ptr::null_mut();
    };
    let bo = s
        .free
        .lock()
        .pop()
        .or_else(|| create_bo(&s.dev, s.width, s.height, s.format, s.flags));
    match bo {
        Some(b) => {
            let raw = as_gbm(b.clone()) as *mut gbm_bo;
            s.busy.lock().push(b);
            raw
        }
        None => core::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gbm_surface_release_buffer(surface: *mut gbm_surface, bo: *mut gbm_bo) {
    let Some(s) = surf_of(surface) else {
        return;
    };
    let Some(b) = bo_of(bo) else {
        return;
    };
    let owned = {
        let mut busy = s.busy.lock();
        busy.iter()
            .position(|x| Arc::ptr_eq(x, &b))
            .map(|pos| busy.remove(pos))
    };
    if let Some(owned) = owned {
        s.free.lock().push(owned);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gbm_surface_has_free_buffers(surface: *mut gbm_surface) -> c_int {
    match surf_of(surface) {
        Some(s) => (!s.free.lock().is_empty()) as c_int,
        None => 0,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gbm_surface_destroy(surface: *mut gbm_surface) {
    if !surface.is_null() {
        drop(Arc::from_raw(surface as *const SurfaceInner));
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gbm_format_get_name(
    format: u32,
    desc: *mut gbm_format_name_desc,
) -> *mut c_char {
    if desc.is_null() {
        return core::ptr::null_mut();
    }
    let b = format.to_le_bytes();
    let name = &mut (*desc).name;
    for (dst, &byte) in name[..4].iter_mut().zip(&b) {
        *dst = byte as c_char;
    }
    name[4] = 0;
    (*desc).name.as_mut_ptr()
}

fn bo_of(b: *mut gbm_bo) -> Option<Arc<BoInner>> {
    if b.is_null() {
        return None;
    }
    let arc = unsafe { Arc::from_raw(b as *const BoInner) };
    let clone = arc.clone();
    core::mem::forget(arc);
    Some(clone)
}

fn surf_of(s: *mut gbm_surface) -> Option<Arc<SurfaceInner>> {
    if s.is_null() {
        return None;
    }
    let arc = unsafe { Arc::from_raw(s as *const SurfaceInner) };
    let clone = arc.clone();
    core::mem::forget(arc);
    Some(clone)
}

fn as_gbm<T>(arc: Arc<T>) -> *mut T {
    Arc::into_raw(arc) as *mut T
}

fn bytes_per_pixel(format: u32) -> u32 {
    match format {
        DRM_FORMAT_ARGB8888 | DRM_FORMAT_XRGB8888 => 4,
        DRM_FORMAT_RGB565 => 2,
        _ => 0, // unsupported
    }
}

/// Create a linear single-plane BO on `dev`. `use_flags` carries the gbm
/// `GBM_BO_USE_*` bits; `format` must be a supported fourcc.
fn create_bo(
    dev: &Arc<Device>,
    width: u32,
    height: u32,
    format: u32,
    use_flags: u32,
) -> Option<Arc<BoInner>> {
    let bpp = bytes_per_pixel(format);
    if bpp == 0 || width == 0 || height == 0 {
        return None;
    }
    let bo = dev.drm.create_dumb(width, height, bpp * 8).ok()?;
    let stride = bo.pitch;
    let size = bo.size;
    Some(Arc::new(BoInner {
        dev: dev.clone(),
        width,
        height,
        format,
        flags: use_flags,
        modifier: DRM_FORMAT_MOD_LINEAR,
        stride,
        size,
        storage: Storage::Dumb(bo),
        user_data: Mutex::new((core::ptr::null_mut(), None)),
    }))
}

/// Import a dmabuf fd as a single-plane linear BO. Takes ownership of one
/// reference on `fd` (caller dups per gbm contract? — Mesa does NOT close
/// the caller's fd; we PRIME-import and keep the caller's fd alive).
fn import_fd_bo(
    dev: &Arc<Device>,
    fd: c_int,
    width: u32,
    height: u32,
    stride: u32,
    format: u32,
    modifier: u64,
    use_flags: u32,
) -> Option<Arc<BoInner>> {
    if modifier != DRM_FORMAT_MOD_LINEAR && modifier != DRM_FORMAT_MOD_INVALID {
        return None; // only linear import supported by the dumb backend
    }
    let bpp = bytes_per_pixel(format);
    if bpp == 0 || width == 0 || height == 0 || stride < width * bpp {
        return None;
    }
    let handle = dev.drm.prime_fd_to_handle(fd).ok()?;
    let len = (stride * height) as usize;
    let ptr = unsafe {
        mmap_impl(
            core::ptr::null_mut(),
            len,
            PROT_READ | PROT_WRITE,
            MAP_SHARED,
            dev.drm_fd(),
            dumb_map_offset(dev, handle)? as i64,
        )
    };
    if ptr.is_null() {
        let _ = dev.drm.gem_close(handle);
        return None;
    }
    Some(Arc::new(BoInner {
        dev: dev.clone(),
        width,
        height,
        format,
        flags: use_flags,
        modifier: DRM_FORMAT_MOD_LINEAR,
        stride,
        size: len as u64,
        storage: Storage::Imported {
            handle,
            ptr: ptr as *mut u8,
            len,
        },
        user_data: Mutex::new((core::ptr::null_mut(), None)),
    }))
}

// ============================================================================
// libc mmap/munmap (same hand-declared pattern as crates/drm)
// ============================================================================

unsafe extern "C" {
    fn mmap(
        addr: *mut c_void,
        len: usize,
        prot: c_int,
        flags: c_int,
        fd: c_int,
        offset: i64,
    ) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> c_int;
}

const PROT_READ: c_int = 1;
const PROT_WRITE: c_int = 2;
const MAP_SHARED: c_int = 1;

unsafe fn mmap_impl(
    addr: *mut c_void,
    len: usize,
    prot: c_int,
    flags: c_int,
    fd: c_int,
    offset: i64,
) -> *mut c_void {
    let p = mmap(addr, len, prot, flags, fd, offset);
    if p as isize == -1 {
        core::ptr::null_mut()
    } else {
        p
    }
}

unsafe fn munmap_impl(addr: *mut c_void, len: usize) {
    munmap(addr, len);
}

/// `DRM_IOCTL_MODE_MAP_DUMB` for an arbitrary GEM handle. vantage-drm only
/// exposes mapping through `create_dumb`; import needs the raw ioctl, so we
/// keep a tiny local version.
fn dumb_map_offset(dev: &Arc<Device>, handle: u32) -> Option<u64> {
    #[repr(C)]
    struct ModeMapDumb {
        handle: u32,
        pad: u32,
        offset: u64,
    }
    unsafe extern "C" {
        fn ioctl(fd: c_int, request: c_ulong, argp: *mut c_void) -> c_int;
    }
    const DRM_IOCTL_MODE_MAP_DUMB: c_ulong = 0xc010_64b3; // IOWR('d', 0xb3, 16)
    let mut m = ModeMapDumb {
        handle,
        pad: 0,
        offset: 0,
    };
    let r = unsafe {
        ioctl(
            dev.drm_fd(),
            DRM_IOCTL_MODE_MAP_DUMB,
            &mut m as *mut _ as *mut c_void,
        )
    };
    if r < 0 { None } else { Some(m.offset) }
}

impl Device {
    fn drm_fd(&self) -> c_int {
        self.drm.fd()
    }
}

// ============================================================================
// tests — over the real DRM node when one is present
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "C" {
        fn close(fd: c_int) -> c_int;
    }

    /// Proves the C-ABI surface end-to-end on real hardware: create device
    /// from the shared node fd, allocate a BO, write + map-read pixels,
    /// export a PRIME fd, import it back, surface ring lock/release.
    #[test]
    fn real_device_c_abi_roundtrip() {
        let dev = match DrmDevice::open_best() {
            Ok(d) => d,
            Err(_) => return, // no DRM hardware in this environment
        };
        let fd = dev.fd();
        core::mem::forget(dev); // gbm_create_device borrows the fd

        let gbm = unsafe { gbm_create_device(fd) };
        assert!(!gbm.is_null(), "gbm_create_device");
        assert_eq!(
            unsafe { gbm_device_is_format_supported(gbm, DRM_FORMAT_XRGB8888, 0) },
            1
        );
        assert_eq!(
            unsafe { gbm_device_is_format_supported(gbm, 0x12345678, 0) },
            0
        );

        // BO allocate + write + map readback.
        let bo = unsafe { gbm_bo_create(gbm, 16, 16, DRM_FORMAT_XRGB8888, GBM_BO_USE_RENDERING) };
        assert!(!bo.is_null(), "gbm_bo_create");
        unsafe {
            assert_eq!(gbm_bo_get_width(bo), 16);
            assert_eq!(gbm_bo_get_height(bo), 16);
            assert_eq!(gbm_bo_get_bpp(bo), 32);
            assert_eq!(gbm_bo_get_modifier(bo), DRM_FORMAT_MOD_LINEAR);
            assert_eq!(
                gbm_bo_get_handle(bo).u32,
                gbm_bo_get_handle_for_plane(bo, 0).u32
            );
        }

        // gbm_bo_write + map readback.
        let px: [u32; 16] = core::array::from_fn(|i| 0xFF00_0000u32 | i as u32);
        unsafe {
            assert_eq!(
                gbm_bo_write(bo, px.as_ptr() as *const c_void, px.len() * 4),
                0
            );
            let mut stride = 0u32;
            let mut map_data: *mut c_void = core::ptr::null_mut();
            let map = gbm_bo_map(
                bo,
                0,
                0,
                16,
                16,
                GBM_BO_TRANSFER_READ,
                &mut stride,
                &mut map_data,
            );
            assert!(!map.is_null(), "gbm_bo_map");
            assert!(stride >= 64);
            let row = core::slice::from_raw_parts(map as *const u32, 16);
            assert_eq!(row[7], 0xFF00_0007);
            gbm_bo_unmap(bo, map_data);
        }

        // PRIME export -> import roundtrip.
        let fd_out = unsafe { gbm_bo_get_fd(bo) };
        assert!(fd_out >= 0, "gbm_bo_get_fd");
        let imp = gbm_import_fd_data {
            fd: fd_out,
            width: 16,
            height: 16,
            stride: unsafe { gbm_bo_get_stride(bo) },
            format: DRM_FORMAT_XRGB8888,
        };
        let bo2 =
            unsafe { gbm_bo_import(gbm, GBM_BO_IMPORT_FD, &imp as *const _ as *mut c_void, 0) };
        assert!(!bo2.is_null(), "gbm_bo_import(FD)");
        unsafe {
            assert_eq!(gbm_bo_get_handle(bo2).u32, gbm_bo_get_handle(bo).u32);
            close(fd_out);
            gbm_bo_destroy(bo2);
            gbm_bo_destroy(bo);
        }

        // Surface ring semantics.
        let surf =
            unsafe { gbm_surface_create(gbm, 8, 8, DRM_FORMAT_XRGB8888, GBM_BO_USE_RENDERING) };
        assert!(!surf.is_null(), "gbm_surface_create");
        unsafe {
            let b1 = gbm_surface_lock_front_buffer(surf);
            let b2 = gbm_surface_lock_front_buffer(surf);
            assert!(!b1.is_null() && !b2.is_null());
            assert_ne!(b1, b2, "distinct front buffers");
            assert_eq!(gbm_surface_has_free_buffers(surf), 0);
            gbm_surface_release_buffer(surf, b1);
            assert_eq!(gbm_surface_has_free_buffers(surf), 1);
            let b3 = gbm_surface_lock_front_buffer(surf);
            assert_eq!(b3, b1, "released buffer is reused");
            gbm_surface_release_buffer(surf, b3);
            gbm_surface_release_buffer(surf, b2);
            gbm_surface_destroy(surf);
            gbm_device_destroy(gbm);
        }
    }
}
