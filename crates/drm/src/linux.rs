//! Vantage DRM — Linux DRM client substrate (`libdrm` replacement).
//!
//! Talks the kernel DRM uapi directly: render-node discovery, `DRM_IOCTL_*`
//! dumb-buffer allocation/mapping, and PRIME fd (`dma-buf`) import/export.
//! No `libc` crate dependency: the handful of libc symbols used are declared
//! `extern "C"` (the final cdylib links the C runtime either way), matching
//! the workspace's hand-declared X11 bindings in `crates/egl`.
//!
//! Scope is deliberately the vantage DRI3/GBM substrate for now; KMS
//! framebuffer/flip ioctls arrive with the compositor-facing path. AMD GPU
//! (amdgpu/radeon) is the target driver class: `DrmDevice::open_best` scans
//! render and card nodes, prefers a matching sysfs driver binding, and
//! verifies dumb-buffer capability before returning.
//!
//! Threading: the device fd is shareable; ioctls are individually atomic at
//! the kernel layer. `GemBuffer` exposes raw mapped memory and is `Send` (the
//! EGL layer already serializes surface access behind its display mutex).

use core::ffi::{c_char, c_int, c_ulong, c_void};

// ============================================================================
// libc externs (see crate docs: no libc crate by design)
// ============================================================================

unsafe extern "C" {
    pub(crate) fn open(path: *const c_char, flags: c_int, ...) -> c_int;
    pub(crate) fn close(fd: c_int) -> c_int;
    pub(crate) fn readlink(path: *const c_char, buf: *mut c_char, bufsiz: usize) -> isize;
    pub(crate) fn ioctl(fd: c_int, request: c_ulong, argp: *mut c_void) -> c_int;
    pub(crate) fn mmap(
        addr: *mut c_void,
        len: usize,
        prot: c_int,
        flags: c_int,
        fd: c_int,
        offset: i64,
    ) -> *mut c_void;
    pub(crate) fn munmap(addr: *mut c_void, len: usize) -> c_int;
    pub(crate) fn __errno_location() -> *mut c_int;
}

const O_RDWR: c_int = 0o2;
const O_CLOEXEC: c_int = 0o2000000;
pub(crate) const PROT_READ: c_int = 1;
pub(crate) const PROT_WRITE: c_int = 2;
pub(crate) const MAP_SHARED: c_int = 1;

pub(crate) fn errno() -> i32 {
    unsafe { *__errno_location() }
}

// ============================================================================
// ioctl encoding (kernel `_IOC` layout, type 'd' = DRM)
// ============================================================================

const _IOC_WRITE: u64 = 1;
const _IOC_READ: u64 = 2;

const fn ioc(dir: u64, nr: u64, typ: u8, size: u64) -> c_ulong {
    ((dir << 30) | ((size & 0x3fff) << 16) | ((typ as u64) << 8) | nr) as c_ulong
}

pub(crate) const fn iowr(nr: u64, typ: u8, size: u64) -> c_ulong {
    ioc(_IOC_READ | _IOC_WRITE, nr, typ, size)
}

pub(crate) const fn iow(nr: u64, typ: u8, size: u64) -> c_ulong {
    ioc(_IOC_WRITE, nr, typ, size)
}

/// `DRM_IOCTL_VERSION` (`struct drm_version`, 64 bytes on LP64).
pub const DRM_IOCTL_VERSION: c_ulong = iowr(0x00, b'd', 64);
/// `DRM_IOCTL_GEM_CLOSE` (`struct drm_gem_close`).
pub const DRM_IOCTL_GEM_CLOSE: c_ulong = iow(0x09, b'd', 8);
/// `DRM_IOCTL_PRIME_HANDLE_TO_FD` (`struct drm_prime_handle`).
pub const DRM_IOCTL_PRIME_HANDLE_TO_FD: c_ulong = iowr(0x2d, b'd', 12);
/// `DRM_IOCTL_PRIME_FD_TO_HANDLE` (`struct drm_prime_handle`).
pub const DRM_IOCTL_PRIME_FD_TO_HANDLE: c_ulong = iowr(0x2e, b'd', 12);
/// `DRM_IOCTL_MODE_CREATE_DUMB` (`struct drm_mode_create_dumb`).
pub const DRM_IOCTL_MODE_CREATE_DUMB: c_ulong = iowr(0xb2, b'd', 32);
/// `DRM_IOCTL_MODE_MAP_DUMB` (`struct drm_mode_map_dumb`).
pub const DRM_IOCTL_MODE_MAP_DUMB: c_ulong = iowr(0xb3, b'd', 16);
/// `DRM_IOCTL_MODE_DESTROY_DUMB` (`struct drm_mode_destroy_dumb`).
pub const DRM_IOCTL_MODE_DESTROY_DUMB: c_ulong = iowr(0xb4, b'd', 4);

// ============================================================================
// uapi structs — layouts match /usr/include/drm/drm{,mode}.h exactly
// ============================================================================

#[repr(C)]
struct DrmVersion {
    major: i32,
    minor: i32,
    patchlevel: i32,
    name_len: usize,
    name: *mut c_char,
    date_len: usize,
    date: *mut c_char,
    desc_len: usize,
    desc: *mut c_char,
}

#[repr(C)]
pub(crate) struct DrmGemClose {
    pub(crate) handle: u32,
    pub(crate) pad: u32,
}

/// PRIME handle/fd exchange. `fd` is out for HANDLE_TO_FD, in for FD_TO_HANDLE.
#[repr(C)]
struct DrmPrimeHandle {
    handle: u32,
    flags: u32,
    fd: i32,
}

#[repr(C)]
struct ModeCreateDumb {
    height: u32,
    width: u32,
    bpp: u32,
    flags: u32,
    // response:
    handle: u32,
    pitch: u32,
    size: u64,
}

#[repr(C)]
struct ModeMapDumb {
    handle: u32,
    pad: u32,
    offset: u64,
}

#[repr(C)]
struct ModeDestroyDumb {
    handle: u32,
}

// ============================================================================
// FourCC / modifiers (from drm_fourcc.h, namespaced `drm::`)
// ============================================================================

/// `fourcc_code(a, b, c, d)` from drm_fourcc.h.
pub const fn fourcc(a: u8, b: u8, c: u8, d: u8) -> u32 {
    (a as u32) | ((b as u32) << 8) | ((c as u32) << 16) | ((d as u32) << 24)
}

/// `DRM_FORMAT_XRGB8888` — 32bpp, what X11 TrueColor depth-24 uses.
pub const DRM_FORMAT_XRGB8888: u32 = fourcc(b'X', b'R', b'2', b'4');
/// `DRM_FORMAT_ARGB8888`.
pub const DRM_FORMAT_ARGB8888: u32 = fourcc(b'A', b'R', b'2', b'4');
/// `DRM_FORMAT_XBGR8888`.
pub const DRM_FORMAT_XBGR8888: u32 = fourcc(b'X', b'B', b'2', b'4');
/// `DRM_FORMAT_ABGR8888`.
pub const DRM_FORMAT_ABGR8888: u32 = fourcc(b'A', b'B', b'2', b'4');
/// `DRM_FORMAT_RGB565`.
pub const DRM_FORMAT_RGB565: u32 = fourcc(b'R', b'G', b'1', b'6');

/// `DRM_FORMAT_MOD_LINEAR` (`fourcc_mod_code(NONE, 0)`).
pub const DRM_FORMAT_MOD_LINEAR: u64 = 0;
/// `DRM_FORMAT_MOD_INVALID` (fourcc_mod_code(NONE, DRM_FORMAT_RESERVED)).
pub const DRM_FORMAT_MOD_INVALID: u64 = u64::MAX;

/// `DRM_IOCTL_PRIME_HANDLE_TO_FD` flag: `O_CLOEXEC` — the kernel validates
/// this field against the O_* flags (glibc value 0o2000000), NOT a namespaced
/// bit. Anything else is rejected with EINVAL.
pub const DRM_PRIME_FD_CLOEXEC: u32 = 0o2000000;

// ============================================================================
// Errors
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// syscall failed with this errno
    Io(i32),
    /// no usable DRM render node found
    NoDevice,
    /// device is not backed by an accepted driver (amdgpu/radeon)
    WrongDriver(&'static [u8]),
    /// the buffer object is not CPU-mapped
    NotMapped,
}

// ============================================================================
// Device
// ============================================================================

/// An open DRM node (usually `/dev/dri/renderD128`).
pub struct DrmDevice {
    fd: c_int,
}

const RENDER_NODE_FIRST: u32 = 128;
const RENDER_NODE_LAST: u32 = 160;

impl DrmDevice {
    /// Opens `path` (NUL-terminated C string) as a DRM node.
    pub fn open_path(path: &[u8]) -> Result<Self, Error> {
        if path.is_empty() || *path.last().unwrap() != 0 {
            return Err(Error::Io(22)); // EINVAL
        }
        let fd = unsafe { open(path.as_ptr() as *const c_char, O_RDWR | O_CLOEXEC, 0) };
        if fd < 0 {
            return Err(Error::Io(errno()));
        }
        Ok(Self { fd })
    }

    /// Wraps an already-open DRM node fd. The device takes ownership:
    /// dropping it closes the fd.
    ///
    /// # Safety
    /// `fd` must be a valid open file descriptor for a DRM node, and not be
    /// closed or owned elsewhere.
    pub const unsafe fn from_fd(fd: c_int) -> Self {
        Self { fd }
    }

    /// The underlying open fd (borrowed; do not close).
    pub const fn fd(&self) -> c_int {
        self.fd
    }

    /// Opens the first usable DRM node with working dumb-buffer ioctls,
    /// preferring nodes whose sysfs driver binding is `amdgpu`/`radeon`.
    ///
    /// Scans render nodes first, then card nodes: some kernel/driver
    /// combinations deny `MODE_CREATE_DUMB` on render nodes (observed:
    /// amdgpu on CachyOS LTS grants dumb ioctls only on card nodes), and
    /// both the GBM and DRI3 paths require dumb buffers.
    pub fn open_best() -> Result<Self, Error> {
        // First pass: amd nodes with verified dumb capability.
        let mut generic: Option<Self> = None;
        for prefix in [b"/dev/dri/renderD".as_slice(), b"/dev/dri/card".as_slice()] {
            let (first, last) = if prefix.last() == Some(&b'D') {
                (RENDER_NODE_FIRST, RENDER_NODE_LAST) // renderD128..renderD159
            } else {
                (0u32, 16u32) // card0..card15
            };
            for n in first..last {
                let mut path = alloc::vec::Vec::new();
                push_ascii(&mut path, prefix);
                push_u32(&mut path, n);
                path.push(0);
                let Ok(dev) = Self::open_path(&path) else {
                    continue;
                };
                if dev.verify_dumb().is_err() {
                    continue;
                }
                let amd = matches!(
                    dev.sysfs_driver_name().as_deref(),
                    Some(b"amdgpu") | Some(b"radeon")
                );
                if amd {
                    return Ok(dev);
                }
                if generic.is_none() {
                    generic = Some(dev);
                }
            }
        }
        generic.ok_or(Error::NoDevice)
    }

    /// Opens the first accessible render node (`/dev/dri/renderD128..159`),
    /// preferring AMDGPU. Render nodes are unprivileged and designed for
    /// 3D rendering, compute, and hardware command submission.
    pub fn open_render() -> Result<Self, Error> {
        let mut generic: Option<Self> = None;
        for n in RENDER_NODE_FIRST..RENDER_NODE_LAST {
            let mut path = alloc::vec::Vec::new();
            push_ascii(&mut path, b"/dev/dri/renderD");
            push_u32(&mut path, n);
            path.push(0);
            let Ok(dev) = Self::open_path(&path) else {
                continue;
            };
            let amd = matches!(
                dev.sysfs_driver_name().as_deref(),
                Some(b"amdgpu") | Some(b"radeon")
            );
            if amd {
                return Ok(dev);
            }
            if generic.is_none() {
                generic = Some(dev);
            }
        }
        generic.ok_or(Error::NoDevice)
    }

    /// Allocates and immediately frees a 64x64x32 buffer to prove the node
    /// permits dumb ioctls.
    fn verify_dumb(&self) -> Result<(), Error> {
        let bo = self.create_dumb(64, 64, 32)?;
        drop(bo);
        Ok(())
    }

    /// Reads `/sys/class/drm/<basename>/device/driver` symlink target and
    /// returns the driver basename (e.g. `amdgpu`). `None` when unavailable.
    pub fn sysfs_driver_name(&self) -> Option<alloc::vec::Vec<u8>> {
        // fd-based procfs path avoids depending on the original path string.
        let mut link = alloc::vec::Vec::new();
        push_ascii(&mut link, b"/proc/self/fd/");
        push_i32(&mut link, self.fd);
        link.push(0);

        let mut buf = [0u8; 256];
        let n = unsafe {
            readlink(
                link.as_ptr() as *const c_char,
                buf.as_mut_ptr() as *mut c_char,
                buf.len(),
            )
        };
        if n <= 0 {
            return None;
        }
        // <real path>/device/driver — walk back to the last '/' for the name.
        let node = &buf[..n as usize];
        let slash = node.iter().rposition(|&b| b == b'/')?;
        let mut sys = alloc::vec::Vec::new();
        push_ascii(&mut sys, b"/sys/class/drm/");
        sys.extend_from_slice(&node[slash + 1..]);
        push_ascii(&mut sys, b"/device/driver");
        sys.push(0);

        let mut tgt = [0u8; 256];
        let m = unsafe {
            readlink(
                sys.as_ptr() as *const c_char,
                tgt.as_mut_ptr() as *mut c_char,
                tgt.len(),
            )
        };
        if m <= 0 {
            return None;
        }
        let tgt = &tgt[..m as usize];
        let slash = tgt.iter().rposition(|&b| b == b'/')?;
        Some(alloc::vec::Vec::from(&tgt[slash + 1..]))
    }

    /// `(major, minor, patchlevel)` of the kernel driver.
    pub fn version(&self) -> Result<(i32, i32, i32), Error> {
        let mut name = [0u8; 64];
        let mut date = [0u8; 32];
        let mut desc = [0u8; 64];
        let mut v = DrmVersion {
            major: 0,
            minor: 0,
            patchlevel: 0,
            name_len: name.len(),
            name: name.as_mut_ptr() as *mut c_char,
            date_len: date.len(),
            date: date.as_mut_ptr() as *mut c_char,
            desc_len: desc.len(),
            desc: desc.as_mut_ptr() as *mut c_char,
        };
        let r = unsafe { ioctl(self.fd, DRM_IOCTL_VERSION, &mut v as *mut _ as *mut c_void) };
        if r < 0 {
            return Err(Error::Io(errno()));
        }
        Ok((v.major, v.minor, v.patchlevel))
    }

    /// Allocates a linear dumb buffer, maps it, and returns the owned object.
    pub fn create_dumb(&self, width: u32, height: u32, bpp: u32) -> Result<GemBuffer, Error> {
        let mut create = ModeCreateDumb {
            height,
            width,
            bpp,
            flags: 0,
            handle: 0,
            pitch: 0,
            size: 0,
        };
        if unsafe {
            ioctl(
                self.fd,
                DRM_IOCTL_MODE_CREATE_DUMB,
                &mut create as *mut _ as *mut c_void,
            )
        } < 0
        {
            return Err(Error::Io(errno()));
        }

        let mut map = ModeMapDumb {
            handle: create.handle,
            pad: 0,
            offset: 0,
        };
        if unsafe {
            ioctl(
                self.fd,
                DRM_IOCTL_MODE_MAP_DUMB,
                &mut map as *mut _ as *mut c_void,
            )
        } < 0
        {
            let _ = self.destroy_dumb(create.handle);
            return Err(Error::Io(errno()));
        }

        let len = create.size as usize;
        let ptr = unsafe {
            mmap(
                core::ptr::null_mut(),
                len,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                self.fd,
                map.offset as i64,
            )
        };
        if ptr as isize == -1 {
            let _ = self.destroy_dumb(create.handle);
            return Err(Error::Io(errno()));
        }

        Ok(GemBuffer {
            dev_fd: self.fd,
            handle: create.handle,
            pitch: create.pitch,
            size: create.size,
            ptr: ptr as *mut u8,
            len,
        })
    }

    /// Exports a GEM handle as a dmabuf fd (`PRIME_HANDLE_TO_FD`).
    pub fn prime_handle_to_fd(&self, handle: u32) -> Result<i32, Error> {
        let mut p = DrmPrimeHandle {
            handle,
            flags: DRM_PRIME_FD_CLOEXEC,
            fd: -1,
        };
        if unsafe {
            ioctl(
                self.fd,
                DRM_IOCTL_PRIME_HANDLE_TO_FD,
                &mut p as *mut _ as *mut c_void,
            )
        } < 0
        {
            return Err(Error::Io(errno()));
        }
        Ok(p.fd)
    }

    /// Imports a dmabuf fd as a GEM handle (`PRIME_FD_TO_HANDLE`).
    pub fn prime_fd_to_handle(&self, fd: i32) -> Result<u32, Error> {
        let mut p = DrmPrimeHandle {
            handle: 0,
            flags: 0,
            fd,
        };
        if unsafe {
            ioctl(
                self.fd,
                DRM_IOCTL_PRIME_FD_TO_HANDLE,
                &mut p as *mut _ as *mut c_void,
            )
        } < 0
        {
            return Err(Error::Io(errno()));
        }
        Ok(p.handle)
    }

    /// Closes a GEM handle. Buffers obtained from [`Self::create_dumb`] do
    /// this themselves on drop.
    pub fn gem_close(&self, handle: u32) -> Result<(), Error> {
        let mut c = DrmGemClose { handle, pad: 0 };
        if unsafe {
            ioctl(
                self.fd,
                DRM_IOCTL_GEM_CLOSE,
                &mut c as *mut _ as *mut c_void,
            )
        } < 0
        {
            return Err(Error::Io(errno()));
        }
        Ok(())
    }

    fn destroy_dumb(&self, handle: u32) -> Result<(), Error> {
        let mut d = ModeDestroyDumb { handle };
        if unsafe {
            ioctl(
                self.fd,
                DRM_IOCTL_MODE_DESTROY_DUMB,
                &mut d as *mut _ as *mut c_void,
            )
        } < 0
        {
            return Err(Error::Io(errno()));
        }
        Ok(())
    }
}

impl Drop for DrmDevice {
    fn drop(&mut self) {
        unsafe { close(self.fd) };
    }
}

// ============================================================================
// Buffer object
// ============================================================================

/// A mapped dumb GEM buffer. Dropping unmaps and destroys the dumb object;
/// `DESTROY_DUMB` releases the GEM handle with it, so no `GEM_CLOSE` follows.
pub struct GemBuffer {
    dev_fd: c_int,
    pub handle: u32,
    /// Bytes per row, as reported by the kernel.
    pub pitch: u32,
    /// Total buffer size in bytes.
    pub size: u64,
    ptr: *mut u8,
    len: usize,
}

// The mapping is plain process memory; the handle bookkeeping is confined to
// Drop. Cross-thread sharing is the caller's (EGL display mutex) concern.
unsafe impl Send for GemBuffer {}
unsafe impl Sync for GemBuffer {}

impl GemBuffer {
    /// Full mapped contents.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Raw mapped pointer, or `None` when mapping failed at creation.
    pub fn map_ptr(&self) -> *mut u8 {
        self.ptr
    }
}

impl Drop for GemBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { munmap(self.ptr as *mut c_void, self.len) };
        }
        let _ = unsafe {
            ioctl(
                self.dev_fd,
                DRM_IOCTL_MODE_DESTROY_DUMB,
                &mut ModeDestroyDumb {
                    handle: self.handle,
                } as *mut _ as *mut c_void,
            )
        };
    }
}

// ============================================================================
// small ascii helpers (no itoa dependency; paths are tiny and cold)
// ============================================================================

fn push_ascii(v: &mut alloc::vec::Vec<u8>, s: &[u8]) {
    v.extend_from_slice(s);
}

fn push_u32(v: &mut alloc::vec::Vec<u8>, mut n: u32) {
    let mut tmp = [0u8; 10];
    let mut i = tmp.len();
    loop {
        i -= 1;
        tmp[i] = b'0' + (n % 10) as u8;
        n /= 10;
        if n == 0 {
            break;
        }
    }
    v.extend_from_slice(&tmp[i..]);
}

fn push_i32(v: &mut alloc::vec::Vec<u8>, n: i32) {
    if n < 0 {
        v.push(b'-');
        push_u32(v, (n as i64).unsigned_abs() as u32);
    } else {
        push_u32(v, n as u32);
    }
}

// ============================================================================
// tests — layout/ioctl pins always; device tests run when a node exists
// ============================================================================

#[cfg(test)]
extern crate std;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::amdgpu::*;

    const fn expect_ioctl(dir: u64, nr: u64, size: u64) -> u64 {
        (dir << 30) | (size << 16) | (0x64 << 8) | nr
    }

    #[test]
    fn ioctl_numbers_match_kernel_encoding() {
        assert_eq!(DRM_IOCTL_VERSION, expect_ioctl(3, 0x00, 64));
        assert_eq!(DRM_IOCTL_GEM_CLOSE, expect_ioctl(1, 0x09, 8));
        assert_eq!(DRM_IOCTL_PRIME_HANDLE_TO_FD, expect_ioctl(3, 0x2d, 12));
        assert_eq!(DRM_IOCTL_PRIME_FD_TO_HANDLE, expect_ioctl(3, 0x2e, 12));
        assert_eq!(DRM_IOCTL_MODE_CREATE_DUMB, expect_ioctl(3, 0xb2, 32));
        assert_eq!(DRM_IOCTL_MODE_MAP_DUMB, expect_ioctl(3, 0xb3, 16));
        assert_eq!(DRM_IOCTL_MODE_DESTROY_DUMB, expect_ioctl(3, 0xb4, 4));
    }

    #[test]
    fn struct_layouts_match_uapi() {
        assert_eq!(core::mem::size_of::<DrmVersion>(), 64);
        assert_eq!(core::mem::size_of::<DrmGemClose>(), 8);
        assert_eq!(core::mem::size_of::<DrmPrimeHandle>(), 12);
        assert_eq!(core::mem::size_of::<ModeCreateDumb>(), 32);
        assert_eq!(core::mem::size_of::<ModeMapDumb>(), 16);
        assert_eq!(core::mem::size_of::<ModeDestroyDumb>(), 4);
    }

    #[test]
    fn fourcc_codes() {
        assert_eq!(DRM_FORMAT_XRGB8888, 0x34325258);
        assert_eq!(DRM_FORMAT_ARGB8888, 0x34325241);
        assert_eq!(DRM_FORMAT_MOD_LINEAR, 0);
        assert_eq!(DRM_FORMAT_MOD_INVALID, u64::MAX);
    }

    /// Exercises the real driver when a render node is present (skipped on
    /// machines without one). Proves: open, sysfs driver readback, dumb
    /// alloc/map, PRIME export + reimport, teardown.
    #[test]
    fn real_device_dumb_prime_roundtrip() {
        let dev = match DrmDevice::open_best() {
            Ok(d) => d,
            Err(_) => return, // no DRM hardware in this environment
        };
        let driver = dev.sysfs_driver_name().expect("sysfs driver name");
        std::assert!(
            driver == b"amdgpu".to_vec() || driver == b"radeon".to_vec() || !driver.is_empty()
        );
        let _ = dev.version().expect("DRM_IOCTL_VERSION");

        let mut bo = dev.create_dumb(64, 64, 32).expect("create_dumb");
        let (h, pitch, size) = (bo.handle, bo.pitch, bo.size);
        assert_eq!(pitch, 256); // 64 px * 4 B, linear
        assert_eq!(size, 256 * 64);
        {
            let map = bo.as_mut_slice();
            map[..4].copy_from_slice(&0x11223344u32.to_le_bytes());
            map[256 * 63..256 * 63 + 4].copy_from_slice(&0x55667788u32.to_le_bytes());
        }

        let fd = dev.prime_handle_to_fd(h).expect("handle_to_fd");
        assert!(fd >= 0);
        let h2 = dev.prime_fd_to_handle(fd).expect("fd_to_handle");
        // Same underlying object: kernel gives back the original handle for a
        // locally-exported dmabuf.
        assert_eq!(h2, h);

        dev.gem_close(h2).expect("gem_close reimported handle");
        unsafe { close(fd) };
        drop(bo);
    }

    #[test]
    fn real_device_amdgpu_queries_and_alloc() {
        let dev = match DrmDevice::open_render() {
            Ok(d) => d,
            Err(_) => return,
        };
        if dev.sysfs_driver_name().as_deref() != Some(b"amdgpu") {
            return; // Only run on AMDGPU hardware
        }

        // Queries + BO alloc only: no command submission, safe on any GPU.
        let amd = AmdgpuDevice::new(dev).expect("amdgpu device");
        let ip = amd.gfx_ip;
        std::println!(
            "AMD GFX IP: {}.{}",
            ip.hw_ip_version_major,
            ip.hw_ip_version_minor
        );
        assert!(ip.hw_ip_version_major >= 9, "expected GFX9 or newer");
        assert!(ip.ib_start_alignment > 0);
        assert!(amd.info.virtual_address_offset > 0);
        let vram_gtt = amd.query_vram_gtt().expect("query vram gtt");
        assert!(vram_gtt.gtt_size > 0);

        let ctx_id = amd.create_context().expect("create context");
        assert_eq!(
            amd.query_context_state(ctx_id).expect("ctx state") & AMDGPU_CTX_QUERY2_FLAGS_GUILTY,
            0
        );

        let va = amd.info.virtual_address_offset.max(0x100_0000);
        let mut bo = amd
            .alloc_bo(
                4096,
                4096,
                AMDGPU_GEM_DOMAIN_GTT,
                AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED,
                va,
                true,
            )
            .expect("alloc and map BO");
        assert_eq!(bo.gpu_va, va);
        {
            let slice = bo.as_slice_mut().expect("cpu slice");
            slice[0..4].copy_from_slice(&0xdeadbeefu32.to_le_bytes());
            slice[4092..4096].copy_from_slice(&0xcafebabau32.to_le_bytes());
        }
        let slice = bo.as_slice().expect("cpu slice readback");
        assert_eq!(&slice[0..4], &0xdeadbeefu32.to_le_bytes());
        assert_eq!(&slice[4092..4096], &0xcafebabau32.to_le_bytes());
        drop(bo);
        amd.destroy_context(ctx_id).expect("destroy context");
    }
}
