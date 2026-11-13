//! Host W^X executable memory allocation, arena bump-allocator, and symbol lookup.

use alloc::ffi::CString;
use core::{fmt, mem, ptr::NonNull};
use volt_platform::{ExecImage, ExecMemory};

#[derive(Debug)]
pub enum ExecError {
    EmptyCode,
    Mapping(std::io::Error),
    Protection(std::io::Error),
    UnsupportedHost,
}

impl fmt::Display for ExecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCode => f.write_str("cannot map empty machine code"),
            Self::Mapping(e) => write!(f, "executable allocation: {e}"),
            Self::Protection(e) => write!(f, "executable protection: {e}"),
            Self::UnsupportedHost => f.write_str("executable mapping requires Linux or macOS"),
        }
    }
}

impl core::error::Error for ExecError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Mapping(e) | Self::Protection(e) => Some(e),
            _ => None,
        }
    }
}

/// Owns immutable executable pages. Never writable and executable simultaneously.
pub struct HostExecImage {
    address: NonNull<u8>,
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    mapped_len: usize,
    code_len: usize,
    /// Set when the code lives in a shared arena chunk, which owns the mapping.
    #[cfg(target_os = "linux")]
    chunk: Option<alloc::sync::Arc<arena::Chunk>>,
}

// SAFETY: the code is immutable after construction (the writable view of an
// arena chunk is never exposed) and the mapping lives as long as the value, so
// it can be shared and run from any thread.
unsafe impl Send for HostExecImage {}
unsafe impl Sync for HostExecImage {}

impl ExecImage for HostExecImage {
    fn as_ptr(&self) -> *const u8 {
        self.address.as_ptr()
    }

    fn len(&self) -> usize {
        self.code_len
    }
}

impl HostExecImage {
    /// Construct from an already-mapped RX region (used by module mapper).
    ///
    /// # Safety
    /// `address` must remain mapped and readable/executable for `mapped_len` bytes
    /// (with `code_len <= mapped_len`) until unmapped by `Drop`.
    #[cfg(target_os = "linux")]
    pub unsafe fn from_raw_linux(address: NonNull<u8>, mapped_len: usize, code_len: usize) -> Self {
        Self {
            address,
            mapped_len,
            code_len,
            chunk: None,
        }
    }

    pub fn code(&self) -> &[u8] {
        ExecImage::code(self)
    }

    /// Return a typed entry pointer.
    ///
    /// # Safety
    /// T must be a function pointer with the generated entry's exact calling
    /// convention/signature. It must not outlive this mapping. Executing the
    /// code requires the host ISA, and all accessed addresses must be valid.
    pub unsafe fn entry<T: Copy>(&self) -> T {
        unsafe {
            self.entry_at(0)
                .expect("function entry pointer has pointer size")
        }
    }

    /// # Safety
    /// Same requirements as `entry`; offset must begin a generated function.
    pub unsafe fn entry_at<T: Copy>(&self, offset: usize) -> Option<T> {
        if mem::size_of::<T>() != mem::size_of::<*const u8>() || offset >= self.code_len {
            return None;
        }
        #[cfg(target_arch = "aarch64")]
        if !offset.is_multiple_of(4) {
            return None;
        }
        let pointer = unsafe { self.address.as_ptr().add(offset) };
        Some(unsafe { mem::transmute_copy(&pointer) })
    }
}

impl Drop for HostExecImage {
    fn drop(&mut self) {
        #[cfg(target_os = "linux")]
        if self.chunk.is_some() {
            return;
        }
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        unsafe {
            libc::munmap(self.address.as_ptr().cast(), self.mapped_len);
        }
    }
}

/// Default host [`ExecMemory`] implementation.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostExecMemory;

impl ExecMemory for HostExecMemory {
    type Image = HostExecImage;
    type Error = ExecError;

    fn map_code(&self, code: &[u8]) -> Result<Self::Image, Self::Error> {
        map_code(code)
    }

    fn instruction_barrier(&self) {
        instruction_barrier();
    }
}

/// Copy generated code into writable pages, protect R+X, synchronize the host
/// instruction cache, and transfer ownership to an executable image.
pub fn map_code(code: impl AsRef<[u8]>) -> Result<HostExecImage, ExecError> {
    let code = code.as_ref();
    if code.is_empty() {
        return Err(ExecError::EmptyCode);
    }
    #[cfg(target_os = "linux")]
    if let Some(image) = arena::place(code) {
        return Ok(image);
    }
    #[cfg(target_os = "linux")]
    {
        let mapped_len = code.len();
        let raw = unsafe {
            libc::mmap(
                core::ptr::null_mut(),
                mapped_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if raw == libc::MAP_FAILED {
            return Err(ExecError::Mapping(std::io::Error::last_os_error()));
        }
        let address = NonNull::new(raw.cast::<u8>()).expect("mmap returned a null mapping");
        let image = HostExecImage {
            address,
            mapped_len,
            code_len: code.len(),
            chunk: None,
        };
        unsafe {
            core::ptr::copy_nonoverlapping(code.as_ptr(), address.as_ptr(), code.len());
        }
        if unsafe { libc::mprotect(raw, mapped_len, libc::PROT_READ | libc::PROT_EXEC) } != 0 {
            return Err(ExecError::Protection(std::io::Error::last_os_error()));
        }
        sync_icache(image.code());
        Ok(image)
    }
    #[cfg(target_os = "macos")]
    {
        let mapped_len = code.len();
        #[cfg(target_arch = "aarch64")]
        let jit_flag = libc::MAP_JIT;
        #[cfg(not(target_arch = "aarch64"))]
        let jit_flag = 0;
        let raw = unsafe {
            libc::mmap(
                core::ptr::null_mut(),
                mapped_len,
                libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | jit_flag,
                -1,
                0,
            )
        };
        if raw == libc::MAP_FAILED {
            return Err(ExecError::Mapping(std::io::Error::last_os_error()));
        }
        let address = NonNull::new(raw.cast::<u8>()).expect("mmap returned a null mapping");
        let image = HostExecImage {
            address,
            mapped_len,
            code_len: code.len(),
        };
        #[cfg(target_arch = "aarch64")]
        unsafe {
            libc::pthread_jit_write_protect_np(0);
            core::ptr::copy_nonoverlapping(code.as_ptr(), address.as_ptr(), code.len());
            libc::pthread_jit_write_protect_np(1);
            sys_icache_invalidate(raw, mapped_len);
        }
        #[cfg(not(target_arch = "aarch64"))]
        unsafe {
            core::ptr::copy_nonoverlapping(code.as_ptr(), address.as_ptr(), code.len());
            if libc::mprotect(raw, mapped_len, libc::PROT_READ | libc::PROT_EXEC) != 0 {
                return Err(ExecError::Protection(std::io::Error::last_os_error()));
            }
        }
        Ok(image)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    Err(ExecError::UnsupportedHost)
}

/// Resolve a dynamic symbol in the host process (`dlsym(RTLD_DEFAULT, name)` on Unix).
pub fn resolve_host_symbol(name: &str) -> Option<u64> {
    #[cfg(unix)]
    {
        let name = CString::new(name).ok()?;
        // SAFETY: null-terminated CString; RTLD_DEFAULT searches the global symbol scope.
        let p = unsafe { libc::dlsym(libc::RTLD_DEFAULT, name.as_ptr()) };
        if p.is_null() {
            None
        } else {
            Some(p as usize as u64)
        }
    }
    #[cfg(not(unix))]
    {
        let _ = name;
        None
    }
}

/// Make instructions another thread wrote visible to instruction fetch on the calling thread.
pub fn instruction_barrier() {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!("isb", options(nostack, preserves_flags));
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe extern "C" {
    fn sys_icache_invalidate(start: *mut core::ffi::c_void, len: usize);
}

pub fn sync_icache(code: &[u8]) {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    unsafe {
        sys_icache_invalidate(code.as_ptr() as *mut core::ffi::c_void, code.len());
    }
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    unsafe {
        let ctr: u64;
        core::arch::asm!("mrs {ctr}, ctr_el0", ctr = out(reg) ctr, options(nomem, nostack, preserves_flags));
        let start = code.as_ptr() as usize;
        let end = start + code.len();
        if ctr & (1 << 28) == 0 {
            let line = 4usize << ((ctr >> 16) & 15);
            let mut at = start & !(line - 1);
            while at < end {
                core::arch::asm!("dc cvau, {at}", at = in(reg) at, options(nostack, preserves_flags));
                at += line;
            }
        }
        core::arch::asm!("dsb ish", options(nostack, preserves_flags));
        if ctr & (1 << 29) == 0 {
            let line = 4usize << (ctr & 15);
            let mut at = start & !(line - 1);
            while at < end {
                core::arch::asm!("ic ivau, {at}", at = in(reg) at, options(nostack, preserves_flags));
                at += line;
            }
            core::arch::asm!("dsb ish", options(nostack, preserves_flags));
        }
        core::arch::asm!("isb", options(nostack, preserves_flags));
    }
    #[cfg(not(target_arch = "aarch64"))]
    let _ = code;
}

/// Shared executable arena for Linux (`memfd_create` mapped RW and RX separately).
#[cfg(target_os = "linux")]
mod arena {
    use super::{HostExecImage, NonNull, sync_icache};
    use alloc::sync::Arc;
    use std::sync::Mutex;

    const CHUNK: usize = 1 << 20;
    const ALIGN: usize = 16;

    pub struct Chunk {
        rw: *mut u8,
        rx: *mut u8,
        len: usize,
    }
    unsafe impl Send for Chunk {}
    unsafe impl Sync for Chunk {}

    impl Drop for Chunk {
        fn drop(&mut self) {
            unsafe {
                libc::munmap(self.rw.cast(), self.len);
                libc::munmap(self.rx.cast(), self.len);
            }
        }
    }

    impl Chunk {
        fn new() -> Option<Self> {
            unsafe {
                let fd = libc::memfd_create(c"volt-jit".as_ptr(), libc::MFD_CLOEXEC);
                if fd < 0 {
                    return None;
                }
                let sized = libc::ftruncate(fd, CHUNK as libc::off_t) == 0;
                let rw = if sized {
                    libc::mmap(
                        core::ptr::null_mut(),
                        CHUNK,
                        libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_SHARED,
                        fd,
                        0,
                    )
                } else {
                    libc::MAP_FAILED
                };
                let rx = if rw != libc::MAP_FAILED {
                    libc::mmap(
                        core::ptr::null_mut(),
                        CHUNK,
                        libc::PROT_READ | libc::PROT_EXEC,
                        libc::MAP_SHARED,
                        fd,
                        0,
                    )
                } else {
                    libc::MAP_FAILED
                };
                libc::close(fd);
                if rx == libc::MAP_FAILED {
                    if rw != libc::MAP_FAILED {
                        libc::munmap(rw, CHUNK);
                    }
                    return None;
                }
                Some(Self {
                    rw: rw.cast(),
                    rx: rx.cast(),
                    len: CHUNK,
                })
            }
        }
    }

    struct Current {
        chunk: Arc<Chunk>,
        used: usize,
    }

    static CURRENT: Mutex<Option<Current>> = Mutex::new(None);

    pub fn place(code: &[u8]) -> Option<HostExecImage> {
        if code.len() > CHUNK / 4 {
            return None;
        }
        let mut guard = CURRENT.lock().unwrap_or_else(|e| e.into_inner());
        let fits = |c: &Current| c.used + code.len() <= c.chunk.len;
        if !guard.as_ref().is_some_and(fits) {
            *guard = Some(Current {
                chunk: Arc::new(Chunk::new()?),
                used: 0,
            });
        }
        let current = guard.as_mut()?;
        let offset = current.used;
        current.used = (offset + code.len()).next_multiple_of(ALIGN);
        let chunk = current.chunk.clone();
        drop(guard);
        let (rx, address) = unsafe {
            core::ptr::copy_nonoverlapping(code.as_ptr(), chunk.rw.add(offset), code.len());
            let rx = chunk.rx.add(offset);
            (rx, NonNull::new(rx)?)
        };
        sync_icache(unsafe { core::slice::from_raw_parts(rx, code.len()) });
        Some(HostExecImage {
            address,
            mapped_len: code.len(),
            code_len: code.len(),
            chunk: Some(chunk),
        })
    }
}
