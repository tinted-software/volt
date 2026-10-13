//! Native host compilation and W^X executable memory.
//! Ported and modified from Vulcan, Copyright 2026 LilithSemi (Apache-2.0).

use alloc::boxed::Box;
use core::{fmt, mem, ptr::NonNull};
use volt_ir::function::Function;

#[cfg(any(
    all(target_arch = "x86_64", feature = "x86_64"),
    all(target_arch = "aarch64", feature = "aarch64")
))]
mod module;
#[cfg(all(
    feature = "aarch64",
    target_os = "linux",
    any(
        all(target_arch = "x86_64", feature = "x86_64"),
        target_arch = "aarch64"
    )
))]
pub use module::map_linked_aarch64;
#[cfg(any(
    all(target_arch = "x86_64", feature = "x86_64"),
    all(target_arch = "aarch64", feature = "aarch64")
))]
pub use module::{Error as ModuleError, JittedModule, jit_module};

#[derive(Debug)]
pub enum Error {
    Compile(Box<dyn core::error::Error + Send + Sync>),
    EmptyCode,
    Mapping(std::io::Error),
    Protection(std::io::Error),
    UnsupportedHost,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compile(e) => write!(f, "native compilation: {e}"),
            Self::EmptyCode => f.write_str("cannot map empty machine code"),
            Self::Mapping(e) => write!(f, "executable allocation: {e}"),
            Self::Protection(e) => write!(f, "executable protection: {e}"),
            Self::UnsupportedHost => f.write_str("native JIT requires Linux x86_64 or AArch64"),
        }
    }
}
impl core::error::Error for Error {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Compile(e) => Some(e.as_ref()),
            Self::Mapping(e) | Self::Protection(e) => Some(e),
            _ => None,
        }
    }
}

/// Owns immutable executable pages. Never writable and executable simultaneously.
pub struct JittedFunction {
    address: NonNull<u8>,
    mapped_len: usize,
    code_len: usize,
    /// Set when the code lives in a shared arena chunk, which owns the mapping.
    chunk: Option<std::sync::Arc<arena::Chunk>>,
}

// SAFETY: the code is immutable after construction (the writable view of an
// arena chunk is never exposed) and the mapping lives as long as the value, so
// it can be shared and run from any thread.
unsafe impl Send for JittedFunction {}
unsafe impl Sync for JittedFunction {}

impl JittedFunction {
    pub fn code(&self) -> &[u8] {
        // SAFETY: mapping is live and readable for its owned code extent.
        unsafe { core::slice::from_raw_parts(self.address.as_ptr(), self.code_len) }
    }

    /// Return a typed entry pointer.
    ///
    /// # Safety
    /// T must be a function pointer with the generated entry's exact calling
    /// convention/signature. It must not outlive this mapping. Executing the
    /// code requires the host ISA, and all accessed addresses must be valid.
    pub unsafe fn entry<T: Copy>(&self) -> T {
        // SAFETY: caller establishes type/signature and lifetime above.
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
        if offset % 4 != 0 {
            return None;
        }
        // SAFETY: offset is in mapping and T is caller-guaranteed function pointer.
        let pointer = unsafe { self.address.as_ptr().add(offset) };
        Some(unsafe { mem::transmute_copy(&pointer) })
    }
}

impl Drop for JittedFunction {
    fn drop(&mut self) {
        if self.chunk.is_some() {
            return;
        }
        #[cfg(target_os = "linux")]
        // SAFETY: these are exactly the mmap address and length owned by self.
        unsafe {
            libc::munmap(self.address.as_ptr().cast(), self.mapped_len);
        }
    }
}

/// Copy generated code into writable pages, protect R+X, synchronize the host
/// instruction cache, and transfer ownership to an executable image.
pub fn map_code(code: impl AsRef<[u8]>) -> Result<JittedFunction, Error> {
    let code = code.as_ref();
    if code.is_empty() {
        return Err(Error::EmptyCode);
    }
    #[cfg(target_os = "linux")]
    if let Some(image) = arena::place(code) {
        return Ok(image);
    }
    #[cfg(target_os = "linux")]
    {
        // mmap/mprotect round their lengths to the runtime host page size. Using
        // the original length also makes this correct on AArch64 64 KiB kernels.
        let mapped_len = code.len();
        // SAFETY: anonymous private mapping, no descriptor or address constraint.
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
            return Err(Error::Mapping(std::io::Error::last_os_error()));
        }
        let address = NonNull::new(raw.cast::<u8>()).expect("mmap returned a null mapping");
        let image = JittedFunction {
            address,
            mapped_len,
            code_len: code.len(),
            chunk: None,
        };
        // SAFETY: new mapping is writable and at least code.len() bytes long.
        unsafe {
            core::ptr::copy_nonoverlapping(code.as_ptr(), address.as_ptr(), code.len());
        }
        // SAFETY: image owns the complete page-aligned mapping.
        if unsafe { libc::mprotect(raw, mapped_len, libc::PROT_READ | libc::PROT_EXEC) } != 0 {
            return Err(Error::Protection(std::io::Error::last_os_error()));
        }
        sync_icache(image.code());
        Ok(image)
    }
    #[cfg(not(target_os = "linux"))]
    Err(Error::UnsupportedHost)
}

/// Compile a Rust SSA function for the architecture executing this process.
pub fn compile(function: &Function) -> Result<JittedFunction, Error> {
    #[cfg(all(target_arch = "x86_64", feature = "x86_64"))]
    {
        let code = crate::x86_64::compile(function).map_err(|e| Error::Compile(Box::new(e)))?;
        map_code(code)
    }
    #[cfg(all(target_arch = "aarch64", feature = "aarch64"))]
    {
        let words =
            crate::aarch64::isel::compile(function).map_err(|e| Error::Compile(Box::new(e)))?;
        // Linux AArch64 host ABI is little endian. Reborrow native words to avoid
        // allocating a second byte vector solely for executable mapping.
        let bytes =
            unsafe { core::slice::from_raw_parts(words.as_ptr().cast::<u8>(), words.len() * 4) };
        map_code(bytes)
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", feature = "x86_64"),
        all(target_arch = "aarch64", feature = "aarch64")
    )))]
    {
        let _ = function;
        Err(Error::UnsupportedHost)
    }
}

/// Like [`compile`], consuming `function` so the backend need not copy it.
pub fn compile_owned(function: Function) -> Result<JittedFunction, Error> {
    #[cfg(all(target_arch = "aarch64", feature = "aarch64"))]
    {
        let words = crate::aarch64::isel::compile_owned(function)
            .map_err(|e| Error::Compile(Box::new(e)))?;
        // SAFETY: `words` is a live slice of plain `u32`s; AArch64 hosts are little endian.
        let bytes =
            unsafe { core::slice::from_raw_parts(words.as_ptr().cast::<u8>(), words.len() * 4) };
        map_code(bytes)
    }
    #[cfg(not(all(target_arch = "aarch64", feature = "aarch64")))]
    {
        compile(&function)
    }
}

/// Make instructions another thread wrote (and synchronized with
/// [`map_code`]) visible to instruction fetch on the calling thread. Call it
/// before first running code that was compiled elsewhere.
pub fn instruction_barrier() {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: `isb` only flushes this core's pipeline.
    unsafe {
        core::arch::asm!("isb", options(nostack, preserves_flags));
    }
}

fn sync_icache(code: &[u8]) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        // CTR_EL0.IDC/DIC allow skipping unneeded cache maintenance. Otherwise
        // use the actual host line sizes, not an assumed 64-byte cache line.
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

/// Shared executable arena. Small functions are bump-allocated from large
/// chunks so each one costs a memcpy instead of mmap + mprotect + munmap.
///
/// W^X is preserved: a chunk is a memfd mapped twice, once read/write (never
/// executable) and once read/execute (never writable).
#[cfg(target_os = "linux")]
mod arena {
    use super::{JittedFunction, NonNull, sync_icache};
    use std::sync::{Arc, Mutex};

    const CHUNK: usize = 1 << 20;
    const ALIGN: usize = 16;

    pub struct Chunk {
        rw: *mut u8,
        rx: *mut u8,
        len: usize,
    }
    // SAFETY: the pointers name process-wide mappings that live as long as the
    // chunk. Writes only go to disjoint, not-yet-published byte ranges.
    unsafe impl Send for Chunk {}
    unsafe impl Sync for Chunk {}

    impl Drop for Chunk {
        fn drop(&mut self) {
            // SAFETY: exact addresses and length returned by mmap in `Chunk::new`.
            unsafe {
                libc::munmap(self.rw.cast(), self.len);
                libc::munmap(self.rx.cast(), self.len);
            }
        }
    }

    impl Chunk {
        fn new() -> Option<Self> {
            // SAFETY: plain syscalls; every failure path releases what it made.
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

    /// Copy `code` into the arena. `None` means the caller should fall back to
    /// a dedicated mapping (oversized code or the host refused memfd/mmap).
    pub fn place(code: &[u8]) -> Option<JittedFunction> {
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
        // SAFETY: [offset, offset + len) lies inside the chunk and was reserved
        // exclusively for this call under the lock.
        let (rx, address) = unsafe {
            core::ptr::copy_nonoverlapping(code.as_ptr(), chunk.rw.add(offset), code.len());
            let rx = chunk.rx.add(offset);
            (rx, NonNull::new(rx)?)
        };
        // SAFETY: the executable view is live and covers the bytes just written.
        sync_icache(unsafe { core::slice::from_raw_parts(rx, code.len()) });
        Some(JittedFunction {
            address,
            mapped_len: code.len(),
            code_len: code.len(),
            chunk: Some(chunk),
        })
    }
}
