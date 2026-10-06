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
pub use module::{
    DataKind, DataReloc, Error as ModuleError, JittedModule, ModuleData, ModuleFunction,
    jit_module, jit_module_data,
};

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
}

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
