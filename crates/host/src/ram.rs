use core::ptr::NonNull;
use volt_platform::PhysMap;

/// Anonymous `mmap`-backed guest physical RAM.
///
/// Pages are demand-zeroed by the host kernel, so large guest memory spaces
/// (such as XNU's 1 GiB configuration) do not fault pages until touched.
pub struct MmapRam {
    ptr: NonNull<u8>,
    len: usize,
}

// SAFETY: `MmapRam` uniquely owns its anonymous mapping for its lifetime.
unsafe impl Send for MmapRam {}
unsafe impl Sync for MmapRam {}

impl MmapRam {
    pub fn allocate(len: usize) -> Result<Self, std::io::Error> {
        if len == 0 {
            return Ok(Self {
                ptr: NonNull::dangling(),
                len: 0,
            });
        }
        #[cfg(unix)]
        {
            let raw = unsafe {
                libc::mmap(
                    core::ptr::null_mut(),
                    len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            if raw == libc::MAP_FAILED {
                return Err(std::io::Error::last_os_error());
            }
            let ptr = NonNull::new(raw.cast::<u8>()).expect("mmap returned null");
            Ok(Self { ptr, len })
        }
        #[cfg(not(unix))]
        {
            Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "mmap requires unix",
            ))
        }
    }
}

impl Drop for MmapRam {
    fn drop(&mut self) {
        #[cfg(unix)]
        if self.len > 0 {
            unsafe {
                libc::munmap(self.ptr.as_ptr().cast(), self.len);
            }
        }
    }
}

impl PhysMap for MmapRam {
    fn len(&self) -> usize {
        self.len
    }

    fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            &[]
        } else {
            unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
        }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        if self.len == 0 {
            &mut []
        } else {
            unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
        }
    }
}
