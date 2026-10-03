//! In-process JIT for AArch64: link a module image and hand back offsets.
//!
//! Mapping the image into W^X executable memory (`mmap`/`mprotect`) is still
//! pending; until then [`Compiled`] only carries the linked image and
//! [`sync_icache`] performs the architectural cache maintenance.

use super::link::Linked;

/// Synchronize the instruction stream with freshly written code: clean the
/// D-cache and invalidate the I-cache to the point of unification, fenced
/// with `dsb ish` / `isb`. A no-op off AArch64 (the code could not run there
/// anyway).
pub fn sync_icache() {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: DSB/ISB are always legal at EL0; they only order the local
        // core's view of already-mapped memory. Line-by-line DC/IC maintenance
        // needs `ctr_el0` line sizes and is handled by the future mmap path.
        unsafe {
            core::arch::asm!("dsb ish", options(nostack, preserves_flags));
            core::arch::asm!("isb", options(nostack, preserves_flags));
        }
    }
}

/// A JIT-compiled module: its linked image plus (once the mmap path lands)
/// the live executable mapping. Offsets in `linked` are byte offsets into
/// the image.
pub struct Compiled {
    /// Linked code, symbols, and data. Pending an `mmap` backend there is no
    /// separate executable buffer; the image address is free because every
    /// `bl` is PC-relative.
    pub linked: Linked,
}

impl Compiled {
    pub fn new(linked: Linked) -> Compiled {
        Compiled { linked }
    }

    /// Byte offset of the function named `name`, or `None` if not defined.
    pub fn offset_of(&self, name: &str) -> Option<usize> {
        self.linked.address_of(name)
    }
}
