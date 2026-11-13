//! In-process AArch64 linked-image JIT with owned W^X code and data mappings.
//! Cache maintenance and global relocations are handled by the shared native mapper.
//! Ported and modified from Vulcan, Copyright 2026 LilithSemi (Apache-2.0).

use super::link::Linked;
use crate::native::{JittedModule, ModuleError};

/// A linked module and its executable mapping. Data/BSS and relocated global
/// pointers remain alive for exactly as long as the generated code.
pub struct Compiled {
    pub linked: Linked,
    mapping: JittedModule,
}

impl Compiled {
    pub fn new(linked: Linked) -> Result<Self, ModuleError> {
        let mapping = crate::native::map_linked_aarch64(&linked)?;
        Ok(Self { linked, mapping })
    }

    pub fn offset_of(&self, name: &str) -> Option<usize> {
        self.linked.address_of(name)
    }

    pub fn code(&self) -> &[u8] {
        self.mapping.code()
    }

    pub fn data_addr(&self, name: &str) -> Option<*mut u8> {
        self.mapping.data_addr(name)
    }

    /// Return the callable entry for a linked function.
    ///
    /// # Safety
    /// T must be the generated function's exact AAPCS64 signature, the host
    /// must be AArch64, and the pointer must not outlive this module.
    pub unsafe fn entry<T: Copy>(&self, name: &str) -> Option<T> {
        let offset = self.offset_of(name)?;
        // SAFETY: the caller establishes the function signature and ISA.
        unsafe { self.mapping.entry_at(offset) }
    }
}
