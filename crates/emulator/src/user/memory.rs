// Copyright 2026 LilithSemi
// SPDX-License-Identifier: Apache-2.0
// Modified: Rust port of Mirage lib/mirage-jit/user/Memory.zig.
//! Owned sparse user mappings. Guest addresses are never host pointers.
use crate::Error;
use crate::memory::{GuestMemory, MemoryError};

#[derive(Clone, Copy, Debug)]
pub(crate) struct Region {
    pub address: u64,
    pub len: usize,
    allocation: usize,
    offset: usize,
    permission: u8,
}
impl Region {
    pub fn end(self) -> u64 {
        self.address + self.len as u64
    }
}

#[derive(Default)]
pub struct Space {
    pub(crate) regions: Vec<Region>,
    allocations: Vec<Option<Vec<u8>>>,
}
impl Space {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn map(&mut self, address: u64, len: usize) -> Result<(), Error> {
        if len == 0 {
            return Err(Error::InvalidMapping);
        }
        let end = address
            .checked_add(len as u64)
            .ok_or(Error::AddressOverflow)?;
        if self
            .regions
            .iter()
            .any(|r| address < r.end() && r.address < end)
        {
            return Err(Error::OverlappingMapping);
        }
        self.regions
            .try_reserve(1)
            .map_err(|_| Error::OutOfMemory)?;
        self.allocations
            .try_reserve(1)
            .map_err(|_| Error::OutOfMemory)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(len)
            .map_err(|_| Error::OutOfMemory)?;
        bytes.resize(len, 0);
        let allocation = self.allocations.len();
        self.allocations.push(Some(bytes));
        self.regions.push(Region {
            address,
            len,
            allocation,
            offset: 0,
            permission: 3,
        });
        Ok(())
    }

    pub fn check(&self, address: u64, len: usize, permission: u8) -> Result<(), MemoryError> {
        let fault = || MemoryError::AccessFault {
            address,
            length: len,
        };
        let end = address.checked_add(len as u64).ok_or_else(fault)?;
        let mut at = address;
        while at < end {
            let region = self
                .regions
                .iter()
                .find(|r| at >= r.address && at < r.end())
                .ok_or_else(fault)?;
            if region.permission & permission != permission {
                return Err(fault());
            }
            at = end.min(region.end());
        }
        Ok(())
    }

    fn span(&self, address: u64, len: usize) -> Result<(usize, usize), MemoryError> {
        self.check(address, len, 0)?;
        let fault = || MemoryError::AccessFault {
            address,
            length: len,
        };
        let first = self
            .regions
            .iter()
            .find(|r| address >= r.address && address < r.end())
            .ok_or_else(fault)?;
        let offset = first.offset + (address - first.address) as usize;
        let end = address.checked_add(len as u64).ok_or_else(fault)?;
        let mut at = address;
        while at < end {
            let region = self
                .regions
                .iter()
                .find(|r| at >= r.address && at < r.end())
                .ok_or_else(fault)?;
            if region.allocation != first.allocation
                || region.offset + (at - region.address) as usize
                    != offset + (at - address) as usize
            {
                return Err(fault());
            }
            at = end.min(region.end());
        }
        Ok((first.allocation, offset))
    }

    pub(crate) fn slice(&self, address: u64, len: usize) -> Result<&[u8], MemoryError> {
        if len == 0 {
            return Ok(&[]);
        }
        let (allocation, offset) = self.span(address, len)?;
        Ok(&self.allocations[allocation].as_ref().unwrap()[offset..offset + len])
    }
    pub(crate) fn slice_mut(&mut self, address: u64, len: usize) -> Result<&mut [u8], MemoryError> {
        if len == 0 {
            return Ok(&mut []);
        }
        let (allocation, offset) = self.span(address, len)?;
        Ok(&mut self.allocations[allocation].as_mut().unwrap()[offset..offset + len])
    }
    /// Native syscall pointers remain valid until mappings are changed. Using
    /// Vec::as_mut_ptr does not create buffer references, so repeated vectors
    /// can legally overlap (as Linux writev permits).
    pub(crate) fn host_pointer(
        &mut self,
        address: u64,
        len: usize,
    ) -> Result<*mut u8, MemoryError> {
        let (allocation, offset) = self.span(address, len)?;
        let base = self.allocations[allocation].as_mut().unwrap().as_mut_ptr();
        // span validated this displacement within the backing allocation.
        Ok(unsafe { base.add(offset) })
    }
    pub(crate) fn chunk_len(&self, address: u64, max: usize) -> Result<usize, MemoryError> {
        let region = self
            .regions
            .iter()
            .find(|r| address >= r.address && address < r.end())
            .ok_or(MemoryError::AccessFault {
                address,
                length: max,
            })?;
        Ok(max.min((region.end() - address) as usize))
    }
    pub(crate) fn initialize(&mut self, address: u64, bytes: &[u8]) -> Result<(), MemoryError> {
        self.check(address, bytes.len(), 0)?;
        let mut offset = 0;
        while offset < bytes.len() {
            let at = address + offset as u64;
            let n = self.chunk_len(at, bytes.len() - offset)?;
            self.slice_mut(at, n)?
                .copy_from_slice(&bytes[offset..offset + n]);
            offset += n;
        }
        Ok(())
    }
    fn split(&mut self, at: u64) {
        if let Some(i) = self
            .regions
            .iter()
            .position(|r| at > r.address && at < r.end())
        {
            let region = self.regions[i];
            let n = (at - region.address) as usize;
            self.regions[i].len = n;
            self.regions.insert(
                i + 1,
                Region {
                    address: at,
                    len: region.len - n,
                    offset: region.offset + n,
                    ..region
                },
            );
        }
    }
    pub fn protect(&mut self, address: u64, len: usize, permission: u8) -> Result<(), Error> {
        if len == 0 || permission & !7 != 0 {
            return Err(Error::InvalidMapping);
        }
        self.check(address, len, 0)?;
        let end = address
            .checked_add(len as u64)
            .ok_or(Error::AddressOverflow)?;
        self.regions
            .try_reserve(2)
            .map_err(|_| Error::OutOfMemory)?;
        self.split(address);
        self.split(end);
        for r in &mut self.regions {
            if r.address >= address && r.address < end {
                r.permission = permission;
            }
        }
        Ok(())
    }
    pub fn unmap(&mut self, address: u64, len: usize) -> Result<(), Error> {
        if len == 0 {
            return Err(Error::InvalidMapping);
        }
        let end = address
            .checked_add(len as u64)
            .ok_or(Error::AddressOverflow)?;
        self.regions
            .try_reserve(2)
            .map_err(|_| Error::OutOfMemory)?;
        self.split(address);
        self.split(end);
        self.regions
            .retain(|r| r.address < address || r.address >= end);
        self.free_unused();
        Ok(())
    }
    pub(crate) fn truncate(&mut self, count: usize) {
        self.regions.truncate(count);
        self.free_unused();
    }
    fn free_unused(&mut self) {
        for (i, allocation) in self.allocations.iter_mut().enumerate() {
            if !self.regions.iter().any(|r| r.allocation == i) {
                *allocation = None;
            }
        }
        while self.allocations.last().is_some_and(Option::is_none) {
            self.allocations.pop();
        }
    }
}
impl GuestMemory for Space {
    fn read(&self, address: u64, bytes: &mut [u8]) -> Result<(), MemoryError> {
        self.check(address, bytes.len(), 1)?;
        let mut offset = 0;
        while offset < bytes.len() {
            let at = address + offset as u64;
            let n = self.chunk_len(at, bytes.len() - offset)?;
            bytes[offset..offset + n].copy_from_slice(self.slice(at, n)?);
            offset += n;
        }
        Ok(())
    }
    fn write(&mut self, address: u64, bytes: &[u8]) -> Result<(), MemoryError> {
        self.check(address, bytes.len(), 2)?;
        self.initialize(address, bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn split_permissions_and_cross_allocation_access() {
        let mut space = Space::new();
        space.map(0x1000, 4096).unwrap();
        space.map(0x2000, 4096).unwrap();
        space.write(0x1ffe, &[1, 2, 3, 4]).unwrap();
        let mut bytes = [0; 4];
        space.read(0x1ffe, &mut bytes).unwrap();
        assert_eq!(bytes, [1, 2, 3, 4]);
        assert!(space.slice(0x1ffe, 4).is_err());
        space.protect(0x2000, 2048, 1).unwrap();
        assert!(space.write(0x1ffe, &[0; 4]).is_err());
        space.read(0x1ffe, &mut bytes).unwrap();
        assert_eq!(bytes, [1, 2, 3, 4]);
        space.unmap(0x1800, 4096).unwrap();
        assert!(space.read(0x1800, &mut [0]).is_err());
        space.write(0x1000, &[9]).unwrap();
        space.write(0x2800, &[8]).unwrap();
    }
    #[test]
    fn overflow_and_execute_permissions() {
        let mut space = Space::new();
        assert!(space.map(u64::MAX, 2).is_err());
        space.map(0x1000, 4096).unwrap();
        assert!(space.check(0x1000, 4, 4).is_err());
        space.protect(0x1000, 4096, 5).unwrap();
        space.check(0x1000, 4, 4).unwrap();
        assert!(space.check(u64::MAX, 2, 0).is_err());
    }
}
