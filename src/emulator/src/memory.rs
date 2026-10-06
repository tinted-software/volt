use alloc::vec::Vec;
use core::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryError {
    AccessFault { address: u64, length: usize },
    PrivateMemory,
}

impl fmt::Display for MemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AccessFault { address, length } => {
                write!(f, "guest access fault at {address:#x} ({length} bytes)")
            }
            Self::PrivateMemory => f.write_str("guest memory has no host mapping"),
        }
    }
}
impl core::error::Error for MemoryError {}

pub trait GuestMemory {
    fn read(&self, address: u64, out: &mut [u8]) -> Result<(), MemoryError>;
    fn write(&mut self, address: u64, bytes: &[u8]) -> Result<(), MemoryError>;
}

/// Host-visible RAM or a physical region without a host mapping.
pub enum Backing {
    Shared(Vec<u8>),
    Private,
}

pub struct Region {
    pub gpa: u64,
    pub len: u64,
    pub backing: Backing,
}

impl Region {
    pub fn ram(gpa: u64, bytes: Vec<u8>) -> Result<Self, MemoryError> {
        let len = bytes.len() as u64;
        gpa.checked_add(len).ok_or(MemoryError::AccessFault {
            address: gpa,
            length: bytes.len(),
        })?;
        Ok(Self {
            gpa,
            len,
            backing: Backing::Shared(bytes),
        })
    }
}

/// Owns RAM storage. Every access must fit one physical region, as in Mirage.
#[derive(Default)]
pub struct PhysicalMemory {
    pub regions: Vec<Region>,
}

impl PhysicalMemory {
    pub fn new(regions: Vec<Region>) -> Result<Self, MemoryError> {
        for region in &regions {
            let end = region
                .gpa
                .checked_add(region.len)
                .ok_or(MemoryError::AccessFault {
                    address: region.gpa,
                    length: usize::MAX,
                })?;
            if let Backing::Shared(bytes) = &region.backing {
                if bytes.len() as u64 != region.len {
                    return Err(MemoryError::AccessFault {
                        address: region.gpa,
                        length: bytes.len(),
                    });
                }
            }
            for previous in &regions {
                if core::ptr::eq(region, previous) {
                    break;
                }
                if region.gpa < previous.gpa + previous.len && previous.gpa < end {
                    return Err(MemoryError::AccessFault {
                        address: region.gpa,
                        length: region.len as usize,
                    });
                }
            }
        }
        Ok(Self { regions })
    }

    fn find(&self, address: u64, length: usize) -> Result<(usize, usize), MemoryError> {
        let fault = MemoryError::AccessFault { address, length };
        let end = address.checked_add(length as u64).ok_or(fault)?;
        self.regions
            .iter()
            .enumerate()
            .find_map(|(index, region)| {
                let region_end = region.gpa.checked_add(region.len)?;
                if address >= region.gpa && end <= region_end {
                    Some((index, (address - region.gpa) as usize))
                } else {
                    None
                }
            })
            .ok_or(fault)
    }

    pub fn slice(&self, address: u64, length: usize) -> Result<&[u8], MemoryError> {
        let (index, offset) = self.find(address, length)?;
        match &self.regions[index].backing {
            Backing::Shared(bytes) => bytes
                .get(offset..offset + length)
                .ok_or(MemoryError::AccessFault { address, length }),
            Backing::Private => Err(MemoryError::PrivateMemory),
        }
    }

    pub fn slice_mut(&mut self, address: u64, length: usize) -> Result<&mut [u8], MemoryError> {
        let (index, offset) = self.find(address, length)?;
        match &mut self.regions[index].backing {
            Backing::Shared(bytes) => bytes
                .get_mut(offset..offset + length)
                .ok_or(MemoryError::AccessFault { address, length }),
            Backing::Private => Err(MemoryError::PrivateMemory),
        }
    }
}

impl GuestMemory for PhysicalMemory {
    fn read(&self, address: u64, out: &mut [u8]) -> Result<(), MemoryError> {
        out.copy_from_slice(self.slice(address, out.len())?);
        Ok(())
    }
    fn write(&mut self, address: u64, bytes: &[u8]) -> Result<(), MemoryError> {
        self.slice_mut(address, bytes.len())?.copy_from_slice(bytes);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn ranges_cannot_wrap_or_cross_regions() {
        let mut memory = PhysicalMemory::new(vec![
            Region::ram(0x1000, vec![0; 8]).unwrap(),
            Region::ram(0x1008, vec![0; 8]).unwrap(),
        ])
        .unwrap();
        memory.write(0x1004, b"volt").unwrap();
        assert_eq!(memory.slice(0x1004, 4).unwrap(), b"volt");
        assert!(matches!(
            memory.slice(0x1004, 8),
            Err(MemoryError::AccessFault { .. })
        ));
        assert_eq!(
            memory.slice(0, 1),
            Err(MemoryError::AccessFault {
                address: 0,
                length: 1
            })
        );
        assert!(matches!(
            memory.slice(u64::MAX, 2),
            Err(MemoryError::AccessFault { .. })
        ));
        assert!(matches!(
            Region::ram(u64::MAX, vec![0; 2]),
            Err(MemoryError::AccessFault { .. })
        ));
    }

    #[test]
    fn private_regions_refuse_host_access() {
        let mut memory = PhysicalMemory::new(vec![Region {
            gpa: 0x8000,
            len: 4096,
            backing: Backing::Private,
        }])
        .unwrap();
        assert_eq!(memory.slice(0x8000, 1), Err(MemoryError::PrivateMemory));
        assert_eq!(memory.write(0x8000, b"x"), Err(MemoryError::PrivateMemory));
    }
}
