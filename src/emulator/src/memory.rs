use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt;
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};
#[cfg(feature = "system")]
use parking_lot::RwLock;

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
    /// Write tracking for pages that hold translated code, if this memory
    /// supports it. `None` means callers must revalidate code by content.
    fn code_tracker(&self) -> Option<&CodeTracker> {
        None
    }
    /// Host address of the start of the 4 KiB guest-physical page at `page`
    /// (page aligned), if it is entirely host-backed RAM. Generated code reads
    /// and writes through this pointer without any lock, so it must stay valid
    /// for the life of the memory: backing storage is never resized or moved.
    fn host_page(&self, _page: u64) -> Option<*mut u8> {
        None
    }
    /// A handle for reading RAM from worker threads, if this memory can offer
    /// one. Enables background compilation.
    fn code_reader(&self) -> Option<Arc<dyn crate::aarch64::ahead::CodeReader>> {
        None
    }
}

const PAGE_SHIFT: u32 = 12;

/// Tracks writes to physical pages that translated code was built from.
///
/// Each RAM page has one atomic word: bit 0 is set once a block has been
/// compiled from the page ("watched"), and the remaining bits count writes to a
/// watched page. A cached block stays valid while its page's word is unchanged,
/// so validating a hit is one atomic load. The tracker is shared (`Arc`) between
/// the memory and every vCPU, so a write by any vCPU invalidates all caches.
#[derive(Default)]
pub struct CodeTracker {
    ranges: Vec<PageRange>,
    /// Bumped whenever a page is watched for the first time. vCPUs compare it
    /// at dispatch to drop cached write permission for pages that just became
    /// code.
    watch_epoch: AtomicU64,
}

struct PageRange {
    first: u64,
    end: u64,
    words: Vec<AtomicU32>,
}

impl CodeTracker {
    fn for_regions(regions: &[Region]) -> Self {
        let ranges = regions
            .iter()
            .filter(|r| matches!(r.backing, Backing::Shared(_)))
            .map(|r| {
                let first = r.gpa >> PAGE_SHIFT;
                let end = (r.gpa + r.len).div_ceil(1 << PAGE_SHIFT);
                PageRange {
                    first,
                    end,
                    words: (first..end).map(|_| AtomicU32::new(0)).collect(),
                }
            })
            .collect();
        Self {
            ranges,
            watch_epoch: AtomicU64::new(0),
        }
    }

    fn word(&self, page: u64) -> Option<&AtomicU32> {
        self.ranges
            .iter()
            .find(|r| page >= r.first && page < r.end)
            .map(|r| &r.words[(page - r.first) as usize])
    }

    /// Start tracking writes to `page` and return its current version. Call this
    /// BEFORE reading the page's bytes: any later write then changes the version.
    pub fn watch(&self, page: u64) -> Option<u32> {
        self.watch_tracked(page).map(|(version, _)| version)
    }

    /// Like `watch`, also reporting whether this call started the tracking.
    pub fn watch_tracked(&self, page: u64) -> Option<(u32, bool)> {
        let previous = self.word(page)?.fetch_or(1, Ordering::AcqRel);
        let new = previous & 1 == 0;
        if new {
            self.watch_epoch.fetch_add(1, Ordering::Release);
        }
        Some((previous | 1, new))
    }

    /// Changes whenever any page becomes watched.
    pub fn epoch(&self) -> u64 {
        self.watch_epoch.load(Ordering::Acquire)
    }

    /// The page's current version, or `None` if it is not tracked RAM.
    pub fn version(&self, page: u64) -> Option<u32> {
        Some(self.word(page)?.load(Ordering::Acquire))
    }

    /// Record a write of `len` bytes at `address`.
    pub fn note_write(&self, address: u64, len: usize) {
        if len == 0 || self.ranges.is_empty() {
            return;
        }
        let first = address >> PAGE_SHIFT;
        let last = address.saturating_add(len as u64 - 1) >> PAGE_SHIFT;
        for page in first..=last {
            if let Some(word) = self.word(page)
                && word.load(Ordering::Relaxed) & 1 != 0
            {
                word.fetch_add(2, Ordering::Release);
            }
        }
    }
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
    tracker: Arc<CodeTracker>,
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
        let tracker = Arc::new(CodeTracker::for_regions(&regions));
        Ok(Self { regions, tracker })
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
        // The caller writes through the returned slice, so it counts as a write
        // already. The borrow (plus the RwLock for shared memory) keeps readers
        // from observing the bumped version with the old bytes.
        self.tracker.note_write(address, length);
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
    fn code_tracker(&self) -> Option<&CodeTracker> {
        Some(&self.tracker)
    }
    fn host_page(&self, page: u64) -> Option<*mut u8> {
        let (index, offset) = self.find(page, 1 << PAGE_SHIFT).ok()?;
        match &self.regions[index].backing {
            Backing::Shared(bytes) if offset + (1 << PAGE_SHIFT) <= bytes.len() => {
                // SAFETY: in bounds. The pointer is only ever handed to code
                // that treats it as raw guest RAM.
                Some(unsafe { bytes.as_ptr().add(offset) }.cast_mut())
            }
            _ => None,
        }
    }
}

/// Thread-safe shared guest memory for multi-vCPU execution.
#[cfg(feature = "system")]
#[derive(Clone, Default)]
pub struct SharedMemory {
    inner: Arc<RwLock<PhysicalMemory>>,
    /// Same tracker as inside `inner`, reachable without taking the lock.
    tracker: Arc<CodeTracker>,
    /// Host-backed regions as `(gpa, len, base)`, copied once at construction.
    /// Regions are never added, removed or resized, so this stays valid and lets
    /// `host_page` skip the lock.
    hosts: Arc<[(u64, u64, *mut u8)]>,
}
// SAFETY: the pointers refer to RAM owned by `inner` for as long as any handle
// lives; they are only ever used as raw guest memory addresses.
#[cfg(feature = "system")]
unsafe impl Send for SharedMemory {}
#[cfg(feature = "system")]
unsafe impl Sync for SharedMemory {}

#[cfg(feature = "system")]
impl SharedMemory {
    pub fn new(memory: PhysicalMemory) -> Self {
        let hosts = memory
            .regions
            .iter()
            .filter_map(|region| match &region.backing {
                Backing::Shared(bytes) => Some((region.gpa, region.len, bytes.as_ptr().cast_mut())),
                Backing::Private => None,
            })
            .collect();
        Self {
            tracker: memory.tracker.clone(),
            hosts,
            inner: Arc::new(RwLock::new(memory)),
        }
    }
}

#[cfg(feature = "system")]
impl SharedMemory {
    /// Host pointer for `[address, address + length)` if it lies wholly inside
    /// one host-backed RAM region. Lock-free: the region table is immutable.
    fn host_range(&self, address: u64, length: usize) -> Option<*mut u8> {
        let end = address.checked_add(length as u64)?;
        self.hosts.iter().find_map(|&(gpa, len, base)| {
            // SAFETY: the range lies inside the region, whose backing is `len` bytes.
            (address >= gpa && end <= gpa + len)
                .then(|| unsafe { base.add((address - gpa) as usize) })
        })
    }
}

#[cfg(feature = "system")]
impl GuestMemory for SharedMemory {
    // RAM accesses skip the `RwLock`: generated code already reads and writes
    // RAM through host pointers without it, so the lock protected nothing, and
    // it made every vCPU contend on one cache line for each fetch, page-table
    // walk and slow-path access. Anything that is not plain RAM (devices,
    // out-of-range addresses) still takes the locked path so errors are unchanged.
    fn read(&self, address: u64, out: &mut [u8]) -> Result<(), MemoryError> {
        if let Some(host) = self.host_range(address, out.len()) {
            // SAFETY: in bounds of a live RAM region; `out` is a distinct buffer.
            unsafe { core::ptr::copy_nonoverlapping(host, out.as_mut_ptr(), out.len()) };
            return Ok(());
        }
        self.inner.read().read(address, out)
    }

    fn write(&mut self, address: u64, bytes: &[u8]) -> Result<(), MemoryError> {
        if let Some(host) = self.host_range(address, bytes.len()) {
            // SAFETY: in bounds of a live RAM region; `bytes` is a distinct buffer.
            unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), host, bytes.len()) };
            self.tracker.note_write(address, bytes.len());
            return Ok(());
        }
        self.inner.write().write(address, bytes)
    }
    fn code_tracker(&self) -> Option<&CodeTracker> {
        Some(&self.tracker)
    }
    fn host_page(&self, page: u64) -> Option<*mut u8> {
        self.host_range(page, 1 << PAGE_SHIFT)
    }
    fn code_reader(&self) -> Option<Arc<dyn crate::aarch64::ahead::CodeReader>> {
        Some(Arc::new(self.clone()))
    }
}

#[cfg(feature = "system")]
impl crate::aarch64::ahead::CodeReader for SharedMemory {
    fn read(&self, physical: u64, out: &mut [u8]) -> bool {
        match self.host_range(physical, out.len()) {
            Some(host) => {
                // SAFETY: in bounds of a live RAM region; `out` is a distinct buffer.
                unsafe { core::ptr::copy_nonoverlapping(host, out.as_mut_ptr(), out.len()) };
                true
            }
            None => false,
        }
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

    #[cfg(feature = "system")]
    #[test]
    fn shared_memory_ram_access_is_lock_free_but_still_tracked() {
        let mut shared = SharedMemory::new(
            PhysicalMemory::new(vec![Region::ram(0x1000, vec![0; 0x2000]).unwrap()]).unwrap(),
        );
        let held = shared.inner.write(); // would deadlock if RAM took the lock
        let mut copy = shared.clone();
        let tracker = shared.code_tracker().unwrap();
        let (version, newly) = tracker.watch_tracked(1).unwrap();
        assert!(newly);
        copy.write(0x1008, b"volt").unwrap();
        let mut out = [0u8; 4];
        shared.read(0x1008, &mut out).unwrap();
        assert_eq!(&out, b"volt");
        assert_ne!(
            tracker.version(1).unwrap(),
            version,
            "write bumped the watched page"
        );
        drop(held);
        // Anything that is not wholly inside RAM keeps the locked path's errors.
        assert!(matches!(
            shared.read(0x2ffc, &mut [0u8; 8]),
            Err(MemoryError::AccessFault { .. })
        ));
        assert!(matches!(
            shared.write(0x9000, &[0]),
            Err(MemoryError::AccessFault { .. })
        ));
        assert_eq!(shared.host_page(0x1000).is_some(), true);
        assert_eq!(shared.host_page(0x3000), None);
    }
}
