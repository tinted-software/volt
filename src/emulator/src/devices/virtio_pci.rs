//! VirtIO block device on PCI: the modern (virtio 1.0) transport.
//!
//! The device is a type-0 function, vendor 0x1af4, device 0x1042 (0x1040 plus the block
//! type). BAR0 holds the four virtio structures, each at its own offset, and the capability
//! list names them:
//!
//! | structure | offset | length | notes                              |
//! |-----------|--------|--------|------------------------------------|
//! | common    | 0x0000 | 0x40   | feature, status, queue registers   |
//! | notify    | 0x1000 | 0x100  | a write of a queue index kicks it  |
//! | ISR       | 0x2000 | 0x4     | never raised: the firmware polls   |
//! | device    | 0x3000 | 0x100  | the block configuration            |
//!
//! Requests are served by the same `BlockQueue` the MMIO transport uses.

use super::pci::{ConfigSpace, PciFunction};
use super::virtio_blk::{
    BlockBackend, BlockQueue, QUEUE_MAX_SIZE, Queue, SECTOR_SIZE, VIRTIO_BLK_F_BLK_SIZE,
    VIRTIO_BLK_F_FLUSH, VIRTIO_F_VERSION_1,
};
use crate::memory::SharedMemory;
use alloc::vec::Vec;

const VIRTIO_VENDOR: u16 = 0x1af4;
/// The modern device ID: 0x1040 plus the virtio device type, 2 for block.
const DEVICE_ID_BLOCK: u16 = 0x1042;
/// The subsystem device ID carries the device type for transitional drivers.
const SUBSYSTEM_BLOCK: u16 = 2;
/// Base class 0x01, mass storage; subclass 0x80, other.
const CLASS_STORAGE_OTHER: u32 = 0x01_80_00;

/// The BAR holding every structure, and its size.
const BAR: usize = 0;
const BAR_SIZE: u64 = 0x4000;

const COMMON_OFFSET: u64 = 0x0000;
const COMMON_LEN: u64 = 0x40;
const NOTIFY_OFFSET: u64 = 0x1000;
const NOTIFY_LEN: u64 = 0x100;
/// Each queue's notify register sits at `queue_notify_off * NOTIFY_MULTIPLIER` into the region.
const NOTIFY_MULTIPLIER: u32 = 4;
const ISR_OFFSET: u64 = 0x2000;
const ISR_LEN: u64 = 0x4;
const DEVICE_OFFSET: u64 = 0x3000;
const DEVICE_LEN: u64 = 0x100;

/// Capability identifier of a vendor-specific capability.
const CAP_VENDOR: u8 = 0x09;
const CFG_COMMON: u8 = 1;
const CFG_NOTIFY: u8 = 2;
const CFG_ISR: u8 = 3;
const CFG_DEVICE: u8 = 4;

/// `MSI-X` vector meaning "none".
const NO_VECTOR: u16 = 0xffff;
const NUM_QUEUES: u16 = 1;
/// The features offered. Only the features this model implements are offered.
const FEATURES: u64 = VIRTIO_F_VERSION_1 | VIRTIO_BLK_F_BLK_SIZE | VIRTIO_BLK_F_FLUSH;

/// Common configuration offsets, from virtio 1.2 section 4.1.4.3.
mod common {
    pub const DEVICE_FEATURE_SELECT: u64 = 0x00;
    pub const DEVICE_FEATURE: u64 = 0x04;
    pub const DRIVER_FEATURE_SELECT: u64 = 0x08;
    pub const DRIVER_FEATURE: u64 = 0x0c;
    pub const MSIX_CONFIG: u64 = 0x10;
    pub const DEVICE_STATUS: u64 = 0x14;
    pub const QUEUE_SELECT: u64 = 0x16;
    pub const QUEUE_SIZE: u64 = 0x18;
    pub const QUEUE_MSIX_VECTOR: u64 = 0x1a;
    pub const QUEUE_ENABLE: u64 = 0x1c;
    pub const QUEUE_DESC: u64 = 0x20;
    pub const QUEUE_DRIVER: u64 = 0x28;
    pub const QUEUE_DEVICE: u64 = 0x30;
}

/// A virtio block device on the modern PCI transport.
pub struct VirtioBlockPci<B> {
    config: ConfigSpace,
    core: BlockQueue<B>,
    device_features_sel: u32,
    driver_features_sel: u32,
    driver_features: u64,
    status: u8,
    queue_select: u16,
    msix_config: u16,
    queue_msix_vector: u16,
}

impl<B: BlockBackend> VirtioBlockPci<B> {
    pub fn new(backend: B, memory: SharedMemory) -> Self {
        let mut config = ConfigSpace::new(
            VIRTIO_VENDOR,
            DEVICE_ID_BLOCK,
            CLASS_STORAGE_OTHER,
            VIRTIO_VENDOR,
            SUBSYSTEM_BLOCK,
        );
        config.set_bar(BAR, BAR_SIZE);
        config.add_capability(
            CAP_VENDOR,
            &structure_body(CFG_COMMON, COMMON_OFFSET, COMMON_LEN, None),
        );
        config.add_capability(
            CAP_VENDOR,
            &structure_body(
                CFG_NOTIFY,
                NOTIFY_OFFSET,
                NOTIFY_LEN,
                Some(NOTIFY_MULTIPLIER),
            ),
        );
        config.add_capability(
            CAP_VENDOR,
            &structure_body(CFG_ISR, ISR_OFFSET, ISR_LEN, None),
        );
        config.add_capability(
            CAP_VENDOR,
            &structure_body(CFG_DEVICE, DEVICE_OFFSET, DEVICE_LEN, None),
        );
        Self {
            config,
            core: BlockQueue::new(backend, memory),
            device_features_sel: 0,
            driver_features_sel: 0,
            driver_features: 0,
            status: 0,
            queue_select: 0,
            msix_config: NO_VECTOR,
            queue_msix_vector: NO_VECTOR,
        }
    }

    fn reset(&mut self) {
        self.device_features_sel = 0;
        self.driver_features_sel = 0;
        self.driver_features = 0;
        self.queue_select = 0;
        self.msix_config = NO_VECTOR;
        self.queue_msix_vector = NO_VECTOR;
        self.core.queue = Queue::default();
    }

    /// A 32-bit word of the common configuration, at a four-byte-aligned offset.
    fn common_dword(&self, at: u64) -> u32 {
        let queue = &self.core.queue;
        let selected = self.queue_select == 0;
        match at {
            common::DEVICE_FEATURE_SELECT => self.device_features_sel,
            common::DEVICE_FEATURE => match self.device_features_sel {
                0 => FEATURES as u32,
                1 => (FEATURES >> 32) as u32,
                _ => 0,
            },
            common::DRIVER_FEATURE_SELECT => self.driver_features_sel,
            common::DRIVER_FEATURE => match self.driver_features_sel {
                0 => self.driver_features as u32,
                1 => (self.driver_features >> 32) as u32,
                _ => 0,
            },
            // msix_config, then num_queues.
            0x10 => u32::from(self.msix_config) | u32::from(NUM_QUEUES) << 16,
            // device_status, config_generation (always 0), then queue_select.
            0x14 => u32::from(self.status) | u32::from(self.queue_select) << 16,
            // queue_size (the maximum until the driver sets one), then queue_msix_vector.
            0x18 => {
                let size = match (selected, queue.num) {
                    (false, _) => 0,
                    (true, 0) => QUEUE_MAX_SIZE,
                    (true, num) => num,
                };
                size | u32::from(self.queue_msix_vector) << 16
            }
            // queue_enable, then queue_notify_off, which is the queue index.
            0x1c => {
                let enable = u32::from(selected && queue.ready);
                enable | u32::from(self.queue_select) << 16
            }
            common::QUEUE_DESC => queue.desc_table as u32,
            0x24 => (queue.desc_table >> 32) as u32,
            common::QUEUE_DRIVER => queue.avail_ring as u32,
            0x2c => (queue.avail_ring >> 32) as u32,
            common::QUEUE_DEVICE => queue.used_ring as u32,
            0x34 => (queue.used_ring >> 32) as u32,
            _ => 0,
        }
    }

    /// Writes one field of the common configuration. Fields are written at their own
    /// offsets; a 64-bit field arrives as one access or as two halves.
    fn common_write(&mut self, at: u64, size: u8, value: u64) {
        if size == 8 {
            self.common_write(at, 4, value & 0xffff_ffff);
            self.common_write(at + 4, 4, value >> 32);
            return;
        }
        let low = value as u32;
        match at {
            common::DEVICE_FEATURE_SELECT => self.device_features_sel = low,
            common::DRIVER_FEATURE_SELECT => self.driver_features_sel = low,
            common::DRIVER_FEATURE => match self.driver_features_sel {
                0 => {
                    self.driver_features = (self.driver_features & !0xffff_ffff) | u64::from(low);
                }
                1 => {
                    self.driver_features =
                        (self.driver_features & 0xffff_ffff) | u64::from(low) << 32;
                }
                _ => {}
            },
            common::MSIX_CONFIG => self.msix_config = value as u16,
            common::DEVICE_STATUS => {
                self.status = value as u8;
                if self.status == 0 {
                    self.reset();
                }
            }
            common::QUEUE_SELECT => self.queue_select = value as u16,
            common::QUEUE_SIZE => {
                if self.queue_select == 0 {
                    self.core.queue.num = (value as u32).min(QUEUE_MAX_SIZE);
                }
            }
            common::QUEUE_MSIX_VECTOR => self.queue_msix_vector = value as u16,
            common::QUEUE_ENABLE => {
                if self.queue_select == 0 {
                    self.core.queue.ready = value & 1 != 0;
                }
            }
            common::QUEUE_DESC => set_low(&mut self.core.queue.desc_table, low),
            0x24 => set_high(&mut self.core.queue.desc_table, low),
            common::QUEUE_DRIVER => set_low(&mut self.core.queue.avail_ring, low),
            0x2c => set_high(&mut self.core.queue.avail_ring, low),
            common::QUEUE_DEVICE => set_low(&mut self.core.queue.used_ring, low),
            0x34 => set_high(&mut self.core.queue.used_ring, low),
            // Read-only fields, and the fields of queues this device does not have.
            _ => {}
        }
    }

    /// A 32-bit word of the device configuration: the capacity in sectors, then the sector size.
    fn device_dword(&self, at: u64) -> u32 {
        match at {
            0x00 => self.core.capacity_sectors() as u32,
            0x04 => (self.core.capacity_sectors() >> 32) as u32,
            0x14 => SECTOR_SIZE as u32,
            _ => 0,
        }
    }
}

/// The body of a virtio vendor capability, after its identifier and `next` pointer. The
/// notify structure adds the multiplier that scales a queue's notify offset.
fn structure_body(cfg_type: u8, offset: u64, length: u64, multiplier: Option<u32>) -> Vec<u8> {
    let total = if multiplier.is_some() { 20 } else { 16 };
    let mut body = Vec::with_capacity(total - 2);
    body.push(total as u8); // cap_len
    body.push(cfg_type);
    body.push(BAR as u8);
    body.push(0); // id, unused
    body.extend_from_slice(&[0, 0]); // padding
    body.extend_from_slice(&(offset as u32).to_le_bytes());
    body.extend_from_slice(&(length as u32).to_le_bytes());
    if let Some(multiplier) = multiplier {
        body.extend_from_slice(&multiplier.to_le_bytes());
    }
    body
}

fn set_low(field: &mut u64, low: u32) {
    *field = (*field & !0xffff_ffff) | u64::from(low);
}

fn set_high(field: &mut u64, high: u32) {
    *field = (*field & 0xffff_ffff) | u64::from(high) << 32;
}

/// Reads `size` bytes at `offset` from a structure whose words `word` supplies, byte by byte.
fn read_bytes(offset: u64, size: u8, word: impl Fn(u64) -> u32) -> u64 {
    (0..u64::from(size)).fold(0, |value, i| {
        let at = offset + i;
        let byte = (word(at & !3) >> ((at & 3) * 8)) as u8;
        value | u64::from(byte) << (8 * i)
    })
}

impl<B: BlockBackend + 'static> PciFunction for VirtioBlockPci<B> {
    fn config(&self) -> &ConfigSpace {
        &self.config
    }

    fn config_mut(&mut self) -> &mut ConfigSpace {
        &mut self.config
    }

    fn bar_read(&mut self, bar: usize, offset: u64, size: u8) -> u64 {
        if bar != BAR {
            return 0;
        }
        match offset {
            o if o < COMMON_OFFSET + COMMON_LEN => {
                read_bytes(o - COMMON_OFFSET, size, |at| self.common_dword(at))
            }
            // The notify register is write-only, and the ISR is never raised.
            o if (NOTIFY_OFFSET..NOTIFY_OFFSET + NOTIFY_LEN).contains(&o)
                || (ISR_OFFSET..ISR_OFFSET + ISR_LEN).contains(&o) =>
            {
                0
            }
            o if (DEVICE_OFFSET..DEVICE_OFFSET + DEVICE_LEN).contains(&o) => {
                read_bytes(o - DEVICE_OFFSET, size, |at| self.device_dword(at))
            }
            _ => 0,
        }
    }

    fn bar_write(&mut self, bar: usize, offset: u64, size: u8, value: u64) {
        if bar != BAR {
            return;
        }
        match offset {
            o if o < COMMON_OFFSET + COMMON_LEN => {
                self.common_write(o - COMMON_OFFSET, size, value);
            }
            // A driver kicks queue 0 with its index. The only queue this device has is 0.
            o if (NOTIFY_OFFSET..NOTIFY_OFFSET + NOTIFY_LEN).contains(&o)
                && value as u16 == 0
                && self.core.queue.ready =>
            {
                self.core.process();
            }
            // The device configuration is read only.
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices::virtio_blk::{
        VIRTIO_BLK_S_OK, VIRTIO_BLK_T_IN, VRING_DESC_F_NEXT, VRING_DESC_F_WRITE,
    };
    use crate::memory::{GuestMemory, PhysicalMemory, Region};

    const RAM: u64 = 0x4000_0000;
    const DESC: u64 = RAM + 0x1_0000;
    const AVAIL: u64 = RAM + 0x1_1000;
    const USED: u64 = RAM + 0x1_2000;
    const HEADER: u64 = RAM + 0x2_0000;
    const DATA: u64 = RAM + 0x2_1000;
    const STATUS: u64 = RAM + 0x2_2000;

    fn device(disk: Vec<u8>) -> (VirtioBlockPci<Vec<u8>>, SharedMemory) {
        let ram = Region::ram(RAM, alloc::vec![0u8; 1024 * 1024]).unwrap();
        let memory = SharedMemory::new(PhysicalMemory::new(alloc::vec![ram]).unwrap());
        (VirtioBlockPci::new(disk, memory.clone()), memory)
    }

    /// The capabilities in list order, as `(structure, offset, length, multiplier)`. The
    /// multiplier is read only for the notify structure, which has the longer capability.
    fn capabilities(config: &ConfigSpace) -> Vec<(u8, u64, u64, Option<u64>)> {
        let mut found = Vec::new();
        let mut at = config.read(0x34, 1) as usize;
        while at != 0 {
            assert_eq!(config.read(at, 1), u64::from(CAP_VENDOR));
            let cfg_type = config.read(at + 3, 1) as u8;
            assert_eq!(config.read(at + 4, 1), 0, "every structure is in BAR0");
            let multiplier = (cfg_type == CFG_NOTIFY).then(|| config.read(at + 16, 4));
            found.push((
                cfg_type,
                config.read(at + 8, 4),
                config.read(at + 12, 4),
                multiplier,
            ));
            at = config.read(at + 1, 1) as usize;
        }
        found
    }

    #[test]
    fn capabilities_locate_the_four_structures_in_bar0() {
        let (dev, _) = device(alloc::vec![0; 512]);
        let config = dev.config();
        assert_ne!(config.read(0x06, 2) & 0x10, 0, "status lists capabilities");
        assert_eq!(
            capabilities(config),
            [
                (CFG_COMMON, COMMON_OFFSET, COMMON_LEN, None),
                (
                    CFG_NOTIFY,
                    NOTIFY_OFFSET,
                    NOTIFY_LEN,
                    Some(u64::from(NOTIFY_MULTIPLIER))
                ),
                (CFG_ISR, ISR_OFFSET, ISR_LEN, None),
                (CFG_DEVICE, DEVICE_OFFSET, DEVICE_LEN, None),
            ]
            .to_vec()
        );
    }

    fn write_descriptor(
        memory: &mut SharedMemory,
        index: u64,
        addr: u64,
        len: u32,
        flags: u16,
        next: u16,
    ) {
        let mut desc = [0u8; 16];
        desc[0..8].copy_from_slice(&addr.to_le_bytes());
        desc[8..12].copy_from_slice(&len.to_le_bytes());
        desc[12..14].copy_from_slice(&flags.to_le_bytes());
        desc[14..16].copy_from_slice(&next.to_le_bytes());
        memory.write(DESC + index * 16, &desc).unwrap();
    }

    /// Drives the device the way virtio-drivers does: reset, acknowledge, negotiate the
    /// features, set FEATURES_OK, and configure and enable queue 0.
    fn negotiate(dev: &mut VirtioBlockPci<Vec<u8>>) {
        dev.bar_write(BAR, COMMON_OFFSET + common::DEVICE_STATUS, 1, 0);
        dev.bar_write(BAR, COMMON_OFFSET + common::DEVICE_STATUS, 1, 0x3);
        dev.bar_write(BAR, COMMON_OFFSET + common::DRIVER_FEATURE_SELECT, 4, 0);
        dev.bar_write(
            BAR,
            COMMON_OFFSET + common::DRIVER_FEATURE,
            4,
            VIRTIO_BLK_F_BLK_SIZE | VIRTIO_BLK_F_FLUSH,
        );
        dev.bar_write(BAR, COMMON_OFFSET + common::DRIVER_FEATURE_SELECT, 4, 1);
        dev.bar_write(BAR, COMMON_OFFSET + common::DRIVER_FEATURE, 4, 1);
        dev.bar_write(BAR, COMMON_OFFSET + common::DEVICE_STATUS, 1, 0xb);
        dev.bar_write(BAR, COMMON_OFFSET + common::QUEUE_SELECT, 2, 0);
        dev.bar_write(BAR, COMMON_OFFSET + common::QUEUE_SIZE, 2, 16);
        dev.bar_write(BAR, COMMON_OFFSET + common::QUEUE_DESC, 8, DESC);
        dev.bar_write(BAR, COMMON_OFFSET + common::QUEUE_DRIVER, 8, AVAIL);
        dev.bar_write(BAR, COMMON_OFFSET + common::QUEUE_DEVICE, 8, USED);
        dev.bar_write(BAR, COMMON_OFFSET + common::QUEUE_ENABLE, 2, 1);
    }

    #[test]
    fn driver_negotiates_features_and_serves_a_read_through_the_pci_transport() {
        let mut disk = alloc::vec![0x42u8; 2048];
        disk[512..1024].fill(0x37); // sector 1 is what the request reads
        let (mut dev, mut memory) = device(disk);

        // The device offers VERSION_1 in the high word of its features.
        dev.bar_write(BAR, COMMON_OFFSET + common::DEVICE_FEATURE_SELECT, 4, 1);
        assert_eq!(
            dev.bar_read(BAR, COMMON_OFFSET + common::DEVICE_FEATURE, 4),
            1
        );

        negotiate(&mut dev);
        assert_eq!(
            dev.bar_read(BAR, COMMON_OFFSET + common::DEVICE_STATUS, 1),
            0xb,
            "FEATURES_OK is kept when the driver's features are a subset of the device's"
        );
        // The configured queue reads back: 8-byte fields come back whole.
        assert_eq!(
            dev.bar_read(BAR, COMMON_OFFSET + common::QUEUE_DESC, 8),
            DESC
        );
        assert_eq!(
            dev.bar_read(BAR, COMMON_OFFSET + common::QUEUE_ENABLE, 2),
            1
        );

        // Read sector 1 into DATA: header, data (device-writable), status (device-writable).
        let mut header = [0u8; 16];
        header[0..4].copy_from_slice(&VIRTIO_BLK_T_IN.to_le_bytes());
        header[8..16].copy_from_slice(&1u64.to_le_bytes());
        memory.write(HEADER, &header).unwrap();
        write_descriptor(&mut memory, 0, HEADER, 16, VRING_DESC_F_NEXT, 1);
        write_descriptor(
            &mut memory,
            1,
            DATA,
            512,
            VRING_DESC_F_WRITE | VRING_DESC_F_NEXT,
            2,
        );
        write_descriptor(&mut memory, 2, STATUS, 1, VRING_DESC_F_WRITE, 0);
        memory.write(STATUS, &[0xff]).unwrap();
        // Avail: idx 1, ring[0] = descriptor 0.
        memory.write(AVAIL + 2, &1u16.to_le_bytes()).unwrap();
        memory.write(AVAIL + 4, &0u16.to_le_bytes()).unwrap();

        dev.bar_write(BAR, NOTIFY_OFFSET, 2, 0);

        let mut data = [0u8; 512];
        memory.read(DATA, &mut data).unwrap();
        assert_eq!(data, [0x37u8; 512]);
        let mut status = [0xffu8];
        memory.read(STATUS, &mut status).unwrap();
        assert_eq!(status[0], VIRTIO_BLK_S_OK);
        let mut used_idx = [0u8; 2];
        memory.read(USED + 2, &mut used_idx).unwrap();
        assert_eq!(
            u16::from_le_bytes(used_idx),
            1,
            "the used ring records the request"
        );
    }

    #[test]
    fn the_device_configuration_reports_capacity_and_sector_size() {
        let (mut dev, _) = device(alloc::vec![0; 2048]);
        // Eight-byte capacity is one access, which the transport splits into two words.
        assert_eq!(dev.bar_read(BAR, DEVICE_OFFSET, 8), 4);
        assert_eq!(dev.bar_read(BAR, DEVICE_OFFSET + 0x14, 4), 512);
        assert_eq!(dev.bar_read(BAR, DEVICE_OFFSET, 4), 4);
    }

    #[test]
    fn resetting_the_status_disables_the_queue() {
        let (mut dev, _) = device(alloc::vec![0; 2048]);
        negotiate(&mut dev);
        assert_eq!(
            dev.bar_read(BAR, COMMON_OFFSET + common::QUEUE_ENABLE, 2),
            1
        );
        dev.bar_write(BAR, COMMON_OFFSET + common::DEVICE_STATUS, 1, 0);
        assert_eq!(
            dev.bar_read(BAR, COMMON_OFFSET + common::QUEUE_ENABLE, 2),
            0
        );
        assert_eq!(
            dev.bar_read(BAR, COMMON_OFFSET + common::DRIVER_FEATURE, 4),
            0
        );
    }
}
