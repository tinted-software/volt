//! VirtIO block device: the request queue, shared by its transports.
//!
//! `VirtioBlock` is the VirtIO-MMIO v2 transport (the Linux virt board's `virtio,mmio`
//! device). `virtio_pci::VirtioBlockPci` is the same device behind the modern PCI transport.
//! Both own the register state and hand the queue to a [`BlockQueue`], which serves the
//! requests the driver made available.

use crate::memory::{GuestMemory, SharedMemory};
use alloc::vec::Vec;

pub const MAGIC_VALUE: u32 = 0x74726976; // "virt"
pub const VERSION: u32 = 2; // Modern VirtIO v2
pub const DEVICE_ID_BLOCK: u32 = 2; // Block device
pub const VENDOR_ID: u32 = 0x554d4551; // "QEMU"

// Feature bits (virtio 1.2, section 5.2.3)
pub const VIRTIO_BLK_F_RO: u64 = 1 << 5;
pub const VIRTIO_BLK_F_BLK_SIZE: u64 = 1 << 6;
pub const VIRTIO_BLK_F_FLUSH: u64 = 1 << 9;
pub const VIRTIO_F_VERSION_1: u64 = 1 << 32;

// Request types
pub const VIRTIO_BLK_T_IN: u32 = 0;
pub const VIRTIO_BLK_T_OUT: u32 = 1;
pub const VIRTIO_BLK_T_FLUSH: u32 = 4;
pub const VIRTIO_BLK_T_GET_ID: u32 = 8;

// Status codes
pub const VIRTIO_BLK_S_OK: u8 = 0;
pub const VIRTIO_BLK_S_IOERR: u8 = 1;

// Virtqueue descriptor flags
pub const VRING_DESC_F_NEXT: u16 = 1;
pub const VRING_DESC_F_WRITE: u16 = 2;

pub const QUEUE_MAX_SIZE: u32 = 128;
/// Block addresses are in sectors of this many bytes.
pub const SECTOR_SIZE: u64 = 512;

/// Storage backend for a block device.
pub trait BlockBackend: Send {
    fn len(&self) -> u64;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<(), ()>;
    fn write_at(&mut self, offset: u64, buf: &[u8]) -> Result<(), ()>;
    fn flush(&mut self) -> Result<(), ()>;
}

impl BlockBackend for Vec<u8> {
    fn len(&self) -> u64 {
        self.len() as u64
    }

    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<(), ()> {
        let start = offset as usize;
        let end = start + buf.len();
        if end <= self.len() {
            buf.copy_from_slice(&self[start..end]);
            Ok(())
        } else {
            Err(())
        }
    }

    fn write_at(&mut self, offset: u64, buf: &[u8]) -> Result<(), ()> {
        let start = offset as usize;
        let end = start + buf.len();
        if end > self.len() {
            self.resize(end, 0);
        }
        self[start..end].copy_from_slice(buf);
        Ok(())
    }

    fn flush(&mut self) -> Result<(), ()> {
        Ok(())
    }
}

/// The virtqueue a transport configures: its size, its three rings, and the progress
/// the device has made through them.
#[derive(Clone, Copy, Default)]
pub struct Queue {
    pub num: u32,
    pub ready: bool,
    pub desc_table: u64,
    pub avail_ring: u64,
    pub used_ring: u64,
    pub last_avail_idx: u16,
    pub last_used_idx: u16,
}

/// The backing store, the guest memory the rings and buffers live in, and the one request
/// queue. Transports set `queue` from their registers and call [`BlockQueue::process`]
/// when the driver kicks it.
pub struct BlockQueue<B> {
    pub backend: B,
    memory: SharedMemory,
    pub queue: Queue,
}

impl<B: BlockBackend> BlockQueue<B> {
    pub fn new(backend: B, memory: SharedMemory) -> Self {
        Self {
            backend,
            memory,
            queue: Queue::default(),
        }
    }

    /// The disk's capacity in sectors, as the device configuration reports it.
    pub fn capacity_sectors(&self) -> u64 {
        self.backend.len() / SECTOR_SIZE
    }

    /// Serves every request the driver made available. Returns whether any was served.
    pub fn process(&mut self) -> bool {
        if self.queue.num == 0 || !self.queue.ready {
            return false;
        }

        let avail_addr = self.queue.avail_ring;
        let desc_addr = self.queue.desc_table;
        let used_addr = self.queue.used_ring;
        let q_num = self.queue.num as u64;

        // Driver increments idx in avail ring
        let avail_idx = self.read_guest_u16(avail_addr + 2);
        let mut processed_any = false;

        while self.queue.last_avail_idx != avail_idx {
            let ring_slot = (self.queue.last_avail_idx as u64) % q_num;
            let head_idx = self.read_guest_u16(avail_addr + 4 + ring_slot * 2) as u64;

            // Walk descriptor chain
            let mut curr = head_idx;
            let mut req_header_addr = None;
            let mut data_buffers: Vec<(u64, u32, bool)> = Vec::new();
            let mut status_addr = None;

            for _ in 0..q_num {
                let d_addr = desc_addr + curr * 16;
                let addr = self.read_guest_u64(d_addr);
                let len = self.read_guest_u32(d_addr + 8);
                let flags = self.read_guest_u16(d_addr + 12);
                let next = self.read_guest_u16(d_addr + 14) as u64;

                let is_write = (flags & VRING_DESC_F_WRITE) != 0;
                let has_next = (flags & VRING_DESC_F_NEXT) != 0;

                if req_header_addr.is_none() {
                    // First descriptor: virtio_blk_req header
                    req_header_addr = Some(addr);
                } else if !has_next && is_write && len == 1 {
                    // Last descriptor in chain: status byte
                    status_addr = Some(addr);
                } else {
                    // Data buffer
                    data_buffers.push((addr, len, is_write));
                }

                if !has_next {
                    break;
                }
                curr = next;
            }

            let mut total_written = 0u32;
            let mut status = VIRTIO_BLK_S_OK;

            if let Some(req_addr) = req_header_addr {
                let req_type = self.read_guest_u32(req_addr);
                let sector = self.read_guest_u64(req_addr + 8);
                let mut disk_offset = sector.saturating_mul(SECTOR_SIZE);

                match req_type {
                    VIRTIO_BLK_T_IN => {
                        // Read from disk to guest memory
                        for (buf_addr, len, _) in data_buffers {
                            let mut tmp = alloc::vec![0u8; len as usize];
                            if self.backend.read_at(disk_offset, &mut tmp).is_ok() {
                                let _ = self.memory.write(buf_addr, &tmp);
                                total_written = total_written.saturating_add(len);
                            } else {
                                status = VIRTIO_BLK_S_IOERR;
                            }
                            disk_offset = disk_offset.saturating_add(len as u64);
                        }
                    }
                    VIRTIO_BLK_T_OUT => {
                        // Write from guest memory to disk
                        for (buf_addr, len, _) in data_buffers {
                            let mut tmp = alloc::vec![0u8; len as usize];
                            let _ = self.memory.read(buf_addr, &mut tmp);
                            if self.backend.write_at(disk_offset, &tmp).is_err() {
                                status = VIRTIO_BLK_S_IOERR;
                            }
                            disk_offset = disk_offset.saturating_add(len as u64);
                        }
                    }
                    VIRTIO_BLK_T_FLUSH => {
                        if self.backend.flush().is_err() {
                            status = VIRTIO_BLK_S_IOERR;
                        }
                    }
                    VIRTIO_BLK_T_GET_ID => {
                        for (buf_addr, len, _) in data_buffers {
                            let id = b"volt-blk\0";
                            let copy_len = (len as usize).min(id.len());
                            let _ = self.memory.write(buf_addr, &id[..copy_len]);
                            total_written = total_written.saturating_add(copy_len as u32);
                        }
                    }
                    _ => {
                        status = VIRTIO_BLK_S_IOERR;
                    }
                }
            } else {
                status = VIRTIO_BLK_S_IOERR;
            }

            // Write status byte (1 byte, written by device)
            if let Some(s_addr) = status_addr {
                let _ = self.memory.write(s_addr, &[status]);
                total_written = total_written.saturating_add(1);
            }

            // Write entry to used ring
            let used_slot = (self.queue.last_used_idx as u64) % q_num;
            let elem_addr = used_addr + 4 + used_slot * 8;
            self.write_guest_u32(elem_addr, head_idx as u32);
            self.write_guest_u32(elem_addr + 4, total_written);

            self.queue.last_used_idx = self.queue.last_used_idx.wrapping_add(1);
            self.write_guest_u16(used_addr + 2, self.queue.last_used_idx);

            self.queue.last_avail_idx = self.queue.last_avail_idx.wrapping_add(1);
            processed_any = true;
        }

        processed_any
    }

    fn read_guest_u16(&self, addr: u64) -> u16 {
        let mut buf = [0u8; 2];
        let _ = self.memory.read(addr, &mut buf);
        u16::from_le_bytes(buf)
    }

    fn read_guest_u32(&self, addr: u64) -> u32 {
        let mut buf = [0u8; 4];
        let _ = self.memory.read(addr, &mut buf);
        u32::from_le_bytes(buf)
    }

    fn read_guest_u64(&self, addr: u64) -> u64 {
        let mut buf = [0u8; 8];
        let _ = self.memory.read(addr, &mut buf);
        u64::from_le_bytes(buf)
    }

    fn write_guest_u16(&mut self, addr: u64, val: u16) {
        let _ = self.memory.write(addr, &val.to_le_bytes());
    }

    fn write_guest_u32(&mut self, addr: u64, val: u32) {
        let _ = self.memory.write(addr, &val.to_le_bytes());
    }
}

/// The VirtIO-MMIO v2 transport for the block device.
pub struct VirtioBlock<B, L = fn(bool)> {
    core: BlockQueue<B>,
    status: u32,
    device_features_sel: u32,
    driver_features_sel: u32,
    driver_features: u64,
    queue_sel: u32,
    interrupt_status: u32,
    line: Option<L>,
}

impl<B: BlockBackend> VirtioBlock<B, fn(bool)> {
    pub fn new(backend: B, memory: SharedMemory) -> Self {
        Self {
            core: BlockQueue::new(backend, memory),
            status: 0,
            device_features_sel: 0,
            driver_features_sel: 0,
            driver_features: 0,
            queue_sel: 0,
            interrupt_status: 0,
            line: None,
        }
    }
}

impl<B: BlockBackend, L: FnMut(bool)> VirtioBlock<B, L> {
    pub fn with_line<L2: FnMut(bool)>(self, line: L2) -> VirtioBlock<B, L2> {
        VirtioBlock {
            core: self.core,
            status: self.status,
            device_features_sel: self.device_features_sel,
            driver_features_sel: self.driver_features_sel,
            driver_features: self.driver_features,
            queue_sel: self.queue_sel,
            interrupt_status: self.interrupt_status,
            line: Some(line),
        }
    }

    fn raise_irq(&mut self) {
        self.interrupt_status |= 1;
        if let Some(line) = &mut self.line {
            line(true);
        }
    }

    fn lower_irq(&mut self) {
        self.interrupt_status = 0;
        if let Some(line) = &mut self.line {
            line(false);
        }
    }

    pub fn read(&mut self, offset: u64, size: u8) -> u64 {
        let value = match offset {
            0x000 => MAGIC_VALUE as u64,
            0x004 => VERSION as u64,
            0x008 => DEVICE_ID_BLOCK as u64,
            0x00c => VENDOR_ID as u64,
            0x010 => {
                let features = VIRTIO_BLK_F_FLUSH | VIRTIO_BLK_F_BLK_SIZE | VIRTIO_F_VERSION_1;
                if self.device_features_sel == 0 {
                    (features & 0xffff_ffff) as u64
                } else {
                    (features >> 32) as u64
                }
            }
            0x034 => {
                if self.queue_sel == 0 {
                    QUEUE_MAX_SIZE as u64
                } else {
                    0
                }
            }
            0x044 => {
                if self.queue_sel == 0 && self.core.queue.ready {
                    1
                } else {
                    0
                }
            }
            0x060 => self.interrupt_status as u64,
            0x070 => self.status as u64,
            0x0fc => 0, // config generation
            // Configuration space:
            // 0x100..0x108: capacity in 512-byte sectors (u64)
            0x100..=0x107 => {
                let capacity = self.core.capacity_sectors();
                let shift = (offset - 0x100) * 8;
                capacity >> shift
            }
            // 0x114..=0x117: blk_size (512)
            0x114 => 512,
            _ => 0,
        };
        // Mask to access width
        match size {
            1 => value & 0xff,
            2 => value & 0xffff,
            4 => value & 0xffff_ffff,
            _ => value,
        }
    }

    pub fn write(&mut self, offset: u64, size: u8, value: u64) {
        let _ = size;
        match offset {
            0x014 => self.device_features_sel = value as u32,
            0x020 => {
                if self.driver_features_sel == 0 {
                    self.driver_features =
                        (self.driver_features & !0xffff_ffff) | (value & 0xffff_ffff);
                } else {
                    self.driver_features = (self.driver_features & 0xffff_ffff) | (value << 32);
                }
            }
            0x024 => self.driver_features_sel = value as u32,
            0x030 => self.queue_sel = value as u32,
            0x038 => {
                if self.queue_sel == 0 {
                    self.core.queue.num = (value as u32).min(QUEUE_MAX_SIZE);
                }
            }
            0x044 => {
                if self.queue_sel == 0 {
                    self.core.queue.ready = (value & 1) != 0;
                }
            }
            0x050 => {
                let queue_idx = value as u32;
                if queue_idx == 0 && self.core.queue.ready && self.core.process() {
                    self.raise_irq();
                }
            }
            0x064 => {
                self.interrupt_status &= !(value as u32);
                if self.interrupt_status == 0 {
                    self.lower_irq();
                }
            }
            0x070 => {
                self.status = value as u32;
                if self.status == 0 {
                    // Reset
                    self.core.queue = Queue::default();
                    self.lower_irq();
                }
            }
            0x080 => {
                self.core.queue.desc_table =
                    (self.core.queue.desc_table & !0xffff_ffff) | (value & 0xffff_ffff)
            }
            0x084 => {
                self.core.queue.desc_table =
                    (self.core.queue.desc_table & 0xffff_ffff) | (value << 32)
            }
            0x090 => {
                self.core.queue.avail_ring =
                    (self.core.queue.avail_ring & !0xffff_ffff) | (value & 0xffff_ffff)
            }
            0x094 => {
                self.core.queue.avail_ring =
                    (self.core.queue.avail_ring & 0xffff_ffff) | (value << 32)
            }
            0x0a0 => {
                self.core.queue.used_ring =
                    (self.core.queue.used_ring & !0xffff_ffff) | (value & 0xffff_ffff)
            }
            0x0a4 => {
                self.core.queue.used_ring =
                    (self.core.queue.used_ring & 0xffff_ffff) | (value << 32)
            }
            _ => {}
        }
    }
}

pub trait VirtioIo {
    fn read(&mut self, offset: u64, size: u8) -> u64;
    fn write(&mut self, offset: u64, size: u8, value: u64);
}

impl<B: BlockBackend, L: FnMut(bool)> VirtioIo for VirtioBlock<B, L> {
    fn read(&mut self, offset: u64, size: u8) -> u64 {
        self.read(offset, size)
    }
    fn write(&mut self, offset: u64, size: u8, value: u64) {
        self.write(offset, size, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{PhysicalMemory, Region};

    fn make_test_device() -> (VirtioBlock<Vec<u8>>, SharedMemory) {
        let ram = Region::ram(0x4000_0000, alloc::vec![0u8; 1024 * 1024]).unwrap();
        let phys = PhysicalMemory::new(alloc::vec![ram]).unwrap();
        let shared = SharedMemory::new(phys);
        let disk = alloc::vec![0x42u8; 2048]; // 4 sectors
        let dev = VirtioBlock::new(disk, shared.clone());
        (dev, shared)
    }

    #[test]
    fn virtio_blk_registers_and_capacity() {
        let (mut dev, _) = make_test_device();
        assert_eq!(dev.read(0x000, 4), MAGIC_VALUE as u64);
        assert_eq!(dev.read(0x004, 4), 2);
        assert_eq!(dev.read(0x008, 4), 2);
        assert_eq!(dev.read(0x00c, 4), VENDOR_ID as u64);

        // Capacity: 2048 bytes / 512 = 4 sectors
        assert_eq!(dev.read(0x100, 8), 4);
    }
    #[test]
    fn virtio_blk_processes_read_request() {
        use alloc::sync::Arc;
        use core::sync::atomic::{AtomicBool, Ordering};

        let (dev, mut mem) = make_test_device();
        let irq_raised = Arc::new(AtomicBool::new(false));
        let irq_flag = irq_raised.clone();
        let mut dev = dev.with_line(move |level| irq_flag.store(level, Ordering::Release));

        // Setup queue at RAM 0x4001_0000
        let desc_table = 0x4001_0000u64;
        let avail_ring = 0x4001_1000u64;
        let used_ring = 0x4001_2000u64;

        dev.write(0x030, 4, 0); // QueueSel = 0
        dev.write(0x038, 4, 16); // QueueNum = 16
        dev.write(0x080, 4, desc_table & 0xffff_ffff);
        dev.write(0x084, 4, desc_table >> 32);
        dev.write(0x090, 4, avail_ring & 0xffff_ffff);
        dev.write(0x094, 4, avail_ring >> 32);
        dev.write(0x0a0, 4, used_ring & 0xffff_ffff);
        dev.write(0x0a4, 4, used_ring >> 32);
        dev.write(0x044, 4, 1); // QueueReady = 1

        // Request header at 0x4002_0000: type=0 (IN), reserved=0, sector=1
        let req_header_addr = 0x4002_0000u64;
        let mut req_header = [0u8; 16];
        req_header[0..4].copy_from_slice(&VIRTIO_BLK_T_IN.to_le_bytes());
        req_header[8..16].copy_from_slice(&1u64.to_le_bytes()); // sector 1
        mem.write(req_header_addr, &req_header).unwrap();

        // Data buffer at 0x4002_1000: 512 bytes
        let data_buf_addr = 0x4002_1000u64;

        // Status byte at 0x4002_2000: 1 byte
        let status_addr = 0x4002_2000u64;
        mem.write(status_addr, &[0xff]).unwrap();

        // Descriptors:
        // desc 0: header (read-only, next=1)
        let mut d0 = [0u8; 16];
        d0[0..8].copy_from_slice(&req_header_addr.to_le_bytes());
        d0[8..12].copy_from_slice(&16u32.to_le_bytes());
        d0[12..14].copy_from_slice(&VRING_DESC_F_NEXT.to_le_bytes());
        d0[14..16].copy_from_slice(&1u16.to_le_bytes());
        mem.write(desc_table, &d0).unwrap();

        // desc 1: data (device-write, next=2)
        let mut d1 = [0u8; 16];
        d1[0..8].copy_from_slice(&data_buf_addr.to_le_bytes());
        d1[8..12].copy_from_slice(&512u32.to_le_bytes());
        d1[12..14].copy_from_slice(&(VRING_DESC_F_WRITE | VRING_DESC_F_NEXT).to_le_bytes());
        d1[14..16].copy_from_slice(&2u16.to_le_bytes());
        mem.write(desc_table + 16, &d1).unwrap();

        // desc 2: status (device-write, no next)
        let mut d2 = [0u8; 16];
        d2[0..8].copy_from_slice(&status_addr.to_le_bytes());
        d2[8..12].copy_from_slice(&1u32.to_le_bytes());
        d2[12..14].copy_from_slice(&VRING_DESC_F_WRITE.to_le_bytes());
        mem.write(desc_table + 32, &d2).unwrap();

        // Avail ring: idx=1, ring[0]=0
        let mut avail = [0u8; 6];
        avail[2..4].copy_from_slice(&1u16.to_le_bytes()); // idx = 1
        avail[4..6].copy_from_slice(&0u16.to_le_bytes()); // ring[0] = desc 0
        mem.write(avail_ring, &avail).unwrap();

        // Notify queue 0
        dev.write(0x050, 4, 0);

        // Check IRQ
        assert!(irq_raised.load(Ordering::Acquire));
        assert_eq!(dev.read(0x060, 4), 1); // InterruptStatus

        // Check data was read from sector 1 (0x42)
        let mut read_data = [0u8; 512];
        mem.read(data_buf_addr, &mut read_data).unwrap();
        assert_eq!(read_data, [0x42u8; 512]);

        // Check status byte is VIRTIO_BLK_S_OK (0)
        let mut status = [0xffu8];
        mem.read(status_addr, &mut status).unwrap();
        assert_eq!(status[0], VIRTIO_BLK_S_OK);

        // Ack interrupt
        dev.write(0x064, 4, 1);
        assert_eq!(dev.read(0x060, 4), 0);
        assert!(!irq_raised.load(Ordering::Acquire));
    }
}
