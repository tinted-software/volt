//! VirtIO-MMIO v2 GPU Device (`VIRTIO_ID_GPU = 16`).
//!
//! Implements VirtIO 1.0 modern MMIO transport and the 2D command set:
//! `GET_DISPLAY_INFO`, `RESOURCE_CREATE_2D`, `RESOURCE_UNREF`,
//! `RESOURCE_ATTACH_BACKING`, `RESOURCE_DETACH_BACKING`, `SET_SCANOUT`,
//! `TRANSFER_TO_HOST_2D`, and `RESOURCE_FLUSH`.
//!
//! Resources are backed by `volt-hal` images (using `ExtMem` when guest
//! backing pages are contiguous host-backed RAM) and scanout presentation
//! goes through `volt_platform::Surface`.

use crate::devices::virtio_blk::{
    MAGIC_VALUE, QUEUE_MAX_SIZE, VENDOR_ID, VERSION, VIRTIO_F_VERSION_1, VRING_DESC_F_NEXT,
    VRING_DESC_F_WRITE, VirtioIo,
};
use crate::memory::GuestMemory;
use alloc::vec;
use alloc::vec::Vec;
use hashbrown::HashMap;
use volt_hal::{Device as HalDevice, Format as HalFormat, ImageId};
use volt_platform::{PixelFormat, Surface};

pub const DEVICE_ID_GPU: u32 = 16;

// Control queue (0) and cursor queue (1)
const NUM_QUEUES: usize = 2;

// 2D command types
pub const VIRTIO_GPU_CMD_GET_DISPLAY_INFO: u32 = 0x0100;
pub const VIRTIO_GPU_CMD_RESOURCE_CREATE_2D: u32 = 0x0101;
pub const VIRTIO_GPU_CMD_RESOURCE_UNREF: u32 = 0x0102;
pub const VIRTIO_GPU_CMD_SET_SCANOUT: u32 = 0x0103;
pub const VIRTIO_GPU_CMD_RESOURCE_FLUSH: u32 = 0x0104;
pub const VIRTIO_GPU_CMD_TRANSFER_TO_HOST_2D: u32 = 0x0105;
pub const VIRTIO_GPU_CMD_RESOURCE_ATTACH_BACKING: u32 = 0x0106;
pub const VIRTIO_GPU_CMD_RESOURCE_DETACH_BACKING: u32 = 0x0107;
pub const VIRTIO_GPU_CMD_UPDATE_CURSOR: u32 = 0x0300;
pub const VIRTIO_GPU_CMD_MOVE_CURSOR: u32 = 0x0301;

// Response types
pub const VIRTIO_GPU_RESP_OK_NODATA: u32 = 0x1100;
pub const VIRTIO_GPU_RESP_OK_DISPLAY_INFO: u32 = 0x1101;
pub const VIRTIO_GPU_RESP_ERR_UNSPEC: u32 = 0x1200;
pub const VIRTIO_GPU_RESP_ERR_INVALID_SCANOUT_ID: u32 = 0x1202;
pub const VIRTIO_GPU_RESP_ERR_INVALID_RESOURCE_ID: u32 = 0x1203;
pub const VIRTIO_GPU_RESP_ERR_INVALID_PARAMETER: u32 = 0x1205;

// Pixel formats (`virtio_gpu_formats`)
pub const VIRTIO_GPU_FORMAT_B8G8R8A8_UNORM: u32 = 1;
pub const VIRTIO_GPU_FORMAT_B8G8R8X8_UNORM: u32 = 2;
pub const VIRTIO_GPU_FORMAT_A8R8G8B8_UNORM: u32 = 3;
pub const VIRTIO_GPU_FORMAT_X8R8G8B8_UNORM: u32 = 4;
pub const VIRTIO_GPU_FORMAT_R8G8B8A8_UNORM: u32 = 67;
pub const VIRTIO_GPU_FORMAT_X8B8G8R8_UNORM: u32 = 68;
pub const VIRTIO_GPU_FORMAT_A8B8G8R8_UNORM: u32 = 121;
pub const VIRTIO_GPU_FORMAT_R8G8B8X8_UNORM: u32 = 134;

const DEFAULT_WIDTH: u32 = 1024;
const DEFAULT_HEIGHT: u32 = 768;
const CTRL_HDR_SIZE: usize = 24;

#[derive(Clone, Copy, Default)]
struct Queue {
    num: u32,
    ready: bool,
    desc_table: u64,
    avail_ring: u64,
    used_ring: u64,
    last_avail_idx: u16,
}

#[derive(Clone, Copy)]
struct BackingEntry {
    addr: u64,
    len: u32,
}

struct GpuResource {
    width: u32,
    height: u32,
    pixel_format: PixelFormat,
    hal_image: ImageId,
    backing: Vec<BackingEntry>,
    /// True when `hal_image` aliases guest RAM directly via `ExtMem`.
    zero_copy_extmem: bool,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Default)]
struct Scanout {
    resource_id: u32,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

pub struct VirtioGpu<M: GuestMemory, S: Surface, L = fn(bool)> {
    memory: M,
    surface: S,
    hal: HalDevice,
    resources: HashMap<u32, GpuResource>,
    scanout: Scanout,
    display_width: u32,
    display_height: u32,
    status: u32,
    device_features_sel: u32,
    driver_features: u64,
    driver_features_sel: u32,
    queue_sel: u32,
    queues: [Queue; NUM_QUEUES],
    interrupt_status: u32,
    line: Option<L>,
}

impl<M: GuestMemory, S: Surface> VirtioGpu<M, S, fn(bool)> {
    pub fn new(memory: M, surface: S) -> Self {
        Self {
            memory,
            surface,
            hal: HalDevice::new(),
            resources: HashMap::new(),
            scanout: Scanout::default(),
            display_width: DEFAULT_WIDTH,
            display_height: DEFAULT_HEIGHT,
            status: 0,
            device_features_sel: 0,
            driver_features: 0,
            driver_features_sel: 0,
            queue_sel: 0,
            queues: [Queue::default(); NUM_QUEUES],
            interrupt_status: 0,
            line: None,
        }
    }
}

impl<M: GuestMemory, S: Surface, L: FnMut(bool)> VirtioGpu<M, S, L> {
    pub fn with_resolution(mut self, width: u32, height: u32) -> Self {
        self.display_width = width.max(1);
        self.display_height = height.max(1);
        self
    }

    pub fn with_line<L2: FnMut(bool)>(self, line: L2) -> VirtioGpu<M, S, L2> {
        VirtioGpu {
            memory: self.memory,
            surface: self.surface,
            hal: self.hal,
            resources: self.resources,
            scanout: self.scanout,
            display_width: self.display_width,
            display_height: self.display_height,
            status: self.status,
            device_features_sel: self.device_features_sel,
            driver_features: self.driver_features,
            driver_features_sel: self.driver_features_sel,
            queue_sel: self.queue_sel,
            queues: self.queues,
            interrupt_status: self.interrupt_status,
            line: Some(line),
        }
    }

    pub fn surface(&self) -> &S {
        &self.surface
    }

    pub fn surface_mut(&mut self) -> &mut S {
        &mut self.surface
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
        let q = self.queue_sel as usize;
        let value = match offset {
            0x000 => MAGIC_VALUE as u64,
            0x004 => VERSION as u64,
            0x008 => DEVICE_ID_GPU as u64,
            0x00c => VENDOR_ID as u64,
            0x010 => {
                let features = VIRTIO_F_VERSION_1;
                if self.device_features_sel == 0 {
                    features & 0xffff_ffff
                } else {
                    features >> 32
                }
            }
            0x034 => {
                if q < NUM_QUEUES {
                    QUEUE_MAX_SIZE as u64
                } else {
                    0
                }
            }
            0x044 => {
                if q < NUM_QUEUES && self.queues[q].ready {
                    1
                } else {
                    0
                }
            }
            0x060 => self.interrupt_status as u64,
            0x070 => self.status as u64,
            0x0fc => 0,
            // virtio_gpu_config: events_read (0x100), events_clear (0x104), num_scanouts (0x108), num_capsets (0x10c)
            0x100 => 0,
            0x104 => 0,
            0x108 => 1,
            0x10c => 0,
            _ => 0,
        };
        match size {
            1 => value & 0xff,
            2 => value & 0xffff,
            4 => value & 0xffff_ffff,
            _ => value,
        }
    }

    pub fn write(&mut self, offset: u64, size: u8, value: u64) {
        let _ = size;
        let q = self.queue_sel as usize;
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
                if q < NUM_QUEUES {
                    self.queues[q].num = (value as u32).min(QUEUE_MAX_SIZE);
                }
            }
            0x044 => {
                if q < NUM_QUEUES {
                    self.queues[q].ready = (value & 1) != 0;
                }
            }
            0x050 => {
                let queue_idx = value as usize;
                if queue_idx < NUM_QUEUES && self.queues[queue_idx].ready {
                    self.process_queue(queue_idx);
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
                    self.queues = [Queue::default(); NUM_QUEUES];
                    let ids: Vec<u32> = self.resources.keys().copied().collect();
                    for id in ids {
                        if let Some(res) = self.resources.remove(&id) {
                            self.hal.destroy_image(res.hal_image);
                        }
                    }
                    self.scanout = Scanout::default();
                    self.lower_irq();
                }
            }
            0x080 => {
                if q < NUM_QUEUES {
                    self.queues[q].desc_table =
                        (self.queues[q].desc_table & !0xffff_ffff) | (value & 0xffff_ffff);
                }
            }
            0x084 => {
                if q < NUM_QUEUES {
                    self.queues[q].desc_table =
                        (self.queues[q].desc_table & 0xffff_ffff) | (value << 32);
                }
            }
            0x090 => {
                if q < NUM_QUEUES {
                    self.queues[q].avail_ring =
                        (self.queues[q].avail_ring & !0xffff_ffff) | (value & 0xffff_ffff);
                }
            }
            0x094 => {
                if q < NUM_QUEUES {
                    self.queues[q].avail_ring =
                        (self.queues[q].avail_ring & 0xffff_ffff) | (value << 32);
                }
            }
            0x0a0 => {
                if q < NUM_QUEUES {
                    self.queues[q].used_ring =
                        (self.queues[q].used_ring & !0xffff_ffff) | (value & 0xffff_ffff);
                }
            }
            0x0a4 if q < NUM_QUEUES => {
                self.queues[q].used_ring = (self.queues[q].used_ring & 0xffff_ffff) | (value << 32);
            }
            _ => {}
        }
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

    fn process_queue(&mut self, qi: usize) {
        let q = self.queues[qi];
        if q.num == 0 || q.desc_table == 0 || q.avail_ring == 0 || q.used_ring == 0 {
            return;
        }
        let avail_idx = self.read_guest_u16(q.avail_ring + 2);
        let mut raised = false;

        while self.queues[qi].last_avail_idx != avail_idx {
            let ring_idx = (self.queues[qi].last_avail_idx as u32) % q.num;
            let head_desc_idx = self.read_guest_u16(q.avail_ring + 4 + (ring_idx as u64) * 2);

            let mut req_buf = Vec::new();
            let mut resp_bufs: Vec<(u64, u32)> = Vec::new();

            let mut desc_idx = head_desc_idx;
            let mut steps = 0u32;
            loop {
                if steps >= q.num {
                    break;
                }
                steps += 1;
                let desc_addr = q.desc_table + (desc_idx as u64) * 16;
                let addr = self.read_guest_u64(desc_addr);
                let len = self.read_guest_u32(desc_addr + 8);
                let flags = self.read_guest_u16(desc_addr + 12);
                let next = self.read_guest_u16(desc_addr + 14);

                if (flags & VRING_DESC_F_WRITE) != 0 {
                    resp_bufs.push((addr, len));
                } else {
                    let start = req_buf.len();
                    req_buf.resize(start + len as usize, 0);
                    let _ = self.memory.read(addr, &mut req_buf[start..]);
                }

                if (flags & VRING_DESC_F_NEXT) != 0 {
                    desc_idx = next;
                } else {
                    break;
                }
            }

            let response = self.handle_command(&req_buf);
            let mut written = 0usize;
            for (addr, len) in resp_bufs {
                if written >= response.len() {
                    break;
                }
                let take = (len as usize).min(response.len() - written);
                let _ = self.memory.write(addr, &response[written..written + take]);
                written += take;
            }

            let used_idx = self.read_guest_u16(q.used_ring + 2);
            let used_ring_idx = (used_idx as u32) % q.num;
            let used_elem_addr = q.used_ring + 4 + (used_ring_idx as u64) * 8;
            self.write_guest_u32(used_elem_addr, head_desc_idx as u32);
            self.write_guest_u32(used_elem_addr + 4, written as u32);
            self.write_guest_u16(q.used_ring + 2, used_idx.wrapping_add(1));

            self.queues[qi].last_avail_idx = self.queues[qi].last_avail_idx.wrapping_add(1);
            raised = true;
        }

        if raised {
            self.raise_irq();
        }
    }

    fn handle_command(&mut self, req: &[u8]) -> Vec<u8> {
        if req.len() < CTRL_HDR_SIZE {
            return ctrl_header(VIRTIO_GPU_RESP_ERR_UNSPEC, 0, 0);
        }
        let cmd_type = u32_le(req, 0);
        let flags = u32_le(req, 4);
        let fence_id = u64_le(req, 8);

        let resp_type = match cmd_type {
            VIRTIO_GPU_CMD_GET_DISPLAY_INFO => {
                // Header (24 bytes) + 16 * virtio_gpu_display_one (24 bytes each) = 408 bytes
                let mut out = ctrl_header(VIRTIO_GPU_RESP_OK_DISPLAY_INFO, flags, fence_id);
                out.resize(CTRL_HDR_SIZE + 16 * 24, 0);
                // Scanout 0: r { x: 0, y: 0, width, height }, enabled = 1, flags = 0
                out[CTRL_HDR_SIZE + 8..CTRL_HDR_SIZE + 12]
                    .copy_from_slice(&self.display_width.to_le_bytes());
                out[CTRL_HDR_SIZE + 12..CTRL_HDR_SIZE + 16]
                    .copy_from_slice(&self.display_height.to_le_bytes());
                out[CTRL_HDR_SIZE + 16..CTRL_HDR_SIZE + 20].copy_from_slice(&1u32.to_le_bytes());
                return out;
            }
            VIRTIO_GPU_CMD_RESOURCE_CREATE_2D => self.cmd_resource_create_2d(req),
            VIRTIO_GPU_CMD_RESOURCE_UNREF => self.cmd_resource_unref(req),
            VIRTIO_GPU_CMD_SET_SCANOUT => self.cmd_set_scanout(req),
            VIRTIO_GPU_CMD_RESOURCE_FLUSH => self.cmd_resource_flush(req),
            VIRTIO_GPU_CMD_TRANSFER_TO_HOST_2D => self.cmd_transfer_to_host_2d(req),
            VIRTIO_GPU_CMD_RESOURCE_ATTACH_BACKING => self.cmd_attach_backing(req),
            VIRTIO_GPU_CMD_RESOURCE_DETACH_BACKING => self.cmd_detach_backing(req),
            VIRTIO_GPU_CMD_UPDATE_CURSOR | VIRTIO_GPU_CMD_MOVE_CURSOR => VIRTIO_GPU_RESP_OK_NODATA,
            _ => VIRTIO_GPU_RESP_ERR_UNSPEC,
        };
        ctrl_header(resp_type, flags, fence_id)
    }

    fn cmd_resource_create_2d(&mut self, req: &[u8]) -> u32 {
        // hdr(24) + resource_id(4) + format(4) + width(4) + height(4) = 40
        if req.len() < 40 {
            return VIRTIO_GPU_RESP_ERR_INVALID_PARAMETER;
        }
        let resource_id = u32_le(req, 24);
        let format = u32_le(req, 28);
        let width = u32_le(req, 32);
        let height = u32_le(req, 36);
        if resource_id == 0 || width == 0 || height == 0 {
            return VIRTIO_GPU_RESP_ERR_INVALID_PARAMETER;
        }
        let Some((hal_fmt, pixel_fmt)) = map_format(format) else {
            return VIRTIO_GPU_RESP_ERR_INVALID_PARAMETER;
        };
        if let Some(old) = self.resources.remove(&resource_id) {
            self.hal.destroy_image(old.hal_image);
        }
        let hal_image = self.hal.create_image(hal_fmt, width, height);
        self.resources.insert(
            resource_id,
            GpuResource {
                width,
                height,
                pixel_format: pixel_fmt,
                hal_image,
                backing: Vec::new(),
                zero_copy_extmem: false,
            },
        );
        VIRTIO_GPU_RESP_OK_NODATA
    }

    fn cmd_resource_unref(&mut self, req: &[u8]) -> u32 {
        // hdr(24) + resource_id(4) + padding(4) = 32
        if req.len() < 28 {
            return VIRTIO_GPU_RESP_ERR_INVALID_PARAMETER;
        }
        let resource_id = u32_le(req, 24);
        if let Some(res) = self.resources.remove(&resource_id) {
            self.hal.destroy_image(res.hal_image);
            if self.scanout.resource_id == resource_id {
                self.scanout.resource_id = 0;
            }
            VIRTIO_GPU_RESP_OK_NODATA
        } else {
            VIRTIO_GPU_RESP_ERR_INVALID_RESOURCE_ID
        }
    }

    fn cmd_attach_backing(&mut self, req: &[u8]) -> u32 {
        // hdr(24) + resource_id(4) + nr_entries(4) + nr_entries * (addr:8, length:4, pad:4)
        if req.len() < 32 {
            return VIRTIO_GPU_RESP_ERR_INVALID_PARAMETER;
        }
        let resource_id = u32_le(req, 24);
        let nr_entries = u32_le(req, 28) as usize;
        let expected_len = 32usize.saturating_add(nr_entries.saturating_mul(16));
        if req.len() < expected_len {
            return VIRTIO_GPU_RESP_ERR_INVALID_PARAMETER;
        }
        let Some(res) = self.resources.get_mut(&resource_id) else {
            return VIRTIO_GPU_RESP_ERR_INVALID_RESOURCE_ID;
        };
        res.backing.clear();
        res.backing.reserve(nr_entries);
        for i in 0..nr_entries {
            let off = 32 + i * 16;
            res.backing.push(BackingEntry {
                addr: u64_le(req, off),
                len: u32_le(req, off + 8),
            });
        }

        // Fast path: if single entry is contiguous in host-backed guest RAM and
        // covers the full linear image, alias it directly with `ExtMem`.
        let needed_bytes = (res.width as usize)
            .saturating_mul(res.height as usize)
            .saturating_mul(4);
        let mut ext_image = None;
        if res.backing.len() == 1
            && (res.backing[0].len as usize) >= needed_bytes
            && let Some(host_ptr) =
                contiguous_host_slice(&self.memory, res.backing[0].addr, needed_bytes)
        {
            let stride = res.width * 4;
            let hal_fmt = match res.pixel_format {
                PixelFormat::B8G8R8A8Unorm | PixelFormat::B8G8R8X8Unorm => HalFormat::B8G8R8A8Unorm,
                PixelFormat::R8G8B8A8Unorm | PixelFormat::R8G8B8X8Unorm => HalFormat::R8G8B8A8Unorm,
            };
            ext_image = Some((host_ptr, stride, hal_fmt, res.width, res.height));
        }
        if let Some((host_ptr, stride, hal_fmt, width, height)) = ext_image {
            if res.zero_copy_extmem {
                // SAFETY: `host_ptr` covers `needed_bytes` inside live guest RAM.
                unsafe {
                    self.hal
                        .set_image_external(res.hal_image, host_ptr, needed_bytes, stride);
                }
            } else {
                let old_id = res.hal_image;
                // SAFETY: `host_ptr` covers `needed_bytes` inside live guest RAM.
                let new_id = unsafe {
                    self.hal.create_image_external(
                        hal_fmt,
                        width,
                        height,
                        host_ptr,
                        needed_bytes,
                        stride,
                    )
                };
                res.hal_image = new_id;
                res.zero_copy_extmem = true;
                self.hal.destroy_image(old_id);
            }
        } else {
            res.zero_copy_extmem = false;
        }
        VIRTIO_GPU_RESP_OK_NODATA
    }

    fn cmd_detach_backing(&mut self, req: &[u8]) -> u32 {
        if req.len() < 28 {
            return VIRTIO_GPU_RESP_ERR_INVALID_PARAMETER;
        }
        let resource_id = u32_le(req, 24);
        let Some(res) = self.resources.get_mut(&resource_id) else {
            return VIRTIO_GPU_RESP_ERR_INVALID_RESOURCE_ID;
        };
        res.backing.clear();
        if res.zero_copy_extmem {
            let hal_fmt = match res.pixel_format {
                PixelFormat::B8G8R8A8Unorm | PixelFormat::B8G8R8X8Unorm => HalFormat::B8G8R8A8Unorm,
                PixelFormat::R8G8B8A8Unorm | PixelFormat::R8G8B8X8Unorm => HalFormat::R8G8B8A8Unorm,
            };
            let new_img = self.hal.create_image(hal_fmt, res.width, res.height);
            self.hal.destroy_image(res.hal_image);
            res.hal_image = new_img;
            res.zero_copy_extmem = false;
        }
        VIRTIO_GPU_RESP_OK_NODATA
    }

    fn cmd_set_scanout(&mut self, req: &[u8]) -> u32 {
        // hdr(24) + r: {x:4, y:4, w:4, h:4} + scanout_id(4) + resource_id(4) = 48
        if req.len() < 48 {
            return VIRTIO_GPU_RESP_ERR_INVALID_PARAMETER;
        }
        let x = u32_le(req, 24);
        let y = u32_le(req, 28);
        let width = u32_le(req, 32);
        let height = u32_le(req, 36);
        let scanout_id = u32_le(req, 40);
        let resource_id = u32_le(req, 44);
        if scanout_id != 0 {
            return VIRTIO_GPU_RESP_ERR_INVALID_SCANOUT_ID;
        }
        if resource_id != 0 && !self.resources.contains_key(&resource_id) {
            return VIRTIO_GPU_RESP_ERR_INVALID_RESOURCE_ID;
        }
        self.scanout = Scanout {
            resource_id,
            x,
            y,
            width,
            height,
        };
        VIRTIO_GPU_RESP_OK_NODATA
    }

    fn cmd_transfer_to_host_2d(&mut self, req: &[u8]) -> u32 {
        // hdr(24) + r: {x:4, y:4, w:4, h:4} + offset(8) + resource_id(4) + pad(4) = 56
        if req.len() < 52 {
            return VIRTIO_GPU_RESP_ERR_INVALID_PARAMETER;
        }
        let x = u32_le(req, 24);
        let y = u32_le(req, 28);
        let w = u32_le(req, 32);
        let h = u32_le(req, 36);
        let offset = u64_le(req, 40);
        let resource_id = u32_le(req, 48);

        let Some(res) = self.resources.get(&resource_id) else {
            return VIRTIO_GPU_RESP_ERR_INVALID_RESOURCE_ID;
        };
        if res.zero_copy_extmem {
            // `ExtMem` directly aliases guest backing memory; no copy needed.
            return VIRTIO_GPU_RESP_OK_NODATA;
        }
        if x.saturating_add(w) > res.width || y.saturating_add(h) > res.height {
            return VIRTIO_GPU_RESP_ERR_INVALID_PARAMETER;
        }
        let Some(img) = self.hal.image_mut(res.hal_image) else {
            return VIRTIO_GPU_RESP_ERR_UNSPEC;
        };
        let pitch = img.row_pitch as usize;
        let dst = img.slice_mut();
        let row_bytes = (w as usize) * 4;
        for row in 0..h as usize {
            let src_off = offset.saturating_add((row as u64) * (pitch as u64));
            let dst_off = ((y as usize) + row) * pitch + (x as usize) * 4;
            if dst_off + row_bytes <= dst.len() {
                let _ = read_from_backing(
                    &self.memory,
                    &res.backing,
                    src_off,
                    &mut dst[dst_off..dst_off + row_bytes],
                );
            }
        }
        VIRTIO_GPU_RESP_OK_NODATA
    }

    fn cmd_resource_flush(&mut self, req: &[u8]) -> u32 {
        // hdr(24) + r: {x:4, y:4, w:4, h:4} + resource_id(4) + pad(4) = 48
        if req.len() < 44 {
            return VIRTIO_GPU_RESP_ERR_INVALID_PARAMETER;
        }
        let resource_id = u32_le(req, 40);
        let Some(res) = self.resources.get(&resource_id) else {
            return VIRTIO_GPU_RESP_ERR_INVALID_RESOURCE_ID;
        };
        if self.scanout.resource_id == resource_id
            && let Some(img) = self.hal.image(res.hal_image)
        {
            self.surface.present(
                res.width,
                res.height,
                img.row_pitch,
                res.pixel_format,
                img.slice(),
            );
        }
        VIRTIO_GPU_RESP_OK_NODATA
    }
}

impl<M: GuestMemory + Send, S: Surface, L: FnMut(bool) + Send> VirtioIo for VirtioGpu<M, S, L> {
    fn read(&mut self, offset: u64, size: u8) -> u64 {
        VirtioGpu::read(self, offset, size)
    }

    fn write(&mut self, offset: u64, size: u8, value: u64) {
        VirtioGpu::write(self, offset, size, value)
    }
}

fn map_format(virtio_fmt: u32) -> Option<(HalFormat, PixelFormat)> {
    match virtio_fmt {
        VIRTIO_GPU_FORMAT_B8G8R8A8_UNORM | VIRTIO_GPU_FORMAT_A8R8G8B8_UNORM => {
            Some((HalFormat::B8G8R8A8Unorm, PixelFormat::B8G8R8A8Unorm))
        }
        VIRTIO_GPU_FORMAT_B8G8R8X8_UNORM | VIRTIO_GPU_FORMAT_X8R8G8B8_UNORM => {
            Some((HalFormat::B8G8R8A8Unorm, PixelFormat::B8G8R8X8Unorm))
        }
        VIRTIO_GPU_FORMAT_R8G8B8A8_UNORM | VIRTIO_GPU_FORMAT_A8B8G8R8_UNORM => {
            Some((HalFormat::R8G8B8A8Unorm, PixelFormat::R8G8B8A8Unorm))
        }
        VIRTIO_GPU_FORMAT_R8G8B8X8_UNORM | VIRTIO_GPU_FORMAT_X8B8G8R8_UNORM => {
            Some((HalFormat::R8G8B8A8Unorm, PixelFormat::R8G8B8X8Unorm))
        }
        _ => None,
    }
}

fn contiguous_host_slice(memory: &impl GuestMemory, gpa: u64, len: usize) -> Option<*mut u8> {
    if len == 0 {
        return None;
    }
    let first_page = gpa & !0xfff;
    let base = memory.host_page(first_page)?;
    let page_off = (gpa & 0xfff) as usize;
    let end = gpa.checked_add(len as u64)?;
    let last_page = (end - 1) & !0xfff;
    let mut expected_ptr = base as usize;
    let mut page = first_page;
    while page <= last_page {
        let p = memory.host_page(page)? as usize;
        if p != expected_ptr {
            return None;
        }
        page = page.checked_add(0x1000)?;
        expected_ptr = expected_ptr.checked_add(0x1000)?;
    }
    Some(unsafe { base.add(page_off) })
}

fn read_from_backing(
    memory: &impl GuestMemory,
    backing: &[BackingEntry],
    mut offset: u64,
    mut out: &mut [u8],
) -> Result<(), ()> {
    for entry in backing {
        if out.is_empty() {
            return Ok(());
        }
        let elen = entry.len as u64;
        if offset >= elen {
            offset -= elen;
            continue;
        }
        let avail = (elen - offset) as usize;
        let take = avail.min(out.len());
        memory
            .read(entry.addr + offset, &mut out[..take])
            .map_err(|_| ())?;
        out = &mut out[take..];
        offset = 0;
    }
    if out.is_empty() { Ok(()) } else { Err(()) }
}

fn ctrl_header(kind: u32, flags: u32, fence_id: u64) -> Vec<u8> {
    let mut buf = vec![0u8; CTRL_HDR_SIZE];
    buf[0..4].copy_from_slice(&kind.to_le_bytes());
    buf[4..8].copy_from_slice(&flags.to_le_bytes());
    buf[8..16].copy_from_slice(&fence_id.to_le_bytes());
    buf
}

fn u32_le(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

fn u64_le(buf: &[u8], off: usize) -> u64 {
    u64::from_le_bytes([
        buf[off],
        buf[off + 1],
        buf[off + 2],
        buf[off + 3],
        buf[off + 4],
        buf[off + 5],
        buf[off + 6],
        buf[off + 7],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{PhysicalMemory, Region, SharedMemory};

    #[derive(Default)]
    struct CaptureSurface {
        width: u32,
        height: u32,
        stride: u32,
        format: Option<PixelFormat>,
        pixels: Vec<u8>,
        presents: u32,
    }

    impl Surface for CaptureSurface {
        fn present(
            &mut self,
            width: u32,
            height: u32,
            stride_bytes: u32,
            format: PixelFormat,
            pixels: &[u8],
        ) {
            self.width = width;
            self.height = height;
            self.stride = stride_bytes;
            self.format = Some(format);
            self.pixels = pixels.to_vec();
            self.presents += 1;
        }
    }

    fn submit_cmd(
        gpu: &mut VirtioGpu<SharedMemory, CaptureSurface>,
        mem: &mut SharedMemory,
        avail_idx: &mut u16,
        req: &[u8],
        resp_len: u32,
    ) -> Vec<u8> {
        let desc_table = 0x4000_1000u64;
        let avail_ring = 0x4000_2000u64;
        let _used_ring = 0x4000_3000u64;
        let req_addr = 0x4000_4000u64;
        let resp_addr = 0x4000_5000u64;

        mem.write(req_addr, req).unwrap();
        mem.write(resp_addr, &vec![0u8; resp_len as usize]).unwrap();

        // Descriptor 0: device-readable request
        let mut d0 = [0u8; 16];
        d0[0..8].copy_from_slice(&req_addr.to_le_bytes());
        d0[8..12].copy_from_slice(&(req.len() as u32).to_le_bytes());
        d0[12..14].copy_from_slice(&VRING_DESC_F_NEXT.to_le_bytes());
        d0[14..16].copy_from_slice(&1u16.to_le_bytes());
        mem.write(desc_table, &d0).unwrap();

        // Descriptor 1: device-writable response
        let mut d1 = [0u8; 16];
        d1[0..8].copy_from_slice(&resp_addr.to_le_bytes());
        d1[8..12].copy_from_slice(&resp_len.to_le_bytes());
        d1[12..14].copy_from_slice(&VRING_DESC_F_WRITE.to_le_bytes());
        mem.write(desc_table + 16, &d1).unwrap();

        let slot = (*avail_idx as u64) % 16;
        mem.write(avail_ring + 4 + slot * 2, &0u16.to_le_bytes())
            .unwrap();
        *avail_idx = avail_idx.wrapping_add(1);
        mem.write(avail_ring + 2, &avail_idx.to_le_bytes()).unwrap();

        gpu.write(0x050, 4, 0); // Notify controlq 0

        let mut out = vec![0u8; resp_len as usize];
        mem.read(resp_addr, &mut out).unwrap();
        out
    }

    #[test]
    fn virtio_gpu_2d_pipeline_presents_to_surface() {
        let phys =
            PhysicalMemory::new(vec![Region::ram(0x4000_0000, vec![0; 0x20000]).unwrap()]).unwrap();
        let mut mem = SharedMemory::new(phys);
        let mut gpu = VirtioGpu::new(mem.clone(), CaptureSurface::default()).with_resolution(4, 4);

        assert_eq!(gpu.read(0x000, 4), MAGIC_VALUE as u64);
        assert_eq!(gpu.read(0x008, 4), DEVICE_ID_GPU as u64);
        assert_eq!(gpu.read(0x108, 4), 1); // num_scanouts

        // Configure controlq 0
        gpu.write(0x030, 4, 0);
        gpu.write(0x038, 4, 16);
        gpu.write(0x080, 4, 0x4000_1000);
        gpu.write(0x084, 4, 0);
        gpu.write(0x090, 4, 0x4000_2000);
        gpu.write(0x094, 4, 0);
        gpu.write(0x0a0, 4, 0x4000_3000);
        gpu.write(0x0a4, 4, 0);
        gpu.write(0x044, 4, 1);

        let mut avail_idx = 0u16;

        // 1. GET_DISPLAY_INFO
        let info_req = ctrl_header(VIRTIO_GPU_CMD_GET_DISPLAY_INFO, 0, 1);
        let info_resp = submit_cmd(&mut gpu, &mut mem, &mut avail_idx, &info_req, 408);
        assert_eq!(u32_le(&info_resp, 0), VIRTIO_GPU_RESP_OK_DISPLAY_INFO);
        assert_eq!(u32_le(&info_resp, 24 + 8), 4);
        assert_eq!(u32_le(&info_resp, 24 + 12), 4);
        assert_eq!(u32_le(&info_resp, 24 + 16), 1);

        // 2. RESOURCE_CREATE_2D (id=1, BGRA8888, 4x4)
        let mut create_req = ctrl_header(VIRTIO_GPU_CMD_RESOURCE_CREATE_2D, 0, 2);
        create_req.extend_from_slice(&1u32.to_le_bytes());
        create_req.extend_from_slice(&VIRTIO_GPU_FORMAT_B8G8R8A8_UNORM.to_le_bytes());
        create_req.extend_from_slice(&4u32.to_le_bytes());
        create_req.extend_from_slice(&4u32.to_le_bytes());
        let resp = submit_cmd(&mut gpu, &mut mem, &mut avail_idx, &create_req, 24);
        assert_eq!(u32_le(&resp, 0), VIRTIO_GPU_RESP_OK_NODATA);

        // 3. RESOURCE_ATTACH_BACKING (contiguous guest memory -> ExtMem zero-copy)
        let fb_gpa = 0x4001_0000u64;
        let fb_len = 4u32 * 4 * 4;
        let mut attach_req = ctrl_header(VIRTIO_GPU_CMD_RESOURCE_ATTACH_BACKING, 0, 3);
        attach_req.extend_from_slice(&1u32.to_le_bytes()); // resource_id
        attach_req.extend_from_slice(&1u32.to_le_bytes()); // nr_entries
        attach_req.extend_from_slice(&fb_gpa.to_le_bytes());
        attach_req.extend_from_slice(&fb_len.to_le_bytes());
        attach_req.extend_from_slice(&0u32.to_le_bytes());
        let resp = submit_cmd(&mut gpu, &mut mem, &mut avail_idx, &attach_req, 24);
        assert_eq!(u32_le(&resp, 0), VIRTIO_GPU_RESP_OK_NODATA);
        assert!(gpu.resources.get(&1).unwrap().zero_copy_extmem);

        // Write pixel pattern into guest backing RAM
        mem.write(fb_gpa, &[0x11, 0x22, 0x33, 0xff]).unwrap();

        // 4. SET_SCANOUT
        let mut scanout_req = ctrl_header(VIRTIO_GPU_CMD_SET_SCANOUT, 0, 4);
        scanout_req.extend_from_slice(&0u32.to_le_bytes());
        scanout_req.extend_from_slice(&0u32.to_le_bytes());
        scanout_req.extend_from_slice(&4u32.to_le_bytes());
        scanout_req.extend_from_slice(&4u32.to_le_bytes());
        scanout_req.extend_from_slice(&0u32.to_le_bytes()); // scanout_id
        scanout_req.extend_from_slice(&1u32.to_le_bytes()); // resource_id
        let resp = submit_cmd(&mut gpu, &mut mem, &mut avail_idx, &scanout_req, 24);
        assert_eq!(u32_le(&resp, 0), VIRTIO_GPU_RESP_OK_NODATA);

        // 5. TRANSFER_TO_HOST_2D
        let mut xfer_req = ctrl_header(VIRTIO_GPU_CMD_TRANSFER_TO_HOST_2D, 0, 5);
        xfer_req.extend_from_slice(&0u32.to_le_bytes());
        xfer_req.extend_from_slice(&0u32.to_le_bytes());
        xfer_req.extend_from_slice(&4u32.to_le_bytes());
        xfer_req.extend_from_slice(&4u32.to_le_bytes());
        xfer_req.extend_from_slice(&0u64.to_le_bytes());
        xfer_req.extend_from_slice(&1u32.to_le_bytes());
        xfer_req.extend_from_slice(&0u32.to_le_bytes());
        let resp = submit_cmd(&mut gpu, &mut mem, &mut avail_idx, &xfer_req, 24);
        assert_eq!(u32_le(&resp, 0), VIRTIO_GPU_RESP_OK_NODATA);

        // 6. RESOURCE_FLUSH
        let mut flush_req = ctrl_header(VIRTIO_GPU_CMD_RESOURCE_FLUSH, 0, 6);
        flush_req.extend_from_slice(&0u32.to_le_bytes());
        flush_req.extend_from_slice(&0u32.to_le_bytes());
        flush_req.extend_from_slice(&4u32.to_le_bytes());
        flush_req.extend_from_slice(&4u32.to_le_bytes());
        flush_req.extend_from_slice(&1u32.to_le_bytes());
        flush_req.extend_from_slice(&0u32.to_le_bytes());
        let resp = submit_cmd(&mut gpu, &mut mem, &mut avail_idx, &flush_req, 24);
        assert_eq!(u32_le(&resp, 0), VIRTIO_GPU_RESP_OK_NODATA);

        assert_eq!(gpu.surface().presents, 1);
        assert_eq!(gpu.surface().width, 4);
        assert_eq!(gpu.surface().height, 4);
        assert_eq!(gpu.surface().format, Some(PixelFormat::B8G8R8A8Unorm));
        assert_eq!(&gpu.surface().pixels[..4], &[0x11, 0x22, 0x33, 0xff]);
    }
}
