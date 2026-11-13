//! Linux DRM transport implementation for `volt-hal`'s AMDGPU driver.

use alloc::sync::Arc;
use volt_drm::{
    AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED, AMDGPU_GEM_DOMAIN_GTT, AMDGPU_HW_IP_GFX,
    AMDGPU_IB_FLAG_EMIT_MEM_SYNC, AmdgpuBo, AmdgpuDevice, DrmDevice, Error as DrmError,
};
use volt_hal::amdgpu::{AmdgpuError, AmdgpuTransport, GpuBo};

fn map_err(err: DrmError) -> AmdgpuError {
    match err {
        DrmError::NoDevice => AmdgpuError::NoDevice,
        DrmError::WrongDriver(msg) => {
            AmdgpuError::WrongDriver(core::str::from_utf8(msg).unwrap_or("wrong driver"))
        }
        DrmError::Io(code) => AmdgpuError::Io(code),
        DrmError::NotMapped => AmdgpuError::Io(22),
    }
}

pub struct DrmBo(AmdgpuBo);

impl GpuBo for DrmBo {
    fn handle(&self) -> u32 {
        self.0.handle
    }
    fn gpu_va(&self) -> u64 {
        self.0.gpu_va
    }
    fn size(&self) -> u64 {
        self.0.size
    }
    fn cpu_ptr(&self) -> *mut u8 {
        self.0.cpu_ptr
    }
    fn as_slice_mut(&mut self) -> Option<&mut [u8]> {
        self.0.as_slice_mut()
    }
}

pub struct DrmAmdgpuTransport {
    dev: Arc<AmdgpuDevice>,
}

impl DrmAmdgpuTransport {
    pub fn probe() -> Result<Self, AmdgpuError> {
        let drm = select_render_node().map_err(map_err)?;
        let dev = AmdgpuDevice::new(drm).map_err(map_err)?;
        Ok(Self { dev })
    }
}

impl AmdgpuTransport for DrmAmdgpuTransport {
    fn ip_discovery_version(&self) -> u32 {
        self.dev.gfx_ip.ip_discovery_version
    }

    fn device_id(&self) -> u16 {
        self.dev.info.device_id as u16
    }

    fn virtual_address_range(&self) -> (u64, u64) {
        (
            self.dev.info.virtual_address_offset,
            self.dev.info.virtual_address_max,
        )
    }

    fn create_context(&self) -> Result<u32, AmdgpuError> {
        self.dev.create_context().map_err(map_err)
    }

    fn destroy_context(&self, ctx_id: u32) -> Result<(), AmdgpuError> {
        self.dev.destroy_context(ctx_id).map_err(map_err)
    }

    fn alloc_gtt_bo(&self, size: u64, align: u64, va: u64) -> Result<Box<dyn GpuBo>, AmdgpuError> {
        let bo = self
            .dev
            .alloc_bo(
                size,
                align,
                AMDGPU_GEM_DOMAIN_GTT,
                AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED,
                va,
                true,
            )
            .map_err(map_err)?;
        Ok(Box::new(DrmBo(bo)))
    }

    fn submit_gfx_ib(
        &self,
        ctx_id: u32,
        ib_gpu_va: u64,
        dwords: u32,
        handles: &[u32],
    ) -> Result<u64, AmdgpuError> {
        self.dev
            .submit(
                ctx_id,
                AMDGPU_HW_IP_GFX,
                ib_gpu_va,
                dwords,
                AMDGPU_IB_FLAG_EMIT_MEM_SYNC,
                handles,
            )
            .map_err(map_err)
    }

    fn wait_gfx_cs(&self, ctx_id: u32, seq: u64, timeout_ns: u64) -> Result<bool, AmdgpuError> {
        self.dev
            .wait_cs(ctx_id, AMDGPU_HW_IP_GFX, seq, timeout_ns)
            .map_err(map_err)
    }
}

/// Probe and construct a `volt-hal` device with AMDGPU hardware acceleration when available.
pub fn new_hal_device() -> volt_hal::Device {
    match DrmAmdgpuTransport::probe() {
        Ok(transport) => volt_hal::Device::with_amdgpu(Box::new(transport)),
        Err(_) => volt_hal::Device::new(),
    }
}

fn select_render_node() -> Result<DrmDevice, DrmError> {
    if let Ok(path) =
        std::env::var("VOLT_AMDGPU_DEVICE").or_else(|_| std::env::var("VANTAGE_AMDGPU_DEVICE"))
    {
        let mut p = path.into_bytes();
        p.push(0);
        return DrmDevice::open_path(&p);
    }

    let mut candidates: Vec<(u32, DrmDevice, bool)> = Vec::new();
    for n in 128u32..160 {
        let path = format!("/dev/dri/renderD{n}\0");
        let Ok(dev) = DrmDevice::open_path(path.as_bytes()) else {
            continue;
        };
        let is_amd = dev
            .sysfs_driver_name()
            .as_deref()
            .map(|d| d == b"amdgpu")
            .unwrap_or(false);
        if !is_amd {
            continue;
        }
        let boot_vga =
            std::fs::read_to_string(format!("/sys/class/drm/renderD{n}/device/boot_vga"))
                .map(|s| s.trim() == "1")
                .unwrap_or(false);
        candidates.push((n, dev, boot_vga));
    }

    if let Some((n, dev, _)) = candidates
        .into_iter()
        .max_by_key(|(_, _, boot_vga)| *boot_vga)
    {
        eprintln!("volt-amdgpu: selecting render node /dev/dri/renderD{n}");
        return Ok(dev);
    }

    Err(DrmError::NoDevice)
}
