//! QEMU fw_cfg's traditional MMIO interface. DMA is not advertised.
use super::bus::Device;

pub const BASE: u64 = 0x0902_0000;
// Exclude the DMA address register at offset 0x10 from the advertised window.
pub const LEN: u64 = 0x10;

pub struct FwCfg {
    selector: u16,
    offset: usize,
    ram_size: [u8; 8],
}

impl FwCfg {
    pub fn new(ram_size: u64) -> Self {
        Self {
            selector: 0,
            offset: 0,
            ram_size: ram_size.to_le_bytes(),
        }
    }

    fn item(&self) -> &[u8] {
        match self.selector {
            0x00 => b"QEMU",
            0x01 => &[1, 0, 0, 0], // Traditional interface only; no DMA.
            0x03 => &self.ram_size,
            0x05 | 0x0f => &[1, 0], // Number and maximum number of CPUs.
            0x19 => &[0, 0, 0, 0],  // Empty file directory (big-endian count).
            _ => &[],
        }
    }
}

impl Device for FwCfg {
    fn read(&mut self, offset: u64, size: u8) -> u64 {
        if offset != 0 || !matches!(size, 1 | 2 | 4 | 8) {
            return 0;
        }
        let item = self.item();
        let mut value = 0;
        for byte in 0..usize::from(size) {
            let at = self.offset.saturating_add(byte);
            value |= u64::from(item.get(at).copied().unwrap_or(0)) << (byte * 8);
        }
        self.offset = self.offset.saturating_add(usize::from(size));
        value
    }

    fn write(&mut self, offset: u64, size: u8, value: u64) {
        if offset == 8 && size == 2 {
            // MMIO selectors are big-endian; Device values are little-endian.
            self.selector = (value as u16).swap_bytes() & !0x4000;
            self.offset = 0;
        }
        // The data interface is read-only, and there is no DMA register.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_preserves_bytes_across_widths_and_reselection() {
        let mut device = FwCfg::new(0x0102_0304_0506_0708);
        assert_eq!(device.read(0, 1), u64::from(b'Q'));
        assert_eq!(device.read(0, 2), u64::from(u16::from_le_bytes(*b"EM")));
        assert_eq!(device.read(0, 4), u64::from(b'U'));
        assert_eq!(device.read(0, 8), 0);
        device.write(8, 2, 0);
        assert_eq!(device.read(0, 4), u64::from(u32::from_le_bytes(*b"QEMU")));
        device.write(8, 2, 0x0300);
        assert_eq!(device.read(0, 8), 0x0102_0304_0506_0708);
        device.write(8, 2, 0x0100);
        assert_eq!(device.read(0, 4), 1);
        // The obsolete write-channel flag does not select a different item.
        device.write(8, 2, 0x0040);
        device.write(0, 4, 0);
        assert_eq!(device.read(0, 4), u64::from(u32::from_le_bytes(*b"QEMU")));
    }

    #[test]
    fn absent_items_and_unsupported_registers_do_not_change_the_stream() {
        let mut device = FwCfg::new(128 << 20);
        device.write(8, 2, 0x1900);
        assert_eq!(device.read(0, 4), 0);
        device.write(8, 2, 0xffff);
        assert_eq!(device.read(0, 8), 0);
        device.write(8, 2, 0);
        device.write(0, 4, 0);
        device.write(8, 1, 1);
        device.write(16, 8, 0);
        assert_eq!(device.read(8, 2), 0);
        assert_eq!(device.read(16, 8), 0);
        assert_eq!(device.read(0, 4), u64::from(u32::from_le_bytes(*b"QEMU")));
    }
}
