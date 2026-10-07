//! Flattened device tree for the one-vCPU JIT machine, following Mirage arm64/fdt.zig.
use std::collections::BTreeMap;

pub const GICD_BASE: u64 = 0x0800_0000;
pub const GIC_CPU_BASE: u64 = 0x0801_0000;
pub const UART_BASE: u64 = 0x0900_0000;
pub const UART_INTID: u32 = 33;
pub const VIRTUAL_TIMER_INTID: u32 = 27;
pub const TIMER_FREQ: u64 = 24_000_000;
pub const VIRTIO_MMIO_BASE: u64 = 0x0a00_0000;
pub const VIRTIO_MMIO_SIZE: u64 = 0x200;
pub const VIRTIO_INTID: u32 = 48; // SPI 16 -> GIC INTID 32 + 16 = 48

struct Builder {
    structure: Vec<u8>,
    strings: Vec<u8>,
    names: BTreeMap<String, u32>,
}
impl Builder {
    fn new() -> Self {
        Self {
            structure: Vec::new(),
            strings: Vec::new(),
            names: BTreeMap::new(),
        }
    }
    fn word(&mut self, word: u32) {
        self.structure.extend_from_slice(&word.to_be_bytes());
    }
    fn padded(&mut self, bytes: &[u8]) {
        self.structure.extend_from_slice(bytes);
        while self.structure.len() % 4 != 0 {
            self.structure.push(0);
        }
    }
    fn begin(&mut self, name: &str) {
        self.word(1);
        self.structure.extend_from_slice(name.as_bytes());
        self.structure.push(0);
        while self.structure.len() % 4 != 0 {
            self.structure.push(0);
        }
    }
    fn end(&mut self) {
        self.word(2);
    }
    fn property(&mut self, name: &str, value: &[u8]) {
        let at = if let Some(&at) = self.names.get(name) {
            at
        } else {
            let at = self.strings.len() as u32;
            self.strings.extend_from_slice(name.as_bytes());
            self.strings.push(0);
            self.names.insert(name.to_owned(), at);
            at
        };
        self.word(3);
        self.word(value.len() as u32);
        self.word(at);
        self.padded(value);
    }
    fn string(&mut self, name: &str, value: &str) {
        let mut bytes = Vec::with_capacity(value.len() + 1);
        bytes.extend_from_slice(value.as_bytes());
        bytes.push(0);
        self.property(name, &bytes);
    }
    fn cells(&mut self, name: &str, value: &[u32]) {
        let mut bytes = Vec::with_capacity(value.len() * 4);
        for cell in value {
            bytes.extend_from_slice(&cell.to_be_bytes());
        }
        self.property(name, &bytes);
    }
    fn finish(mut self) -> Vec<u8> {
        self.word(9);
        let structure_offset = 56; // 40-byte header, then terminating reserve-map entry.
        let strings_offset = structure_offset + self.structure.len();
        let mut blob = Vec::with_capacity(strings_offset + self.strings.len());
        for word in [
            0xd00dfeed,
            (strings_offset + self.strings.len()) as u32,
            structure_offset as u32,
            strings_offset as u32,
            40,
            17,
            16,
            0,
            self.strings.len() as u32,
            self.structure.len() as u32,
        ] {
            blob.extend_from_slice(&word.to_be_bytes());
        }
        blob.extend_from_slice(&[0; 16]);
        blob.extend_from_slice(&self.structure);
        blob.extend_from_slice(&self.strings);
        blob
    }
}
fn range(address: u64, size: u64) -> [u32; 4] {
    [
        (address >> 32) as u32,
        address as u32,
        (size >> 32) as u32,
        size as u32,
    ]
}

pub fn build(ram_base: u64, ram_size: u64, cmdline: &str) -> Vec<u8> {
    build_smp(ram_base, ram_size, cmdline, 1, None, false)
}

pub fn build_smp(
    ram_base: u64,
    ram_size: u64,
    cmdline: &str,
    cpus: u32,
    initrd: Option<(u64, u64)>,
    has_virtio: bool,
) -> Vec<u8> {
    let mut b = Builder::new();
    b.begin("");
    b.cells("#address-cells", &[2]);
    b.cells("#size-cells", &[2]);
    b.string("compatible", "linux,dummy-virt");
    b.string("model", "volt");
    b.cells("interrupt-parent", &[2]);
    b.begin("chosen");
    b.string("bootargs", cmdline);
    b.string("stdout-path", "/pl011@9000000");
    if let Some((start, end)) = initrd {
        b.cells("linux,initrd-start", &[(start >> 32) as u32, start as u32]);
        b.cells("linux,initrd-end", &[(end >> 32) as u32, end as u32]);
    }
    b.end();
    b.begin(&format!("memory@{ram_base:x}"));
    b.string("device_type", "memory");
    b.cells("reg", &range(ram_base, ram_size));
    b.end();
    b.begin("cpus");
    b.cells("#address-cells", &[1]);
    b.cells("#size-cells", &[0]);
    for cpu in 0..cpus.max(1) {
        b.begin(&format!("cpu@{cpu}"));
        b.string("device_type", "cpu");
        b.string("compatible", "arm,armv8");
        b.cells("reg", &[cpu]);
        b.string("enable-method", "psci");
        b.end();
    }
    b.end();
    b.begin("psci");
    b.string("compatible", "arm,psci-0.2");
    b.string("method", "hvc");
    b.end();
    b.begin("timer");
    b.string("compatible", "arm,armv8-timer");
    b.cells(
        "interrupts",
        &[1, 13, 0xf08, 1, 14, 0xf08, 1, 11, 0xf08, 1, 10, 0xf08],
    );
    b.property("always-on", &[]);
    b.end();
    b.begin("intc@8000000");
    b.cells("#interrupt-cells", &[3]);
    b.cells("#address-cells", &[2]);
    b.cells("#size-cells", &[2]);
    b.property("interrupt-controller", &[]);
    b.string("compatible", "arm,cortex-a15-gic");
    let mut registers = Vec::from(range(GICD_BASE, 0x1000));
    registers.extend_from_slice(&range(GIC_CPU_BASE, 0x2000));
    b.cells("reg", &registers);
    b.cells("phandle", &[2]);
    b.end();
    b.begin("pl011@9000000");
    b.string("compatible", "arm,pl011\0arm,primecell");
    b.cells("reg", &range(UART_BASE, 0x1000));
    b.cells("interrupts", &[0, 1, 4]);
    b.cells("clocks", &[1, 1]);
    b.string("clock-names", "uartclk\0apb_pclk");
    b.end();
    if has_virtio {
        b.begin("virtio@a000000");
        b.string("compatible", "virtio,mmio");
        b.cells("reg", &range(VIRTIO_MMIO_BASE, VIRTIO_MMIO_SIZE));
        b.cells("interrupts", &[0, 16, 4]); // SPI 16, level high
        b.end();
    }
    b.begin("apb-pclk");
    b.string("compatible", "fixed-clock");
    b.cells("#clock-cells", &[0]);
    b.cells("clock-frequency", &[TIMER_FREQ as u32]);
    b.cells("phandle", &[1]);
    b.end();
    b.end();
    b.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn blob_header_offsets_and_tokens_are_consistent() {
        let tree = build(0x40000000, 128 << 20, "console=ttyAMA0");
        let word = |at| u32::from_be_bytes(tree[at..at + 4].try_into().unwrap()) as usize;
        assert_eq!(word(0), 0xd00dfeed);
        assert_eq!(word(4), tree.len());
        assert_eq!(word(8), 56);
        assert_eq!(word(12) + word(32), tree.len());
        assert_eq!(word(12), word(8) + word(36));
        assert_eq!(word(word(12) - 4), 9);
        assert!(tree.windows(19).any(|part| part == b"arm,cortex-a15-gic\0"));
    }
    #[test]
    fn fdt_includes_initrd_properties_when_provided() {
        let tree = build_smp(
            0x40000000,
            128 << 20,
            "console=ttyAMA0",
            1,
            Some((0x42000000, 0x42100000)),
            true,
        );
        assert!(tree.windows(19).any(|part| part == b"linux,initrd-start\0"));
        assert!(tree.windows(17).any(|part| part == b"linux,initrd-end\0"));
        assert!(tree.windows(12).any(|part| part == b"virtio,mmio\0"));
    }
}
