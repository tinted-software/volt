//! Apple flattened device tree for the XNU QEMU `virt` board, the tree
//! `pexpert` looks nodes up in. Port of the AFDT builder the XNU fork's
//! `qemu/mkafdt.py` ships; each node here exists because XNU panics or loses
//! its console without it:
//!
//! * `/chosen` `dram-base`/`dram-size`: `arm_init` panics without them.
//! * `/defaults` `serial-device` plus `/arm-io/uart0` `AAPL,phandle`: the
//!   serial console lookup in `pe_serial.c`.
//! * `/arm-io` `ranges`: its second value becomes the SoC base address that
//!   every `reg` below it is relative to.
//! * `/arm-io/gic` `reg`: GICD and GICR windows for `pe_fiq.c`.
//! * `/options`: `IODTPlatformExpert` creates the NVRAM node from it.
//!
//! Encoding (little-endian): `u32 nProperties, u32 nChildren`, then per property
//! a 32-byte NUL-padded name, `u32 length` and the value padded to four bytes,
//! then the children recursively.
use super::fdt::{GICD_BASE, UART_BASE};
use crate::aarch64::identification::counter_hz;

pub const GICD_SIZE: u64 = 0x1_0000;
/// The redistributor window of QEMU `virt`: 0x20000 per PE.
pub const GICR_BASE: u64 = 0x080a_0000;
pub const GICR_SIZE: u64 = 0x00f6_0000;
pub const RTC_BASE: u64 = 0x0901_0000;
const UART_PHANDLE: u32 = 0x100;
/// Size of the panic log the tree places in the last of RAM (`chosen/pram`).
pub const PANIC_LOG_SIZE: u32 = 0x8_0000;
const NVRAM_BANK_SIZE: usize = 0x2000;
const RANDOM_SEED_SIZE: usize = 256;

fn property(name: &str, value: &[u8]) -> Vec<u8> {
    assert!(
        name.len() < 32,
        "device tree property name {name:?} is too long"
    );
    let mut bytes = Vec::with_capacity(36 + value.len() + 3);
    bytes.extend_from_slice(name.as_bytes());
    bytes.resize(32, 0);
    bytes.extend_from_slice(&(value.len() as u32).to_le_bytes());
    bytes.extend_from_slice(value);
    bytes.resize(bytes.len().next_multiple_of(4), 0);
    bytes
}
fn string(name: &str, value: &str) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(value.len() + 1);
    bytes.extend_from_slice(value.as_bytes());
    bytes.push(0);
    property(name, &bytes)
}
fn word(name: &str, value: u32) -> Vec<u8> {
    property(name, &value.to_le_bytes())
}
fn words64(name: &str, values: &[u64]) -> Vec<u8> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    property(name, &bytes)
}
fn node(properties: &[Vec<u8>], children: &[Vec<u8>]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(properties.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&(children.len() as u32).to_le_bytes());
    for part in properties.iter().chain(children) {
        bytes.extend_from_slice(part);
    }
    bytes
}

fn adler32(bytes: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in bytes {
        a = (a + u32::from(byte)) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

/// An empty, valid CHRP NVRAM image: the bank header and one `common` partition.
fn nvram_image() -> Vec<u8> {
    let mut image = vec![0u8; NVRAM_BANK_SIZE];
    let mut header = |offset: usize, length: usize, name: &[u8]| {
        image[offset] = 0x70;
        image[offset + 2..offset + 4].copy_from_slice(&((length / 16) as u16).to_le_bytes());
        image[offset + 4..offset + 4 + name.len()].copy_from_slice(name);
        let mut checksum = u32::from(image[offset])
            + image[offset + 2..offset + 16]
                .iter()
                .map(|&b| u32::from(b))
                .sum::<u32>();
        while checksum > 0xff {
            checksum = (checksum & 0xff) + (checksum >> 8);
        }
        image[offset + 1] = checksum as u8;
    };
    header(0, 32, b"nvram");
    header(32, NVRAM_BANK_SIZE - 32, b"common");
    let sum = adler32(&image[20..]);
    image[16..20].copy_from_slice(&sum.to_le_bytes());
    image
}

/// Deterministic non-zero bytes. XNU rejects an all-zero seed; a fixed stream
/// keeps boots reproducible, which matters more here than entropy.
fn random_seed() -> Vec<u8> {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut seed = Vec::with_capacity(RANDOM_SEED_SIZE);
    while seed.len() < RANDOM_SEED_SIZE {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        seed.extend_from_slice(&(z ^ (z >> 31)).to_le_bytes());
    }
    seed
}

fn chosen(ram_base: u64, ram_size: u64, boot_args: &str) -> Vec<u8> {
    node(
        &[
            string("name", "chosen"),
            property("dram-base", &ram_base.to_le_bytes()),
            property("dram-size", &ram_size.to_le_bytes()),
            string("firmware-version", "volt"),
            string("system-firmware-version", "volt"),
            string("boot-args", boot_args),
            property("unique-chip-id", &[1, 2, 3, 4, 5, 6, 7, 8]),
            word("embedded-panic-log-size", PANIC_LOG_SIZE),
            word("kernel-ctrr-to-be-enabled", 0),
            word("debug-enabled", 1),
            property("random-seed", &random_seed()),
            word("nvram-bank-size", NVRAM_BANK_SIZE as u32),
            word("nvram-bank-count", 2),
            word("nvram-current-bank", 0),
            property("nvram-proxy-data", &nvram_image()),
        ],
        &[node(&[string("name", "memory-map")], &[])],
    )
}

fn arm_io() -> Vec<u8> {
    let uart = node(
        &[
            string("name", "uart0"),
            string("compatible", "arm,pl011"),
            words64("reg", &[UART_BASE - GICD_BASE, 0x1000]),
            word("AAPL,phandle", UART_PHANDLE),
        ],
        &[],
    );
    let gic = node(
        &[
            string("name", "gic"),
            string("interrupt-controller", "master"),
            string("compatible", "arm,gic-v3"),
            words64("reg", &[0, GICD_SIZE, GICR_BASE - GICD_BASE, GICR_SIZE]),
        ],
        &[],
    );
    let timer = node(
        &[
            string("name", "timer"),
            string("device_type", "timer"),
            words64("reg", &[RTC_BASE - GICD_BASE, 0x1000]),
        ],
        &[],
    );
    node(
        &[
            string("name", "arm-io"),
            string("device_type", "arm-io"),
            words64("ranges", &[0, GICD_BASE]),
        ],
        &[uart, gic, timer],
    )
}

fn cpus() -> Vec<u8> {
    let cpu0 = node(
        &[
            string("name", "cpu0"),
            string("state", "running"),
            word("reg", 0),
            word("timebase-frequency", counter_hz as u32),
            word("clock-frequency", 1_000_000_000),
        ],
        &[],
    );
    node(&[string("name", "cpus")], &[cpu0])
}

/// The device tree for one CPU, RAM at `[ram_base, ram_base + ram_size)` and
/// `boot_args` as the kernel's command line.
pub fn build(ram_base: u64, ram_size: u64, boot_args: &str) -> Vec<u8> {
    let panic_log = ram_base + ram_size - u64::from(PANIC_LOG_SIZE);
    node(
        &[
            string("name", "device-tree"),
            string("compatible", "qemu,virt"),
            string("model", "QEMU,virt"),
        ],
        &[
            chosen(ram_base, ram_size, boot_args),
            node(
                &[
                    string("name", "defaults"),
                    word("serial-device", UART_PHANDLE),
                ],
                &[],
            ),
            node(&[string("name", "options")], &[]),
            arm_io(),
            cpus(),
            node(
                &[
                    string("name", "pram"),
                    words64("reg", &[panic_log, u64::from(PANIC_LOG_SIZE)]),
                ],
                &[],
            ),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A parsed node: its properties by name and its children in order.
    struct Parsed {
        properties: Vec<(String, Vec<u8>)>,
        children: Vec<Parsed>,
    }
    fn u32_at(bytes: &[u8], at: usize) -> usize {
        u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize
    }
    /// Walk the tree the way `pexpert/gen/device_tree.c` does.
    fn parse(bytes: &[u8], at: &mut usize) -> Parsed {
        let (count, children) = (u32_at(bytes, *at), u32_at(bytes, *at + 4));
        *at += 8;
        let mut properties = Vec::new();
        for _ in 0..count {
            let name = &bytes[*at..*at + 32];
            let end = name
                .iter()
                .position(|&b| b == 0)
                .expect("name is NUL-terminated");
            let length = u32_at(bytes, *at + 32);
            let value = bytes[*at + 36..*at + 36 + length].to_vec();
            properties.push((String::from_utf8(name[..end].to_vec()).unwrap(), value));
            *at += 36 + length.next_multiple_of(4);
        }
        let children = (0..children).map(|_| parse(bytes, at)).collect();
        Parsed {
            properties,
            children,
        }
    }
    impl Parsed {
        fn get(&self, name: &str) -> &[u8] {
            &self
                .properties
                .iter()
                .find(|(n, _)| n == name)
                .unwrap_or_else(|| panic!("no property {name}"))
                .1
        }
        fn child(&self, name: &str) -> &Parsed {
            self.children
                .iter()
                .find(|c| {
                    c.properties.iter().any(|(n, v)| {
                        n == "name" && v.starts_with(name.as_bytes()) && v[name.len()] == 0
                    })
                })
                .unwrap_or_else(|| panic!("no node {name}"))
        }
    }

    #[test]
    fn the_tree_parses_to_its_exact_length_with_the_nodes_xnu_requires() {
        let tree = build(0x4000_0000, 1 << 30, "-v serial=3");
        let mut at = 0;
        let root = parse(&tree, &mut at);
        assert_eq!(at, tree.len(), "properties and children cover the blob");
        let chosen = root.child("chosen");
        assert_eq!(chosen.get("dram-base"), 0x4000_0000u64.to_le_bytes());
        assert_eq!(chosen.get("dram-size"), (1u64 << 30).to_le_bytes());
        assert_eq!(chosen.get("boot-args"), b"-v serial=3\0");
        assert_eq!(
            root.child("defaults").get("serial-device"),
            root.child("arm-io").child("uart0").get("AAPL,phandle")
        );
        let gic = root.child("arm-io").child("gic");
        assert_eq!(gic.get("compatible"), b"arm,gic-v3\0");
        assert_eq!(gic.get("reg").len(), 32);
        root.child("options");
        root.child("cpus").child("cpu0");
    }

    #[test]
    fn unaligned_values_are_padded_so_the_next_property_stays_aligned() {
        let mut bytes = property("odd", &[1, 2, 3]);
        assert_eq!(bytes.len(), 32 + 4 + 4);
        assert_eq!(&bytes[36..], &[1, 2, 3, 0]);
        bytes.extend(property("next", &[]));
        assert_eq!(bytes.len() % 4, 0);
    }

    #[test]
    fn the_random_seed_is_not_all_zero() {
        let seed = random_seed();
        assert_eq!(seed.len(), RANDOM_SEED_SIZE);
        assert!(seed.iter().any(|&b| b != 0));
    }

    #[test]
    fn the_nvram_image_carries_a_valid_adler_checksum() {
        let image = nvram_image();
        assert_eq!(u32_at(&image, 16) as u32, adler32(&image[20..]));
        assert_eq!(image[0], 0x70);
        assert_eq!(&image[4..9], b"nvram");
        assert_eq!(&image[36..42], b"common");
    }
}
