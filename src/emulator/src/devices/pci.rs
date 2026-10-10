//! PCI for the `virt` board: the root bus's configuration space (ECAM), and the memory
//! BARs of the functions on it.
//!
//! Only bus 0 is populated, one function per slot, and no bridges sit below it. Firmware
//! assigns the BARs itself, as it does under QEMU, so a function claims the addresses its
//! BARs hold once the command register enables memory decoding.

use alloc::boxed::Box;
use alloc::vec::Vec;

/// The ECAM window of QEMU `virt`'s high PCIe region: 256 buses of 4 KiB functions.
pub const ECAM_BASE: u64 = 0x40_1000_0000;
pub const ECAM_SIZE: u64 = 0x1000_0000;

/// The host bridge QEMU's GPEX reports at 00:00.0.
const HOST_BRIDGE_VENDOR: u16 = 0x1b36;
const HOST_BRIDGE_DEVICE: u16 = 0x0008;
/// Base class 0x06, bridge; subclass 0x00, host bridge.
const HOST_BRIDGE_CLASS: u32 = 0x06_00_00;

const CONFIG_LEN: usize = 256;
pub const BAR_COUNT: usize = 6;
const BAR_OFFSET: usize = 0x10;
const CAPABILITIES_POINTER: usize = 0x34;
/// Capabilities start after the 64-byte standard header.
const FIRST_CAPABILITY: usize = 0x40;
const STATUS: usize = 0x06;
const STATUS_CAPABILITIES: u8 = 0x10;
const COMMAND: usize = 0x04;
/// The command bits a driver may set: I/O, memory, bus master, special cycles, MWI,
/// VGA palette snoop, parity, SERR, fast back-to-back and INTx disable.
const COMMAND_WRITABLE: [u8; 2] = [0x47, 0x05];
const COMMAND_MEMORY: u8 = 0x2;

/// The 256-byte configuration space of a type-0 function. Writes honour a per-byte
/// writable mask, and a memory BAR keeps only the address bits its size allows, so the
/// standard sizing probe reads back the size.
pub struct ConfigSpace {
    bytes: [u8; CONFIG_LEN],
    writable: [u8; CONFIG_LEN],
    /// Size of each memory BAR in bytes, zero when the BAR is not implemented.
    bar_size: [u64; BAR_COUNT],
    /// Offset of the last capability, whose `next` pointer is filled in by the next one.
    last_capability: Option<usize>,
    /// The first offset free for another capability, four-byte aligned.
    capabilities_end: usize,
}

impl ConfigSpace {
    /// `class` is base class, subclass and programming interface, as `0xBBSSPP`.
    pub fn new(
        vendor: u16,
        device: u16,
        class: u32,
        subsystem_vendor: u16,
        subsystem: u16,
    ) -> Self {
        let mut config = Self {
            bytes: [0; CONFIG_LEN],
            writable: [0; CONFIG_LEN],
            bar_size: [0; BAR_COUNT],
            last_capability: None,
            capabilities_end: FIRST_CAPABILITY,
        };
        config.bytes[0x00..0x02].copy_from_slice(&vendor.to_le_bytes());
        config.bytes[0x02..0x04].copy_from_slice(&device.to_le_bytes());
        config.bytes[0x09] = class as u8;
        config.bytes[0x0a] = (class >> 8) as u8;
        config.bytes[0x0b] = (class >> 16) as u8;
        config.bytes[0x2c..0x2e].copy_from_slice(&subsystem_vendor.to_le_bytes());
        config.bytes[0x2e..0x30].copy_from_slice(&subsystem.to_le_bytes());
        config.writable[COMMAND] = COMMAND_WRITABLE[0];
        config.writable[COMMAND + 1] = COMMAND_WRITABLE[1];
        config
    }

    /// Declares memory BAR `index` of `size` bytes, a power of two of at least 16.
    pub fn set_bar(&mut self, index: usize, size: u64) {
        assert!(index < BAR_COUNT && size.is_power_of_two() && (16..=1 << 32).contains(&size));
        self.bar_size[index] = size;
        let at = BAR_OFFSET + 4 * index;
        self.writable[at..at + 4].fill(0xff);
    }

    /// Appends a capability with identifier `id`. `body` is what follows the `next`
    /// pointer, starting with the capability's own length byte.
    pub fn add_capability(&mut self, id: u8, body: &[u8]) {
        let at = self.capabilities_end;
        let length = 2 + body.len();
        assert!(
            at + length <= CONFIG_LEN,
            "capability does not fit configuration space"
        );
        self.bytes[at] = id;
        self.bytes[at + 1] = 0;
        self.bytes[at + 2..at + length].copy_from_slice(body);
        match self.last_capability {
            None => self.bytes[CAPABILITIES_POINTER] = at as u8,
            Some(previous) => self.bytes[previous + 1] = at as u8,
        }
        self.bytes[STATUS] |= STATUS_CAPABILITIES;
        self.last_capability = Some(at);
        self.capabilities_end = (at + length + 3) & !3;
    }

    /// The base and size of memory BAR `index`, once firmware has assigned it.
    pub fn memory_bar(&self, index: usize) -> Option<(u64, u64)> {
        let size = self.bar_size[index];
        if size == 0 {
            return None;
        }
        let at = BAR_OFFSET + 4 * index;
        let base = u32::from_le_bytes(self.bytes[at..at + 4].try_into().unwrap());
        Some((u64::from(base & !0xf), size))
    }

    /// Whether the command register lets the function decode its memory BARs.
    pub fn memory_decoding(&self) -> bool {
        self.bytes[COMMAND] & COMMAND_MEMORY != 0
    }

    /// Reads `size` bytes, little-endian. Bytes past the header read as zero.
    pub fn read(&self, offset: usize, size: usize) -> u64 {
        (0..size).fold(0, |value, i| {
            let byte = self.bytes.get(offset + i).copied().unwrap_or(0);
            value | u64::from(byte) << (8 * i)
        })
    }

    /// Writes `size` bytes, little-endian, keeping the bytes the writable mask protects.
    pub fn write(&mut self, offset: usize, size: usize, value: u64) {
        for i in 0..size {
            let at = offset + i;
            if at >= CONFIG_LEN {
                break;
            }
            let mask = self.writable[at];
            let byte = (value >> (8 * i)) as u8;
            self.bytes[at] = (self.bytes[at] & !mask) | (byte & mask);
        }
        for index in 0..BAR_COUNT {
            let size = self.bar_size[index];
            if size == 0 {
                continue;
            }
            let at = BAR_OFFSET + 4 * index;
            let written = u32::from_le_bytes(self.bytes[at..at + 4].try_into().unwrap());
            let kept = written & !((size - 1) as u32) & !0xf;
            self.bytes[at..at + 4].copy_from_slice(&kept.to_le_bytes());
        }
    }
}

/// A function on the root bus. Its BARs are reached through the bus, which decides which
/// function a bus address belongs to.
pub trait PciFunction {
    fn config(&self) -> &ConfigSpace;
    fn config_mut(&mut self) -> &mut ConfigSpace;
    /// Reads `size` bytes at `offset` into memory BAR `bar`.
    fn bar_read(&mut self, bar: usize, offset: u64, size: u8) -> u64 {
        let _ = (bar, offset, size);
        0
    }
    /// Writes `size` bytes at `offset` into memory BAR `bar`.
    fn bar_write(&mut self, bar: usize, offset: u64, size: u8, value: u64) {
        let _ = (bar, offset, size, value);
    }
}

/// The root bus: functions attached to slots, served through ECAM and their BARs.
#[derive(Default)]
pub struct PciHost {
    slots: Vec<(u8, Box<dyn PciFunction>)>,
}

impl PciHost {
    /// A bus with the GPEX host bridge at 00:00.0 and nothing else.
    pub fn new() -> Self {
        let mut host = Self::default();
        host.attach(0, Box::new(HostBridge::new()));
        host
    }

    /// Attaches `function` at `slot` on bus 0.
    pub fn attach(&mut self, slot: u8, function: Box<dyn PciFunction>) {
        assert!(slot < 32, "slot {slot} is not on bus 0");
        assert!(
            self.slots.iter().all(|(taken, _)| *taken != slot),
            "slot {slot} is already occupied"
        );
        self.slots.push((slot, function));
    }

    /// Reads `size` bytes at a physical address the bus claims, or `None` for one it does not.
    pub fn read(&mut self, address: u64, size: u8) -> Option<u64> {
        if let Some(ecam) = ecam_offset(address) {
            return Some(self.config_read(ecam, size));
        }
        let (index, bar, offset) = self.claim(address)?;
        Some(self.slots[index].1.bar_read(bar, offset, size))
    }

    /// Writes `size` bytes at a physical address the bus claims. `false` declines one it does not.
    pub fn write(&mut self, address: u64, size: u8, value: u64) -> bool {
        if let Some(ecam) = ecam_offset(address) {
            self.config_write(ecam, size, value);
            return true;
        }
        let Some((index, bar, offset)) = self.claim(address) else {
            return false;
        };
        self.slots[index].1.bar_write(bar, offset, size, value);
        true
    }

    fn config_read(&self, ecam: u64, size: u8) -> u64 {
        let all_ones = size_mask(size);
        match self.function(ecam) {
            Some((function, register)) => function.config().read(register, size as usize),
            None => all_ones,
        }
    }

    fn config_write(&mut self, ecam: u64, size: u8, value: u64) {
        let Some(slot) = self.function_index(ecam) else {
            return;
        };
        let register = (ecam & 0xfff) as usize;
        self.slots[slot]
            .1
            .config_mut()
            .write(register, size as usize, value);
    }

    /// The function an ECAM offset names, and the register within it. Only bus 0 and
    /// function 0 are populated.
    fn function(&self, ecam: u64) -> Option<(&dyn PciFunction, usize)> {
        let index = self.function_index(ecam)?;
        Some((self.slots[index].1.as_ref(), (ecam & 0xfff) as usize))
    }

    fn function_index(&self, ecam: u64) -> Option<usize> {
        let bus = ecam >> 20;
        let slot = ((ecam >> 15) & 0x1f) as u8;
        let function = (ecam >> 12) & 0x7;
        if bus != 0 || function != 0 {
            return None;
        }
        self.slots.iter().position(|(taken, _)| *taken == slot)
    }

    /// The function, BAR and offset a bus address falls in. Only functions that enable
    /// memory decoding answer.
    fn claim(&self, address: u64) -> Option<(usize, usize, u64)> {
        for (index, (_, function)) in self.slots.iter().enumerate() {
            let config = function.config();
            if !config.memory_decoding() {
                continue;
            }
            for bar in 0..BAR_COUNT {
                if let Some((base, size)) = config.memory_bar(bar)
                    && (base..base + size).contains(&address)
                {
                    return Some((index, bar, address - base));
                }
            }
        }
        None
    }
}

fn ecam_offset(address: u64) -> Option<u64> {
    (ECAM_BASE..ECAM_BASE + ECAM_SIZE)
        .contains(&address)
        .then(|| address - ECAM_BASE)
}

fn size_mask(size: u8) -> u64 {
    if size >= 8 {
        u64::MAX
    } else {
        (1u64 << (8 * size)) - 1
    }
}

/// The GPEX host bridge at 00:00.0. It has no BARs; its configuration space is what
/// enumeration reads first.
pub struct HostBridge {
    config: ConfigSpace,
}

impl HostBridge {
    fn new() -> Self {
        Self {
            config: ConfigSpace::new(
                HOST_BRIDGE_VENDOR,
                HOST_BRIDGE_DEVICE,
                HOST_BRIDGE_CLASS,
                0,
                0,
            ),
        }
    }
}

impl PciFunction for HostBridge {
    fn config(&self) -> &ConfigSpace {
        &self.config
    }

    fn config_mut(&mut self) -> &mut ConfigSpace {
        &mut self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::rc::Rc;
    use core::cell::RefCell;

    type Writes = Rc<RefCell<Vec<(usize, u64, u8, u64)>>>;

    /// A function with one 16 KiB BAR. Its BAR reads return the offset, and its writes are
    /// recorded where the test can see them.
    struct Probe {
        config: ConfigSpace,
        writes: Writes,
    }

    impl Probe {
        fn new(writes: Writes) -> Self {
            let mut config = ConfigSpace::new(0x1af4, 0x1042, 0x018000, 0x1af4, 2);
            config.set_bar(0, 0x4000);
            // A capability, so the status register's capability-list bit is set as on virtio.
            let mut body = [0u8; 14];
            body[0] = 16;
            config.add_capability(0x09, &body);
            Self { config, writes }
        }
    }

    impl PciFunction for Probe {
        fn config(&self) -> &ConfigSpace {
            &self.config
        }
        fn config_mut(&mut self) -> &mut ConfigSpace {
            &mut self.config
        }
        fn bar_read(&mut self, bar: usize, offset: u64, _size: u8) -> u64 {
            if bar == 0 { offset } else { 0 }
        }
        fn bar_write(&mut self, bar: usize, offset: u64, size: u8, value: u64) {
            self.writes.borrow_mut().push((bar, offset, size, value));
        }
    }

    fn ecam(bus: u64, slot: u64, function: u64, register: u64) -> u64 {
        ECAM_BASE | bus << 20 | slot << 15 | function << 12 | register
    }

    fn probe_host() -> (PciHost, Writes) {
        let writes = Writes::default();
        let mut host = PciHost::new();
        host.attach(1, Box::new(Probe::new(writes.clone())));
        (host, writes)
    }

    #[test]
    fn host_bridge_identifies_itself_and_empty_slots_read_as_all_ones() {
        let mut host = PciHost::new();
        assert_eq!(
            host.read(ecam(0, 0, 0, 0), 4),
            Some(u64::from(HOST_BRIDGE_VENDOR) | u64::from(HOST_BRIDGE_DEVICE) << 16)
        );
        assert_eq!(host.read(ecam(0, 5, 0, 0), 4), Some(0xffff_ffff));
        assert_eq!(host.read(ecam(0, 5, 0, 0), 2), Some(0xffff));
        assert_eq!(host.read(ecam(1, 0, 0, 0), 4), Some(0xffff_ffff));
        assert_eq!(host.read(ecam(0, 0, 3, 0), 4), Some(0xffff_ffff));
    }

    #[test]
    fn bar_sizing_probe_reads_back_the_size_mask_and_restores_the_address() {
        let (mut host, _) = probe_host();
        let bar = ecam(0, 1, 0, 0x10);
        host.write(bar, 4, 0x1000_0000);
        assert_eq!(host.read(bar, 4), Some(0x1000_0000));
        // All ones reads back as the writable address bits, which name 16 KiB.
        host.write(bar, 4, 0xffff_ffff);
        assert_eq!(host.read(bar, 4), Some(0xffff_c000));
        host.write(bar, 4, 0x1000_0000);
        assert_eq!(host.read(bar, 4), Some(0x1000_0000));
    }

    #[test]
    fn a_bar_answers_only_while_memory_decoding_is_enabled() {
        let (mut host, _) = probe_host();
        host.write(ecam(0, 1, 0, 0x10), 4, 0x1000_0000);
        assert_eq!(host.read(0x1000_0004, 4), None);
        host.write(ecam(0, 1, 0, 0x04), 2, 0x0002);
        assert_eq!(host.read(0x1000_0004, 4), Some(4));
        assert_eq!(
            host.read(0x1000_4000, 4),
            None,
            "one past the BAR is unclaimed"
        );
        assert!(!host.write(0x1000_4000, 4, 0));
    }

    #[test]
    fn a_write_into_a_bar_reaches_the_function_at_its_offset() {
        let (mut host, writes) = probe_host();
        host.write(ecam(0, 1, 0, 0x10), 4, 0x1000_0000);
        host.write(ecam(0, 1, 0, 0x04), 2, 0x0002);
        assert!(host.write(0x1000_0040, 2, 0xabcd));
        assert_eq!(writes.borrow().as_slice(), &[(0, 0x40, 2, 0xabcd)]);
    }

    #[test]
    fn the_command_register_keeps_only_writable_bits_and_status_lists_capabilities() {
        let (mut host, _) = probe_host();
        // Status is read only: its capability-list bit (bit 4) survives the write.
        host.write(ecam(0, 1, 0, 0x04), 4, 0xffff_ffff);
        assert_eq!(host.read(ecam(0, 1, 0, 0x04), 2), Some(0x0547));
        assert_eq!(host.read(ecam(0, 1, 0, 0x06), 2), Some(0x0010));
    }
}
