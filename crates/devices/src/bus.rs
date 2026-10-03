//! The bus that carries a guest memory access to the device owning the address.
//!
//! The guest chooses the address, so an access matching no device is a
//! recoverable fault and never an assertion. It is counted, because a guest is
//! allowed to be wrong and a fault recovered in silence is a bug that hides
//! itself.

use alloc::boxed::Box;
use alloc::vec::Vec;

/// A memory-mapped device behind the bus.
pub trait Device {
    /// Read `size` bytes at `offset` (relative to the device base).
    fn read(&mut self, offset: u64, size: u8) -> u64;
    /// Write the low `size` bytes of `value` at `offset`.
    fn write(&mut self, offset: u64, size: u8, value: u64);
}

/// One device's window on the bus.
pub struct BusDevice {
    base: u64,
    len: u64,
    device: Box<dyn Device>,
}

impl BusDevice {
    /// Map `device` at `[base, base + len)`.
    pub fn new(base: u64, len: u64, device: impl Device + 'static) -> Self {
        Self {
            base,
            len,
            device: Box::new(device),
        }
    }
}

/// Routes guest-physical accesses to devices by address.
#[derive(Default)]
pub struct Bus {
    devices: Vec<BusDevice>,
    /// Accesses that matched no device. Never a host error: the guest asked
    /// for something that does not exist, and its own handler hears about it.
    pub unmapped: u64,
}

impl Bus {
    /// An empty bus; add windows with [`Bus::attach`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach a device window to the bus.
    pub fn attach(&mut self, device: BusDevice) {
        self.devices.push(device);
    }

    fn find(&mut self, address: u64, size: u8) -> Option<(u64, &mut Box<dyn Device>)> {
        // Both the address and the width come from the guest. Unchecked
        // arithmetic wraps and then passes a bounds test it should fail.
        let end = address.checked_add(size as u64)?;
        for owner in &mut self.devices {
            if address < owner.base {
                continue;
            }
            if end > owner.base + owner.len {
                continue;
            }
            return Some((address - owner.base, &mut owner.device));
        }
        None
    }

    /// Read `size` bytes at guest-physical `address`, or `None` when unmapped.
    pub fn read(&mut self, address: u64, size: u8) -> Option<u64> {
        match self.find(address, size) {
            Some((offset, device)) => Some(device.read(offset, size)),
            None => {
                self.unmapped += 1;
                None
            }
        }
    }

    /// Write the low `size` bytes of `value` at `address`; `false` when unmapped.
    pub fn write(&mut self, address: u64, size: u8, value: u64) -> bool {
        match self.find(address, size) {
            Some((offset, device)) => {
                device.write(offset, size, value);
                true
            }
            None => {
                self.unmapped += 1;
                false
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Bus, BusDevice, Device};

    struct Counter {
        reads: u32,
        writes: u32,
        last_offset: u64,
        last_value: u64,
    }

    impl Device for Counter {
        fn read(&mut self, offset: u64, _size: u8) -> u64 {
            self.reads += 1;
            self.last_offset = offset;
            0x42
        }

        fn write(&mut self, offset: u64, _size: u8, value: u64) {
            self.writes += 1;
            self.last_offset = offset;
            self.last_value = value;
        }
    }

    #[test]
    fn routes_by_address_and_counts_unmapped() {
        let mut bus = Bus::new();
        bus.attach(BusDevice::new(
            0x9000_0000,
            0x1000,
            Counter {
                reads: 0,
                writes: 0,
                last_offset: 0,
                last_value: 0,
            },
        ));
        assert_eq!(bus.read(0x9000_0000, 1), Some(0x42));
        assert!(bus.write(0x9000_0040, 4, 65));
        assert_eq!(bus.read(0x8000_0000, 4), None);
        assert!(!bus.write(0x8000_0000, 4, 1));
        assert_eq!(bus.unmapped, 2);
        // An access straddling the end of the window matches nothing.
        assert_eq!(bus.read(0x9000_0fff, 2), None);
        assert_eq!(bus.unmapped, 3);
        // Wrapping arithmetic matches nothing rather than aliasing device zero.
        assert_eq!(bus.read(u64::MAX, 2), None);
        assert_eq!(bus.unmapped, 4);
    }
}
