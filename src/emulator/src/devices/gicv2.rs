//! GICv2 distributor and banked CPU interfaces, ported from Mirage's Gicv2.zig.
//! Register values and interrupt identifiers are always guest-controlled.

use super::bus::{BusDevice, Device};
use alloc::rc::Rc;
use core::cell::RefCell;

pub const INTERRUPTS: usize = 256;
pub const PRIVATE: usize = 32;
pub const MAX_CPUS: usize = 8;
pub const SPURIOUS: u32 = 1023;
pub const DISTRIBUTOR_SIZE: u64 = 0x1000;
pub const CPU_SIZE: u64 = 0x2000;
const WORDS: usize = INTERRUPTS / 32;

#[derive(Clone, Copy, Default)]
struct Cpu {
    enabled: u32,
    pending: u32,
    active: u32,
    priority: [u8; PRIVATE],
    interface_on: bool,
    mask: u8,
    holding: Option<u32>,
    binary_point: u8,
}

pub struct Gicv2 {
    pub cpus: u32,
    pub acting: u32,
    banked: [Cpu; MAX_CPUS],
    enabled: [u32; WORDS],
    pending: [u32; WORDS],
    active: [u32; WORDS],
    priority: [u8; INTERRUPTS],
    target: [u8; INTERRUPTS],
    configuration: [u32; INTERRUPTS / 16],
    distributor_on: bool,
    pub dropped: u64,
}

impl Default for Gicv2 {
    fn default() -> Self {
        Self {
            cpus: 1,
            acting: 0,
            banked: [Cpu::default(); MAX_CPUS],
            enabled: [0; WORDS],
            pending: [0; WORDS],
            active: [0; WORDS],
            priority: [0; INTERRUPTS],
            target: [0; INTERRUPTS],
            configuration: [0; INTERRUPTS / 16],
            distributor_on: false,
            dropped: 0,
        }
    }
}

impl Gicv2 {
    pub fn cpu_index(&self, cpu: u32) -> usize {
        if cpu < self.cpus && (cpu as usize) < MAX_CPUS {
            cpu as usize
        } else {
            0
        }
    }
    pub fn holding_for(&self, cpu: u32) -> Option<u32> {
        self.banked.get(cpu as usize).and_then(|own| own.holding)
    }
    pub fn raise(&mut self, intid: u32) {
        self.shared_level(intid, true);
    }
    pub fn lower(&mut self, intid: u32) {
        self.shared_level(intid, false);
    }
    fn shared_level(&mut self, intid: u32, on: bool) {
        if !(PRIVATE as u32..INTERRUPTS as u32).contains(&intid) {
            self.dropped += 1;
            return;
        }
        let bit = 1 << (intid % 32);
        let word = &mut self.pending[intid as usize / 32];
        if on {
            *word |= bit;
        } else {
            *word &= !bit;
        }
    }
    pub fn raise_on(&mut self, cpu: u32, intid: u32) {
        self.private_level(cpu, intid, true);
    }
    pub fn lower_on(&mut self, cpu: u32, intid: u32) {
        self.private_level(cpu, intid, false);
    }
    fn private_level(&mut self, cpu: u32, intid: u32, on: bool) {
        if cpu >= self.cpus || cpu as usize >= MAX_CPUS || intid >= PRIVATE as u32 {
            self.dropped += 1;
            return;
        }
        let own = &mut self.banked[cpu as usize];
        if on {
            own.pending |= 1 << intid;
        } else {
            own.pending &= !(1 << intid);
        }
    }
    fn highest_for(&self, cpu: u32) -> Option<u32> {
        if !self.distributor_on || cpu >= self.cpus || cpu as usize >= MAX_CPUS {
            return None;
        }
        let own = &self.banked[cpu as usize];
        if !own.interface_on || own.active != 0 || own.holding.is_some() {
            return None;
        }
        // Almost always nothing is pending. Rule that out with one pass over the
        // bitmaps before scanning every interrupt individually.
        let shared_ready =
            (PRIVATE / 32..WORDS).any(|w| self.enabled[w] & self.pending[w] & !self.active[w] != 0);
        if own.enabled & own.pending == 0 && !shared_ready {
            return None;
        }
        let mut best = None;
        let mut priority = 0xff;
        for intid in 0..INTERRUPTS {
            let bit = 1 << (intid % 32);
            let ready = if intid < PRIVATE {
                own.enabled & own.pending & bit != 0
            } else {
                self.enabled[intid / 32] & self.pending[intid / 32] & !self.active[intid / 32] & bit
                    != 0
                    && self.target[intid] & (1 << cpu) != 0
            };
            let candidate = if intid < PRIVATE {
                own.priority[intid]
            } else {
                self.priority[intid]
            };
            if ready && candidate < own.mask && candidate < priority {
                best = Some(intid as u32);
                priority = candidate;
            }
        }
        best
    }
    pub fn signalled(&self, cpu: u32) -> bool {
        self.highest_for(cpu).is_some()
    }
    pub fn send_message_from(&mut self, sender: u32, value: u32) -> u8 {
        let intid = value & 15;
        let filter = (value >> 24) & 3;
        let listed = ((value >> 16) & 0xff) as u8;
        let mut targets = 0u8;
        for cpu in 0..self.cpus.min(MAX_CPUS as u32) {
            let wanted = match filter {
                0 => listed & (1 << cpu) != 0,
                1 => cpu != sender,
                2 => cpu == sender,
                _ => false,
            };
            if wanted {
                self.banked[cpu as usize].pending |= 1 << intid;
                targets |= 1 << cpu;
            }
        }
        targets
    }
    fn word_at(offset: u64, base: u64, words: usize) -> Option<usize> {
        if (base..base + words as u64 * 4).contains(&offset) {
            Some(((offset - base) / 4) as usize)
        } else {
            None
        }
    }
    pub fn read_distributor(&mut self, offset: u64, size: u8) -> u64 {
        self.read_distributor_for(self.acting, offset, size)
    }
    pub fn read_distributor_for(&mut self, cpu: u32, offset: u64, size: u8) -> u64 {
        let own = self.cpu_index(cpu);
        // Priority and target registers are byte-addressable, including word accesses.
        if (0x400..0x900).contains(&offset) && ((offset < 0x500) || offset >= 0x800) {
            let base = if offset < 0x500 { 0x400 } else { 0x800 };
            let mut value = 0;
            for byte in 0..size.min(8) {
                let intid = (offset - base + byte as u64) as usize;
                if intid >= INTERRUPTS {
                    break;
                }
                let part = if base == 0x400 {
                    if intid < PRIVATE {
                        self.banked[own].priority[intid]
                    } else {
                        self.priority[intid]
                    }
                } else if intid < PRIVATE {
                    1 << own
                } else {
                    self.target[intid]
                };
                value |= (part as u64) << (byte * 8);
            }
            return value;
        }
        for (base, shared, banked) in [
            (0x100, &self.enabled, self.banked[own].enabled),
            (0x180, &self.enabled, self.banked[own].enabled),
            (0x200, &self.pending, self.banked[own].pending),
            (0x280, &self.pending, self.banked[own].pending),
            (0x300, &self.active, self.banked[own].active),
            (0x380, &self.active, self.banked[own].active),
        ] {
            if let Some(index) = Self::word_at(offset, base, WORDS) {
                return if index == 0 {
                    banked as u64
                } else {
                    shared[index] as u64
                };
            }
        }
        if let Some(index) = Self::word_at(offset, 0xc00, INTERRUPTS / 16) {
            return self.configuration[index] as u64;
        }
        match offset {
            0 => self.distributor_on as u64,
            4 => (WORDS as u64 - 1) | ((self.cpus.clamp(1, MAX_CPUS as u32) as u64 - 1) << 5),
            8 => 0x0200_043b,
            _ => 0,
        }
    }
    pub fn write_distributor(&mut self, offset: u64, size: u8, value: u64) {
        self.write_distributor_for(self.acting, offset, size, value);
    }
    pub fn write_distributor_for(&mut self, cpu: u32, offset: u64, size: u8, value: u64) -> u8 {
        let own = self.cpu_index(cpu);
        if (0x400..0x500).contains(&offset) || (0x800..0x900).contains(&offset) {
            let base = if offset < 0x500 { 0x400 } else { 0x800 };
            for byte in 0..size.min(8) {
                let intid = (offset - base + byte as u64) as usize;
                if intid >= INTERRUPTS {
                    break;
                }
                let part = (value >> (byte * 8)) as u8;
                if base == 0x400 {
                    if intid < PRIVATE {
                        self.banked[own].priority[intid] = part;
                    } else {
                        self.priority[intid] = part;
                    }
                } else if intid >= PRIVATE {
                    self.target[intid] = part;
                }
            }
            return 0;
        }
        let word = value as u32;
        for (base, set, which) in [
            (0x100, true, 0),
            (0x180, false, 0),
            (0x200, true, 1),
            (0x280, false, 1),
            (0x300, true, 2),
            (0x380, false, 2),
        ] {
            if let Some(index) = Self::word_at(offset, base, WORDS) {
                let destination = match (index == 0, which) {
                    (true, 0) => &mut self.banked[own].enabled,
                    (true, 1) => &mut self.banked[own].pending,
                    (true, _) => &mut self.banked[own].active,
                    (false, 0) => &mut self.enabled[index],
                    (false, 1) => &mut self.pending[index],
                    (false, _) => &mut self.active[index],
                };
                if set {
                    *destination |= word;
                } else {
                    *destination &= !word;
                }
                return 0;
            }
        }
        if let Some(index) = Self::word_at(offset, 0xc00, INTERRUPTS / 16) {
            // SGIs are permanently edge-triggered; PPIs/SPIs retain their configuration.
            self.configuration[index] = if index == 0 {
                0xaaaa_aaaa
            } else {
                word & 0xaaaa_aaaa
            };
            return 0;
        }
        match offset {
            0 => self.distributor_on = word & 1 != 0,
            0xf00 => return self.send_message_from(cpu, word),
            _ => {}
        }
        0
    }
    pub fn read_cpu(&mut self, offset: u64, size: u8) -> u64 {
        self.read_cpu_for(self.acting, offset, size)
    }
    pub fn read_cpu_for(&mut self, cpu: u32, offset: u64, _size: u8) -> u64 {
        let own = self.cpu_index(cpu);
        match offset {
            0 => self.banked[own].interface_on as u64,
            4 => self.banked[own].mask as u64,
            8 => self.banked[own].binary_point as u64,
            0xc => {
                let Some(found) = self.highest_for(own as u32) else {
                    return SPURIOUS as u64;
                };
                let bit = 1 << (found % 32);
                if found < PRIVATE as u32 {
                    self.banked[own].pending &= !bit;
                    self.banked[own].active |= bit;
                } else {
                    self.pending[found as usize / 32] &= !bit;
                    self.active[found as usize / 32] |= bit;
                    self.banked[own].holding = Some(found);
                }
                found as u64
            }
            0x18 => self.highest_for(own as u32).unwrap_or(SPURIOUS) as u64,
            0xfc => 0x0002_043b,
            _ => 0,
        }
    }
    pub fn write_cpu(&mut self, offset: u64, size: u8, value: u64) {
        self.write_cpu_for(self.acting, offset, size, value);
    }
    pub fn write_cpu_for(&mut self, cpu: u32, offset: u64, _size: u8, value: u64) {
        let own = self.cpu_index(cpu);
        match offset {
            0 => self.banked[own].interface_on = value & 1 != 0,
            4 => self.banked[own].mask = value as u8,
            8 => self.banked[own].binary_point = value as u8 & 7,
            0x10 => {
                let intid = (value & 0x3ff) as usize;
                if intid >= INTERRUPTS {
                    return;
                }
                if intid < PRIVATE {
                    self.banked[own].active &= !(1 << intid);
                } else if self.banked[own].holding == Some(intid as u32) {
                    self.active[intid / 32] &= !(1 << (intid % 32));
                    self.banked[own].holding = None;
                }
            }
            _ => {}
        }
    }
    pub fn devices(
        controller: &Rc<RefCell<Self>>,
        distributor_base: u64,
        cpu_base: u64,
    ) -> [BusDevice; 2] {
        [
            BusDevice::new(
                distributor_base,
                DISTRIBUTOR_SIZE,
                Window {
                    controller: controller.clone(),
                    cpu: false,
                },
            ),
            BusDevice::new(
                cpu_base,
                CPU_SIZE,
                Window {
                    controller: controller.clone(),
                    cpu: true,
                },
            ),
        ]
    }
}

struct Window {
    controller: Rc<RefCell<Gicv2>>,
    cpu: bool,
}
impl Device for Window {
    fn read(&mut self, offset: u64, size: u8) -> u64 {
        let mut controller = self.controller.borrow_mut();
        if self.cpu {
            controller.read_cpu(offset, size)
        } else {
            controller.read_distributor(offset, size)
        }
    }
    fn write(&mut self, offset: u64, size: u8, value: u64) {
        let mut controller = self.controller.borrow_mut();
        if self.cpu {
            controller.write_cpu(offset, size, value);
        } else {
            controller.write_distributor(offset, size, value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn enabled(gic: &mut Gicv2, cpu: u32) {
        gic.acting = cpu;
        gic.write_distributor(0, 4, 1);
        gic.write_cpu(0, 4, 1);
        gic.write_cpu(4, 4, 0xff);
    }
    #[test]
    fn private_interrupt_acknowledges_and_retires_on_its_cpu() {
        let mut gic = Gicv2 {
            cpus: 2,
            ..Gicv2::default()
        };
        enabled(&mut gic, 0);
        gic.write_distributor(0x100, 4, 1 << 27);
        enabled(&mut gic, 1);
        gic.write_distributor(0x100, 4, 1 << 27);
        gic.raise_on(0, 27);
        assert!(gic.signalled(0));
        assert!(!gic.signalled(1));
        gic.acting = 0;
        assert_eq!(gic.read_cpu(0xc, 4), 27);
        assert!(!gic.signalled(0));
        gic.raise_on(0, 27);
        assert!(!gic.signalled(0));
        gic.write_cpu(0x10, 4, 27 | (3 << 10));
        assert!(gic.signalled(0));
    }
    #[test]
    fn word_priority_and_target_accesses_cover_four_interrupts() {
        let mut gic = Gicv2::default();
        gic.write_distributor(0x420, 4, 0x12345678);
        gic.write_distributor(0x820, 4, 0x01010101);
        assert_eq!(gic.read_distributor(0x420, 4), 0x12345678);
        assert_eq!(gic.read_distributor(0x821, 1), 1);
    }
    #[test]
    fn shared_interrupt_can_only_be_retired_by_owner() {
        let mut gic = Gicv2 {
            cpus: 2,
            ..Gicv2::default()
        };
        enabled(&mut gic, 0);
        enabled(&mut gic, 1);
        gic.write_distributor(0x104, 4, 2);
        gic.write_distributor(0x821, 1, 3);
        gic.raise(33);
        gic.acting = 0;
        assert_eq!(gic.read_cpu(0xc, 4), 33);
        gic.acting = 1;
        gic.write_cpu(0x10, 4, 33);
        assert_eq!(gic.holding_for(0), Some(33));
        gic.acting = 0;
        gic.write_cpu(0x10, 4, 33);
        assert_eq!(gic.holding_for(0), None);
    }
}
