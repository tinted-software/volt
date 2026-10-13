//! QEMU virt's PL031 real-time clock, driven by the host monotonic clock.
//!
//! Like QEMU's PL031, the counter is permanently enabled. Alarm status is
//! latched when the device is accessed; no interrupt line is connected.

use core::time::Duration;
use std::time::Instant;

use super::bus::Device;

pub const BASE: u64 = 0x0901_0000;
pub const LEN: u64 = 0x1000;

const DR: u64 = 0x00;
const MR: u64 = 0x04;
const LR: u64 = 0x08;
const CR: u64 = 0x0c;
const IMSC: u64 = 0x10;
const RIS: u64 = 0x14;
const MIS: u64 = 0x18;
const ICR: u64 = 0x1c;
const PERIPH_ID0: u64 = 0xfe0;
const IDENTIFICATION: [u32; 8] = [0x31, 0x10, 0x14, 0x00, 0x0d, 0xf0, 0x05, 0xb1];

/// A wrapping 32-bit seconds counter with PL031 alarm and identification registers.
pub struct Pl031 {
    origin: Instant,
    tick_offset: u32,
    load: u32,
    match_value: u32,
    interrupt_mask: u32,
    raw_interrupt: u32,
    alarm_deadline: Option<Duration>,
}

impl Pl031 {
    /// Start at `epoch_seconds`; subsequent time is elapsed host monotonic time.
    pub fn new(epoch_seconds: u32) -> Self {
        Self::new_at(epoch_seconds, Instant::now())
    }

    fn new_at(epoch_seconds: u32, now: Instant) -> Self {
        Self {
            origin: now,
            tick_offset: epoch_seconds,
            load: 0,
            match_value: 0,
            interrupt_mask: 0,
            raw_interrupt: 0,
            alarm_deadline: None,
        }
    }

    fn count(&self, elapsed: Duration) -> u32 {
        self.tick_offset.wrapping_add(elapsed.as_secs() as u32)
    }

    fn refresh_alarm(&mut self, elapsed: Duration) {
        if self
            .alarm_deadline
            .is_some_and(|deadline| elapsed >= deadline)
        {
            self.raw_interrupt = 1;
            self.alarm_deadline = None;
        }
    }

    fn set_alarm(&mut self, elapsed: Duration) {
        // Unsigned subtraction schedules a past match after the counter wraps.
        // QEMU raises an already matching alarm immediately, otherwise schedules
        // it this many seconds after programming MR or LR.
        let ticks = self.match_value.wrapping_sub(self.count(elapsed));
        if ticks == 0 {
            self.raw_interrupt = 1;
            self.alarm_deadline = None;
        } else {
            self.alarm_deadline = Some(elapsed + Duration::from_secs(u64::from(ticks)));
        }
    }

    fn access(offset: u64, size: u8) -> Option<(u64, u32, u32)> {
        // The APB register width is 32 bits. Byte and halfword accesses select
        // little-endian lanes, including the byte-sized PrimeCell IDs.
        if !matches!(size, 1 | 2 | 4) || offset >= LEN || (offset & 3) + u64::from(size) > 4 {
            return None;
        }
        let shift = ((offset & 3) * 8) as u32;
        let mask = u32::MAX >> (32 - u32::from(size) * 8);
        Some((offset & !3, shift, mask))
    }

    fn read_at(&mut self, offset: u64, size: u8, now: Instant) -> u64 {
        let Some((register, shift, mask)) = Self::access(offset, size) else {
            return 0;
        };
        let elapsed = now.duration_since(self.origin);
        self.refresh_alarm(elapsed);
        let value = match register {
            DR => self.count(elapsed),
            MR => self.match_value,
            LR => self.load,
            CR => 1, // QEMU's RTC is permanently enabled.
            IMSC => self.interrupt_mask,
            RIS => self.raw_interrupt,
            MIS => self.raw_interrupt & self.interrupt_mask,
            PERIPH_ID0..=0xffc => IDENTIFICATION[((register - PERIPH_ID0) / 4) as usize],
            _ => 0, // Includes the write-only interrupt clear register.
        };
        u64::from((value >> shift) & mask)
    }

    fn write_at(&mut self, offset: u64, size: u8, value: u64, now: Instant) {
        let Some((register, shift, mask)) = Self::access(offset, size) else {
            return;
        };
        let elapsed = now.duration_since(self.origin);
        self.refresh_alarm(elapsed);
        let bits = ((value as u32) & mask) << shift;
        let lane_mask = mask << shift;
        match register {
            MR => {
                self.match_value = (self.match_value & !lane_mask) | bits;
                self.set_alarm(elapsed);
            }
            LR => {
                self.load = (self.load & !lane_mask) | bits;
                // Keep the oscillator's subsecond phase when setting the time.
                self.tick_offset = self
                    .tick_offset
                    .wrapping_add(self.load.wrapping_sub(self.count(elapsed)));
                self.set_alarm(elapsed);
            }
            IMSC => self.interrupt_mask = ((self.interrupt_mask & !lane_mask) | bits) & 1,
            ICR => self.raw_interrupt &= !bits,
            // CR writes are ignored, as in QEMU; all remaining registers are
            // read-only or reserved.
            _ => {}
        }
    }
}

impl Device for Pl031 {
    fn read(&mut self, offset: u64, size: u8) -> u64 {
        self.read_at(offset, size, Instant::now())
    }

    fn write(&mut self, offset: u64, size: u8, value: u64) {
        self.write_at(offset, size, value, Instant::now());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_advances_in_whole_seconds_and_wraps() {
        let start = Instant::now();
        let mut rtc = Pl031::new_at(u32::MAX - 1, start);
        assert_eq!(rtc.read_at(DR, 4, start), u64::from(u32::MAX - 1));
        assert_eq!(
            rtc.read_at(DR, 4, start + Duration::from_millis(999)),
            u64::from(u32::MAX - 1)
        );
        assert_eq!(
            rtc.read_at(DR, 4, start + Duration::from_secs(1)),
            u64::from(u32::MAX)
        );
        assert_eq!(rtc.read_at(DR, 4, start + Duration::from_secs(2)), 0);
        assert_eq!(rtc.read_at(DR, 4, start + Duration::from_secs(3)), 1);
    }

    #[test]
    fn loading_time_preserves_fractional_phase_and_load_readback() {
        let start = Instant::now();
        let mut rtc = Pl031::new_at(100, start);
        rtc.write_at(LR, 4, 500, start + Duration::from_millis(1750));
        assert_eq!(rtc.read_at(LR, 4, start + Duration::from_millis(1750)), 500);
        assert_eq!(rtc.read_at(DR, 4, start + Duration::from_millis(1999)), 500);
        assert_eq!(rtc.read_at(DR, 4, start + Duration::from_secs(2)), 501);
        assert_eq!(rtc.read_at(LR, 4, start + Duration::from_secs(2)), 500);
        rtc.write_at(LR, 4, 20, start + Duration::from_secs(3));
        assert_eq!(rtc.read_at(DR, 4, start + Duration::from_secs(3)), 20);
        assert_eq!(rtc.read_at(DR, 4, start + Duration::from_secs(4)), 21);
    }

    #[test]
    fn control_is_permanently_enabled_like_qemu() {
        let start = Instant::now();
        let mut rtc = Pl031::new_at(100, start);
        assert_eq!(rtc.read_at(CR, 4, start), 1);
        rtc.write_at(CR, 4, 0, start);
        assert_eq!(rtc.read_at(CR, 4, start), 1);
        assert_eq!(rtc.read_at(DR, 4, start + Duration::from_secs(2)), 102);
        rtc.write_at(CR, 4, u64::MAX, start + Duration::from_secs(2));
        assert_eq!(rtc.read_at(CR, 4, start + Duration::from_secs(2)), 1);
    }

    #[test]
    fn match_latches_raw_status_independently_of_the_mask() {
        let start = Instant::now();
        let mut rtc = Pl031::new_at(100, start);
        rtc.write_at(MR, 4, 102, start);
        assert_eq!(rtc.read_at(MR, 4, start), 102);
        assert_eq!(rtc.read_at(RIS, 4, start + Duration::from_secs(1)), 0);
        let after = start + Duration::from_secs(5);
        // A missed matching second still latches the interrupt.
        assert_eq!(rtc.read_at(RIS, 4, after), 1);
        assert_eq!(rtc.read_at(MIS, 4, after), 0);
        rtc.write_at(IMSC, 4, 0xffff, after);
        assert_eq!(rtc.read_at(IMSC, 4, after), 1);
        assert_eq!(rtc.read_at(MIS, 4, after), 1);
        rtc.write_at(IMSC, 4, 0, after);
        assert_eq!(rtc.read_at(MIS, 4, after), 0);
        assert_eq!(rtc.read_at(RIS, 4, after), 1);
        rtc.write_at(ICR, 4, 2, after);
        assert_eq!(rtc.read_at(RIS, 4, after), 1);
        rtc.write_at(ICR, 4, 1, after);
        assert_eq!(rtc.read_at(RIS, 4, after), 0);
        assert_eq!(rtc.read_at(ICR, 4, after), 0);
    }

    #[test]
    fn match_scheduling_wraps_and_load_reschedules_the_alarm() {
        let start = Instant::now();
        let mut rtc = Pl031::new_at(u32::MAX - 1, start);
        rtc.write_at(MR, 4, 1, start);
        assert_eq!(rtc.read_at(RIS, 4, start + Duration::from_secs(2)), 0);
        assert_eq!(rtc.read_at(RIS, 4, start + Duration::from_secs(3)), 1);
        let now = start + Duration::from_secs(3);
        rtc.write_at(ICR, 4, 1, now);
        assert_eq!(rtc.read_at(RIS, 4, now), 0);
        rtc.write_at(MR, 4, 1, now); // Equal to the current count.
        assert_eq!(rtc.read_at(RIS, 4, now), 1);
        rtc.write_at(ICR, 4, 1, now);
        rtc.write_at(MR, 4, 10, now);
        rtc.write_at(LR, 4, 9, now);
        assert_eq!(rtc.read_at(RIS, 4, now), 0);
        assert_eq!(rtc.read_at(RIS, 4, now + Duration::from_secs(1)), 1);
        rtc.write_at(ICR, 4, 1, now + Duration::from_secs(1));
        rtc.write_at(LR, 4, 10, now + Duration::from_secs(1));
        assert_eq!(rtc.read_at(RIS, 4, now + Duration::from_secs(1)), 1);
    }

    #[test]
    fn primecell_ids_and_little_endian_register_lanes_are_readable() {
        let start = Instant::now();
        let mut rtc = Pl031::new_at(0x1234_5678, start);
        for (index, expected) in IDENTIFICATION.into_iter().enumerate() {
            let offset = PERIPH_ID0 + index as u64 * 4;
            assert_eq!(rtc.read_at(offset, 1, start), u64::from(expected));
            assert_eq!(rtc.read_at(offset, 4, start), u64::from(expected));
            assert_eq!(rtc.read_at(offset + 1, 1, start), 0);
        }
        assert_eq!(rtc.read_at(DR, 1, start), 0x78);
        assert_eq!(rtc.read_at(DR + 1, 1, start), 0x56);
        assert_eq!(rtc.read_at(DR + 2, 2, start), 0x1234);
        rtc.write_at(MR, 4, 0x1122_3344, start);
        rtc.write_at(MR + 1, 1, 0xaa, start);
        assert_eq!(rtc.read_at(MR, 4, start), 0x1122_aa44);
        rtc.write_at(LR, 4, 0x0102_0304, start);
        rtc.write_at(LR + 2, 2, 0xaabb, start);
        assert_eq!(rtc.read_at(LR, 4, start), 0xaabb_0304);
        assert_eq!(rtc.read_at(DR, 4, start), 0xaabb_0304);
    }

    #[test]
    fn read_only_reserved_and_invalid_accesses_do_not_change_registers() {
        let start = Instant::now();
        let mut rtc = Pl031::new_at(100, start);
        for offset in [DR, RIS, MIS, PERIPH_ID0, 0x20, LEN, u64::MAX] {
            rtc.write_at(offset, 4, u64::MAX, start);
        }
        assert_eq!(rtc.read_at(DR, 4, start), 100);
        assert_eq!(rtc.read_at(RIS, 4, start), 0);
        assert_eq!(rtc.read_at(MIS, 4, start), 0);
        assert_eq!(rtc.read_at(PERIPH_ID0, 1, start), 0x31);
        for (offset, size) in [(MR, 0), (MR, 3), (MR, 8), (MR + 3, 2), (LEN, 4)] {
            rtc.write_at(offset, size, u64::MAX, start);
            assert_eq!(rtc.read_at(offset, size, start), 0);
        }
        assert_eq!(rtc.read_at(MR, 4, start), 0);
        assert_eq!(rtc.read_at(0x20, 4, start), 0);
        assert_eq!(rtc.read_at(LEN, 4, start), 0);
    }
}
