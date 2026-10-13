//! Apple's power manager (PMGR), as the M1 Linux power-domain driver drives it.
//!
//! Each power domain has one 32-bit power-state register (`apple-pmgr-pwrstate`):
//! `PS_TARGET` in bits 3:0 is the state the driver asks for, and `PS_ACTUAL` in bits
//! 7:4 is the state the hardware is in. The driver writes the target and polls until
//! the actual state matches. This model acknowledges a write at once, so the actual
//! state is the target, and every register starts in the active state iBoot leaves
//! the always-on domains in. Other bits are stored as written.
use alloc::vec;
use alloc::vec::Vec;

/// `PS_ACTIVE`: the target and the actual state both name the active state.
const ACTIVE: u32 = 0xff;
const TARGET: u32 = 0x0f;
const ACTUAL: u32 = 0xf0;

/// The power-state registers of one PMGR block of `len` bytes.
pub struct Pmgr {
    words: Vec<u32>,
}

impl Pmgr {
    /// A block of `len` bytes (a multiple of four) with every power domain active.
    pub fn new(len: u64) -> Self {
        Self {
            words: vec![ACTIVE; (len / 4) as usize],
        }
    }

    /// Read `size` bytes at `offset` (relative to the block base).
    pub fn read(&self, offset: u64, size: u8) -> u64 {
        (0..u64::from(size)).fold(0, |value, i| {
            let at = offset + i;
            let word = self.words[(at / 4) as usize];
            value | u64::from((word >> (8 * (at % 4))) as u8) << (8 * i)
        })
    }

    /// Write the low `size` bytes of `value` at `offset`. Each 32-bit register the write
    /// touches takes its target power state at once.
    pub fn write(&mut self, offset: u64, size: u8, value: u64) {
        for i in 0..u64::from(size) {
            let at = offset + i;
            let shift = 8 * (at % 4);
            let byte = ((value >> (8 * i)) & 0xff) as u32;
            let word = &mut self.words[(at / 4) as usize];
            *word = (*word & !(0xff << shift)) | byte << shift;
        }
        let first = (offset / 4) as usize;
        let last = ((offset + u64::from(size) - 1) / 4) as usize;
        for word in &mut self.words[first..=last] {
            *word = (*word & !ACTUAL) | (*word & TARGET) << 4;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PS: u64 = 0x100;
    const LEN: u64 = 0x4000;

    #[test]
    fn every_power_domain_starts_active() {
        let pmgr = Pmgr::new(LEN);
        assert_eq!(pmgr.read(PS, 4) & 0xff, 0xff);
    }

    #[test]
    fn a_target_is_acknowledged_as_the_actual_state_at_once() {
        let mut pmgr = Pmgr::new(LEN);
        pmgr.write(PS, 4, 0x0);
        let state = pmgr.read(PS, 4);
        assert_eq!(state & 0x0f, 0, "target is what was written");
        assert_eq!((state >> 4) & 0x0f, 0, "actual follows the target");
        pmgr.write(PS, 4, 0x0f | 1 << 31);
        assert_eq!(
            pmgr.read(PS, 4) & 0xff,
            0xff,
            "active again, reset bit stored"
        );
    }

    #[test]
    fn narrow_writes_change_only_their_bytes_and_flags_are_stored() {
        let mut pmgr = Pmgr::new(LEN);
        // WAS_PWRGATED (bit 8) lives in the second byte: a byte write leaves the rest.
        pmgr.write(PS + 1, 1, 0x1);
        assert_eq!(pmgr.read(PS + 1, 1), 0x1);
        assert_eq!(pmgr.read(PS, 1), 0xff);
    }
}
