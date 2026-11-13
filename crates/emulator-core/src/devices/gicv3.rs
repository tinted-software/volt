//! GICv3 interrupt controller for one processing element: the part XNU's QEMU
//! platform (`pexpert/arm/pe_fiq.c`) drives.
//!
//! It brings the redistributor out of sleep, puts the virtual timer's PPI 27 in
//! group 0 (delivered as a FIQ) and enables it. The CPU interface is the `ICC_*`
//! system registers, which live in the CPU's system state; this device holds the
//! memory-mapped distributor and redistributor and decides whether PPI 27 is
//! forwarded to the CPU.
//!
//! Everything else (SPIs, SGIs, LPIs, group 1, affinity routing) reads as zero and
//! ignores writes: nothing in the XNU boot path uses it.

pub const DISTRIBUTOR_SIZE: u64 = 0x1_0000;
/// The redistributor window of QEMU `virt`: 0x20000 (two frames) per PE.
pub const REDISTRIBUTOR_SIZE: u64 = 0xf6_0000;
const FRAME: u64 = 0x1_0000;

const GICD_CTLR: u64 = 0x0;
const GICD_TYPER: u64 = 0x4;
const GICR_CTLR: u64 = 0x0;
const GICR_TYPER: u64 = 0x8;
const GICR_WAKER: u64 = 0x14;
/// `PIDR2`: the architecture revision (3 = GICv3) in bits 7:4.
const PIDR2: u64 = 0xffe8;
const PIDR2_GICV3: u64 = 0x3b;
// SGI frame, relative to its base.
const IGROUPR0: u64 = 0x80;
const ISENABLER0: u64 = 0x100;
const ICENABLER0: u64 = 0x180;
const ISPENDR0: u64 = 0x200;
const ICPENDR0: u64 = 0x280;
const IPRIORITYR: u64 = 0x400;

/// `GICD_CTLR.EnableGrp0`, `EnableGrp1NS` and `ARE_NS`: the writable bits.
const GICD_CTLR_WRITABLE: u32 = 0x1 | 0x2 | 0x10;
const GICD_CTLR_ENABLE_GRP0: u32 = 0x1;
/// `GICR_TYPER.Last`: this is the final redistributor in the window.
const TYPER_LAST: u64 = 1 << 4;
const WAKER_PROCESSOR_SLEEP: u32 = 1 << 1;
const WAKER_CHILDREN_ASLEEP: u32 = 1 << 2;
/// The virtual timer's private peripheral interrupt.
pub const VIRTUAL_TIMER: u32 = 27;

#[derive(Debug, Clone)]
pub struct Gicv3 {
    distributor_ctlr: u32,
    /// Only `ProcessorSleep` is stored; `ChildrenAsleep` follows it at once.
    processor_sleep: bool,
    group: u32,
    enabled: u32,
    pending: u32,
    priority: [u8; 32],
}

impl Default for Gicv3 {
    fn default() -> Self {
        Self {
            distributor_ctlr: 0,
            // A redistributor comes out of reset asleep.
            processor_sleep: true,
            group: 0,
            enabled: 0,
            pending: 0,
            priority: [0; 32],
        }
    }
}

/// The bytes of `value` an access of `size` bytes at `offset` within a naturally
/// aligned 64-bit register sees.
fn window(value: u64, offset: u64, size: u8) -> u64 {
    let shifted = value >> ((offset & 7) * 8);
    if size >= 8 {
        shifted
    } else {
        shifted & ((1 << (u32::from(size) * 8)) - 1)
    }
}

impl Gicv3 {
    pub fn read_distributor(&self, offset: u64, size: u8) -> u64 {
        // All of these are 32-bit registers.
        let register = match offset & !3 {
            GICD_CTLR => u64::from(self.distributor_ctlr),
            // `ITLinesNumber` 0 (32 INTIDs), one PE, 16 ID bits.
            GICD_TYPER => 15 << 19,
            PIDR2 => PIDR2_GICV3,
            _ => 0,
        };
        window(register, offset & 3, size)
    }
    pub fn write_distributor(&mut self, offset: u64, _size: u8, value: u64) {
        if offset == GICD_CTLR {
            self.distributor_ctlr = value as u32 & GICD_CTLR_WRITABLE;
        }
    }

    fn waker(&self) -> u32 {
        if self.processor_sleep {
            WAKER_PROCESSOR_SLEEP | WAKER_CHILDREN_ASLEEP
        } else {
            0
        }
    }
    pub fn read_redistributor(&self, offset: u64, size: u8) -> u64 {
        let frame = offset / FRAME;
        let at = offset % FRAME;
        // Only PE 0's frames exist; the rest of the window is reserved.
        if frame >= 2 {
            return 0;
        }
        if frame == 0 {
            return match at & !7 {
                GICR_CTLR => 0,
                // Affinity 0.0.0.0, processor number 0, and the last in the window.
                GICR_TYPER => window(TYPER_LAST, at, size),
                _ if at & !3 == GICR_WAKER => window(u64::from(self.waker()), at & 3, size),
                _ if at & !3 == PIDR2 & !3 => {
                    window(PIDR2_GICV3 << ((PIDR2 & 3) * 8), at & 3, size)
                }
                _ => 0,
            };
        }
        let word = |value: u32| window(u64::from(value), at & 3, size);
        match at {
            IGROUPR0 => word(self.group),
            ISENABLER0 | ICENABLER0 => word(self.enabled),
            ISPENDR0 | ICPENDR0 => word(self.pending),
            IPRIORITYR..=0x41f => {
                let start = (at - IPRIORITYR) as usize;
                (0..usize::from(size).min(4))
                    .map(|i| u64::from(self.priority[start + i]) << (8 * i))
                    .fold(0, |a, b| a | b)
            }
            _ => 0,
        }
    }
    pub fn write_redistributor(&mut self, offset: u64, size: u8, value: u64) {
        let frame = offset / FRAME;
        let at = offset % FRAME;
        if frame == 0 {
            if at == GICR_WAKER {
                self.processor_sleep = value as u32 & WAKER_PROCESSOR_SLEEP != 0;
            }
            return;
        }
        if frame != 1 {
            return;
        }
        let bits = value as u32;
        match at {
            IGROUPR0 => self.group = bits,
            ISENABLER0 => self.enabled |= bits,
            ICENABLER0 => self.enabled &= !bits,
            ISPENDR0 => self.pending |= bits,
            ICPENDR0 => self.pending &= !bits,
            IPRIORITYR..=0x41f => {
                let start = (at - IPRIORITYR) as usize;
                for i in 0..usize::from(size).min(4).min(32 - start) {
                    self.priority[start + i] = (value >> (8 * i)) as u8;
                }
            }
            _ => {}
        }
    }

    /// The priority the virtual timer's interrupt is forwarded with, or `None` if
    /// the controller would not deliver it as a FIQ: the redistributor must be
    /// awake, group 0 enabled at the distributor, and PPI 27 enabled in group 0
    /// (a group 1 interrupt is an IRQ, not a FIQ).
    pub fn vtimer_fiq_priority(&self) -> Option<u8> {
        let bit = 1 << VIRTUAL_TIMER;
        let forwarded = !self.processor_sleep
            && self.distributor_ctlr & GICD_CTLR_ENABLE_GRP0 != 0
            && self.enabled & bit != 0
            && self.group & bit == 0;
        forwarded.then(|| self.priority[VIRTUAL_TIMER as usize])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sequence of `pe_init_fiq`: find this PE's redistributor, wake it,
    /// make the timer a group 0 interrupt, enable it and the distributor's group 0.
    fn init_like_xnu(gic: &mut Gicv3) {
        let typer = gic.read_redistributor(GICR_TYPER, 8);
        assert_eq!(typer >> 32 & 0xffff, 0, "affinity 0.0 is this PE");
        assert_ne!(typer & TYPER_LAST, 0, "the scan stops at the first frame");
        let waker = gic.read_redistributor(GICR_WAKER, 4) as u32;
        assert_ne!(waker & WAKER_CHILDREN_ASLEEP, 0);
        gic.write_redistributor(GICR_WAKER, 4, u64::from(waker & !WAKER_PROCESSOR_SLEEP));
        assert_eq!(
            gic.read_redistributor(GICR_WAKER, 4) as u32 & WAKER_CHILDREN_ASLEEP,
            0
        );
        gic.write_redistributor(FRAME + IGROUPR0, 4, 0x81ff_ffff);
        gic.write_redistributor(FRAME + ISENABLER0, 4, 1 << 27);
        let ctlr = gic.read_distributor(GICD_CTLR, 4) as u32;
        gic.write_distributor(GICD_CTLR, 4, u64::from(ctlr | GICD_CTLR_ENABLE_GRP0));
    }

    #[test]
    fn the_xnu_init_sequence_forwards_the_virtual_timer_as_a_fiq() {
        let mut gic = Gicv3::default();
        assert_eq!(gic.vtimer_fiq_priority(), None);
        init_like_xnu(&mut gic);
        assert_eq!(gic.vtimer_fiq_priority(), Some(0));
    }

    #[test]
    fn each_gate_on_the_path_to_the_cpu_blocks_the_timer() {
        let ready = |edit: &dyn Fn(&mut Gicv3)| {
            let mut gic = Gicv3::default();
            init_like_xnu(&mut gic);
            edit(&mut gic);
            gic.vtimer_fiq_priority()
        };
        // The redistributor asleep again.
        assert_eq!(
            ready(&|g| g.write_redistributor(GICR_WAKER, 4, u64::from(WAKER_PROCESSOR_SLEEP))),
            None
        );
        // Distributor group 0 disabled.
        assert_eq!(ready(&|g| g.write_distributor(GICD_CTLR, 4, 0)), None);
        // PPI 27 disabled (ICENABLER clears it).
        assert_eq!(
            ready(&|g| g.write_redistributor(FRAME + ICENABLER0, 4, 1 << 27)),
            None
        );
        // PPI 27 in group 1: an IRQ, never a FIQ.
        assert_eq!(
            ready(&|g| g.write_redistributor(FRAME + IGROUPR0, 4, 1 << 27)),
            None
        );
        // The priority the guest set is reported for the CPU interface's mask check.
        assert_eq!(
            ready(&|g| g.write_redistributor(FRAME + IPRIORITYR + 27, 1, 0x80)),
            Some(0x80)
        );
    }

    #[test]
    fn typer_is_readable_as_two_words_and_other_frames_are_reserved() {
        let gic = Gicv3::default();
        assert_eq!(gic.read_redistributor(GICR_TYPER, 4), TYPER_LAST);
        assert_eq!(gic.read_redistributor(GICR_TYPER + 4, 4), 0);
        // PE 1 does not exist: the second PE's frames read as zero.
        assert_eq!(gic.read_redistributor(2 * FRAME + GICR_TYPER, 8), 0);
        assert_eq!(gic.read_distributor(PIDR2, 4) & 0xf0, 0x30, "GICv3");
    }
}
