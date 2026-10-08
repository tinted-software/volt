//! Apple Interrupt Controller, AIC v1: the interrupt controller of the t8103
//! (M1) SoC, as the Asahi Linux driver drives it on the `apple,t8103-aic` path.
//!
//! The guest sees 896 level-triggered hardware IRQ lines. Each IRQ has a mask
//! bit, a software trigger ORed with its line, and a target CPU. Each CPU has
//! one event register. Reading it returns the lowest-numbered IRQ that is
//! pending, unmasked and targeted at that CPU, and the read itself masks that
//! IRQ (auto-ack). The driver unmasks it again in its EOI handler. Because the
//! IRQ stays masked until then, a level line that is still high does not
//! re-enter the handler.
//!
//! IPIs on this SoC go through the fast-IPI system registers, which are not
//! part of this device. The driver still writes the MMIO IPI registers during
//! CPU bring-up (ACK, and MASK_SET for the SELF and OTHER bits). Those writes
//! are accepted: the mask registers store state and read it back. SEND, ACK,
//! and the per-CPU SET/CLR triggers are write-only no-ops, since no MMIO IPI is
//! ever sent on this path.
//!
//! Reset state: every IRQ is masked and targeted at CPU 0. The driver unmasks
//! an IRQ when it is requested. A guest never sees an IRQ that has not been
//! explicitly enabled.

/// Mapping length of the AIC register window (the DT `reg` length).
pub const LEN: u64 = 0x8000;

/// Hardware IRQ lines the guest sees. `AIC_INFO` reports this value, and the
/// driver's header comment documents 896 level-triggered hardware IRQs. Any IRQ
/// number at or above this is not a real pin: it has no effect on delivery, and
/// its register slots read as zero.
const NR_IRQ: usize = 896;
/// 32-bit words that cover `NR_IRQ` in each per-IRQ bitmap.
const IRQ_WORDS: usize = NR_IRQ / 32;

const INFO: u64 = 0x0004;
const WHOAMI: u64 = 0x2000;
const EVENT: u64 = 0x2004;
const IPI_SEND: u64 = 0x2008;
const IPI_ACK: u64 = 0x200c;
const IPI_MASK_SET: u64 = 0x2024;
const IPI_MASK_CLR: u64 = 0x2028;

/// `TARGET_CPU`: one word per IRQ at `0x3000 + 4 * irq`. The word holds the CPU
/// bit(s) that receive the IRQ. The driver writes `BIT(cpu)`.
const TARGET_CPU: u64 = 0x3000;

/// Per-IRQ bitmap register blocks. The driver lays them out in this order, and
/// each block is 32 words (128 bytes) long, one bit per IRQ.
const SW_SET: u64 = 0x4000;
#[cfg(test)]
const SW_CLR: u64 = 0x4080;
#[cfg(test)]
const MASK_SET: u64 = 0x4100;
#[cfg(test)]
const MASK_CLR: u64 = 0x4180;
const HW_STATE: u64 = 0x4200;
const BITMAP_STRIDE: u64 = 0x80;

/// Explicit per-CPU IPI views: `0x5000 + cpu * 0x80`, with the registers at
/// fixed offsets within each 128-byte block.
const CPU_IPI_BASE: u64 = 0x5000;
const CPU_IPI_STRIDE: u64 = 0x80;
const CPU_IPI_SET: u64 = 0x08;
const CPU_IPI_CLR: u64 = 0x0c;
const CPU_IPI_MASK_SET: u64 = 0x24;
const CPU_IPI_MASK_CLR: u64 = 0x28;

/// Event word for an IRQ: type `IRQ` (1) in bits 23:16, IRQ number in 15:0.
const EVENT_TYPE_IRQ: u32 = 1 << 16;

/// Which per-IRQ bitmap a register offset selects.
#[derive(Clone, Copy)]
enum Bitmap {
    SwSet,
    SwClr,
    MaskSet,
    MaskClr,
    HwState,
}

/// MMIO IPI registers, both the default view and the explicit per-CPU view.
#[derive(Clone, Copy)]
enum IpiReg {
    Send,
    Ack,
    Set,
    Clr,
    MaskSet,
    MaskClr,
}

/// A decoded, word-aligned register offset.
#[derive(Clone, Copy)]
enum Reg {
    Info,
    WhoAmI,
    Event,
    Target(usize),
    Bitmap(Bitmap, usize),
    Ipi { cpu: u32, reg: IpiReg },
    Reserved,
}

/// Apple Interrupt Controller (AIC v1) state for `cpus` CPUs.
pub struct Aic {
    cpus: u32,
    /// Hardware line state, one bit per IRQ.
    hw: [u32; IRQ_WORDS],
    /// Software trigger state (`SW_SET`/`SW_CLR`), one bit per IRQ.
    sw: [u32; IRQ_WORDS],
    /// Mask state (`MASK_SET`/`MASK_CLR`), one bit per IRQ. A set bit masks.
    mask: [u32; IRQ_WORDS],
    /// CPU routing per IRQ (`TARGET_CPU`). Only the first `NR_IRQ` entries are
    /// meaningful.
    target: [u32; NR_IRQ],
    /// Per-CPU IPI mask state, indexed by CPU. Only the first `cpus` entries
    /// are reachable.
    ipi_mask: [u32; 32],
}

/// Mask of the low `n` bytes (`n` in 1..=4) of a 32-bit word.
fn byte_mask(n: u32) -> u32 {
    if n >= 4 {
        u32::MAX
    } else {
        (1u32 << (n * 8)) - 1
    }
}

impl Aic {
    /// `cpus` = number of CPUs wired to the IRQ pins (affects per-CPU IPI registers and target masks).
    ///
    /// Panics if `cpus` is outside `1..=32`, since target registers are 32-bit CPU masks.
    pub fn new(cpus: u32) -> Self {
        assert!((1..=32).contains(&cpus), "AIC routes to 1..=32 CPUs");
        Self {
            cpus,
            hw: [0; IRQ_WORDS],
            sw: [0; IRQ_WORDS],
            mask: [u32::MAX; IRQ_WORDS],
            target: [1; NR_IRQ],
            ipi_mask: [u32::MAX; 32],
        }
    }

    /// Drive hardware IRQ `irq` (level-triggered source line).
    ///
    /// IRQ numbers at or above `NR_IRQ` have no pin and are ignored.
    pub fn set_irq(&mut self, irq: u32, asserted: bool) {
        let irq = irq as usize;
        if irq >= NR_IRQ {
            return;
        }
        let (w, bit) = (irq / 32, 1u32 << (irq % 32));
        if asserted {
            self.hw[w] |= bit;
        } else {
            self.hw[w] &= !bit;
        }
    }

    /// Whether the IRQ pin of CPU `cpu` is asserted right now (an enabled, unmasked, targeted source is pending).
    pub fn irq_pending(&self, cpu: u32) -> bool {
        self.first_pending(cpu).is_some()
    }

    /// MMIO read at `offset` from the AIC base, performed by CPU `cpu` (AIC_WHOAMI and AIC_EVENT are per reading CPU). Takes &mut self because reading AIC_EVENT has side effects.
    ///
    /// `size` is 1, 2, 4 or 8 bytes. Accesses that cross a 32-bit register are
    /// split per register. Unmapped offsets read as zero.
    pub fn read(&mut self, cpu: u32, offset: u64, size: u8) -> u64 {
        let size = u64::from(size.min(8));
        let mut value = 0u64;
        let mut done = 0u64;
        while done < size {
            let addr = offset.wrapping_add(done);
            let lane = (addr & 3) as u32;
            let n = (4 - u64::from(lane)).min(size - done) as u32;
            let word = self.read_word(cpu, addr & !3);
            let bytes = u64::from((word >> (lane * 8)) & byte_mask(n));
            value |= bytes << (done * 8);
            done += u64::from(n);
        }
        value
    }

    /// MMIO write. Only the low `size` bytes of `value` are written. Writes to
    /// set/clear registers only affect the bits they name. Writes to read-only
    /// or unmapped offsets are ignored.
    pub fn write(&mut self, cpu: u32, offset: u64, size: u8, value: u64) {
        let size = u64::from(size.min(8));
        let mut done = 0u64;
        while done < size {
            let addr = offset.wrapping_add(done);
            let lane = (addr & 3) as u32;
            let n = (4 - u64::from(lane)).min(size - done) as u32;
            let lanes = byte_mask(n) << (lane * 8);
            let bits = (((value >> (done * 8)) as u32) << (lane * 8)) & lanes;
            self.write_word(cpu, addr & !3, bits, lanes);
            done += u64::from(n);
        }
    }

    /// Lowest IRQ that is pending, unmasked, and targeted at `cpu`. A CPU index
    /// of 32 or more has no routing bit, so it never matches.
    fn first_pending(&self, cpu: u32) -> Option<usize> {
        let bit = 1u32.checked_shl(cpu).unwrap_or(0);
        for w in 0..IRQ_WORDS {
            let mut candidates = (self.hw[w] | self.sw[w]) & !self.mask[w];
            while candidates != 0 {
                let irq = w * 32 + candidates.trailing_zeros() as usize;
                if self.target[irq] & bit != 0 {
                    return Some(irq);
                }
                candidates &= candidates - 1;
            }
        }
        None
    }

    /// Reading AIC_EVENT: deliver the lowest pending IRQ for `cpu` and mask it
    /// (auto-ack). Returns 0 when nothing is pending.
    fn claim(&mut self, cpu: u32) -> u32 {
        let Some(irq) = self.first_pending(cpu) else {
            return 0;
        };
        self.mask[irq / 32] |= 1 << (irq % 32);
        EVENT_TYPE_IRQ | irq as u32
    }

    fn own_ipi(&self, cpu: u32, reg: IpiReg) -> Reg {
        if cpu < self.cpus {
            Reg::Ipi { cpu, reg }
        } else {
            Reg::Reserved
        }
    }

    fn decode(&self, offset: u64, cpu: u32) -> Reg {
        match offset {
            INFO => Reg::Info,
            WHOAMI => Reg::WhoAmI,
            EVENT => Reg::Event,
            IPI_SEND => self.own_ipi(cpu, IpiReg::Send),
            IPI_ACK => self.own_ipi(cpu, IpiReg::Ack),
            IPI_MASK_SET => self.own_ipi(cpu, IpiReg::MaskSet),
            IPI_MASK_CLR => self.own_ipi(cpu, IpiReg::MaskClr),
            o if (TARGET_CPU..SW_SET).contains(&o) => {
                let irq = ((o - TARGET_CPU) / 4) as usize;
                if irq < NR_IRQ {
                    Reg::Target(irq)
                } else {
                    Reg::Reserved
                }
            }
            o if (SW_SET..HW_STATE + BITMAP_STRIDE).contains(&o) => {
                let block = (o - SW_SET) / BITMAP_STRIDE;
                let word = (((o - SW_SET) % BITMAP_STRIDE) / 4) as usize;
                let which = match block {
                    0 => Bitmap::SwSet,
                    1 => Bitmap::SwClr,
                    2 => Bitmap::MaskSet,
                    3 => Bitmap::MaskClr,
                    _ => Bitmap::HwState,
                };
                if word < IRQ_WORDS {
                    Reg::Bitmap(which, word)
                } else {
                    Reg::Reserved
                }
            }
            o if (CPU_IPI_BASE..CPU_IPI_BASE + CPU_IPI_STRIDE * u64::from(self.cpus))
                .contains(&o) =>
            {
                let rel = o - CPU_IPI_BASE;
                let reg = match rel % CPU_IPI_STRIDE {
                    CPU_IPI_SET => IpiReg::Set,
                    CPU_IPI_CLR => IpiReg::Clr,
                    CPU_IPI_MASK_SET => IpiReg::MaskSet,
                    CPU_IPI_MASK_CLR => IpiReg::MaskClr,
                    _ => return Reg::Reserved,
                };
                Reg::Ipi {
                    cpu: (rel / CPU_IPI_STRIDE) as u32,
                    reg,
                }
            }
            _ => Reg::Reserved,
        }
    }

    /// Read the 32-bit register at word-aligned `offset`.
    fn read_word(&mut self, cpu: u32, offset: u64) -> u32 {
        match self.decode(offset, cpu) {
            Reg::Info => NR_IRQ as u32,
            Reg::WhoAmI => cpu,
            Reg::Event => self.claim(cpu),
            Reg::Target(irq) => self.target[irq],
            Reg::Bitmap(Bitmap::SwSet | Bitmap::SwClr, w) => self.sw[w],
            Reg::Bitmap(Bitmap::MaskSet | Bitmap::MaskClr, w) => self.mask[w],
            Reg::Bitmap(Bitmap::HwState, w) => self.hw[w],
            Reg::Ipi {
                cpu: c,
                reg: IpiReg::MaskSet | IpiReg::MaskClr,
            } => self.ipi_mask[c as usize],
            Reg::Ipi { .. } | Reg::Reserved => 0,
        }
    }

    /// Write the lanes in `lanes` of the 32-bit register at word-aligned
    /// `offset`. `bits` holds the written value already shifted into those
    /// lanes and zero elsewhere.
    fn write_word(&mut self, cpu: u32, offset: u64, bits: u32, lanes: u32) {
        match self.decode(offset, cpu) {
            Reg::Target(irq) => self.target[irq] = (self.target[irq] & !lanes) | bits,
            Reg::Bitmap(Bitmap::SwSet, w) => self.sw[w] |= bits,
            Reg::Bitmap(Bitmap::SwClr, w) => self.sw[w] &= !bits,
            Reg::Bitmap(Bitmap::MaskSet, w) => self.mask[w] |= bits,
            Reg::Bitmap(Bitmap::MaskClr, w) => self.mask[w] &= !bits,
            Reg::Ipi {
                cpu: c,
                reg: IpiReg::MaskSet,
            } => self.ipi_mask[c as usize] |= bits,
            Reg::Ipi {
                cpu: c,
                reg: IpiReg::MaskClr,
            } => self.ipi_mask[c as usize] &= !bits,
            // Read-only registers, write-only IPI triggers, and reserved
            // offsets: accepted and ignored.
            Reg::Info
            | Reg::WhoAmI
            | Reg::Event
            | Reg::Bitmap(Bitmap::HwState, _)
            | Reg::Ipi { .. }
            | Reg::Reserved => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Address of the MASK_CLR bit for `irq`.
    fn mask_clr(irq: u32) -> u64 {
        MASK_CLR + 4 * u64::from(irq / 32)
    }

    /// Clear the mask of `irq` on CPU 0, as the driver does when it enables an IRQ.
    fn unmask(aic: &mut Aic, irq: u32) {
        aic.write(0, mask_clr(irq), 4, 1u64 << (irq % 32));
    }

    fn event(aic: &mut Aic, cpu: u32) -> u64 {
        aic.read(cpu, EVENT, 4)
    }

    #[test]
    fn event_returns_lowest_pending_irq_and_auto_masks() {
        let mut aic = Aic::new(1);
        unmask(&mut aic, 40);
        unmask(&mut aic, 5);
        aic.set_irq(40, true);
        aic.set_irq(5, true);

        assert_eq!(event(&mut aic, 0), u64::from(EVENT_TYPE_IRQ | 5));
        // IRQ 5 was auto-masked by the read, so IRQ 40 is next.
        assert!(aic.irq_pending(0));
        assert_eq!(event(&mut aic, 0), u64::from(EVENT_TYPE_IRQ | 40));
        assert!(!aic.irq_pending(0));
        assert_eq!(event(&mut aic, 0), 0);
    }

    #[test]
    fn masked_irq_does_not_assert_pin_until_unmasked() {
        let mut aic = Aic::new(1);
        aic.set_irq(3, true);
        assert!(!aic.irq_pending(0));
        assert_eq!(event(&mut aic, 0), 0);

        unmask(&mut aic, 3);
        assert!(aic.irq_pending(0));
        assert_eq!(event(&mut aic, 0), u64::from(EVENT_TYPE_IRQ | 3));
    }

    #[test]
    fn reasserting_while_auto_masked_does_not_raise_pin() {
        let mut aic = Aic::new(1);
        unmask(&mut aic, 3);
        aic.set_irq(3, true);
        assert_eq!(event(&mut aic, 0), u64::from(EVENT_TYPE_IRQ | 3));

        // The line drops and rises again while the IRQ is auto-masked.
        aic.set_irq(3, false);
        aic.set_irq(3, true);
        assert!(!aic.irq_pending(0));
        assert_eq!(event(&mut aic, 0), 0);
    }

    #[test]
    fn mask_clr_reenables_and_pin_rises_again() {
        let mut aic = Aic::new(1);
        unmask(&mut aic, 3);
        aic.set_irq(3, true);
        event(&mut aic, 0);
        assert!(!aic.irq_pending(0));

        // The driver's EOI unmasks the level-triggered source, which is still high.
        unmask(&mut aic, 3);
        assert!(aic.irq_pending(0));
        assert_eq!(event(&mut aic, 0), u64::from(EVENT_TYPE_IRQ | 3));
    }

    #[test]
    fn target_register_routes_irq_away_from_reading_cpu() {
        let mut aic = Aic::new(2);
        unmask(&mut aic, 7);
        aic.set_irq(7, true);
        // Reset routes every IRQ to CPU 0.
        assert!(aic.irq_pending(0));

        // The driver writes BIT(1) to route IRQ 7 to CPU 1.
        aic.write(0, TARGET_CPU + 4 * 7, 4, 1 << 1);
        assert!(!aic.irq_pending(0));
        assert_eq!(event(&mut aic, 0), 0);
        assert!(aic.irq_pending(1));
        assert_eq!(event(&mut aic, 1), u64::from(EVENT_TYPE_IRQ | 7));
    }

    #[test]
    fn software_trigger_asserts_without_hardware_line() {
        let mut aic = Aic::new(1);
        unmask(&mut aic, 40);
        // IRQ 40 is bit 8 of the second word of each bitmap block.
        aic.write(0, SW_SET + 4, 4, 1 << 8);
        assert!(aic.irq_pending(0));
        assert_eq!(event(&mut aic, 0), u64::from(EVENT_TYPE_IRQ | 40));

        aic.write(0, SW_SET + 4, 4, 1 << 8);
        aic.write(0, SW_CLR + 4, 4, 1 << 8);
        assert!(!aic.irq_pending(0));
    }

    #[test]
    fn whoami_reports_the_reading_cpu() {
        let mut aic = Aic::new(2);
        assert_eq!(aic.read(0, WHOAMI, 4), 0);
        assert_eq!(aic.read(1, WHOAMI, 4), 1);
    }

    #[test]
    fn unknown_offsets_read_zero_and_ignore_writes() {
        let mut aic = Aic::new(1);
        // Reserved gap, AIC_CONFIG (not used by the driver on this path), a
        // reserved word in the mask block, a target slot past the last IRQ, a
        // bitmap block past HW_STATE, and a per-CPU view for a CPU that is absent.
        let offsets = [
            0x0000,
            0x0010,
            0x2010,
            MASK_SET + 4 * 28,
            TARGET_CPU + 4 * NR_IRQ as u64,
            HW_STATE + BITMAP_STRIDE,
            CPU_IPI_BASE + CPU_IPI_STRIDE,
        ];
        for offset in offsets {
            aic.write(0, offset, 4, u64::from(u32::MAX));
            assert_eq!(aic.read(0, offset, 4), 0, "offset {offset:#x}");
        }
    }
}
