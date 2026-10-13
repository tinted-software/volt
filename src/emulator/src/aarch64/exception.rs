use super::cpu::{Cpu, HCR_TGE};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum Kind {
    #[default]
    Sync = 0,
    Irq = 1,
    Fiq = 2,
    Serror = 3,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reason {
    pub kind: Kind,
    pub ec: u8,
    pub instruction: bool,
    pub status: u8,
    /// Further ISS bits: a BRK's immediate. Syndromes with a DFSC leave this 0.
    pub iss: u32,
}
impl Default for Reason {
    fn default() -> Self {
        Self {
            kind: Kind::Sync,
            ec: 0,
            instruction: false,
            status: status::TRANSLATION,
            iss: 0,
        }
    }
}
pub mod status {
    pub const TRANSLATION: u8 = 0b000100;
    pub const PERMISSION: u8 = 0b001100;
    pub const PERMISSION_FETCH: u8 = 0b001101;
    /// Synchronous external abort: the access reached no memory or device.
    pub const SYNC_EXTERNAL: u8 = 0b010000;
}
pub fn take(cpu: &mut Cpu, reason: Reason, fault_address: u64) -> u64 {
    let from = cpu.system.el;
    let to = if from == 2 || (from == 0 && cpu.system.hcr_el2 & HCR_TGE != 0) {
        2
    } else {
        1
    };
    let spsr = (u64::from(cpu.flags) & 0xf000_0000)
        | (cpu.system.pan << 22)
        | ((cpu.system.daif & 15) << 6)
        | (u64::from(from & 3) << 2)
        | u64::from(cpu.system.spsel);
    let esr = (u64::from(reason.ec & 63) << 26)
        | (u64::from(reason.instruction) << 25)
        | u64::from(reason.status & 63)
        | u64::from(reason.iss & 0x1ff_ffff);
    if to == 2 {
        cpu.system.spsr_el2 = spsr;
        cpu.system.elr_el2 = cpu.pc;
        cpu.system.esr_el2 = esr;
        cpu.system.far_el2 = fault_address;
    } else {
        cpu.system.spsr_el1 = spsr;
        cpu.system.elr_el1 = cpu.pc;
        cpu.system.esr_el1 = esr;
        cpu.system.far_el1 = fault_address;
    }
    // Same-EL entries pick the vector group by SPSel (SP_EL0 or SP_ELx); entries from a
    // lower EL use the lower-EL groups.
    let group = if from != to {
        2
    } else if cpu.system.spsel {
        1
    } else {
        0
    };
    let vbar = if to == 2 {
        cpu.system.vbar_el2
    } else {
        cpu.system.vbar_el1
    };
    let vector = vbar
        .wrapping_add(group * 0x200)
        .wrapping_add(reason.kind as u64 * 0x80);
    // The stack live at the exception is banked: SP_ELx when SPSel was set, else SP_EL0.
    let live_stack = if cpu.system.spsel { from & 3 } else { 0 };
    cpu.system.sp_el[live_stack as usize] = cpu.sp;
    cpu.system.el = to;
    cpu.system.spsel = true;
    cpu.sp = cpu.system.sp_el[to as usize];
    cpu.system.daif = 15;
    // SCTLR_EL1.SPAN clear: entry to EL1 sets PAN. PAN belongs to EL1 only.
    if to == 1 && cpu.system.sctlr_el1 & (1 << 23) == 0 {
        cpu.system.pan = 1;
    }
    cpu.monitor_valid = false;
    cpu.fault_address = fault_address;
    cpu.pc = vector;
    cpu.pc
}
pub fn eret(cpu: &mut Cpu) -> u64 {
    let el = cpu.system.el;
    let (spsr, elr) = if el == 2 {
        (cpu.system.spsr_el2, cpu.system.elr_el2)
    } else {
        (cpu.system.spsr_el1, cpu.system.elr_el1)
    };
    let live_stack = if cpu.system.spsel { el & 3 } else { 0 };
    cpu.system.sp_el[live_stack as usize] = cpu.sp;
    cpu.monitor_valid = false;
    let to = ((spsr >> 2) & 3) as u8;
    cpu.system.el = to;
    cpu.system.spsel = to != 0 && spsr & 1 != 0;
    let restored = if cpu.system.spsel { to } else { 0 };
    cpu.sp = cpu.system.sp_el[restored as usize];
    cpu.system.daif = (spsr >> 6) & 15;
    cpu.system.pan = (spsr >> 22) & 1;
    cpu.flags = (spsr & 0xf000_0000) as u32;
    cpu.pc = elr;
    cpu.pc
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pan_is_set_on_entry_unless_span_and_comes_back_with_eret() {
        let mut cpu = Cpu::default();
        cpu.pc = 0x4000;
        cpu.system.vbar_el1 = 0x10000;
        // SCTLR_EL1.SPAN clear: entry sets PAN, and the SPSR records the old value.
        cpu.system.pan = 0;
        take(&mut cpu, Reason::default(), 0);
        assert_eq!(cpu.system.pan, 1);
        assert_eq!((cpu.system.spsr_el1 >> 22) & 1, 0);
        cpu.system.pan = 1;
        take(&mut cpu, Reason::default(), 0);
        assert_eq!((cpu.system.spsr_el1 >> 22) & 1, 1);
        // SCTLR_EL1.SPAN set: PAN keeps the interrupted value.
        cpu.system.sctlr_el1 = 1 << 23;
        cpu.system.pan = 0;
        take(&mut cpu, Reason::default(), 0);
        assert_eq!(cpu.system.pan, 0);
        // ERET restores PAN from SPSR bit 22.
        for saved in [1u64, 0] {
            cpu.system.spsr_el1 = saved << 22;
            eret(&mut cpu);
            assert_eq!(cpu.system.pan, saved);
        }
    }
    #[test]
    fn el1t_exception_banks_both_stacks_and_restores_status() {
        let mut cpu = Cpu::default();
        cpu.pc = 0x4000;
        cpu.sp = 0x8000;
        cpu.flags = 0xa000_0000;
        cpu.system.spsel = false;
        cpu.system.sp_el[1] = 0x9000;
        cpu.system.vbar_el1 = 0x10000;
        cpu.system.daif = 2;
        cpu.monitor_valid = true;
        assert_eq!(
            take(
                &mut cpu,
                Reason {
                    kind: Kind::Irq,
                    ..Reason::default()
                },
                0x1234
            ),
            0x10080
        );
        assert_eq!(cpu.sp, 0x9000);
        assert_eq!(cpu.system.sp_el[0], 0x8000);
        assert_eq!(cpu.system.far_el1, 0x1234);
        assert!(!cpu.monitor_valid);
        cpu.sp = 0x8f00;
        assert_eq!(eret(&mut cpu), 0x4000);
        assert_eq!(cpu.sp, 0x8000);
        assert_eq!(cpu.system.sp_el[1], 0x8f00);
        assert_eq!(cpu.flags, 0xa000_0000);
        assert_eq!(cpu.system.daif, 2);
        assert!(!cpu.system.spsel);
    }
    #[test]
    fn lower_level_abort_uses_lower_aarch64_vector() {
        let mut cpu = Cpu::default();
        cpu.system.el = 0;
        cpu.system.spsel = false;
        cpu.sp = 0x2000;
        cpu.system.sp_el[1] = 0x3000;
        cpu.system.vbar_el1 = 0x10000;
        let reason = Reason {
            ec: 0x20,
            instruction: true,
            status: status::PERMISSION_FETCH,
            ..Reason::default()
        };
        assert_eq!(take(&mut cpu, reason, 0x4444), 0x10400);
        assert_eq!(cpu.system.esr_el1, (0x20 << 26) | (1 << 25) | 13);
        eret(&mut cpu);
        assert_eq!(cpu.system.el, 0);
        assert_eq!(cpu.sp, 0x2000);
    }
    #[test]
    fn exception_from_el2_stays_in_el2_and_banks_only_el2_state() {
        let mut cpu = Cpu::default();
        cpu.system.el = 2;
        cpu.system.spsel = true;
        cpu.sp = 0x7000;
        cpu.pc = 0x1234;
        cpu.system.vbar_el2 = 0x20000;
        cpu.system.vbar_el1 = 0x10000;
        cpu.system.elr_el1 = 0xaaaa;
        cpu.system.spsr_el1 = 0xbbbb;
        // Same EL with SPSel set: the SP_EL2 group, synchronous.
        assert_eq!(take(&mut cpu, Reason::default(), 0x55), 0x20200);
        assert_eq!(cpu.system.el, 2);
        assert_eq!(cpu.system.elr_el2, 0x1234);
        assert_eq!(cpu.system.far_el2, 0x55);
        assert_eq!((cpu.system.spsr_el2 >> 2) & 3, 2);
        assert_eq!(cpu.system.elr_el1, 0xaaaa);
        assert_eq!(cpu.system.spsr_el1, 0xbbbb);
        assert_eq!(cpu.sp, 0x7000);
    }
    #[test]
    fn eret_from_el2_returns_to_el1h_on_its_stack() {
        let mut cpu = Cpu::default();
        cpu.system.el = 2;
        cpu.system.spsel = true;
        cpu.sp = 0x7000;
        cpu.system.sp_el[1] = 0x9000;
        cpu.system.spsr_el2 = (1 << 2) | 1;
        cpu.system.elr_el2 = 0x4444;
        assert_eq!(eret(&mut cpu), 0x4444);
        assert_eq!(cpu.system.el, 1);
        assert!(cpu.system.spsel);
        assert_eq!(cpu.sp, 0x9000);
        assert_eq!(cpu.system.sp_el[2], 0x7000);
    }
    #[test]
    fn el0_svc_routes_to_el2_when_hcr_el2_tge_is_set() {
        let mut cpu = Cpu::default();
        cpu.system.el = 0;
        cpu.system.spsel = false;
        cpu.system.hcr_el2 = HCR_TGE;
        cpu.pc = 0x1000;
        cpu.system.vbar_el2 = 0x20000;
        cpu.system.vbar_el1 = 0x10000;
        let reason = Reason {
            ec: 0x15,
            ..Reason::default()
        };
        // Lower EL, AArch64, synchronous.
        assert_eq!(take(&mut cpu, reason, 0), 0x20400);
        assert_eq!(cpu.system.el, 2);
        assert_eq!(cpu.system.elr_el2, 0x1000);
        assert_eq!(cpu.system.elr_el1, 0);
        assert_eq!(cpu.system.esr_el2 >> 26, 0x15);
    }
    #[test]
    fn el0_svc_still_routes_to_el1_without_tge() {
        let mut cpu = Cpu::default();
        cpu.system.el = 0;
        cpu.system.spsel = false;
        cpu.system.vbar_el2 = 0x20000;
        cpu.system.vbar_el1 = 0x10000;
        let reason = Reason {
            ec: 0x15,
            ..Reason::default()
        };
        assert_eq!(take(&mut cpu, reason, 0), 0x10400);
        assert_eq!(cpu.system.el, 1);
        assert_eq!(cpu.system.elr_el2, 0);
    }
}

#[cfg(test)]
mod return_boundaries {
    use super::*;
    #[test]
    fn eret_from_el1t_banks_sp_el0_without_overwriting_kernel_stack() {
        let mut cpu = Cpu::default();
        cpu.sp = 0x8000;
        cpu.system.spsel = false;
        cpu.system.sp_el[1] = 0x9000;
        cpu.system.spsr_el1 = 0;
        cpu.system.elr_el1 = 0x1000;
        assert_eq!(eret(&mut cpu), 0x1000);
        assert_eq!(cpu.system.sp_el[0], 0x8000);
        assert_eq!(cpu.system.sp_el[1], 0x9000);
        assert_eq!(cpu.sp, 0x8000);
        take(&mut cpu, Reason::default(), 0);
        assert_eq!(cpu.sp, 0x9000);
    }
}
