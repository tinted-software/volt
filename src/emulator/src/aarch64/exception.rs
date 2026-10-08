use super::cpu::Cpu;

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
}
impl Default for Reason {
    fn default() -> Self {
        Self {
            kind: Kind::Sync,
            ec: 0,
            instruction: false,
            status: status::TRANSLATION,
        }
    }
}
pub mod status {
    pub const TRANSLATION: u8 = 0b000100;
    pub const PERMISSION: u8 = 0b001100;
    pub const PERMISSION_FETCH: u8 = 0b001101;
}
pub fn take(cpu: &mut Cpu, reason: Reason, fault_address: u64) -> u64 {
    let from = cpu.system.el;
    cpu.system.spsr_el1 = (u64::from(cpu.flags) & 0xf000_0000)
        | (cpu.system.pan << 22)
        | ((cpu.system.daif & 15) << 6)
        | (u64::from(from & 3) << 2)
        | u64::from(cpu.system.spsel);
    cpu.system.elr_el1 = cpu.pc;
    cpu.system.esr_el1 = (u64::from(reason.ec & 63) << 26)
        | (u64::from(reason.instruction) << 25)
        | u64::from(reason.status & 63);
    cpu.system.far_el1 = fault_address;
    let group = if from != 1 {
        2
    } else if cpu.system.spsel {
        1
    } else {
        0
    };
    let vector = cpu
        .system
        .vbar_el1
        .wrapping_add(group * 0x200)
        .wrapping_add(reason.kind as u64 * 0x80);
    let live_stack = if from == 1 && !cpu.system.spsel {
        0
    } else {
        from & 3
    };
    cpu.system.sp_el[live_stack as usize] = cpu.sp;
    cpu.system.el = 1;
    cpu.system.spsel = true;
    cpu.sp = cpu.system.sp_el[1];
    cpu.system.daif = 15;
    // SCTLR_EL1.SPAN clear: entry to EL1 sets PAN.
    if cpu.system.sctlr_el1 & (1 << 23) == 0 {
        cpu.system.pan = 1;
    }
    cpu.monitor_valid = false;
    cpu.fault_address = fault_address;
    cpu.pc = vector;
    cpu.pc
}
pub fn eret(cpu: &mut Cpu) -> u64 {
    let spsr = cpu.system.spsr_el1;
    let from_stack = if cpu.system.el == 1 && !cpu.system.spsel {
        0
    } else {
        cpu.system.el & 3
    };
    cpu.system.sp_el[from_stack as usize] = cpu.sp;
    cpu.monitor_valid = false;
    let to = ((spsr >> 2) & 3) as u8;
    cpu.system.el = to;
    cpu.system.spsel = to == 1 && spsr & 1 != 0;
    let live_stack = if to == 1 && !cpu.system.spsel { 0 } else { to };
    cpu.sp = cpu.system.sp_el[live_stack as usize];
    cpu.system.daif = (spsr >> 6) & 15;
    cpu.system.pan = (spsr >> 22) & 1;
    cpu.flags = (spsr & 0xf000_0000) as u32;
    cpu.pc = cpu.system.elr_el1;
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
