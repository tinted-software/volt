use super::decode::SignExtend;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum Exclusive {
    #[default]
    None,
    Load,
    Store,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum Trap {
    #[default]
    None,
    Psci,
    Wfi,
    Wfe,
    Load,
    Store,
    Svc,
    Brk,
    Sync,
    Translation,
    DcZva,
    Eret,
    Undefined,
}

#[derive(Clone, Debug, Default)]
#[repr(C)]
pub struct Cpu {
    pub x: [u64; 31],
    pub sp: u64,
    pub pc: u64,
    pub flags: u32,
    pub trap: Trap,
    pub address: u64,
    pub value: u64,
    pub width: u8,
    pub dest: u8,
    pub load_signed: SignExtend,
    pub exclusive: Exclusive,
    pub status_dest: u8,
    pub monitor_valid: bool,
    pub monitor_address: u64,
    pub monitor_width: u8,
    pub event_set: bool,
    pub second_pending: bool,
    pub second_address: u64,
    pub second_value: u64,
    pub second_width: u8,
    pub second_dest: u8,
    pub second_signed: SignExtend,
    pub writeback: bool,
    pub writeback_value: u64,
    pub writeback_dest: u8,
    pub system: System,
    pub fault_address: u64,
    pub fault_status: u64,
}

#[derive(Clone, Debug)]
#[repr(C)]
pub struct System {
    pub el: u8,
    pub spsel: bool,
    pub sp_el: [u64; 4],
    pub spsr_el1: u64,
    pub elr_el1: u64,
    pub esr_el1: u64,
    pub far_el1: u64,
    pub vbar_el1: u64,
    pub cpacr_el1: u64,
    pub sctlr_el1: u64,
    pub ttbr0_el1: u64,
    pub ttbr1_el1: u64,
    pub tcr_el1: u64,
    pub mair_el1: u64,
    pub tpidr_el0: u64,
    pub tpidrro_el0: u64,
    pub tpidr_el1: u64,
    pub mdscr_el1: u64,
    pub cntkctl_el1: u64,
    pub oslar_el1: u64,
    pub osdlr_el1: u64,
    pub fpcr: u64,
    pub fpsr: u64,
    pub cntvct_el0: u64,
    pub cntv_ctl_el0: u64,
    pub cntv_cval_el0: u64,
    pub daif: u64,
}
impl Default for System {
    fn default() -> Self {
        Self {
            el: 1,
            spsel: true,
            sp_el: [0; 4],
            spsr_el1: 0,
            elr_el1: 0,
            esr_el1: 0,
            far_el1: 0,
            vbar_el1: 0,
            cpacr_el1: 0,
            sctlr_el1: 0,
            ttbr0_el1: 0,
            ttbr1_el1: 0,
            tcr_el1: 0,
            mair_el1: 0,
            tpidr_el0: 0,
            tpidrro_el0: 0,
            tpidr_el1: 0,
            mdscr_el1: 0,
            cntkctl_el1: 0,
            oslar_el1: 1,
            osdlr_el1: 1,
            fpcr: 0,
            fpsr: 0,
            cntvct_el0: 0,
            cntv_ctl_el0: 0,
            cntv_cval_el0: 0,
            daif: 0xf,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reset_state_matches_kernel_entry_contract() {
        let cpu = Cpu::default();
        assert_eq!(cpu.x, [0; 31]);
        assert_eq!(cpu.sp, 0);
        assert_eq!(cpu.pc, 0);
        assert_eq!(cpu.trap, Trap::None);
        assert_eq!(cpu.load_signed, SignExtend::None);
        assert_eq!(cpu.second_signed, SignExtend::None);
        assert_eq!(cpu.exclusive, Exclusive::None);
        assert!(!cpu.monitor_valid);
        assert!(!cpu.event_set);
        assert!(!cpu.second_pending);
        assert!(!cpu.writeback);
        assert_eq!(cpu.system.el, 1);
        assert!(cpu.system.spsel);
        assert_eq!(cpu.system.daif, 15);
        assert_eq!(cpu.system.oslar_el1, 1);
        assert_eq!(cpu.system.osdlr_el1, 1);
        assert_eq!(cpu.system.sp_el, [0; 4]);
        assert_eq!(cpu.system.sctlr_el1, 0);
        assert_eq!(cpu.system.cntvct_el0, 0);
    }
}
