use volt_target::aarch64::decode::{AtomicKind, SignExtend};
use volt_target::aarch64::simd_struct::StructDesc;

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
    /// Context synchronization (`ISB`): page-table writes become visible to the walker.
    Isb,
    /// TLB invalidation (`TLBI`).
    Tlbi,
    Translation,
    DcZva,
    Eret,
    Undefined,
    /// LSE atomic (`ldadd`, `cas`, `swp`, ...); see `Cpu::atomic_kind`.
    Atomic,
    /// SIMD structure load/store (`ld2`, `st4`, `ld1r`, ...); see `Cpu::structure`.
    SimdStruct,
    /// `at s1e*`: translate `Cpu::address` (`value` bit 0 = write, bit 1 = as EL0)
    /// into `PAR_EL1`.
    AddressTranslate,
    /// An instruction the host runs: `Cpu::value` holds its word (see `host::run`).
    Host,
}

/// Number of data-TLB entries. The JIT indexes with `(va >> 12) & (DTLB_ENTRIES - 1)`.
pub const DTLB_ENTRIES: usize = 256;
/// Tag that never matches a lookup: real tags are page aligned, and a lookup
/// value is `va & (!0xfff | (size - 1))`, which is never `u64::MAX`.
pub const DTLB_INVALID: u64 = u64::MAX;

/// One inline data-TLB entry, read directly by generated code (hence `repr(C)`
/// and the 32-byte stride, which turns indexing into a shift).
///
/// A tag is the guest virtual page base if that page may be read (or written)
/// through `addend`, else `DTLB_INVALID`. The host address of a byte is
/// `va.wrapping_add(addend)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct DtlbEntry {
    pub read: u64,
    pub write: u64,
    pub addend: u64,
    pub _pad: u64,
}
impl Default for DtlbEntry {
    fn default() -> Self {
        Self {
            read: DTLB_INVALID,
            write: DTLB_INVALID,
            addend: 0,
            _pad: 0,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Dtlb {
    pub entries: [DtlbEntry; DTLB_ENTRIES],
}
impl Default for Dtlb {
    fn default() -> Self {
        Self {
            entries: [DtlbEntry::default(); DTLB_ENTRIES],
        }
    }
}
impl Dtlb {
    pub fn index(address: u64) -> usize {
        ((address >> 12) as usize) & (DTLB_ENTRIES - 1)
    }
    pub fn flush(&mut self) {
        self.entries = [DtlbEntry::default(); DTLB_ENTRIES];
    }
    /// Drop write permission everywhere, keeping read entries.
    pub fn purge_writes(&mut self) {
        for entry in &mut self.entries {
            entry.write = DTLB_INVALID;
        }
    }
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
    pub vector_dest: bool,
    /// LSE atomic in flight (`Trap::Atomic`): the operation, with `address`,
    /// `width` and `dest` as for a load. `atomic_operands` is `[Rs]` for the memory
    /// ops, `[Rs, Rt]` for CAS and `[Rs, Rs+1, Rt, Rt+1]` for CASP.
    pub atomic_kind: AtomicKind,
    pub atomic_operands: [u64; 4],
    /// Registers that receive the words read (31 discards): `[Rt, Rt2]` for `LDXP`,
    /// `[Rs, Rs+1]` for `CASP`, `[Rt, _]` for the memory operations.
    pub atomic_dests: [u8; 2],
    /// SIMD structure access in flight (`Trap::SimdStruct`), with `address` as its base.
    pub structure: StructDesc,

    pub exclusive: Exclusive,
    pub status_dest: u8,
    pub monitor_valid: bool,
    pub monitor_address: u64,
    pub monitor_width: u8,
    /// Value the last exclusive load observed. A store-exclusive succeeds only
    /// if memory still holds it (checked with a host compare-exchange), which
    /// stays correct with other vCPUs and inline stores that never touch a
    /// shared monitor.
    pub monitor_value: u64,
    /// The second word of an exclusive pair (`LDXP`); `monitor_value` is the first.
    pub monitor_value2: u64,
    pub event_set: bool,
    pub second_pending: bool,
    pub second_address: u64,
    pub second_value: u64,

    pub second_width: u8,
    pub second_dest: u8,
    pub second_signed: SignExtend,
    pub second_vector: bool,
    pub writeback: bool,
    pub writeback_value: u64,
    pub writeback_dest: u8,
    pub system: System,
    pub fault_address: u64,
    pub fault_status: u64,
    pub v: [[u64; 2]; 32],
    /// Inline data TLB for generated loads and stores. Filled by the machine.
    pub dtlb: Dtlb,
    /// Apple IMP-DEF registers of the `op1` 0 to 6 banks the SPTM monitor configures. It
    /// is the last field so growing it never moves the fields generated code reaches
    /// with short immediate offsets (`v`, `dtlb`).
    pub apple_bank: ImpDefBank,
}

/// `S3_<op1>_c15` registers, indexed by `volt_target::aarch64::decode::apple_bank_slot`.
#[derive(Clone, Debug)]
pub struct ImpDefBank(pub [u64; volt_target::aarch64::decode::APPLE_BANK_SLOTS]);

impl Default for ImpDefBank {
    fn default() -> Self {
        Self([0; volt_target::aarch64::decode::APPLE_BANK_SLOTS])
    }
}

/// `HCR_EL2.TGE` (bit 27): exceptions taken from EL0 route to EL2.
pub const HCR_TGE: u64 = 1 << 27;
/// `HCR_EL2.E2H` (bit 34): VHE. EL1 register names accessed from EL2 reach the EL2 banks.
pub const HCR_E2H: u64 = 1 << 34;

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
    pub par_el1: u64,
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
    pub contextidr_el1: u64,
    pub csselr_el1: u64,
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
    /// PSTATE.PAN (0 or 1). Saved in `SPSR_EL1` bit 22 and set on exception entry
    /// when `SCTLR_EL1.SPAN` is clear; not enforced by translation.
    pub pan: u64,
    pub isr_el1: u64,
    pub icc_sre_el1: u64,
    pub icc_pmr_el1: u64,
    pub icc_bpr0_el1: u64,
    pub icc_ctlr_el1: u64,
    pub icc_igrpen0_el1: u64,
    pub icc_igrpen1_el1: u64,
    pub icc_bpr1_el1: u64,
    pub cntp_ctl_el0: u64,
    pub cntp_cval_el0: u64,
    pub actlr_el1: u64,
    /// EL2 banks (SPTM runs at EL2, GL2). `SP_EL2` is `sp_el[2]`.
    pub sctlr_el2: u64,
    pub tcr_el2: u64,
    pub ttbr0_el2: u64,
    pub ttbr1_el2: u64,
    pub mair_el2: u64,
    pub vbar_el2: u64,
    pub elr_el2: u64,
    pub spsr_el2: u64,
    pub esr_el2: u64,
    pub far_el2: u64,
    pub hcr_el2: u64,
    pub tpidr_el2: u64,
    pub cntvoff_el2: u64,
    /// Apple IMP-DEF configuration registers, indexed by `SystemRegister::Apple`.
    pub apple: [u64; volt_target::aarch64::decode::APPLE_SLOTS],
    /// `MIDR_EL1`, `ID_AA64PFR0_EL1` and `ID_AA64MMFR0_EL1`: the identity this CPU
    /// advertises, set by the board.
    pub midr_el1: u64,
    pub id_aa64pfr0_el1: u64,
    pub id_aa64mmfr0_el1: u64,
    /// PAuth keys, low word first: APIA, APIB, APDA, APDB, APGA.
    pub pac_keys: [u64; 10],
    /// `ID_AA64ISAR1_EL1`: which pointer authentication algorithms the CPU advertises.
    pub id_aa64isar1_el1: u64,
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
            par_el1: 0,
            vbar_el1: 0,
            cpacr_el1: 0,
            sctlr_el1: 0,
            ttbr0_el1: 0,
            ttbr1_el1: 0,
            tcr_el1: 0,
            sctlr_el2: 0,
            tcr_el2: 0,
            ttbr0_el2: 0,
            ttbr1_el2: 0,
            mair_el2: 0,
            vbar_el2: 0,
            elr_el2: 0,
            spsr_el2: 0,
            esr_el2: 0,
            far_el2: 0,
            hcr_el2: 0,
            tpidr_el2: 0,
            cntvoff_el2: 0,
            mair_el1: 0,
            tpidr_el0: 0,
            tpidrro_el0: 0,
            tpidr_el1: 0,
            contextidr_el1: 0,
            csselr_el1: 0,
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
            pan: 0,
            isr_el1: 0,
            icc_sre_el1: 0,
            icc_pmr_el1: 0,
            icc_bpr0_el1: 0,
            // PRIbits 7: eight priority bits.
            icc_ctlr_el1: 7 << 8,
            icc_igrpen0_el1: 0,
            icc_igrpen1_el1: 0,
            icc_bpr1_el1: 0,
            cntp_ctl_el0: 0,
            cntp_cval_el0: 0,
            actlr_el1: 0,
            apple: [0; volt_target::aarch64::decode::APPLE_SLOTS],
            midr_el1: crate::aarch64::identification::DEFAULT_MIDR_EL1,
            id_aa64pfr0_el1: crate::aarch64::identification::DEFAULT_ID_AA64PFR0_EL1,
            id_aa64mmfr0_el1: crate::aarch64::identification::DEFAULT_ID_AA64MMFR0_EL1,
            pac_keys: [0; 10],
            id_aa64isar1_el1: crate::aarch64::identification::DEFAULT_ID_AA64ISAR1_EL1,
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

impl Cpu {
    /// Complete `LDXP`/`STXP` against the `memory` pair just read at `address`, and
    /// return the pair to leave in memory. The monitor is value-based like the
    /// single-register exclusives: `STXP` succeeds if the monitor is held on the same
    /// address and the pair still has the values `LDXP` saw. The status goes to
    /// `status_dest`.
    pub fn exclusive_pair(&mut self, kind: AtomicKind, memory: [u64; 2]) -> [u64; 2] {
        if kind == AtomicKind::LoadPair {
            self.monitor_valid = true;
            self.monitor_address = self.address;
            self.monitor_width = self.width;
            self.monitor_value = memory[0];
            self.monitor_value2 = memory[1];
            return memory;
        }
        let held = self.monitor_valid
            && self.monitor_address == self.address
            && self.monitor_width == self.width
            && [self.monitor_value, self.monitor_value2] == memory;
        self.monitor_valid = false;
        if self.status_dest != 31 {
            self.x[usize::from(self.status_dest)] = u64::from(!held);
        }
        if held {
            [self.atomic_operands[0], self.atomic_operands[1]]
        } else {
            memory
        }
    }
    /// Reads the byte-sized memory-request fields JIT code stores one byte at a
    /// time. Plain reads of these adjacent bytes get merged into a single wide
    /// load, which cannot be store-forwarded from several narrow stores and
    /// stalls the pipeline on every guest load and store. Volatile keeps each
    /// read a byte load that matches its store.
    #[inline(always)]
    pub fn memory_request(&self) -> (u8, u8, SignExtend, Exclusive) {
        // SAFETY: each pointer is derived from a live field reference.
        unsafe {
            (
                core::ptr::read_volatile(&self.width),
                core::ptr::read_volatile(&self.dest),
                core::ptr::read_volatile(&self.load_signed),
                core::ptr::read_volatile(&self.exclusive),
            )
        }
    }
    /// The second access of a pair, with the same byte-load rationale as
    /// `memory_request`.
    #[inline(always)]
    pub fn second_memory_request(&self) -> (u8, u8, SignExtend) {
        // SAFETY: each pointer is derived from a live field reference.
        unsafe {
            (
                core::ptr::read_volatile(&self.second_width),
                core::ptr::read_volatile(&self.second_dest),
                core::ptr::read_volatile(&self.second_signed),
            )
        }
    }
}
