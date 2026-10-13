//! Branches, system instructions and the families that are not data processing:
//! pointer authentication and guarded execution.
//!
//! Owns `TestBranch`, `Call`, `Indirect`, `B`, `BCond`, `Eret`, `System` (`mrs`/`msr`,
//! with every `SystemRegister` name), `DcZva`, `AddressTranslate`, `CacheOp`, `Tlbi`,
//! `Clrex`, `Nop`, `Hint`, `Dsb`, `Dmb`, `Isb`, `Sev`, `Sevl`, `Wfe`, `Wfi`, `Yield`,
//! `Brk`, `Svc`, `Psci`, `Pauth` and `Gxf`.

use super::{
    Context,
    reg::{condition, gpr},
};
use crate::decode::{self, IndirectKind, Instruction, SysOp, SystemRegister, Width};
use crate::gxf;
use crate::pauth::{self, KeyId, Modifier};
use core::fmt::{self, Formatter};

pub(super) fn write(
    f: &mut Formatter<'_>,
    cx: &Context<'_>,
    instruction: &Instruction,
) -> fmt::Result {
    use Instruction::*;
    match *instruction {
        TestBranch(test) => test_branch(f, cx, &test),
        Call(call) => {
            f.write_str(if call.link { "bl " } else { "b " })?;
            cx.target(f, call.target)
        }
        B(offset) => {
            f.write_str("b ")?;
            cx.target(f, offset)
        }
        BCond(branch) => {
            write!(f, "b.{} ", condition(branch.cond))?;
            cx.target(f, branch.offset)
        }
        Indirect(indirect) => match indirect.kind {
            IndirectKind::Jump => {
                f.write_str("br ")?;
                gpr(f, Width::X64, indirect.rn)
            }
            IndirectKind::Call => {
                f.write_str("blr ")?;
                gpr(f, Width::X64, indirect.rn)
            }
            IndirectKind::Return if indirect.rn == 30 => f.write_str("ret"),
            IndirectKind::Return => {
                f.write_str("ret ")?;
                gpr(f, Width::X64, indirect.rn)
            }
        },
        Eret => f.write_str("eret"),
        System(system) => system_register_move(f, &system),
        DcZva(rt) => {
            f.write_str("dc zva, ")?;
            gpr(f, Width::X64, rt)
        }
        AddressTranslate(at) => {
            write!(
                f,
                "at s1e{}{}, ",
                if at.user { '0' } else { '1' },
                if at.write { 'w' } else { 'r' }
            )?;
            gpr(f, Width::X64, at.rt)
        }
        CacheOp(op) => cache_op(f, &op),
        Tlbi(op) => tlbi(f, &op),
        Clrex(option) => option_or_default(f, "clrex", option),
        Nop => f.write_str("nop"),
        Hint(n) => hint(f, n),
        Dsb(option) => match option {
            0 => f.write_str("ssbb"),
            4 => f.write_str("pssbb"),
            _ => barrier(f, "dsb", option),
        },
        Dmb(option) => barrier(f, "dmb", option),
        Isb(option) => option_or_default(f, "isb", option),
        Sev => f.write_str("sev"),
        Sevl => f.write_str("sevl"),
        Wfe => f.write_str("wfe"),
        Wfi => f.write_str("wfi"),
        Yield => f.write_str("yield"),
        Brk(imm) => write!(f, "brk #{imm:#x}"),
        Svc(imm) => write!(f, "svc #{imm:#x}"),
        Psci => f.write_str("hvc #0x0"),
        Pauth(op) => pauth_op(f, op),
        Gxf(op) => match op {
            gxf::Op::Gexit => f.write_str(".inst 0x00201400"),
            gxf::Op::Genter { imm5 } => write!(f, ".inst {:#010x}", 0x0020_1420 | u32::from(imm5)),
        },
        _ => unreachable!("{instruction:?} is not a system instruction"),
    }
}

fn test_branch(f: &mut Formatter<'_>, cx: &Context<'_>, test: &decode::Test) -> fmt::Result {
    match test.bit_index {
        None => f.write_str(if test.negate { "cbnz " } else { "cbz " })?,
        Some(_) => f.write_str(if test.negate { "tbnz " } else { "tbz " })?,
    }
    gpr(f, test.width, test.rt)?;
    f.write_str(", ")?;
    if let Some(bit) = test.bit_index {
        write!(f, "#{bit}, ")?;
    }
    cx.target(f, test.offset)
}

/// `mnemonic` alone for the architectural default option 15, else `mnemonic #imm`.
fn option_or_default(f: &mut Formatter<'_>, mnemonic: &str, option: u8) -> fmt::Result {
    if option == 15 {
        f.write_str(mnemonic)
    } else {
        write!(f, "{mnemonic} #{option:#x}")
    }
}

/// `dsb`/`dmb` with the barrier option named when binutils names it.
fn barrier(f: &mut Formatter<'_>, mnemonic: &str, option: u8) -> fmt::Result {
    let name = match option {
        1 => "oshld",
        2 => "oshst",
        3 => "osh",
        5 => "nshld",
        6 => "nshst",
        7 => "nsh",
        9 => "ishld",
        10 => "ishst",
        11 => "ish",
        13 => "ld",
        14 => "st",
        15 => "sy",
        _ => return write!(f, "{mnemonic} #{option:#04x}"),
    };
    write!(f, "{mnemonic} {name}")
}

fn hint(f: &mut Formatter<'_>, n: u8) -> fmt::Result {
    let text = match n {
        0 => "nop",
        1 => "yield",
        2 => "wfe",
        3 => "wfi",
        4 => "sev",
        5 => "sevl",
        6 => "dgh",
        // Hints 7, 8, 10, 12, 14 and 24 to 31 (`xpaclri`, `pacia1716`, `paciasp`, ...) are
        // pointer authentication: the decoder yields `Pauth`, never `Hint`, for them.
        16 => "esb",
        17 => "psb csync",
        18 => "tsb csync",
        19 => "gcsb dsync",
        20 => "csdb",
        22 => "clrbhb",
        32 => "bti r",
        34 => "bti c",
        36 => "bti j",
        38 => "bti jc",
        40 => "chkfeat x16",
        48 => "stshh keep",
        49 => "stshh strm",
        50 => "shuh",
        51 => "shuh ph",
        52 => "stcph",
        _ => return write!(f, "hint #{n:#x}"),
    };
    f.write_str(text)
}

/// How a named `sys` alias spells its `Xt` operand.
#[derive(Clone, Copy)]
enum Xt {
    /// Never printed.
    Never,
    /// Always printed (`xzr` for 31).
    Always,
    /// Printed unless it is 31.
    UnlessZero,
}

const IC: &[(u8, u8, u8, &str, Xt)] = &[
    (0, 1, 0, "ialluis", Xt::Never),
    (0, 5, 0, "iallu", Xt::Never),
    (3, 5, 1, "ivau", Xt::Always),
];

const DC: &[(u8, u8, u8, &str, Xt)] = &[
    (0, 6, 1, "ivac", Xt::Always),
    (0, 6, 2, "isw", Xt::Always),
    (0, 6, 3, "igvac", Xt::Always),
    (0, 6, 4, "igsw", Xt::Always),
    (0, 6, 5, "igdvac", Xt::Always),
    (0, 6, 6, "igdsw", Xt::Always),
    (0, 10, 2, "csw", Xt::Always),
    (0, 10, 4, "cgsw", Xt::Always),
    (0, 10, 6, "cgdsw", Xt::Always),
    (0, 14, 2, "cisw", Xt::Always),
    (0, 14, 4, "cigsw", Xt::Always),
    (0, 14, 6, "cigdsw", Xt::Always),
    (3, 10, 1, "cvac", Xt::Always),
    (3, 10, 3, "cgvac", Xt::Always),
    (3, 10, 5, "cgdvac", Xt::Always),
    (3, 11, 0, "cvaoc", Xt::Always),
    (3, 11, 1, "cvau", Xt::Always),
    (3, 11, 7, "cgdvaoc", Xt::Always),
    (3, 14, 1, "civac", Xt::Always),
    (3, 14, 3, "cigvac", Xt::Always),
    (3, 14, 5, "cigdvac", Xt::Always),
    (4, 14, 0, "cipae", Xt::Always),
    (4, 14, 7, "cigdpae", Xt::Always),
    (6, 14, 1, "cipapa", Xt::Always),
    (6, 14, 5, "cigdpapa", Xt::Always),
];

const TLBI: &[(u8, u8, u8, &str, Xt)] = &[
    (0, 1, 0, "vmalle1os", Xt::UnlessZero),
    (0, 1, 1, "vae1os", Xt::Always),
    (0, 1, 2, "aside1os", Xt::Always),
    (0, 1, 3, "vaae1os", Xt::Always),
    (0, 1, 5, "vale1os", Xt::Always),
    (0, 1, 7, "vaale1os", Xt::Always),
    (0, 2, 1, "rvae1is", Xt::Always),
    (0, 2, 3, "rvaae1is", Xt::Always),
    (0, 2, 5, "rvale1is", Xt::Always),
    (0, 2, 7, "rvaale1is", Xt::Always),
    (0, 3, 0, "vmalle1is", Xt::UnlessZero),
    (0, 3, 1, "vae1is", Xt::Always),
    (0, 3, 2, "aside1is", Xt::Always),
    (0, 3, 3, "vaae1is", Xt::Always),
    (0, 3, 5, "vale1is", Xt::Always),
    (0, 3, 7, "vaale1is", Xt::Always),
    (0, 5, 1, "rvae1os", Xt::Always),
    (0, 5, 3, "rvaae1os", Xt::Always),
    (0, 5, 5, "rvale1os", Xt::Always),
    (0, 5, 7, "rvaale1os", Xt::Always),
    (0, 6, 1, "rvae1", Xt::Always),
    (0, 6, 3, "rvaae1", Xt::Always),
    (0, 6, 5, "rvale1", Xt::Always),
    (0, 6, 7, "rvaale1", Xt::Always),
    (0, 7, 0, "vmalle1", Xt::Never),
    (0, 7, 1, "vae1", Xt::Always),
    (0, 7, 2, "aside1", Xt::Always),
    (0, 7, 3, "vaae1", Xt::Always),
    (0, 7, 5, "vale1", Xt::Always),
    (0, 7, 7, "vaale1", Xt::Always),
    (4, 0, 1, "ipas2e1is", Xt::Always),
    (4, 0, 2, "ripas2e1is", Xt::Always),
    (4, 0, 5, "ipas2le1is", Xt::Always),
    (4, 0, 6, "ripas2le1is", Xt::Always),
    (4, 1, 0, "alle2os", Xt::UnlessZero),
    (4, 1, 1, "vae2os", Xt::Always),
    (4, 1, 4, "alle1os", Xt::UnlessZero),
    (4, 1, 5, "vale2os", Xt::Always),
    (4, 1, 6, "vmalls12e1os", Xt::UnlessZero),
    (4, 2, 1, "rvae2is", Xt::Always),
    (4, 2, 2, "vmallws2e1is", Xt::UnlessZero),
    (4, 2, 5, "rvale2is", Xt::Always),
    (4, 3, 0, "alle2is", Xt::UnlessZero),
    (4, 3, 1, "vae2is", Xt::Always),
    (4, 3, 4, "alle1is", Xt::UnlessZero),
    (4, 3, 5, "vale2is", Xt::Always),
    (4, 3, 6, "vmalls12e1is", Xt::UnlessZero),
    (4, 4, 0, "ipas2e1os", Xt::Always),
    (4, 4, 1, "ipas2e1", Xt::Always),
    (4, 4, 2, "ripas2e1", Xt::Always),
    (4, 4, 3, "ripas2e1os", Xt::Always),
    (4, 4, 4, "ipas2le1os", Xt::Always),
    (4, 4, 5, "ipas2le1", Xt::Always),
    (4, 4, 6, "ripas2le1", Xt::Always),
    (4, 4, 7, "ripas2le1os", Xt::Always),
    (4, 5, 1, "rvae2os", Xt::Always),
    (4, 5, 2, "vmallws2e1os", Xt::UnlessZero),
    (4, 5, 5, "rvale2os", Xt::Always),
    (4, 6, 1, "rvae2", Xt::Always),
    (4, 6, 3, "vmallws2e1", Xt::Never),
    (4, 6, 5, "rvale2", Xt::Always),
    (4, 7, 0, "alle2", Xt::Never),
    (4, 7, 1, "vae2", Xt::Always),
    (4, 7, 4, "alle1", Xt::Never),
    (4, 7, 5, "vale2", Xt::Always),
    (4, 7, 6, "vmalls12e1", Xt::Never),
    (6, 1, 0, "alle3os", Xt::Never),
    (6, 1, 1, "vae3os", Xt::Always),
    (6, 1, 4, "paallos", Xt::Never),
    (6, 1, 5, "vale3os", Xt::Always),
    (6, 2, 1, "rvae3is", Xt::Always),
    (6, 2, 5, "rvale3is", Xt::Always),
    (6, 3, 0, "alle3is", Xt::Never),
    (6, 3, 1, "vae3is", Xt::Always),
    (6, 3, 5, "vale3is", Xt::Always),
    (6, 4, 3, "rpaos", Xt::Always),
    (6, 4, 7, "rpalos", Xt::Always),
    (6, 5, 1, "rvae3os", Xt::Always),
    (6, 5, 5, "rvale3os", Xt::Always),
    (6, 6, 1, "rvae3", Xt::Always),
    (6, 6, 5, "rvale3", Xt::Always),
    (6, 7, 0, "alle3", Xt::Never),
    (6, 7, 1, "vae3", Xt::Always),
    (6, 7, 4, "paall", Xt::Never),
    (6, 7, 5, "vale3", Xt::Always),
];

fn lookup(table: &[(u8, u8, u8, &'static str, Xt)], op: &SysOp) -> Option<(&'static str, Xt)> {
    table
        .iter()
        .find(|&&(op1, crm, op2, _, _)| (op1, crm, op2) == (op.op1, op.crm, op.op2))
        .map(|&(_, _, _, name, xt)| (name, xt))
}

/// `mnemonic name{, xt}` for a named alias, or the generic `sys` form.
fn named_sys(
    f: &mut Formatter<'_>,
    mnemonic: &str,
    named: Option<(&str, Xt)>,
    op: &SysOp,
) -> fmt::Result {
    let Some((name, xt)) = named else {
        return generic_sys(f, op);
    };
    write!(f, "{mnemonic} {name}")?;
    match xt {
        Xt::Never => Ok(()),
        Xt::UnlessZero if op.rt == 31 => Ok(()),
        Xt::Always | Xt::UnlessZero => {
            f.write_str(", ")?;
            gpr(f, Width::X64, op.rt)
        }
    }
}

fn generic_sys(f: &mut Formatter<'_>, op: &SysOp) -> fmt::Result {
    write!(f, "sys #{}, C{}, C{}, #{}", op.op1, op.crn, op.crm, op.op2)?;
    if op.rt != 31 {
        f.write_str(", ")?;
        gpr(f, Width::X64, op.rt)?;
    }
    Ok(())
}

fn cache_op(f: &mut Formatter<'_>, op: &SysOp) -> fmt::Result {
    match lookup(IC, op) {
        Some(ic) => named_sys(f, "ic", Some(ic), op),
        None => named_sys(f, "dc", lookup(DC, op), op),
    }
}

fn tlbi(f: &mut Formatter<'_>, op: &SysOp) -> fmt::Result {
    named_sys(f, "tlbi", lookup(TLBI, op), op)
}

/// `(op1, crm, op2)` of the Apple IMP-DEF register in `System::apple[slot]`
/// (`S3_<op1>_C15_C<crm>_<op2>`).
const APPLE: [(u8, u8, u8); 21] = [
    (1, 0, 0),
    (1, 1, 0),
    (1, 2, 0),
    (1, 3, 0),
    (1, 4, 0),
    (1, 5, 0),
    (1, 6, 0),
    (2, 0, 0),
    (2, 1, 0),
    (2, 2, 0),
    (2, 3, 0),
    (2, 4, 0),
    (2, 5, 0),
    (2, 6, 0),
    (2, 7, 0),
    (2, 9, 0),
    (2, 10, 0),
    (5, 0, 0),
    (5, 0, 1),
    (5, 3, 1),
    (7, 0, 4),
];

fn generic_register(
    f: &mut Formatter<'_>,
    op0: u8,
    op1: u8,
    crn: u8,
    crm: u8,
    op2: u8,
) -> fmt::Result {
    write!(f, "s{op0}_{op1}_c{crn}_c{crm}_{op2}")
}

fn register_name(f: &mut Formatter<'_>, register: SystemRegister) -> fmt::Result {
    use SystemRegister as R;
    let name = match register {
        R::SpEl0 => "sp_el0",
        R::SpEl1 => "sp_el1",
        R::SpEl2 => "sp_el2",
        R::SctlrEl2 => "sctlr_el2",
        R::TcrEl2 => "tcr_el2",
        R::Ttbr0El2 => "ttbr0_el2",
        R::Ttbr1El2 => "ttbr1_el2",
        R::MairEl2 => "mair_el2",
        R::VbarEl2 => "vbar_el2",
        R::ElrEl2 => "elr_el2",
        R::SpsrEl2 => "spsr_el2",
        R::EsrEl2 => "esr_el2",
        R::FarEl2 => "far_el2",
        R::HcrEl2 => "hcr_el2",
        R::TpidrEl2 => "tpidr_el2",
        R::CntvoffEl2 => "cntvoff_el2",
        R::CurrentEl => "currentel",
        R::SpsrEl1 => "spsr_el1",
        R::ElrEl1 => "elr_el1",
        R::EsrEl1 => "esr_el1",
        R::FarEl1 => "far_el1",
        R::ParEl1 => "par_el1",
        R::VbarEl1 => "vbar_el1",
        R::CpacrEl1 => "cpacr_el1",
        R::SctlrEl1 => "sctlr_el1",
        R::Ttbr0El1 => "ttbr0_el1",
        R::Ttbr1El1 => "ttbr1_el1",
        R::TcrEl1 => "tcr_el1",
        R::MairEl1 => "mair_el1",
        R::TpidrEl0 => "tpidr_el0",
        R::TpidrroEl0 => "tpidrro_el0",
        R::CntvctEl0 => "cntvct_el0",
        R::CntvCtlEl0 => "cntv_ctl_el0",
        R::CntvCvalEl0 => "cntv_cval_el0",
        R::CntvTvalEl0 => "cntv_tval_el0",
        R::TpidrEl1 => "tpidr_el1",
        R::ContextidrEl1 => "contextidr_el1",
        R::MdscrEl1 => "mdscr_el1",
        R::OslarEl1 => "oslar_el1",
        R::OsdlrEl1 => "osdlr_el1",
        R::OslsrEl1 => "oslsr_el1",
        R::CntkctlEl1 => "cntkctl_el1",
        R::Fpcr => "fpcr",
        R::Fpsr => "fpsr",
        R::Daif => "daif",
        R::SpSel => "spsel",
        R::Pan => "pan",
        R::IsrEl1 => "isr_el1",
        R::IccSreEl1 => "icc_sre_el1",
        R::IccPmrEl1 => "icc_pmr_el1",
        R::IccBpr0El1 => "icc_bpr0_el1",
        R::IccCtlrEl1 => "icc_ctlr_el1",
        R::IccIgrpen0El1 => "icc_igrpen0_el1",
        R::IccIgrpen1El1 => "icc_igrpen1_el1",
        R::IccBpr1El1 => "icc_bpr1_el1",
        R::CntpCtlEl0 => "cntp_ctl_el0",
        R::CntpctEl0 => "cntpct_el0",
        R::CntpTvalEl0 => "cntp_tval_el0",
        R::CntpCvalEl0 => "cntp_cval_el0",
        R::IccIar1El1 => "icc_iar1_el1",
        R::IccEoir1El1 => "icc_eoir1_el1",
        R::ActlrEl1 => "actlr_el1",
        R::MidrEl1 => "midr_el1",
        R::MpidrEl1 => "mpidr_el1",
        R::RevidrEl1 => "revidr_el1",
        R::CtrEl0 => "ctr_el0",
        R::DczidEl0 => "dczid_el0",
        R::CntfrqEl0 => "cntfrq_el0",
        R::ClidrEl1 => "clidr_el1",
        R::CcsidrEl1 => "ccsidr_el1",
        R::CsselrEl1 => "csselr_el1",
        R::IdAa64pfr0El1 => "id_aa64pfr0_el1",
        R::IdAa64pfr1El1 => "id_aa64pfr1_el1",
        R::IdAa64pfr2El1 => "id_aa64pfr2_el1",
        R::IdAa64zfr0El1 => "id_aa64zfr0_el1",
        R::IdAa64smfr0El1 => "id_aa64smfr0_el1",
        R::IdAa64fpfr0El1 => "id_aa64fpfr0_el1",
        R::IdAa64isar3El1 => "id_aa64isar3_el1",
        R::AidrEl1 => "aidr_el1",
        R::IdAa64dfr0El1 => "id_aa64dfr0_el1",
        R::IdAa64dfr1El1 => "id_aa64dfr1_el1",
        R::IdAa64isar0El1 => "id_aa64isar0_el1",
        R::IdAa64isar1El1 => "id_aa64isar1_el1",
        R::IdAa64isar2El1 => "id_aa64isar2_el1",
        R::IdAa64mmfr0El1 => "id_aa64mmfr0_el1",
        R::IdAa64mmfr1El1 => "id_aa64mmfr1_el1",
        R::IdAa64mmfr2El1 => "id_aa64mmfr2_el1",
        R::IdAa64mmfr3El1 => "id_aa64mmfr3_el1",
        R::IdAa64mmfr4El1 => "id_aa64mmfr4_el1",
        R::Nzcv => "nzcv",
        R::DaifSet => "daifset",
        R::DaifClear => "daifclr",
        R::SpSelSet => "spsel",
        R::PanSet => "pan",
        R::Apple(slot) => {
            let (op1, crm, op2) = APPLE[usize::from(slot) % APPLE.len()];
            return generic_register(f, 3, op1, 15, crm, op2);
        }
        R::AppleBank(slot) => {
            let op1 = ((slot >> 7) & 7) as u8;
            let crm = ((slot >> 3) & 15) as u8;
            let op2 = (slot & 7) as u8;
            return generic_register(f, 3, op1, 15, crm, op2);
        }
        R::AppleZero { op1, crm, op2 } => return generic_register(f, 3, op1, 15, crm, op2),
        R::RazWi { crm, op2 } => {
            let name = ["dbgbvr", "dbgbcr", "dbgwvr", "dbgwcr"][usize::from(op2 & 3)];
            return write!(f, "{name}{crm}_el1");
        }
        R::PacKey(slot) => {
            let (key, half) = [
                ("apia", "lo"),
                ("apia", "hi"),
                ("apib", "lo"),
                ("apib", "hi"),
                ("apda", "lo"),
                ("apda", "hi"),
                ("apdb", "lo"),
                ("apdb", "hi"),
                ("apga", "lo"),
                ("apga", "hi"),
            ][usize::from(slot) % 10];
            return write!(f, "{key}key{half}_el1");
        }
    };
    f.write_str(name)
}

fn system_register_move(f: &mut Formatter<'_>, system: &decode::System) -> fmt::Result {
    use SystemRegister as R;
    if system.read {
        f.write_str("mrs ")?;
        gpr(f, Width::X64, system.rt)?;
        f.write_str(", ")?;
        return register_name(f, system.register);
    }
    f.write_str("msr ")?;
    register_name(f, system.register)?;
    f.write_str(", ")?;
    match system.register {
        R::DaifSet | R::DaifClear | R::SpSelSet | R::PanSet => {
            write!(f, "#{:#x}", system.immediate)
        }
        _ => gpr(f, Width::X64, system.rt),
    }
}

/// The key letter of `braa`/`retab`/`eretaa`: `a` then `a` or `b`.
fn branch_key(key: KeyId) -> &'static str {
    match key {
        KeyId::Ib => "ab",
        _ => "aa",
    }
}

/// The key as `<i|d><a|b>` (`ga` for the generic key); with a zero modifier the
/// `z` goes between the two letters (`paciza`, `autdzb`).
fn key_name(f: &mut Formatter<'_>, key: KeyId, zero: bool) -> fmt::Result {
    let (kind, which) = match key {
        KeyId::Ia => ("i", "a"),
        KeyId::Ib => ("i", "b"),
        KeyId::Da => ("d", "a"),
        KeyId::Db => ("d", "b"),
        KeyId::Ga => ("g", "a"),
    };
    write!(f, "{kind}{}{which}", if zero { "z" } else { "" })
}

/// `sp` for register 31, else `xN`: the modifier register of pointer authentication.
fn modifier_register(f: &mut Formatter<'_>, r: u8) -> fmt::Result {
    if r == 31 {
        f.write_str("sp")
    } else {
        write!(f, "x{r}")
    }
}

fn pauth_op(f: &mut Formatter<'_>, op: pauth::Op) -> fmt::Result {
    use pauth::Op::{Auth, Branch, ExceptionReturn, Pacga, Return, Sign, Strip};
    match op {
        Sign {
            key,
            reg,
            modifier,
            hint,
        }
        | Auth {
            key,
            reg,
            modifier,
            hint,
        } => {
            f.write_str(if matches!(op, Sign { .. }) {
                "pac"
            } else {
                "aut"
            })?;
            if hint {
                // `paciaz`, `paciasp`, `pacia1716`: no operands.
                key_name(f, key, false)?;
                return f.write_str(match modifier {
                    Modifier::Zero => "z",
                    Modifier::Reg(31) => "sp",
                    Modifier::Reg(_) => "1716",
                });
            }
            key_name(f, key, matches!(modifier, Modifier::Zero))?;
            f.write_str(" ")?;
            gpr(f, Width::X64, reg)?;
            if let Modifier::Reg(m) = modifier {
                f.write_str(", ")?;
                modifier_register(f, m)?;
            }
            Ok(())
        }
        Strip { hint: true, .. } => f.write_str("xpaclri"),
        Strip { data, reg, .. } => {
            f.write_str(if data { "xpacd " } else { "xpaci " })?;
            gpr(f, Width::X64, reg)
        }
        Pacga { rd, rn, modifier } => {
            f.write_str("pacga ")?;
            gpr(f, Width::X64, rd)?;
            f.write_str(", ")?;
            gpr(f, Width::X64, rn)?;
            f.write_str(", ")?;
            match modifier {
                Modifier::Reg(m) => modifier_register(f, m),
                Modifier::Zero => f.write_str("xzr"),
            }
        }
        Branch {
            key,
            link,
            target,
            modifier,
        } => {
            write!(f, "{}{}", if link { "blr" } else { "br" }, branch_key(key))?;
            if matches!(modifier, Modifier::Zero) {
                f.write_str("z")?;
            }
            f.write_str(" ")?;
            gpr(f, Width::X64, target)?;
            if let Modifier::Reg(m) = modifier {
                f.write_str(", ")?;
                modifier_register(f, m)?;
            }
            Ok(())
        }
        Return { key } => write!(f, "ret{}", branch_key(key)),
        ExceptionReturn { key } => write!(f, "eret{}", branch_key(key)),
    }
}

#[cfg(test)]
mod tests {
    use crate::format::{word, word_at};

    fn check(w: u32, expected: &str) {
        assert_eq!(
            std::string::ToString::to_string(&word(w)),
            expected,
            "{w:#010x}"
        );
    }

    fn check_at(w: u32, pc: u64, expected: &str) {
        assert_eq!(
            std::string::ToString::to_string(&word_at(w, pc)),
            expected,
            "{w:#010x} at {pc:#x}"
        );
    }

    #[test]
    fn relative_branches_without_an_address() {
        check(0x14000002, "b .+8");
        check(0x17ffffff, "b .-4");
        check(0x94000040, "bl .+256");
        check(0x97fffffe, "bl .-8");
        check(0x54000040, "b.eq .+8");
        check(0x54ffffe1, "b.ne .-4");
        check(0x34000040, "cbz w0, .+8");
        check(0xb5fffff5, "cbnz x21, .-4");
        check(0x36180040, "tbz w0, #3, .+8");
        check(0xb7f80041, "tbnz x1, #63, .+8");
    }

    #[test]
    fn relative_branches_at_an_address() {
        check_at(0x14000002, 0x10100000, "b 0x10100008");
        check_at(0x94000040, 0x10100004, "bl 0x10100104");
        check_at(0x17ffffff, 0x10100008, "b 0x10100004");
        check_at(0x97fffffe, 0x1010000c, "bl 0x10100004");
        check_at(0x15ffffff, 0x10100010, "b 0x1810000c");
        check_at(0x96000000, 0x10100014, "bl 0x8100014");
        check_at(0x54000040, 0x10100018, "b.eq 0x10100020");
        check_at(0x54ffffe1, 0x1010001c, "b.ne 0x10100018");
        check_at(0x543fffe2, 0x10100020, "b.cs 0x1018001c");
        check_at(0x54800003, 0x10100024, "b.cc 0x10000024");
        check_at(0x54000024, 0x10100028, "b.mi 0x1010002c");
        check_at(0x54000025, 0x1010002c, "b.pl 0x10100030");
        check_at(0x54000026, 0x10100030, "b.vs 0x10100034");
        check_at(0x54000027, 0x10100034, "b.vc 0x10100038");
        check_at(0x54000028, 0x10100038, "b.hi 0x1010003c");
        check_at(0x54000029, 0x1010003c, "b.ls 0x10100040");
        check_at(0x5400002a, 0x10100040, "b.ge 0x10100044");
        check_at(0x5400002b, 0x10100044, "b.lt 0x10100048");
        check_at(0x5400002c, 0x10100048, "b.gt 0x1010004c");
        check_at(0x5400002d, 0x1010004c, "b.le 0x10100050");
        check_at(0x34000040, 0x10100050, "cbz w0, 0x10100058");
        check_at(0x35ffffc5, 0x10100054, "cbnz w5, 0x1010004c");
        check_at(0xb400009f, 0x10100058, "cbz xzr, 0x10100068");
        check_at(0xb580001e, 0x1010005c, "cbnz x30, 0x1000005c");
        check_at(0x357fffff, 0x10100060, "cbnz wzr, 0x1020005c");
        check_at(0x36000040, 0x10100064, "tbz w0, #0, 0x1010006c");
        check_at(0x372fffc3, 0x10100068, "tbnz w3, #5, 0x10100060");
        check_at(0x36f8ffff, 0x1010006c, "tbz wzr, #31, 0x10102068");
        check_at(0xb704001e, 0x10100070, "tbnz x30, #32, 0x100f8070");
        check_at(0xb6f80205, 0x10100074, "tbz x5, #63, 0x101000b4");
        check_at(0xb7080020, 0x10100078, "tbnz x0, #33, 0x1010007c");
    }

    #[test]
    fn hints() {
        check(0xd503201f, "nop");
        check(0xd503203f, "yield");
        check(0xd503205f, "wfe");
        check(0xd503207f, "wfi");
        check(0xd503209f, "sev");
        check(0xd50320bf, "sevl");
        check(0xd50320df, "dgh");
        check(0xd50320ff, "xpaclri");
        check(0xd503211f, "pacia1716");
        check(0xd503215f, "pacib1716");
        check(0xd503219f, "autia1716");
        check(0xd50321df, "autib1716");
        check(0xd50321ff, "hint #0xf");
        check(0xd503221f, "esb");
        check(0xd503223f, "psb csync");
        check(0xd503225f, "tsb csync");
        check(0xd503227f, "gcsb dsync");
        check(0xd503229f, "csdb");
        check(0xd50322bf, "hint #0x15");
        check(0xd50322df, "clrbhb");
        check(0xd50322ff, "hint #0x17");
        check(0xd503231f, "paciaz");
        check(0xd503233f, "paciasp");
        check(0xd503235f, "pacibz");
        check(0xd503237f, "pacibsp");
        check(0xd503239f, "autiaz");
        check(0xd50323bf, "autiasp");
        check(0xd50323df, "autibz");
        check(0xd50323ff, "autibsp");
        check(0xd503241f, "bti r");
        check(0xd503243f, "hint #0x21");
        check(0xd503245f, "bti c");
        check(0xd503249f, "bti j");
        check(0xd50324df, "bti jc");
        check(0xd503251f, "chkfeat x16");
        check(0xd503261f, "stshh keep");
        check(0xd503263f, "stshh strm");
        check(0xd503265f, "shuh");
        check(0xd503267f, "shuh ph");
        check(0xd503269f, "stcph");
        check(0xd50326bf, "hint #0x35");
        check(0xd5032fff, "hint #0x7f");
    }

    #[test]
    fn barriers_and_clrex() {
        check(0xd503305f, "clrex #0x0");
        check(0xd503315f, "clrex #0x1");
        check(0xd503325f, "clrex #0x2");
        check(0xd503335f, "clrex #0x3");
        check(0xd503345f, "clrex #0x4");
        check(0xd503355f, "clrex #0x5");
        check(0xd503365f, "clrex #0x6");
        check(0xd503375f, "clrex #0x7");
        check(0xd503385f, "clrex #0x8");
        check(0xd503395f, "clrex #0x9");
        check(0xd5033a5f, "clrex #0xa");
        check(0xd5033b5f, "clrex #0xb");
        check(0xd5033c5f, "clrex #0xc");
        check(0xd5033d5f, "clrex #0xd");
        check(0xd5033e5f, "clrex #0xe");
        check(0xd5033f5f, "clrex");
        check(0xd503309f, "ssbb");
        check(0xd503319f, "dsb oshld");
        check(0xd503329f, "dsb oshst");
        check(0xd503339f, "dsb osh");
        check(0xd503349f, "pssbb");
        check(0xd503359f, "dsb nshld");
        check(0xd503369f, "dsb nshst");
        check(0xd503379f, "dsb nsh");
        check(0xd503389f, "dsb #0x08");
        check(0xd503399f, "dsb ishld");
        check(0xd5033a9f, "dsb ishst");
        check(0xd5033b9f, "dsb ish");
        check(0xd5033c9f, "dsb #0x0c");
        check(0xd5033d9f, "dsb ld");
        check(0xd5033e9f, "dsb st");
        check(0xd5033f9f, "dsb sy");
        check(0xd50330bf, "dmb #0x00");
        check(0xd50331bf, "dmb oshld");
        check(0xd50332bf, "dmb oshst");
        check(0xd50333bf, "dmb osh");
        check(0xd50334bf, "dmb #0x04");
        check(0xd50335bf, "dmb nshld");
        check(0xd50336bf, "dmb nshst");
        check(0xd50337bf, "dmb nsh");
        check(0xd50338bf, "dmb #0x08");
        check(0xd50339bf, "dmb ishld");
        check(0xd5033abf, "dmb ishst");
        check(0xd5033bbf, "dmb ish");
        check(0xd5033cbf, "dmb #0x0c");
        check(0xd5033dbf, "dmb ld");
        check(0xd5033ebf, "dmb st");
        check(0xd5033fbf, "dmb sy");
        check(0xd50330df, "isb #0x0");
        check(0xd50331df, "isb #0x1");
        check(0xd50332df, "isb #0x2");
        check(0xd50333df, "isb #0x3");
        check(0xd50334df, "isb #0x4");
        check(0xd50335df, "isb #0x5");
        check(0xd50336df, "isb #0x6");
        check(0xd50337df, "isb #0x7");
        check(0xd50338df, "isb #0x8");
        check(0xd50339df, "isb #0x9");
        check(0xd5033adf, "isb #0xa");
        check(0xd5033bdf, "isb #0xb");
        check(0xd5033cdf, "isb #0xc");
        check(0xd5033ddf, "isb #0xd");
        check(0xd5033edf, "isb #0xe");
        check(0xd5033fdf, "isb");
    }

    #[test]
    fn every_system_register_reads() {
        check(0xd5384100, "mrs x0, sp_el0");
        check(0xd53c4100, "mrs x0, sp_el1");
        check(0xd53e4100, "mrs x0, sp_el2");
        check(0xd53c1000, "mrs x0, sctlr_el2");
        check(0xd53c2040, "mrs x0, tcr_el2");
        check(0xd53c2000, "mrs x0, ttbr0_el2");
        check(0xd53c2020, "mrs x0, ttbr1_el2");
        check(0xd53ca200, "mrs x0, mair_el2");
        check(0xd53cc000, "mrs x0, vbar_el2");
        check(0xd53c4020, "mrs x0, elr_el2");
        check(0xd53c4000, "mrs x0, spsr_el2");
        check(0xd53c5200, "mrs x0, esr_el2");
        check(0xd53c6000, "mrs x0, far_el2");
        check(0xd53c1100, "mrs x0, hcr_el2");
        check(0xd53cd040, "mrs x0, tpidr_el2");
        check(0xd53ce060, "mrs x0, cntvoff_el2");
        check(0xd5384240, "mrs x0, currentel");
        check(0xd5384000, "mrs x0, spsr_el1");
        check(0xd5384020, "mrs x0, elr_el1");
        check(0xd5385200, "mrs x0, esr_el1");
        check(0xd5386000, "mrs x0, far_el1");
        check(0xd5387400, "mrs x0, par_el1");
        check(0xd538c000, "mrs x0, vbar_el1");
        check(0xd5381040, "mrs x0, cpacr_el1");
        check(0xd5381000, "mrs x0, sctlr_el1");
        check(0xd5382000, "mrs x0, ttbr0_el1");
        check(0xd5382020, "mrs x0, ttbr1_el1");
        check(0xd5382040, "mrs x0, tcr_el1");
        check(0xd538a200, "mrs x0, mair_el1");
        check(0xd53bd040, "mrs x0, tpidr_el0");
        check(0xd53bd060, "mrs x0, tpidrro_el0");
        check(0xd53be040, "mrs x0, cntvct_el0");
        check(0xd53be320, "mrs x0, cntv_ctl_el0");
        check(0xd53be340, "mrs x0, cntv_cval_el0");
        check(0xd53be300, "mrs x0, cntv_tval_el0");
        check(0xd538d080, "mrs x0, tpidr_el1");
        check(0xd538d020, "mrs x0, contextidr_el1");
        check(0xd5300240, "mrs x0, mdscr_el1");
        check(0xd5301080, "mrs x0, oslar_el1");
        check(0xd5301380, "mrs x0, osdlr_el1");
        check(0xd5301180, "mrs x0, oslsr_el1");
        check(0xd538e100, "mrs x0, cntkctl_el1");
        check(0xd53b4400, "mrs x0, fpcr");
        check(0xd53b4420, "mrs x0, fpsr");
        check(0xd53b4220, "mrs x0, daif");
        check(0xd5384200, "mrs x0, spsel");
        check(0xd5384260, "mrs x0, pan");
        check(0xd538c100, "mrs x0, isr_el1");
        check(0xd538cca0, "mrs x0, icc_sre_el1");
        check(0xd5384600, "mrs x0, icc_pmr_el1");
        check(0xd538c860, "mrs x0, icc_bpr0_el1");
        check(0xd538cc80, "mrs x0, icc_ctlr_el1");
        check(0xd538ccc0, "mrs x0, icc_igrpen0_el1");
        check(0xd538cce0, "mrs x0, icc_igrpen1_el1");
        check(0xd538cc60, "mrs x0, icc_bpr1_el1");
        check(0xd53be220, "mrs x0, cntp_ctl_el0");
        check(0xd53be020, "mrs x0, cntpct_el0");
        check(0xd53be200, "mrs x0, cntp_tval_el0");
        check(0xd53be240, "mrs x0, cntp_cval_el0");
        check(0xd538cc00, "mrs x0, icc_iar1_el1");
        check(0xd538cc20, "mrs x0, icc_eoir1_el1");
        check(0xd5381020, "mrs x0, actlr_el1");
        check(0xd5380000, "mrs x0, midr_el1");
        check(0xd53800a0, "mrs x0, mpidr_el1");
        check(0xd53800c0, "mrs x0, revidr_el1");
        check(0xd53b0020, "mrs x0, ctr_el0");
        check(0xd53b00e0, "mrs x0, dczid_el0");
        check(0xd53be000, "mrs x0, cntfrq_el0");
        check(0xd5390020, "mrs x0, clidr_el1");
        check(0xd5390000, "mrs x0, ccsidr_el1");
        check(0xd53a0000, "mrs x0, csselr_el1");
        check(0xd5380400, "mrs x0, id_aa64pfr0_el1");
        check(0xd5380420, "mrs x0, id_aa64pfr1_el1");
        check(0xd5380440, "mrs x0, id_aa64pfr2_el1");
        check(0xd5380480, "mrs x0, id_aa64zfr0_el1");
        check(0xd53804a0, "mrs x0, id_aa64smfr0_el1");
        check(0xd53804e0, "mrs x0, id_aa64fpfr0_el1");
        check(0xd5380660, "mrs x0, id_aa64isar3_el1");
        check(0xd53900e0, "mrs x0, aidr_el1");
        check(0xd5380500, "mrs x0, id_aa64dfr0_el1");
        check(0xd5380520, "mrs x0, id_aa64dfr1_el1");
        check(0xd5380600, "mrs x0, id_aa64isar0_el1");
        check(0xd5380620, "mrs x0, id_aa64isar1_el1");
        check(0xd5380640, "mrs x0, id_aa64isar2_el1");
        check(0xd5380700, "mrs x0, id_aa64mmfr0_el1");
        check(0xd5380720, "mrs x0, id_aa64mmfr1_el1");
        check(0xd5380740, "mrs x0, id_aa64mmfr2_el1");
        check(0xd5380760, "mrs x0, id_aa64mmfr3_el1");
        check(0xd5380780, "mrs x0, id_aa64mmfr4_el1");
        check(0xd53b4200, "mrs x0, nzcv");
    }

    #[test]
    fn every_system_register_writes() {
        check(0xd5184100, "msr sp_el0, x0");
        check(0xd51c4100, "msr sp_el1, x0");
        check(0xd51e4100, "msr sp_el2, x0");
        check(0xd51c1000, "msr sctlr_el2, x0");
        check(0xd51c2040, "msr tcr_el2, x0");
        check(0xd51c2000, "msr ttbr0_el2, x0");
        check(0xd51c2020, "msr ttbr1_el2, x0");
        check(0xd51ca200, "msr mair_el2, x0");
        check(0xd51cc000, "msr vbar_el2, x0");
        check(0xd51c4020, "msr elr_el2, x0");
        check(0xd51c4000, "msr spsr_el2, x0");
        check(0xd51c5200, "msr esr_el2, x0");
        check(0xd51c6000, "msr far_el2, x0");
        check(0xd51c1100, "msr hcr_el2, x0");
        check(0xd51cd040, "msr tpidr_el2, x0");
        check(0xd51ce060, "msr cntvoff_el2, x0");
        check(0xd5184000, "msr spsr_el1, x0");
        check(0xd5184020, "msr elr_el1, x0");
        check(0xd5185200, "msr esr_el1, x0");
        check(0xd5186000, "msr far_el1, x0");
        check(0xd5187400, "msr par_el1, x0");
        check(0xd518c000, "msr vbar_el1, x0");
        check(0xd5181040, "msr cpacr_el1, x0");
        check(0xd5181000, "msr sctlr_el1, x0");
        check(0xd5182000, "msr ttbr0_el1, x0");
        check(0xd5182020, "msr ttbr1_el1, x0");
        check(0xd5182040, "msr tcr_el1, x0");
        check(0xd518a200, "msr mair_el1, x0");
        check(0xd51bd040, "msr tpidr_el0, x0");
        check(0xd51bd060, "msr tpidrro_el0, x0");
        check(0xd51be040, "msr cntvct_el0, x0");
        check(0xd51be320, "msr cntv_ctl_el0, x0");
        check(0xd51be340, "msr cntv_cval_el0, x0");
        check(0xd51be300, "msr cntv_tval_el0, x0");
        check(0xd518d080, "msr tpidr_el1, x0");
        check(0xd518d020, "msr contextidr_el1, x0");
        check(0xd5100240, "msr mdscr_el1, x0");
        check(0xd5101080, "msr oslar_el1, x0");
        check(0xd5101380, "msr osdlr_el1, x0");
        check(0xd5101180, "msr oslsr_el1, x0");
        check(0xd518e100, "msr cntkctl_el1, x0");
        check(0xd51b4400, "msr fpcr, x0");
        check(0xd51b4420, "msr fpsr, x0");
        check(0xd51b4220, "msr daif, x0");
        check(0xd5184200, "msr spsel, x0");
        check(0xd5184260, "msr pan, x0");
        check(0xd518cca0, "msr icc_sre_el1, x0");
        check(0xd5184600, "msr icc_pmr_el1, x0");
        check(0xd518c860, "msr icc_bpr0_el1, x0");
        check(0xd518cc80, "msr icc_ctlr_el1, x0");
        check(0xd518ccc0, "msr icc_igrpen0_el1, x0");
        check(0xd518cce0, "msr icc_igrpen1_el1, x0");
        check(0xd518cc60, "msr icc_bpr1_el1, x0");
        check(0xd51be220, "msr cntp_ctl_el0, x0");
        check(0xd51be020, "msr cntpct_el0, x0");
        check(0xd51be200, "msr cntp_tval_el0, x0");
        check(0xd51be240, "msr cntp_cval_el0, x0");
        check(0xd518cc00, "msr icc_iar1_el1, x0");
        check(0xd518cc20, "msr icc_eoir1_el1, x0");
        check(0xd5181020, "msr actlr_el1, x0");
        check(0xd51a0000, "msr csselr_el1, x0");
        check(0xd51b4200, "msr nzcv, x0");
    }

    #[test]
    fn system_register_edge_cases() {
        check(0xd5034fdf, "msr daifset, #0xf");
        check(0xd50342df, "msr daifset, #0x2");
        check(0xd5034cdf, "msr daifset, #0xc");
        check(0xd5034fff, "msr daifclr, #0xf");
        check(0xd50342ff, "msr daifclr, #0x2");
        check(0xd500409f, "msr pan, #0x0");
        check(0xd500419f, "msr pan, #0x1");
        check(0xd50040bf, "msr spsel, #0x0");
        check(0xd50041bf, "msr spsel, #0x1");
        check(0xd5382100, "mrs x0, apiakeylo_el1");
        check(0xd5382120, "mrs x0, apiakeyhi_el1");
        check(0xd5382140, "mrs x0, apibkeylo_el1");
        check(0xd5382160, "mrs x0, apibkeyhi_el1");
        check(0xd5382200, "mrs x0, apdakeylo_el1");
        check(0xd5382300, "mrs x0, apgakeylo_el1");
        check(0xd5382320, "mrs x0, apgakeyhi_el1");
    }

    #[test]
    fn cache_maintenance() {
        check(0xd5087100, "ic ialluis");
        check(0xd5087500, "ic iallu");
        check(0xd5087620, "dc ivac, x0");
        check(0xd5087640, "dc isw, x0");
        check(0xd5087660, "dc igvac, x0");
        check(0xd5087680, "dc igsw, x0");
        check(0xd50876a0, "dc igdvac, x0");
        check(0xd50876c0, "dc igdsw, x0");
        check(0xd5087a40, "dc csw, x0");
        check(0xd5087a80, "dc cgsw, x0");
        check(0xd5087ac0, "dc cgdsw, x0");
        check(0xd5087e40, "dc cisw, x0");
        check(0xd5087e80, "dc cigsw, x0");
        check(0xd5087ec0, "dc cigdsw, x0");
        check(0xd50b7520, "ic ivau, x0");
        check(0xd50b7a20, "dc cvac, x0");
        check(0xd50b7a60, "dc cgvac, x0");
        check(0xd50b7aa0, "dc cgdvac, x0");
        check(0xd50b7b00, "dc cvaoc, x0");
        check(0xd50b7b20, "dc cvau, x0");
        check(0xd50b7be0, "dc cgdvaoc, x0");
        check(0xd50b7e20, "dc civac, x0");
        check(0xd50b7e60, "dc cigvac, x0");
        check(0xd50b7ea0, "dc cigdvac, x0");
        check(0xd50c7e00, "dc cipae, x0");
        check(0xd50c7ee0, "dc cigdpae, x0");
        check(0xd50e7e20, "dc cipapa, x0");
        check(0xd50e7ea0, "dc cigdpapa, x0");
        check(0xd508711f, "ic ialluis");
        check(0xd508751f, "ic iallu");
        check(0xd508763f, "dc ivac, xzr");
        check(0xd508765f, "dc isw, xzr");
        check(0xd508767f, "dc igvac, xzr");
        check(0xd508769f, "dc igsw, xzr");
        check(0xd50876bf, "dc igdvac, xzr");
        check(0xd50876df, "dc igdsw, xzr");
        check(0xd5087a5f, "dc csw, xzr");
        check(0xd5087a9f, "dc cgsw, xzr");
        check(0xd5087adf, "dc cgdsw, xzr");
        check(0xd5087e5f, "dc cisw, xzr");
        check(0xd5087e9f, "dc cigsw, xzr");
        check(0xd5087edf, "dc cigdsw, xzr");
        check(0xd50b753f, "ic ivau, xzr");
        check(0xd50b7a3f, "dc cvac, xzr");
        check(0xd50b7a7f, "dc cgvac, xzr");
        check(0xd50b7abf, "dc cgdvac, xzr");
        check(0xd50b7b1f, "dc cvaoc, xzr");
        check(0xd50b7b3f, "dc cvau, xzr");
        check(0xd50b7bff, "dc cgdvaoc, xzr");
        check(0xd50b7e3f, "dc civac, xzr");
        check(0xd50b7e7f, "dc cigvac, xzr");
        check(0xd50b7ebf, "dc cigdvac, xzr");
        check(0xd50c7e1f, "dc cipae, xzr");
        check(0xd50c7eff, "dc cigdpae, xzr");
        check(0xd50e7e3f, "dc cipapa, xzr");
        check(0xd50e7ebf, "dc cigdpapa, xzr");
    }

    #[test]
    fn tlbi_named() {
        check(0xd5088100, "tlbi vmalle1os, x0");
        check(0xd50881e0, "tlbi vaale1os, x0");
        check(0xd5088300, "tlbi vmalle1is, x0");
        check(0xd50883e0, "tlbi vaale1is, x0");
        check(0xd5088620, "tlbi rvae1, x0");
        check(0xd5088720, "tlbi vae1, x0");
        check(0xd50c8020, "tlbi ipas2e1is, x0");
        check(0xd50c8120, "tlbi vae2os, x0");
        check(0xd50c8240, "tlbi vmallws2e1is, x0");
        check(0xd50c83a0, "tlbi vale2is, x0");
        check(0xd50c8460, "tlbi ripas2e1os, x0");
        check(0xd50c8520, "tlbi rvae2os, x0");
        check(0xd50c86a0, "tlbi rvale2, x0");
        check(0xd50c87c0, "tlbi vmalls12e1");
        check(0xd50e8220, "tlbi rvae3is, x0");
        check(0xd50e8460, "tlbi rpaos, x0");
        check(0xd50e86a0, "tlbi rvale3, x0");
        check(0xd508811f, "tlbi vmalle1os");
        check(0xd50881ff, "tlbi vaale1os, xzr");
        check(0xd508831f, "tlbi vmalle1is");
        check(0xd50883ff, "tlbi vaale1is, xzr");
        check(0xd508863f, "tlbi rvae1, xzr");
        check(0xd508873f, "tlbi vae1, xzr");
        check(0xd50c803f, "tlbi ipas2e1is, xzr");
        check(0xd50c813f, "tlbi vae2os, xzr");
        check(0xd50c825f, "tlbi vmallws2e1is");
        check(0xd50c83bf, "tlbi vale2is, xzr");
        check(0xd50c847f, "tlbi ripas2e1os, xzr");
        check(0xd50c853f, "tlbi rvae2os, xzr");
        check(0xd50c86bf, "tlbi rvale2, xzr");
        check(0xd50c87df, "tlbi vmalls12e1");
        check(0xd50e823f, "tlbi rvae3is, xzr");
        check(0xd50e847f, "tlbi rpaos, xzr");
        check(0xd50e86bf, "tlbi rvale3, xzr");
    }

    #[test]
    fn generic_sys() {
        check(0xd5087120, "sys #0, C7, C1, #1, x0");
        check(0xd508713f, "sys #0, C7, C1, #1");
        check(0xd50a883f, "sys #2, C8, C8, #1");
        check(0xd50a8840, "sys #2, C8, C8, #2, x0");
        check(0xd50f8fff, "sys #7, C8, C15, #7");
        check(0xd50f8fe0, "sys #7, C8, C15, #7, x0");
        check(0xd50b7420, "dc zva, x0");
        check(0xd50b743f, "dc zva, xzr");
        check(0xd5087800, "at s1e1r, x0");
        check(0xd5087820, "at s1e1w, x0");
        check(0xd5087840, "at s1e0r, x0");
        check(0xd508785f, "at s1e0r, xzr");
        check(0xd5087860, "at s1e0w, x0");
    }

    #[test]
    fn exceptions() {
        check(0xd4000002, "hvc #0x0");
        check(0xd4000001, "svc #0x0");
        check(0xd4001001, "svc #0x80");
        check(0xd4200020, "brk #0x1");
        check(0xd4210000, "brk #0x800");
        check(0xd43fffe0, "brk #0xffff");
        check(0xd69f03e0, "eret");
    }

    #[test]
    fn pointer_authentication() {
        check(0xdac10000, "pacia x0, x0");
        check(0xdac10020, "pacia x0, x1");
        check(0xdac103e0, "pacia x0, sp");
        check(0xdac10400, "pacib x0, x0");
        check(0xdac10800, "pacda x0, x0");
        check(0xdac10c00, "pacdb x0, x0");
        check(0xdac11000, "autia x0, x0");
        check(0xdac11400, "autib x0, x0");
        check(0xdac11800, "autda x0, x0");
        check(0xdac11c00, "autdb x0, x0");
        check(0xdac113e0, "autia x0, sp");
        check(0xdac123e0, "paciza x0");
        check(0xdac127e0, "pacizb x0");
        check(0xdac12be0, "pacdza x0");
        check(0xdac12fe0, "pacdzb x0");
        check(0xdac133e0, "autiza x0");
        check(0xdac137e0, "autizb x0");
        check(0xdac13be0, "autdza x0");
        check(0xdac13fe0, "autdzb x0");
        check(0xdac143e0, "xpaci x0");
        check(0x9ac033e0, "pacga x0, xzr, x0");
        check(0x9adf3000, "pacga x0, x0, sp");
        check(0xd71f0800, "braa x0, x0");
        check(0xd71f0fe3, "brab xzr, x3");
        check(0xd65f0bff, "retaa");
        check(0xd65f0fff, "retab");
        check(0xd69f0bff, "eretaa");
        check(0xd69f0fff, "eretab");
        check(0xd503231f, "paciaz");
        check(0xd503233f, "paciasp");
        check(0xd503235f, "pacibz");
        check(0xd503237f, "pacibsp");
        check(0xd503239f, "autiaz");
        check(0xd50323bf, "autiasp");
        check(0xd50323df, "autibz");
        check(0xd50323ff, "autibsp");
        check(0xd503211f, "pacia1716");
        check(0xd503215f, "pacib1716");
        check(0xd503219f, "autia1716");
        check(0xd50321df, "autib1716");
        check(0xd50320ff, "xpaclri");
        // The register forms of the same operations are not the hint spellings.
        check(0xdac103fe, "pacia x30, sp");
        check(0xdac10211, "pacia x17, x16");
        check(0xdac10611, "pacib x17, x16");
        check(0xdac113fe, "autia x30, sp");
        check(0xdac11211, "autia x17, x16");
        check(0xdac11611, "autib x17, x16");
        check(0xdac123fe, "paciza x30");
        check(0xdac127fe, "pacizb x30");
        check(0xdac133fe, "autiza x30");
        check(0xdac137fe, "autizb x30");
        check(0xdac143fe, "xpaci x30");
        check(0xdac147e0, "xpacd x0");
        check(0xdac147fe, "xpacd x30");
    }

    #[test]
    fn register_branches() {
        check(0xd61f0000, "br x0");
        check(0xd61f0220, "br x17");
        check(0xd61f03e0, "br xzr");
        check(0xd63f0000, "blr x0");
        check(0xd63f03c0, "blr x30");
        check(0xd63f03e0, "blr xzr");
        check(0xd65f0000, "ret x0");
        check(0xd65f03c0, "ret");
        check(0xd65f0220, "ret x17");
        check(0xd65f03e0, "ret xzr");
    }

    #[test]
    fn debug_and_apple_zero_registers_keep_their_spelling() {
        check(0xd5300080, "mrs x0, dbgbvr0_el1");
        check(0xd53000a0, "mrs x0, dbgbcr0_el1");
        check(0xd53000c0, "mrs x0, dbgwvr0_el1");
        check(0xd53000e0, "mrs x0, dbgwcr0_el1");
        check(0xd5300fe0, "mrs x0, dbgwcr15_el1");
        check(0xd5300ac1, "mrs x1, dbgwvr10_el1");
        check(0xd5100180, "msr dbgbvr1_el1, x0");
        check(0xd51001a0, "msr dbgbcr1_el1, x0");
        check(0xd51001c0, "msr dbgwvr1_el1, x0");
        check(0xd51001e0, "msr dbgwcr1_el1, x0");
        check(0xd53df600, "mrs x0, s3_5_c15_c6_0");
        check(0xd53ff680, "mrs x0, s3_7_c15_c6_4");
        check(0xd539fd00, "mrs x0, s3_1_c15_c13_0");
        check(0xd53df120, "mrs x0, s3_5_c15_c1_1");
    }

    #[test]
    fn implementation_defined_registers() {
        check(0xd539f000, "mrs x0, s3_1_c15_c0_0");
        check(0xd539f600, "mrs x0, s3_1_c15_c6_0");
        check(0xd53afa00, "mrs x0, s3_2_c15_c10_0");
        check(0xd53df320, "mrs x0, s3_5_c15_c3_1");
        check(0xd53ff080, "mrs x0, s3_7_c15_c0_4");
        check(0xd539fd00, "mrs x0, s3_1_c15_c13_0");
        check(0xd53df120, "mrs x0, s3_5_c15_c1_1");
        check(0xd53df300, "mrs x0, s3_5_c15_c3_0");
        check(0xd53df700, "mrs x0, s3_5_c15_c7_0");
        check(0xd53cf3a0, "mrs x0, s3_4_c15_c3_5");
        check(0xd53effe0, "mrs x0, s3_6_c15_c15_7");
        check(0xd538f000, "mrs x0, s3_0_c15_c0_0");
        check(0xd538ffa0, "mrs x0, s3_0_c15_c15_5");
        check(0xd53bfa00, "mrs x0, s3_3_c15_c10_0");
        check(0xd539f840, "mrs x0, s3_1_c15_c8_2");
        check(0xd5382100, "mrs x0, apiakeylo_el1");
        check(0xd5382120, "mrs x0, apiakeyhi_el1");
        check(0xd5382140, "mrs x0, apibkeylo_el1");
        check(0xd5382160, "mrs x0, apibkeyhi_el1");
        check(0xd5382200, "mrs x0, apdakeylo_el1");
        check(0xd5382220, "mrs x0, apdakeyhi_el1");
        check(0xd5382240, "mrs x0, apdbkeylo_el1");
        check(0xd5382260, "mrs x0, apdbkeyhi_el1");
        check(0xd5382300, "mrs x0, apgakeylo_el1");
        check(0xd5382320, "mrs x0, apgakeyhi_el1");
        check(0xd519f000, "msr s3_1_c15_c0_0, x0");
        check(0xd519f600, "msr s3_1_c15_c6_0, x0");
        check(0xd51afa00, "msr s3_2_c15_c10_0, x0");
        check(0xd5182100, "msr apiakeylo_el1, x0");
        check(0xd5182120, "msr apiakeyhi_el1, x0");
    }
}
