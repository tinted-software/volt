use crate::simd_struct::{Shape, StructDesc};
use crate::{fp, gxf, pauth, simd};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeError {
    UnsupportedInstruction,
}
impl core::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("unsupported AArch64 instruction")
    }
}
impl core::error::Error for DecodeError {}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Width {
    W32 = 32,
    X64 = 64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Size {
    Byte = 1,
    Half = 2,
    Word = 4,
    Double = 8,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum SignExtend {
    #[default]
    None,
    To32,
    To64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArithOp {
    Add,
    Sub,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryOp {
    Load,
    Store,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shift {
    Lsl,
    Lsr,
    Asr,
    Ror,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Condition {
    Eq,
    Ne,
    Cs,
    Cc,
    Mi,
    Pl,
    Vs,
    Vc,
    Hi,
    Ls,
    Ge,
    Lt,
    Gt,
    Le,
    Al,
    Nv,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogicOp {
    BitAnd,
    BitOr,
    BitXor,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VariableOp {
    Udiv,
    Sdiv,
    Lsl,
    Lsr,
    Asr,
    Ror,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Extend {
    Uxtb,
    Uxth,
    Uxtw,
    Uxtx,
    Sxtb,
    Sxth,
    Sxtw,
    Sxtx,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    Rbit,
    Rev16,
    Rev32,
    Rev,
    Clz,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BitfieldOp {
    Unsigned,
    Signed,
    Insert,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Wide {
    pub width: Width,
    pub rd: u8,
    pub shift: u8,
    pub immediate: u16,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Arithmetic {
    pub op: ArithOp,
    pub flags: bool,
    pub width: Width,
    pub rn: u8,
    pub rd: u8,
    /// The 12-bit immediate field as encoded, before `shifted` applies.
    pub immediate: u16,
    /// `lsl #12` on the immediate: `add x0, x1, #0, lsl #12` differs from `add x0, x1, #0`.
    pub shifted: bool,
}
impl Arithmetic {
    /// The operand value: the immediate field, shifted left by 12 when `shifted`.
    pub fn operand(self) -> u32 {
        u32::from(self.immediate) << if self.shifted { 12 } else { 0 }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Addressing {
    /// `[xn, #imm]` with the scaled unsigned 12-bit offset (`ldr`, `str`, `prfm`) or, for
    /// pairs, the scaled signed 7-bit one (`ldp`, `stp`, `ldnp`, `stnp`).
    Offset(i64),
    /// `[xn, #simm]` with the unscaled signed 9-bit offset: `ldur`, `stur`, `prfum`,
    /// `ldapur`, `stlur`. The offset may be representable as `Offset` too.
    Unscaled(i64),
    /// `[xn, #simm]` of `ldtr`/`sttr`: the unscaled signed 9-bit offset, but the
    /// access is checked as an unprivileged (EL0) one.
    Unprivileged(i64),
    /// `amount` is `None` when the S bit is clear and `Some(n)` when the offset register
    /// is shifted by `n`: for a byte access `Some(0)` is spelled `lsl #0`.
    Register {
        rm: u8,
        extend: Extend,
        amount: Option<u8>,
    },
    PreIndex(i64),
    PostIndex(i64),
}
/// The acquire/release class of an `Offset(0)` or `Unscaled` access that is neither a
/// plain load or store nor an exclusive. `ldar`, `stlr` and `ldapr` are the ARMv8
/// RCsc/RCpc pair; the `L` forms are the LORegion limited-ordering ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ordered {
    /// `ldar`: load-acquire.
    Acquire,
    /// `ldapr`, `ldapur`: load-acquire with only RCpc (processor consistent) ordering.
    AcquirePc,
    /// `ldlar`: limited-ordering load-acquire.
    LimitedAcquire,
    /// `stlr`, `stlur`: store-release.
    Release,
    /// `stllr`: limited-ordering store-release.
    LimitedRelease,
}
/// The acquire (`a`) and release (`l`) bits of an exclusive or LSE atomic, spelled
/// `ldaxr`, `stlxr`, `casal`, `ldaddl`, ...
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryOrder {
    pub acquire: bool,
    pub release: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Memory {
    pub op: MemoryOp,
    pub size: Size,
    pub rn: u8,
    pub rt: u8,
    pub addressing: Addressing,
    pub signed: SignExtend,
    /// `Some` for `ldar`, `stlr`, `ldapr`, `ldlar`, `stllr` and the unscaled `ldapur`,
    /// `stlur`, `ldapurs*`; the machines run them as the plain access.
    pub ordered: Option<Ordered>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pair {
    pub op: MemoryOp,
    pub size: Size,
    pub rn: u8,
    pub rt: u8,
    pub rt2: u8,
    pub addressing: Addressing,
    pub signed: SignExtend,
    /// `ldnp`, `stnp`: the non-temporal hint form (always `Addressing::Offset`).
    pub non_temporal: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Multiply {
    pub op: ArithOp,
    pub width: Width,
    pub rn: u8,
    pub rm: u8,
    pub ra: u8,
    pub rd: u8,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MultiplyLong {
    pub signed: bool,
    pub op: ArithOp,
    pub rn: u8,
    pub rm: u8,
    pub ra: u8,
    pub rd: u8,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MultiplyHigh {
    pub signed: bool,
    pub rn: u8,
    pub rm: u8,
    pub rd: u8,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimdLiteral {
    /// 4, 8 or 16.
    pub bytes: u8,
    pub rt: u8,
    pub offset: i64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Literal {
    pub size: Size,
    pub rt: u8,
    pub offset: i64,
    /// `ldrsw` (literal) sign-extends its word to 64 bits.
    pub signed: SignExtend,
}
/// `LDXR`, `LDAXR`, `STXR`, `STLXR` and their byte and halfword forms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Exclusive {
    pub op: MemoryOp,
    pub size: Size,
    pub rn: u8,
    pub rt: u8,
    pub rs: u8,
    /// `ldaxr` acquires, `stlxr` releases. The machines ignore it.
    pub order: MemoryOrder,
}

/// Operation of an LSE atomic or exclusive pair. The acquire and release bits are kept
/// in `Atomic::order`; the machines do not model them: each host read-modify-write is
/// sequentially consistent.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum AtomicKind {
    #[default]
    Add,
    Clr,
    Eor,
    Set,
    SMax,
    SMin,
    UMax,
    UMin,
    Swap,
    /// `CAS`, `CASB`, `CASH`: compare with `Rs`, store `Rt` on a match.
    Cas,
    /// `CASP`: compare `Rs`, `Rs+1` with the pair, store `Rt`, `Rt+1` on a match.
    Casp,
    /// `LDXP`: load a pair and set the exclusive monitor on it.
    LoadPair,
    /// `STXP`: store a pair if the monitor still holds and memory is unchanged.
    StorePair,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Atomic {
    pub kind: AtomicKind,
    pub size: Size,
    pub rs: u8,
    pub rn: u8,
    pub rt: u8,
    /// The second register of `LoadPair` and `StorePair`.
    pub rt2: u8,
    /// The `a` and `l` suffixes of `casal`, `ldaddal`, `swpa`, `ldaxp`, `stlxp`, ...
    pub order: MemoryOrder,
}
impl AtomicKind {
    /// Number of adjacent words the operation reads and writes.
    pub fn words(self) -> usize {
        if matches!(self, Self::Casp | Self::LoadPair | Self::StorePair) {
            2
        } else {
            1
        }
    }
    /// `LDXP` or `STXP`: the monitor-based pair accesses.
    pub fn is_exclusive_pair(self) -> bool {
        matches!(self, Self::LoadPair | Self::StorePair)
    }
    /// The words left in memory after the operation. `memory` holds the words read
    /// (the second one only matters to `Casp`). `operands` are `[Rs]` for the memory
    /// ops, `[Rs, Rt]` for `Cas`, and `[Rs, Rs+1, Rt, Rt+1]` for `Casp`. Results are
    /// masked to `width` bytes.
    pub fn apply(self, memory: [u64; 2], operands: [u64; 4], width: u8) -> [u64; 2] {
        let mask = if width >= 8 {
            u64::MAX
        } else {
            (1u64 << (width * 8)) - 1
        };
        let [old, second] = memory.map(|v| v & mask);
        let [a, b, c, d] = operands.map(|v| v & mask);
        match self {
            Self::Add => [old.wrapping_add(a) & mask, second],
            Self::Clr => [old & !a, second],
            Self::Eor => [old ^ a, second],
            Self::Set => [old | a, second],
            Self::SMax => [
                if sext_width(old, width) >= sext_width(a, width) {
                    old
                } else {
                    a
                },
                second,
            ],
            Self::SMin => [
                if sext_width(old, width) <= sext_width(a, width) {
                    old
                } else {
                    a
                },
                second,
            ],
            Self::UMax => [old.max(a), second],
            Self::UMin => [old.min(a), second],
            Self::Swap => [a, second],
            Self::Cas => [if old == a { b } else { old }, second],
            // Exclusive pairs are completed by `Cpu::exclusive_pair`, which owns the
            // monitor; memory is not combined with anything here.
            Self::LoadPair | Self::StorePair => [old, second],
            Self::Casp => {
                if old == a && second == b {
                    [c, d]
                } else {
                    [old, second]
                }
            }
        }
    }
}
/// Sign-extend the low `width` bytes of `value`.
fn sext_width(value: u64, width: u8) -> i64 {
    let drop = 64 - 8 * u32::from(width);
    ((value << drop) as i64) >> drop
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Address {
    pub page: bool,
    pub rd: u8,
    pub offset: i64,
}
/// The operands of a `SYS`-space system instruction (`DC`, `IC`, `TLBI`, `AT`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SysOp {
    pub op1: u8,
    pub crn: u8,
    pub crm: u8,
    pub op2: u8,
    pub rt: u8,
}
/// `PRFM`/`PRFUM` with a register-based address. `op` is the prefetch operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Prefetch {
    pub op: u8,
    pub rn: u8,
    pub addressing: Addressing,
}
/// `RPRFM <rprfop>, Xm, [Xn]`: a range prefetch. It is a hint and runs as a no-op.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RangePrefetch {
    /// The six-bit range prefetch operation.
    pub op: u8,
    pub rm: u8,
    pub rn: u8,
}
/// `PRFM` (literal): a prefetch from a PC-relative address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefetchLiteral {
    pub op: u8,
    pub offset: i64,
}
/// How a modified-immediate instruction combines its immediate with `rd`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImmCombine {
    /// `movi`, `fmov`: replace the register.
    Set,
    /// `mvni`: replace the register with the (already inverted) pattern.
    SetInverted,
    /// `orr`: `rd |= imm`.
    Or,
    /// `bic`: `rd &= !imm`.
    AndNot,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimdImmediate {
    pub rd: u8,
    /// The encoded 8-bit immediate (`abc:defgh`).
    pub imm8: u8,
    /// The lane layout and shift the encoding (`cmode`, `op`) selects.
    pub form: ImmForm,
    /// The expanded immediate: the low and high 64 bits (the high half is zero for
    /// a 64-bit `q == false` form, which also clears the register's upper half).
    /// Inverted for `mvni`.
    pub low: u64,
    pub high: u64,
    pub combine: ImmCombine,
    pub q: bool,
}
/// The lane layout of a modified immediate (`AdvSIMDExpandImm` `cmode` classes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImmForm {
    /// `.8b`/`.16b`: `imm8` in every byte.
    Byte,
    /// `.4h`/`.8h`: `imm8 << shift` with `shift` 0 or 8 (`lsl`).
    Half { shift: u8 },
    /// `.2s`/`.4s`: `imm8 << shift` with `shift` 0, 8, 16 or 24 (`lsl`).
    Word { shift: u8 },
    /// `.2s`/`.4s`: `imm8 << shift` with the vacated low bits set, `shift` 8 or 16
    /// (`msl`).
    WordMask { shift: u8 },
    /// `movi d`/`.2d`: each bit of `imm8` stands for a 0x00 or 0xff byte.
    Double,
    /// `fmov .4h`/`.8h`.
    FmovHalf,
    /// `fmov .2s`/`.4s`.
    FmovSingle,
    /// `fmov .2d`.
    FmovDouble,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimdPair {
    pub op: MemoryOp,
    pub bytes: u8,
    pub rn: u8,
    pub rt: u8,
    pub rt2: u8,
    pub addressing: Addressing,
    /// `ldnp`, `stnp`: the non-temporal hint form (always `Addressing::Offset`).
    pub non_temporal: bool,
}
/// How a [`SimdMemory`] is spelled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimdAccess {
    /// `ldr`/`str` (`ldur`/`stur` with an unscaled offset) of a `b`, `h`, `s`, `d` or `q`
    /// register.
    Register,
    /// One-register `ld1`/`st1` of a whole `.8b`..`.2d` vector whose elements are `esize`
    /// bytes. It moves the same bytes as the `d`/`q` register form.
    Ld1 { esize: u8 },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimdMemory {
    pub op: MemoryOp,
    pub bytes: u8,
    pub rn: u8,
    pub rt: u8,
    pub addressing: Addressing,
    pub access: SimdAccess,
}
/// How a SIMD structure access updates its base register afterwards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StructPost {
    None,
    /// Advance by the bytes transferred.
    Immediate,
    Register(u8),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimdStruct {
    pub desc: StructDesc,
    pub rn: u8,
    pub post: StructPost,
}
/// `at s1e{1,0}{r,w}, Xt`: a stage-1 walk whose result lands in `PAR_EL1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AddressTranslate {
    pub rt: u8,
    pub write: bool,
    /// Translate as EL0 (`s1e0*`) rather than EL1.
    pub user: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimdCompareZero {
    pub rd: u8,
    pub rn: u8,
    /// Log2 of the element width in bytes.
    pub size: u8,
    pub q: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimdExt {
    pub rd: u8,
    pub rn: u8,
    pub rm: u8,
    /// The byte index into the `rm:rn` pair, below 8 when `q` is clear.
    pub imm: u8,
    /// The 128-bit form (`.16b`); clear for `.8b`.
    pub q: bool,
}
/// `dup` from an element of a vector register: replicate lane `index` of
/// `rn` across every lane of the destination.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimdDupElement {
    pub rd: u8,
    pub rn: u8,
    /// Log2 of the element width in bytes.
    pub esize: u8,
    pub index: u8,
    pub q: bool,
}
/// `mov <Vd>.<T>[index], <R>`: insert a general register's bits into one lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimdInsert {
    pub rd: u8,
    pub rn: u8,
    /// Log2 of the element width in bytes.
    pub esize: u8,
    pub index: u8,
}
/// `mov Vd.T[dst], Vn.T[src]`: copy one lane, keeping the rest of the destination.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimdInsertElement {
    pub rd: u8,
    pub rn: u8,
    /// Log2 of the element width in bytes.
    pub esize: u8,
    pub dst: u8,
    pub src: u8,
}
/// `mov <R>, <Vn>.<T>[index]` / `umov`: extract one lane into a general register.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimdExtract {
    pub rd: u8,
    pub rn: u8,
    /// Log2 of the element width in bytes.
    pub esize: u8,
    pub index: u8,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimdFmov {
    pub rd: u8,
    pub rn: u8,
    /// True for GPR-to-FPR, false for FPR-to-GPR.
    pub to_fp: bool,
    /// True for 64-bit (`x`/`d`), false for 32-bit (`w`/`s`).
    pub double: bool,
    /// The upper doubleword of the vector (`Vd.D[1]`); only with `double`. A move
    /// to it keeps the lower doubleword.
    pub upper: bool,
}
/// Three-registers-same integer vector operation (`U`, `size` and `opcode`
/// fields verified against `llvm-mc` encodings).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimdAluOp {
    Add,
    Sub,
    Mul,
    Mla,
    Mls,
    And,
    Bic,
    Orr,
    Orn,
    Eor,
    Bsl,
    Bit,
    Bif,
    CmEq,
    CmGt,
    CmGe,
    CmHi,
    CmHs,
    CmTst,
    UMin,
    UMax,
    SMin,
    SMax,
    UAbd,
    SAbd,
    SQAdd,
    UQAdd,
    SQSub,
    UQSub,
    SHAdd,
    UHAdd,
    SHSub,
    UHSub,
    SRHAdd,
    URHAdd,
    SSHl,
    USHl,
    SQShl,
    UQShl,
    SQRShl,
    UQRShl,
    /// Horizontal add of all lanes (only the low byte/half/word of the result
    /// is meaningful; the rest of the destination is zeroed).
    AddV,
    /// Minimum (unsigned) across all lanes; result layout matches `AddV`.
    UMinV,
    /// Maximum (unsigned) across all lanes; result layout matches `AddV`.
    UMaxV,
    /// Sum of all lanes into a lane twice as wide (`uaddlv`).
    UAddLV,
    /// Adjacent-lane pair operations between two registers.
    UMaxP,
    SMaxP,
    UMinP,
    SMinP,
    AddP,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimdAlu {
    pub op: SimdAluOp,
    pub rd: u8,
    pub rn: u8,
    pub rm: u8,
    /// Log2 of the element width in bytes.
    pub size: u8,
    pub q: bool,
}
/// Table vector lookup (`tbl`/`tbx`): byte indices select from one to four
/// consecutive table registers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimdTable {
    pub rd: u8,
    /// First table register.
    pub rn: u8,
    /// Index register.
    pub rm: u8,
    /// Number of table registers minus one.
    pub len: u8,
    /// True for `tbx`, whose out-of-range lanes keep the destination bytes.
    pub extend: bool,
    pub q: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SystemRegister {
    SpEl0,
    SpEl1,
    SpEl2,
    SctlrEl2,
    TcrEl2,
    Ttbr0El2,
    Ttbr1El2,
    MairEl2,
    VbarEl2,
    ElrEl2,
    SpsrEl2,
    EsrEl2,
    FarEl2,
    HcrEl2,
    TpidrEl2,
    CntvoffEl2,
    CurrentEl,
    SpsrEl1,
    ElrEl1,
    EsrEl1,
    FarEl1,
    ParEl1,
    VbarEl1,
    CpacrEl1,
    SctlrEl1,
    Ttbr0El1,
    Ttbr1El1,
    TcrEl1,
    MairEl1,
    TpidrEl0,
    TpidrroEl0,
    CntvctEl0,
    CntvCtlEl0,
    CntvCvalEl0,
    CntvTvalEl0,
    TpidrEl1,
    ContextidrEl1,
    MdscrEl1,
    CntkctlEl1,
    OslarEl1,
    OsdlrEl1,
    OslsrEl1,
    Fpcr,
    Fpsr,
    Daif,
    MidrEl1,
    MpidrEl1,
    RevidrEl1,
    CtrEl0,
    DczidEl0,
    CntfrqEl0,
    ClidrEl1,
    /// Geometry of the cache selected by `CsselrEl1` (see `identification`).
    CcsidrEl1,
    CsselrEl1,
    IdAa64pfr0El1,
    IdAa64pfr1El1,
    IdAa64pfr2El1,
    IdAa64zfr0El1,
    IdAa64smfr0El1,
    IdAa64fpfr0El1,
    IdAa64isar3El1,
    AidrEl1,
    IdAa64dfr0El1,
    IdAa64dfr1El1,
    IdAa64isar0El1,
    IdAa64isar1El1,
    IdAa64isar2El1,
    IdAa64mmfr0El1,
    IdAa64mmfr1El1,
    IdAa64mmfr2El1,
    IdAa64mmfr3El1,
    IdAa64mmfr4El1,
    DaifSet,
    DaifClear,
    /// `mrs`/`msr SPSel, Xt`.
    SpSel,
    /// `msr SPSel, #imm`.
    SpSelSet,
    /// `mrs`/`msr PAN, Xt`: PSTATE.PAN, stored but not enforced by the page walk.
    Pan,
    /// `msr PAN, #imm`.
    PanSet,
    /// Pending-interrupt status (bit 7 IRQ, bit 6 FIQ), kept current by the machine.
    IsrEl1,
    /// GICv3 CPU interface registers, held as plain state: the redistributor and
    /// the machine consult them when deciding whether the timer FIQ is delivered.
    IccSreEl1,
    IccPmrEl1,
    IccBpr0El1,
    IccCtlrEl1,
    IccIgrpen0El1,
    /// Group 1 enable and binary point: the group the firmware and Linux use for IRQs.
    IccIgrpen1El1,
    IccBpr1El1,
    /// Group 1 acknowledge and end of interrupt. No group 1 interrupt is delivered by this
    /// model, so an acknowledge reads spurious and an end of interrupt has nothing to end.
    IccIar1El1,
    IccEoir1El1,
    /// Stored, but the physical timer never fires.
    CntpCtlEl0,
    /// The physical counter (the virtual count, as CNTVOFF_EL2 is zero without EL2).
    CntpctEl0,
    /// Physical timer compare value and its 32-bit view, sharing the counter. It never fires.
    CntpTvalEl0,
    CntpCvalEl0,
    ActlrEl1,
    Nzcv,
    /// Read-as-zero, write-ignored debug register `DBG{B,W}{V,C}R<crm>_EL1` (`op0` 2, `op2` 4 to 7:
    /// `DBGBVR`, `DBGBCR`, `DBGWVR`, `DBGWCR`).
    RazWi {
        crm: u8,
        op2: u8,
    },
    /// Read-as-zero, write-ignored: Apple IMP-DEF status registers (`op0` 3).
    AppleZero {
        op1: u8,
        crm: u8,
        op2: u8,
    },
    /// Apple IMP-DEF configuration register the guest reads back, stored in
    /// `System::apple[slot]`.
    Apple(u8),
    /// Apple IMP-DEF register of the banks the SPTM monitor configures (`op1` 4 to 6, and the
    /// `op1` 0, 1 and 3 registers in `MONITOR_LOW_BANKS`), which
    /// the SPTM monitor reads back; stored in `Cpu::apple_bank`.
    AppleBank(u16),
    /// PAuth key register `APxAKey{Lo,Hi}_EL1`, stored in `System::pac_keys[slot]`.
    PacKey(u8),
}
impl SystemRegister {
    pub fn read_only(self) -> bool {
        use SystemRegister::*;
        matches!(
            self,
            CurrentEl
                | MidrEl1
                | MpidrEl1
                | RevidrEl1
                | CtrEl0
                | DczidEl0
                | CntfrqEl0
                | ClidrEl1
                | CcsidrEl1
                | IsrEl1
                | IdAa64pfr0El1
                | IdAa64pfr1El1
                | IdAa64pfr2El1
                | IdAa64zfr0El1
                | IdAa64smfr0El1
                | IdAa64fpfr0El1
                | IdAa64isar3El1
                | AidrEl1
                | IdAa64dfr0El1
                | IdAa64dfr1El1
                | IdAa64isar0El1
                | IdAa64isar1El1
                | IdAa64isar2El1
                | IdAa64mmfr0El1
                | IdAa64mmfr1El1
                | IdAa64mmfr2El1
                | IdAa64mmfr3El1
                | IdAa64mmfr4El1
        )
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct System {
    pub read: bool,
    pub register: SystemRegister,
    pub rt: u8,
    /// The `#imm` of `msr <pstatefield>, #imm` (`DaifSet`, `DaifClear`, `SpSelSet`,
    /// `PanSet`); zero for the register forms.
    pub immediate: u8,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArithmeticReg {
    pub op: ArithOp,
    pub flags: bool,
    pub width: Width,
    pub rn: u8,
    pub rm: u8,
    pub rd: u8,
    pub shift: Shift,
    pub amount: u8,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Logic {
    pub op: LogicOp,
    pub flags: bool,
    pub width: Width,
    pub rn: u8,
    pub rm: u8,
    pub rd: u8,
    pub shift: Shift,
    pub amount: u8,
    pub invert: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LogicImm {
    pub op: LogicOp,
    pub flags: bool,
    pub width: Width,
    pub rn: u8,
    pub rd: u8,
    pub immediate: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Test {
    pub rt: u8,
    pub bit_index: Option<u8>,
    pub negate: bool,
    pub width: Width,
    pub offset: i64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Call {
    pub target: i64,
    pub link: bool,
}
/// Which register branch an [`Indirect`] is. `RET` is architecturally a `BR` with a
/// return hint, but analysis (and branch prediction) must tell them apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndirectKind {
    /// `BR Xn`.
    Jump,
    /// `BLR Xn`: also writes the return address to `X30`.
    Call,
    /// `RET {Xn}`.
    Return,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Indirect {
    pub rn: u8,
    pub kind: IndirectKind,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Variable {
    pub op: VariableOp,
    pub width: Width,
    pub rn: u8,
    pub rm: u8,
    pub rd: u8,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompareOperand {
    Register(u8),
    Immediate(u8),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CondCompare {
    pub op: ArithOp,
    pub width: Width,
    pub rn: u8,
    pub operand: CompareOperand,
    pub cond: Condition,
    pub nzcv: u8,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AddCarry {
    pub op: ArithOp,
    pub flags: bool,
    pub width: Width,
    pub rn: u8,
    pub rm: u8,
    pub rd: u8,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArithmeticExtended {
    pub op: ArithOp,
    pub flags: bool,
    pub width: Width,
    pub rn: u8,
    pub rm: u8,
    pub rd: u8,
    pub extend: Extend,
    pub amount: u8,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Unary {
    pub op: UnaryOp,
    pub width: Width,
    pub rn: u8,
    pub rd: u8,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Extract {
    pub width: Width,
    pub rn: u8,
    pub rm: u8,
    pub rd: u8,
    pub lsb: u8,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bitfield {
    pub op: BitfieldOp,
    pub extract: bool,
    pub width: Width,
    pub rn: u8,
    pub rd: u8,
    pub lsb: u8,
    pub len: u8,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BranchCondition {
    pub cond: Condition,
    pub offset: i64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Select {
    pub cond: Condition,
    pub width: Width,
    pub rn: u8,
    pub rm: u8,
    pub rd: u8,
    pub opc: u8,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Instruction {
    Movz(Wide),
    Movn(Wide),
    Movk(Wide),
    ArithImm(Arithmetic),
    ArithReg(ArithmeticReg),
    ArithExt(ArithmeticExtended),
    LogicReg(Logic),
    LogicImm(LogicImm),
    TestBranch(Test),
    Call(Call),
    Indirect(Indirect),
    Memory(Memory),
    Pair(Pair),
    Mul(Multiply),
    MulLong(MultiplyLong),
    MulHigh(MultiplyHigh),
    Literal(Literal),
    SimdLiteral(SimdLiteral),
    Exclusive(Exclusive),
    Atomic(Atomic),
    Clrex(u8),
    Adr(Address),
    System(System),
    DcZva(u8),
    AddressTranslate(AddressTranslate),
    Eret,
    CacheOp(SysOp),
    Tlbi(SysOp),
    Nop,
    /// A hint-space `HINT #n` other than `nop` and the named hints (`bti`, `esb`,
    /// `csdb`, ...). Executes as `nop`.
    Hint(u8),
    Prefetch(Prefetch),
    PrefetchLiteral(PrefetchLiteral),
    RangePrefetch(RangePrefetch),
    Dsb(u8),
    Dmb(u8),
    Isb(u8),
    Sev,
    Sevl,
    Wfe,
    Yield,
    Brk(u16),
    B(i64),
    BCond(BranchCondition),
    Csel(Select),
    Variable(Variable),
    CondCompare(CondCompare),
    ArithCarry(AddCarry),
    Unary(Unary),
    Bitfield(Bitfield),
    Extract(Extract),
    SimdImm(SimdImmediate),
    SimdPair(SimdPair),
    SimdMemory(SimdMemory),
    SimdStruct(SimdStruct),
    SimdCompareZero(SimdCompareZero),
    SimdFmov(SimdFmov),
    SimdDup {
        rd: u8,
        rn: u8,
        esize: u8,
        q: bool,
    },
    SimdDupElement(SimdDupElement),
    SimdInsert(SimdInsert),
    SimdInsertElement(SimdInsertElement),
    SimdExtract(SimdExtract),
    SimdExt(SimdExt),
    SimdAlu(SimdAlu),
    SimdTable(SimdTable),
    Svc(u16),
    Psci,
    Wfi,
    /// Scalar floating point; see [`fp`].
    Fp(fp::Op),
    /// Integer Advanced SIMD instructions without a structured form; see [`simd`].
    Simd(simd::Op),
    /// Pointer authentication; see [`pauth`].
    Pauth(pauth::Op),
    /// Guarded execution; see [`gxf`].
    Gxf(gxf::Op),
}
/// Whether `word` is architecturally UNDEFINED on this CPU, as opposed to an
/// instruction the emulator does not implement. Execution takes an Undefined
/// Instruction exception (XNU's `TRAP_DEBUGGER` word `0xe7ffdeff` relies on it). These
/// are the permanently-undefined `udf` space and the unallocated top-level groups, plus
/// the SVE group: the CPU does not advertise SVE.
pub fn is_undefined(word: u32) -> bool {
    match (word >> 25) & 0xf {
        // `udf #imm16` is the reserved group with bits 31:16 clear.
        0b0000 => word >> 16 == 0,
        0b0001 | 0b0010 | 0b0011 => true,
        _ => false,
    }
}
fn reg(word: u32, shift: u32) -> u8 {
    ((word >> shift) & 31) as u8
}
fn width(word: u32) -> Width {
    if word & 0x8000_0000 != 0 {
        Width::X64
    } else {
        Width::W32
    }
}
fn arith_op(word: u32) -> ArithOp {
    if word & 0x4000_0000 != 0 {
        ArithOp::Sub
    } else {
        ArithOp::Add
    }
}
fn size(word: u32) -> Size {
    [Size::Byte, Size::Half, Size::Word, Size::Double][(word >> 30) as usize]
}
fn condition(n: u32) -> Condition {
    [
        Condition::Eq,
        Condition::Ne,
        Condition::Cs,
        Condition::Cc,
        Condition::Mi,
        Condition::Pl,
        Condition::Vs,
        Condition::Vc,
        Condition::Hi,
        Condition::Ls,
        Condition::Ge,
        Condition::Lt,
        Condition::Gt,
        Condition::Le,
        Condition::Al,
        Condition::Nv,
    ][(n & 15) as usize]
}
fn extension(n: u32) -> Extend {
    [
        Extend::Uxtb,
        Extend::Uxth,
        Extend::Uxtw,
        Extend::Uxtx,
        Extend::Sxtb,
        Extend::Sxth,
        Extend::Sxtw,
        Extend::Sxtx,
    ][(n & 7) as usize]
}
/// The `[xn, rm{, ext{ #amount}}]` address of a register-offset load, store or prefetch.
/// `scale` is log2 of the access size, the amount applied when the S bit is set. Options
/// 0, 1, 4 and 5 (the byte and halfword extensions) and a non-zero bits 11:10 are
/// unallocated.
fn register_offset(word: u32, scale: u32) -> Result<Addressing, DecodeError> {
    let option = (word >> 13) & 7;
    if option & 2 == 0 || word & 0x0c00 != 0x0800 {
        return Err(DecodeError::UnsupportedInstruction);
    }
    Ok(Addressing::Register {
        rm: reg(word, 16),
        extend: extension(option),
        amount: (word & 0x1000 != 0).then_some(scale as u8),
    })
}
/// The address of a load/store-register word without the unsigned-offset or register
/// form: bits 11:10 select unscaled, post-index, unprivileged or pre-index with the
/// signed 9-bit offset. The SIMD&FP forms have no unprivileged variant.
fn immediate_offset(word: u32, vector: bool) -> Result<Addressing, DecodeError> {
    let offset = sign_extend(word >> 12, 9);
    match (word >> 10) & 3 {
        0 => Ok(Addressing::Unscaled(offset)),
        1 => Ok(Addressing::PostIndex(offset)),
        2 if !vector => Ok(Addressing::Unprivileged(offset)),
        3 => Ok(Addressing::PreIndex(offset)),
        _ => Err(DecodeError::UnsupportedInstruction),
    }
}
/// The sign extension of a general-register load or store: `opc` is bits 23:22 and
/// `size` the access size. `opc` 0 stores and 1 loads zero-extended; 2 is `ldrs[bhw] Xt`
/// and 3 is `ldrs[bh] Wt`.
fn load_extension(opc: u32, size: Size) -> Result<SignExtend, DecodeError> {
    match opc {
        0 | 1 => Ok(SignExtend::None),
        2 if size != Size::Double => Ok(SignExtend::To64),
        3 if size == Size::Byte || size == Size::Half => Ok(SignExtend::To32),
        _ => Err(DecodeError::UnsupportedInstruction),
    }
}
fn sign_extend(n: u32, bits: u32) -> i64 {
    ((n << (32 - bits)) as i32 >> (32 - bits)) as i64
}
fn logic_op(word: u32) -> LogicOp {
    match (word >> 29) & 3 {
        0 | 3 => LogicOp::BitAnd,
        1 => LogicOp::BitOr,
        _ => LogicOp::BitXor,
    }
}
fn bitmask(word: u32, width: Width) -> Result<u64, DecodeError> {
    let n = (word >> 22) & 1;
    let immr = (word >> 16) & 63;
    let imms = (word >> 10) & 63;
    let combined = (n << 6) | ((!imms) & 63);
    if combined == 0 || (width == Width::W32 && n != 0) {
        return Err(DecodeError::UnsupportedInstruction);
    }
    let esize = 1u32 << (31 - combined.leading_zeros());
    let levels = esize - 1;
    if imms & levels == levels {
        return Err(DecodeError::UnsupportedInstruction);
    }
    let run = imms & levels;
    let rotation = immr & levels;
    let ones = (1u64 << (run + 1)) - 1;
    let within = if esize == 64 {
        u64::MAX
    } else {
        (1u64 << esize) - 1
    };
    let element = if rotation == 0 {
        ones
    } else {
        ((ones >> rotation) | (ones << (esize - rotation))) & within
    };
    let mut immediate = 0;
    let mut offset = 0;
    while offset < width as u32 {
        immediate |= element << offset;
        offset += esize;
    }
    Ok(immediate)
}
/// Decode an Advanced SIMD structure load/store word (`word & 0xbe00_0000 ==
/// 0x0c00_0000`) into its descriptor and post-index.
fn simd_struct(word: u32) -> Result<(StructDesc, StructPost), DecodeError> {
    use DecodeError::UnsupportedInstruction as Bad;
    let q = word & 0x4000_0000 != 0;
    let rm = reg(word, 16);
    let size = ((word >> 10) & 3) as u8;
    let post = if word & 0x0080_0000 == 0 {
        // Without post-index the Rm field is zero.
        if rm != 0 {
            return Err(Bad);
        }
        StructPost::None
    } else if rm == 31 {
        StructPost::Immediate
    } else {
        StructPost::Register(rm)
    };
    let mut desc = StructDesc {
        store: word & 0x0040_0000 == 0,
        q,
        rt: reg(word, 0),
        esize: 1 << size,
        ..StructDesc::default()
    };
    if word & 0x0100_0000 == 0 {
        if word & 0x0020_0000 != 0 {
            return Err(Bad);
        }
        (desc.registers, desc.interleaved) = match (word >> 12) & 0xf {
            0b0000 => (4, true),
            0b0100 => (3, true),
            0b1000 => (2, true),
            0b0010 => (4, false),
            0b0110 => (3, false),
            0b1010 => (2, false),
            0b0111 => (1, false),
            _ => return Err(Bad),
        };
        // `.1d` exists only for LD1/ST1.
        if desc.interleaved && !q && size == 3 {
            return Err(Bad);
        }
    } else {
        // Bits 15:13 give the element scale (and, with R, the structure count);
        // the lane index is spread over Q, S and size.
        let opcode = (word >> 13) & 7;
        let s = u8::from(word & 0x1000 != 0);
        let qbit = u8::from(q);
        desc.registers = ((opcode & 1) << 1 | u32::from(word & 0x0020_0000 != 0)) as u8 + 1;
        let (shape, esize, lane) = match opcode >> 1 {
            0 => (Shape::Lane, 1, qbit << 3 | s << 2 | size),
            1 if size & 1 == 0 => (Shape::Lane, 2, qbit << 2 | s << 1 | size >> 1),
            2 if size == 0 => (Shape::Lane, 4, qbit << 1 | s),
            2 if size == 1 && s == 0 => (Shape::Lane, 8, qbit),
            // Replication is a load, and has no lane or S bit.
            3 if !desc.store && s == 0 => (Shape::Replicate, 1 << size, 0),
            _ => return Err(Bad),
        };
        (desc.shape, desc.esize, desc.lane) = (shape, esize, lane);
    }
    Ok((desc, post))
}
/// Number of Apple IMP-DEF configuration registers `System::apple` holds.
pub const APPLE_SLOTS: usize = 21;
/// Number of `S3_<op1>_c15` registers `Cpu::apple_bank` holds: eight banks of
/// `crm` x `op2` (16 x 8).
pub const APPLE_BANK_SLOTS: usize = 8 * 128;

/// The `Cpu::apple_bank` slot of `S3_<op1>_c15_c<crm>_<op2>`.
pub const fn apple_bank_slot(op1: u32, crm: u32, op2: u32) -> usize {
    ((op1 as usize) << 7) | ((crm as usize) << 3) | op2 as usize
}

/// The `S3_{0,1,3}_c15` registers (`op1`, `crm`, `op2`) the SPTM monitor's code accesses
/// that the kernel-side table above does not name, found by scanning its `mrs`/`msr`
/// words. Their roles are not identified; they are stored so the monitor reads back what
/// it wrote. Other registers in those banks stay unsupported.
const MONITOR_LOW_BANKS: [(u32, u32, u32); 25] = [
    (0, 0, 0),
    (0, 1, 0),
    (0, 1, 2),
    (0, 1, 3),
    (0, 2, 0),
    (0, 2, 1),
    (0, 3, 0),
    (0, 4, 0),
    (0, 4, 1),
    (0, 5, 0),
    (0, 6, 0),
    (0, 7, 0),
    (0, 9, 0),
    (0, 9, 1),
    (0, 10, 1),
    (0, 11, 0),
    (0, 11, 2),
    (0, 14, 0),
    (0, 15, 0),
    (0, 15, 2),
    (0, 15, 5),
    (1, 8, 2),
    (3, 8, 0),
    (3, 9, 0),
    (3, 10, 0),
];

/// The `S3_*` implementation-defined registers the M1 kernel and PMU driver touch
/// at EL1 (`arch/arm64/include/asm/apple_m1_pmu.h`, `irq-apple-aic.c`). Configuration
/// is stored so the guest reads back what it wrote. Status registers (IPI pending,
/// PMU interrupt active, PMU state) read as zero: this model raises no fast IPIs and
/// no PMU interrupts. Anything else falls through to the unsupported path.
fn apple_register(op0: u32, op1: u32, crn: u32, crm: u32, op2: u32) -> Option<SystemRegister> {
    use SystemRegister::*;
    if op0 != 3 || crn != 15 {
        return None;
    }
    let zero = AppleZero {
        op1: op1 as u8,
        crm: crm as u8,
        op2: op2 as u8,
    };
    Some(match (op1, crm, op2) {
        (1, 0, 0) => Apple(0),
        (1, 1, 0) => Apple(1),
        (1, 2, 0) => Apple(2),
        (1, 3, 0) => Apple(3),
        (1, 4, 0) => Apple(4),
        (1, 5, 0) => Apple(5),
        (1, 6, 0) => Apple(6),
        (1, 13, 0) => zero,
        (2, 0, 0) => Apple(7),
        (2, 1, 0) => Apple(8),
        (2, 2, 0) => Apple(9),
        (2, 3, 0) => Apple(10),
        (2, 4, 0) => Apple(11),
        (2, 5, 0) => Apple(12),
        (2, 6, 0) => Apple(13),
        (2, 7, 0) => Apple(14),
        (2, 9, 0) => Apple(15),
        (2, 10, 0) => Apple(16),
        (5, 0, 0) => Apple(17),
        (5, 0, 1) => Apple(18),
        (5, 1, 1) => zero,
        (5, 3, 1) => Apple(19),
        (7, 0, 4) => Apple(20),
        (7, 6, 4) => zero,
        // The S3_5_c15 state registers the kernel reads during early boot. None are named
        // in the XNU source tree that is available here, so they read as zero and writes
        // are dropped until their roles are identified.
        (5, 2..=7, 0) => zero,
        // The EL2 and guarded-level banks the SPTM monitor configures: stored, so a
        // read-modify-write or a read-back sees what was written.
        (4..=6, _, _) => AppleBank(apple_bank_slot(op1, crm, op2) as u16),
        _ if MONITOR_LOW_BANKS.contains(&(op1, crm, op2)) => {
            AppleBank(apple_bank_slot(op1, crm, op2) as u16)
        }
        _ => return None,
    })
}

/// The PAuth key registers `S3_0_C2_C{1,2,3}_*`: APIA, APIB, APDA, APDB and APGA, low
/// word first (`System::pac_keys`).
fn pac_key_register(op0: u32, op1: u32, crn: u32, crm: u32, op2: u32) -> Option<SystemRegister> {
    if op0 != 3 || op1 != 0 || crn != 2 {
        return None;
    }
    let slot = match (crm, op2) {
        (1, 0) => 0,
        (1, 1) => 1,
        (1, 2) => 2,
        (1, 3) => 3,
        (2, 0) => 4,
        (2, 1) => 5,
        (2, 2) => 6,
        (2, 3) => 7,
        (3, 0) => 8,
        (3, 1) => 9,
        _ => return None,
    };
    Some(SystemRegister::PacKey(slot))
}

fn system_register(
    op0: u32,
    op1: u32,
    crn: u32,
    crm: u32,
    op2: u32,
) -> Result<SystemRegister, DecodeError> {
    use SystemRegister::*;
    if op0 == 2 && op1 == 0 && crn == 0 && (4..=7).contains(&op2) {
        return Ok(RazWi {
            crm: crm as u8,
            op2: op2 as u8,
        });
    }
    if let Some(register) = apple_register(op0, op1, crn, crm, op2) {
        return Ok(register);
    }
    if let Some(register) = pac_key_register(op0, op1, crn, crm, op2) {
        return Ok(register);
    }
    if op1 == 0 && crn == 4 && op2 == 5 {
        return Ok(SpSelSet);
    }
    if op1 == 0 && crn == 4 && op2 == 4 {
        return Ok(PanSet);
    }
    if op1 == 3 && crn == 4 {
        if op2 == 6 {
            return Ok(DaifSet);
        }
        if op2 == 7 {
            return Ok(DaifClear);
        }
    }
    Ok(match op1 << 12 | crn << 8 | crm << 4 | op2 {
        0x0410 => SpEl0,
        0x4410 => SpEl1,
        0x6410 => SpEl2,
        0x4100 => SctlrEl2,
        0x4202 => TcrEl2,
        0x4200 => Ttbr0El2,
        0x4201 => Ttbr1El2,
        0x4a20 => MairEl2,
        0x4c00 => VbarEl2,
        0x4401 => ElrEl2,
        0x4400 => SpsrEl2,
        0x4520 => EsrEl2,
        0x4600 => FarEl2,
        0x4110 => HcrEl2,
        0x4d02 => TpidrEl2,
        0x4e03 => CntvoffEl2,
        0x0422 => CurrentEl,
        0x0400 => SpsrEl1,
        0x0401 => ElrEl1,
        0x0520 => EsrEl1,
        0x0600 => FarEl1,
        0x0740 => ParEl1,
        0x0c00 => VbarEl1,
        0x0102 => CpacrEl1,
        0x0100 => SctlrEl1,
        0x0200 => Ttbr0El1,
        0x0201 => Ttbr1El1,
        0x0202 => TcrEl1,
        0x0a20 => MairEl1,
        0x3d02 => TpidrEl0,
        0x3d03 => TpidrroEl0,
        0x3e02 => CntvctEl0,
        0x3e31 => CntvCtlEl0,
        0x3e32 => CntvCvalEl0,
        0x3e30 => CntvTvalEl0,
        0x0d04 => TpidrEl1,
        0x0d01 => ContextidrEl1,
        0x0022 => MdscrEl1,
        0x0104 => OslarEl1,
        0x0134 => OsdlrEl1,
        0x0114 => OslsrEl1,
        0x0e10 => CntkctlEl1,
        0x3440 => Fpcr,
        0x3441 => Fpsr,
        0x3421 => Daif,
        0x0420 => SpSel,
        0x0423 => Pan,
        0x0c10 => IsrEl1,
        0x0cc5 => IccSreEl1,
        0x0460 => IccPmrEl1,
        0x0c83 => IccBpr0El1,
        0x0cc4 => IccCtlrEl1,
        0x0cc6 => IccIgrpen0El1,
        0x0cc7 => IccIgrpen1El1,
        0x0cc3 => IccBpr1El1,
        0x3e21 => CntpCtlEl0,
        0x3e01 => CntpctEl0,
        0x3e20 => CntpTvalEl0,
        0x3e22 => CntpCvalEl0,
        0x0cc0 => IccIar1El1,
        0x0cc1 => IccEoir1El1,
        0x0101 => ActlrEl1,
        0x0000 => MidrEl1,
        0x0005 => MpidrEl1,
        0x0006 => RevidrEl1,
        0x3001 => CtrEl0,
        0x3007 => DczidEl0,
        0x3e00 => CntfrqEl0,
        0x1001 => ClidrEl1,
        0x1000 => CcsidrEl1,
        0x2000 => CsselrEl1,
        0x0040 => IdAa64pfr0El1,
        0x0041 => IdAa64pfr1El1,
        0x0042 => IdAa64pfr2El1,
        0x0044 => IdAa64zfr0El1,
        0x0045 => IdAa64smfr0El1,
        0x0047 => IdAa64fpfr0El1,
        0x0063 => IdAa64isar3El1,
        0x1007 => AidrEl1,
        0x0050 => IdAa64dfr0El1,
        0x0051 => IdAa64dfr1El1,
        0x0060 => IdAa64isar0El1,
        0x0061 => IdAa64isar1El1,
        0x0062 => IdAa64isar2El1,
        0x0070 => IdAa64mmfr0El1,
        0x0071 => IdAa64mmfr1El1,
        0x0072 => IdAa64mmfr2El1,
        0x0073 => IdAa64mmfr3El1,
        0x0074 => IdAa64mmfr4El1,
        0x3420 => Nzcv,
        _ => return Err(DecodeError::UnsupportedInstruction),
    })
}

/// The 16-bit pattern of the 8-bit floating-point immediate `imm8` (`VFPExpandImm`
/// for half precision), as the vector `fmov` takes it.
pub fn expand_half_immediate(imm8: u64) -> u64 {
    let sign = imm8 >> 7;
    let b = (imm8 >> 6) & 1;
    (sign << 15) | ((b ^ 1) << 14) | (b * 0b11 << 12) | ((imm8 & 0x3f) << 6)
}

/// The 32- or 64-bit pattern of the 8-bit floating-point immediate `imm8`
/// (`VFPExpandImm`), as `fmov` and the vector `fmov` take it.
pub fn expand_immediate(imm8: u64, double: bool) -> u64 {
    let sign = imm8 >> 7;
    let b = (imm8 >> 6) & 1;
    let fraction = imm8 & 0x3f;
    if double {
        (sign << 63) | ((b ^ 1) << 62) | (b * 0xff << 54) | (fraction << 48)
    } else {
        (sign << 31) | ((b ^ 1) << 30) | (b * 0x1f << 25) | (fraction << 19)
    }
}

/// Decode one instruction word.
///
/// The families in [`fp`], [`simd`], [`pauth`] and [`gxf`] are tried first and take
/// precedence over the structured forms of this module wherever both recognise a word:
/// the emulator executes those families from their decoded form, never by lifting them.
pub fn decode(word: u32) -> Result<Instruction, DecodeError> {
    if let Some(op) = fp::decode(word) {
        return Ok(Instruction::Fp(op));
    }
    if let Some(op) = simd::decode(word) {
        return Ok(Instruction::Simd(op));
    }
    if let Some(op) = pauth::decode(word) {
        return Ok(Instruction::Pauth(op));
    }
    if let Some(op) = gxf::decode(word) {
        return Ok(Instruction::Gxf(op));
    }
    structured(word)
}

fn structured(word: u32) -> Result<Instruction, DecodeError> {
    use DecodeError::UnsupportedInstruction;
    use Instruction::*;
    let width = width(word);
    let rn = reg(word, 5);
    let rm = reg(word, 16);
    let rd = reg(word, 0);
    let flags = word & 0x2000_0000 != 0;
    let op = arith_op(word);
    if word & 0x1f80_0000 == 0x1280_0000 {
        let hw = ((word >> 21) & 3) as u8;
        if width == Width::W32 && hw > 1 {
            return Err(UnsupportedInstruction);
        }
        let wide = Wide {
            width,
            rd,
            shift: hw * 16,
            immediate: (word >> 5) as u16,
        };
        return match (word >> 29) & 3 {
            0 => Ok(Movn(wide)),
            2 => Ok(Movz(wide)),
            3 => Ok(Movk(wide)),
            _ => Err(UnsupportedInstruction),
        };
    }
    if word & 0x1fe0_0000 == 0x0b20_0000 {
        let amount = ((word >> 10) & 7) as u8;
        if amount > 4 {
            return Err(UnsupportedInstruction);
        }
        return Ok(ArithExt(ArithmeticExtended {
            op,
            flags,
            width,
            rn,
            rm,
            rd,
            extend: extension(word >> 13),
            amount,
        }));
    }
    if word & 0x5f20_0000 == 0x0b00_0000 || word & 0x5f20_0000 == 0x4b00_0000 {
        let shift = match (word >> 22) & 3 {
            0 => Shift::Lsl,
            1 => Shift::Lsr,
            2 => Shift::Asr,
            _ => return Err(UnsupportedInstruction),
        };
        let amount = ((word >> 10) & 63) as u8;
        if (shift != Shift::Lsl && amount == 0) || (width == Width::W32 && amount >= 32) {
            return Err(UnsupportedInstruction);
        }
        return Ok(ArithReg(ArithmeticReg {
            op,
            flags,
            width,
            rn,
            rm,
            rd,
            shift,
            amount,
        }));
    }
    if word & 0x1f80_0000 == 0x1300_0000 {
        if (word >> 22) & 1 != u32::from(width == Width::X64) {
            return Err(UnsupportedInstruction);
        }
        let op = match (word >> 29) & 3 {
            0 => BitfieldOp::Signed,
            1 => BitfieldOp::Insert,
            2 => BitfieldOp::Unsigned,
            _ => return Err(UnsupportedInstruction),
        };
        let immr = ((word >> 16) & 63) as u8;
        let imms = ((word >> 10) & 63) as u8;
        let bits = width as u8;
        if immr >= bits || imms >= bits {
            return Err(UnsupportedInstruction);
        }
        let extract = imms >= immr;
        return Ok(Bitfield(self::Bitfield {
            op,
            extract,
            width,
            rn,
            rd,
            lsb: if extract { immr } else { bits - immr },
            len: if extract { imms - immr + 1 } else { imms + 1 },
        }));
    }
    if word & 0x7fa0_0000 == 0x1380_0000 {
        let lsb = ((word >> 10) & 63) as u8;
        if (word >> 22) & 1 != u32::from(width == Width::X64) || (width == Width::W32 && lsb >= 32)
        {
            return Err(UnsupportedInstruction);
        }
        return Ok(Extract(self::Extract {
            width,
            rn,
            rm,
            rd,
            lsb,
        }));
    }
    if word & 0x1f80_0000 == 0x1200_0000 {
        return Ok(LogicImm(self::LogicImm {
            op: logic_op(word),
            flags: (word >> 29) & 3 == 3,
            immediate: bitmask(word, width)?,
            width,
            rn,
            rd,
        }));
    }
    if word & 0x1f00_0000 == 0x0a00_0000 {
        let shift = [Shift::Lsl, Shift::Lsr, Shift::Asr, Shift::Ror][((word >> 22) & 3) as usize];
        let amount = ((word >> 10) & 63) as u8;
        if (shift != Shift::Lsl && shift != Shift::Ror && amount == 0)
            || (width == Width::W32 && amount >= 32)
        {
            return Err(UnsupportedInstruction);
        }
        return Ok(LogicReg(Logic {
            op: logic_op(word),
            flags: (word >> 29) & 3 == 3,
            width,
            rn,
            rm,
            rd,
            shift,
            amount,
            invert: word & 0x0020_0000 != 0,
        }));
    }
    if word & 0x7e00_0000 == 0x3400_0000 {
        return Ok(TestBranch(Test {
            rt: rd,
            bit_index: None,
            negate: word & 0x0100_0000 != 0,
            width,
            offset: sign_extend(word >> 5, 19) << 2,
        }));
    }
    if word & 0x7e00_0000 == 0x3600_0000 {
        let bit = (((word >> 31) << 5) | ((word >> 19) & 31)) as u8;
        return Ok(TestBranch(Test {
            rt: rd,
            bit_index: Some(bit),
            negate: word & 0x0100_0000 != 0,
            width,
            offset: sign_extend(word >> 5, 14) << 2,
        }));
    }
    if word & 0xfc00_0000 == 0x9400_0000 {
        return Ok(Call(self::Call {
            target: sign_extend(word, 26) << 2,
            link: true,
        }));
    }
    if word & 0xfe00_0000 == 0xd600_0000 {
        if word & 0x001f_0000 != 0x001f_0000 || word & 0x0000_fc1f != 0 {
            return Err(UnsupportedInstruction);
        }
        // opc is bits 24:21; bit 24 set is the authenticated forms (`braa`, `retaa`, ...).
        return match (word >> 21) & 15 {
            0 => Ok(Indirect(self::Indirect {
                rn,
                kind: IndirectKind::Jump,
            })),
            1 => Ok(Indirect(self::Indirect {
                rn,
                kind: IndirectKind::Call,
            })),
            2 => Ok(Indirect(self::Indirect {
                rn,
                kind: IndirectKind::Return,
            })),
            4 if rn == 31 => Ok(Eret),
            _ => Err(UnsupportedInstruction),
        };
    }
    if word & 0x1f00_0000 == 0x1100_0000 {
        if word & 0x0080_0000 != 0 {
            return Err(UnsupportedInstruction);
        }
        return Ok(ArithImm(Arithmetic {
            op,
            flags,
            width,
            rn,
            rd,
            immediate: ((word >> 10) & 0xfff) as u16,
            shifted: word & 0x0040_0000 != 0,
        }));
    }
    // PRFM (unsigned offset), PRFUM (unscaled offset) and PRFM (register). They
    // keep their operand and execute as `nop`.
    if word & 0xffc0_0000 == 0xf980_0000 {
        return Ok(Prefetch(self::Prefetch {
            op: rd,
            rn,
            addressing: Addressing::Offset((((word >> 10) & 0xfff) << 3) as i64),
        }));
    }
    if word & 0xffe0_0c00 == 0xf880_0000 {
        return Ok(Prefetch(self::Prefetch {
            op: rd,
            rn,
            addressing: Addressing::Unscaled(sign_extend(word >> 12, 9)),
        }));
    }
    if word & 0xffe0_0c00 == 0xf8a0_0800 {
        let addressing = register_offset(word, 3)?;
        // RPRFM takes the PRFM (register) slots whose Rt is 0b11xxx; its 6-bit operation
        // is `option<2>:option<0>:S:Rt<2:0>`.
        if rd >= 24 {
            let option = (word >> 13) & 7;
            return Ok(RangePrefetch(self::RangePrefetch {
                op: ((option >> 2) << 5
                    | (option & 1) << 4
                    | ((word >> 12) & 1) << 3
                    | u32::from(rd & 7)) as u8,
                rm,
                rn,
            }));
        }
        return Ok(Prefetch(self::Prefetch {
            op: rd,
            rn,
            addressing,
        }));
    }
    // PRFM (literal).
    if word & 0xff00_0000 == 0xd800_0000 {
        return Ok(PrefetchLiteral(self::PrefetchLiteral {
            op: rd,
            offset: sign_extend(word >> 5, 19) << 2,
        }));
    }
    if word & 0x3f00_0000 == 0x1800_0000 {
        let (size, signed) = match word >> 30 {
            0 => (Size::Word, SignExtend::None),
            1 => (Size::Double, SignExtend::None),
            // LDRSW (literal).
            2 => (Size::Word, SignExtend::To64),
            _ => return Err(UnsupportedInstruction),
        };
        return Ok(Literal(self::Literal {
            size,
            rt: rd,
            offset: sign_extend(word >> 5, 19) << 2,
            signed,
        }));
    }
    // LDR (literal, SIMD&FP): S, D or Q register from a PC-relative address.
    if word & 0x3f00_0000 == 0x1c00_0000 {
        let bytes = match word >> 30 {
            0 => 4,
            1 => 8,
            2 => 16,
            _ => return Err(UnsupportedInstruction),
        };
        return Ok(SimdLiteral(self::SimdLiteral {
            bytes,
            rt: rd,
            offset: sign_extend(word >> 5, 19) << 2,
        }));
    }
    // LSE atomics (ARMv8.1): LDADD/LDCLR/LDEOR/LDSET/LDSMAX/LDSMIN/LDUMAX/LDUMIN and
    // SWP share one encoding, split by o3 (bit 15) and opc (bits 14:12); A (bit 23) is
    // acquire and R (bit 22) release. LDAPR, the RCpc load, is the A=1, R=0, o3=1,
    // opc=100 slot with Rs unused.
    if word & 0x3f20_0c00 == 0x3820_0000 {
        let order = MemoryOrder {
            acquire: word & 0x0080_0000 != 0,
            release: word & 0x0040_0000 != 0,
        };
        let kind = match (word & 0x8000 != 0, (word >> 12) & 7) {
            (false, 0) => AtomicKind::Add,
            (false, 1) => AtomicKind::Clr,
            (false, 2) => AtomicKind::Eor,
            (false, 3) => AtomicKind::Set,
            (false, 4) => AtomicKind::SMax,
            (false, 5) => AtomicKind::SMin,
            (false, 6) => AtomicKind::UMax,
            (false, 7) => AtomicKind::UMin,
            (true, 0) => AtomicKind::Swap,
            (true, 4) if rm == 31 && order.acquire && !order.release => {
                return Ok(Memory(self::Memory {
                    op: MemoryOp::Load,
                    size: size(word),
                    rn,
                    rt: rd,
                    addressing: Addressing::Offset(0),
                    signed: SignExtend::None,
                    ordered: Some(Ordered::AcquirePc),
                }));
            }
            _ => return Err(UnsupportedInstruction),
        };
        return Ok(Atomic(self::Atomic {
            kind,
            size: size(word),
            rs: rm,
            rn,
            rt: rd,
            rt2: 0,
            order,
        }));
    }
    // CAS/CASP (ARMv8.1): compare with Rs (the pair Rs, Rs+1 for CASP); store Rt on a match.
    // L (bit 22) is acquire and o0 (bit 15) release.
    let cas_order = MemoryOrder {
        acquire: word & 0x0040_0000 != 0,
        release: word & 0x8000 != 0,
    };
    if word & 0x3fa0_7c00 == 0x08a0_7c00 {
        return Ok(Atomic(self::Atomic {
            kind: AtomicKind::Cas,
            size: size(word),
            rs: rm,
            rn,
            rt: rd,
            rt2: 0,
            order: cas_order,
        }));
    }
    if word & 0xbfa0_7c00 == 0x0820_7c00 {
        // CASP register pairs must be even; odd pairs are unallocated.
        if rm % 2 != 0 || rd % 2 != 0 {
            return Err(UnsupportedInstruction);
        }
        let size = if word & 0x4000_0000 != 0 {
            Size::Double
        } else {
            Size::Word
        };
        return Ok(Atomic(self::Atomic {
            kind: AtomicKind::Casp,
            size,
            rs: rm,
            rn,
            rt: rd,
            rt2: 0,
            order: cas_order,
        }));
    }
    if word & 0x3f00_0000 == 0x0800_0000 {
        // o0 (bit 15): release for a store, acquire for a load.
        let load = word & 0x0040_0000 != 0;
        let o0 = word & 0x8000 != 0;
        let order = MemoryOrder {
            acquire: load && o0,
            release: !load && o0,
        };
        if word & 0x0020_0000 != 0 {
            // LDXP/LDAXP/STXP/STLXP: a pair of words or doublewords; the load form has
            // no status register.
            if word & 0x8000_0000 == 0 || word & 0x0080_0000 != 0 || (load && rm != 31) {
                return Err(UnsupportedInstruction);
            }
            return Ok(Atomic(self::Atomic {
                kind: if load {
                    AtomicKind::LoadPair
                } else {
                    AtomicKind::StorePair
                },
                size: if word & 0x4000_0000 != 0 {
                    Size::Double
                } else {
                    Size::Word
                },
                rs: rm,
                rn,
                rt: rd,
                rt2: reg(word, 10),
                order,
            }));
        }
        let size = size(word);
        let op = if load {
            MemoryOp::Load
        } else {
            MemoryOp::Store
        };
        let rs = rm;
        if reg(word, 10) != 31 {
            return Err(UnsupportedInstruction);
        }
        if word & 0x0080_0000 != 0 {
            // LDAR/STLR (o0 set) and the LORegion LDLAR/STLLR (o0 clear).
            if rs != 31 {
                return Err(UnsupportedInstruction);
            }
            return Ok(Memory(self::Memory {
                op,
                size,
                rn,
                rt: rd,
                addressing: Addressing::Offset(0),
                signed: SignExtend::None,
                ordered: Some(match (load, o0) {
                    (true, true) => Ordered::Acquire,
                    (true, false) => Ordered::LimitedAcquire,
                    (false, true) => Ordered::Release,
                    (false, false) => Ordered::LimitedRelease,
                }),
            }));
        }
        if load && rs != 31 {
            return Err(UnsupportedInstruction);
        }
        return Ok(Exclusive(self::Exclusive {
            op,
            size,
            rn,
            rt: rd,
            rs,
            order,
        }));
    }
    // LD1..LD4 / ST1..ST4 (multiple and single structures) and LD1R..LD4R. A single
    // register without a register post-index moves a whole vector like `ldr d`/`ldr q`,
    // which the JIT inlines: that form is a `SimdMemory` that remembers its arrangement.
    if word & 0xbe00_0000 == 0x0c00_0000 {
        let (desc, post) = simd_struct(word)?;
        if desc.shape == Shape::Multiple
            && desc.registers == 1
            && !matches!(post, StructPost::Register(_))
        {
            let bytes = desc.vector_bytes();
            return Ok(SimdMemory(self::SimdMemory {
                op: if desc.store {
                    MemoryOp::Store
                } else {
                    MemoryOp::Load
                },
                bytes,
                rn,
                rt: desc.rt,
                addressing: if post == StructPost::Immediate {
                    Addressing::PostIndex(i64::from(bytes))
                } else {
                    Addressing::Offset(0)
                },
                access: SimdAccess::Ld1 { esize: desc.esize },
            }));
        }
        return Ok(SimdStruct(self::SimdStruct { desc, rn, post }));
    }
    // LDAPUR/STLUR and LDAPURS* (FEAT_LRCPC2): the RCpc/release accesses with an unscaled
    // offset.
    if word & 0x3f20_0c00 == 0x1900_0000 {
        let opc = (word >> 22) & 3;
        let size = size(word);
        return Ok(Memory(self::Memory {
            op: if opc == 0 {
                MemoryOp::Store
            } else {
                MemoryOp::Load
            },
            size,
            rn,
            rt: rd,
            addressing: Addressing::Unscaled(sign_extend(word >> 12, 9)),
            signed: load_extension(opc, size)?,
            ordered: Some(if opc == 0 {
                Ordered::Release
            } else {
                Ordered::AcquirePc
            }),
        }));
    }
    if word & 0x3a00_0000 == 0x3800_0000 {
        let vector = word & 0x0400_0000 != 0;
        if vector {
            let opc = (word >> 22) & 3;
            let size_bits = word >> 30;
            let (bytes, is_load) = match (size_bits, opc) {
                (0, 0) => (1, false),
                (0, 1) => (1, true),
                (1, 0) => (2, false),
                (1, 1) => (2, true),
                (2, 0) => (4, false),
                (2, 1) => (4, true),
                (3, 0) => (8, false),
                (3, 1) => (8, true),
                (0, 2) => (16, false),
                (0, 3) => (16, true),
                _ => return Err(UnsupportedInstruction),
            };
            let addressing = if word & 0x0100_0000 != 0 {
                Addressing::Offset((((word >> 10) & 0xfff) * bytes as u32) as i64)
            } else if word & 0x0020_0000 != 0 {
                register_offset(word, (bytes as u32).trailing_zeros())?
            } else {
                immediate_offset(word, true)?
            };
            return Ok(SimdMemory(self::SimdMemory {
                op: if is_load {
                    MemoryOp::Load
                } else {
                    MemoryOp::Store
                },
                bytes,
                rn,
                rt: rd,
                addressing,
                access: SimdAccess::Register,
            }));
        }
        let opc = (word >> 22) & 3;
        let size = size(word);
        let signed = load_extension(opc, size)?;
        let addressing = if word & 0x0100_0000 != 0 {
            Addressing::Offset((((word >> 10) & 0xfff) * size as u32) as i64)
        } else if word & 0x0020_0000 != 0 {
            register_offset(word, (size as u32).trailing_zeros())?
        } else {
            immediate_offset(word, false)?
        };
        return Ok(Memory(self::Memory {
            op: if opc == 0 {
                MemoryOp::Store
            } else {
                MemoryOp::Load
            },
            size,
            rn,
            rt: rd,
            addressing,
            signed,
            ordered: None,
        }));
    }
    if word & 0xffff_f01f == 0xd503_201f {
        let hint = ((word >> 5) & 127) as u8;
        return Ok(match hint {
            0 => Nop,
            1 => Yield,
            2 => Wfe,
            3 => Wfi,
            4 => Sev,
            5 => Sevl,
            _ => Hint(hint),
        });
    }
    if word & 0xffff_f01f == 0xd503_301f {
        let option = ((word >> 8) & 15) as u8;
        return match (word >> 5) & 7 {
            2 => Ok(Clrex(option)),
            4 => Ok(Dsb(option)),
            5 => Ok(Dmb(option)),
            6 => Ok(Isb(option)),
            _ => Err(UnsupportedInstruction),
        };
    }
    if word & 0xffff_ffe0 == 0xd50b_7420 {
        return Ok(DcZva(rd));
    }
    // AT S1E1R / S1E1W / S1E0R / S1E0W: op1 = 0, CRn = 7, CRm = 8, op2 = 0..=3.
    if word & 0xffff_ff00 == 0xd508_7800 && (word >> 5) & 7 < 4 {
        let op2 = (word >> 5) & 7;
        return Ok(AddressTranslate(self::AddressTranslate {
            rt: rd,
            write: op2 & 1 != 0,
            user: op2 & 2 != 0,
        }));
    }
    if word & 0xfff8_0000 == 0xd508_0000 {
        let crn = (word >> 12) & 15;
        let crm = (word >> 8) & 15;
        let sys = SysOp {
            op1: ((word >> 16) & 7) as u8,
            crn: crn as u8,
            crm: crm as u8,
            op2: ((word >> 5) & 7) as u8,
            rt: rd,
        };
        if crn == 8 {
            return Ok(Tlbi(sys));
        }
        if crn == 7 && matches!(crm, 1 | 5 | 6 | 10 | 11 | 14) {
            return Ok(CacheOp(sys));
        }
        return Err(UnsupportedInstruction);
    }
    if word & 0xffc0_0000 == 0xd500_0000 {
        let read = word & 0x0020_0000 != 0;
        let register = system_register(
            (word >> 19) & 3,
            (word >> 16) & 7,
            (word >> 12) & 15,
            (word >> 8) & 15,
            (word >> 5) & 7,
        )?;
        if !read && register.read_only() {
            return Err(UnsupportedInstruction);
        }
        let expected = match register {
            SystemRegister::DaifSet
            | SystemRegister::DaifClear
            | SystemRegister::SpSelSet
            | SystemRegister::PanSet => 0,
            SystemRegister::MdscrEl1
            | SystemRegister::OslarEl1
            | SystemRegister::OsdlrEl1
            | SystemRegister::OslsrEl1
            | SystemRegister::RazWi { .. } => 2,
            _ => 3,
        };
        if (word >> 19) & 3 != expected || (read && expected == 0) {
            return Err(UnsupportedInstruction);
        }
        // `msr <pstatefield>, #imm` has no register operand (Rt must be 31), and PAN and
        // SPSel take a one-bit immediate; any other word is unallocated.
        let immediate = ((word >> 8) & 15) as u8;
        let pstate = expected == 0;
        if pstate
            && (rd != 31
                || (matches!(register, SystemRegister::SpSelSet | SystemRegister::PanSet)
                    && immediate > 1))
        {
            return Err(UnsupportedInstruction);
        }
        return Ok(System(self::System {
            read,
            register,
            rt: rd,
            immediate: if pstate { immediate } else { 0 },
        }));
    }
    // BRK #imm16: opc 001 with LL 00. Its immediate is the whole field, so any value
    // decodes; the trap carries it to the exception syndrome.
    if word & 0xffe0_001f == 0xd420_0000 {
        return Ok(Brk((word >> 5) as u16));
    }
    if word & 0x1f00_0000 == 0x1000_0000 {
        let page = word & 0x8000_0000 != 0;
        let immediate = (((word >> 5) & 0x7ffff) << 2) | ((word >> 29) & 3);
        return Ok(Adr(Address {
            page,
            rd,
            offset: sign_extend(immediate, 21) * if page { 4096 } else { 1 },
        }));
    }
    if word & 0xff00_0000 == 0x9b00_0000 {
        let form = (word >> 21) & 7;
        let ra = reg(word, 10);
        match form {
            1 | 5 => {
                return Ok(MulLong(MultiplyLong {
                    signed: form == 1,
                    op: if word & 0x8000 != 0 {
                        ArithOp::Sub
                    } else {
                        ArithOp::Add
                    },
                    rn,
                    rm,
                    ra,
                    rd,
                }));
            }
            2 | 6 => {
                if word & 0x8000 != 0 || ra != 31 {
                    return Err(UnsupportedInstruction);
                }
                return Ok(MulHigh(MultiplyHigh {
                    signed: form == 2,
                    rn,
                    rm,
                    rd,
                }));
            }
            _ => {}
        }
    }
    if word & 0x7fe0_0000 == 0x1b00_0000 {
        return Ok(Mul(Multiply {
            op: if word & 0x8000 != 0 {
                ArithOp::Sub
            } else {
                ArithOp::Add
            },
            width,
            rn,
            rm,
            ra: reg(word, 10),
            rd,
        }));
    }
    if word & 0x3a00_0000 == 0x2800_0000 {
        let vector = word & 0x0400_0000 != 0;
        let load = word & 0x0040_0000 != 0;
        let opc = word >> 30;
        // Bits 24:23 are 00 for the no-allocate pair (LDNP, STNP).
        let non_temporal = (word >> 23) & 7 == 0;
        if vector {
            let bytes = match opc {
                0 => 4,  // S
                1 => 8,  // D
                2 => 16, // Q
                _ => return Err(UnsupportedInstruction),
            };
            let displacement = sign_extend(word >> 15, 7) * bytes as i64;
            let addressing = match (word >> 23) & 7 {
                0 | 2 => Addressing::Offset(displacement),
                3 => Addressing::PreIndex(displacement),
                1 => Addressing::PostIndex(displacement),
                _ => return Err(UnsupportedInstruction),
            };
            return Ok(SimdPair(self::SimdPair {
                op: if load {
                    MemoryOp::Load
                } else {
                    MemoryOp::Store
                },
                bytes,
                rn,
                rt: rd,
                rt2: reg(word, 10),
                addressing,
                non_temporal,
            }));
        }
        let load = word & 0x0040_0000 != 0;
        let opc = word >> 30;
        if opc == 1 && (!load || non_temporal) {
            return Err(UnsupportedInstruction);
        }
        let size = match opc {
            0 | 1 => Size::Word,
            2 => Size::Double,
            _ => return Err(UnsupportedInstruction),
        };
        let displacement = sign_extend(word >> 15, 7) * size as i64;
        let addressing = match (word >> 23) & 7 {
            0 | 2 => Addressing::Offset(displacement),
            3 => Addressing::PreIndex(displacement),
            1 => Addressing::PostIndex(displacement),
            _ => return Err(UnsupportedInstruction),
        };
        return Ok(Pair(self::Pair {
            op: if load {
                MemoryOp::Load
            } else {
                MemoryOp::Store
            },
            size,
            rn,
            rt: rd,
            rt2: reg(word, 10),
            addressing,
            non_temporal,
            signed: if opc == 1 {
                SignExtend::To64
            } else {
                SignExtend::None
            },
        }));
    }
    // Advanced SIMD modified immediate: MOVI, MVNI, ORR, BIC and FMOV (vector).
    // `AdvSIMDExpandImm` selects the lane pattern from `cmode` and `op`.
    // `immh` (bits 22:19) is zero here; a non-zero `immh` is a shift by immediate.
    // Bit 11 (`o2`) is zero except in the half-precision `fmov`.
    if word & 0x9ff8_0400 == 0x0f00_0400 {
        let q = word & 0x4000_0000 != 0;
        let op = word & 0x2000_0000 != 0;
        let o2 = word & 0x0000_0800 != 0;
        let cmode = (word >> 12) & 0xf;
        let imm8 = (((word >> 16) & 7) << 5) | ((word >> 5) & 0x1f);
        let rd = (word & 0x1f) as u8;
        let twice = |v: u64| v | (v << 32);
        let wide = |lane: u64| lane * 0x0001_0001_0001_0001;
        let wide_imm = u64::from(imm8);
        let (form, pattern) = match cmode {
            // 32-bit lanes, shifted left by 0, 8, 16 or 24.
            0..=7 if !o2 => {
                let shift = 8 * (cmode >> 1);
                (
                    ImmForm::Word { shift: shift as u8 },
                    twice(wide_imm << shift),
                )
            }
            // 16-bit lanes, shifted left by 0 or 8.
            8..=11 if !o2 => {
                let shift = 8 * ((cmode >> 1) & 1);
                (
                    ImmForm::Half { shift: shift as u8 },
                    wide(wide_imm << shift),
                )
            }
            // 32-bit lanes with the vacated low bits set to ones.
            12 | 13 if !o2 => {
                let shift = 8 + 8 * (cmode & 1);
                (
                    ImmForm::WordMask { shift: shift as u8 },
                    twice((wide_imm << shift) | ((1 << shift) - 1)),
                )
            }
            // Bytes: imm8 in every byte.
            14 if !o2 && !op => (ImmForm::Byte, wide_imm * 0x0101_0101_0101_0101),
            // Each bit of imm8 as a 0x00/0xff byte, in the one 64-bit lane.
            14 if !o2 => (
                ImmForm::Double,
                (0..8).fold(0, |acc, bit| {
                    acc | (if (wide_imm >> bit) & 1 != 0 { 0xff } else { 0 }) << (8 * bit)
                }),
            ),
            // FMOV: the 8-bit float immediate widened to f16, f32 (twice) or f64.
            15 if !op && o2 => (ImmForm::FmovHalf, wide(expand_half_immediate(wide_imm))),
            15 if !op => (
                ImmForm::FmovSingle,
                twice(expand_immediate(wide_imm, false)),
            ),
            15 if q && !o2 => (ImmForm::FmovDouble, expand_immediate(wide_imm, true)),
            _ => return Err(UnsupportedInstruction),
        };
        // Odd cmode below 12 is ORR/BIC; MOVI becomes MVNI (inverted) with op set.
        let (value, combine) = if cmode < 12 && cmode & 1 == 1 {
            (
                pattern,
                if op {
                    ImmCombine::AndNot
                } else {
                    ImmCombine::Or
                },
            )
        } else if cmode < 14 && op {
            (!pattern, ImmCombine::SetInverted)
        } else {
            (pattern, ImmCombine::Set)
        };
        return Ok(SimdImm(SimdImmediate {
            rd,
            imm8: imm8 as u8,
            form,
            combine,
            q,
            low: value,
            high: if q { value } else { 0 },
        }));
    }
    if word & 0xbf3f_fc00 == 0x0e20_9800 {
        let q = word & 0x4000_0000 != 0;
        let size = ((word >> 22) & 3) as u8;
        if !q && size == 3 {
            return Err(UnsupportedInstruction);
        }
        return Ok(SimdCompareZero(self::SimdCompareZero { rd, rn, size, q }));
    }
    // FMOV between a general register and an S/D register: a pure bit move. Bit 24
    // must be clear; the floating-point 3-source group (FMADD and friends) has it set.
    if word & 0x7fbe_fc00 == 0x1e26_0000 {
        let double64 = word & 0x8000_0000 != 0;
        let double_type = word & 0x0040_0000 != 0;
        if double64 != double_type {
            return Err(UnsupportedInstruction);
        }
        return Ok(SimdFmov(self::SimdFmov {
            rd,
            rn,
            to_fp: word & 0x0001_0000 != 0,
            double: double64,
            upper: false,
        }));
    }
    // `fmov Vd.D[1], Xn` / `fmov Xd, Vn.D[1]`: the upper doubleword of the vector.
    if word & 0xfffe_fc00 == 0x9eae_0000 {
        return Ok(SimdFmov(self::SimdFmov {
            rd,
            rn,
            to_fp: word & 0x0001_0000 != 0,
            double: true,
            upper: true,
        }));
    }
    // Across-lanes reductions: bits 21:17 = 11000, bits 11:10 = 10. They share
    // the three-same class bits, so they are matched first. ADDV (U=0, opcode
    // 11011) and, with U=1, UMAXV (01010), UMINV (11010) and UADDLV (00011).
    if word & 0xbf3f_fc00 == 0x0e31_b800
        || word & 0xbf3f_fc00 == 0x2e30_a800
        || word & 0xbf3f_fc00 == 0x2e31_a800
        || word & 0xbf3f_fc00 == 0x2e30_3800
    {
        let q = word & 0x4000_0000 != 0;
        let size = ((word >> 22) & 3) as u8;
        // 64-bit lanes need Q, and `.2s` is reserved for every reduction.
        if size == 3 || (size == 2 && !q) {
            return Err(UnsupportedInstruction);
        }
        let op = match (word & 0x2000_0000 != 0, (word >> 12) & 0x1f) {
            (false, 0b11011) => SimdAluOp::AddV,
            (true, 0b01010) => SimdAluOp::UMaxV,
            (true, 0b11010) => SimdAluOp::UMinV,
            (true, 0b00011) => SimdAluOp::UAddLV,
            _ => return Err(UnsupportedInstruction),
        };
        return Ok(SimdAlu(self::SimdAlu {
            op,
            rd,
            rn,
            rm: 0,
            size,
            q,
        }));
    }
    // Integer three-registers-same (`U`, `size`, `opcode` verified with llvm-mc).
    // Bit 10 is 1 in this class; bit-10-clear encodings are handled above.
    if word & 0x9f20_0400 == 0x0e20_0400 {
        use SimdAluOp::*;
        let q = word & 0x4000_0000 != 0;
        let u = word & 0x2000_0000 != 0;
        let size = ((word >> 22) & 3) as u8;
        let opc = ((word >> 11) & 31) as u8;
        let op = match (u, opc) {
            (false, 0b10000) => Add,
            (true, 0b10000) => Sub,
            (false, 0b10011) => Mul,
            (false, 0b10010) => Mla,
            (true, 0b10010) => Mls,
            (true, 0b10001) => CmEq,
            (false, 0b00110) => CmGt,
            (false, 0b00111) => CmGe,
            (true, 0b00110) => CmHi,
            (true, 0b00111) => CmHs,
            (false, 0b10001) => CmTst,
            (true, 0b01100) => UMax,
            (true, 0b01101) => UMin,
            (false, 0b01100) => SMax,
            (false, 0b01101) => SMin,
            (true, 0b01110) => UAbd,
            (false, 0b01110) => SAbd,
            (false, 0b00001) => SQAdd,
            (true, 0b00001) => UQAdd,
            (false, 0b00101) => SQSub,
            (true, 0b00101) => UQSub,
            (false, 0b00000) => SHAdd,
            (true, 0b00000) => UHAdd,
            (false, 0b00100) => SHSub,
            (true, 0b00100) => UHSub,
            (false, 0b00010) => SRHAdd,
            (true, 0b00010) => URHAdd,
            (false, 0b01000) => SSHl,
            (true, 0b01000) => USHl,
            (false, 0b01001) => SQShl,
            (true, 0b01001) => UQShl,
            (false, 0b01011) => SQRShl,
            (true, 0b01011) => UQRShl,
            (false, 0b10111) => AddP,
            (true, 0b10100) => UMaxP,
            (false, 0b10100) => SMaxP,
            (true, 0b10101) => UMinP,
            (false, 0b10101) => SMinP,
            // The two low opcode bits select AND/BIC/ORR/ORN (and, with U=1,
            // EOR/BSL/BIT/BIF).
            (false, 0b00011) if size == 0 => And,
            (false, 0b00011) if size == 1 => Bic,
            (false, 0b00011) if size == 2 => Orr,
            (false, 0b00011) if size == 3 => Orn,
            (true, 0b00011) if size == 0 => Eor,
            (true, 0b00011) if size == 1 => Bsl,
            (true, 0b00011) if size == 2 => Bit,
            (true, 0b00011) if size == 3 => Bif,
            _ => return Err(UnsupportedInstruction),
        };
        // `orn` and `bif` use `size` as an opcode; every other 64-bit lane needs
        // the full vector and exists only in these groups.
        if size == 3
            && !matches!(op, Orn | Bif)
            && !(q
                && matches!(
                    op,
                    Add | Sub
                        | CmEq
                        | CmGt
                        | CmGe
                        | CmHi
                        | CmHs
                        | CmTst
                        | SSHl
                        | USHl
                        | SQShl
                        | UQShl
                        | SQRShl
                        | UQRShl
                        | SQAdd
                        | UQAdd
                        | SQSub
                        | UQSub
                        | AddP
                ))
        {
            return Err(UnsupportedInstruction);
        }
        return Ok(SimdAlu(self::SimdAlu {
            op,
            rd,
            rn,
            rm,
            size,
            q,
        }));
    }
    // Table vector lookup: `tbl` and `tbx` over one to four table registers.
    // Bits 14:13 hold the table length, bit 12 selects tbl/tbx, and bits 11:10
    // are reserved, so the class mask covers everything else.
    if word & 0xbfe0_8c00 == 0x0e00_0000 {
        return Ok(SimdTable(self::SimdTable {
            rd: (word & 0x1f) as u8,
            rn: ((word >> 5) & 0x1f) as u8,
            rm: ((word >> 16) & 0x1f) as u8,
            len: ((word >> 13) & 3) as u8,
            extend: word & 0x0000_1000 != 0,
            q: word & 0x4000_0000 != 0,
        }));
    }
    // `mov <Vd>.<T>[index], <R>` (`ins`, general): only the 128-bit form exists.
    if word & 0xffe0_fc00 == 0x4e00_1c00 {
        let imm5 = (word >> 16) & 0x1f;
        if imm5 != 0 {
            let esize = imm5.trailing_zeros() as u8;
            if esize <= 3 {
                return Ok(SimdInsert(self::SimdInsert {
                    rd,
                    rn,
                    esize,
                    index: (imm5 >> (esize + 1)) as u8,
                }));
            }
        }
    }
    // `umov`/`mov <R>, <Vn>.<T>[index]`: `Q` is set exactly for the doubleword lane.
    if word & 0xbfe0_fc00 == 0x0e00_3c00 {
        let imm5 = (word >> 16) & 0x1f;
        if imm5 != 0 {
            let esize = imm5.trailing_zeros() as u8;
            if esize <= 3 && (word & 0x4000_0000 != 0) == (esize == 3) {
                return Ok(SimdExtract(self::SimdExtract {
                    rd,
                    rn,
                    esize,
                    index: (imm5 >> (esize + 1)) as u8,
                }));
            }
        }
    }
    // `mov Vd.T[i], Vn.T[j]` (`ins`, element): imm5 gives the size and the
    // destination index; imm4 gives the source index, scaled by the size.
    if word & 0xffe0_8400 == 0x6e00_0400 {
        let imm5 = (word >> 16) & 0x1f;
        let esize = imm5.trailing_zeros();
        if imm5 != 0 && esize <= 3 {
            return Ok(SimdInsertElement(self::SimdInsertElement {
                rd,
                rn,
                esize: esize as u8,
                dst: (imm5 >> (esize + 1)) as u8,
                src: (((word >> 11) & 0xf) >> esize) as u8,
            }));
        }
    }
    // `dup` from an element: the imm5 field encodes the element size as the
    // position of its lowest set bit and the index in the bits above.
    if word & 0xbfe0_fc00 == 0x0e00_0400 {
        let q = word & 0x4000_0000 != 0;
        let imm5 = (word >> 16) & 0x1f;
        if imm5 != 0 {
            let esize = imm5.trailing_zeros() as u8;
            if esize <= 3 && (q || esize < 3) {
                return Ok(SimdDupElement(self::SimdDupElement {
                    rd,
                    rn,
                    esize,
                    index: (imm5 >> (esize + 1)) as u8,
                    q,
                }));
            }
        }
    }
    // `dup <Vd>.<T>, <R>`: the doubleword arrangement needs the full vector.
    if word & 0xbfe0_fc00 == 0x0e00_0c00 {
        let q = (word >> 30) & 1 != 0;
        let imm5 = (word >> 16) & 0x1f;
        if imm5 != 0 {
            let esize = imm5.trailing_zeros() as u8;
            if esize <= 3 && (q || esize < 3) {
                let rn = ((word >> 5) & 0x1f) as u8;
                let rd = (word & 0x1f) as u8;
                return Ok(SimdDup { rd, rn, esize, q });
            }
        }
    }
    if word & 0xfc00_0000 == 0x1400_0000 {
        return Ok(B(sign_extend(word, 26) << 2));
    }
    if word & 0xff00_0010 == 0x5400_0000 {
        let cond = condition(word);
        if matches!(cond, Condition::Al | Condition::Nv) {
            return Err(UnsupportedInstruction);
        }
        return Ok(BCond(BranchCondition {
            cond,
            offset: sign_extend(word >> 5, 19) << 2,
        }));
    }
    if word & 0xbfe0_8400 == 0x2e00_0000 {
        let q = word & 0x4000_0000 != 0;
        let imm = ((word >> 11) & 0xf) as u8;
        // The 64-bit form extracts from a 16-byte pair, so the index is below 8.
        if !q && imm >= 8 {
            return Err(UnsupportedInstruction);
        }
        let rm = ((word >> 16) & 0x1f) as u8;
        let rn = ((word >> 5) & 0x1f) as u8;
        let rd = (word & 0x1f) as u8;
        return Ok(SimdExt(self::SimdExt { rd, rn, rm, imm, q }));
    }
    if word & 0x7fff_0000 == 0x5ac0_0000 {
        let op = match (word >> 10) & 63 {
            0 => UnaryOp::Rbit,
            1 => UnaryOp::Rev16,
            2 => {
                if width == Width::X64 {
                    UnaryOp::Rev32
                } else {
                    UnaryOp::Rev
                }
            }
            3 if width == Width::X64 => UnaryOp::Rev,
            4 => UnaryOp::Clz,
            _ => return Err(UnsupportedInstruction),
        };
        return Ok(Unary(self::Unary { op, width, rn, rd }));
    }
    if word & 0x1fe0_fc00 == 0x1a00_0000 {
        return Ok(ArithCarry(AddCarry {
            op,
            flags,
            width,
            rn,
            rm,
            rd,
        }));
    }
    if word & 0x3fe0_0410 == 0x3a40_0000 {
        let operand = if word & 0x800 != 0 {
            CompareOperand::Immediate(rm)
        } else {
            CompareOperand::Register(rm)
        };
        return Ok(CondCompare(self::CondCompare {
            op,
            width,
            rn,
            operand,
            cond: condition(word >> 12),
            nzcv: (word & 15) as u8,
        }));
    }
    if word & 0x1fe0_0000 == 0x1a80_0000 {
        if word & 0x2000_0800 != 0 {
            return Err(UnsupportedInstruction);
        }
        return Ok(Csel(Select {
            cond: condition(word >> 12),
            width,
            rn,
            rm,
            rd,
            opc: ((u8::from(word & 0x4000_0000 != 0) << 1) | u8::from(word & 0x400 != 0)),
        }));
    }
    if word & 0x7fe0_0000 == 0x1ac0_0000 {
        let op = match (word >> 10) & 63 {
            2 => VariableOp::Udiv,
            3 => VariableOp::Sdiv,
            8 => VariableOp::Lsl,
            9 => VariableOp::Lsr,
            10 => VariableOp::Asr,
            11 => VariableOp::Ror,
            _ => return Err(UnsupportedInstruction),
        };
        return Ok(Variable(self::Variable {
            op,
            width,
            rn,
            rm,
            rd,
        }));
    }
    // SVC #imm16: the immediate is the whole field.
    if word & 0xffe0_001f == 0xd400_0001 {
        return Ok(Svc((word >> 5) as u16));
    }
    match word {
        0xd400_0002 => Ok(Psci),
        0xd503_207f => Ok(Wfi),
        _ => Err(UnsupportedInstruction),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn assembler_memory_addressing_vectors() {
        let rows = [
            (0xf9400420, Size::Double, 1, 0, Addressing::Offset(8)),
            (0x39401062, Size::Byte, 3, 2, Addressing::Offset(4)),
            (
                0xf86668a4,
                Size::Double,
                5,
                4,
                Addressing::Register {
                    rm: 6,
                    extend: Extend::Uxtx,
                    amount: None,
                },
            ),
            (
                0xf86678a4,
                Size::Double,
                5,
                4,
                Addressing::Register {
                    rm: 6,
                    extend: Extend::Uxtx,
                    amount: Some(3),
                },
            ),
            (0xf8410d07, Size::Double, 8, 7, Addressing::PreIndex(16)),
            (0xf8410549, Size::Double, 10, 9, Addressing::PostIndex(16)),
            (0xf85f818b, Size::Double, 12, 11, Addressing::Unscaled(-8)),
            (0xb85fc1cd, Size::Word, 14, 13, Addressing::Unscaled(-4)),
        ];
        for (word, size, rn, rt, addressing) in rows {
            assert_eq!(
                decode(word),
                Ok(Instruction::Memory(Memory {
                    op: MemoryOp::Load,
                    size,
                    rn,
                    rt,
                    addressing,
                    signed: SignExtend::None,
                    ordered: None,
                }))
            );
        }
        assert_eq!(
            decode(0xb81fc0e6),
            Ok(Instruction::Memory(Memory {
                op: MemoryOp::Store,
                size: Size::Word,
                rn: 7,
                rt: 6,
                addressing: Addressing::Unscaled(-4),
                signed: SignExtend::None,
                ordered: None,
            }))
        );
        for (word, op, size, rn, rt, rt2, addressing) in [
            (
                0xa9bf53f3,
                MemoryOp::Store,
                Size::Double,
                31,
                19,
                20,
                Addressing::PreIndex(-16),
            ),
            (
                0xa88153f3,
                MemoryOp::Store,
                Size::Double,
                31,
                19,
                20,
                Addressing::PostIndex(16),
            ),
            (
                0xa9015bf5,
                MemoryOp::Store,
                Size::Double,
                31,
                21,
                22,
                Addressing::Offset(16),
            ),
            (
                0xa9fe6879,
                MemoryOp::Load,
                Size::Double,
                3,
                25,
                26,
                Addressing::PreIndex(-32),
            ),
            (
                0xa8c2709b,
                MemoryOp::Load,
                Size::Double,
                4,
                27,
                28,
                Addressing::PostIndex(32),
            ),
            (
                0xa94478bd,
                MemoryOp::Load,
                Size::Double,
                5,
                29,
                30,
                Addressing::Offset(64),
            ),
            (
                0xa8300c02,
                MemoryOp::Store,
                Size::Double,
                0,
                2,
                3,
                Addressing::Offset(-256),
            ),
            (
                0x29410861,
                MemoryOp::Load,
                Size::Word,
                3,
                1,
                2,
                Addressing::Offset(8),
            ),
        ] {
            assert_eq!(
                decode(word),
                Ok(Instruction::Pair(Pair {
                    op,
                    size,
                    rn,
                    rt,
                    rt2,
                    addressing,
                    signed: SignExtend::None,
                    // 0xa8300c02 is `stnp x2, x3, [x0, #-256]`; the rest are `ldp`/`stp`.
                    non_temporal: word == 0xa8300c02,
                }))
            );
        }
    }
    #[test]
    fn assembler_system_register_vectors() {
        use SystemRegister::*;
        for (word, register) in [
            (0xd5384100, SpEl0),
            (0xd53c4100, SpEl1),
            (0xd53e4100, SpEl2),
            (0xd5384240, CurrentEl),
            (0xd5384000, SpsrEl1),
            (0xd5384020, ElrEl1),
            (0xd5385200, EsrEl1),
            (0xd5386000, FarEl1),
            (0xd538c000, VbarEl1),
            (0xd5381040, CpacrEl1),
            (0xd5381000, SctlrEl1),
            (0xd5382000, Ttbr0El1),
            (0xd5382020, Ttbr1El1),
            (0xd5382040, TcrEl1),
            (0xd538a200, MairEl1),
            (0xd53bd040, TpidrEl0),
            (0xd53bd060, TpidrroEl0),
            (0xd53be040, CntvctEl0),
            (0xd53b4220, Daif),
            (0xd53b4200, Nzcv),
            (0xd5390000, CcsidrEl1),
            (0xd53a0000, CsselrEl1),
        ] {
            assert_eq!(
                decode(word),
                Ok(Instruction::System(System {
                    read: true,
                    register,
                    rt: 0,
                    immediate: 0
                }))
            );
        }
        for (word, register) in [
            (0xd51c4100, SpEl1),
            (0xd51bd040, TpidrEl0),
            (0xd5184000, SpsrEl1),
            (0xd518c000, VbarEl1),
            (0xd518d020, ContextidrEl1),
            (0xd51a0000, CsselrEl1),
        ] {
            assert_eq!(
                decode(word),
                Ok(Instruction::System(System {
                    read: false,
                    register,
                    rt: 0,
                    immediate: 0
                }))
            );
        }
        // Writing the read-only CCSIDR_EL1 is unallocated.
        for word in [
            0xd5380060, 0xd5383000, 0xd53b0000, 0xd5385000, 0xd5180000, 0xd5190000,
        ] {
            assert_eq!(decode(word), Err(DecodeError::UnsupportedInstruction));
        }
        assert_eq!(
            decode(0xd50342ff),
            Ok(Instruction::System(System {
                read: false,
                register: DaifClear,
                rt: 31,
                immediate: 2
            }))
        );
    }
    #[test]
    fn arithmetic_immediate_keeps_its_shift() {
        let imm = |op, rn, rd, width, immediate, shifted| {
            Ok(Instruction::ArithImm(Arithmetic {
                op,
                flags: false,
                width,
                rn,
                rd,
                immediate,
                shifted,
            }))
        };
        // `add x0, x1, #0`, `add x0, x1, #0, lsl #12`, `add sp, x1, #0, lsl #12`.
        assert_eq!(
            decode(0x91000020),
            imm(ArithOp::Add, 1, 0, Width::X64, 0, false)
        );
        assert_eq!(
            decode(0x91400020),
            imm(ArithOp::Add, 1, 0, Width::X64, 0, true)
        );
        assert_eq!(
            decode(0x9140003f),
            imm(ArithOp::Add, 1, 31, Width::X64, 0, true)
        );
        // `add w0, w1, #3, lsl #12`, `sub x0, x1, #5, lsl #12`.
        assert_eq!(
            decode(0x11400c20),
            imm(ArithOp::Add, 1, 0, Width::W32, 3, true)
        );
        assert_eq!(
            decode(0xd1401420),
            imm(ArithOp::Sub, 1, 0, Width::X64, 5, true)
        );
        let Ok(Instruction::ArithImm(a)) = decode(0x917ffc20) else {
            panic!("add x0, x1, #0xfff, lsl #12 decodes");
        };
        assert_eq!((a.immediate, a.operand()), (0xfff, 0xfff000));
    }

    #[test]
    fn debug_and_apple_zero_registers_keep_their_coordinates() {
        use SystemRegister::*;
        for (word, register) in [
            (0xd5300080, RazWi { crm: 0, op2: 4 }),
            (0xd53000e0, RazWi { crm: 0, op2: 7 }),
            (0xd5300fe0, RazWi { crm: 15, op2: 7 }),
            (
                0xd53df600,
                AppleZero {
                    op1: 5,
                    crm: 6,
                    op2: 0,
                },
            ),
            (
                0xd53ff680,
                AppleZero {
                    op1: 7,
                    crm: 6,
                    op2: 4,
                },
            ),
            (
                0xd539fd00,
                AppleZero {
                    op1: 1,
                    crm: 13,
                    op2: 0,
                },
            ),
            (
                0xd53df120,
                AppleZero {
                    op1: 5,
                    crm: 1,
                    op2: 1,
                },
            ),
        ] {
            assert_eq!(
                decode(word),
                Ok(Instruction::System(System {
                    read: true,
                    register,
                    rt: 0,
                    immediate: 0
                })),
                "{word:#010x}"
            );
        }
    }

    #[test]
    fn authenticated_branch_space_is_not_plain_branches() {
        // Opcode bit 24 set with no authentication operand: objdump prints `.inst ...; undefined`.
        for word in [0xd71f0000, 0xd73f0000, 0xd75f0000, 0xd79f03e0, 0xd71f03e0] {
            assert_eq!(
                decode(word),
                Err(DecodeError::UnsupportedInstruction),
                "{word:#010x}"
            );
        }
        assert!(matches!(decode(0xd61f0000), Ok(Instruction::Indirect(_))));
        assert!(matches!(decode(0xd69f03e0), Ok(Instruction::Eret)));
    }

    #[test]
    fn msr_immediate_needs_rt_31_and_a_valid_immediate() {
        // `msr pan, #0` and `msr daifset, #0` with Rt = 0, PAN/SPSel with CRm >= 2 (objdump
        // spells these `msr s0_..., x0`).
        for word in [
            0xd5004080, 0xd50041a0, 0xd50340c0, 0xd50342e0, 0xd500429f, 0xd50042bf, 0xd50043bf,
            0xd500439f,
        ] {
            assert_eq!(
                decode(word),
                Err(DecodeError::UnsupportedInstruction),
                "{word:#010x}"
            );
        }
        for (word, register, immediate) in [
            (0xd500409f, SystemRegister::PanSet, 0),
            (0xd500419f, SystemRegister::PanSet, 1),
            (0xd50040bf, SystemRegister::SpSelSet, 0),
            (0xd50041bf, SystemRegister::SpSelSet, 1),
            (0xd50340df, SystemRegister::DaifSet, 0),
            (0xd5034fdf, SystemRegister::DaifSet, 15),
            (0xd50342ff, SystemRegister::DaifClear, 2),
        ] {
            assert_eq!(
                decode(word),
                Ok(Instruction::System(System {
                    read: false,
                    register,
                    rt: 31,
                    immediate
                })),
                "{word:#010x}"
            );
        }
    }

    #[test]
    fn assembler_hints_barriers_and_traps() {
        use Instruction::*;
        for (word, expected) in [
            (0xd5033f9f, Dsb(0xf)),
            (0xd5033fbf, Dmb(0xf)),
            (0xd5033fdf, Isb(0xf)),
            (0xd503201f, Nop),
            (0xd503305f, Clrex(0)),
            (0xd503203f, Yield),
            (0xd503205f, Wfe),
            (0xd503207f, Wfi),
            (0xd503209f, Sev),
            (0xd50320bf, Sevl),
            (0xd503241f, Hint(32)),
            (0xd503245f, Hint(34)),
            (0xd503249f, Hint(36)),
            (0xd50324df, Hint(38)),
            (0xd69f03e0, Eret),
            (0xd4000001, Svc(0)),
            (0xd4001001, Svc(0x80)),
            (0xd4000002, Psci),
            (0xd4200020, Brk(1)),
            (0xd4210000, Brk(0x800)),
        ] {
            assert_eq!(decode(word), Ok(expected));
        }
        assert_eq!(decode(0xd50b7423), Ok(DcZva(3)));
        assert_eq!(decode(0xd503309f), Ok(Dsb(0)));
    }
    #[test]
    fn prefetch_and_system_operations_keep_their_operands() {
        use Instruction::*;
        assert_eq!(
            decode(0xf9800420),
            Ok(Prefetch(super::Prefetch {
                op: 0,
                rn: 1,
                addressing: Addressing::Offset(8),
            }))
        );
        assert_eq!(
            decode(0xf89f8040),
            Ok(Prefetch(super::Prefetch {
                op: 0,
                rn: 2,
                addressing: Addressing::Unscaled(-8),
            }))
        );
        assert_eq!(
            decode(0xf8a47860),
            Ok(Prefetch(super::Prefetch {
                op: 0,
                rn: 3,
                addressing: Addressing::Register {
                    rm: 4,
                    extend: Extend::Uxtx,
                    amount: Some(3),
                },
            }))
        );
        assert_eq!(
            decode(0xd8000040),
            Ok(PrefetchLiteral(super::PrefetchLiteral { op: 0, offset: 8 }))
        );
        assert_eq!(
            decode(0xd50b7e20),
            Ok(CacheOp(super::SysOp {
                op1: 3,
                crn: 7,
                crm: 14,
                op2: 1,
                rt: 0,
            }))
        );
        assert_eq!(
            decode(0xd508871f),
            Ok(Tlbi(super::SysOp {
                op1: 0,
                crn: 8,
                crm: 7,
                op2: 0,
                rt: 31,
            }))
        );
    }
    #[test]
    fn logical_immediates_wrap_and_replicate_without_crossing_elements() {
        for (word, immediate) in [
            (0x121c3c21, 0xffff0),
            (0x121c6c21, 0xfffffff0),
            (0x121c7021, 0xfffffff1),
            (0x121f7821, 0xfffffffe),
            (0x92400021, 1),
            (0x92410021, 0x8000_0000_0000_0000),
            (0x9200f021, 0x5555_5555_5555_5555),
            (0x9201f021, 0xaaaa_aaaa_aaaa_aaaa),
        ] {
            let Instruction::LogicImm(logic) = decode(word).unwrap() else {
                panic!("not a logical immediate")
            };
            assert_eq!(logic.immediate, immediate);
        }
        for word in [0x9240fc21, 0x12400021, 0x1200fc21] {
            assert_eq!(decode(word), Err(DecodeError::UnsupportedInstruction));
        }
    }
    #[test]
    fn reserved_integer_memory_and_branch_encodings_are_rejected() {
        for word in [
            0x52c00000,              // MOVZ W with halfword 2
            0x32800000,              // reserved wide opcode
            0x8bc00000,              // add rotated register
            0x0b008000,              // shift >= W width
            0x8b201400,              // extended shift 5
            0x91400000 | 0x00800000, // reserved immediate shift
            0x13400000,              // bitfield N disagrees with width
            0x13808000,              // W EXTR bit position 32
            0x5400000e,              // unconditional condition in conditional branch
            0xd61f0001,              // reserved low branch bits
            0xb8c00000,              // reserved signed word to W load
            0xf8620800,              // unsupported index extension
            0x5ac00c00,              // W REV64
            0x1ac00400,              // unsupported variable opcode
            0xd50330ff,              // unsupported barrier opcode
        ] {
            assert_eq!(
                decode(word),
                Err(DecodeError::UnsupportedInstruction),
                "{word:08x}"
            );
        }
    }
    #[test]
    fn simd_immediate_zeroes_vector_registers() {
        let Instruction::SimdImm(simd) = decode(0x4f00041f).unwrap() else {
            panic!("not a simd immediate")
        };
        assert_eq!(simd.rd, 31);
        assert_eq!(simd.low, 0);
        assert_eq!(simd.high, 0);
    }
    #[test]
    fn simd_modified_immediates_follow_the_architectural_expansion() {
        // (word, low, high, combine): the vector `movi`/`mvni`/`orr`/`bic`/`fmov` families.
        use ImmCombine::*;
        for (word, low, high, combine) in [
            // `movi v0.2d, #0` is cmode 1110 with op 1: a per-bit byte mask, not a NOT.
            (0x6f00e400, 0, 0, Set),
            (
                0x6f05e540,
                0xff00_ff00_ff00_ff00,
                0xff00_ff00_ff00_ff00,
                Set,
            ),
            // `movi d1, #imm` is the scalar form: the upper half is cleared.
            (0x2f04e421, 0xff00_0000_0000_00ff, 0, Set),
            (
                0x4f02e6a1,
                0x5555_5555_5555_5555,
                0x5555_5555_5555_5555,
                Set,
            ),
            (
                0x6f000422,
                0xffff_fffe_ffff_fffe,
                0xffff_fffe_ffff_fffe,
                SetInverted,
            ),
            (0x2f006489, 0xfbff_ffff_fbff_ffff, 0, SetInverted),
            (
                0x4f00a468,
                0x0300_0300_0300_0300,
                0x0300_0300_0300_0300,
                Set,
            ),
            // The ones-shifting `msl` forms.
            (
                0x4f00c427,
                0x0000_01ff_0000_01ff,
                0x0000_01ff_0000_01ff,
                Set,
            ),
            (
                0x4f00d427,
                0x0001_ffff_0001_ffff,
                0x0001_ffff_0001_ffff,
                Set,
            ),
            (
                0x4f03f605,
                0x3f80_0000_3f80_0000,
                0x3f80_0000_3f80_0000,
                Set,
            ),
            (
                0x6f04f406,
                0xc000_0000_0000_0000,
                0xc000_0000_0000_0000,
                Set,
            ),
            // `orr`/`bic` combine with the register instead of replacing it.
            (0x4f001443, 0x0000_0002_0000_0002, 0x0000_0002_0000_0002, Or),
            (
                0x6f00b424,
                0x0100_0100_0100_0100,
                0x0100_0100_0100_0100,
                AndNot,
            ),
        ] {
            let Ok(Instruction::SimdImm(simd)) = decode(word) else {
                panic!("{word:08x} is not a simd immediate")
            };
            assert_eq!(
                (simd.low, simd.high, simd.combine),
                (low, high, combine),
                "{word:08x}"
            );
        }
        // `fmov v.2d` needs Q = 1; `fmov d, #imm` is a different instruction.
        assert_eq!(decode(0x2f04f406), Err(DecodeError::UnsupportedInstruction));
        // `ushll.8h v1, v0.8b, #0`, `shl.4s` and `sshr.16b` share the class bits of a
        // modified immediate but have a non-zero `immh`; they are not `movi`.
        for word in [0x2f08a401, 0x4f255420, 0x4f090420, 0x0f0c8420] {
            assert!(
                !matches!(decode(word), Ok(Instruction::SimdImm(_))),
                "{word:08x}"
            );
        }
    }
    #[test]
    fn simd_modified_immediates_keep_their_encoding_class() {
        use ImmForm::*;
        // (word, form, imm8, q); every word checked against GNU `objdump`.
        for (word, form, imm8, q) in [
            (0x4f000400, Word { shift: 0 }, 0, true),
            (0x0f000400, Word { shift: 0 }, 0, false),
            (0x6f00e401, Double, 0, true),
            (0x2f00e400, Double, 0, false),
            (0x4f00e400, Byte, 0, true),
            (0x0f00e460, Byte, 3, false),
            (0x6f0707ff, Word { shift: 0 }, 0xff, true),
            (0x6f00c400, WordMask { shift: 8 }, 0, true),
            (0x4f00d420, WordMask { shift: 16 }, 1, true),
            (0x0f006400, Word { shift: 24 }, 0, false),
            (0x4f00a420, Half { shift: 8 }, 1, true),
            (0x0f008420, Half { shift: 0 }, 1, false),
            (0x4f00f400, FmovSingle, 0, true),
            (0x6f00f400, FmovDouble, 0, true),
            (0x4f00fc00, FmovHalf, 0, true),
            (0x0f00fc00, FmovHalf, 0, false),
        ] {
            let Ok(Instruction::SimdImm(simd)) = decode(word) else {
                panic!("{word:08x} is not a simd immediate")
            };
            assert_eq!(
                (simd.form, simd.imm8, simd.q),
                (form, imm8, q),
                "{word:08x}"
            );
        }
        // `mvni` and `movi` with the same pattern differ in `combine`.
        let Ok(Instruction::SimdImm(mvni)) = decode(0x6f00045f) else {
            panic!()
        };
        assert_eq!(mvni.combine, ImmCombine::SetInverted);
        // `fmov v.4h, #2.0` expands to the half-precision pattern in every lane.
        let Ok(Instruction::SimdImm(half)) = decode(0x0f00fc00) else {
            panic!()
        };
        assert_eq!((half.low, half.high), (0x4000_4000_4000_4000, 0));
        // Unallocated in the modified-immediate class (`objdump`: `.inst`, undefined):
        // bit 11 set outside the half-precision `fmov`, `fmov .2d` without `Q`, and
        // `op` set with the half-precision form.
        for word in [0x0f000ced, 0x4f00af48, 0x0f009c28, 0x2f00f420, 0x6f00fc20] {
            assert_eq!(
                decode(word),
                Err(DecodeError::UnsupportedInstruction),
                "{word:08x}"
            );
        }
    }
    #[test]
    fn signed_branches_and_address_generation_preserve_negative_offsets() {
        assert_eq!(decode(0x17ffffff), Ok(Instruction::B(-4)));
        assert_eq!(
            decode(0x54ffffe0),
            Ok(Instruction::BCond(BranchCondition {
                cond: Condition::Eq,
                offset: -4
            }))
        );
        assert_eq!(
            decode(0x10ffffe0),
            Ok(Instruction::Adr(Address {
                page: false,
                rd: 0,
                offset: -4
            }))
        );
        assert_eq!(
            decode(0x90ffffe0),
            Ok(Instruction::Adr(Address {
                page: true,
                rd: 0,
                offset: -16384
            }))
        );
    }
    #[test]
    fn test_decode_stp_q() {
        let insn = decode(0xad05ffbf).unwrap();
        let Instruction::SimdPair(pair) = insn else {
            panic!("expected SimdPair, got {insn:?}");
        };
        assert_eq!(pair.bytes, 16);
        assert_eq!(pair.rt, 31);
        assert_eq!(pair.rt2, 31);
        assert_eq!(pair.rn, 29);
        assert_eq!(pair.op, MemoryOp::Store);
        assert_eq!(pair.addressing, Addressing::Offset(176));
    }
    #[test]
    fn test_decode_dup_general() {
        let insn = decode(0x4e010c20).unwrap();
        let Instruction::SimdDup { rd, rn, esize, q } = insn else {
            panic!("expected SimdDup, got {insn:?}");
        };
        assert_eq!(rd, 0);
        assert_eq!(rn, 1);
        assert_eq!(esize, 0);
        assert!(q);
    }
    #[test]
    fn test_decode_ext() {
        let insn = decode(0x6e0043bf).unwrap();
        let Instruction::SimdExt(ext) = insn else {
            panic!("expected SimdExt, got {insn:?}");
        };
        assert_eq!(ext.rd, 31);
        assert_eq!(ext.rn, 29);
        assert_eq!(ext.rm, 0);
        assert_eq!(ext.imm, 8);
        assert!(ext.q);
        // `ext v0.8b, v1.8b, v2.8b, #7` has Q clear; with an index of 8 or more
        // (`objdump`: undefined) there is no `.8b` form.
        let Ok(Instruction::SimdExt(ext)) = decode(0x2e023820) else {
            panic!("not ext")
        };
        assert_eq!((ext.imm, ext.q), (7, false));
        assert_eq!(decode(0x2e164372), Err(DecodeError::UnsupportedInstruction));
    }
    /// `objdump` calls every one of these `.inst ... ; undefined` or decodes them as a
    /// different instruction (half-precision `fmla`/`fmulx`/`frecps`/`fmaxnm`, `luti2`,
    /// `luti4`, indexed multiply); none may decode as the vector move, `tbl`, ALU or
    /// immediate instruction whose mask they once satisfied.
    #[test]
    fn unallocated_or_other_simd_encodings_are_rejected() {
        for word in [
            0x4e5a0d5f, // fmla v31.8h (was `dup`)
            0x0e471c94, // fmulx v20.4h (was `mov v.b[n], w`)
            0x0e5b3cd8, // frecps v24.4h (was `umov`)
            0x4e4105db, // fmaxnm v27.8h (was `dup` element)
            0x4e8b21d7, // undefined table encoding
            0x4e86516b, // luti2
            0x4e5c136f, // luti4
            0x6eee6d74, // umin .2d
            0x4efb2521, // shsub .2d
            0x4ee6a4d4, // smaxp .2d
            0x4ef99d5a, // mul .2d
            0x0e1c1fba, // ins (general) with Q clear
            0x4e013c20, // umov w, v.b with Q set
            0x0e083c20, // umov x, v.d with Q clear
            0x4fab1d6c, // indexed-element class, bit 24 set
            0x2e5a0d5f, // fp16 class under U=1
            0x6e000420, // dup/ins element with imm5 = 0
        ] {
            assert_eq!(
                decode(word),
                Err(DecodeError::UnsupportedInstruction),
                "{word:08x}"
            );
        }
    }
    #[test]
    fn decodes_bitwise_three_same_with_the_64_bit_form_selected_by_size() {
        // `bif v0.8b, v31.8b, v30.8b` and `orn v0.8b, v1.8b, v0.8b`: `size` is part of
        // the opcode, so `size == 3` with `Q` clear is valid here.
        for (word, op) in [(0x2efe1fe0, SimdAluOp::Bif), (0x0ee01c20, SimdAluOp::Orn)] {
            let Ok(Instruction::SimdAlu(alu)) = decode(word) else {
                panic!("{word:08x}")
            };
            assert_eq!((alu.op, alu.size, alu.q), (op, 3, false), "{word:08x}");
        }
    }
    #[test]
    fn decodes_single_structure_simd_memory() {
        assert_eq!(
            decode(0x4c407020),
            Ok(Instruction::SimdMemory(SimdMemory {
                op: MemoryOp::Load,
                bytes: 16,
                rn: 1,
                rt: 0,
                addressing: Addressing::Offset(0),
                access: SimdAccess::Ld1 { esize: 1 },
            }))
        );
        assert_eq!(
            decode(0x4c9f7020),
            Ok(Instruction::SimdMemory(SimdMemory {
                op: MemoryOp::Store,
                bytes: 16,
                rn: 1,
                rt: 0,
                addressing: Addressing::PostIndex(16),
                access: SimdAccess::Ld1 { esize: 1 },
            }))
        );
    }
    #[test]
    fn only_architecturally_undefined_words_are_undefined() {
        // `udf #0`, XNU's `TRAP_DEBUGGER`, and an SVE word (the CPU has no SVE).
        for word in [0x0000_0000, 0x0000_ffff, 0xe7ff_deff, 0x0420_0000] {
            assert!(is_undefined(word), "{word:08x}");
            assert!(decode(word).is_err(), "{word:08x} is not an instruction");
        }
        // Valid instructions, and an instruction that is real but not implemented
        // (half-precision `fadd`), are not undefined.
        for word in [0xd503_201f, 0x8b00_0000, 0x1ee2_2820] {
            assert!(!is_undefined(word), "{word:08x}");
        }
    }
    #[test]
    fn exclusive_pairs_decode_with_both_transfer_registers() {
        let pair = |kind, size, rs, rn, rt, rt2| {
            Ok(Instruction::Atomic(Atomic {
                kind,
                size,
                rs,
                rn,
                rt,
                rt2,
                order: MemoryOrder::default(),
            }))
        };
        // ldxp x9, x21, [x8]; stxp w11, x19, x10, [x8]; ldxp w0, wzr, [x0]
        // (verified against llvm-mc).
        assert_eq!(
            decode(0xc87f5509),
            pair(AtomicKind::LoadPair, Size::Double, 31, 8, 9, 21)
        );
        assert_eq!(
            decode(0xc82b2913),
            pair(AtomicKind::StorePair, Size::Double, 11, 8, 19, 10)
        );
        assert_eq!(
            decode(0x887f7c00),
            pair(AtomicKind::LoadPair, Size::Word, 31, 0, 0, 31)
        );
        // The load form has no status register; byte and halfword pairs do not exist.
        assert!(decode(0xc87e5509).is_err());
        assert!(decode(0x487f5509).is_err());
    }
    #[test]
    fn structure_loads_and_stores_decode_to_descriptors() {
        use StructPost::*;
        // (word, store, shape, registers, interleaved, esize, lane, q, rt, rn, post),
        // every word verified against llvm-mc.
        let cases = [
            (
                0x4c400044,
                false,
                Shape::Multiple,
                4,
                true,
                1,
                0,
                true,
                4,
                2,
                None,
            ), // ld4.16b
            (
                0x4cdf4c20,
                false,
                Shape::Multiple,
                3,
                true,
                8,
                0,
                true,
                0,
                1,
                Immediate,
            ), // ld3.2d, #48
            (
                0x4c9f8c7e,
                true,
                Shape::Multiple,
                2,
                true,
                8,
                0,
                true,
                30,
                3,
                Immediate,
            ), // st2.2d
            (
                0x0c408481,
                false,
                Shape::Multiple,
                2,
                true,
                2,
                0,
                false,
                1,
                4,
                None,
            ), // ld2.4h
            // LD1 of whole registers is a structure access too; only with one register
            // and no register post-index does it stay a `SimdMemory`.
            (
                0x4c402d50,
                false,
                Shape::Multiple,
                4,
                false,
                8,
                0,
                true,
                16,
                10,
                None,
            ),
            (
                0x4cc2a800,
                false,
                Shape::Multiple,
                2,
                false,
                4,
                0,
                true,
                0,
                0,
                Register(2),
            ),
            (
                0x0d4090a3,
                false,
                Shape::Lane,
                1,
                false,
                4,
                1,
                false,
                3,
                5,
                None,
            ), // ld1.s[1]
            (
                0x4ddf84a3,
                false,
                Shape::Lane,
                1,
                false,
                8,
                1,
                true,
                3,
                5,
                Immediate,
            ), // ld1.d[1]
            (
                0x4dc61ca3,
                false,
                Shape::Lane,
                1,
                false,
                1,
                15,
                true,
                3,
                5,
                Register(6),
            ), // ld1.b[15]
            (
                0x4d4058a3,
                false,
                Shape::Lane,
                1,
                false,
                2,
                7,
                true,
                3,
                5,
                None,
            ), // ld1.h[7]
            (
                0x4d40cc20,
                false,
                Shape::Replicate,
                1,
                false,
                8,
                0,
                true,
                0,
                1,
                None,
            ), // ld1r.2d
            (
                0x0ddfc020,
                false,
                Shape::Replicate,
                1,
                false,
                1,
                0,
                false,
                0,
                1,
                Immediate,
            ), // ld1r.8b
            (
                0x4d40e820,
                false,
                Shape::Replicate,
                3,
                false,
                4,
                0,
                true,
                0,
                1,
                None,
            ), // ld3r.4s
            (
                0x4d60a020,
                false,
                Shape::Lane,
                4,
                false,
                4,
                2,
                true,
                0,
                1,
                None,
            ), // ld4.s[2]
            (
                0x4d9fb020,
                true,
                Shape::Lane,
                3,
                false,
                4,
                3,
                true,
                0,
                1,
                Immediate,
            ), // st3.s[3]
            (
                0x0d008020,
                true,
                Shape::Lane,
                1,
                false,
                4,
                0,
                false,
                0,
                1,
                None,
            ), // st1.s[0]
        ];
        for (word, store, shape, registers, interleaved, esize, lane, q, rt, rn, post) in cases {
            let expected = Instruction::SimdStruct(SimdStruct {
                desc: StructDesc {
                    store,
                    shape,
                    registers,
                    interleaved,
                    esize,
                    lane,
                    q,
                    rt,
                },
                rn,
                post,
            });
            assert_eq!(decode(word), Ok(expected), "{word:08x}");
        }
        // A single register with no or an immediate post-index stays the inlinable form.
        assert!(matches!(decode(0x4c407020), Ok(Instruction::SimdMemory(_))));
        assert!(matches!(decode(0x4c9f7020), Ok(Instruction::SimdMemory(_))));
        // `ld2`..`ld4` of `.1d` and a stray Rm without post-index are reserved.
        assert!(decode(0x0c408c00).is_err());
        assert!(decode(0x4c428c00).is_err());
        // `st1r` does not exist: replicate is a load.
        assert!(decode(0x4d00c020).is_err());
    }
    #[test]
    fn decodes_simd_compare_zero() {
        assert_eq!(
            decode(0x4e209801),
            Ok(Instruction::SimdCompareZero(SimdCompareZero {
                rd: 1,
                rn: 0,
                size: 0,
                q: true,
            }))
        );
        // Half-width form zeroes the upper doubleword at runtime.
        assert_eq!(
            decode(0x0e209801),
            Ok(Instruction::SimdCompareZero(SimdCompareZero {
                rd: 1,
                rn: 0,
                size: 0,
                q: false,
            }))
        );
    }
    #[test]
    fn decodes_simd_fmov() {
        // fmov x2, d2 / fmov d2, x2 / fmov w2, s1 / fmov s1, w2.
        assert_eq!(
            decode(0x9e660042),
            Ok(Instruction::SimdFmov(SimdFmov {
                rd: 2,
                rn: 2,
                to_fp: false,
                double: true,
                upper: false,
            }))
        );
        assert_eq!(
            decode(0x9e670042),
            Ok(Instruction::SimdFmov(SimdFmov {
                rd: 2,
                rn: 2,
                to_fp: true,
                double: true,
                upper: false,
            }))
        );
        assert_eq!(
            decode(0x1e260022),
            Ok(Instruction::SimdFmov(SimdFmov {
                rd: 2,
                rn: 1,
                to_fp: false,
                double: false,
                upper: false,
            }))
        );
        assert_eq!(
            decode(0x1e270041),
            Ok(Instruction::SimdFmov(SimdFmov {
                rd: 1,
                rn: 2,
                to_fp: true,
                double: false,
                upper: false,
            }))
        );
        // `fmov v0.d[1], x3` / `fmov x3, v0.d[1]` move the upper doubleword.
        assert_eq!(
            decode(0x9eaf0060),
            Ok(Instruction::SimdFmov(SimdFmov {
                rd: 0,
                rn: 3,
                to_fp: true,
                double: true,
                upper: true,
            }))
        );
        assert_eq!(
            decode(0x9eae0003),
            Ok(Instruction::SimdFmov(SimdFmov {
                rd: 3,
                rn: 0,
                to_fp: false,
                double: true,
                upper: true,
            }))
        );
    }
    #[test]
    fn decodes_table_vector_lookup() {
        assert_eq!(
            decode(0x4e020020),
            Ok(Instruction::SimdTable(SimdTable {
                rd: 0,
                rn: 1,
                rm: 2,
                len: 0,
                extend: false,
                q: true,
            }))
        );
        assert_eq!(
            decode(0x4e032020),
            Ok(Instruction::SimdTable(SimdTable {
                rd: 0,
                rn: 1,
                rm: 3,
                len: 1,
                extend: false,
                q: true,
            }))
        );
        assert_eq!(
            decode(0x4e021020),
            Ok(Instruction::SimdTable(SimdTable {
                rd: 0,
                rn: 1,
                rm: 2,
                len: 0,
                extend: true,
                q: true,
            }))
        );
        assert_eq!(
            decode(0x0e020020),
            Ok(Instruction::SimdTable(SimdTable {
                rd: 0,
                rn: 1,
                rm: 2,
                len: 0,
                extend: false,
                q: false,
            }))
        );
    }
    #[test]
    fn decodes_integer_three_same_vectors() {
        use SimdAluOp::*;
        // Encodings below are `llvm-mc` output for v0/v1/v2 `.16b` forms.
        let cases: [(u32, SimdAluOp); 41] = [
            (0x4e228420, Add),
            (0x6e228420, Sub),
            (0x4e221c20, And),
            (0x4ea21c20, Orr),
            (0x6e221c20, Eor),
            (0x6e228c20, CmEq),
            (0x4e223420, CmGt),
            (0x4e223c20, CmGe),
            (0x6e223420, CmHi),
            (0x6e223c20, CmHs),
            (0x4e228c20, CmTst),
            (0x6e226c20, UMin),
            (0x6e226420, UMax),
            (0x4e226c20, SMin),
            (0x4e226420, SMax),
            (0x6e227420, UAbd),
            (0x4e227420, SAbd),
            (0x4e229c20, Mul),
            (0x4e229420, Mla),
            (0x6e229420, Mls),
            (0x4e220c20, SQAdd),
            (0x6e220c20, UQAdd),
            (0x4e222c20, SQSub),
            (0x6e222c20, UQSub),
            (0x4e220420, SHAdd),
            (0x6e220420, UHAdd),
            (0x4e224420, SSHl),
            (0x6e224420, USHl),
            (0x4e621c20, Bic),
            (0x4ee21c20, Orn),
            (0x4e222420, SHSub),
            (0x6e222420, UHSub),
            (0x4e221420, SRHAdd),
            (0x6e221420, URHAdd),
            (0x4e224c20, SQShl),
            (0x6e224c20, UQShl),
            (0x4e225c20, SQRShl),
            (0x6e225c20, UQRShl),
            (0x4e31b820, AddV),
            (0x4e71b820, AddV),
            (0x4eb1b820, AddV),
        ];
        for (word, op) in cases {
            let decoded = decode(word).unwrap_or_else(|e| panic!("{word:08x}: {e:?}"));
            let Instruction::SimdAlu(alu) = decoded else {
                panic!("{word:08x}: not a vector ALU op: {decoded:?}");
            };
            assert_eq!(alu.op, op, "{word:08x}");
            assert_eq!((alu.rn, alu.rd), (1, 0), "{word:08x}");
            assert!(alu.q, "{word:08x}");
            if op != AddV {
                assert_eq!(alu.rm, 2, "{word:08x}");
            }
        }
    }
    fn memory(op: MemoryOp, size: Size, rt: u8, rn: u8, addressing: Addressing) -> Memory {
        Memory {
            op,
            size,
            rn,
            rt,
            addressing,
            signed: SignExtend::None,
            ordered: None,
        }
    }
    #[test]
    fn ordered_memory_accesses_keep_their_class() {
        use MemoryOp::{Load, Store};
        let ordered = |op, size, rt, rn, ordered| {
            Ok(Instruction::Memory(Memory {
                ordered: Some(ordered),
                ..memory(op, size, rt, rn, Addressing::Offset(0))
            }))
        };
        // ldar w0, [x1]; ldar x0, [sp]; ldarb w0, [x1]; stlr x0, [sp]; stlrh w0, [x1]
        assert_eq!(
            decode(0x88dffc20),
            ordered(Load, Size::Word, 0, 1, Ordered::Acquire)
        );
        assert_eq!(
            decode(0xc8dfffe0),
            ordered(Load, Size::Double, 0, 31, Ordered::Acquire)
        );
        assert_eq!(
            decode(0x08dffc20),
            ordered(Load, Size::Byte, 0, 1, Ordered::Acquire)
        );
        assert_eq!(
            decode(0xc89fffe0),
            ordered(Store, Size::Double, 0, 31, Ordered::Release)
        );
        assert_eq!(
            decode(0x489ffc20),
            ordered(Store, Size::Half, 0, 1, Ordered::Release)
        );
        // ldlar x0, [x1]; stllrb w0, [x1]
        assert_eq!(
            decode(0xc8df7c20),
            ordered(Load, Size::Double, 0, 1, Ordered::LimitedAcquire)
        );
        assert_eq!(
            decode(0x089f7c20),
            ordered(Store, Size::Byte, 0, 1, Ordered::LimitedRelease)
        );
        // ldapr w0, [x1]; ldaprh w0, [x1]
        assert_eq!(
            decode(0xb8bfc020),
            ordered(Load, Size::Word, 0, 1, Ordered::AcquirePc)
        );
        assert_eq!(
            decode(0x78bfc020),
            ordered(Load, Size::Half, 0, 1, Ordered::AcquirePc)
        );
        // The unscaled RCpc forms: ldapur w0, [x1, #4]; stlur w0, [x1, #4];
        // ldapursb x0, [x1, #4].
        assert_eq!(
            decode(0x99404020),
            Ok(Instruction::Memory(Memory {
                ordered: Some(Ordered::AcquirePc),
                ..memory(Load, Size::Word, 0, 1, Addressing::Unscaled(4))
            }))
        );
        assert_eq!(
            decode(0x99004020),
            Ok(Instruction::Memory(Memory {
                ordered: Some(Ordered::Release),
                ..memory(Store, Size::Word, 0, 1, Addressing::Unscaled(4))
            }))
        );
        assert_eq!(
            decode(0x19804020),
            Ok(Instruction::Memory(Memory {
                signed: SignExtend::To64,
                ordered: Some(Ordered::AcquirePc),
                ..memory(Load, Size::Byte, 0, 1, Addressing::Unscaled(4))
            }))
        );
    }
    #[test]
    fn exclusives_and_atomics_keep_acquire_and_release() {
        let exclusive = |op, size, rs, rt, acquire, release| {
            Ok(Instruction::Exclusive(Exclusive {
                op,
                size,
                rn: 1,
                rt,
                rs,
                order: MemoryOrder { acquire, release },
            }))
        };
        // ldxr w0, [x1]; ldaxr w0, [x1]; stxr w2, w0, [x1]; stlxr w2, w0, [x1]
        assert_eq!(
            decode(0x885f7c20),
            exclusive(MemoryOp::Load, Size::Word, 31, 0, false, false)
        );
        assert_eq!(
            decode(0x885ffc20),
            exclusive(MemoryOp::Load, Size::Word, 31, 0, true, false)
        );
        assert_eq!(
            decode(0x88027c20),
            exclusive(MemoryOp::Store, Size::Word, 2, 0, false, false)
        );
        assert_eq!(
            decode(0x8802fc20),
            exclusive(MemoryOp::Store, Size::Word, 2, 0, false, true)
        );
        let atomic = |kind, size, rs, rt, rt2, acquire, release| {
            Ok(Instruction::Atomic(Atomic {
                kind,
                size,
                rs,
                rn: 2,
                rt,
                rt2,
                order: MemoryOrder { acquire, release },
            }))
        };
        // cas w0, w1, [x2]; casa; casl; casal x0, x1, [x2]
        for (word, size, acquire, release) in [
            (0x88a07c41, Size::Word, false, false),
            (0x88e07c41, Size::Word, true, false),
            (0x88a0fc41, Size::Word, false, true),
            (0xc8e0fc41, Size::Double, true, true),
        ] {
            assert_eq!(
                decode(word),
                atomic(AtomicKind::Cas, size, 0, 1, 0, acquire, release),
                "{word:#010x}"
            );
        }
        // casp, caspa, caspl, caspal x0, x1, x2, x3, [x4] share one kind.
        let casp = |word, acquire, release| {
            assert_eq!(
                decode(word),
                Ok(Instruction::Atomic(Atomic {
                    kind: AtomicKind::Casp,
                    size: Size::Double,
                    rs: 0,
                    rn: 4,
                    rt: 2,
                    rt2: 0,
                    order: MemoryOrder { acquire, release },
                })),
                "{word:#010x}"
            );
        };
        casp(0x48207c82, false, false);
        casp(0x48607c82, true, false);
        casp(0x4820fc82, false, true);
        casp(0x4860fc82, true, true);
        // swpal x0, xzr, [x2]; ldadda w0, wzr, [x1] and ldaddl w0, w1, [x2]
        assert_eq!(
            decode(0xf8e0805f),
            atomic(AtomicKind::Swap, Size::Double, 0, 31, 0, true, true)
        );
        assert_eq!(
            decode(0xb8a0003f),
            Ok(Instruction::Atomic(Atomic {
                kind: AtomicKind::Add,
                size: Size::Word,
                rs: 0,
                rn: 1,
                rt: 31,
                rt2: 0,
                order: MemoryOrder {
                    acquire: true,
                    release: false
                },
            }))
        );
        assert_eq!(
            decode(0xb8600041),
            atomic(AtomicKind::Add, Size::Word, 0, 1, 0, false, true)
        );
        // ldaxp w0, w1, [x2]; stlxp w3, x0, x1, [x2]
        assert_eq!(
            decode(0x887f8440),
            atomic(AtomicKind::LoadPair, Size::Word, 31, 0, 1, true, false)
        );
        assert_eq!(
            decode(0xc8238440),
            atomic(AtomicKind::StorePair, Size::Double, 3, 0, 1, false, true)
        );
    }
    #[test]
    fn offset_forms_keep_their_encoding() {
        use MemoryOp::Load;
        // ldr x0, [x1, #8]; ldur x0, [x1, #8]; ldtr w0, [x1, #4]; ldur w0, [x1]
        assert_eq!(
            decode(0xf9400420),
            Ok(Instruction::Memory(memory(
                Load,
                Size::Double,
                0,
                1,
                Addressing::Offset(8)
            )))
        );
        assert_eq!(
            decode(0xf8408020),
            Ok(Instruction::Memory(memory(
                Load,
                Size::Double,
                0,
                1,
                Addressing::Unscaled(8)
            )))
        );
        assert_eq!(
            decode(0xb8404820),
            Ok(Instruction::Memory(memory(
                Load,
                Size::Word,
                0,
                1,
                Addressing::Unprivileged(4)
            )))
        );
        assert_eq!(
            decode(0xb8400020),
            Ok(Instruction::Memory(memory(
                Load,
                Size::Word,
                0,
                1,
                Addressing::Unscaled(0)
            )))
        );
        // ldr q0, [x1]; ldur q0, [x1, #16]; ldur b0, [x1, #1]; ldr b0, [x1, #1]
        let simd = |word, bytes, addressing| {
            assert_eq!(
                decode(word),
                Ok(Instruction::SimdMemory(SimdMemory {
                    op: MemoryOp::Load,
                    bytes,
                    rn: 1,
                    rt: 0,
                    addressing,
                    access: SimdAccess::Register,
                })),
                "{word:#010x}"
            );
        };
        simd(0x3dc00020, 16, Addressing::Offset(0));
        simd(0x3cc10020, 16, Addressing::Unscaled(16));
        simd(0x3c401020, 1, Addressing::Unscaled(1));
        simd(0x3d400420, 1, Addressing::Offset(1));
        // prfm pldl1keep, [x1, #8]; prfum pldl1keep, [x1, #8]
        assert_eq!(
            decode(0xf9800420),
            Ok(Instruction::Prefetch(Prefetch {
                op: 0,
                rn: 1,
                addressing: Addressing::Offset(8)
            }))
        );
        assert_eq!(
            decode(0xf8808020),
            Ok(Instruction::Prefetch(Prefetch {
                op: 0,
                rn: 1,
                addressing: Addressing::Unscaled(8),
            }))
        );
    }
    #[test]
    fn register_offsets_keep_whether_they_are_shifted() {
        let register = |word, extend, amount| {
            assert_eq!(
                decode(word),
                Ok(Instruction::Memory(memory(
                    MemoryOp::Load,
                    Size::Byte,
                    0,
                    1,
                    Addressing::Register {
                        rm: 2,
                        extend,
                        amount
                    }
                ))),
                "{word:#010x}"
            );
        };
        // ldrb w0, [x1, x2, lsl #0]; ldrb w0, [x1, x2]; ldrb w0, [x1, w2, sxtw #0];
        // ldrb w0, [x1, w2, sxtw]
        register(0x38627820, Extend::Uxtx, Some(0));
        register(0x38626820, Extend::Uxtx, None);
        register(0x3862d820, Extend::Sxtw, Some(0));
        register(0x3862c820, Extend::Sxtw, None);
        // str b2, [x11, x0, lsl #0]; str q2, [x11, x0, lsl #4]; str q2, [x11, x0]
        let simd = |word, bytes, amount| {
            assert_eq!(
                decode(word),
                Ok(Instruction::SimdMemory(SimdMemory {
                    op: MemoryOp::Store,
                    bytes,
                    rn: 11,
                    rt: 2,
                    addressing: Addressing::Register {
                        rm: 0,
                        extend: Extend::Uxtx,
                        amount
                    },
                    access: SimdAccess::Register,
                })),
                "{word:#010x}"
            );
        };
        simd(0x3c207962, 1, Some(0));
        simd(0x3ca07962, 16, Some(4));
        simd(0x3ca06962, 16, None);
    }
    #[test]
    fn non_temporal_pairs_keep_their_identity() {
        let Ok(Instruction::Pair(pair)) = decode(0xa8400440) else {
            panic!("ldnp x0, x1, [x2]");
        };
        assert_eq!(
            (pair.op, pair.size, pair.addressing, pair.non_temporal),
            (MemoryOp::Load, Size::Double, Addressing::Offset(0), true)
        );
        let Ok(Instruction::Pair(pair)) = decode(0x28010440) else {
            panic!("stnp w0, w1, [x2, #8]");
        };
        assert_eq!(
            (pair.op, pair.size, pair.addressing, pair.non_temporal),
            (MemoryOp::Store, Size::Word, Addressing::Offset(8), true)
        );
        let Ok(Instruction::Pair(pair)) = decode(0xa9400440) else {
            panic!("ldp x0, x1, [x2]");
        };
        assert!(!pair.non_temporal);
        let Ok(Instruction::SimdPair(pair)) = decode(0xac3f0440) else {
            panic!("stnp q0, q1, [x2, #-32]");
        };
        assert_eq!(
            (pair.op, pair.bytes, pair.addressing, pair.non_temporal),
            (MemoryOp::Store, 16, Addressing::Offset(-32), true)
        );
        let Ok(Instruction::SimdPair(pair)) = decode(0xadc08440) else {
            panic!("ldp q0, q1, [x2, #16]!");
        };
        assert!(!pair.non_temporal);
    }
    #[test]
    fn one_register_ld1_keeps_its_arrangement() {
        let ld1 = |word, op, bytes, addressing, esize| {
            assert_eq!(
                decode(word),
                Ok(Instruction::SimdMemory(SimdMemory {
                    op,
                    bytes,
                    rn: 1,
                    rt: 0,
                    addressing,
                    access: SimdAccess::Ld1 { esize },
                })),
                "{word:#010x}"
            );
        };
        // ld1 {v0.8b}, [x1]; ld1 {v0.8h}, [x1]; ld1 {v0.2s}, [x1]; ld1 {v0.1d}, [x1]
        ld1(0x0c407020, MemoryOp::Load, 8, Addressing::Offset(0), 1);
        ld1(0x4c407420, MemoryOp::Load, 16, Addressing::Offset(0), 2);
        ld1(0x0c407820, MemoryOp::Load, 8, Addressing::Offset(0), 4);
        ld1(0x0c407c20, MemoryOp::Load, 8, Addressing::Offset(0), 8);
        // ld1 {v0.4s}, [x1], #16; st1 {v0.2s}, [x1], #8
        ld1(0x4cdf7820, MemoryOp::Load, 16, Addressing::PostIndex(16), 4);
        ld1(0x0c9f7820, MemoryOp::Store, 8, Addressing::PostIndex(8), 4);
        // A register post-index is a structure access.
        assert!(matches!(decode(0x4cc27020), Ok(Instruction::SimdStruct(_))));
    }
    #[test]
    fn range_prefetch_and_ldrsw_literal_decode() {
        // rprfm pldkeep, x2, [x1]; rprfm pldstrm, xzr, [sp]; rprfm #2, x2, [x1]
        for (word, op, rm, rn) in [
            (0xf8a24838, 0, 2, 1),
            (0xf8bf4bfc, 4, 31, 31),
            (0xf8a2483a, 2, 2, 1),
        ] {
            assert_eq!(
                decode(word),
                Ok(Instruction::RangePrefetch(RangePrefetch { op, rm, rn })),
                "{word:#010x}"
            );
        }
        // ldrsw x0, #8
        assert_eq!(
            decode(0x98000040),
            Ok(Instruction::Literal(Literal {
                size: Size::Word,
                rt: 0,
                offset: 8,
                signed: SignExtend::To64,
            }))
        );
    }
    #[test]
    fn unallocated_memory_words_are_rejected() {
        // Each is `.inst ... ; undefined` for GNU objdump 2.46: PRFM (register) with a
        // byte or halfword extend, LDAPR with the wrong A/R bits or a non-zero Rs, an
        // unprivileged SIMD&FP access, LDNP with the `ldpsw` opc, and LDAPURS* with an
        // impossible size.
        for word in [
            0xf8a20838, 0xf8a21838, 0xf8a28838, 0xf8a29838, 0xf8a2a838, 0xf8a20837, 0xb8ffc020,
            0xb87fc020, 0xb83fc020, 0x78ffc020, 0xb8bdc020, 0x3c400820, 0x3c000820, 0x68400440,
            0x68000440, 0x99c04020, 0xd9804020,
        ] {
            assert_eq!(
                decode(word),
                Err(DecodeError::UnsupportedInstruction),
                "{word:#010x}"
            );
        }
    }
}
