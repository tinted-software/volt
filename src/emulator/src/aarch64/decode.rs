use super::simd_struct::{Shape, StructDesc};

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
    pub immediate: u32,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Addressing {
    Offset(i64),
    Register { rm: u8, extend: Extend, amount: u8 },
    PreIndex(i64),
    PostIndex(i64),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Memory {
    pub op: MemoryOp,
    pub size: Size,
    pub rn: u8,
    pub rt: u8,
    pub addressing: Addressing,
    pub signed: SignExtend,
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
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Exclusive {
    pub op: MemoryOp,
    pub size: Size,
    pub rn: u8,
    pub rt: u8,
    pub rs: u8,
}

/// Operation of an LSE atomic. Acquire and release ordering are not modelled
/// separately: each host read-modify-write is sequentially consistent.
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
/// How a modified-immediate instruction combines its immediate with `rd`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImmCombine {
    /// `movi`, `mvni`, `fmov`: replace the register.
    Set,
    /// `orr`: `rd |= imm`.
    Or,
    /// `bic`: `rd &= !imm`.
    AndNot,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimdImmediate {
    pub rd: u8,
    /// The expanded immediate: the low and high 64 bits (the high half is zero for
    /// a 64-bit `q == false` form, which also clears the register's upper half).
    pub low: u64,
    pub high: u64,
    pub combine: ImmCombine,
    pub q: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimdPair {
    pub op: MemoryOp,
    pub bytes: u8,
    pub rn: u8,
    pub rt: u8,
    pub rt2: u8,
    pub addressing: Addressing,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimdMemory {
    pub op: MemoryOp,
    pub bytes: u8,
    pub rn: u8,
    pub rt: u8,
    pub addressing: Addressing,
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
    pub imm: u8,
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
    /// Stored, but the physical timer never fires.
    CntpCtlEl0,
    ActlrEl1,
    Nzcv,
    RazWi,
    /// Read-as-zero, write-ignored: Apple IMP-DEF status registers (`op0` 3).
    AppleZero,
    /// Apple IMP-DEF configuration register the guest reads back, stored in
    /// `System::apple[slot]`.
    Apple(u8),
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Indirect {
    pub rn: u8,
    pub link: bool,
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
    Clrex,
    Adr(Address),
    System(System),
    DcZva(u8),
    AddressTranslate(AddressTranslate),
    Eret,
    CacheOp,
    Tlbi,
    Nop,
    Dsb,
    Dmb,
    Isb,
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
    SimdMove {
        rd: u8,
        rn: u8,
    },
    SimdDup {
        rd: u8,
        rn: u8,
        esize: u8,
        q: bool,
    },
    SimdDupElement(SimdDupElement),
    /// An instruction the host executes from its raw word (see `host`).
    Host(u32),
    SimdInsert(SimdInsert),
    SimdInsertElement(SimdInsertElement),
    SimdExtract(SimdExtract),
    SimdExt(SimdExt),
    SimdAlu(SimdAlu),
    SimdTable(SimdTable),
    Svc,
    Psci,
    Wfi,
}
impl Instruction {
    /// Whether this instruction ends a translated block. With `inline_memory`,
    /// plain loads and stores are compiled with an inline fast path and no
    /// longer end the block (their slow path exits mid-block instead).
    pub fn terminates_with(&self, inline_memory: bool) -> bool {
        if inline_memory
            && matches!(
                self,
                Instruction::Memory(_)
                    | Instruction::Pair(_)
                    | Instruction::Literal(_)
                    | Instruction::SimdMemory(_)
                    | Instruction::SimdPair(_)
            )
        {
            return false;
        }
        self.terminates()
    }
    pub fn terminates(&self) -> bool {
        use Instruction::*;
        matches!(
            self,
            Memory(_)
                | Exclusive(_)
                | Atomic(_)
                | Literal(_)
                | SimdLiteral(_)
                | Pair(_)
                | SimdPair(_)
                | SimdMemory(_)
                | SimdStruct(_)
                | B(_)
                | BCond(_)
                | TestBranch(_)
                | Call(_)
                | Indirect(_)
                | Svc
                | Psci
                | Wfi
                | Wfe
                | Brk(_)
                | Isb
                | DcZva(_)
                | AddressTranslate(_)
                | Host(_)
                | Eret
                | Tlbi
        )
    }
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
/// 0x0c00_0000`) into its descriptor and post-index, or `None` for the
/// whole-register LD1/ST1 forms that `SimdMemory` handles.
fn simd_struct(word: u32) -> Result<Option<(StructDesc, StructPost)>, DecodeError> {
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
        if desc.registers == 1 && !matches!(post, StructPost::Register(_)) {
            return Ok(None);
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
    Ok(Some((desc, post)))
}
/// Number of Apple IMP-DEF configuration registers `System::apple` holds.
pub const APPLE_SLOTS: usize = 21;

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
    Some(match (op1, crm, op2) {
        (1, 0, 0) => Apple(0),
        (1, 1, 0) => Apple(1),
        (1, 2, 0) => Apple(2),
        (1, 3, 0) => Apple(3),
        (1, 4, 0) => Apple(4),
        (1, 5, 0) => Apple(5),
        (1, 6, 0) => Apple(6),
        (1, 13, 0) => AppleZero,
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
        (5, 1, 1) => AppleZero,
        (5, 3, 1) => Apple(19),
        (7, 0, 4) => Apple(20),
        (7, 6, 4) => AppleZero,
        // The S3_5_c15_cN_0 state registers early init reads, none named in the XNU
        // source tree. Read as zero until their roles are identified.
        (5, 2..=7, 0) => AppleZero,
        _ => return None,
    })
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
        return Ok(RazWi);
    }
    if let Some(register) = apple_register(op0, op1, crn, crm, op2) {
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
        0x3e21 => CntpCtlEl0,
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

pub fn decode(word: u32) -> Result<Instruction, DecodeError> {
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
        return match (word >> 21) & 7 {
            0 | 2 => Ok(Indirect(self::Indirect { rn, link: false })),
            1 => Ok(Indirect(self::Indirect { rn, link: true })),
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
            immediate: ((word >> 10) & 0xfff) << if word & 0x0040_0000 != 0 { 12 } else { 0 },
        }));
    }
    if word & 0xffc0_0000 == 0xf980_0000
        || word & 0xffe0_0c00 == 0xf8a0_0800
        || word & 0xffe0_0c00 == 0xf880_0000
        || word & 0xff00_0000 == 0xd800_0000
    {
        return Ok(Nop);
    }
    if word & 0x3f00_0000 == 0x1800_0000 {
        let size = match word >> 30 {
            0 => Size::Word,
            1 => Size::Double,
            _ => return Err(UnsupportedInstruction),
        };
        return Ok(Literal(self::Literal {
            size,
            rt: rd,
            offset: sign_extend(word >> 5, 19) << 2,
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
    // SWP share one encoding, split by o3 (bit 15) and opc (bits 14:12). LDAPR, the
    // RCpc load, is the o3=1, opc=100 slot with Rs unused.
    if word & 0x3f20_0c00 == 0x3820_0000 {
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
            (true, 4) if rm == 31 => {
                return Ok(Memory(self::Memory {
                    op: MemoryOp::Load,
                    size: size(word),
                    rn,
                    rt: rd,
                    addressing: Addressing::Offset(0),
                    signed: SignExtend::None,
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
        }));
    }
    // CAS/CASP (ARMv8.1): compare with Rs (the pair Rs, Rs+1 for CASP); store Rt on a match.
    if word & 0x3fa0_7c00 == 0x08a0_7c00 {
        return Ok(Atomic(self::Atomic {
            kind: AtomicKind::Cas,
            size: size(word),
            rs: rm,
            rn,
            rt: rd,
            rt2: 0,
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
        }));
    }
    if word & 0x3f00_0000 == 0x0800_0000 {
        if word & 0x0020_0000 != 0 {
            // LDXP/LDAXP/STXP/STLXP: a pair of words or doublewords; the load form has
            // no status register.
            let load = word & 0x0040_0000 != 0;
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
            }));
        }
        let size = size(word);
        let load = word & 0x0040_0000 != 0;
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
            if rs != 31 || (load && word & 0x8000 == 0) {
                return Err(UnsupportedInstruction);
            }
            return Ok(Memory(self::Memory {
                op,
                size,
                rn,
                rt: rd,
                addressing: Addressing::Offset(0),
                signed: SignExtend::None,
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
        }));
    }
    // LD1..LD4 / ST1..ST4 (multiple and single structures) and LD1R..LD4R.
    // A single-register LD1/ST1 without a register post-index is left to the
    // `SimdMemory` blocks below, which the JIT inlines.
    if word & 0xbe00_0000 == 0x0c00_0000
        && let Some((desc, post)) = simd_struct(word)?
    {
        return Ok(SimdStruct(self::SimdStruct { desc, rn, post }));
    }
    // LD1/ST1 (one structure): a full or half-width SIMD register.
    //
    // bits 29:24 are `001100`, bits 14:12 are `111`, bit 23 selects the
    // no-offset vs post-index addressing; bits 21:16 are the `size:opc`
    // structure field, which must be 0b000000 or 0b011111.
    if word & 0xbf00_8000 == 0x0c00_0000 && word & 0x0000_7000 == 0x0000_7000 {
        let bytes = if word & 0x4000_0000 != 0 { 16 } else { 8 };
        let post_index = word & 0x0080_0000 != 0;
        let index = (word >> 16) & 0x3f;
        let addressing = if post_index {
            if index != 31 {
                return Err(UnsupportedInstruction);
            }
            Addressing::PostIndex(bytes as i64)
        } else {
            if index != 0 {
                return Err(UnsupportedInstruction);
            }
            Addressing::Offset(0)
        };
        return Ok(SimdMemory(self::SimdMemory {
            op: if word & 0x0040_0000 != 0 {
                MemoryOp::Load
            } else {
                MemoryOp::Store
            },
            bytes: bytes as u8,
            rn,
            rt: rd,
            addressing,
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
                if word & 0x0000_0c00 != 0x800 {
                    return Err(UnsupportedInstruction);
                }
                let extend = match (word >> 13) & 7 {
                    2 => Extend::Uxtw,
                    3 => Extend::Uxtx,
                    6 => Extend::Sxtw,
                    7 => Extend::Sxtx,
                    _ => return Err(UnsupportedInstruction),
                };
                Addressing::Register {
                    rm,
                    extend,
                    amount: if word & 0x1000 != 0 {
                        (bytes as u32).trailing_zeros() as u8
                    } else {
                        0
                    },
                }
            } else {
                let offset = sign_extend(word >> 12, 9);
                match (word >> 10) & 3 {
                    0 | 2 => Addressing::Offset(offset),
                    1 => Addressing::PostIndex(offset),
                    3 => Addressing::PreIndex(offset),
                    _ => return Err(UnsupportedInstruction),
                }
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
            }));
        }
        let opc = (word >> 22) & 3;
        let size = size(word);
        let signed = match opc {
            0 | 1 => SignExtend::None,
            2 if size != Size::Double => SignExtend::To64,
            3 if size == Size::Byte || size == Size::Half => SignExtend::To32,
            _ => return Err(UnsupportedInstruction),
        };
        let addressing = if word & 0x0100_0000 != 0 {
            Addressing::Offset((((word >> 10) & 0xfff) * size as u32) as i64)
        } else if word & 0x0020_0000 != 0 {
            if word & 0x0000_0c00 != 0x800 {
                return Err(UnsupportedInstruction);
            }
            let extend = match (word >> 13) & 7 {
                2 => Extend::Uxtw,
                3 => Extend::Uxtx,
                6 => Extend::Sxtw,
                7 => Extend::Sxtx,
                _ => return Err(UnsupportedInstruction),
            };
            Addressing::Register {
                rm,
                extend,
                amount: if word & 0x1000 != 0 {
                    (size as u8).trailing_zeros() as u8
                } else {
                    0
                },
            }
        } else {
            let offset = sign_extend(word >> 12, 9);
            match (word >> 10) & 3 {
                0 | 2 => Addressing::Offset(offset),
                1 => Addressing::PostIndex(offset),
                3 => Addressing::PreIndex(offset),
                _ => return Err(UnsupportedInstruction),
            }
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
        }));
    }
    if word & 0xffff_f01f == 0xd503_201f {
        return Ok(match (word >> 5) & 127 {
            1 => Yield,
            2 => Wfe,
            3 => Wfi,
            4 => Sev,
            5 => Sevl,
            _ => Nop,
        });
    }
    if word & 0xffff_f01f == 0xd503_301f {
        return match (word >> 5) & 7 {
            2 => Ok(Clrex),
            4 => Ok(Dsb),
            5 => Ok(Dmb),
            6 => Ok(Isb),
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
        if crn == 8 {
            return Ok(Tlbi);
        }
        if crn == 7 && matches!(crm, 1 | 5 | 6 | 10 | 11 | 14) {
            return Ok(CacheOp);
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
            | SystemRegister::RazWi => 2,
            _ => 3,
        };
        if (word >> 19) & 3 != expected || (read && expected == 0) {
            return Err(UnsupportedInstruction);
        }
        return Ok(System(self::System {
            read,
            register,
            rt: rd,
            immediate: ((word >> 8) & 15) as u8,
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
            }));
        }
        let load = word & 0x0040_0000 != 0;
        let opc = word >> 30;
        if opc == 1 && !load {
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
    if word & 0x9ff8_0400 == 0x0f00_0400 {
        let q = word & 0x4000_0000 != 0;
        let op = word & 0x2000_0000 != 0;
        let cmode = (word >> 12) & 0xf;
        let imm8 = u64::from((((word >> 16) & 7) << 5) | ((word >> 5) & 0x1f));
        let rd = (word & 0x1f) as u8;
        let twice = |v: u64| v | (v << 32);
        let imm = match cmode {
            // 32-bit lanes, shifted left by 0, 8, 16 or 24.
            0..=7 => twice(imm8 << (8 * (cmode >> 1))),
            // 16-bit lanes, shifted left by 0 or 8.
            8..=11 => {
                let lane = imm8 << (8 * ((cmode >> 1) & 1));
                twice(lane | (lane << 16))
            }
            // 32-bit lanes with the vacated low bits set to ones.
            12 => twice((imm8 << 8) | 0xff),
            13 => twice((imm8 << 16) | 0xffff),
            // Bytes: imm8 in every byte, or each bit of imm8 as a 0x00/0xff byte.
            14 if !op => imm8 * 0x0101_0101_0101_0101,
            14 => (0..8).fold(0, |acc, bit| {
                acc | (if (imm8 >> bit) & 1 != 0 { 0xff } else { 0 }) << (8 * bit)
            }),
            // FMOV: the 8-bit float immediate widened to f32 (twice) or f64.
            _ if !op => twice(super::fp::expand_immediate(imm8, false)),
            _ if q => super::fp::expand_immediate(imm8, true),
            _ => return Err(UnsupportedInstruction),
        };
        // Odd cmode below 12 is ORR/BIC; MOVI becomes MVNI (inverted) with op set.
        let (value, combine) = if cmode < 12 && cmode & 1 == 1 {
            (
                imm,
                if op {
                    ImmCombine::AndNot
                } else {
                    ImmCombine::Or
                },
            )
        } else if cmode < 14 && op {
            (!imm, ImmCombine::Set)
        } else {
            (imm, ImmCombine::Set)
        };
        return Ok(SimdImm(SimdImmediate {
            rd,
            low: value,
            high: if q { value } else { 0 },
            combine,
            q,
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
    // FMOV between a general register and an S/D register: a pure bit move.
    if word & 0x7ebe_fc00 == 0x1e26_0000 {
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
        }));
    }
    // Scalar floating point and integer SIMD the JIT does not compile: `host::run`
    // executes the word.
    if super::host::is_supported(word) {
        return Ok(Host(word));
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
        if !q && size == 3 {
            return Err(UnsupportedInstruction);
        }
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
    if word & 0xbf20_8c00 == 0x0e00_0000 {
        return Ok(SimdTable(self::SimdTable {
            rd: (word & 0x1f) as u8,
            rn: ((word >> 5) & 0x1f) as u8,
            rm: ((word >> 16) & 0x1f) as u8,
            len: ((word >> 13) & 3) as u8,
            extend: word & 0x0000_1000 != 0,
            q: word & 0x4000_0000 != 0,
        }));
    }
    // `mov <Vd>.<T>[index], <R>` and the `umov`/`mov` lane-to-general form.
    if word & 0xbf80_fc00 == 0x0e00_1c00 || word & 0xbf80_fc00 == 0x0e00_3c00 {
        let imm5 = (word >> 16) & 0x1f;
        if imm5 != 0 {
            let esize = imm5.trailing_zeros() as u8;
            let index = (imm5 >> (esize + 1)) as u8;
            let q = word & 0x4000_0000 != 0;
            if esize <= 3 && (q || esize < 3) {
                return if word & 0x0000_2000 == 0 {
                    Ok(SimdInsert(self::SimdInsert {
                        rd,
                        rn,
                        esize,
                        index,
                    }))
                } else {
                    Ok(SimdExtract(self::SimdExtract {
                        rd,
                        rn,
                        esize,
                        index,
                    }))
                };
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
    if word & 0xbf80_fc00 == 0x0e00_0400 {
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
    if word & 0xbf80_fc00 == 0x0e00_0c00 {
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
    if word & 0xbef0_fc00 == 0x0ea0_1c00 {
        let rn = ((word >> 5) & 0x1f) as u8;
        let rm = ((word >> 16) & 0x1f) as u8;
        if rn == rm {
            return Ok(SimdMove { rd, rn });
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
        let rm = ((word >> 16) & 0x1f) as u8;
        let imm = ((word >> 11) & 0xf) as u8;
        let rn = ((word >> 5) & 0x1f) as u8;
        let rd = (word & 0x1f) as u8;
        return Ok(SimdExt(self::SimdExt { rd, rn, rm, imm }));
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
    match word {
        0xd400_0001 => Ok(Svc),
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
                    amount: 0,
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
                    amount: 3,
                },
            ),
            (0xf8410d07, Size::Double, 8, 7, Addressing::PreIndex(16)),
            (0xf8410549, Size::Double, 10, 9, Addressing::PostIndex(16)),
            (0xf85f818b, Size::Double, 12, 11, Addressing::Offset(-8)),
            (0xb85fc1cd, Size::Word, 14, 13, Addressing::Offset(-4)),
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
                    signed: SignExtend::None
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
                addressing: Addressing::Offset(-4),
                signed: SignExtend::None
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
                    signed: SignExtend::None
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
                    immediate: ((word >> 8) & 15) as u8
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
                    immediate: ((word >> 8) & 15) as u8
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
    fn assembler_hints_barriers_and_traps() {
        use Instruction::*;
        for (word, expected) in [
            (0xd5033f9f, Dsb),
            (0xd5033fbf, Dmb),
            (0xd5033fdf, Isb),
            (0xd503201f, Nop),
            (0xd503203f, Yield),
            (0xd503205f, Wfe),
            (0xd503207f, Wfi),
            (0xd503209f, Sev),
            (0xd50320bf, Sevl),
            (0xd503241f, Nop),
            (0xd503245f, Nop),
            (0xd503249f, Nop),
            (0xd50324df, Nop),
            (0xd69f03e0, Eret),
            (0xd4000001, Svc),
            (0xd4000002, Psci),
            (0xd4200020, Brk(1)),
            (0xd4210000, Brk(0x800)),
        ] {
            assert_eq!(decode(word), Ok(expected));
        }
        assert_eq!(decode(0xd50b7423), Ok(DcZva(3)));
        assert_eq!(decode(0xd503309f), Ok(Dsb));
        assert!(!Dsb.terminates());
        assert!(Isb.terminates());
        assert!(Eret.terminates());
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
                Set,
            ),
            (0x2f006489, 0xfbff_ffff_fbff_ffff, 0, Set),
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
                matches!(decode(word), Ok(Instruction::Host(_))),
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
                double: true
            }))
        );
        assert_eq!(
            decode(0x9e670042),
            Ok(Instruction::SimdFmov(SimdFmov {
                rd: 2,
                rn: 2,
                to_fp: true,
                double: true
            }))
        );
        assert_eq!(
            decode(0x1e260022),
            Ok(Instruction::SimdFmov(SimdFmov {
                rd: 2,
                rn: 1,
                to_fp: false,
                double: false
            }))
        );
        assert_eq!(
            decode(0x1e270041),
            Ok(Instruction::SimdFmov(SimdFmov {
                rd: 1,
                rn: 2,
                to_fp: true,
                double: false
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
}
