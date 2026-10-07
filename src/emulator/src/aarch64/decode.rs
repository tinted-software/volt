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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Address {
    pub page: bool,
    pub rd: u8,
    pub offset: i64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimdImmediate {
    pub rd: u8,
    pub low: u64,
    pub high: u64,
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
    /// Consecutive vector registers touched, starting at `rt`. The
    /// multi-structure forms use 2..4; the single-register forms use 1.
    pub registers: u8,
    pub addressing: Addressing,
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PermuteOp {
    Uzp1,
    Uzp2,
    Zip1,
    Zip2,
    Trn1,
    Trn2,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimdPermute {
    pub op: PermuteOp,
    pub rd: u8,
    pub rn: u8,
    pub rm: u8,
    pub size: u8,
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
    TpidrEl1,
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
    Nzcv,
    RazWi,
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
    Exclusive(Exclusive),
    Clrex,
    Adr(Address),
    System(System),
    DcZva(u8),
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
    Trap,
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
    SimdCompareZero(SimdCompareZero),
    SimdFmov(SimdFmov),
    SimdMove { rd: u8, rn: u8 },
    SimdDup { rd: u8, rn: u8, esize: u8, q: bool },
    SimdDupElement(SimdDupElement),
    SimdInsert(SimdInsert),
    SimdExtract(SimdExtract),
    SimdExt(SimdExt),
    SimdPermute(SimdPermute),
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
                | Literal(_)
                | Pair(_)
                | SimdPair(_)
                | SimdMemory(_)
                | B(_)
                | BCond(_)
                | TestBranch(_)
                | Call(_)
                | Indirect(_)
                | Svc
                | Psci
                | Wfi
                | Wfe
                | Trap
                | Isb
                | DcZva(_)
                | Eret
                | Tlbi
        )
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
        0x0d04 => TpidrEl1,
        0x0022 => MdscrEl1,
        0x0104 => OslarEl1,
        0x0134 => OsdlrEl1,
        0x0114 => OslsrEl1,
        0x0e10 => CntkctlEl1,
        0x3440 => Fpcr,
        0x3441 => Fpsr,
        0x3421 => Daif,
        0x0000 => MidrEl1,
        0x0005 => MpidrEl1,
        0x0006 => RevidrEl1,
        0x3001 => CtrEl0,
        0x3007 => DczidEl0,
        0x3e00 => CntfrqEl0,
        0x1001 => ClidrEl1,
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
    if word & 0x3f00_0000 == 0x0800_0000 {
        if word & 0x0020_0000 != 0 {
            return Err(UnsupportedInstruction);
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
    // LD1/ST1 (multiple structures): the opcode field at bits 15:12 selects
    // how many consecutive vector registers the structure spans, starting at
    // `rt`: `1010` -> 2, `0110` -> 3, `0010` -> 4. The `0111` opcode is the
    // single-register form handled below; the remaining encodings are the
    // interleaved LD2/LD3/LD4 single-structure forms, which spread one
    // element across registers and are not supported here.
    if word & 0x3e00_0000 == 0x0c00_0000 && word & 0x0000_f000 != 0x0000_7000 {
        let registers = match (word >> 12) & 0xf {
            0b1010 => 2,
            0b0110 => 3,
            0b0010 => 4,
            _ => return Err(UnsupportedInstruction),
        };
        // Each register holds the full vector width selected by Q.
        let bytes = if word & 0x4000_0000 != 0 { 16 } else { 8 };
        let post_index = word & 0x0080_0000 != 0;
        let index = (word >> 16) & 0x3f;
        let span = (registers as u32 * bytes) as u64;
        let addressing = if post_index {
            if index != 31 {
                return Err(UnsupportedInstruction);
            }
            Addressing::PostIndex(span as i64)
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
            registers: registers as u8,
            addressing,
        }));
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
            registers: 1,
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
                registers: 1,
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
            SystemRegister::DaifSet | SystemRegister::DaifClear => 0,
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
    if word & 0xffe0_f000 == 0xd420_0000 {
        return if word & 0x0000_ffe0 == 0x20 {
            Ok(Trap)
        } else {
            Err(UnsupportedInstruction)
        };
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
    if word & 0x9f00_0400 == 0x0f00_0400 {
        let q = (word >> 30) & 1;
        let op = (word >> 29) & 1;
        let cmode = (word >> 12) & 0xf;
        let abc = (word >> 16) & 7;
        let defgh = (word >> 5) & 0x1f;
        let imm8 = (abc << 5) | defgh;
        let rd = (word & 0x1f) as u8;

        let val32: u64 = match cmode {
            0 | 1 => imm8 as u64,
            2 | 3 => (imm8 as u64) << 8,
            4 | 5 => (imm8 as u64) << 16,
            6 | 7 => (imm8 as u64) << 24,
            8 | 9 => {
                let v16 = imm8 as u64;
                v16 | (v16 << 16)
            }
            10 | 11 => {
                let v16 = (imm8 as u64) << 8;
                v16 | (v16 << 16)
            }
            14 => {
                let b = imm8 as u64;
                b | (b << 8) | (b << 16) | (b << 24)
            }
            _ => imm8 as u64,
        };

        let mut val64 = val32 | (val32 << 32);
        if op == 1 {
            val64 = !val64;
        }

        let low = val64;
        let high = if q == 1 { val64 } else { 0 };
        return Ok(SimdImm(SimdImmediate { rd, low, high }));
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
    // Integer three-registers-same (`U`, `size`, `opcode` verified with llvm-mc).
    if word & 0x9f20_0000 == 0x0e20_0000 {
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
            (false, 0b10111) if rm == 0b10001 => AddV,
            (false, 0b10111) if rm != 0b10001 => AddP,
            (true, 0b10100) => UMaxP,
            (false, 0b10100) => SMaxP,
            (true, 0b10101) => UMinP,
            (false, 0b10101) => SMinP,
            (false, 0b10111) if size != 3 => AddP,
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
    if word & 0xbf20_8400 == 0x0e00_0000 {
        let op = match (word >> 10) & 0x3f {
            0b000110 => PermuteOp::Uzp1,
            0b001110 => PermuteOp::Uzp2,
            0b000101 => PermuteOp::Zip1,
            0b001101 => PermuteOp::Zip2,
            0b000111 => PermuteOp::Trn1,
            0b001111 => PermuteOp::Trn2,
            _ => return Err(UnsupportedInstruction),
        };
        let size = ((word >> 22) & 3) as u8;
        let rm = ((word >> 16) & 0x1f) as u8;
        let rn = ((word >> 5) & 0x1f) as u8;
        let rd = (word & 0x1f) as u8;
        return Ok(SimdPermute(self::SimdPermute {
            op,
            rd,
            rn,
            rm,
            size,
        }));
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
        for word in [0xd5380060, 0xd5383000, 0xd53b0000, 0xd5385000, 0xd5180000] {
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
            (0xd4200020, Trap),
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
            0x887f7c00,              // pair-exclusive class
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
                registers: 1,
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
                registers: 1,
                addressing: Addressing::PostIndex(16),
            }))
        );
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
