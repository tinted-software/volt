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
    Svc,
    Psci,
    Wfi,
}
impl Instruction {
    /// Whether this instruction ends a translated block. With `inline_memory`,
    /// plain loads and stores are compiled with an inline fast path and no
    /// longer end the block (their slow path exits mid-block instead).
    pub fn terminates_with(&self, inline_memory: bool) -> bool {
        if inline_memory && matches!(self, Instruction::Memory(_) | Instruction::Literal(_)) {
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
    if word & 0x3e00_0000 == 0x3800_0000 {
        if word & 0x0400_0000 != 0 {
            return Err(UnsupportedInstruction);
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
    if word & 0x3e00_0000 == 0x2800_0000 {
        if word & 0x0400_0000 != 0 {
            return Err(UnsupportedInstruction);
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
            2 => Addressing::Offset(displacement),
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
        assert_eq!(decode(0x6d410861), Err(DecodeError::UnsupportedInstruction));
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
            0x3dc00000,              // SIMD single memory
            0x6d000000,              // SIMD pair memory
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
}
