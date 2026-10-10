//! Textual disassembly of decoded instructions, in GNU binutils (`objdump`) spelling.
//!
//! The text is a presentation of an already decoded [`Instruction`]: nothing here
//! decodes bits. Assembly aliases (`mov`, `cmp`, `ret`, `lsl #n`, ...) are chosen
//! here, from the decoded fields, the way the Arm ARM's alias conditions describe.
//!
//! * [`Instruction::display`] renders without an address: PC-relative operands
//!   print as `.+N` / `.-N`.
//! * [`Instruction::display_at`] renders at `pc`: targets print as absolute
//!   addresses (`0x1008`).
//! * [`Instruction::display_with`] additionally annotates targets with the symbol
//!   that contains them (`0x1008 <callee+0x4>`), through [`Symbols`].
//! * [`word`], [`word_at`] and [`word_with`] start from a raw word and render
//!   `.word 0x...` for one the decoder does not recognise.

use crate::decode::{Instruction, decode};
use core::fmt::{self, Display, Formatter};

mod fp;
mod integer;
mod memory;
mod reg;
mod system;
mod vector;

/// Resolves a code or data address to the symbol that contains it.
pub trait Symbols {
    /// The symbol containing `addr` and the byte offset of `addr` into it.
    fn lookup(&self, addr: u64) -> Option<(&str, u64)>;
}

/// No symbols: targets print bare.
impl Symbols for () {
    fn lookup(&self, _: u64) -> Option<(&str, u64)> {
        None
    }
}

/// Where an instruction is, and what is known about its surroundings.
#[derive(Clone, Copy)]
pub(crate) struct Context<'a> {
    /// The instruction's address, when known. Without it a PC-relative operand
    /// prints as an offset from `.`.
    pub pc: Option<u64>,
    pub symbols: &'a dyn Symbols,
}

impl Context<'_> {
    /// A PC-relative code or data target `offset` bytes from the instruction:
    /// `.+N` / `.-N` without a pc, else `0x<addr>` followed by ` <sym>` or
    /// ` <sym+0xN>` when a symbol contains it. Address arithmetic wraps; a
    /// disassembler must never panic on the bytes it is given.
    pub fn target(&self, f: &mut Formatter<'_>, offset: i64) -> fmt::Result {
        match self.pc {
            None => {
                let sign = if offset < 0 { '-' } else { '+' };
                write!(f, ".{sign}{}", offset.unsigned_abs())
            }
            Some(pc) => self.address(f, pc.wrapping_add(offset as u64)),
        }
    }

    /// Like [`Self::target`] for `adrp`: `offset` is relative to the 4 KiB page
    /// containing the instruction.
    pub fn page_target(&self, f: &mut Formatter<'_>, offset: i64) -> fmt::Result {
        match self.pc {
            None => {
                let sign = if offset < 0 { '-' } else { '+' };
                write!(f, ".{sign}{}", offset.unsigned_abs())
            }
            Some(pc) => self.address(f, (pc & !0xfff).wrapping_add(offset as u64)),
        }
    }

    /// An absolute address with its symbol annotation.
    fn address(&self, f: &mut Formatter<'_>, addr: u64) -> fmt::Result {
        write!(f, "{addr:#x}")?;
        match self.symbols.lookup(addr) {
            Some((name, 0)) => write!(f, " <{name}>"),
            Some((name, offset)) => write!(f, " <{name}+{offset:#x}>"),
            None => Ok(()),
        }
    }
}

/// An [`Instruction`] ready to print; see [`Instruction::display`].
pub struct Disassembly<'a> {
    instruction: &'a Instruction,
    cx: Context<'a>,
}

impl Instruction {
    /// Render without an address.
    pub fn display(&self) -> Disassembly<'_> {
        Disassembly {
            instruction: self,
            cx: Context {
                pc: None,
                symbols: &(),
            },
        }
    }

    /// Render at address `pc`.
    pub fn display_at(&self, pc: u64) -> Disassembly<'_> {
        Disassembly {
            instruction: self,
            cx: Context {
                pc: Some(pc),
                symbols: &(),
            },
        }
    }

    /// Render at address `pc`, annotating targets from `symbols`.
    pub fn display_with<'a>(&'a self, pc: u64, symbols: &'a dyn Symbols) -> Disassembly<'a> {
        Disassembly {
            instruction: self,
            cx: Context {
                pc: Some(pc),
                symbols,
            },
        }
    }
}

impl Display for Disassembly<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write(f, &self.cx, self.instruction)
    }
}

/// A raw instruction word ready to print; see [`word`].
pub struct Word<'a> {
    word: u32,
    cx: Context<'a>,
}

/// Render `word` without an address; `.word 0x...` when it does not decode.
pub fn word(word: u32) -> Word<'static> {
    Word {
        word,
        cx: Context {
            pc: None,
            symbols: &(),
        },
    }
}

/// Render `word` at `pc`; `.word 0x...` when it does not decode.
pub fn word_at(word: u32, pc: u64) -> Word<'static> {
    Word {
        word,
        cx: Context {
            pc: Some(pc),
            symbols: &(),
        },
    }
}

/// Render `word` at `pc`, annotating targets from `symbols`.
pub fn word_with(word: u32, pc: u64, symbols: &dyn Symbols) -> Word<'_> {
    Word {
        word,
        cx: Context {
            pc: Some(pc),
            symbols,
        },
    }
}

impl Display for Word<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match decode(self.word) {
            Ok(instruction) => write(f, &self.cx, &instruction),
            Err(_) => write!(f, ".word {:#010x}", self.word),
        }
    }
}

/// Route `instruction` to the module that owns its family. Every variant is named
/// here, so a new one does not compile until it has a formatter.
fn write(f: &mut Formatter<'_>, cx: &Context<'_>, instruction: &Instruction) -> fmt::Result {
    use Instruction::*;
    match instruction {
        Movz(_) | Movn(_) | Movk(_) | ArithImm(_) | ArithReg(_) | ArithExt(_) | LogicReg(_)
        | LogicImm(_) | Mul(_) | MulLong(_) | MulHigh(_) | Adr(_) | Csel(_) | Variable(_)
        | CondCompare(_) | ArithCarry(_) | Unary(_) | Bitfield(_) | Extract(_) => {
            integer::write(f, cx, instruction)
        }
        Memory(_) | Pair(_) | Literal(_) | SimdLiteral(_) | Exclusive(_) | Atomic(_)
        | Prefetch(_) | PrefetchLiteral(_) | RangePrefetch(_) | SimdPair(_) | SimdMemory(_)
        | SimdStruct(_) => memory::write(f, cx, instruction),
        TestBranch(_) | Call(_) | Indirect(_) | B(_) | BCond(_) | Eret | System(_) | DcZva(_)
        | AddressTranslate(_) | CacheOp(_) | Tlbi(_) | Clrex(_) | Nop | Hint(_) | Dsb(_)
        | Dmb(_) | Isb(_) | Sev | Sevl | Wfe | Wfi | Yield | Brk(_) | Svc(_) | Psci | Pauth(_)
        | Gxf(_) => system::write(f, cx, instruction),
        SimdImm(_)
        | SimdCompareZero(_)
        | SimdDup { .. }
        | SimdDupElement(_)
        | SimdInsert(_)
        | SimdInsertElement(_)
        | SimdExtract(_)
        | SimdExt(_)
        | SimdAlu(_)
        | SimdTable(_)
        | Simd(_) => vector::write(f, cx, instruction),
        Fp(_) | SimdFmov(_) => fp::write(f, cx, instruction),
    }
}
