//! Vulcan text-format parser (Vulcan functional text): the round-trip partner
//! of the function printer. Values and blocks are named positionally (`v{n}`,
//! `block{n}`); the parser creates them in textual order and resolves
//! references by number. Ported from `vulcan/libs/vulcan-ir/parser.zig`.
//!
//! # Memory accesses
//!
//! `load` and `store` take optional modifier words (the `MemFlags`) between the mnemonic
//! and the operands. Each may appear at most once, in any order; the canonical order
//! (the one a printer writes) is:
//!
//! ```text
//! let v1 = load [volatile] [atomic <ordering>] [align <n>] [be|le] <ty>, <ptr>
//! store [volatile] [atomic <ordering>] [align <n>] [be|le] <value>, <ptr>
//! ```
//!
//! - `volatile`: the access may not be removed, merged or reordered.
//! - `atomic <ordering>`: `relaxed`, `acquire`, `release`, `seq_cst` (loads take
//!   relaxed/acquire/seq_cst, stores relaxed/release/seq_cst; `acq_rel` parses but does not
//!   verify on a load or store).
//! - `align <n>`: the address is aligned to `n` bytes (a power of two).
//! - `be` / `le`: big / little endian memory image; omitted means native.
//!
//! # Signedness lives on the operation
//!
//! Integer types are plain `iN` (there is no `uN`); an operation decides whether it reads its
//! operands as signed or unsigned. Binary operators are infix (`let v2 = v0 <op> v1`, or
//! `v0 <op> <integer literal>` for the immediate form); the signed / unsigned variant of an
//! operator is the same spelling with an `s` / `u` suffix:
//!
//! ```text
//! + - * & | ^ << >>     add sub mul and or xor shl, logical shift right (integer)
//! / %                   FLOAT division / remainder only
//! /u /s  %u %s          unsigned / signed integer division / remainder
//! >>s                   arithmetic shift right (integer)
//! *hu *hs               high half of the unsigned / signed double-width product
//! == !=                 equality of any scalar
//! < <= > >=             FLOAT ordered comparison only
//! <s <=s >s >=s         signed integer comparison
//! <u <=u >u >=u         unsigned integer / pointer comparison
//! ```
//!
//! Conversions name the kind first, then the result type:
//!
//! ```text
//! let v1 = convert <kind> <ty>, v0
//! ```
//!
//! with `kind` one of `trunc zext sext sitofp uitofp fptosi fptoui fpresize` (the legal type
//! pairs of each are listed on `ConvertKind`). A same-width integer cast does not exist.
//!
//! `dot` and the atomic min/max spell the signedness as a word:
//!
//! ```text
//! let v3 = dot signed|unsigned v0, v1, v2
//! let v2 = atomic_rmw add|smin|smax|umin|umax|bit_and|bit_or|bit_xor|exchange <scope> <order> v0, v1
//! ```
//!
//! Words naming a `BinOp` (`reduce add, v0`) are `add sub mul div rem udiv sdiv urem srem
//! bit_and bit_or bit_xor shl shr sar umulh smulh`.
//!
//! # Intrinsics
//!
//! An intrinsic is a target-defined operation named by a dotted identifier:
//!
//! ```text
//! let v2 = intrinsic <ty> @aarch64.aese(v0, v1) [imm <n>] [pure]
//! intrinsic @aarch64.dmb() [imm <n>] [pure]
//! ```
//!
//! `imm` is the one signed immediate operand (omitted means 0, and the printer omits it
//! then). `pure` marks the intrinsic free of side effects (`Intrinsic::side_effects ==
//! false`); without it the intrinsic is assumed to have side effects.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use super::attribute::{AttrValue, Attribute, Custom, Endianness};
use super::function::{
    Alloca, Arith, AtomicOp, AtomicOrdering, AtomicRmw, AtomicScope, AttrTarget, BarrierScope,
    BinOp, Block, Call, CallIndirect, CmpOp, Compare, ConvertKind, EdgeDesc, Extract, Function,
    GlobalAddr, InputSigns, LowFloatConvert, MatMul, MatMulQuant, MatMulQuantOut, MatMulScale,
    MatMulType, MemFlags, NvFp4Convert, Opcode, Reduce, Ret, RetPiece, Select, Terminator, Unary,
    UnaryOp, Value,
};
use super::low_float::Format as LowFloatFormat;
use super::nvfp4::ScaleApplication;
use super::types::{self, FloatKind, IntDesc, Type, TypeKind};

/// Parse errors: bad function syntax out in the world, or an invalid type text.
pub type Error = ParseError;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ParseError {
    InvalidSyntax,
    Type(types::ParseError),
}

impl core::fmt::Display for ParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ParseError::InvalidSyntax => write!(f, "invalid syntax"),
            ParseError::Type(_) => write!(f, "invalid type"),
        }
    }
}

impl From<types::ParseError> for ParseError {
    fn from(e: types::ParseError) -> Self {
        ParseError::Type(e)
    }
}

/// The infix spellings of the binary operators. A spelling that is a prefix of another
/// (`>>` of `>>s`, `*` of `*hu`) comes after it, so the first match wins.
const ARITH_OPS: [(&str, BinOp); 17] = [
    ("<<", BinOp::Shl),
    (">>s", BinOp::Sar),
    (">>", BinOp::Shr),
    ("*hu", BinOp::UMulh),
    ("*hs", BinOp::SMulh),
    ("/u", BinOp::UDiv),
    ("/s", BinOp::SDiv),
    ("%u", BinOp::URem),
    ("%s", BinOp::SRem),
    ("+", BinOp::Add),
    ("-", BinOp::Sub),
    ("*", BinOp::Mul),
    ("/", BinOp::Div),
    ("%", BinOp::Rem),
    ("&", BinOp::BitAnd),
    ("|", BinOp::BitOr),
    ("^", BinOp::BitXor),
];

/// The infix spellings of the comparisons, ordered like [`ARITH_OPS`].
const CMP_OPS: [(&str, CmpOp); 14] = [
    ("==", CmpOp::Eq),
    ("!=", CmpOp::Ne),
    ("<=s", CmpOp::Sle),
    ("<=u", CmpOp::Ule),
    ("<=", CmpOp::Le),
    ("<s", CmpOp::Slt),
    ("<u", CmpOp::Ult),
    ("<", CmpOp::Lt),
    (">=s", CmpOp::Sge),
    (">=u", CmpOp::Uge),
    (">=", CmpOp::Ge),
    (">s", CmpOp::Sgt),
    (">u", CmpOp::Ugt),
    (">", CmpOp::Gt),
];

/// Parse a function from its text form. The returned function owns its memory.
pub fn parse(text: &str) -> Result<Function, ParseError> {
    let func = Function::new();
    let mut p = FunctionParser {
        func,
        src: text.as_bytes(),
        pos: 0,
        value_names: Vec::new(),
    };
    p.parse_function()?;
    Ok(p.func)
}

fn is_digit(c: u8) -> bool {
    c.is_ascii_digit()
}

/// An underscore is part of a word. The printer writes mnemonics such as `call_indirect`
/// and `global_addr`, and attribute keys such as `local_size_x`, so a word that stops at
/// the underscore cannot read back what the printer wrote.
fn is_word_char(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

/// Decimal text to binary128 bit pattern. f128 is unstable on this toolchain
/// (`#![feature(f128)]`), so only exact integers are accepted: the accumulator
/// is exact for magnitudes below 2^113, which covers every constant the
/// current frontends emit. Anything else is InvalidSyntax, never a wrong
/// value.
fn decimal_str_to_f128_bits(text: &str) -> Option<u128> {
    let bytes = text.as_bytes();
    let mut i = 0;
    let mut neg = false;
    if bytes.first() == Some(&b'-') {
        neg = true;
        i = 1;
    } else if bytes.first() == Some(&b'+') {
        i = 1;
    }
    if i >= bytes.len() {
        return None;
    }
    // Reject anything that is not a plain integer: fraction, exponent, NaN/inf.
    for &c in &bytes[i..] {
        if !c.is_ascii_digit() {
            return None;
        }
    }
    let mut mant: u128 = 0;
    for &c in &bytes[i..] {
        mant = mant.checked_mul(10)?.checked_add((c - b'0') as u128)?;
        if mant >= (1 << 113) {
            return None;
        }
    }
    // Normalize: shift mantissa to the 112-bit fraction field.
    let bits: u128 = if mant == 0 {
        0
    } else {
        let top: u32 = 127 - mant.leading_zeros();
        let exp: u128 = (top as u128 + 16383) << 112;
        let frac: u128 = if top >= 112 {
            mant >> (top - 112)
        } else {
            mant << (112 - top)
        };
        exp | (frac & ((1 << 112) - 1))
    };
    Some(if neg { bits | (1 << 127) } else { bits })
}

fn all_digits(s: &[u8]) -> bool {
    !s.is_empty() && s.iter().all(|&c| is_digit(c))
}

/// BinOp from its word (`add`, `sub`, `bit_and`, `udiv`, ...). The text is untrusted,
/// so an unknown op word is a parse error, never an invalid enum.
fn bin_op_of(word: &str) -> Option<BinOp> {
    match word {
        "add" => Some(BinOp::Add),
        "sub" => Some(BinOp::Sub),
        "mul" => Some(BinOp::Mul),
        "div" => Some(BinOp::Div),
        "rem" => Some(BinOp::Rem),
        "udiv" => Some(BinOp::UDiv),
        "sdiv" => Some(BinOp::SDiv),
        "urem" => Some(BinOp::URem),
        "srem" => Some(BinOp::SRem),
        "bit_and" => Some(BinOp::BitAnd),
        "bit_or" => Some(BinOp::BitOr),
        "bit_xor" => Some(BinOp::BitXor),
        "shl" => Some(BinOp::Shl),
        "shr" => Some(BinOp::Shr),
        "sar" => Some(BinOp::Sar),
        "umulh" => Some(BinOp::UMulh),
        "smulh" => Some(BinOp::SMulh),
        _ => None,
    }
}

fn convert_kind_of(word: &str) -> Option<ConvertKind> {
    match word {
        "trunc" => Some(ConvertKind::Trunc),
        "zext" => Some(ConvertKind::Zext),
        "sext" => Some(ConvertKind::Sext),
        "sitofp" => Some(ConvertKind::SiToFp),
        "uitofp" => Some(ConvertKind::UiToFp),
        "fptosi" => Some(ConvertKind::FpToSi),
        "fptoui" => Some(ConvertKind::FpToUi),
        "fpresize" => Some(ConvertKind::FpResize),
        _ => None,
    }
}

fn unary_op_of(word: &str) -> Option<UnaryOp> {
    match word {
        "reinterpret" => Some(UnaryOp::Reinterpret),
        "sqrt" => Some(UnaryOp::Sqrt),
        "ceil" => Some(UnaryOp::Ceil),
        "floor" => Some(UnaryOp::Floor),
        "trunc" => Some(UnaryOp::Trunc),
        "nearest" => Some(UnaryOp::Nearest),
        _ => None,
    }
}

fn low_float_format_of(word: &str) -> Option<LowFloatFormat> {
    match word {
        "bf16" => Some(LowFloatFormat::Bf16),
        "f8_e4m3" => Some(LowFloatFormat::F8E4M3),
        "f8_e5m2" => Some(LowFloatFormat::F8E5M2),
        _ => None,
    }
}

fn scale_application_of(word: &str) -> Option<ScaleApplication> {
    match word {
        "multiply" => Some(ScaleApplication::Multiply),
        "divide" => Some(ScaleApplication::Divide),
        _ => None,
    }
}

fn matmul_dtype_of(word: &str) -> Option<MatMulType> {
    match word {
        "fp32" => Some(MatMulType::Fp32),
        "fp16" => Some(MatMulType::Fp16),
        "int8" => Some(MatMulType::Int8),
        "uint8" => Some(MatMulType::Uint8),
        _ => None,
    }
}

fn matmul_quant_out_of(word: &str) -> Option<MatMulQuantOut> {
    match word {
        "i8" => Some(MatMulQuantOut::I8),
        "u8" => Some(MatMulQuantOut::U8),
        _ => None,
    }
}

fn barrier_scope_of(word: &str) -> Option<BarrierScope> {
    match word {
        "workgroup" => Some(BarrierScope::Workgroup),
        "subgroup" => Some(BarrierScope::Subgroup),
        _ => None,
    }
}

fn atomic_op_of(word: &str) -> Option<AtomicOp> {
    match word {
        "add" => Some(AtomicOp::Add),
        "smin" => Some(AtomicOp::SMin),
        "smax" => Some(AtomicOp::SMax),
        "umin" => Some(AtomicOp::UMin),
        "umax" => Some(AtomicOp::UMax),
        "bit_and" => Some(AtomicOp::BitAnd),
        "bit_or" => Some(AtomicOp::BitOr),
        "bit_xor" => Some(AtomicOp::BitXor),
        "exchange" => Some(AtomicOp::Exchange),
        "compare_exchange" => Some(AtomicOp::CompareExchange),
        _ => None,
    }
}

fn atomic_ordering_of(word: &str) -> Option<AtomicOrdering> {
    match word {
        "relaxed" => Some(AtomicOrdering::Relaxed),
        "acquire" => Some(AtomicOrdering::Acquire),
        "release" => Some(AtomicOrdering::Release),
        "acq_rel" => Some(AtomicOrdering::AcqRel),
        "seq_cst" => Some(AtomicOrdering::SeqCst),
        _ => None,
    }
}

fn atomic_scope_of(word: &str) -> Option<AtomicScope> {
    match word {
        "workgroup" => Some(AtomicScope::Workgroup),
        "device" => Some(AtomicScope::Device),
        "system" => Some(AtomicScope::System),
        _ => None,
    }
}

#[derive(Clone, Debug, Default)]
struct CallExtras {
    is_variadic: bool,
    num_fixed: u32,
    ret_dest: Option<Value>,
    ret_regs: u8,
    ret_pieces: [RetPiece; 4],
    sret: bool,
}

struct FunctionParser<'a> {
    func: Function,
    src: &'a [u8],
    pos: usize,
    /// Values indexed by their positional name number.
    value_names: Vec<Value>,
}

impl<'a> FunctionParser<'a> {
    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while matches!(self.src.get(self.pos), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn eat(&mut self, c: u8) -> Result<(), ParseError> {
        if self.peek() == Some(c) {
            self.pos += 1;
            Ok(())
        } else {
            Err(ParseError::InvalidSyntax)
        }
    }

    fn try_char(&mut self, c: u8) -> bool {
        if self.peek() == Some(c) {
            self.pos += 1;
            return true;
        }
        false
    }

    fn read_word(&mut self) -> &'a str {
        let start = self.pos;
        while matches!(self.src.get(self.pos), Some(&c) if is_word_char(c)) {
            self.pos += 1;
        }
        // Word characters are ASCII alphanumerics by construction.
        core::str::from_utf8(&self.src[start..self.pos]).unwrap_or("")
    }

    fn expect_word(&mut self, word: &str) -> Result<(), ParseError> {
        if self.read_word() == word {
            Ok(())
        } else {
            Err(ParseError::InvalidSyntax)
        }
    }

    fn read_unsigned(&mut self) -> Result<u64, ParseError> {
        let start = self.pos;
        while matches!(self.src.get(self.pos), Some(&c) if is_digit(c)) {
            self.pos += 1;
        }
        if self.pos == start {
            return Err(ParseError::InvalidSyntax);
        }
        let s = core::str::from_utf8(&self.src[start..self.pos]).unwrap_or("");
        s.parse::<u64>().map_err(|_| ParseError::InvalidSyntax)
    }

    fn read_signed(&mut self) -> Result<i64, ParseError> {
        let neg = self.try_char(b'-');
        let mag = self.read_unsigned()?;
        if neg {
            // -(2^63) is a valid i64 even though +2^63 is not, so handle the
            // boundary explicitly; any larger magnitude is out of range.
            if mag == (i64::MAX as u64) + 1 {
                return Ok(i64::MIN);
            }
            if mag > i64::MAX as u64 {
                return Err(ParseError::InvalidSyntax);
            }
            return Ok(-(mag as i64));
        }
        if mag > i64::MAX as u64 {
            return Err(ParseError::InvalidSyntax);
        }
        Ok(mag as i64)
    }

    fn read_bool(&mut self) -> Result<bool, ParseError> {
        self.skip_ws();
        match self.read_word() {
            "true" => Ok(true),
            "false" => Ok(false),
            _ => Err(ParseError::InvalidSyntax),
        }
    }

    /// Read an unsigned literal that may carry a `0x` prefix, as the printer writes the
    /// quant scale bits.
    fn read_radix_unsigned(&mut self) -> Result<u64, ParseError> {
        self.skip_ws();
        let start = self.pos;
        let w = self.read_word();
        if w.is_empty() {
            return Err(ParseError::InvalidSyntax);
        }
        u64::from_str_radix(
            core::str::from_utf8(&self.src[start..self.pos]).unwrap_or(""),
            0,
        )
        .map_err(|_| ParseError::InvalidSyntax)
    }

    /// Parse a type embedded at the cursor, advancing past it.
    fn parse_type(&mut self) -> Result<Type, ParseError> {
        let start = self.pos;
        // Scan a balanced type token: types nest <> [] {} and carry `x`, digits,
        // commas, spaces, and address-space parens.
        let mut depth = 0usize;
        let mut end = self.pos;
        while let Some(&c) = self.src.get(end) {
            match c {
                b'<' | b'[' | b'{' | b'(' => {
                    depth += 1;
                    end += 1;
                }
                b'>' | b']' | b'}' | b')' => {
                    if depth == 0 {
                        break;
                    }
                    depth -= 1;
                    end += 1;
                    if depth == 0 {
                        break;
                    }
                }
                b',' => {
                    if depth == 0 {
                        break;
                    }
                    end += 1;
                }
                b' ' | b'\t' if depth > 0 => {
                    end += 1;
                }
                c if c.is_ascii_alphanumeric() || c == b'_' => {
                    end += 1;
                }
                _ => break,
            }
        }
        let text =
            core::str::from_utf8(&self.src[start..end]).map_err(|_| ParseError::InvalidSyntax)?;
        let ty = self.func.types.parse_type(text)?;
        self.pos = end;
        Ok(ty)
    }

    /// Parse a value definition `vN`, asserting N is the next positional name.
    fn define_value_name(&mut self) -> Result<(), ParseError> {
        self.eat(b'v')?;
        let num = self.read_unsigned()?;
        if num as usize != self.value_names.len() {
            return Err(ParseError::InvalidSyntax);
        }
        Ok(())
    }

    fn record_value(&mut self, value: Value) {
        self.value_names.push(value);
    }

    /// Parse a value reference `vN`, resolving it.
    fn parse_value_ref(&mut self) -> Result<Value, ParseError> {
        self.eat(b'v')?;
        let num = self.read_unsigned()?;
        if num as usize >= self.value_names.len() {
            return Err(ParseError::InvalidSyntax);
        }
        Ok(self.value_names[num as usize])
    }

    /// Convert a parsed block number to a `Block`, rejecting an out-of-range value
    /// that would later index the block list out of bounds (blocks are precreated
    /// up front, so the count is final here).
    fn checked_block(&self, bnum: u32) -> Result<Block, ParseError> {
        if bnum as usize >= self.func.block_count() {
            return Err(ParseError::InvalidSyntax);
        }
        Ok(Block(bnum))
    }

    fn parse_function(&mut self) -> Result<(), ParseError> {
        self.precreate_blocks()?;
        self.skip_ws();
        self.parse_attrs(AttrTarget::Func)?;
        self.skip_ws();
        self.expect_word("fn")?;
        self.parse_function_modifiers()?;
        self.skip_ws();
        self.eat(b'{')?;

        loop {
            self.skip_ws();
            if self.peek() == Some(b'}') {
                self.pos += 1;
                break;
            }
            // Attributes here sit above a block label, so they belong to that block. The
            // label is read first, so they attach to the block the label names.
            let mut pending: Vec<Attribute> = Vec::new();
            self.collect_attrs(&mut pending, &mut None)?;
            self.parse_block(&pending)?;
        }
        Ok(())
    }

    /// Read the whole-function metadata words the printer writes between `fn` and `{`:
    /// `local`, `variadic(n)` and `sret`, in any order and any number. An ordinary
    /// function has none of them and this stops at once.
    fn parse_function_modifiers(&mut self) -> Result<(), ParseError> {
        loop {
            let save = self.pos;
            self.skip_ws();
            if self.peek() == Some(b'{') {
                return Ok(());
            }
            let word = self.read_word();
            if word == "local" {
                self.func.is_local = true;
            } else if word == "sret" {
                self.func.sret = true;
            } else if word == "variadic" {
                self.eat(b'(')?;
                let n = self.read_unsigned()?;
                self.func.num_fixed_params =
                    u32::try_from(n).map_err(|_| ParseError::InvalidSyntax)?;
                self.eat(b')')?;
                self.func.is_variadic = true;
            } else {
                self.pos = save;
                return Ok(());
            }
        }
    }

    /// Try to read `word` at the cursor. On a different word the cursor does not move, so
    /// the caller can fall through to another spelling.
    fn try_word(&mut self, word: &str) -> bool {
        let save = self.pos;
        self.skip_ws();
        if self.read_word() == word {
            return true;
        }
        self.pos = save;
        false
    }

    /// Parse zero or more `#[...]` attributes, attaching each to `target`.
    fn parse_attrs(&mut self, target: AttrTarget) -> Result<(), ParseError> {
        loop {
            self.skip_ws();
            if self.peek() != Some(b'#') {
                break;
            }
            self.eat(b'#')?;
            self.eat(b'[')?;
            self.skip_ws();
            let attr = self.parse_attr_body()?;
            self.skip_ws();
            self.eat(b']')?;
            self.func.add_attr(target, attr);
        }
        Ok(())
    }

    fn parse_attr_body(&mut self) -> Result<Attribute, ParseError> {
        let word_start = self.pos;
        let word = self.read_word().to_string();
        if word == "inline" {
            return Ok(Attribute::Inline);
        }
        if word == "noreturn" {
            return Ok(Attribute::Noreturn);
        }
        if word == "cold" {
            return Ok(Attribute::Cold);
        }
        if word == "align" {
            self.eat(b'(')?;
            let n = self.read_unsigned()?;
            self.eat(b')')?;
            let align = u32::try_from(n).map_err(|_| ParseError::InvalidSyntax)?;
            return Ok(Attribute::Align(align));
        }
        // Namespaced: `namespace.key` with an optional `= value`. A namespace may itself
        // hold dots (`vulcan.gpu` is the one the whole GPU path uses), so read the full
        // dotted path and split it at the LAST dot: everything before it is the
        // namespace, the final segment is the key.
        while self.peek() == Some(b'.') {
            self.pos += 1;
            if self.read_word().is_empty() {
                return Err(ParseError::InvalidSyntax);
            }
        }
        let path = &self.src[word_start..self.pos];
        let dot = path
            .iter()
            .rposition(|&c| c == b'.')
            .ok_or(ParseError::InvalidSyntax)?;
        let namespace = String::from_utf8_lossy(&path[..dot]).into_owned();
        let key = String::from_utf8_lossy(&path[dot + 1..]).into_owned();
        self.skip_ws();
        let mut value = AttrValue::Flag;
        if self.try_char(b'=') {
            self.skip_ws();
            value = self.parse_attr_value()?;
        }
        Ok(Attribute::Custom(Custom {
            namespace,
            key,
            value,
        }))
    }

    fn parse_attr_value(&mut self) -> Result<AttrValue, ParseError> {
        if self.peek() == Some(b'"') {
            self.pos += 1;
            let start = self.pos;
            while matches!(self.src.get(self.pos), Some(&c) if c != b'"') {
                self.pos += 1;
            }
            let s = String::from_utf8_lossy(&self.src[start..self.pos]).into_owned();
            self.eat(b'"')?;
            return Ok(AttrValue::Str(s));
        }
        Ok(AttrValue::Int(self.read_signed()?))
    }

    /// Pre-create one block per label line so forward references resolve.
    fn precreate_blocks(&mut self) -> Result<(), ParseError> {
        let mut count = 0usize;
        for line in self.src.split(|&c| c == b'\n') {
            let mut trimmed = line;
            while matches!(trimmed.first(), Some(b' ' | b'\t' | b'\r')) {
                trimmed = &trimmed[1..];
            }
            while matches!(trimmed.last(), Some(b' ' | b'\t' | b'\r')) {
                trimmed = &trimmed[..trimmed.len() - 1];
            }
            if trimmed.starts_with(b"block") && trimmed.ends_with(b":") {
                count += 1;
            }
        }
        for _ in 0..count {
            self.func.append_block();
        }
        Ok(())
    }

    fn parse_block(&mut self, block_attrs: &[Attribute]) -> Result<(), ParseError> {
        self.skip_ws();
        let label_start = self.pos;
        let label = self.read_word().to_string();
        if !label.starts_with("block") {
            return Err(ParseError::InvalidSyntax);
        }
        let bnum = self.label_to_block_num(label_start, "block")?;
        let block = self.checked_block(bnum)?;
        for attr in block_attrs {
            self.func.add_attr(AttrTarget::Block(block), attr.clone());
        }

        self.eat(b'(')?;
        self.skip_ws();
        if self.peek() != Some(b')') {
            loop {
                self.skip_ws();
                self.parse_block_param(block)?;
                self.skip_ws();
                if self.try_char(b',') {
                    continue;
                }
                break;
            }
        }
        self.skip_ws();
        self.eat(b')')?;
        self.skip_ws();
        self.eat(b':')?;

        self.parse_body(block)
    }

    fn parse_block_param(&mut self, block: Block) -> Result<(), ParseError> {
        self.define_value_name()?;
        self.skip_ws();
        self.eat(b':')?;
        self.skip_ws();
        let ty = self.parse_type()?;
        let value = self.func.append_block_param(block, ty);
        self.record_value(value);
        // Attributes that follow the parameter attach to it, not to the block.
        self.parse_attrs(AttrTarget::Value(value))
    }

    /// Parse instructions until a terminator ends the block.
    fn parse_body(&mut self, block: Block) -> Result<(), ParseError> {
        loop {
            self.skip_ws();
            let mut pending: Vec<Attribute> = Vec::new();
            let mut pending_inst: Vec<Attribute> = Vec::new();
            self.collect_attrs(&mut pending, &mut Some(&mut pending_inst))?;
            let insts_before = self.func.block_insts(block).len();
            self.skip_ws();
            let word = self.read_word().to_string();
            if word == "const" {
                let v = self.parse_const(block)?;
                self.attach_all(&pending, v)?;
            } else if word == "let" {
                let v = self.parse_let(block)?;
                self.attach_all(&pending, v)?;
            } else if word.len() >= 2
                && word.as_bytes()[0] == b'v'
                && all_digits(&word.as_bytes()[1..])
            {
                let v = self.parse_select(block, &word)?;
                self.attach_all(&pending, v)?;
            } else if word == "if" {
                if !pending.is_empty() {
                    return Err(ParseError::InvalidSyntax);
                }
                self.parse_if(block)?;
            } else if word == "store" {
                if !pending.is_empty() {
                    return Err(ParseError::InvalidSyntax);
                }
                self.parse_store(block)?;
            } else if word == "barrier" {
                if !pending.is_empty() {
                    return Err(ParseError::InvalidSyntax);
                }
                self.parse_barrier(block)?;
            } else if word == "atomic_rmw" {
                if !pending.is_empty() {
                    return Err(ParseError::InvalidSyntax);
                }
                let rmw = self.parse_atomic_rmw()?;
                self.func.append_atomic_rmw_stmt(block, rmw);
            } else if word == "call" {
                if !pending.is_empty() {
                    return Err(ParseError::InvalidSyntax);
                }
                self.parse_void_call(block)?;
            } else if word == "intrinsic" {
                if !pending.is_empty() {
                    return Err(ParseError::InvalidSyntax);
                }
                let (name, args, imm, side_effects) = self.parse_intrinsic_body()?;
                self.func
                    .append_intrinsic_void(block, &name, &args, imm, side_effects);
            } else if word == "call_indirect" {
                if !pending.is_empty() {
                    return Err(ParseError::InvalidSyntax);
                }
                self.parse_void_call_indirect(block)?;
            } else if word == "prefetch" {
                if !pending.is_empty() {
                    return Err(ParseError::InvalidSyntax);
                }
                self.skip_ws();
                let ptr = self.parse_value_ref()?;
                self.func.append_prefetch(block, ptr);
            } else if word == "va_start" {
                if !pending.is_empty() {
                    return Err(ParseError::InvalidSyntax);
                }
                self.skip_ws();
                let ptr = self.parse_value_ref()?;
                self.func.append_va_start(block, ptr);
            } else if word == "va_end" {
                if !pending.is_empty() {
                    return Err(ParseError::InvalidSyntax);
                }
                self.skip_ws();
                let ptr = self.parse_value_ref()?;
                self.func.append_va_end(block, ptr);
            } else if word == "matmul" {
                if !pending.is_empty() {
                    return Err(ParseError::InvalidSyntax);
                }
                self.parse_matmul(block)?;
            } else if word == "ret" {
                if !pending.is_empty() || !pending_inst.is_empty() {
                    return Err(ParseError::InvalidSyntax);
                }
                self.parse_ret(block)?;
                return Ok(());
            } else if word.starts_with("block") {
                if !pending.is_empty() || !pending_inst.is_empty() {
                    return Err(ParseError::InvalidSyntax);
                }
                self.parse_jump(block, &word)?;
                return Ok(());
            } else {
                return Err(ParseError::InvalidSyntax);
            }
            if !pending_inst.is_empty() {
                let insts = self.func.block_insts(block);
                if insts.len() == insts_before {
                    return Err(ParseError::InvalidSyntax);
                }
                let inst = insts[insts.len() - 1];
                for attr in &pending_inst {
                    self.func.add_attr(AttrTarget::Inst(inst), attr.clone());
                }
            }
        }
    }

    /// Collect leading attributes without attaching them yet.
    fn collect_attrs(
        &mut self,
        list: &mut Vec<Attribute>,
        inst_list: &mut Option<&mut Vec<Attribute>>,
    ) -> Result<(), ParseError> {
        loop {
            self.skip_ws();
            if self.peek() != Some(b'#') {
                break;
            }
            self.eat(b'#')?;
            let on_inst = self.try_char(b'!');
            self.eat(b'[')?;
            self.skip_ws();
            let attr = self.parse_attr_body()?;
            if on_inst {
                if inst_list.is_none() {
                    return Err(ParseError::InvalidSyntax);
                }
                // Re-borrow each iteration: `&mut Option<&mut Vec>` has no Copy.
                match inst_list.as_deref_mut() {
                    Some(target) => target.push(attr),
                    None => return Err(ParseError::InvalidSyntax),
                }
            } else {
                list.push(attr);
            }
            self.skip_ws();
            self.eat(b']')?;
        }
        Ok(())
    }

    fn read_float(&mut self) -> Result<f64, ParseError> {
        let start = self.pos;
        while matches!(
            self.src.get(self.pos),
            Some(b'0'..=b'9' | b'.' | b'-' | b'+' | b'e' | b'E')
        ) {
            self.pos += 1;
        }
        core::str::from_utf8(&self.src[start..self.pos])
            .ok()
            .and_then(|t| t.parse::<f64>().ok())
            .ok_or(ParseError::InvalidSyntax)
    }

    /// f128 literals: f64 parsing cannot hold binary128 precision. f128 is
    /// unstable on this toolchain, so only exact integers are accepted (the
    /// printer's float decimals are rejected until stable f128 FromStr lands).
    fn read_float128(&mut self) -> Result<u128, ParseError> {
        let start = self.pos;
        while matches!(
            self.src.get(self.pos),
            Some(b'0'..=b'9' | b'.' | b'-' | b'+' | b'e' | b'E')
        ) {
            self.pos += 1;
        }
        let text = core::str::from_utf8(&self.src[start..self.pos])
            .map_err(|_| ParseError::InvalidSyntax)?;
        decimal_str_to_f128_bits(text).ok_or(ParseError::InvalidSyntax)
    }

    fn finish_arith(&mut self, block: Block, lhs: Value, bop: BinOp) -> Result<Value, ParseError> {
        self.skip_ws();
        let ty = self.func.value_type(lhs);
        let ch = self.peek().unwrap_or(0);
        let result = if ch == b'-' || ch.is_ascii_digit() {
            let imm = self.read_signed()?;
            self.func.append_arith_imm(block, ty, bop, lhs, imm)
        } else {
            let rhs = self.parse_value_ref()?;
            self.func
                .append_inst(block, ty, Opcode::Arith(Arith { op: bop, lhs, rhs }))
        };
        self.record_value(result);
        Ok(result)
    }

    fn parse_const(&mut self, block: Block) -> Result<Value, ParseError> {
        self.skip_ws();
        self.define_value_name()?;
        self.skip_ws();
        self.eat(b':')?;
        self.skip_ws();
        let ty = self.parse_type()?;
        self.skip_ws();
        self.eat(b'=')?;
        self.skip_ws();
        let op = match self.func.types.type_kind(ty) {
            TypeKind::Float(FloatKind::F128) => Opcode::Fconst128(self.read_float128()?),
            TypeKind::Float(_) => Opcode::Fconst(self.read_float()?),
            _ => Opcode::Iconst(self.read_signed()?),
        };
        let result = self.func.append_inst(block, ty, op);
        self.record_value(result);
        Ok(result)
    }

    fn label_to_block_num(&self, word_start: usize, prefix: &str) -> Result<u32, ParseError> {
        let digits = &self.src[word_start + prefix.len()..];
        let mut end = 0;
        while end < digits.len() && is_digit(digits[end]) {
            end += 1;
        }
        if end == 0 {
            return Err(ParseError::InvalidSyntax);
        }
        core::str::from_utf8(&digits[..end])
            .unwrap_or("")
            .parse::<u32>()
            .map_err(|_| ParseError::InvalidSyntax)
    }

    /// Consume the binary arithmetic operator at the cursor, if there is one.
    fn read_arith_op(&mut self) -> Option<BinOp> {
        let rest = &self.src[self.pos..];
        let (text, op) = ARITH_OPS
            .iter()
            .find(|(text, _)| rest.starts_with(text.as_bytes()))?;
        self.pos += text.len();
        Some(*op)
    }

    fn read_cmp_op(&mut self) -> Result<CmpOp, ParseError> {
        let rest = &self.src[self.pos..];
        let (text, op) = CMP_OPS
            .iter()
            .find(|(text, _)| rest.starts_with(text.as_bytes()))
            .ok_or(ParseError::InvalidSyntax)?;
        self.pos += text.len();
        Ok(*op)
    }

    fn parse_let(&mut self, block: Block) -> Result<Value, ParseError> {
        self.skip_ws();
        self.define_value_name()?;
        self.skip_ws();
        self.eat(b'=')?;
        self.skip_ws();
        if self.peek() == Some(b'v') {
            let lhs = self.parse_value_ref()?;
            if self.peek() == Some(b'.') {
                self.pos += 1;
                self.eat(b'#')?;
                let index =
                    u32::try_from(self.read_unsigned()?).map_err(|_| ParseError::InvalidSyntax)?;
                let field_ty = match self.func.types.type_kind(self.func.value_type(lhs)) {
                    TypeKind::Struct(flds) => flds
                        .get(index as usize)
                        .copied()
                        .ok_or(ParseError::InvalidSyntax)?,
                    _ => return Err(ParseError::InvalidSyntax),
                };
                let result = self.func.append_inst(
                    block,
                    field_ty,
                    Opcode::Extract(Extract {
                        aggregate: lhs,
                        index,
                    }),
                );
                self.record_value(result);
                return Ok(result);
            }
            self.skip_ws();
            if let Some(bop) = self.read_arith_op() {
                return self.finish_arith(block, lhs, bop);
            }
            let op = self.read_cmp_op()?;
            self.skip_ws();
            let rhs = self.parse_value_ref()?;
            let bool_t = self.func.types.intern(TypeKind::Bool);
            let result =
                self.func
                    .append_inst(block, bool_t, Opcode::Icmp(Compare { op, lhs, rhs }));
            self.record_value(result);
            return Ok(result);
        }
        let op = self.read_word().to_string();
        if op == "load" {
            let mem = self.parse_mem_flags()?;
            self.skip_ws();
            let ty = self.parse_type()?;
            self.skip_ws();
            self.eat(b',')?;
            self.skip_ws();
            let ptr = self.parse_value_ref()?;
            let result = self.func.append_load(block, ty, ptr, mem);
            self.record_value(result);
            return Ok(result);
        }
        if op == "intrinsic" {
            self.skip_ws();
            let ty = self.parse_type()?;
            let (name, args, imm, side_effects) = self.parse_intrinsic_body()?;
            let result = self
                .func
                .append_intrinsic(block, ty, &name, &args, imm, side_effects);
            self.record_value(result);
            return Ok(result);
        }
        if op == "atomic_rmw" {
            let rmw = self.parse_atomic_rmw()?;
            let result = self.func.append_atomic_rmw(block, rmw);
            self.record_value(result);
            return Ok(result);
        }
        if op == "global_addr" {
            let via_got = self.try_word("got");
            self.skip_ws();
            self.eat(b'@')?;
            let name = self.read_word().to_string();
            let symbol = self.func.intern_symbol(&name);
            let ptr_t = self.func.types.ptr_global();
            let result = self.func.append_inst(
                block,
                ptr_t,
                Opcode::GlobalAddr(GlobalAddr { symbol, via_got }),
            );
            self.record_value(result);
            return Ok(result);
        }
        if op == "dot" {
            self.skip_ws();
            let signed = match self.read_word() {
                "signed" => true,
                "unsigned" => false,
                _ => return Err(ParseError::InvalidSyntax),
            };
            self.skip_ws();
            let acc = self.parse_value_ref()?;
            self.skip_ws();
            self.eat(b',')?;
            self.skip_ws();
            let a = self.parse_value_ref()?;
            self.skip_ws();
            self.eat(b',')?;
            self.skip_ws();
            let b = self.parse_value_ref()?;
            let result = self.func.append_dot(block, acc, a, b, signed);
            self.record_value(result);
            return Ok(result);
        }
        if op == "reduce" {
            self.skip_ws();
            let bin_op = bin_op_of(self.read_word()).ok_or(ParseError::InvalidSyntax)?;
            self.skip_ws();
            self.eat(b',')?;
            self.skip_ws();
            let vector = self.parse_value_ref()?;
            let elem = match self.func.types.type_kind(self.func.value_type(vector)) {
                TypeKind::Vector(v) => v.elem,
                _ => return Err(ParseError::InvalidSyntax),
            };
            let result =
                self.func
                    .append_inst(block, elem, Opcode::Reduce(Reduce { vector, op: bin_op }));
            self.record_value(result);
            return Ok(result);
        }
        if op == "splat" {
            self.skip_ws();
            let ty = self.parse_type()?;
            self.skip_ws();
            self.eat(b',')?;
            self.skip_ws();
            let scalar = self.parse_value_ref()?;
            let result = self.func.append_splat(block, ty, scalar);
            self.record_value(result);
            return Ok(result);
        }
        if op == "va_arg" {
            self.skip_ws();
            let ty = self.parse_type()?;
            self.skip_ws();
            self.eat(b',')?;
            self.skip_ws();
            let list = self.parse_value_ref()?;
            let result = self.func.append_va_arg(block, list, ty);
            self.record_value(result);
            return Ok(result);
        }
        if let Some(uop) = unary_op_of(&op) {
            self.skip_ws();
            let ty = self.parse_type()?;
            self.skip_ws();
            self.eat(b',')?;
            self.skip_ws();
            let value = self.parse_value_ref()?;
            let result = self
                .func
                .append_inst(block, ty, Opcode::Unary(Unary { op: uop, value }));
            self.record_value(result);
            return Ok(result);
        }
        if op == "call_indirect" {
            self.skip_ws();
            let ty = self.parse_type()?;
            self.skip_ws();
            let target = self.parse_value_ref()?;
            let mut args: Vec<Value> = Vec::new();
            self.parse_call_args(&mut args)?;
            let extras = self.parse_call_extras()?;
            let list = self.func.intern_values(&args);
            let result = self.func.append_inst(
                block,
                ty,
                Opcode::CallIndirect(CallIndirect {
                    target,
                    args: list,
                    is_variadic: extras.is_variadic,
                    num_fixed: extras.num_fixed,
                    ret_dest: extras.ret_dest,
                    ret_regs: extras.ret_regs,
                    ret_pieces: extras.ret_pieces,
                    sret: extras.sret,
                }),
            );
            self.record_value(result);
            return Ok(result);
        }
        if op == "alloca" {
            self.skip_ws();
            let elem = self.parse_type()?;
            let ptr_t = self.func.types.ptr_global();
            let result = self
                .func
                .append_inst(block, ptr_t, Opcode::Alloca(Alloca { elem }));
            self.record_value(result);
            return Ok(result);
        }
        if op == "call" {
            self.skip_ws();
            let ty = self.parse_type()?;
            self.skip_ws();
            self.eat(b'@')?;
            let name = self.read_word().to_string();
            let mut args: Vec<Value> = Vec::new();
            self.parse_call_args(&mut args)?;
            let extras = self.parse_call_extras()?;
            let symbol = self.func.intern_symbol(&name);
            let list = self.func.intern_values(&args);
            let result = self.func.append_inst(
                block,
                ty,
                Opcode::Call(Call {
                    symbol,
                    args: list,
                    is_variadic: extras.is_variadic,
                    num_fixed: extras.num_fixed,
                    ret_dest: extras.ret_dest,
                    ret_regs: extras.ret_regs,
                    ret_pieces: extras.ret_pieces,
                    sret: extras.sret,
                }),
            );
            self.record_value(result);
            return Ok(result);
        }
        if op == "convert" {
            self.skip_ws();
            let kind = convert_kind_of(self.read_word()).ok_or(ParseError::InvalidSyntax)?;
            self.skip_ws();
            let ty = self.parse_type()?;
            self.skip_ws();
            self.eat(b',')?;
            self.skip_ws();
            let value = self.parse_value_ref()?;
            let result = self.func.append_convert(block, ty, kind, value);
            self.record_value(result);
            return Ok(result);
        }
        if op == "decode_low_float" || op == "encode_low_float" {
            self.skip_ws();
            let format = low_float_format_of(self.read_word()).ok_or(ParseError::InvalidSyntax)?;
            self.skip_ws();
            self.eat(b',')?;
            self.skip_ws();
            let value = self.parse_value_ref()?;
            let decode = op == "decode_low_float";
            let ty = if decode {
                self.func.types.intern(TypeKind::Float(FloatKind::F32))
            } else {
                self.func.types.intern(TypeKind::Int(IntDesc {
                    bits: format.payload_bits(),
                }))
            };
            let conversion = LowFloatConvert { value, format };
            let result = if decode {
                self.func
                    .append_inst(block, ty, Opcode::DecodeLowFloat(conversion))
            } else {
                self.func
                    .append_inst(block, ty, Opcode::EncodeLowFloat(conversion))
            };
            self.record_value(result);
            return Ok(result);
        }
        if op == "dequantize_nvfp4" || op == "quantize_nvfp4" {
            self.skip_ws();
            let block_application =
                scale_application_of(self.read_word()).ok_or(ParseError::InvalidSyntax)?;
            self.skip_ws();
            self.eat(b',')?;
            self.skip_ws();
            let global_application =
                scale_application_of(self.read_word()).ok_or(ParseError::InvalidSyntax)?;
            let mut operands = [Value::from_u32(0); 3];
            for operand in operands.iter_mut() {
                self.skip_ws();
                self.eat(b',')?;
                self.skip_ws();
                *operand = self.parse_value_ref()?;
            }
            let dequantize = op == "dequantize_nvfp4";
            let ty = if dequantize {
                self.func.types.intern(TypeKind::Float(FloatKind::F32))
            } else {
                self.func.types.intern(TypeKind::Int(IntDesc { bits: 8 }))
            };
            let conversion = NvFp4Convert {
                value: operands[0],
                block_scale: operands[1],
                global_scale: operands[2],
                block_application,
                global_application,
            };
            let result = if dequantize {
                self.func
                    .append_inst(block, ty, Opcode::DequantizeNvfp4(conversion))
            } else {
                self.func
                    .append_inst(block, ty, Opcode::QuantizeNvfp4(conversion))
            };
            self.record_value(result);
            return Ok(result);
        }
        if op == "struct" {
            self.skip_ws();
            self.eat(b'{')?;
            let mut fields: Vec<Value> = Vec::new();
            self.skip_ws();
            if self.peek() != Some(b'}') {
                loop {
                    self.skip_ws();
                    fields.push(self.parse_value_ref()?);
                    self.skip_ws();
                    if self.try_char(b',') {
                        continue;
                    }
                    break;
                }
            }
            self.skip_ws();
            self.eat(b'}')?;
            let mut field_types: Vec<Type> = Vec::new();
            for f in &fields {
                field_types.push(self.func.value_type(*f));
            }
            let st = self.func.types.intern(TypeKind::Struct(field_types));
            let result = self.func.append_struct_new(block, st, &fields);
            self.record_value(result);
            return Ok(result);
        }
        Err(ParseError::InvalidSyntax)
    }

    fn parse_atomic_rmw(&mut self) -> Result<AtomicRmw, ParseError> {
        self.skip_ws();
        let op = atomic_op_of(self.read_word()).ok_or(ParseError::InvalidSyntax)?;
        self.skip_ws();
        let scope = atomic_scope_of(self.read_word()).ok_or(ParseError::InvalidSyntax)?;
        self.skip_ws();
        let ordering = atomic_ordering_of(self.read_word()).ok_or(ParseError::InvalidSyntax)?;
        self.skip_ws();
        let ptr = self.parse_value_ref()?;
        self.skip_ws();
        self.eat(b',')?;
        self.skip_ws();
        let value = self.parse_value_ref()?;
        let mut compare: Option<Value> = None;
        if op == AtomicOp::CompareExchange {
            self.skip_ws();
            self.eat(b',')?;
            self.skip_ws();
            compare = Some(self.parse_value_ref()?);
        }
        Ok(AtomicRmw {
            op,
            ptr,
            value,
            compare,
            ordering,
            scope,
        })
    }

    /// Read `be` or `le` at the cursor; any other word leaves the cursor where it was.
    fn try_endian_word(&mut self) -> Option<Endianness> {
        let save = self.pos;
        self.skip_ws();
        match self.read_word() {
            "be" => Some(Endianness::Big),
            "le" => Some(Endianness::Little),
            _ => {
                self.pos = save;
                None
            }
        }
    }

    /// Parse the optional modifier words after `load`/`store`: `volatile`, `atomic <ordering>`,
    /// `align <n>`, `be`, `le`, in any order, each at most once. Absent modifiers keep the
    /// `MemFlags::new()` default (native byte order, not atomic, no alignment promise).
    fn parse_mem_flags(&mut self) -> Result<MemFlags, ParseError> {
        let mut mem = MemFlags::new();
        let (mut volatile, mut atomic, mut align, mut endian) = (false, false, false, false);
        loop {
            if self.try_word("volatile") {
                if core::mem::replace(&mut volatile, true) {
                    return Err(ParseError::InvalidSyntax);
                }
                mem.volatile = true;
            } else if self.try_word("atomic") {
                if core::mem::replace(&mut atomic, true) {
                    return Err(ParseError::InvalidSyntax);
                }
                self.skip_ws();
                mem.ordering =
                    Some(atomic_ordering_of(self.read_word()).ok_or(ParseError::InvalidSyntax)?);
            } else if self.try_word("align") {
                if core::mem::replace(&mut align, true) {
                    return Err(ParseError::InvalidSyntax);
                }
                self.skip_ws();
                mem.align =
                    u32::try_from(self.read_unsigned()?).map_err(|_| ParseError::InvalidSyntax)?;
            } else if let Some(order) = self.try_endian_word() {
                if core::mem::replace(&mut endian, true) {
                    return Err(ParseError::InvalidSyntax);
                }
                mem.endian = order;
            } else {
                return Ok(mem);
            }
        }
    }

    fn parse_store(&mut self, block: Block) -> Result<(), ParseError> {
        let mem = self.parse_mem_flags()?;
        self.skip_ws();
        let value = self.parse_value_ref()?;
        self.skip_ws();
        self.eat(b',')?;
        self.skip_ws();
        let ptr = self.parse_value_ref()?;
        self.func.append_store_mem(block, value, ptr, mem);
        Ok(())
    }

    /// The part of an intrinsic after its optional result type:
    /// `@name(args) [imm N] [pure]`. The name is a dotted identifier (`aarch64.aese`);
    /// `pure` clears `side_effects`, which is otherwise set.
    fn parse_intrinsic_body(&mut self) -> Result<(String, Vec<Value>, i64, bool), ParseError> {
        self.skip_ws();
        self.eat(b'@')?;
        let start = self.pos;
        while matches!(self.src.get(self.pos), Some(&c) if is_word_char(c) || c == b'.') {
            self.pos += 1;
        }
        if self.pos == start {
            return Err(ParseError::InvalidSyntax);
        }
        let name = core::str::from_utf8(&self.src[start..self.pos])
            .map_err(|_| ParseError::InvalidSyntax)?
            .to_string();
        let mut args: Vec<Value> = Vec::new();
        self.parse_call_args(&mut args)?;
        let (mut imm, mut side_effects) = (0, true);
        let (mut saw_imm, mut saw_pure) = (false, false);
        loop {
            if self.try_word("imm") {
                if core::mem::replace(&mut saw_imm, true) {
                    return Err(ParseError::InvalidSyntax);
                }
                self.skip_ws();
                imm = self.read_signed()?;
            } else if self.try_word("pure") {
                if core::mem::replace(&mut saw_pure, true) {
                    return Err(ParseError::InvalidSyntax);
                }
                side_effects = false;
            } else {
                return Ok((name, args, imm, side_effects));
            }
        }
    }

    fn parse_barrier(&mut self, block: Block) -> Result<(), ParseError> {
        self.skip_ws();
        let scope = barrier_scope_of(self.read_word()).ok_or(ParseError::InvalidSyntax)?;
        self.func.append_barrier(block, scope);
        Ok(())
    }

    fn parse_call_args(&mut self, list: &mut Vec<Value>) -> Result<(), ParseError> {
        self.skip_ws();
        self.eat(b'(')?;
        self.skip_ws();
        if self.peek() != Some(b')') {
            loop {
                self.skip_ws();
                list.push(self.parse_value_ref()?);
                self.skip_ws();
                if self.try_char(b',') {
                    continue;
                }
                break;
            }
        }
        self.skip_ws();
        self.eat(b')')
    }

    fn parse_call_extras(&mut self) -> Result<CallExtras, ParseError> {
        let mut out = CallExtras::default();
        loop {
            let save = self.pos;
            self.skip_ws();
            let word = self.read_word().to_string();
            if word == "variadic" {
                self.eat(b'(')?;
                out.num_fixed =
                    u32::try_from(self.read_unsigned()?).map_err(|_| ParseError::InvalidSyntax)?;
                self.eat(b')')?;
                out.is_variadic = true;
            } else if word == "sret" {
                out.sret = true;
            } else if word == "retdest" {
                self.eat(b'(')?;
                if self.peek() == Some(b'v') {
                    out.ret_dest = Some(self.parse_value_ref()?);
                } else {
                    self.expect_word("none")?;
                }
                self.eat(b',')?;
                self.eat(b'[')?;
                let mut count: usize = 0;
                if self.peek() != Some(b']') {
                    loop {
                        if count >= out.ret_pieces.len() {
                            return Err(ParseError::InvalidSyntax);
                        }
                        out.ret_pieces[count] = self.parse_ret_piece()?;
                        count += 1;
                        if self.try_char(b',') {
                            continue;
                        }
                        break;
                    }
                }
                self.eat(b']')?;
                self.eat(b')')?;
                out.ret_regs = count as u8;
            } else {
                self.pos = save;
                return Ok(out);
            }
        }
    }

    fn parse_ret_piece(&mut self) -> Result<RetPiece, ParseError> {
        let bank = self.read_word().to_string();
        let fp = if bank == "f" {
            true
        } else if bank == "i" {
            false
        } else {
            return Err(ParseError::InvalidSyntax);
        };
        self.eat(b'@')?;
        let offset = u8::try_from(self.read_unsigned()?).map_err(|_| ParseError::InvalidSyntax)?;
        self.eat(b':')?;
        let bytes = u8::try_from(self.read_unsigned()?).map_err(|_| ParseError::InvalidSyntax)?;
        Ok(RetPiece { fp, offset, bytes })
    }

    fn parse_named_operand(&mut self, name: &str) -> Result<Value, ParseError> {
        self.expect_word_after_ws(name)?;
        self.eat(b'=')?;
        self.parse_value_ref()
    }

    fn expect_word_after_ws(&mut self, word: &str) -> Result<(), ParseError> {
        self.skip_ws();
        self.expect_word(word)
    }

    fn parse_void_call(&mut self, block: Block) -> Result<(), ParseError> {
        self.skip_ws();
        self.eat(b'@')?;
        let name = self.read_word().to_string();
        let mut args: Vec<Value> = Vec::new();
        self.parse_call_args(&mut args)?;
        let extras = self.parse_call_extras()?;
        let symbol = self.func.intern_symbol(&name);
        let list = self.func.intern_values(&args);
        self.func.append_stmt_raw(
            block,
            Opcode::Call(Call {
                symbol,
                args: list,
                is_variadic: extras.is_variadic,
                num_fixed: extras.num_fixed,
                ret_dest: extras.ret_dest,
                ret_regs: extras.ret_regs,
                ret_pieces: extras.ret_pieces,
                sret: extras.sret,
            }),
        );
        Ok(())
    }

    fn parse_void_call_indirect(&mut self, block: Block) -> Result<(), ParseError> {
        self.skip_ws();
        let target = self.parse_value_ref()?;
        let mut args: Vec<Value> = Vec::new();
        self.parse_call_args(&mut args)?;
        let extras = self.parse_call_extras()?;
        let list = self.func.intern_values(&args);
        self.func.append_stmt_raw(
            block,
            Opcode::CallIndirect(CallIndirect {
                target,
                args: list,
                is_variadic: extras.is_variadic,
                num_fixed: extras.num_fixed,
                ret_dest: extras.ret_dest,
                ret_regs: extras.ret_regs,
                ret_pieces: extras.ret_pieces,
                sret: extras.sret,
            }),
        );
        Ok(())
    }

    fn parse_matmul_quant(&mut self) -> Result<MatMulQuant, ParseError> {
        self.eat(b'(')?;
        self.skip_ws();
        let scale_word = self.read_word().to_string();
        let scale = if scale_word == "scalar" {
            self.eat(b'=')?;
            MatMulScale::Scalar(
                u32::try_from(self.read_radix_unsigned()?)
                    .map_err(|_| ParseError::InvalidSyntax)?,
            )
        } else if scale_word == "per_col" {
            self.eat(b'[')?;
            let mut scales: Vec<u32> = Vec::new();
            if self.peek() != Some(b']') {
                loop {
                    scales.push(
                        u32::try_from(self.read_radix_unsigned()?)
                            .map_err(|_| ParseError::InvalidSyntax)?,
                    );
                    if self.try_char(b',') {
                        continue;
                    }
                    break;
                }
            }
            self.eat(b']')?;
            MatMulScale::PerColumn(self.func.intern_scales(&scales))
        } else {
            return Err(ParseError::InvalidSyntax);
        };
        self.eat(b',')?;
        self.expect_word_after_ws("relu")?;
        self.eat(b'=')?;
        let relu = self.read_bool()?;
        self.eat(b',')?;
        self.skip_ws();
        let out = matmul_quant_out_of(self.read_word()).ok_or(ParseError::InvalidSyntax)?;
        self.eat(b',')?;
        self.expect_word_after_ws("bias")?;
        let bias = if self.try_char(b'=') {
            self.expect_word_after_ws("none")?;
            None
        } else {
            self.eat(b'[')?;
            let mut values: Vec<i32> = Vec::new();
            if self.peek() != Some(b']') {
                loop {
                    values.push(
                        i32::try_from(self.read_signed()?)
                            .map_err(|_| ParseError::InvalidSyntax)?,
                    );
                    if self.try_char(b',') {
                        continue;
                    }
                    break;
                }
            }
            self.eat(b']')?;
            Some(self.func.intern_bias(&values))
        };
        let mut zero_point: i32 = 0;
        if self.try_char(b',') {
            self.expect_word_after_ws("zp")?;
            self.eat(b'=')?;
            zero_point =
                i32::try_from(self.read_signed()?).map_err(|_| ParseError::InvalidSyntax)?;
        }
        self.eat(b')')?;
        Ok(MatMulQuant {
            scale,
            relu,
            out,
            bias,
            zero_point,
        })
    }

    fn parse_matmul(&mut self, block: Block) -> Result<(), ParseError> {
        let c = self.parse_named_operand("c")?;
        self.skip_ws();
        self.eat(b',')?;
        let a = self.parse_named_operand("a")?;
        self.skip_ws();
        self.eat(b',')?;
        let b = self.parse_named_operand("b")?;
        self.skip_ws();
        self.eat(b'[')?;
        let m =
            u16::try_from(self.read_radix_unsigned()?).map_err(|_| ParseError::InvalidSyntax)?;
        self.expect_word_after_ws("x")?;
        let n =
            u16::try_from(self.read_radix_unsigned()?).map_err(|_| ParseError::InvalidSyntax)?;
        self.expect_word_after_ws("x")?;
        let k =
            u16::try_from(self.read_radix_unsigned()?).map_err(|_| ParseError::InvalidSyntax)?;
        self.skip_ws();
        self.eat(b']')?;
        self.skip_ws();
        let dtype = matmul_dtype_of(self.read_word()).ok_or(ParseError::InvalidSyntax)?;
        let accumulate = self.try_word("acc");
        let embedded = self.try_word("embedded");
        let mut input_signs: Option<InputSigns> = None;
        if self.try_word("a_uns") {
            self.eat(b'=')?;
            let a_unsigned = self.read_bool()?;
            self.eat(b',')?;
            self.expect_word_after_ws("b_uns")?;
            self.eat(b'=')?;
            let b_unsigned = self.read_bool()?;
            input_signs = Some(InputSigns {
                a_unsigned,
                b_unsigned,
            });
        }
        let mut quant: Option<MatMulQuant> = None;
        if self.try_word("quant") {
            quant = Some(self.parse_matmul_quant()?);
        }
        self.func.append_stmt_raw(
            block,
            Opcode::Matmul(MatMul {
                a,
                b,
                c,
                m,
                n,
                k,
                dtype,
                accumulate,
                embedded,
                quant,
                input_signs,
            }),
        );
        Ok(())
    }

    fn parse_select(&mut self, block: Block, name: &str) -> Result<Value, ParseError> {
        let num: usize = name[1..].parse().map_err(|_| ParseError::InvalidSyntax)?;
        if num != self.value_names.len() {
            return Err(ParseError::InvalidSyntax);
        }
        self.skip_ws();
        self.eat(b':')?;
        self.eat(b'=')?;
        self.skip_ws();
        self.expect_word("if")?;
        self.skip_ws();
        let cond = self.parse_value_ref()?;
        self.skip_ws();
        self.eat(b'{')?;
        self.skip_ws();
        let then_v = self.parse_value_ref()?;
        self.skip_ws();
        self.eat(b'}')?;
        self.skip_ws();
        self.expect_word("else")?;
        self.skip_ws();
        self.eat(b'{')?;
        self.skip_ws();
        let else_v = self.parse_value_ref()?;
        self.skip_ws();
        self.eat(b'}')?;
        let ty = self.func.value_type(then_v);
        let result = self.func.append_inst(
            block,
            ty,
            Opcode::Select(Select {
                cond,
                then: then_v,
                else_: else_v,
            }),
        );
        self.record_value(result);
        Ok(result)
    }

    fn parse_if(&mut self, block: Block) -> Result<(), ParseError> {
        self.skip_ws();
        let cond = self.parse_value_ref()?;
        let mut then_args: Vec<Value> = Vec::new();
        let mut else_args: Vec<Value> = Vec::new();
        self.skip_ws();
        self.eat(b'{')?;
        let then_target = self.parse_edge_into(&mut then_args)?;
        self.skip_ws();
        self.eat(b'}')?;
        self.skip_ws();
        self.expect_word("else")?;
        self.skip_ws();
        self.eat(b'{')?;
        let else_target = self.parse_edge_into(&mut else_args)?;
        self.skip_ws();
        self.eat(b'}')?;
        self.func.append_if(
            block,
            cond,
            EdgeDesc {
                target: then_target,
                args: then_args,
            },
            EdgeDesc {
                target: else_target,
                args: else_args,
            },
        );
        Ok(())
    }

    fn parse_edge_into(&mut self, list: &mut Vec<Value>) -> Result<Block, ParseError> {
        self.skip_ws();
        let word_start = self.pos;
        let label = self.read_word().to_string();
        if !label.starts_with("block") {
            return Err(ParseError::InvalidSyntax);
        }
        let bnum = self.label_to_block_num(word_start, "block")?;
        let target = self.checked_block(bnum)?;
        self.parse_edge_args(list)?;
        Ok(target)
    }

    fn parse_jump(&mut self, block: Block, label: &str) -> Result<(), ParseError> {
        let word_start = self.pos - label.len();
        let bnum = self.label_to_block_num(word_start, "block")?;
        let target = self.checked_block(bnum)?;
        let mut args: Vec<Value> = Vec::new();
        self.parse_edge_args(&mut args)?;
        self.func.set_jump(block, target, &args);
        Ok(())
    }

    fn parse_edge_args(&mut self, list: &mut Vec<Value>) -> Result<(), ParseError> {
        self.eat(b'(')?;
        self.skip_ws();
        if self.peek() != Some(b')') {
            loop {
                self.skip_ws();
                list.push(self.parse_value_ref()?);
                self.skip_ws();
                if self.try_char(b',') {
                    continue;
                }
                break;
            }
        }
        self.skip_ws();
        self.eat(b')')
    }

    fn parse_ret(&mut self, block: Block) -> Result<(), ParseError> {
        self.skip_ws();
        if self.peek() == Some(b'v') {
            let rest = &self.src[self.pos..];
            if rest.starts_with(b"void")
                && (self.pos + 4 >= self.src.len() || !is_word_char(self.src[self.pos + 4]))
            {
                self.pos += 4;
                self.func
                    .set_terminator(block, Terminator::Ret(Ret::none()));
                return Ok(());
            }
            let mut values = [Value::from_u32(0); 4];
            let mut count: usize = 0;
            loop {
                if count >= values.len() {
                    return Err(ParseError::InvalidSyntax);
                }
                values[count] = self.parse_value_ref()?;
                count += 1;
                self.skip_ws();
                if self.try_char(b',') {
                    self.skip_ws();
                    continue;
                }
                break;
            }
            self.func
                .set_terminator(block, Terminator::Ret(Ret::many(&values[..count])));
            return Ok(());
        }
        Err(ParseError::InvalidSyntax)
    }

    fn attach_all(&mut self, list: &[Attribute], value: Value) -> Result<(), ParseError> {
        for attr in list {
            self.func.add_attr(AttrTarget::Value(value), attr.clone());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verify;

    fn parse_ok(text: &str) -> Function {
        parse(text).unwrap_or_else(|e| panic!("parse failed: {e:?}\n{text}"))
    }

    #[test]
    fn round_trips_a_float_constant() {
        let func = parse_ok("fn {\n  block0():\n    const v0: f32 = 1.5\n    ret v0\n}");
        assert!(matches!(
            func.opcode(func.defining_inst(Value::from_u32(0)).unwrap()),
            Opcode::Fconst(bits) if (bits - 1.5f64).abs() < f64::EPSILON
        ));
    }

    #[test]
    fn round_trips_a_minimal_function() {
        let func = parse_ok("fn {\n  block0():\n    const v0: i32 = 42\n    ret v0\n}");
        assert_eq!(func.block_count(), 1);
        assert_eq!(func.value_count(), 1);
    }

    #[test]
    fn round_trips_a_function_level_attribute() {
        let func = parse_ok("#[inline]\nfn {\n  block0():\n    ret void\n}");
        let mut found = false;
        let mut it = func.attributes_of(AttrTarget::Func);
        while it.next_attr().is_some() {
            found = true;
        }
        assert!(found);
    }

    #[test]
    fn round_trips_a_namespaced_attribute() {
        let func = parse_ok("#[target.clone = \"rv64gcv\"]\nfn {\n  block0():\n    ret void\n}");
        let mut it = func.attributes_of(AttrTarget::Func);
        let attr = it.next_attr().expect("custom attr").clone();
        match attr {
            Attribute::Custom(c) => {
                assert_eq!(c.namespace, "target");
                assert_eq!(c.key, "clone");
            }
            _ => panic!("expected custom"),
        }
    }

    #[test]
    fn round_trips_the_arithmetic_operators() {
        let func =
            parse_ok("fn {\n  block0(v0: i32, v1: i32):\n    let v2 = v0 + v1\n    ret v2\n}");
        assert!(matches!(
            func.opcode(func.defining_inst(Value::from_u32(2)).unwrap()),
            Opcode::Arith(a) if a.op == BinOp::Add
        ));
    }

    #[test]
    fn round_trips_a_value_attribute() {
        let func = parse_ok(
            "fn {\n  block0(v0: i32, v1: i32):\n    #[align(16)]\n    let v2 = v0 + v1\n    ret v2\n}",
        );
        let mut it = func.attributes_of(AttrTarget::Value(Value::from_u32(2)));
        assert!(it.next_attr().is_some());
    }

    #[test]
    fn round_trips_a_function_with_a_conditional() {
        let func = parse_ok(
            "fn {\n  block0():\n    const v0: bool = 1\n    const v1: i32 = 5\n    if v0 { block1(v1) } else { block2() }\n    ret void\n\n  block1():\n    ret void\n  block2():\n    ret void\n}",
        );
        assert_eq!(func.block_count(), 3);
    }

    #[test]
    fn round_trips_a_function_with_iadd_and_a_jump() {
        let func = parse_ok(
            "fn {\n  block0(v0: i32, v1: i32):\n    let v2 = v0 + v1\n    block1(v2)\n\n  block1(v3: i32):\n    ret v3\n}",
        );
        assert_eq!(func.block_count(), 2);
        assert!(matches!(
            func.terminator(Block::from_u32(0)),
            Some(Terminator::Jump(_))
        ));
    }

    #[test]
    fn rejects_an_out_of_range_block_label() {
        assert!(parse("fn {\n  block7():\n    ret\n}").is_err());
    }

    #[test]
    fn round_trips_the_canonical_max_function() {
        let func = parse_ok(
            "fn {\n  block0(v0: i32, v1: i32):\n    let v2 = v0 >s v1\n    if v2 { block1(v0) } else { block1(v1) }\n    ret void\n\n  block1(v3: i32):\n    ret v3\n}",
        );
        let diags = verify::verify(&func, verify::Profile::High);
        assert!(diags.ok());
        let diags = verify::verify(&func, verify::Profile::Low);
        assert!(diags.ok());
    }

    fn only_mem(func: &Function, which: usize) -> MemFlags {
        let mut seen = 0;
        for bi in 0..func.block_count() {
            for &inst in func.block_insts(Block::from_u32(bi as u32)) {
                let mem = match func.opcode(inst) {
                    Opcode::Load(l) => l.mem,
                    Opcode::Store(st) => st.mem,
                    _ => continue,
                };
                if seen == which {
                    return mem;
                }
                seen += 1;
            }
        }
        panic!("no memory access #{which}");
    }

    /// The canonical spelling of `mem` as documented in the module comment.
    fn mem_words(mem: &MemFlags) -> String {
        let mut out = String::new();
        if mem.volatile {
            out.push_str("volatile ");
        }
        if let Some(o) = mem.ordering {
            out.push_str(match o {
                AtomicOrdering::Relaxed => "atomic relaxed ",
                AtomicOrdering::Acquire => "atomic acquire ",
                AtomicOrdering::Release => "atomic release ",
                AtomicOrdering::AcqRel => "atomic acq_rel ",
                AtomicOrdering::SeqCst => "atomic seq_cst ",
            });
        }
        if mem.align != 0 {
            out.push_str(&alloc::format!("align {} ", mem.align));
        }
        match mem.endian {
            Endianness::Native => {}
            Endianness::Big => out.push_str("be "),
            Endianness::Little => out.push_str("le "),
        }
        out
    }

    #[test]
    fn parses_load_and_store_without_modifiers_as_plain() {
        let func = parse_ok(
            "fn {\n  block0(v0: ptr, v1: i32):\n    let v2 = load i32, v0\n    store v1, v0\n    ret v2\n}",
        );
        assert_eq!(only_mem(&func, 0), MemFlags::new());
        assert_eq!(only_mem(&func, 1), MemFlags::new());
        assert!(only_mem(&func, 0).is_plain());
    }

    #[test]
    fn mem_flags_text_round_trips_every_modifier() {
        let orderings = [
            None,
            Some(AtomicOrdering::Relaxed),
            Some(AtomicOrdering::Acquire),
            Some(AtomicOrdering::Release),
            Some(AtomicOrdering::AcqRel),
            Some(AtomicOrdering::SeqCst),
        ];
        let endians = [Endianness::Native, Endianness::Little, Endianness::Big];
        for volatile in [false, true] {
            for align in [0u32, 1, 8, 4096] {
                for ordering in orderings {
                    for endian in endians {
                        let mem = MemFlags {
                            volatile,
                            align,
                            ordering,
                            endian,
                        };
                        let words = mem_words(&mem);
                        let text = alloc::format!(
                            "fn {{\n  block0(v0: ptr, v1: i32):\n    let v2 = load {words}i32, v0\n    store {words}v1, v0\n    ret v2\n}}"
                        );
                        let func = parse_ok(&text);
                        assert_eq!(only_mem(&func, 0), mem, "{text}");
                        assert_eq!(only_mem(&func, 1), mem, "{text}");
                    }
                }
            }
        }
    }

    #[test]
    fn mem_flag_modifiers_parse_in_any_order() {
        let func = parse_ok(
            "fn {\n  block0(v0: ptr):\n    let v1 = load be align 8 atomic seq_cst volatile i64, v0\n    ret v1\n}",
        );
        assert_eq!(
            only_mem(&func, 0),
            MemFlags {
                volatile: true,
                align: 8,
                ordering: Some(AtomicOrdering::SeqCst),
                endian: Endianness::Big,
            }
        );
    }

    #[test]
    fn rejects_malformed_mem_flags() {
        for words in [
            "volatile volatile ",
            "align ",
            "align x ",
            "atomic ",
            "atomic strong ",
            "atomic acquire atomic acquire ",
            "be le ",
            "align 4 align 4 ",
        ] {
            let text = alloc::format!(
                "fn {{\n  block0(v0: ptr):\n    let v1 = load {words}i32, v0\n    ret v1\n}}"
            );
            assert!(parse(&text).is_err(), "{text}");
        }
    }

    #[test]
    fn round_trips_a_valued_intrinsic() {
        let func = parse_ok(
            "fn {\n  block0(v0: i32, v1: i32):\n    let v2 = intrinsic i32 @aarch64.aese(v0, v1) imm -3\n    ret v2\n}",
        );
        let inst = func.defining_inst(Value::from_u32(2)).unwrap();
        let Opcode::Intrinsic(i) = func.opcode(inst) else {
            panic!("expected intrinsic");
        };
        assert_eq!(func.symbol_name(i.symbol), "aarch64.aese");
        assert_eq!(
            func.value_list(i.args),
            &[Value::from_u32(0), Value::from_u32(1)]
        );
        assert_eq!(i.imm, -3);
        assert!(i.side_effects);
    }

    #[test]
    fn round_trips_a_pure_and_a_void_intrinsic() {
        let func = parse_ok(
            "fn {\n  block0(v0: i32):\n    let v1 = intrinsic i32 @x86.bswap(v0) pure\n    intrinsic @arm.dmb() imm 15\n    intrinsic @nop.pure() pure imm 1\n    ret v1\n}",
        );
        let insts = func.block_insts(Block::from_u32(0));
        let get = |n: usize| match func.opcode(insts[n]) {
            Opcode::Intrinsic(i) => i,
            _ => panic!("expected intrinsic"),
        };
        let (a, b, c) = (get(0), get(1), get(2));
        assert!(!a.side_effects && a.imm == 0);
        assert!(func.inst_result(insts[1]).is_none());
        assert!(b.side_effects && b.imm == 15 && b.args.len == 0);
        assert!(!c.side_effects && c.imm == 1);
        assert_eq!(func.symbol_name(b.symbol), "arm.dmb");
    }

    // --- Signedness on the operation ---------------------------------------------

    const ALL_BIN_OPS: [BinOp; 17] = [
        BinOp::Add,
        BinOp::Sub,
        BinOp::Mul,
        BinOp::Div,
        BinOp::Rem,
        BinOp::BitAnd,
        BinOp::BitOr,
        BinOp::BitXor,
        BinOp::Shl,
        BinOp::Shr,
        BinOp::Sar,
        BinOp::UDiv,
        BinOp::SDiv,
        BinOp::URem,
        BinOp::SRem,
        BinOp::UMulh,
        BinOp::SMulh,
    ];

    const ALL_CMP_OPS: [CmpOp; 14] = [
        CmpOp::Eq,
        CmpOp::Ne,
        CmpOp::Lt,
        CmpOp::Le,
        CmpOp::Gt,
        CmpOp::Ge,
        CmpOp::Slt,
        CmpOp::Sle,
        CmpOp::Sgt,
        CmpOp::Sge,
        CmpOp::Ult,
        CmpOp::Ule,
        CmpOp::Ugt,
        CmpOp::Uge,
    ];

    fn assert_verifies(func: &Function, what: &str) {
        let diags = verify::verify(func, verify::Profile::High);
        assert!(diags.ok(), "{what}: {:?}", diags.items());
    }

    fn type_text(func: &Function, v: u32) -> String {
        func.types
            .display(func.value_type(Value::from_u32(v)))
            .to_string()
    }

    #[test]
    fn every_binary_operator_parses_from_its_own_symbol() {
        for op in ALL_BIN_OPS {
            let ty = if matches!(op, BinOp::Div | BinOp::Rem) {
                "f32"
            } else {
                "i32"
            };
            let text = alloc::format!(
                "fn {{\n  block0(v0: {ty}, v1: {ty}):\n    let v2 = v0 {} v1\n    ret v2\n}}",
                op.symbol()
            );
            let func = parse_ok(&text);
            let inst = func.defining_inst(Value::from_u32(2)).unwrap();
            assert!(
                matches!(func.opcode(inst), Opcode::Arith(a)
                    if a.op == op && a.lhs == Value::from_u32(0) && a.rhs == Value::from_u32(1)),
                "{op:?}: {text}"
            );
            assert_verifies(&func, &text);
        }
    }

    #[test]
    fn every_binary_operator_has_an_immediate_form() {
        for op in ALL_BIN_OPS {
            let ty = if matches!(op, BinOp::Div | BinOp::Rem) {
                "f32"
            } else {
                "i32"
            };
            let text = alloc::format!(
                "fn {{\n  block0(v0: {ty}):\n    let v1 = v0 {} 3\n    ret v1\n}}",
                op.symbol()
            );
            let func = parse_ok(&text);
            let inst = func.defining_inst(Value::from_u32(1)).unwrap();
            assert!(
                matches!(func.opcode(inst), Opcode::ArithImm(a) if a.op == op && a.imm == 3),
                "{op:?}: {text}"
            );
            assert_verifies(&func, &text);
        }
    }

    #[test]
    fn operator_suffixes_pick_the_signed_or_unsigned_variant() {
        let op_of = |sym: &str| {
            let func = parse_ok(&alloc::format!(
                "fn {{\n  block0(v0: i64, v1: i64):\n    let v2 = v0 {sym} v1\n    ret v2\n}}"
            ));
            let inst = func.defining_inst(Value::from_u32(2)).unwrap();
            match func.opcode(inst) {
                Opcode::Arith(a) => a.op,
                other => panic!("expected arith, got {other:?}"),
            }
        };
        assert_eq!(op_of(">>"), BinOp::Shr);
        assert_eq!(op_of(">>s"), BinOp::Sar);
        assert_eq!(op_of("/u"), BinOp::UDiv);
        assert_eq!(op_of("/s"), BinOp::SDiv);
        assert_eq!(op_of("%u"), BinOp::URem);
        assert_eq!(op_of("%s"), BinOp::SRem);
        assert_eq!(op_of("*"), BinOp::Mul);
        assert_eq!(op_of("*hu"), BinOp::UMulh);
        assert_eq!(op_of("*hs"), BinOp::SMulh);
    }

    #[test]
    fn every_comparison_parses_from_its_own_symbol() {
        for op in ALL_CMP_OPS {
            let ty = if matches!(op, CmpOp::Lt | CmpOp::Le | CmpOp::Gt | CmpOp::Ge) {
                "f32"
            } else {
                "i32"
            };
            let text = alloc::format!(
                "fn {{\n  block0(v0: {ty}, v1: {ty}):\n    let v2 = v0 {} v1\n    ret v2\n}}",
                op.symbol()
            );
            let func = parse_ok(&text);
            let inst = func.defining_inst(Value::from_u32(2)).unwrap();
            assert!(
                matches!(func.opcode(inst), Opcode::Icmp(c) if c.op == op),
                "{op:?}: {text}"
            );
            assert_eq!(type_text(&func, 2), "bool");
            assert_verifies(&func, &text);
        }
        let ptrs =
            parse_ok("fn {\n  block0(v0: ptr, v1: ptr):\n    let v2 = v0 <u v1\n    ret v2\n}");
        assert_verifies(&ptrs, "ptr <u ptr");
    }

    #[test]
    fn the_verifier_rejects_an_operator_applied_to_the_wrong_domain() {
        for (ty, sym) in [
            ("i32", "/"),
            ("i32", "%"),
            ("f32", "/u"),
            ("f32", "/s"),
            ("f32", "%u"),
            ("f32", ">>"),
            ("f32", ">>s"),
            ("f32", "*hu"),
            ("i32", "<"),
            ("i32", ">="),
            ("f32", "<s"),
            ("f32", ">u"),
            ("ptr", "<s"),
        ] {
            let text = alloc::format!(
                "fn {{\n  block0(v0: {ty}, v1: {ty}):\n    let v2 = v0 {sym} v1\n    ret v2\n}}"
            );
            let func = parse_ok(&text);
            let diags = verify::verify(&func, verify::Profile::High);
            assert!(
                matches!(diags.items(), [verify::Diagnostic::InvalidOperation(_)]),
                "{text}: {:?}",
                diags.items()
            );
        }
    }

    #[test]
    fn every_conversion_kind_parses_with_its_result_type() {
        use ConvertKind::*;
        for (kind, src, dst) in [
            (Trunc, "i64", "i8"),
            (Zext, "bool", "i32"),
            (Zext, "i8", "i64"),
            (Sext, "i8", "i64"),
            (SiToFp, "i32", "f32"),
            (UiToFp, "i64", "f64"),
            (FpToSi, "f32", "i32"),
            (FpToUi, "f64", "i64"),
            (FpResize, "f32", "f64"),
            (FpResize, "f128", "f16"),
            (Zext, "<4 x i8>", "<4 x i32>"),
        ] {
            let text = alloc::format!(
                "fn {{\n  block0(v0: {src}):\n    let v1 = convert {} {dst}, v0\n    ret v1\n}}",
                kind.name()
            );
            let func = parse_ok(&text);
            let inst = func.defining_inst(Value::from_u32(1)).unwrap();
            assert!(
                matches!(func.opcode(inst), Opcode::Convert(c)
                    if c.kind == kind && c.value == Value::from_u32(0)),
                "{text}"
            );
            assert_eq!(type_text(&func, 1), dst, "{text}");
            assert_verifies(&func, &text);
        }
    }

    #[test]
    fn a_conversion_needs_a_known_kind() {
        for text in [
            "fn {\n  block0(v0: i8):\n    let v1 = convert i32, v0\n    ret v1\n}",
            "fn {\n  block0(v0: i8):\n    let v1 = convert widen i32, v0\n    ret v1\n}",
            "fn {\n  block0(v0: i8):\n    let v1 = convert sext, v0\n    ret v1\n}",
        ] {
            assert!(parse(text).is_err(), "{text}");
        }
    }

    #[test]
    fn dot_spells_its_signedness() {
        for (word, signed) in [("signed", true), ("unsigned", false)] {
            let func = parse_ok(&alloc::format!(
                "fn {{\n  block0(v0: <4 x i32>, v1: <16 x i8>, v2: <16 x i8>):\n    let v3 = dot {word} v0, v1, v2\n    ret v3\n}}"
            ));
            let inst = func.defining_inst(Value::from_u32(3)).unwrap();
            assert!(
                matches!(func.opcode(inst), Opcode::Dot(d)
                    if d.signed == signed && d.acc == Value::from_u32(0)
                        && d.a == Value::from_u32(1) && d.b == Value::from_u32(2)),
                "{word}"
            );
            assert_eq!(type_text(&func, 3), "<4 x i32>");
        }
        for text in [
            "fn {\n  block0(v0: <4 x i32>, v1: <16 x i8>, v2: <16 x i8>):\n    let v3 = dot v0, v1, v2\n    ret v3\n}",
            "fn {\n  block0(v0: <4 x i32>, v1: <16 x i8>, v2: <16 x i8>):\n    let v3 = dot sgn v0, v1, v2\n    ret v3\n}",
        ] {
            assert!(parse(text).is_err(), "{text}");
        }
    }

    #[test]
    fn atomic_min_and_max_are_signed_or_unsigned() {
        for (word, op) in [
            ("add", AtomicOp::Add),
            ("smin", AtomicOp::SMin),
            ("smax", AtomicOp::SMax),
            ("umin", AtomicOp::UMin),
            ("umax", AtomicOp::UMax),
            ("bit_and", AtomicOp::BitAnd),
            ("bit_or", AtomicOp::BitOr),
            ("bit_xor", AtomicOp::BitXor),
            ("exchange", AtomicOp::Exchange),
        ] {
            let func = parse_ok(&alloc::format!(
                "fn {{\n  block0(v0: ptr, v1: i32):\n    let v2 = atomic_rmw {word} device seq_cst v0, v1\n    ret v2\n}}"
            ));
            let inst = func.defining_inst(Value::from_u32(2)).unwrap();
            assert!(
                matches!(func.opcode(inst), Opcode::AtomicRmw(a) if a.op == op),
                "{word}"
            );
            assert_verifies(&func, word);
        }
        for word in ["min", "max"] {
            let text = alloc::format!(
                "fn {{\n  block0(v0: ptr, v1: i32):\n    let v2 = atomic_rmw {word} device seq_cst v0, v1\n    ret v2\n}}"
            );
            assert!(parse(&text).is_err(), "{word}");
        }
    }

    #[test]
    fn integer_types_have_no_unsigned_spelling_and_mulh_has_no_bare_word() {
        assert!(parse("fn {\n  block0(v0: u32):\n    ret v0\n}").is_err());
        assert!(parse("fn {\n  block0():\n    const v0: u8 = 1\n    ret v0\n}").is_err());
        assert!(parse("fn {\n  block0():\n    const v0: i8 = 1\n    ret v0\n}").is_ok());
        for word in ["mulh", "min", "max"] {
            let text = alloc::format!(
                "fn {{\n  block0(v0: <4 x i32>):\n    let v1 = reduce {word}, v0\n    ret v1\n}}"
            );
            assert!(parse(&text).is_err(), "{word}");
        }
        for word in ["umulh", "sdiv", "sar"] {
            let text = alloc::format!(
                "fn {{\n  block0(v0: <4 x i32>):\n    let v1 = reduce {word}, v0\n    ret v1\n}}"
            );
            // The word parses; the verifier is what limits reductions to the lane-wise ops.
            let func = parse_ok(&text);
            assert!(!verify::verify(&func, verify::Profile::High).ok(), "{word}");
        }
    }

    #[test]
    fn low_float_and_nvfp4_results_are_plain_integers() {
        let func = parse_ok(
            "fn {\n  block0(v0: f32, v1: i8, v2: f32):\n    let v3 = encode_low_float bf16, v0\n    let v4 = encode_low_float f8_e4m3, v0\n    let v5 = quantize_nvfp4 multiply, divide, v0, v1, v2\n    ret v3\n}",
        );
        assert_eq!(type_text(&func, 3), "i16");
        assert_eq!(type_text(&func, 4), "i8");
        assert_eq!(type_text(&func, 5), "i8");
    }
}
