//! Integer data processing: moves, add/subtract, logical, multiply, divide, shifts,
//! bitfields, conditional select/compare, `adr`/`adrp`.
//!
//! Owns `Movz`, `Movn`, `Movk`, `ArithImm`, `ArithReg`, `ArithExt`, `LogicReg`,
//! `LogicImm`, `Mul`, `MulLong`, `MulHigh`, `Adr`, `Csel`, `Variable`, `CondCompare`,
//! `ArithCarry`, `Unary`, `Bitfield` and `Extract`.

use super::Context;
use super::reg::{condition, gpr, gpr_sp};
use crate::decode::{
    AddCarry, Address, ArithOp, Arithmetic, ArithmeticExtended, ArithmeticReg, Bitfield,
    BitfieldOp, CompareOperand, CondCompare, Condition, Extend, Extract, Instruction, Logic,
    LogicImm, LogicOp, Multiply, MultiplyHigh, MultiplyLong, Select, Shift, Unary, UnaryOp,
    Variable, VariableOp, Wide, Width,
};
use core::fmt::{self, Formatter};

pub(super) fn write(
    f: &mut Formatter<'_>,
    cx: &Context<'_>,
    instruction: &Instruction,
) -> fmt::Result {
    match *instruction {
        Instruction::Movz(w) => movz(f, w),
        Instruction::Movn(w) => movn(f, w),
        Instruction::Movk(w) => {
            f.write_str("movk ")?;
            gpr(f, w.width, w.rd)?;
            write!(f, ", #{:#x}", w.immediate)?;
            lsl_suffix(f, w.shift)
        }
        Instruction::ArithImm(a) => arith_imm(f, a),
        Instruction::ArithReg(a) => arith_reg(f, a),
        Instruction::ArithExt(a) => arith_ext(f, a),
        Instruction::LogicReg(l) => logic_reg(f, l),
        Instruction::LogicImm(l) => logic_imm(f, l),
        Instruction::Mul(m) => mul(f, m),
        Instruction::MulLong(m) => mul_long(f, m),
        Instruction::MulHigh(m) => mul_high(f, m),
        Instruction::Adr(a) => adr(f, cx, a),
        Instruction::Csel(s) => csel(f, s),
        Instruction::Variable(v) => variable(f, v),
        Instruction::CondCompare(c) => cond_compare(f, c),
        Instruction::ArithCarry(a) => arith_carry(f, a),
        Instruction::Unary(u) => unary(f, u),
        Instruction::Bitfield(b) => bitfield(f, b),
        Instruction::Extract(e) => extract(f, e),
        _ => unreachable!("{instruction:?} is not an integer instruction"),
    }
}

/// `, lsl #n` unless `n` is zero.
fn lsl_suffix(f: &mut Formatter<'_>, amount: u8) -> fmt::Result {
    if amount == 0 {
        Ok(())
    } else {
        write!(f, ", lsl #{amount}")
    }
}

fn shift_name(shift: Shift) -> &'static str {
    match shift {
        Shift::Lsl => "lsl",
        Shift::Lsr => "lsr",
        Shift::Asr => "asr",
        Shift::Ror => "ror",
    }
}

fn extend_name(extend: Extend) -> &'static str {
    match extend {
        Extend::Uxtb => "uxtb",
        Extend::Uxth => "uxth",
        Extend::Uxtw => "uxtw",
        Extend::Uxtx => "uxtx",
        Extend::Sxtb => "sxtb",
        Extend::Sxth => "sxth",
        Extend::Sxtw => "sxtw",
        Extend::Sxtx => "sxtx",
    }
}

/// `, <shift> #n`; a zero `lsl` is not printed (a zero `ror` is).
fn shift_operand(f: &mut Formatter<'_>, shift: Shift, amount: u8) -> fmt::Result {
    if shift == Shift::Lsl && amount == 0 {
        Ok(())
    } else {
        write!(f, ", {} #{amount}", shift_name(shift))
    }
}

fn invert(cond: Condition) -> Condition {
    use Condition::*;
    match cond {
        Eq => Ne,
        Ne => Eq,
        Cs => Cc,
        Cc => Cs,
        Mi => Pl,
        Pl => Mi,
        Vs => Vc,
        Vc => Vs,
        Hi => Ls,
        Ls => Hi,
        Ge => Lt,
        Lt => Ge,
        Gt => Le,
        Le => Gt,
        Al => Nv,
        Nv => Al,
    }
}

fn bits(width: Width) -> u16 {
    u16::from(width as u8)
}

/// The width's all-ones mask.
fn mask(width: Width) -> u64 {
    match width {
        Width::W32 => u64::from(u32::MAX),
        Width::X64 => u64::MAX,
    }
}

fn movz(f: &mut Formatter<'_>, w: Wide) -> fmt::Result {
    if w.immediate == 0 && w.shift != 0 {
        f.write_str("movz ")?;
        gpr(f, w.width, w.rd)?;
        write!(f, ", #{:#x}", w.immediate)?;
        return lsl_suffix(f, w.shift);
    }
    let value = (u64::from(w.immediate) << (w.shift & 63)) & mask(w.width);
    f.write_str("mov ")?;
    gpr(f, w.width, w.rd)?;
    write!(f, ", #{value:#x}")
}

fn movn(f: &mut Formatter<'_>, w: Wide) -> fmt::Result {
    // `mov` only where a `movz` could not have produced the same value.
    if (w.immediate == 0 && w.shift != 0) || (w.width == Width::W32 && w.immediate == 0xffff) {
        f.write_str("movn ")?;
        gpr(f, w.width, w.rd)?;
        write!(f, ", #{:#x}", w.immediate)?;
        return lsl_suffix(f, w.shift);
    }
    let value = !(u64::from(w.immediate) << (w.shift & 63)) & mask(w.width);
    f.write_str("mov ")?;
    gpr(f, w.width, w.rd)?;
    write!(f, ", #{value:#x}")
}

fn arith_imm(f: &mut Formatter<'_>, a: Arithmetic) -> fmt::Result {
    let add = a.op == ArithOp::Add;
    if !a.flags && add && a.immediate == 0 && !a.shifted && (a.rd == 31 || a.rn == 31) {
        f.write_str("mov ")?;
        gpr_sp(f, a.width, a.rd)?;
        f.write_str(", ")?;
        return gpr_sp(f, a.width, a.rn);
    }
    if a.flags && a.rd == 31 {
        f.write_str(if add { "cmn " } else { "cmp " })?;
    } else {
        f.write_str(match (add, a.flags) {
            (true, false) => "add ",
            (true, true) => "adds ",
            (false, false) => "sub ",
            (false, true) => "subs ",
        })?;
        if a.flags {
            gpr(f, a.width, a.rd)?;
        } else {
            gpr_sp(f, a.width, a.rd)?;
        }
        f.write_str(", ")?;
    }
    gpr_sp(f, a.width, a.rn)?;
    write!(f, ", #{:#x}", a.immediate)?;
    if a.shifted {
        f.write_str(", lsl #12")?;
    }
    Ok(())
}

fn arith_reg(f: &mut Formatter<'_>, a: ArithmeticReg) -> fmt::Result {
    let add = a.op == ArithOp::Add;
    let mut rn = Some(a.rn);
    if a.flags && a.rd == 31 {
        f.write_str(if add { "cmn " } else { "cmp " })?;
    } else if !add && a.rn == 31 {
        f.write_str(if a.flags { "negs " } else { "neg " })?;
        gpr(f, a.width, a.rd)?;
        f.write_str(", ")?;
        rn = None;
    } else {
        f.write_str(match (add, a.flags) {
            (true, false) => "add ",
            (true, true) => "adds ",
            (false, false) => "sub ",
            (false, true) => "subs ",
        })?;
        gpr(f, a.width, a.rd)?;
        f.write_str(", ")?;
    }
    if let Some(rn) = rn {
        gpr(f, a.width, rn)?;
        f.write_str(", ")?;
    }
    gpr(f, a.width, a.rm)?;
    shift_operand(f, a.shift, a.amount)
}

fn arith_ext(f: &mut Formatter<'_>, a: ArithmeticExtended) -> fmt::Result {
    let add = a.op == ArithOp::Add;
    // `rd` is the stack pointer only where the instruction does not set flags.
    let rd_sp = !a.flags && a.rd == 31;
    if a.flags && a.rd == 31 {
        f.write_str(if add { "cmn " } else { "cmp " })?;
    } else {
        f.write_str(match (add, a.flags) {
            (true, false) => "add ",
            (true, true) => "adds ",
            (false, false) => "sub ",
            (false, true) => "subs ",
        })?;
        if a.flags {
            gpr(f, a.width, a.rd)?;
        } else {
            gpr_sp(f, a.width, a.rd)?;
        }
        f.write_str(", ")?;
    }
    gpr_sp(f, a.width, a.rn)?;
    f.write_str(", ")?;
    let wide_rm = a.width == Width::X64 && matches!(a.extend, Extend::Uxtx | Extend::Sxtx);
    gpr(f, if wide_rm { Width::X64 } else { Width::W32 }, a.rm)?;
    let native = match a.width {
        Width::W32 => Extend::Uxtw,
        Width::X64 => Extend::Uxtx,
    };
    if (rd_sp || a.rn == 31) && a.extend == native {
        lsl_suffix(f, a.amount)
    } else {
        write!(f, ", {}", extend_name(a.extend))?;
        if a.amount != 0 {
            write!(f, " #{}", a.amount)?;
        }
        Ok(())
    }
}

fn logic_reg(f: &mut Formatter<'_>, l: Logic) -> fmt::Result {
    let plain_shift = l.shift == Shift::Lsl && l.amount == 0;
    if l.op == LogicOp::BitOr && !l.flags && l.rn == 31 {
        if !l.invert && plain_shift {
            f.write_str("mov ")?;
            gpr(f, l.width, l.rd)?;
            f.write_str(", ")?;
            return gpr(f, l.width, l.rm);
        }
        if l.invert {
            f.write_str("mvn ")?;
            gpr(f, l.width, l.rd)?;
            f.write_str(", ")?;
            gpr(f, l.width, l.rm)?;
            return shift_operand(f, l.shift, l.amount);
        }
    }
    if l.op == LogicOp::BitAnd && l.flags && !l.invert && l.rd == 31 {
        f.write_str("tst ")?;
    } else {
        f.write_str(match (l.op, l.invert, l.flags) {
            (LogicOp::BitAnd, false, false) => "and ",
            (LogicOp::BitAnd, false, true) => "ands ",
            (LogicOp::BitAnd, true, false) => "bic ",
            (LogicOp::BitAnd, true, true) => "bics ",
            (LogicOp::BitOr, false, _) => "orr ",
            (LogicOp::BitOr, true, _) => "orn ",
            (LogicOp::BitXor, false, _) => "eor ",
            (LogicOp::BitXor, true, _) => "eon ",
        })?;
        gpr(f, l.width, l.rd)?;
        f.write_str(", ")?;
    }
    gpr(f, l.width, l.rn)?;
    f.write_str(", ")?;
    gpr(f, l.width, l.rm)?;
    shift_operand(f, l.shift, l.amount)
}

/// Whether a single `movz` or `movn` can load `value` (already masked to `width`).
fn wide_constant(value: u64, width: Width) -> bool {
    let single_chunk =
        |v: u64| v != 0 && [0, 16, 32, 48].iter().any(|&s| v & !(0xffffu64 << s) == 0);
    single_chunk(value) || single_chunk(!value & mask(width))
}

fn logic_imm(f: &mut Formatter<'_>, l: LogicImm) -> fmt::Result {
    let value = l.immediate & mask(l.width);
    if l.op == LogicOp::BitOr
        && !l.flags
        && l.rn == 31
        && (l.rd == 31 || !wide_constant(value, l.width))
    {
        f.write_str("mov ")?;
        gpr_sp(f, l.width, l.rd)?;
        return write!(f, ", #{value:#x}");
    }
    if l.op == LogicOp::BitAnd && l.flags && l.rd == 31 {
        f.write_str("tst ")?;
    } else {
        f.write_str(match (l.op, l.flags) {
            (LogicOp::BitAnd, false) => "and ",
            (LogicOp::BitAnd, true) => "ands ",
            (LogicOp::BitOr, _) => "orr ",
            (LogicOp::BitXor, _) => "eor ",
        })?;
        if l.flags {
            gpr(f, l.width, l.rd)?;
        } else {
            gpr_sp(f, l.width, l.rd)?;
        }
        f.write_str(", ")?;
    }
    gpr(f, l.width, l.rn)?;
    write!(f, ", #{value:#x}")
}

fn mul(f: &mut Formatter<'_>, m: Multiply) -> fmt::Result {
    let add = m.op == ArithOp::Add;
    let alias = m.ra == 31;
    f.write_str(match (add, alias) {
        (true, true) => "mul ",
        (false, true) => "mneg ",
        (true, false) => "madd ",
        (false, false) => "msub ",
    })?;
    gpr(f, m.width, m.rd)?;
    f.write_str(", ")?;
    gpr(f, m.width, m.rn)?;
    f.write_str(", ")?;
    gpr(f, m.width, m.rm)?;
    if !alias {
        f.write_str(", ")?;
        gpr(f, m.width, m.ra)?;
    }
    Ok(())
}

fn mul_long(f: &mut Formatter<'_>, m: MultiplyLong) -> fmt::Result {
    let add = m.op == ArithOp::Add;
    let alias = m.ra == 31;
    f.write_str(match (m.signed, add, alias) {
        (true, true, true) => "smull ",
        (true, false, true) => "smnegl ",
        (true, true, false) => "smaddl ",
        (true, false, false) => "smsubl ",
        (false, true, true) => "umull ",
        (false, false, true) => "umnegl ",
        (false, true, false) => "umaddl ",
        (false, false, false) => "umsubl ",
    })?;
    gpr(f, Width::X64, m.rd)?;
    f.write_str(", ")?;
    gpr(f, Width::W32, m.rn)?;
    f.write_str(", ")?;
    gpr(f, Width::W32, m.rm)?;
    if !alias {
        f.write_str(", ")?;
        gpr(f, Width::X64, m.ra)?;
    }
    Ok(())
}

fn mul_high(f: &mut Formatter<'_>, m: MultiplyHigh) -> fmt::Result {
    f.write_str(if m.signed { "smulh " } else { "umulh " })?;
    gpr(f, Width::X64, m.rd)?;
    f.write_str(", ")?;
    gpr(f, Width::X64, m.rn)?;
    f.write_str(", ")?;
    gpr(f, Width::X64, m.rm)
}

fn adr(f: &mut Formatter<'_>, cx: &Context<'_>, a: Address) -> fmt::Result {
    f.write_str(if a.page { "adrp " } else { "adr " })?;
    gpr(f, Width::X64, a.rd)?;
    f.write_str(", ")?;
    if a.page {
        cx.page_target(f, a.offset)
    } else {
        cx.target(f, a.offset)
    }
}

fn csel(f: &mut Formatter<'_>, s: Select) -> fmt::Result {
    let real = !matches!(s.cond, Condition::Al | Condition::Nv);
    let same = s.rn == s.rm;
    // opc: 0 csel, 1 csinc, 2 csinv, 3 csneg.
    if real && same {
        let alias = match s.opc {
            1 if s.rn == 31 => Some(("cset ", false)),
            1 => Some(("cinc ", true)),
            2 if s.rn == 31 => Some(("csetm ", false)),
            2 => Some(("cinv ", true)),
            3 => Some(("cneg ", true)),
            _ => None,
        };
        if let Some((name, with_rn)) = alias {
            f.write_str(name)?;
            gpr(f, s.width, s.rd)?;
            f.write_str(", ")?;
            if with_rn {
                gpr(f, s.width, s.rn)?;
                f.write_str(", ")?;
            }
            return f.write_str(condition(invert(s.cond)));
        }
    }
    f.write_str(match s.opc {
        0 => "csel ",
        1 => "csinc ",
        2 => "csinv ",
        _ => "csneg ",
    })?;
    gpr(f, s.width, s.rd)?;
    f.write_str(", ")?;
    gpr(f, s.width, s.rn)?;
    f.write_str(", ")?;
    gpr(f, s.width, s.rm)?;
    write!(f, ", {}", condition(s.cond))
}

fn variable(f: &mut Formatter<'_>, v: Variable) -> fmt::Result {
    f.write_str(match v.op {
        VariableOp::Udiv => "udiv ",
        VariableOp::Sdiv => "sdiv ",
        VariableOp::Lsl => "lsl ",
        VariableOp::Lsr => "lsr ",
        VariableOp::Asr => "asr ",
        VariableOp::Ror => "ror ",
    })?;
    gpr(f, v.width, v.rd)?;
    f.write_str(", ")?;
    gpr(f, v.width, v.rn)?;
    f.write_str(", ")?;
    gpr(f, v.width, v.rm)
}

fn cond_compare(f: &mut Formatter<'_>, c: CondCompare) -> fmt::Result {
    f.write_str(if c.op == ArithOp::Add {
        "ccmn "
    } else {
        "ccmp "
    })?;
    gpr(f, c.width, c.rn)?;
    f.write_str(", ")?;
    match c.operand {
        CompareOperand::Register(rm) => gpr(f, c.width, rm)?,
        CompareOperand::Immediate(imm) => write!(f, "#{imm:#x}")?,
    }
    write!(f, ", #{:#x}, {}", c.nzcv, condition(c.cond))
}

fn arith_carry(f: &mut Formatter<'_>, a: AddCarry) -> fmt::Result {
    if a.op == ArithOp::Sub && a.rn == 31 {
        f.write_str(if a.flags { "ngcs " } else { "ngc " })?;
        gpr(f, a.width, a.rd)?;
        f.write_str(", ")?;
        return gpr(f, a.width, a.rm);
    }
    f.write_str(match (a.op, a.flags) {
        (ArithOp::Add, false) => "adc ",
        (ArithOp::Add, true) => "adcs ",
        (ArithOp::Sub, false) => "sbc ",
        (ArithOp::Sub, true) => "sbcs ",
    })?;
    gpr(f, a.width, a.rd)?;
    f.write_str(", ")?;
    gpr(f, a.width, a.rn)?;
    f.write_str(", ")?;
    gpr(f, a.width, a.rm)
}

fn unary(f: &mut Formatter<'_>, u: Unary) -> fmt::Result {
    f.write_str(match u.op {
        UnaryOp::Rbit => "rbit ",
        UnaryOp::Rev16 => "rev16 ",
        UnaryOp::Rev32 => "rev32 ",
        UnaryOp::Rev => "rev ",
        UnaryOp::Clz => "clz ",
    })?;
    gpr(f, u.width, u.rd)?;
    f.write_str(", ")?;
    gpr(f, u.width, u.rn)
}

/// `name rd, rn, #a` or `name rd, rn, #a, #b`.
fn bitfield_alias(
    f: &mut Formatter<'_>,
    name: &str,
    b: Bitfield,
    first: u16,
    second: Option<u16>,
) -> fmt::Result {
    write!(f, "{name} ")?;
    gpr(f, b.width, b.rd)?;
    f.write_str(", ")?;
    gpr(f, b.width, b.rn)?;
    write!(f, ", #{first}")?;
    if let Some(second) = second {
        write!(f, ", #{second}")?;
    }
    Ok(())
}

fn bitfield(f: &mut Formatter<'_>, b: Bitfield) -> fmt::Result {
    let lsb = u16::from(b.lsb);
    let len = u16::from(b.len);
    let top = lsb.wrapping_add(len) == bits(b.width);
    match b.op {
        BitfieldOp::Signed => {
            if b.extract && lsb == 0 {
                let ext = match (len, b.width) {
                    (8, _) => Some("sxtb "),
                    (16, _) => Some("sxth "),
                    (32, Width::X64) => Some("sxtw "),
                    _ => None,
                };
                if let Some(name) = ext {
                    f.write_str(name)?;
                    gpr(f, b.width, b.rd)?;
                    f.write_str(", ")?;
                    return gpr(f, Width::W32, b.rn);
                }
            }
            if b.extract && top {
                bitfield_alias(f, "asr", b, lsb, None)
            } else if b.extract {
                bitfield_alias(f, "sbfx", b, lsb, Some(len))
            } else {
                bitfield_alias(f, "sbfiz", b, lsb, Some(len))
            }
        }
        BitfieldOp::Unsigned => {
            if b.extract && lsb == 0 && b.width == Width::W32 && (len == 8 || len == 16) {
                f.write_str(if len == 8 { "uxtb " } else { "uxth " })?;
                gpr(f, b.width, b.rd)?;
                f.write_str(", ")?;
                return gpr(f, b.width, b.rn);
            }
            if !b.extract && top {
                bitfield_alias(f, "lsl", b, lsb, None)
            } else if b.extract && top {
                bitfield_alias(f, "lsr", b, lsb, None)
            } else if b.extract {
                bitfield_alias(f, "ubfx", b, lsb, Some(len))
            } else {
                bitfield_alias(f, "ubfiz", b, lsb, Some(len))
            }
        }
        BitfieldOp::Insert => {
            if !b.extract && b.rn == 31 {
                write!(f, "bfc ")?;
                gpr(f, b.width, b.rd)?;
                write!(f, ", #{lsb}, #{len}")
            } else if b.extract {
                bitfield_alias(f, "bfxil", b, lsb, Some(len))
            } else {
                bitfield_alias(f, "bfi", b, lsb, Some(len))
            }
        }
    }
}

fn extract(f: &mut Formatter<'_>, e: Extract) -> fmt::Result {
    let ror = e.rn == e.rm;
    f.write_str(if ror { "ror " } else { "extr " })?;
    gpr(f, e.width, e.rd)?;
    f.write_str(", ")?;
    gpr(f, e.width, e.rn)?;
    if !ror {
        f.write_str(", ")?;
        gpr(f, e.width, e.rm)?;
    }
    write!(f, ", #{}", e.lsb)
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::decode::decode;
    use core::fmt::Write as _;
    use std::string::String;

    struct At<'a>(&'a Instruction, Option<u64>);

    impl core::fmt::Display for At<'_> {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            let cx = Context {
                pc: self.1,
                symbols: &(),
            };
            write(f, &cx, self.0)
        }
    }

    fn text(word: u32, pc: Option<u64>) -> String {
        let instruction = decode(word).expect("decodes");
        let mut out = String::new();
        write!(out, "{}", At(&instruction, pc)).unwrap();
        out
    }

    /// Words and their GNU objdump rendering.
    const CASES: &[(u32, &str)] = &[
        (0xd2800000, "mov x0, #0x0"),
        (0xd2a00000, "movz x0, #0x0, lsl #16"),
        (0x52a00000, "movz w0, #0x0, lsl #16"),
        (0xd2e000a0, "mov x0, #0x5000000000000"),
        (0x92800000, "mov x0, #0xffffffffffffffff"),
        (0x12800000, "mov w0, #0xffffffff"),
        (0x92a00000, "movn x0, #0x0, lsl #16"),
        (0x12a00000, "movn w0, #0x0, lsl #16"),
        (0x129fffe0, "movn w0, #0xffff"),
        (0x12bfffe0, "movn w0, #0xffff, lsl #16"),
        (0x929fffe0, "mov x0, #0xffffffffffff0000"),
        (0x12a000a0, "mov w0, #0xfffaffff"),
        (0xf28000a0, "movk x0, #0x5"),
        (0x72a000a0, "movk w0, #0x5, lsl #16"),
        (0xf2ffffe0, "movk x0, #0xffff, lsl #48"),
        (0x91400420, "add x0, x1, #0x1, lsl #12"),
        (0x917ffc20, "add x0, x1, #0xfff, lsl #12"),
        (0x91400020, "add x0, x1, #0x0, lsl #12"),
        (0x9140003f, "add sp, x1, #0x0, lsl #12"),
        (0x914003e0, "add x0, sp, #0x0, lsl #12"),
        (0xd140003f, "sub sp, x1, #0x0, lsl #12"),
        (0xf140003f, "cmp x1, #0x0, lsl #12"),
        (0xb140003f, "cmn x1, #0x0, lsl #12"),
        (0x11400c20, "add w0, w1, #0x3, lsl #12"),
        (0x7140001f, "cmp w0, #0x0, lsl #12"),
        (0x9100003f, "mov sp, x1"),
        (0x910003e0, "mov x0, sp"),
        (0x91000020, "add x0, x1, #0x0"),
        (0xb10003e0, "adds x0, sp, #0x0"),
        (0xb1000c3f, "cmn x1, #0x3"),
        (0x71000fff, "cmp wsp, #0x3"),
        (0xf1000fe0, "subs x0, sp, #0x3"),
        (0xd10043ff, "sub sp, sp, #0x10"),
        (0xcb0103e0, "neg x0, x1"),
        (0xeb010fe0, "negs x0, x1, lsl #3"),
        (0xeb0103ff, "cmp xzr, x1"),
        (0xab82103f, "cmn x1, x2, asr #4"),
        (0xab0203ff, "cmn xzr, x2"),
        (0x4b810fe0, "neg w0, w1, asr #3"),
        (0x8b020020, "add x0, x1, x2"),
        (0x8b224020, "add x0, x1, w2, uxtw"),
        (0x8b224820, "add x0, x1, w2, uxtw #2"),
        (0x8b226020, "add x0, x1, x2, uxtx"),
        (0x8b226c20, "add x0, x1, x2, uxtx #3"),
        (0x8b22e420, "add x0, x1, x2, sxtx #1"),
        (0x8b22d020, "add x0, x1, w2, sxtw #4"),
        (0x8b2263ff, "add sp, sp, x2"),
        (0x8b226bff, "add sp, sp, x2, lsl #2"),
        (0x8b226c3f, "add sp, x1, x2, lsl #3"),
        (0x8b226fe0, "add x0, sp, x2, lsl #3"),
        (0x0b2247e0, "add w0, wsp, w2, lsl #1"),
        (0x0b2243e0, "add w0, wsp, w2"),
        (0x0b226020, "add w0, w1, w2, uxtx"),
        (0x0b224020, "add w0, w1, w2, uxtw"),
        (0x0b228820, "add w0, w1, w2, sxtb #2"),
        (0xab2267e0, "adds x0, sp, x2, lsl #1"),
        (0xab2263e0, "adds x0, sp, x2"),
        (0xab2267ff, "cmn sp, x2, lsl #1"),
        (0xab22643f, "cmn x1, x2, uxtx #1"),
        (0xeb2203ff, "cmp sp, w2, uxtb"),
        (0x6b22a7ff, "cmp wsp, w2, sxth #1"),
        (0xab22c03f, "cmn x1, w2, sxtw"),
        (0xcb22703f, "sub sp, x1, x2, lsl #4"),
        (0xcb228020, "sub x0, x1, w2, sxtb"),
        (0x129fffc0, "mov w0, #0xffff0001"),
        (0x12bfffc0, "mov w0, #0x1ffff"),
        (0x92bfffe0, "mov x0, #0xffffffff0000ffff"),
        (0x92dfffe0, "mov x0, #0xffff0000ffffffff"),
        (0x92ffffe0, "mov x0, #0xffffffffffff"),
        (0x92e00020, "mov x0, #0xfffeffffffffffff"),
        (0x52bfffe0, "mov w0, #0xffff0000"),
        (0x529fffe0, "mov w0, #0xffff"),
        (0xd2ffffe0, "mov x0, #0xffff000000000000"),
        (0x8a020020, "and x0, x1, x2"),
        (0x8a020c20, "and x0, x1, x2, lsl #3"),
        (0x8ac20020, "and x0, x1, x2, ror #0"),
        (0x8ac20c20, "and x0, x1, x2, ror #3"),
        (0xaa0203e0, "mov x0, x2"),
        (0xaa020fe0, "orr x0, xzr, x2, lsl #3"),
        (0xaa1f0020, "orr x0, x1, xzr"),
        (0x2a0203e0, "mov w0, w2"),
        (0xaa1f03e0, "mov x0, xzr"),
        (0xaa2203e0, "mvn x0, x2"),
        (0xaa6213e0, "mvn x0, x2, lsr #4"),
        (0xaa220020, "orn x0, x1, x2"),
        (0xca2203e0, "eon x0, xzr, x2"),
        (0xca0203e0, "eor x0, xzr, x2"),
        (0x8a220020, "bic x0, x1, x2"),
        (0xea220020, "bics x0, x1, x2"),
        (0xea22003f, "bics xzr, x1, x2"),
        (0xea02003f, "tst x1, x2"),
        (0xea02143f, "tst x1, x2, lsl #5"),
        (0xea0203ff, "tst xzr, x2"),
        (0xea020020, "ands x0, x1, x2"),
        (0xaae203e0, "mvn x0, x2, ror #0"),
        (0xaac203e0, "orr x0, xzr, x2, ror #0"),
        (0xaa8207e0, "orr x0, xzr, x2, asr #1"),
        (0x92401c20, "and x0, x1, #0xff"),
        (0x92401c3f, "and sp, x1, #0xff"),
        (0xb2401fe0, "orr x0, xzr, #0xff"),
        (0xb200f3e0, "mov x0, #0x5555555555555555"),
        (0x32089fe0, "mov w0, #0xff00ff00"),
        (0x320003e0, "orr w0, wzr, #0x1"),
        (0x320003ff, "mov wsp, #0x1"),
        (0xb24003ff, "mov sp, #0x1"),
        (0xb278dfe0, "orr x0, xzr, #0xffffffffffffff00"),
        (0xb2410020, "orr x0, x1, #0x8000000000000000"),
        (0x321f7820, "orr w0, w1, #0xfffffffe"),
        (0xf27c0c3f, "tst x1, #0xf0"),
        (0xf27c0c20, "ands x0, x1, #0xf0"),
        (0x721c0c3f, "tst w1, #0xf0"),
        (0xd2401fe0, "eor x0, xzr, #0xff"),
        (0x52007820, "eor w0, w1, #0x7fffffff"),
        (0xb27f7be0, "mov x0, #0xfffffffe"),
        (0xb240fbe0, "orr x0, xzr, #0x7fffffffffffffff"),
        (0x321f7be0, "orr w0, wzr, #0xfffffffe"),
        (0x9b027c20, "mul x0, x1, x2"),
        (0x9b020c20, "madd x0, x1, x2, x3"),
        (0x9b02fc20, "mneg x0, x1, x2"),
        (0x9b028c20, "msub x0, x1, x2, x3"),
        (0x1b02fc20, "mneg w0, w1, w2"),
        (0x1b027c20, "mul w0, w1, w2"),
        (0x9b220c20, "smaddl x0, w1, w2, x3"),
        (0x9b227c20, "smull x0, w1, w2"),
        (0x9b228c20, "smsubl x0, w1, w2, x3"),
        (0x9b22fc20, "smnegl x0, w1, w2"),
        (0x9ba20c20, "umaddl x0, w1, w2, x3"),
        (0x9ba27c20, "umull x0, w1, w2"),
        (0x9ba28c20, "umsubl x0, w1, w2, x3"),
        (0x9ba2fc20, "umnegl x0, w1, w2"),
        (0x9b427c20, "smulh x0, x1, x2"),
        (0x9bc27c20, "umulh x0, x1, x2"),
        (0x9a820020, "csel x0, x1, x2, eq"),
        (0x9a82e020, "csel x0, x1, x2, al"),
        (0x9a82f020, "csel x0, x1, x2, nv"),
        (0x9a821420, "csinc x0, x1, x2, ne"),
        (0x9a811420, "cinc x0, x1, eq"),
        (0x9a81e420, "csinc x0, x1, x1, al"),
        (0x9a81f420, "csinc x0, x1, x1, nv"),
        (0x9a9f17e0, "cset x0, eq"),
        (0x9a9fe7e0, "csinc x0, xzr, xzr, al"),
        (0x9a9ff7e0, "csinc x0, xzr, xzr, nv"),
        (0x9a8117e0, "csinc x0, xzr, x1, ne"),
        (0x9a9f1420, "csinc x0, x1, xzr, ne"),
        (0xda9f13e0, "csetm x0, eq"),
        (0xda9fe3e0, "csinv x0, xzr, xzr, al"),
        (0xda811020, "cinv x0, x1, eq"),
        (0xda81f020, "csinv x0, x1, x1, nv"),
        (0xda8113e0, "csinv x0, xzr, x1, ne"),
        (0xda811420, "cneg x0, x1, eq"),
        (0xda81e420, "csneg x0, x1, x1, al"),
        (0xda9f17e0, "cneg x0, xzr, eq"),
        (0xda9fe7e0, "csneg x0, xzr, xzr, al"),
        (0xda821420, "csneg x0, x1, x2, ne"),
        (0x1a9f27e0, "cset w0, cc"),
        (0x1a9f87e0, "cset w0, ls"),
        (0x1a9f47e0, "cset w0, pl"),
        (0x1a9f77e0, "cset w0, vs"),
        (0x1a9fa7e0, "cset w0, lt"),
        (0x1a9fd7e0, "cset w0, gt"),
        (0x9ac20820, "udiv x0, x1, x2"),
        (0x1ac20c20, "sdiv w0, w1, w2"),
        (0x9ac22020, "lsl x0, x1, x2"),
        (0x1ac22420, "lsr w0, w1, w2"),
        (0x9ac22820, "asr x0, x1, x2"),
        (0x1ac22c20, "ror w0, w1, w2"),
        (0xfa410005, "ccmp x0, x1, #0x5, eq"),
        (0xfa5f180f, "ccmp x0, #0x1f, #0xf, ne"),
        (0x3a41e000, "ccmn w0, w1, #0x0, al"),
        (0x3a40f800, "ccmn w0, #0x0, #0x0, nv"),
        (0xfa5f23e1, "ccmp xzr, xzr, #0x1, cs"),
        (0x9a020020, "adc x0, x1, x2"),
        (0x3a020020, "adcs w0, w1, w2"),
        (0xda020020, "sbc x0, x1, x2"),
        (0xda0203e0, "ngc x0, x2"),
        (0xfa0203e0, "ngcs x0, x2"),
        (0x7a0203e0, "ngcs w0, w2"),
        (0x5a1f03e0, "ngc w0, wzr"),
        (0x9a1f03e0, "adc x0, xzr, xzr"),
        (0xdac00020, "rbit x0, x1"),
        (0x5ac00020, "rbit w0, w1"),
        (0xdac00420, "rev16 x0, x1"),
        (0x5ac00420, "rev16 w0, w1"),
        (0xdac00820, "rev32 x0, x1"),
        (0x5ac00820, "rev w0, w1"),
        (0xdac00c20, "rev x0, x1"),
        (0xdac01020, "clz x0, x1"),
        (0x5ac013e0, "clz w0, wzr"),
        (0x93401c20, "sxtb x0, w1"),
        (0x93403c20, "sxth x0, w1"),
        (0x93407c20, "sxtw x0, w1"),
        (0x13001c20, "sxtb w0, w1"),
        (0x13003c20, "sxth w0, w1"),
        (0x13007c20, "asr w0, w1, #0"),
        (0x9340fc20, "asr x0, x1, #0"),
        (0x9345fc20, "asr x0, x1, #5"),
        (0x13057c20, "asr w0, w1, #5"),
        (0x93452420, "sbfx x0, x1, #5, #5"),
        (0x93491420, "sbfiz x0, x1, #55, #6"),
        (0x937ff820, "sbfiz x0, x1, #1, #63"),
        (0x131f7820, "sbfiz w0, w1, #1, #31"),
        (0x93410020, "sbfiz x0, x1, #63, #1"),
        (0x93400020, "sbfx x0, x1, #0, #1"),
        (0x93402020, "sbfx x0, x1, #0, #9"),
        (0x93430c20, "sbfx x0, x1, #3, #1"),
        (0xd3401c20, "ubfx x0, x1, #0, #8"),
        (0x53001c20, "uxtb w0, w1"),
        (0x53003c20, "uxth w0, w1"),
        (0xd3403c20, "ubfx x0, x1, #0, #16"),
        (0xd3407c20, "ubfx x0, x1, #0, #32"),
        (0xd340fc20, "lsr x0, x1, #0"),
        (0x53007c20, "lsr w0, w1, #0"),
        (0xd345fc20, "lsr x0, x1, #5"),
        (0x53057c20, "lsr w0, w1, #5"),
        (0xd3452420, "ubfx x0, x1, #5, #5"),
        (0xd3491420, "ubfiz x0, x1, #55, #6"),
        (0xd37ff820, "lsl x0, x1, #1"),
        (0x531f7820, "lsl w0, w1, #1"),
        (0xd3410020, "lsl x0, x1, #63"),
        (0xd3400020, "ubfx x0, x1, #0, #1"),
        (0x53010020, "lsl w0, w1, #31"),
        (0xd3430c20, "ubfx x0, x1, #3, #1"),
        (0xd340f820, "ubfx x0, x1, #0, #63"),
        (0xd37df020, "lsl x0, x1, #3"),
        (0xb3452420, "bfxil x0, x1, #5, #5"),
        (0xb3491420, "bfi x0, x1, #55, #6"),
        (0xb34917e0, "bfc x0, #55, #6"),
        (0xb34527e0, "bfxil x0, xzr, #5, #5"),
        (0xb34027e0, "bfxil x0, xzr, #0, #10"),
        (0xb3400020, "bfxil x0, x1, #0, #1"),
        (0xb37ff820, "bfi x0, x1, #1, #63"),
        (0x330103e0, "bfc w0, #31, #1"),
        (0x33007c20, "bfxil w0, w1, #0, #32"),
        (0xb340fc20, "bfxil x0, x1, #0, #64"),
        (0xb340ffe0, "bfxil x0, xzr, #0, #64"),
        (0x93c21420, "extr x0, x1, x2, #5"),
        (0x93c11420, "ror x0, x1, #5"),
        (0x13810020, "ror w0, w1, #0"),
        (0x93c20020, "extr x0, x1, x2, #0"),
        (0x13827c20, "extr w0, w1, w2, #31"),
        (0x93dfffe0, "ror x0, xzr, #63"),
    ];

    #[test]
    fn matches_objdump() {
        for &(word, expected) in CASES {
            assert_eq!(text(word, None), expected, "{word:#010x}");
        }
    }

    #[test]
    fn pc_relative() {
        assert_eq!(text(0x1000_0020, None), "adr x0, .+4");
        assert_eq!(text(0x10ff_ffe0, None), "adr x0, .-4");
        assert_eq!(text(0xb000_0000, None), "adrp x0, .+4096");
        assert_eq!(text(0x90ff_ffe0, None), "adrp x0, .-16384");
        assert_eq!(text(0x1000_0020, Some(0x1000)), "adr x0, 0x1004");
        assert_eq!(text(0xb000_0000, Some(0x1234)), "adrp x0, 0x2000");
        assert_eq!(text(0xb000_001f, Some(0x1234)), "adrp xzr, 0x2000");
    }
}
