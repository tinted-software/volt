//! Scalar floating point and the general-register / SIMD&FP register `fmov`.
//!
//! Owns `Fp` and `SimdFmov`.

use super::Context;
use super::reg::{arrangement, fp_scalar, gpr};
use crate::decode::{Instruction, SimdFmov, Width};
use crate::fp::{Binary, Fused, Op, Precision, Unary};
use core::fmt::{self, Formatter};

pub(super) fn write(
    f: &mut Formatter<'_>,
    cx: &Context<'_>,
    instruction: &Instruction,
) -> fmt::Result {
    let _ = cx;
    match instruction {
        Instruction::Fp(op) => write_op(f, *op),
        Instruction::SimdFmov(op) => write_fmov(f, *op),
        other => unreachable!("{other:?} is not a floating-point instruction"),
    }
}

/// Size in bytes of the scalar register for a precision.
fn size(p: Precision) -> u8 {
    match p {
        Precision::Single => 4,
        Precision::Double => 8,
    }
}

fn width(wide: bool) -> Width {
    if wide { Width::X64 } else { Width::W32 }
}

/// The condition field as binutils spells it; the field is four bits.
fn condition(f: &mut Formatter<'_>, cond: u32) -> fmt::Result {
    const NAMES: [&str; 16] = [
        "eq", "ne", "cs", "cc", "mi", "pl", "vs", "vc", "hi", "ls", "ge", "lt", "gt", "le", "al",
        "nv",
    ];
    f.write_str(NAMES[(cond & 15) as usize])
}

fn write_fmov(f: &mut Formatter<'_>, op: SimdFmov) -> fmt::Result {
    f.write_str("fmov ")?;
    // The upper doubleword is named as the lane `Vn.D[1]`.
    let vector = |f: &mut Formatter<'_>, r: u8| -> fmt::Result {
        if op.upper {
            write!(f, "v{r}.d[1]")
        } else {
            fp_scalar(f, if op.double { 8 } else { 4 }, r)
        }
    };
    if op.to_fp {
        vector(f, op.rd)?;
        f.write_str(", ")?;
        gpr(f, width(op.double), op.rn)
    } else {
        gpr(f, width(op.double), op.rd)?;
        f.write_str(", ")?;
        vector(f, op.rn)
    }
}

/// `mnemonic rd, rn` over scalar registers of precisions `pd` and `pn`.
fn two(
    f: &mut Formatter<'_>,
    mnemonic: &str,
    (pd, rd): (Precision, u8),
    (pn, rn): (Precision, u8),
) -> fmt::Result {
    write!(f, "{mnemonic} ")?;
    fp_scalar(f, size(pd), rd)?;
    f.write_str(", ")?;
    fp_scalar(f, size(pn), rn)
}

fn regs(f: &mut Formatter<'_>, p: Precision, list: &[u8]) -> fmt::Result {
    for (i, &r) in list.iter().enumerate() {
        if i != 0 {
            f.write_str(", ")?;
        }
        fp_scalar(f, size(p), r)?;
    }
    Ok(())
}

fn write_op(f: &mut Formatter<'_>, op: Op) -> fmt::Result {
    match op {
        Op::Unary { kind, p, rd, rn } => {
            let mnemonic = match kind {
                Unary::Move => "fmov",
                Unary::Abs => "fabs",
                Unary::Neg => "fneg",
                Unary::Sqrt => "fsqrt",
            };
            two(f, mnemonic, (p, rd), (p, rn))
        }
        Op::Convert { p, rd, rn } => {
            let to = match p {
                Precision::Single => Precision::Double,
                Precision::Double => Precision::Single,
            };
            two(f, "fcvt", (to, rd), (p, rn))
        }
        Op::Binary {
            kind,
            p,
            rd,
            rn,
            rm,
        } => {
            let mnemonic = match kind {
                Binary::Mul => "fmul",
                Binary::Div => "fdiv",
                Binary::Add => "fadd",
                Binary::Sub => "fsub",
                Binary::Max => "fmax",
                Binary::Min => "fmin",
                Binary::MaxNum => "fmaxnm",
                Binary::MinNum => "fminnm",
                Binary::Nmul => "fnmul",
            };
            write!(f, "{mnemonic} ")?;
            regs(f, p, &[rd, rn, rm])
        }
        Op::Fused {
            kind,
            p,
            rd,
            rn,
            rm,
            ra,
        } => {
            let mnemonic = match kind {
                Fused::Madd => "fmadd",
                Fused::Msub => "fmsub",
                Fused::Nmadd => "fnmadd",
                Fused::Nmsub => "fnmsub",
            };
            write!(f, "{mnemonic} ")?;
            regs(f, p, &[rd, rn, rm, ra])
        }
        Op::Compare {
            p,
            rn,
            rm,
            signaling,
        } => {
            f.write_str(if signaling { "fcmpe " } else { "fcmp " })?;
            fp_scalar(f, size(p), rn)?;
            f.write_str(", ")?;
            match rm {
                Some(rm) => fp_scalar(f, size(p), rm),
                None => f.write_str("#0.0"),
            }
        }
        Op::CondCompare {
            p,
            rn,
            rm,
            cond,
            nzcv,
        } => {
            f.write_str("fccmp ")?;
            regs(f, p, &[rn, rm])?;
            write!(f, ", #{:#x}, ", nzcv & 15)?;
            condition(f, cond)
        }
        Op::Select {
            p,
            rd,
            rn,
            rm,
            cond,
        } => {
            f.write_str("fcsel ")?;
            regs(f, p, &[rd, rn, rm])?;
            f.write_str(", ")?;
            condition(f, cond)
        }
        Op::Immediate { p, rd, imm8 } => {
            f.write_str("fmov ")?;
            fp_scalar(f, size(p), rd)?;
            f.write_str(", #")?;
            expanded_immediate(f, imm8 as u8)
        }
        Op::IntToFloat {
            signed,
            p,
            wide,
            rd,
            rn,
        } => {
            f.write_str(if signed { "scvtf " } else { "ucvtf " })?;
            fp_scalar(f, size(p), rd)?;
            f.write_str(", ")?;
            gpr(f, width(wide), rn)
        }
        Op::FloatToInt {
            signed,
            p,
            wide,
            rd,
            rn,
        } => {
            f.write_str(if signed { "fcvtzs " } else { "fcvtzu " })?;
            gpr(f, width(wide), rd)?;
            f.write_str(", ")?;
            fp_scalar(f, size(p), rn)
        }
        Op::ScalarIntToFloat { signed, p, rd, rn } => {
            two(f, if signed { "scvtf" } else { "ucvtf" }, (p, rd), (p, rn))
        }
        Op::ScalarFloatToInt { signed, p, rd, rn } => two(
            f,
            if signed { "fcvtzs" } else { "fcvtzu" },
            (p, rd),
            (p, rn),
        ),
        Op::VectorSign {
            negate,
            double,
            q,
            rd,
            rn,
        } => {
            f.write_str(if negate { "fneg " } else { "fabs " })?;
            let esize = if double { 3 } else { 2 };
            write!(f, "v{rd}.")?;
            arrangement(f, esize, q)?;
            write!(f, ", v{rn}.")?;
            arrangement(f, esize, q)
        }
    }
}

/// Print the VFPExpandImm value of `imm8` the way binutils does: `%.18e`.
///
/// With `imm8 = a:bcd:efgh` the value is `±(16 + efgh) / 16 * 2^exp` where `exp` is
/// the signed three-bit `NOT(b):c:d` plus one (`-3..=4`). That is `n / 2^j` with
/// `n` in `16..=31` and `j` in `0..=7`, so `n * 5^j / 10^j` is an exact decimal
/// integer of at most seven digits.
fn expanded_immediate(f: &mut Formatter<'_>, imm8: u8) -> fmt::Result {
    let negative = imm8 & 0x80 != 0;
    let e = i32::from((imm8 >> 4) & 7);
    let exp = if e < 4 { e + 1 } else { e - 7 };
    let n = 16 + u32::from(imm8 & 15);
    let j = (4 - exp) as u32;
    // `j` is `0..=7`; `n * 5^j` is at most 31 * 78125.
    let digits_value = n * 5u32.pow(j);
    let mut digits = [0u8; 10];
    let mut count = 0;
    let mut rest = digits_value;
    while rest != 0 {
        digits[count] = (rest % 10) as u8;
        rest /= 10;
        count += 1;
    }
    // `digits[count - 1]` is the most significant digit.
    let decimal_exponent = count as i32 - 1 - j as i32;
    if negative {
        f.write_str("-")?;
    }
    write!(f, "{}.", digits[count - 1])?;
    for i in 1..19 {
        let d = if i < count { digits[count - 1 - i] } else { 0 };
        write!(f, "{d}")?;
    }
    let sign = if decimal_exponent < 0 { '-' } else { '+' };
    write!(f, "e{sign}{:02}", decimal_exponent.unsigned_abs())
}

#[cfg(test)]
mod tests {
    use crate::format::word;
    use core::fmt::Write;

    struct Buf {
        bytes: [u8; 96],
        len: usize,
    }

    impl Write for Buf {
        fn write_str(&mut self, s: &str) -> core::fmt::Result {
            self.bytes[self.len..self.len + s.len()].copy_from_slice(s.as_bytes());
            self.len += s.len();
            Ok(())
        }
    }

    fn check(w: u32, expected: &str) {
        let mut buf = Buf {
            bytes: [0; 96],
            len: 0,
        };
        write!(buf, "{}", word(w)).unwrap();
        assert_eq!(
            core::str::from_utf8(&buf.bytes[..buf.len]).unwrap(),
            expected,
            "{w:#010x}"
        );
    }

    #[test]
    fn scalar_floating_point() {
        check(0x1e602008, "fcmp d0, #0.0");
        check(0x1e222020, "fcmp s1, s2");
        check(0x1e222030, "fcmpe s1, s2");
        check(0x1e202038, "fcmpe s1, #0.0");
        check(0x1e7f2010, "fcmpe d0, d31");
        check(0x1e221425, "fccmp s1, s2, #0x5, ne");
        check(0x1e60e7ef, "fccmp d31, d0, #0xf, al");
        check(0x1e633c41, "fcsel d1, d2, d3, cc");
        check(0x1e22c020, "fcvt d0, s1");
        check(0x1e6243e0, "fcvt s0, d31");
        check(0x1e638841, "fnmul d1, d2, d3");
        check(0x1f239041, "fnmsub s1, s2, s3, s4");
        check(0x1e204041, "fmov s1, s2");
        check(0x1e21c041, "fsqrt s1, s2");
        check(0x1e636841, "fmaxnm d1, d2, d3");
    }

    #[test]
    fn fmov_to_and_from_the_upper_doubleword() {
        check(0x9eaf0060, "fmov v0.d[1], x3");
        check(0x9eae0003, "fmov x3, v0.d[1]");
        check(0x9e670042, "fmov d2, x2");
    }

    #[test]
    fn immediates() {
        check(0x1e78101f, "fmov d31, #-1.250000000000000000e-01");
        check(0x1e27f000, "fmov s0, #3.100000000000000000e+01");
        check(0x1e281000, "fmov s0, #1.250000000000000000e-01");
    }

    #[test]
    fn conversions() {
        check(0x1e620041, "scvtf d1, w2");
        check(0x9e230041, "ucvtf s1, x2");
        check(0x1e780020, "fcvtzs w0, d1");
        check(0x9e390020, "fcvtzu x0, s1");
        check(0x5e21d841, "scvtf s1, s2");
        check(0x7e61d841, "ucvtf d1, d2");
        check(0x5ee1b841, "fcvtzs d1, d2");
        check(0x7ea1b841, "fcvtzu s1, s2");
        check(0x0ea0f841, "fabs v1.2s, v2.2s");
        check(0x6ea0f841, "fneg v1.4s, v2.4s");
        check(0x4ee0f841, "fabs v1.2d, v2.2d");
        check(0x9e670020, "fmov d0, x1");
        check(0x9e660020, "fmov x0, d1");
        check(0x1e270020, "fmov s0, w1");
        check(0x1e260020, "fmov w0, s1");
    }
}
