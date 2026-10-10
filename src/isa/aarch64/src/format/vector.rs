//! Advanced SIMD data processing: modified immediates, moves, `dup`/`ins`/`umov`,
//! `ext`, `tbl`/`tbx`, the integer three-same group, compares against zero, and the
//! integer instruction forms of `crate::simd`.
//!
//! Owns `SimdImm`, `SimdCompareZero`, `SimdDup`, `SimdDupElement`,
//! `SimdInsert`, `SimdInsertElement`, `SimdExtract`, `SimdExt`, `SimdAlu`, `SimdTable`
//! and `Simd`.

use super::Context;
use super::reg::{fp_scalar, gpr, vector};
use crate::decode::{
    ImmCombine, ImmForm, Instruction, SimdAlu, SimdAluOp, SimdImmediate, SimdTable, Width,
};
use crate::simd::{Long, Narrow, Op, Permute, Scalar, Zero};
use core::fmt::{self, Formatter, Write};

pub(super) fn write(
    f: &mut Formatter<'_>,
    cx: &Context<'_>,
    instruction: &Instruction,
) -> fmt::Result {
    let _ = cx;
    match *instruction {
        Instruction::SimdImm(imm) => immediate(f, imm),
        Instruction::SimdCompareZero(c) => {
            f.write_str("cmeq ")?;
            two_same(f, c.rd, c.rn, c.size, c.q)?;
            f.write_str(", #0")
        }
        Instruction::SimdDup { rd, rn, esize, q } => {
            f.write_str("dup ")?;
            vector(f, rd, esize & 3, q)?;
            f.write_str(", ")?;
            gpr(f, lane_width(esize), rn)
        }
        Instruction::SimdDupElement(d) => {
            f.write_str("dup ")?;
            vector(f, d.rd, d.esize & 3, d.q)?;
            f.write_str(", ")?;
            element(f, d.rn, d.esize, d.index)
        }
        Instruction::SimdInsert(i) => {
            f.write_str("mov ")?;
            element(f, i.rd, i.esize, i.index)?;
            f.write_str(", ")?;
            gpr(f, lane_width(i.esize), i.rn)
        }
        Instruction::SimdInsertElement(i) => {
            f.write_str("mov ")?;
            element(f, i.rd, i.esize, i.dst)?;
            f.write_str(", ")?;
            element(f, i.rn, i.esize, i.src)
        }
        // `umov` of a word or doubleword lane is spelled `mov`.
        Instruction::SimdExtract(e) => {
            f.write_str(if e.esize & 3 >= 2 { "mov " } else { "umov " })?;
            gpr(f, lane_width(e.esize), e.rd)?;
            f.write_str(", ")?;
            element(f, e.rn, e.esize, e.index)
        }
        Instruction::SimdExt(e) => {
            f.write_str("ext ")?;
            vector(f, e.rd, 0, e.q)?;
            f.write_str(", ")?;
            vector(f, e.rn, 0, e.q)?;
            f.write_str(", ")?;
            vector(f, e.rm, 0, e.q)?;
            write!(f, ", #{}", e.imm)
        }
        Instruction::SimdAlu(alu) => simd_alu(f, alu),
        Instruction::SimdTable(t) => table(f, t),
        Instruction::Simd(op) => simd(f, op),
        _ => unreachable!("{instruction:?} is not an Advanced SIMD vector instruction"),
    }
}

/// `X` for a doubleword lane, `W` for every narrower one.
fn lane_width(esize: u8) -> Width {
    if esize & 3 == 3 {
        Width::X64
    } else {
        Width::W32
    }
}

/// The element letter of `esize` (log2 of the byte width).
fn lane_letter(esize: u8) -> char {
    ['b', 'h', 's', 'd'][usize::from(esize & 3)]
}

/// One lane of a vector register: `v1.s[2]`.
fn element(f: &mut Formatter<'_>, r: u8, esize: u8, index: u8) -> fmt::Result {
    write!(f, "v{r}.{}[{index}]", lane_letter(esize))
}

/// `vd.T, vn.T` with one arrangement.
fn two_same(f: &mut Formatter<'_>, rd: u8, rn: u8, size: u8, q: bool) -> fmt::Result {
    vector(f, rd, size & 3, q)?;
    f.write_str(", ")?;
    vector(f, rn, size & 3, q)
}

/// `vd.T, vn.T, vm.T` with one arrangement.
fn three_same(f: &mut Formatter<'_>, rd: u8, rn: u8, rm: u8, size: u8, q: bool) -> fmt::Result {
    two_same(f, rd, rn, size, q)?;
    f.write_str(", ")?;
    vector(f, rm, size & 3, q)
}

fn simd_alu(f: &mut Formatter<'_>, alu: SimdAlu) -> fmt::Result {
    use SimdAluOp::*;
    let SimdAlu {
        op,
        rd,
        rn,
        rm,
        size,
        q,
    } = alu;
    let size = size & 3;
    let name = match op {
        Add => "add",
        Sub => "sub",
        Mul => "mul",
        Mla => "mla",
        Mls => "mls",
        And => "and",
        Bic => "bic",
        Orr if rn == rm => {
            f.write_str("mov ")?;
            return two_same(f, rd, rn, 0, q);
        }
        Orr => "orr",
        Orn => "orn",
        Eor => "eor",
        Bsl => "bsl",
        Bit => "bit",
        Bif => "bif",
        CmEq => "cmeq",
        CmGt => "cmgt",
        CmGe => "cmge",
        CmHi => "cmhi",
        CmHs => "cmhs",
        CmTst => "cmtst",
        UMin => "umin",
        UMax => "umax",
        SMin => "smin",
        SMax => "smax",
        UAbd => "uabd",
        SAbd => "sabd",
        SQAdd => "sqadd",
        UQAdd => "uqadd",
        SQSub => "sqsub",
        UQSub => "uqsub",
        SHAdd => "shadd",
        UHAdd => "uhadd",
        SHSub => "shsub",
        UHSub => "uhsub",
        SRHAdd => "srhadd",
        URHAdd => "urhadd",
        SSHl => "sshl",
        USHl => "ushl",
        SQShl => "sqshl",
        UQShl => "uqshl",
        SQRShl => "sqrshl",
        UQRShl => "uqrshl",
        AddV => "addv",
        UMinV => "uminv",
        UMaxV => "umaxv",
        UAddLV => "uaddlv",
        UMaxP => "umaxp",
        SMaxP => "smaxp",
        UMinP => "uminp",
        SMinP => "sminp",
        AddP => "addp",
    };
    write!(f, "{name} ")?;
    match op {
        // Across lanes: a scalar result, as wide as the lanes (twice as wide for
        // `uaddlv`), from a vector.
        AddV | UMinV | UMaxV | UAddLV => {
            let bytes = 1u8 << (size + u8::from(op == UAddLV));
            fp_scalar(f, bytes, rd)?;
            f.write_str(", ")?;
            vector(f, rn, size, q)
        }
        And | Bic | Orr | Orn | Eor | Bsl | Bit | Bif => three_same(f, rd, rn, rm, 0, q),
        _ => three_same(f, rd, rn, rm, size, q),
    }
}

fn table(f: &mut Formatter<'_>, t: SimdTable) -> fmt::Result {
    f.write_str(if t.extend { "tbx " } else { "tbl " })?;
    vector(f, t.rd, 0, t.q)?;
    f.write_str(", {")?;
    vector(f, t.rn, 0, true)?;
    if t.len & 3 != 0 {
        f.write_str("-")?;
        vector(f, t.rn.wrapping_add(t.len & 3) & 31, 0, true)?;
    }
    f.write_str("}, ")?;
    vector(f, t.rm, 0, t.q)
}

/// The mnemonic suffix of the upper-half (`2`) form of a long or narrow operation.
fn two(upper: bool) -> &'static str {
    if upper { "2" } else { "" }
}

fn simd(f: &mut Formatter<'_>, op: Op) -> fmt::Result {
    match op {
        Op::Permute {
            kind,
            size,
            q,
            rd,
            rn,
            rm,
        } => {
            f.write_str(match kind {
                Permute::Uzp1 => "uzp1 ",
                Permute::Uzp2 => "uzp2 ",
                Permute::Zip1 => "zip1 ",
                Permute::Zip2 => "zip2 ",
                Permute::Trn1 => "trn1 ",
                Permute::Trn2 => "trn2 ",
            })?;
            three_same(f, rd, rn, rm, size as u8, q)
        }
        Op::Reverse {
            group,
            size,
            q,
            rd,
            rn,
        } => {
            write!(f, "rev{group} ")?;
            two_same(f, rd, rn, size as u8, q)
        }
        Op::Count { q, rd, rn } => {
            f.write_str("cnt ")?;
            two_same(f, rd, rn, 0, q)
        }
        Op::Not { q, rd, rn } => {
            f.write_str("mvn ")?;
            two_same(f, rd, rn, 0, q)
        }
        Op::ReverseBits { q, rd, rn } => {
            f.write_str("rbit ")?;
            two_same(f, rd, rn, 0, q)
        }
        Op::CompareZero {
            kind,
            size,
            q,
            rd,
            rn,
        } => {
            f.write_str(match kind {
                Zero::Gt => "cmgt ",
                Zero::Ge => "cmge ",
                Zero::Le => "cmle ",
                Zero::Lt => "cmlt ",
            })?;
            two_same(f, rd, rn, size as u8, q)?;
            f.write_str(", #0")
        }
        Op::Abs {
            negate,
            size,
            q,
            rd,
            rn,
        } => {
            f.write_str(if negate { "neg " } else { "abs " })?;
            two_same(f, rd, rn, size as u8, q)
        }
        Op::Narrow {
            kind,
            size,
            upper,
            rd,
            rn,
        } => {
            let name = match kind {
                Narrow::Plain => "xtn",
                Narrow::SignedSat => "sqxtn",
                Narrow::UnsignedSat => "uqxtn",
                Narrow::SignedToUnsignedSat => "sqxtun",
            };
            let size = size as u8 & 3;
            write!(f, "{name}{} ", two(upper))?;
            vector(f, rd, size, upper)?;
            f.write_str(", ")?;
            vector(f, rn, size.wrapping_add(1) & 3, true)
        }
        Op::PairLong {
            signed,
            accumulate,
            size,
            q,
            rd,
            rn,
        } => {
            let size = size as u8 & 3;
            write!(
                f,
                "{}{} ",
                if signed { 's' } else { 'u' },
                if accumulate { "adalp" } else { "addlp" }
            )?;
            vector(f, rd, size.wrapping_add(1) & 3, q)?;
            f.write_str(", ")?;
            vector(f, rn, size, q)
        }
        Op::Long {
            kind,
            signed,
            size,
            upper,
            rd,
            rn,
            rm,
        } => {
            let name = match kind {
                Long::Add => "addl",
                Long::AddWide => "addw",
                Long::Sub => "subl",
                Long::SubWide => "subw",
                Long::Abd => "abdl",
                Long::Mul => "mull",
                Long::Mla => "mlal",
                Long::Mls => "mlsl",
            };
            let size = size as u8 & 3;
            write!(f, "{}{name}{} ", if signed { 's' } else { 'u' }, two(upper))?;
            vector(f, rd, size.wrapping_add(1) & 3, true)?;
            f.write_str(", ")?;
            if matches!(kind, Long::AddWide | Long::SubWide) {
                vector(f, rn, size.wrapping_add(1) & 3, true)?;
            } else {
                vector(f, rn, size, upper)?;
            }
            f.write_str(", ")?;
            vector(f, rm, size, upper)
        }
        Op::ScalarAddp { rd, rn } => write!(f, "addp d{rd}, v{rn}.2d"),
        Op::ScalarDup {
            size,
            index,
            rd,
            rn,
        } => {
            let size = size as u8 & 3;
            f.write_str("mov ")?;
            fp_scalar(f, 1 << size, rd)?;
            f.write_str(", ")?;
            element(f, rn, size, index as u8)
        }
        Op::ScalarThree { kind, rd, rn, rm } => {
            let name = match kind {
                Scalar::Add => "add",
                Scalar::Sub => "sub",
                Scalar::Gt => "cmgt",
                Scalar::Hi => "cmhi",
                Scalar::Ge => "cmge",
                Scalar::Hs => "cmhs",
                Scalar::Eq => "cmeq",
                Scalar::Tst => "cmtst",
            };
            write!(f, "{name} d{rd}, d{rn}, d{rm}")
        }
        Op::ShiftImm {
            right,
            signed,
            size,
            q,
            shift,
            rd,
            rn,
        } => {
            f.write_str(match (right, signed) {
                (false, _) => "shl ",
                (true, true) => "sshr ",
                (true, false) => "ushr ",
            })?;
            two_same(f, rd, rn, size as u8, q)?;
            write!(f, ", #{shift}")
        }
        Op::ShiftLong {
            signed,
            size,
            shift,
            upper,
            rd,
            rn,
        } => {
            let size = size as u8 & 3;
            let sign = if signed { 's' } else { 'u' };
            if shift == 0 {
                write!(f, "{sign}xtl{} ", two(upper))?;
            } else {
                write!(f, "{sign}shll{} ", two(upper))?;
            }
            vector(f, rd, size.wrapping_add(1) & 3, true)?;
            f.write_str(", ")?;
            vector(f, rn, size, upper)?;
            if shift != 0 {
                write!(f, ", #{shift}")?;
            }
            Ok(())
        }
        Op::ShiftNarrow {
            size,
            shift,
            upper,
            rd,
            rn,
        } => {
            let size = size as u8 & 3;
            write!(f, "shrn{} ", two(upper))?;
            vector(f, rd, size, upper)?;
            f.write_str(", ")?;
            vector(f, rn, size.wrapping_add(1) & 3, true)?;
            write!(f, ", #{shift}")
        }
    }
}

/// A floating-point immediate the way binutils prints it: C's `%.18e`.
fn float_immediate(f: &mut Formatter<'_>, value: f64) -> fmt::Result {
    // `fmov` immediates are always finite; other bit patterns only come from
    // hand-built values and print as C's `%f` would.
    if !value.is_finite() {
        return f.write_str(if value.is_nan() {
            "nan"
        } else if value < 0.0 {
            "-inf"
        } else {
            "inf"
        });
    }
    let mut text = Text {
        bytes: [0; 64],
        len: 0,
    };
    write!(text, "{value:.18e}")?;
    let text = core::str::from_utf8(&text.bytes[..text.len]).map_err(|_| fmt::Error)?;
    let (mantissa, exponent) = text.split_once('e').ok_or(fmt::Error)?;
    let exponent: i32 = exponent.parse().map_err(|_| fmt::Error)?;
    let sign = if exponent < 0 { '-' } else { '+' };
    write!(f, "{mantissa}e{sign}{:02}", exponent.unsigned_abs())
}

/// A fixed buffer to format a float into; every value printed here fits.
struct Text {
    bytes: [u8; 64],
    len: usize,
}

impl Write for Text {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let end = self.len.checked_add(s.len()).ok_or(fmt::Error)?;
        self.bytes
            .get_mut(self.len..end)
            .ok_or(fmt::Error)?
            .copy_from_slice(s.as_bytes());
        self.len = end;
        Ok(())
    }
}

/// `name v<rd>.<lanes>, #<imm8>[, <kind> #<shift>]`.
fn lane_form(
    f: &mut Formatter<'_>,
    name: &str,
    rd: u8,
    lanes: &str,
    imm8: u8,
    kind: &str,
    shift: u8,
) -> fmt::Result {
    write!(f, "{name} v{rd}.{lanes}, #{imm8:#x}")?;
    if shift != 0 {
        write!(f, ", {kind} #{shift}")?;
    }
    Ok(())
}

/// A half-precision value (as `fmov`'s expanded immediates are: always normal) as
/// a double.
fn half_to_f64(bits: u16) -> f64 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = u64::from((bits >> 10) & 0x1f);
    let fraction = f64::from(bits & 0x3ff);
    // 2^(e - 25) and 2^-24 as double exponents.
    let scale = |e: u64| f64::from_bits((1023 + e - 25) << 52);
    match exponent {
        0 => sign * fraction * scale(1),
        31 if fraction == 0.0 => sign * f64::INFINITY,
        31 => f64::NAN,
        e => sign * (1024.0 + fraction) * scale(e),
    }
}

/// A vector modified immediate, spelled from the encoding the decoder kept
/// (`form`, `imm8`, `combine`, `q`); `low` is only read for the 64-bit `movi`
/// and the `fmov` forms, whose operand is the expanded pattern.
fn immediate(f: &mut Formatter<'_>, imm: SimdImmediate) -> fmt::Result {
    let SimdImmediate {
        rd,
        imm8,
        form,
        combine,
        q,
        low,
        ..
    } = imm;
    let name = match combine {
        ImmCombine::Set => "movi",
        ImmCombine::SetInverted => "mvni",
        ImmCombine::Or => "orr",
        ImmCombine::AndNot => "bic",
    };
    match form {
        ImmForm::Byte => lane_form(f, name, rd, if q { "16b" } else { "8b" }, imm8, "lsl", 0),
        ImmForm::Half { shift } => {
            lane_form(f, name, rd, if q { "8h" } else { "4h" }, imm8, "lsl", shift)
        }
        ImmForm::Word { shift } => {
            lane_form(f, name, rd, if q { "4s" } else { "2s" }, imm8, "lsl", shift)
        }
        ImmForm::WordMask { shift } => {
            lane_form(f, name, rd, if q { "4s" } else { "2s" }, imm8, "msl", shift)
        }
        // The 64-bit lane: the whole expanded constant, as a scalar when `Q` is clear.
        ImmForm::Double if q => write!(f, "{name} v{rd}.2d, #{low:#x}"),
        ImmForm::Double => write!(f, "{name} d{rd}, #{low:#x}"),
        ImmForm::FmovHalf => {
            write!(f, "fmov v{rd}.{}, #", if q { "8h" } else { "4h" })?;
            float_immediate(f, half_to_f64(low as u16))
        }
        ImmForm::FmovSingle => {
            write!(f, "fmov v{rd}.{}, #", if q { "4s" } else { "2s" })?;
            float_immediate(f, f64::from(f32::from_bits(low as u32)))
        }
        ImmForm::FmovDouble => {
            write!(f, "fmov v{rd}.2d, #")?;
            float_immediate(f, f64::from_bits(low))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::{ImmCombine, SimdAlu, SimdDupElement, SimdExtract, SimdInsert, decode};

    struct Show<'a>(&'a Instruction);

    impl core::fmt::Display for Show<'_> {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            write(
                f,
                &Context {
                    pc: None,
                    symbols: &(),
                },
                self.0,
            )
        }
    }

    /// The rendered text, kept in a fixed buffer.
    struct Rendered(Text);

    impl Rendered {
        fn as_str(&self) -> &str {
            core::str::from_utf8(&self.0.bytes[..self.0.len]).unwrap()
        }
    }

    impl PartialEq<&str> for Rendered {
        fn eq(&self, other: &&str) -> bool {
            self.as_str() == *other
        }
    }

    impl core::fmt::Debug for Rendered {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            write!(f, "{:?}", self.as_str())
        }
    }

    fn text(instruction: &Instruction) -> Rendered {
        let mut text = Text {
            bytes: [0; 64],
            len: 0,
        };
        write!(text, "{}", Show(instruction)).unwrap();
        Rendered(text)
    }

    /// Words and the `objdump` text (binutils 2.46) for them. Excluded on purpose:
    /// `movi v0.16b, #0xff`, `movi d31, #0xffffffffffffffff` and the other
    /// encodings that expand to the same pattern as a form listed here (`mvni v0.4s,
    /// #0`); the decoder keeps only the expanded pattern, so it cannot tell them apart.
    #[test]
    fn matches_objdump() {
        let cases: [(u32, &str); 247] = [
            (0x4f000400, "movi v0.4s, #0x0"),
            (0x0f000401, "movi v1.2s, #0x0"),
            (0x4f05657f, "movi v31.4s, #0xab, lsl #24"),
            (0x0f002420, "movi v0.2s, #0x1, lsl #8"),
            (0x4f0347e0, "movi v0.4s, #0x7f, lsl #16"),
            (0x6f0747e0, "mvni v0.4s, #0xff, lsl #16"),
            (0x2f002423, "mvni v3.2s, #0x1, lsl #8"),
            (0x6f000423, "mvni v3.4s, #0x1"),
            (0x6f000400, "mvni v0.4s, #0x0"),
            (0x2f000400, "mvni v0.2s, #0x0"),
            (0x4f008640, "movi v0.8h, #0x12"),
            (0x0f00a420, "movi v0.4h, #0x1, lsl #8"),
            (0x6f00a460, "mvni v0.8h, #0x3, lsl #8"),
            (0x2f008460, "mvni v0.4h, #0x3"),
            (0x4f00c420, "movi v0.4s, #0x1, msl #8"),
            (0x0f07d7e0, "movi v0.2s, #0xff, msl #16"),
            (0x6f00d420, "mvni v0.4s, #0x1, msl #16"),
            (0x2f00c4e0, "mvni v0.2s, #0x7, msl #8"),
            (0x4f00e640, "movi v0.16b, #0x12"),
            (0x0f00e640, "movi v0.8b, #0x12"),
            (0x2f05e4a0, "movi d0, #0xff00ff0000ff00ff"),
            (0x6f05e4a0, "movi v0.2d, #0xff00ff0000ff00ff"),
            (0x6f05e600, "movi v0.2d, #0xff00ffff00000000"),
            (0x4f03f600, "fmov v0.4s, #1.000000000000000000e+00"),
            (0x0f04f400, "fmov v0.2s, #-2.000000000000000000e+00"),
            (0x6f01f7e0, "fmov v0.2d, #3.100000000000000000e+01"),
            (0x6f06f400, "fmov v0.2d, #-1.250000000000000000e-01"),
            (0x4f03f401, "fmov v1.4s, #5.000000000000000000e-01"),
            (0x4f03f7e1, "fmov v1.4s, #1.937500000000000000e+00"),
            (0x6f07f7e1, "fmov v1.2d, #-1.937500000000000000e+00"),
            (0x0f02f401, "fmov v1.2s, #1.250000000000000000e-01"),
            (0x4f001420, "orr v0.4s, #0x1"),
            (0x4f0777e0, "orr v0.4s, #0xff, lsl #24"),
            (0x0f0054e0, "orr v0.2s, #0x7, lsl #16"),
            (0x4f00b420, "orr v0.8h, #0x1, lsl #8"),
            (0x0f049400, "orr v0.4h, #0x80"),
            (0x2f005440, "bic v0.2s, #0x2, lsl #16"),
            (0x2f0094e0, "bic v0.4h, #0x7"),
            (0x6f00361f, "bic v31.4s, #0x10, lsl #8"),
            (0x4e209801, "cmeq v1.16b, v0.16b, #0"),
            (0x0e209801, "cmeq v1.8b, v0.8b, #0"),
            (0x0ea09be1, "cmeq v1.2s, v31.2s, #0"),
            (0x4ee0981f, "cmeq v31.2d, v0.2d, #0"),
            (0x4e609801, "cmeq v1.8h, v0.8h, #0"),
            (0x4e040c20, "dup v0.4s, w1"),
            (0x4e080c20, "dup v0.2d, x1"),
            (0x0e010fe0, "dup v0.8b, wzr"),
            (0x4e010c60, "dup v0.16b, w3"),
            (0x0e020c40, "dup v0.4h, w2"),
            (0x0e040fff, "dup v31.2s, wzr"),
            (0x4e140420, "dup v0.4s, v1.s[2]"),
            (0x4e1f0420, "dup v0.16b, v1.b[15]"),
            (0x4e1807e0, "dup v0.2d, v31.d[1]"),
            (0x4e1e0420, "dup v0.8h, v1.h[7]"),
            (0x0e040420, "dup v0.2s, v1.s[0]"),
            (0x0e0e0420, "dup v0.4h, v1.h[3]"),
            (0x4e0c1c20, "mov v0.s[1], w1"),
            (0x4e181fe0, "mov v0.d[1], xzr"),
            (0x4e1f1c40, "mov v0.b[15], w2"),
            (0x4e021fe0, "mov v0.h[0], wzr"),
            (0x4e081fdf, "mov v31.d[0], x30"),
            (0x6e0c4420, "mov v0.s[1], v1.s[2]"),
            (0x6e1f07e0, "mov v0.b[15], v31.b[0]"),
            (0x6e180420, "mov v0.d[1], v1.d[0]"),
            (0x6e1e3420, "mov v0.h[7], v1.h[3]"),
            (0x6e1c6420, "mov v0.s[3], v1.s[3]"),
            (0x0e073c20, "umov w0, v1.b[3]"),
            (0x0e1e3c20, "umov w0, v1.h[7]"),
            (0x0e1c3c20, "mov w0, v1.s[3]"),
            (0x4e183c20, "mov x0, v1.d[1]"),
            (0x0e043ffe, "mov w30, v31.s[0]"),
            (0x0e1f3fe0, "umov w0, v31.b[15]"),
            (0x6e021820, "ext v0.16b, v1.16b, v2.16b, #3"),
            (0x6e027820, "ext v0.16b, v1.16b, v2.16b, #15"),
            (0x6e1f03ff, "ext v31.16b, v31.16b, v31.16b, #0"),
            (0x4e228420, "add v0.16b, v1.16b, v2.16b"),
            (0x4ee28420, "add v0.2d, v1.2d, v2.2d"),
            (0x0e628420, "add v0.4h, v1.4h, v2.4h"),
            (0x6e628420, "sub v0.8h, v1.8h, v2.8h"),
            (0x4ea29c20, "mul v0.4s, v1.4s, v2.4s"),
            (0x0ea29420, "mla v0.2s, v1.2s, v2.2s"),
            (0x6e229420, "mls v0.16b, v1.16b, v2.16b"),
            (0x4e221c20, "and v0.16b, v1.16b, v2.16b"),
            (0x0e221c20, "and v0.8b, v1.8b, v2.8b"),
            (0x4e621c20, "bic v0.16b, v1.16b, v2.16b"),
            (0x4ea21c20, "orr v0.16b, v1.16b, v2.16b"),
            (0x0ea21c20, "orr v0.8b, v1.8b, v2.8b"),
            (0x4ea11c20, "mov v0.16b, v1.16b"),
            (0x0ebf1fff, "mov v31.8b, v31.8b"),
            (0x4ee21c20, "orn v0.16b, v1.16b, v2.16b"),
            (0x6e221c20, "eor v0.16b, v1.16b, v2.16b"),
            (0x6e621c20, "bsl v0.16b, v1.16b, v2.16b"),
            (0x2ea21c20, "bit v0.8b, v1.8b, v2.8b"),
            (0x6ee21c20, "bif v0.16b, v1.16b, v2.16b"),
            (0x6ea28c20, "cmeq v0.4s, v1.4s, v2.4s"),
            (0x4ee23420, "cmgt v0.2d, v1.2d, v2.2d"),
            (0x4e623c20, "cmge v0.8h, v1.8h, v2.8h"),
            (0x6e223420, "cmhi v0.16b, v1.16b, v2.16b"),
            (0x6ea23c20, "cmhs v0.4s, v1.4s, v2.4s"),
            (0x0ea28c20, "cmtst v0.2s, v1.2s, v2.2s"),
            (0x6e226c20, "umin v0.16b, v1.16b, v2.16b"),
            (0x6e626420, "umax v0.8h, v1.8h, v2.8h"),
            (0x4ea26c20, "smin v0.4s, v1.4s, v2.4s"),
            (0x0e226420, "smax v0.8b, v1.8b, v2.8b"),
            (0x6e227420, "uabd v0.16b, v1.16b, v2.16b"),
            (0x0e627420, "sabd v0.4h, v1.4h, v2.4h"),
            (0x4ee20c20, "sqadd v0.2d, v1.2d, v2.2d"),
            (0x6e220c20, "uqadd v0.16b, v1.16b, v2.16b"),
            (0x4e622c20, "sqsub v0.8h, v1.8h, v2.8h"),
            (0x6ea22c20, "uqsub v0.4s, v1.4s, v2.4s"),
            (0x4e220420, "shadd v0.16b, v1.16b, v2.16b"),
            (0x6e620420, "uhadd v0.8h, v1.8h, v2.8h"),
            (0x4ea22420, "shsub v0.4s, v1.4s, v2.4s"),
            (0x2e222420, "uhsub v0.8b, v1.8b, v2.8b"),
            (0x4e221420, "srhadd v0.16b, v1.16b, v2.16b"),
            (0x6e621420, "urhadd v0.8h, v1.8h, v2.8h"),
            (0x4ee24420, "sshl v0.2d, v1.2d, v2.2d"),
            (0x6e224420, "ushl v0.16b, v1.16b, v2.16b"),
            (0x4ea24c20, "sqshl v0.4s, v1.4s, v2.4s"),
            (0x6e624c20, "uqshl v0.8h, v1.8h, v2.8h"),
            (0x4e225c20, "sqrshl v0.16b, v1.16b, v2.16b"),
            (0x6ee25c20, "uqrshl v0.2d, v1.2d, v2.2d"),
            (0x4ea2bc20, "addp v0.4s, v1.4s, v2.4s"),
            (0x4e22bc20, "addp v0.16b, v1.16b, v2.16b"),
            (0x6e62a420, "umaxp v0.8h, v1.8h, v2.8h"),
            (0x4ea2a420, "smaxp v0.4s, v1.4s, v2.4s"),
            (0x6e22ac20, "uminp v0.16b, v1.16b, v2.16b"),
            (0x0ea2ac20, "sminp v0.2s, v1.2s, v2.2s"),
            (0x4e31b820, "addv b0, v1.16b"),
            (0x0e71b83f, "addv h31, v1.4h"),
            (0x4eb1bbe0, "addv s0, v31.4s"),
            (0x0e31b820, "addv b0, v1.8b"),
            (0x6e71a820, "uminv h0, v1.8h"),
            (0x6e30a820, "umaxv b0, v1.16b"),
            (0x6eb0a820, "umaxv s0, v1.4s"),
            (0x6e303820, "uaddlv h0, v1.16b"),
            (0x6e703820, "uaddlv s0, v1.8h"),
            (0x6eb03820, "uaddlv d0, v1.4s"),
            (0x2e303820, "uaddlv h0, v1.8b"),
            (0x4e030020, "tbl v0.16b, {v1.16b}, v3.16b"),
            (0x4e032020, "tbl v0.16b, {v1.16b-v2.16b}, v3.16b"),
            (0x4e044020, "tbl v0.16b, {v1.16b-v3.16b}, v4.16b"),
            (0x0e1f6020, "tbl v0.8b, {v1.16b-v4.16b}, v31.8b"),
            (0x4e0323e0, "tbl v0.16b, {v31.16b-v0.16b}, v3.16b"),
            (0x4e0373c0, "tbx v0.16b, {v30.16b-v1.16b}, v3.16b"),
            (0x0e0053bf, "tbx v31.8b, {v29.16b-v31.16b}, v0.8b"),
            (0x0e031020, "tbx v0.8b, {v1.16b}, v3.8b"),
            (0x4e021820, "uzp1 v0.16b, v1.16b, v2.16b"),
            (0x4e425820, "uzp2 v0.8h, v1.8h, v2.8h"),
            (0x4e823820, "zip1 v0.4s, v1.4s, v2.4s"),
            (0x4ec27820, "zip2 v0.2d, v1.2d, v2.2d"),
            (0x0e022820, "trn1 v0.8b, v1.8b, v2.8b"),
            (0x0e826820, "trn2 v0.2s, v1.2s, v2.2s"),
            (0x4e201820, "rev16 v0.16b, v1.16b"),
            (0x6e600820, "rev32 v0.8h, v1.8h"),
            (0x2e200820, "rev32 v0.8b, v1.8b"),
            (0x4ea00820, "rev64 v0.4s, v1.4s"),
            (0x4e200820, "rev64 v0.16b, v1.16b"),
            (0x4e205820, "cnt v0.16b, v1.16b"),
            (0x0e205820, "cnt v0.8b, v1.8b"),
            (0x6e205820, "mvn v0.16b, v1.16b"),
            (0x2e205820, "mvn v0.8b, v1.8b"),
            (0x6e605820, "rbit v0.16b, v1.16b"),
            (0x2e605820, "rbit v0.8b, v1.8b"),
            (0x4ea08820, "cmgt v0.4s, v1.4s, #0"),
            (0x6e208820, "cmge v0.16b, v1.16b, #0"),
            (0x6e609820, "cmle v0.8h, v1.8h, #0"),
            (0x4ee0a820, "cmlt v0.2d, v1.2d, #0"),
            (0x0ea0a820, "cmlt v0.2s, v1.2s, #0"),
            (0x4ea0b820, "abs v0.4s, v1.4s"),
            (0x4ee0b820, "abs v0.2d, v1.2d"),
            (0x6e20b820, "neg v0.16b, v1.16b"),
            (0x2e60b820, "neg v0.4h, v1.4h"),
            (0x0e212820, "xtn v0.8b, v1.8h"),
            (0x4e212820, "xtn2 v0.16b, v1.8h"),
            (0x0e612820, "xtn v0.4h, v1.4s"),
            (0x4ea12820, "xtn2 v0.4s, v1.2d"),
            (0x0ea14820, "sqxtn v0.2s, v1.2d"),
            (0x4e614820, "sqxtn2 v0.8h, v1.4s"),
            (0x2e214820, "uqxtn v0.8b, v1.8h"),
            (0x6ea14820, "uqxtn2 v0.4s, v1.2d"),
            (0x2e612820, "sqxtun v0.4h, v1.4s"),
            (0x6e212820, "sqxtun2 v0.16b, v1.8h"),
            (0x4e202820, "saddlp v0.8h, v1.16b"),
            (0x0e202820, "saddlp v0.4h, v1.8b"),
            (0x6e602820, "uaddlp v0.4s, v1.8h"),
            (0x2ea02820, "uaddlp v0.1d, v1.2s"),
            (0x6ea02820, "uaddlp v0.2d, v1.4s"),
            (0x4e606820, "sadalp v0.4s, v1.8h"),
            (0x6e206820, "uadalp v0.8h, v1.16b"),
            (0x0e220020, "saddl v0.8h, v1.8b, v2.8b"),
            (0x4e220020, "saddl2 v0.8h, v1.16b, v2.16b"),
            (0x2e620020, "uaddl v0.4s, v1.4h, v2.4h"),
            (0x6ea20020, "uaddl2 v0.2d, v1.4s, v2.4s"),
            (0x0ea22020, "ssubl v0.2d, v1.2s, v2.2s"),
            (0x6e622020, "usubl2 v0.4s, v1.8h, v2.8h"),
            (0x0e221020, "saddw v0.8h, v1.8h, v2.8b"),
            (0x4ea21020, "saddw2 v0.2d, v1.2d, v2.4s"),
            (0x2e621020, "uaddw v0.4s, v1.4s, v2.4h"),
            (0x6e221020, "uaddw2 v0.8h, v1.8h, v2.16b"),
            (0x0ea23020, "ssubw v0.2d, v1.2d, v2.2s"),
            (0x6e623020, "usubw2 v0.4s, v1.4s, v2.8h"),
            (0x0e227020, "sabdl v0.8h, v1.8b, v2.8b"),
            (0x6e627020, "uabdl2 v0.4s, v1.8h, v2.8h"),
            (0x0e628020, "smlal v0.4s, v1.4h, v2.4h"),
            (0x6ea28020, "umlal2 v0.2d, v1.4s, v2.4s"),
            (0x0e22a020, "smlsl v0.8h, v1.8b, v2.8b"),
            (0x6e62a020, "umlsl2 v0.4s, v1.8h, v2.8h"),
            (0x0ea2c020, "smull v0.2d, v1.2s, v2.2s"),
            (0x6e22c020, "umull2 v0.8h, v1.16b, v2.16b"),
            (0x5ef1b820, "addp d0, v1.2d"),
            (0x5e070420, "mov b0, v1.b[3]"),
            (0x5e1e043f, "mov h31, v1.h[7]"),
            (0x5e140420, "mov s0, v1.s[2]"),
            (0x5e1807e0, "mov d0, v31.d[1]"),
            (0x5ee28420, "add d0, d1, d2"),
            (0x7ee28420, "sub d0, d1, d2"),
            (0x5ee23420, "cmgt d0, d1, d2"),
            (0x7ee23420, "cmhi d0, d1, d2"),
            (0x5ee23c20, "cmge d0, d1, d2"),
            (0x7ee23c20, "cmhs d0, d1, d2"),
            (0x7ee28c20, "cmeq d0, d1, d2"),
            (0x5ee28c3f, "cmtst d31, d1, d2"),
            (0x4f205420, "shl v0.4s, v1.4s, #0"),
            (0x4f0f5420, "shl v0.16b, v1.16b, #7"),
            (0x4f7f5420, "shl v0.2d, v1.2d, #63"),
            (0x0f1f5420, "shl v0.4h, v1.4h, #15"),
            (0x0f080420, "sshr v0.8b, v1.8b, #8"),
            (0x4f400420, "sshr v0.2d, v1.2d, #64"),
            (0x6f3f0420, "ushr v0.4s, v1.4s, #1"),
            (0x6f100420, "ushr v0.8h, v1.8h, #16"),
            (0x0f0fa420, "sshll v0.8h, v1.8b, #7"),
            (0x4f1fa420, "sshll2 v0.4s, v1.8h, #15"),
            (0x0f3fa420, "sshll v0.2d, v1.2s, #31"),
            (0x2f09a420, "ushll v0.8h, v1.8b, #1"),
            (0x6f21a420, "ushll2 v0.2d, v1.4s, #1"),
            (0x0f08a420, "sxtl v0.8h, v1.8b"),
            (0x4f10a420, "sxtl2 v0.4s, v1.8h"),
            (0x0f20a420, "sxtl v0.2d, v1.2s"),
            (0x2f08a420, "uxtl v0.8h, v1.8b"),
            (0x6f20a420, "uxtl2 v0.2d, v1.4s"),
            (0x2f10a420, "uxtl v0.4s, v1.4h"),
            (0x0f088420, "shrn v0.8b, v1.8h, #8"),
            (0x4f0f8420, "shrn2 v0.16b, v1.8h, #1"),
            (0x0f208420, "shrn v0.2s, v1.2d, #32"),
            (0x4f3b8420, "shrn2 v0.4s, v1.2d, #5"),
            (0x0f108420, "shrn v0.4h, v1.4s, #16"),
        ];
        for (word, expected) in cases {
            let instruction = decode(word).unwrap_or_else(|e| panic!("{word:08x}: {e:?}"));
            assert_eq!(text(&instruction), expected, "{word:08x}");
        }
    }

    /// Words verified against GNU `objdump`.
    #[test]
    fn vector_moves_match_objdump() {
        for (word, expected) in [
            (0x2efe1fe0, "bif v0.8b, v31.8b, v30.8b"),
            (0x0ee01c20, "orn v0.8b, v1.8b, v0.8b"),
            (0x0ea21c20, "orr v0.8b, v1.8b, v2.8b"),
            (0x0e211c20, "and v0.8b, v1.8b, v1.8b"),
            (0x4e1c1c20, "mov v0.s[3], w1"),
            (0x4e083c20, "mov x0, v1.d[0]"),
            (0x0e013c20, "umov w0, v1.b[0]"),
        ] {
            let instruction = decode(word).unwrap_or_else(|e| panic!("{word:08x}: {e:?}"));
            assert_eq!(text(&instruction), expected, "{word:08x}");
        }
    }

    /// Words checked against GNU `objdump`: the modified-immediate forms keep the
    /// arrangement and shift, so encodings that expand to the same pattern differ.
    #[test]
    fn modified_immediates_keep_their_encoding() {
        let cases: [(u32, &str); 27] = [
            (0x4f000400, "movi v0.4s, #0x0"),
            (0x0f000400, "movi v0.2s, #0x0"),
            (0x6f00e401, "movi v1.2d, #0x0"),
            (0x2f00e400, "movi d0, #0x0"),
            (0x4f00e400, "movi v0.16b, #0x0"),
            (0x0f00e460, "movi v0.8b, #0x3"),
            (0x6f0707ff, "mvni v31.4s, #0xff"),
            (0x6f00045f, "mvni v31.4s, #0x2"),
            (0x6f00c400, "mvni v0.4s, #0x0, msl #8"),
            (0x4f00d420, "movi v0.4s, #0x1, msl #16"),
            (0x0f008420, "movi v0.4h, #0x1"),
            (0x4f00a420, "movi v0.8h, #0x1, lsl #8"),
            (0x6f00a420, "mvni v0.8h, #0x1, lsl #8"),
            (0x4f009420, "orr v0.8h, #0x1"),
            (0x4f00b420, "orr v0.8h, #0x1, lsl #8"),
            (0x2f00b420, "bic v0.4h, #0x1, lsl #8"),
            (0x0f00540a, "orr v10.2s, #0x0, lsl #16"),
            (0x0f006400, "movi v0.2s, #0x0, lsl #24"),
            (0x4f00f400, "fmov v0.4s, #2.000000000000000000e+00"),
            (0x6f00f400, "fmov v0.2d, #2.000000000000000000e+00"),
            (0x4f00fc00, "fmov v0.8h, #2.000000000000000000e+00"),
            (0x0f00fc00, "fmov v0.4h, #2.000000000000000000e+00"),
            (0x4f04fc80, "fmov v0.8h, #-2.500000000000000000e+00"),
            (0x2f00e420, "movi d0, #0xff"),
            (0x2f0367e1, "mvni v1.2s, #0x7f, lsl #24"),
            (0x2e023820, "ext v0.8b, v1.8b, v2.8b, #7"),
            (0x6e024820, "ext v0.16b, v1.16b, v2.16b, #9"),
        ];
        for (word, expected) in cases {
            let instruction = decode(word).unwrap_or_else(|e| panic!("{word:08x}: {e:?}"));
            assert_eq!(text(&instruction), expected, "{word:08x}");
        }
        // Not instructions (`objdump`: `.inst ... ; undefined`) and so not decoded: an
        // `ext .8b` index above 7, a modified immediate with bit 11 set outside the
        // half-precision `fmov`, `fmov .2d` with `Q` clear.
        for word in [0x2e164372, 0x0f000ced, 0x4f00af48, 0x2f00f420, 0x6f00fc20] {
            assert!(decode(word).is_err(), "{word:08x}");
        }
    }

    /// Fields beyond what the decoder produces must still print, not panic.
    #[test]
    fn extreme_fields_do_not_panic() {
        use crate::simd::Op;
        let ops = [
            Op::ShiftImm {
                right: true,
                signed: true,
                size: u32::MAX,
                q: true,
                shift: u32::MAX,
                rd: 255,
                rn: 255,
            },
            Op::ShiftLong {
                signed: false,
                size: 7,
                shift: 1,
                upper: true,
                rd: 255,
                rn: 255,
            },
            Op::ScalarDup {
                size: 99,
                index: u32::MAX,
                rd: 255,
                rn: 255,
            },
            Op::Narrow {
                kind: crate::simd::Narrow::Plain,
                size: 3,
                upper: true,
                rd: 0,
                rn: 0,
            },
        ];
        for op in ops {
            let _ = text(&Instruction::Simd(op));
        }
        let _ = text(&Instruction::SimdAlu(SimdAlu {
            op: SimdAluOp::UAddLV,
            rd: 255,
            rn: 255,
            rm: 255,
            size: 255,
            q: true,
        }));
        let _ = text(&Instruction::SimdDupElement(SimdDupElement {
            rd: 255,
            rn: 255,
            esize: 255,
            index: 255,
            q: false,
        }));
        let _ = text(&Instruction::SimdInsert(SimdInsert {
            rd: 255,
            rn: 255,
            esize: 255,
            index: 255,
        }));
        let _ = text(&Instruction::SimdExtract(SimdExtract {
            rd: 255,
            rn: 255,
            esize: 255,
            index: 255,
        }));
        let _ = text(&Instruction::SimdTable(SimdTable {
            rd: 255,
            rn: 255,
            rm: 255,
            len: 255,
            extend: true,
            q: true,
        }));
        for low in [0, 1, u64::MAX, 0x0123_4567_89ab_cdef, 0x8000_0000_8000_0000] {
            for combine in [
                ImmCombine::Set,
                ImmCombine::SetInverted,
                ImmCombine::Or,
                ImmCombine::AndNot,
            ] {
                for form in [
                    ImmForm::Byte,
                    ImmForm::Half { shift: 255 },
                    ImmForm::Word { shift: 255 },
                    ImmForm::WordMask { shift: 255 },
                    ImmForm::Double,
                    ImmForm::FmovHalf,
                    ImmForm::FmovSingle,
                    ImmForm::FmovDouble,
                ] {
                    for q in [false, true] {
                        let _ = text(&Instruction::SimdImm(SimdImmediate {
                            rd: 255,
                            imm8: 255,
                            form,
                            low,
                            high: if q { low } else { 0 },
                            combine,
                            q,
                        }));
                    }
                }
            }
        }
    }
}
