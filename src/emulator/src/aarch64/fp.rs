//! Scalar floating point (single and double precision) and the vector `fabs`/`fneg`
//! forms, executed with host `f32`/`f64` arithmetic.
//!
//! Decode only classifies a word with [`parse`]; the JIT stores the word in
//! `Cpu::value` and traps, and the machine (or user mode) runs [`run`] on the CPU.
//! Results follow the Arm rules where they differ from the host's: an invalid
//! operation yields the default NaN (positive, quiet), a NaN operand propagates
//! quietened, and `fmax(+0, -0)` is `+0`.
//!
//! Not modelled: half precision, `FPCR` rounding modes and flush-to-zero (the
//! rounding is always to nearest, ties to even), and the `FPSR` exception flags.

use super::Cpu;
use volt_target::aarch64::decode::expand_immediate;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Precision {
    Single,
    Double,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Unary {
    Move,
    Abs,
    Neg,
    Sqrt,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Binary {
    Mul,
    Div,
    Add,
    Sub,
    Max,
    Min,
    MaxNum,
    MinNum,
    Nmul,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fused {
    Madd,
    Msub,
    Nmadd,
    Nmsub,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    Unary {
        kind: Unary,
        p: Precision,
        rd: u8,
        rn: u8,
    },
    /// `fcvt`: between single and double; `p` is the source precision.
    Convert {
        p: Precision,
        rd: u8,
        rn: u8,
    },
    Binary {
        kind: Binary,
        p: Precision,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    Fused {
        kind: Fused,
        p: Precision,
        rd: u8,
        rn: u8,
        rm: u8,
        ra: u8,
    },
    /// `fcmp`/`fcmpe`; no `rm` compares with zero.
    Compare {
        p: Precision,
        rn: u8,
        rm: Option<u8>,
    },
    CondCompare {
        p: Precision,
        rn: u8,
        rm: u8,
        cond: u32,
        nzcv: u32,
    },
    Select {
        p: Precision,
        rd: u8,
        rn: u8,
        rm: u8,
        cond: u32,
    },
    Immediate {
        p: Precision,
        rd: u8,
        imm8: u64,
    },
    /// `scvtf`/`ucvtf` from a general register.
    IntToFloat {
        signed: bool,
        p: Precision,
        wide: bool,
        rd: u8,
        rn: u8,
    },
    /// `fcvtzs`/`fcvtzu` to a general register.
    FloatToInt {
        signed: bool,
        p: Precision,
        wide: bool,
        rd: u8,
        rn: u8,
    },
    /// `scvtf`/`ucvtf` between SIMD&FP registers (the source holds an integer).
    ScalarIntToFloat {
        signed: bool,
        p: Precision,
        rd: u8,
        rn: u8,
    },
    /// `fcvtzs`/`fcvtzu` between SIMD&FP registers.
    ScalarFloatToInt {
        signed: bool,
        p: Precision,
        rd: u8,
        rn: u8,
    },
    /// Vector `fabs`/`fneg`.
    VectorSign {
        negate: bool,
        double: bool,
        q: bool,
        rd: u8,
        rn: u8,
    },
}

fn precision(ty: u32) -> Option<Precision> {
    match ty {
        0 => Some(Precision::Single),
        1 => Some(Precision::Double),
        // 3 is half precision, which is not modelled; 2 is reserved.
        _ => None,
    }
}

fn parse(word: u32) -> Option<Op> {
    let rd = (word & 31) as u8;
    let rn = ((word >> 5) & 31) as u8;
    let rm = ((word >> 16) & 31) as u8;
    let ty = (word >> 22) & 3;
    // Scalar floating point: `M=0 S=0 01111 0 type 1 ...`.
    if word & 0xff20_0c00 == 0x1e20_0800 {
        let kind = match (word >> 12) & 15 {
            0 => Binary::Mul,
            1 => Binary::Div,
            2 => Binary::Add,
            3 => Binary::Sub,
            4 => Binary::Max,
            5 => Binary::Min,
            6 => Binary::MaxNum,
            7 => Binary::MinNum,
            8 => Binary::Nmul,
            _ => return None,
        };
        return Some(Op::Binary {
            kind,
            p: precision(ty)?,
            rd,
            rn,
            rm,
        });
    }
    // One-source: bits 14:10 = 10000; the opcode (bits 20:15) includes bit 15.
    if word & 0xff20_7c00 == 0x1e20_4000 {
        let p = precision(ty)?;
        let kind = match (word >> 15) & 0x3f {
            0 => Unary::Move,
            1 => Unary::Abs,
            2 => Unary::Neg,
            3 => Unary::Sqrt,
            // `fcvt` to single from double, to double from single.
            4 if p == Precision::Double => return Some(Op::Convert { p, rd, rn }),
            5 if p == Precision::Single => return Some(Op::Convert { p, rd, rn }),
            _ => return None,
        };
        return Some(Op::Unary { kind, p, rd, rn });
    }
    if word & 0xff20_fc07 == 0x1e20_2000 {
        let zero = word & 8 != 0;
        return Some(Op::Compare {
            p: precision(ty)?,
            rn,
            rm: (!zero).then_some(rm),
        });
    }
    if word & 0xff20_0c10 == 0x1e20_0400 {
        return Some(Op::CondCompare {
            p: precision(ty)?,
            rn,
            rm,
            cond: (word >> 12) & 15,
            nzcv: word & 15,
        });
    }
    if word & 0xff20_0c00 == 0x1e20_0c00 {
        return Some(Op::Select {
            p: precision(ty)?,
            rd,
            rn,
            rm,
            cond: (word >> 12) & 15,
        });
    }
    if word & 0xff20_1fe0 == 0x1e20_1000 {
        return Some(Op::Immediate {
            p: precision(ty)?,
            rd,
            imm8: u64::from((word >> 13) & 0xff),
        });
    }
    if word & 0xff00_0000 == 0x1f00_0000 {
        let kind = match ((word >> 21) & 1, (word >> 15) & 1) {
            (0, 0) => Fused::Madd,
            (0, _) => Fused::Msub,
            (_, 0) => Fused::Nmadd,
            _ => Fused::Nmsub,
        };
        return Some(Op::Fused {
            kind,
            p: precision(ty)?,
            rd,
            rn,
            rm,
            ra: ((word >> 10) & 31) as u8,
        });
    }
    // Conversions between floating point and general registers. The `fmov` forms
    // (rmode 0, opcode 6 and 7) are `SimdFmov`.
    if word & 0x7f20_fc00 == 0x1e20_0000 {
        let (p, wide) = (precision(ty)?, word & 0x8000_0000 != 0);
        return match ((word >> 19) & 3, (word >> 16) & 7) {
            (0, 2) => Some(Op::IntToFloat {
                signed: true,
                p,
                wide,
                rd,
                rn,
            }),
            (0, 3) => Some(Op::IntToFloat {
                signed: false,
                p,
                wide,
                rd,
                rn,
            }),
            (3, 0) => Some(Op::FloatToInt {
                signed: true,
                p,
                wide,
                rd,
                rn,
            }),
            (3, 1) => Some(Op::FloatToInt {
                signed: false,
                p,
                wide,
                rd,
                rn,
            }),
            _ => None,
        };
    }
    // The same conversions between SIMD&FP registers: bit 22 is `sz`.
    let p = if word & 0x0040_0000 != 0 {
        Precision::Double
    } else {
        Precision::Single
    };
    match word & 0xffbf_fc00 {
        0x5e21_d800 => {
            return Some(Op::ScalarIntToFloat {
                signed: true,
                p,
                rd,
                rn,
            });
        }
        0x7e21_d800 => {
            return Some(Op::ScalarIntToFloat {
                signed: false,
                p,
                rd,
                rn,
            });
        }
        0x5ea1_b800 => {
            return Some(Op::ScalarFloatToInt {
                signed: true,
                p,
                rd,
                rn,
            });
        }
        0x7ea1_b800 => {
            return Some(Op::ScalarFloatToInt {
                signed: false,
                p,
                rd,
                rn,
            });
        }
        _ => {}
    }
    // Vector `fabs` (U=0) and `fneg` (U=1).
    if word & 0x9fbf_fc00 == 0x0ea0_f800 {
        let (q, double) = (word & 0x4000_0000 != 0, word & 0x0040_0000 != 0);
        // `.1d` is reserved.
        if double && !q {
            return None;
        }
        return Some(Op::VectorSign {
            negate: word & 0x2000_0000 != 0,
            double,
            q,
            rd,
            rn,
        });
    }
    None
}

/// Whether [`run`] implements `word`.
pub fn is_supported(word: u32) -> bool {
    parse(word).is_some()
}

trait Float: Copy + PartialOrd {
    const SIGN: u64;
    const QUIET: u64;
    fn from_u64(bits: u64) -> Self;
    fn to_u64(self) -> u64;
    fn is_nan(self) -> bool;
    fn default_nan() -> Self;
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    fn mul(self, other: Self) -> Self;
    fn div(self, other: Self) -> Self;
    fn sqrt(self) -> Self;
    /// `self * a + b`, rounded once.
    fn fma(self, a: Self, b: Self) -> Self;
    fn negative(self) -> bool {
        self.to_u64() & Self::SIGN != 0
    }
    fn neg(self) -> Self {
        Self::from_u64(self.to_u64() ^ Self::SIGN)
    }
    fn quiet(self) -> Self {
        Self::from_u64(self.to_u64() | Self::QUIET)
    }
}
macro_rules! float {
    ($t:ty, $bits:ty, $sign:expr, $quiet:expr, $nan:expr) => {
        impl Float for $t {
            const SIGN: u64 = $sign;
            const QUIET: u64 = $quiet;
            fn from_u64(bits: u64) -> Self {
                <$t>::from_bits(bits as $bits)
            }
            fn to_u64(self) -> u64 {
                u64::from(<$t>::to_bits(self))
            }
            fn is_nan(self) -> bool {
                <$t>::is_nan(self)
            }
            fn default_nan() -> Self {
                <$t>::from_bits($nan)
            }
            fn add(self, other: Self) -> Self {
                self + other
            }
            fn sub(self, other: Self) -> Self {
                self - other
            }
            fn mul(self, other: Self) -> Self {
                self * other
            }
            fn div(self, other: Self) -> Self {
                self / other
            }
            fn sqrt(self) -> Self {
                <$t>::sqrt(self)
            }
            fn fma(self, a: Self, b: Self) -> Self {
                <$t>::mul_add(self, a, b)
            }
        }
    };
}
float!(f32, u32, 1 << 31, 1 << 22, 0x7fc0_0000);
float!(f64, u64, 1 << 63, 1 << 51, 0x7ff8_0000_0000_0000);

/// An operation that produced a NaN returns the first NaN operand, quietened, or
/// the default NaN when the operands were numbers (the host's NaN has its own sign).
fn nan_or<T: Float>(result: T, operands: &[T]) -> T {
    if !result.is_nan() {
        return result;
    }
    operands
        .iter()
        .find(|o| o.is_nan())
        .map_or(T::default_nan(), |o| o.quiet())
}

fn extremum<T: Float>(a: T, b: T, max: bool, numbers_win: bool) -> T {
    if a.is_nan() || b.is_nan() {
        // `fmaxnm`/`fminnm` prefer the number over a quiet NaN.
        if numbers_win && !(a.is_nan() && b.is_nan()) {
            return if a.is_nan() { b } else { a };
        }
        return nan_or(T::default_nan(), &[a, b]);
    }
    if a == b {
        // Equal values: only the zeros differ, by sign. `max` prefers +0, `min` -0.
        return if a.negative() == max { b } else { a };
    }
    if (a > b) == max { a } else { b }
}

fn binary<T: Float>(kind: Binary, a: T, b: T) -> T {
    let result = match kind {
        Binary::Mul => a.mul(b),
        Binary::Div => a.div(b),
        Binary::Add => a.add(b),
        Binary::Sub => a.sub(b),
        Binary::Max => return extremum(a, b, true, false),
        Binary::Min => return extremum(a, b, false, false),
        Binary::MaxNum => return extremum(a, b, true, true),
        Binary::MinNum => return extremum(a, b, false, true),
        Binary::Nmul => {
            let product = nan_or(a.mul(b), &[a, b]);
            return if product.is_nan() {
                product
            } else {
                product.neg()
            };
        }
    };
    nan_or(result, &[a, b])
}

fn fused<T: Float>(kind: Fused, n: T, m: T, a: T) -> T {
    // fmadd: a + n*m; fmsub: a - n*m; fnmadd: -a - n*m; fnmsub: n*m - a.
    let (n, a) = match kind {
        Fused::Madd => (n, a),
        Fused::Msub => (n.neg(), a),
        Fused::Nmadd => (n.neg(), a.neg()),
        Fused::Nmsub => (n, a.neg()),
    };
    nan_or(n.fma(m, a), &[n, m, a])
}

fn unary<T: Float>(kind: Unary, a: T) -> T {
    match kind {
        Unary::Move => a,
        Unary::Abs if a.is_nan() => a,
        Unary::Abs => T::from_u64(a.to_u64() & !T::SIGN),
        Unary::Neg => a.neg(),
        Unary::Sqrt => nan_or(a.sqrt(), &[a]),
    }
}

/// `NZCV` (in bits 31:28) of comparing `a` with `b`.
fn compare<T: Float>(a: T, b: T) -> u32 {
    if a.is_nan() || b.is_nan() {
        0x3000_0000
    } else if a == b {
        0x6000_0000
    } else if a < b {
        0x8000_0000
    } else {
        0x2000_0000
    }
}

fn condition_holds(cond: u32, flags: u32) -> bool {
    let (n, z, c, v) = (
        flags >> 31 & 1 != 0,
        flags >> 30 & 1 != 0,
        flags >> 29 & 1 != 0,
        flags >> 28 & 1 != 0,
    );
    let base = match cond >> 1 {
        0 => z,
        1 => c,
        2 => n,
        3 => v,
        4 => c && !z,
        5 => n == v,
        6 => n == v && !z,
        _ => true,
    };
    // Odd conditions invert, except `nv` (0b1111), which also always holds.
    if cond & 1 != 0 && cond != 15 {
        !base
    } else {
        base
    }
}

fn read(cpu: &Cpu, p: Precision, reg: u8) -> u64 {
    match p {
        Precision::Single => cpu.v[usize::from(reg)][0] & 0xffff_ffff,
        Precision::Double => cpu.v[usize::from(reg)][0],
    }
}
/// Write a scalar result: the rest of the 128-bit register is zeroed.
fn write(cpu: &mut Cpu, reg: u8, bits: u64) {
    cpu.v[usize::from(reg)] = [bits, 0];
}

/// Apply `f32` or `f64` code to the registers of precision `p`.
macro_rules! at {
    ($p:expr, $t:ident => $body:expr) => {
        match $p {
            Precision::Single => {
                type $t = f32;
                $body
            }
            Precision::Double => {
                type $t = f64;
                $body
            }
        }
    };
}

/// Run the floating-point instruction `word` on `cpu`. `word` must satisfy
/// [`is_supported`].
pub fn run(cpu: &mut Cpu, word: u32) {
    let op = parse(word).expect("an unsupported floating-point word reached the executor");
    match op {
        Op::Unary { kind, p, rd, rn } => {
            let a = read(cpu, p, rn);
            let r = at!(p, T => unary(kind, T::from_u64(a)).to_u64());
            write(cpu, rd, r);
        }
        Op::Convert { p, rd, rn } => {
            let a = read(cpu, p, rn);
            let r = match p {
                Precision::Double => {
                    let d = f64::from_bits(a);
                    // Rounding to single keeps a NaN's class; quiet it like Arm does.
                    // A NaN keeps its sign and payload (truncated) and is quietened.
                    let sign = ((a >> 63) as u32) << 31;
                    u64::from(if d.is_nan() {
                        sign | 0x7fc0_0000 | ((a >> 29) as u32 & 0x3f_ffff)
                    } else {
                        (d as f32).to_bits()
                    })
                }
                Precision::Single => {
                    let s = f32::from_bits(a as u32);
                    if s.is_nan() {
                        f64::from_bits(
                            (((a >> 31) & 1) << 63)
                                | 0x7ff8_0000_0000_0000
                                | ((a & 0x3f_ffff) << 29),
                        )
                        .to_bits()
                    } else {
                        f64::from(s).to_bits()
                    }
                }
            };
            write(cpu, rd, r);
        }
        Op::Binary {
            kind,
            p,
            rd,
            rn,
            rm,
        } => {
            let (a, b) = (read(cpu, p, rn), read(cpu, p, rm));
            let r = at!(p, T => binary(kind, T::from_u64(a), T::from_u64(b)).to_u64());
            write(cpu, rd, r);
        }
        Op::Fused {
            kind,
            p,
            rd,
            rn,
            rm,
            ra,
        } => {
            let (n, m, a) = (read(cpu, p, rn), read(cpu, p, rm), read(cpu, p, ra));
            let r =
                at!(p, T => fused(kind, T::from_u64(n), T::from_u64(m), T::from_u64(a)).to_u64());
            write(cpu, rd, r);
        }
        Op::Compare { p, rn, rm } => {
            let a = read(cpu, p, rn);
            let b = rm.map_or(0, |rm| read(cpu, p, rm));
            let nzcv = at!(p, T => compare(T::from_u64(a), T::from_u64(b)));
            cpu.flags = (cpu.flags & 0x0fff_ffff) | nzcv;
        }
        Op::CondCompare {
            p,
            rn,
            rm,
            cond,
            nzcv,
        } => {
            let flags = if condition_holds(cond, cpu.flags) {
                let (a, b) = (read(cpu, p, rn), read(cpu, p, rm));
                at!(p, T => compare(T::from_u64(a), T::from_u64(b)))
            } else {
                nzcv << 28
            };
            cpu.flags = (cpu.flags & 0x0fff_ffff) | flags;
        }
        Op::Select {
            p,
            rd,
            rn,
            rm,
            cond,
        } => {
            let source = if condition_holds(cond, cpu.flags) {
                rn
            } else {
                rm
            };
            let bits = read(cpu, p, source);
            write(cpu, rd, bits);
        }
        Op::Immediate { p, rd, imm8 } => {
            write(cpu, rd, expand_immediate(imm8, p == Precision::Double));
        }
        Op::IntToFloat {
            signed,
            p,
            wide,
            rd,
            rn,
        } => {
            let raw = if rn == 31 { 0 } else { cpu.x[usize::from(rn)] };
            let r = int_to_float(raw, signed, wide, p);
            write(cpu, rd, r);
        }
        Op::FloatToInt {
            signed,
            p,
            wide,
            rd,
            rn,
        } => {
            let a = read(cpu, p, rn);
            let r = float_to_int(a, signed, wide, p);
            if rd != 31 {
                cpu.x[usize::from(rd)] = r;
            }
        }
        Op::ScalarIntToFloat { signed, p, rd, rn } => {
            let raw = cpu.v[usize::from(rn)][0];
            let r = int_to_float(raw, signed, p == Precision::Double, p);
            write(cpu, rd, r);
        }
        Op::ScalarFloatToInt { signed, p, rd, rn } => {
            let a = read(cpu, p, rn);
            let r = float_to_int(a, signed, p == Precision::Double, p);
            write(cpu, rd, r);
        }
        Op::VectorSign {
            negate,
            double,
            q,
            rd,
            rn,
        } => {
            let sign = if double {
                1 << 63
            } else {
                0x8000_0000_8000_0000
            };
            let source = cpu.v[usize::from(rn)];
            let apply = |half: u64| if negate { half ^ sign } else { half & !sign };
            cpu.v[usize::from(rd)] = [apply(source[0]), if q { apply(source[1]) } else { 0 }];
        }
    }
}

/// The float of precision `p` nearest to the `wide` (64-bit) or 32-bit integer in
/// `raw`, read as signed or unsigned. Integer-to-float casts round to nearest even.
fn int_to_float(raw: u64, signed: bool, wide: bool, p: Precision) -> u64 {
    let value: i128 = match (signed, wide) {
        (true, true) => i128::from(raw as i64),
        (true, false) => i128::from(raw as u32 as i32),
        (false, true) => i128::from(raw),
        (false, false) => i128::from(raw as u32),
    };
    match p {
        Precision::Single => u64::from((value as f32).to_bits()),
        Precision::Double => (value as f64).to_bits(),
    }
}

/// Truncate the float in `bits` toward zero into a 32- or 64-bit integer, saturating
/// out-of-range values and turning NaN into zero (which Rust's `as` already does).
fn float_to_int(bits: u64, signed: bool, wide: bool, p: Precision) -> u64 {
    let value = match p {
        Precision::Single => f64::from(f32::from_bits(bits as u32)),
        Precision::Double => f64::from_bits(bits),
    };
    match (signed, wide) {
        (true, true) => (value as i64) as u64,
        (true, false) => u64::from(value as i32 as u32),
        (false, true) => value as u64,
        (false, false) => u64::from(value as u32),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu() -> Cpu {
        Cpu::default()
    }
    fn set64(cpu: &mut Cpu, reg: usize, value: f64) {
        cpu.v[reg] = [value.to_bits(), u64::MAX];
    }
    fn set32(cpu: &mut Cpu, reg: usize, value: f32) {
        cpu.v[reg] = [u64::from(value.to_bits()), u64::MAX];
    }
    fn get64(cpu: &Cpu, reg: usize) -> f64 {
        f64::from_bits(cpu.v[reg][0])
    }
    fn get32(cpu: &Cpu, reg: usize) -> f32 {
        f32::from_bits(cpu.v[reg][0] as u32)
    }

    #[test]
    fn arithmetic_rounds_in_the_register_precision_and_clears_the_upper_half() {
        let mut c = cpu();
        set32(&mut c, 4, 16_777_216.0);
        set32(&mut c, 5, 1.0);
        // fadd s3, s4, s5: 2^24 + 1 is not representable in single precision.
        run(&mut c, 0x1e252883);
        assert_eq!(get32(&c, 3), 16_777_216.0);
        assert_eq!(c.v[3][1], 0, "scalar writes clear bits 127:64");
        set64(&mut c, 1, 1.5);
        set64(&mut c, 2, 4.0);
        run(&mut c, 0x1e620820); // fmul d0, d1, d2
        assert_eq!(get64(&c, 0), 6.0);
        set64(&mut c, 7, 1.0);
        set64(&mut c, 8, 4.0);
        run(&mut c, 0x1e6838e6); // fsub d6, d7, d8
        assert_eq!(get64(&c, 6), -3.0);
        set32(&mut c, 10, 1.0);
        set32(&mut c, 11, 4.0);
        run(&mut c, 0x1e2b1949); // fdiv s9, s10, s11
        assert_eq!(get32(&c, 9), 0.25);
    }

    #[test]
    fn invalid_operations_give_the_default_nan_and_nan_operands_propagate_quietened() {
        let mut c = cpu();
        set64(&mut c, 1, 0.0);
        set64(&mut c, 2, 0.0);
        run(&mut c, 0x1e621820); // fdiv d0, d1, d2 = 0/0
        assert_eq!(c.v[0][0], 0x7ff8_0000_0000_0000, "positive default NaN");
        // A signalling NaN operand comes back quietened.
        c.v[1][0] = 0x7ff0_0000_0000_0001;
        set64(&mut c, 2, 1.0);
        run(&mut c, 0x1e620820); // fmul
        assert_eq!(c.v[0][0], 0x7ff8_0000_0000_0001);
    }

    #[test]
    fn max_min_follow_the_zero_and_nan_rules() {
        let mut c = cpu();
        let (zero, neg_zero) = (0.0f64, -0.0f64);
        set64(&mut c, 1, zero);
        set64(&mut c, 2, neg_zero);
        run(&mut c, 0x1e624820); // fmax d0, d1, d2
        assert!(!get64(&c, 0).is_sign_negative(), "fmax(+0, -0) = +0");
        // fmin s0, s1, s2 on (+0, -0): -0.
        set32(&mut c, 1, 0.0);
        set32(&mut c, 2, -0.0);
        run(&mut c, 0x1e225820);
        assert!(get32(&c, 0).is_sign_negative());
        // fmax with a NaN is NaN; fmaxnm prefers the number.
        set64(&mut c, 1, f64::NAN);
        set64(&mut c, 2, 3.0);
        run(&mut c, 0x1e624820);
        assert!(get64(&c, 0).is_nan());
        run(&mut c, 0x1e626820); // fmaxnm d0, d1, d2
        assert_eq!(get64(&c, 0), 3.0);
        // fminnm s0, s1, s2 with a NaN in the second operand.
        set32(&mut c, 1, 2.0);
        set32(&mut c, 2, f32::NAN);
        run(&mut c, 0x1e227820);
        assert_eq!(get32(&c, 0), 2.0);
    }

    #[test]
    fn fused_multiply_add_rounds_once() {
        let mut c = cpu();
        let x = 1.0 + 2f64.powi(-30);
        set64(&mut c, 1, x);
        set64(&mut c, 2, x);
        set64(&mut c, 3, -(1.0 + 2f64.powi(-29)));
        run(&mut c, 0x1f420c20); // fmadd d0, d1, d2, d3 = d3 + d1*d2
        assert_eq!(
            get64(&c, 0),
            2f64.powi(-60),
            "an unfused multiply-add gives 0"
        );
        // fmsub s0, s1, s2, s3 = s3 - s1*s2.
        set32(&mut c, 1, 2.0);
        set32(&mut c, 2, 3.0);
        set32(&mut c, 3, 10.0);
        run(&mut c, 0x1f028c20);
        assert_eq!(get32(&c, 0), 4.0);
        // fnmadd d0 = -d3 - d1*d2; fnmsub s0 = s1*s2 - s3.
        set64(&mut c, 1, 2.0);
        set64(&mut c, 2, 3.0);
        set64(&mut c, 3, 1.0);
        run(&mut c, 0x1f620c20);
        assert_eq!(get64(&c, 0), -7.0);
        set32(&mut c, 1, 2.0);
        set32(&mut c, 2, 3.0);
        set32(&mut c, 3, 1.0);
        run(&mut c, 0x1f228c20);
        assert_eq!(get32(&c, 0), 5.0);
    }

    #[test]
    fn compares_set_the_arm_flags_and_conditional_forms_select_by_them() {
        let mut c = cpu();
        c.flags = 0x0abc_def0;
        let cases = [
            (1.0, 2.0, 0x8000_0000),
            (2.0, 2.0, 0x6000_0000),
            (3.0, 2.0, 0x2000_0000),
            (f64::NAN, 2.0, 0x3000_0000),
        ];
        for (a, b, nzcv) in cases {
            set64(&mut c, 0, a);
            set64(&mut c, 1, b);
            run(&mut c, 0x1e612000); // fcmp d0, d1
            assert_eq!(c.flags, 0x0abc_def0 & 0x0fff_ffff | nzcv, "{a} vs {b}");
        }
        // fcmp s0, #0.0 ignores Rm.
        set32(&mut c, 0, 0.0);
        run(&mut c, 0x1e202008);
        assert_eq!(c.flags >> 28, 0x6);
        // fccmp d0, d1, #4, eq: with Z set it compares, otherwise it loads #4.
        set64(&mut c, 0, 1.0);
        set64(&mut c, 1, 2.0);
        c.flags = 0x4000_0000;
        run(&mut c, 0x1e610404);
        assert_eq!(c.flags >> 28, 0x8);
        c.flags = 0;
        run(&mut c, 0x1e610404);
        assert_eq!(c.flags >> 28, 0x4);
        // fcsel d0, d1, d2, ne.
        set64(&mut c, 1, 7.0);
        set64(&mut c, 2, 9.0);
        c.flags = 0;
        run(&mut c, 0x1e621c20);
        assert_eq!(get64(&c, 0), 7.0);
        c.flags = 0x4000_0000;
        run(&mut c, 0x1e621c20);
        assert_eq!(get64(&c, 0), 9.0);
    }

    #[test]
    fn integer_conversions_saturate_truncate_and_round_to_nearest_even() {
        let mut c = cpu();
        // fcvtzs x0, d1 truncates toward zero, saturates, and sends NaN to zero.
        for (input, expected) in [
            (-2.9, (-2i64) as u64),
            (1e30, i64::MAX as u64),
            (-1e30, i64::MIN as u64),
            (f64::NAN, 0),
        ] {
            set64(&mut c, 1, input);
            run(&mut c, 0x9e780020);
            assert_eq!(c.x[0], expected, "{input}");
        }
        // fcvtzu w0, d1: negative inputs give 0; the result is zero-extended.
        set64(&mut c, 1, -5.0);
        run(&mut c, 0x1e790020);
        assert_eq!(c.x[0], 0);
        set64(&mut c, 1, 5e9);
        run(&mut c, 0x1e790020);
        assert_eq!(c.x[0], u64::from(u32::MAX));
        // fcvtzs w0, s1 of a negative value is a 32-bit two's complement, zero-extended.
        set32(&mut c, 1, -1.5);
        run(&mut c, 0x1e380020);
        assert_eq!(c.x[0], 0xffff_ffff);
        // ucvtf s0, x1: 2^24 + 1 rounds to even (2^24); scvtf d0, x1 of a negative.
        c.x[1] = (1 << 24) + 1;
        run(&mut c, 0x9e230020);
        assert_eq!(get32(&c, 0), 16_777_216.0);
        c.x[1] = (-3i64) as u64;
        run(&mut c, 0x9e620020);
        assert_eq!(get64(&c, 0), -3.0);
        // ucvtf d0, w1 reads only the low 32 bits.
        c.x[1] = 0xffff_ffff_0000_0007;
        run(&mut c, 0x1e630020);
        assert_eq!(get64(&c, 0), 7.0);
        // The SIMD&FP-register forms: ucvtf s0, s1 and fcvtzs s0, s1.
        c.v[1] = [0x8000_0000, 0];
        run(&mut c, 0x7e21d820);
        assert_eq!(get32(&c, 0), 2_147_483_648.0);
        set32(&mut c, 1, -7.9);
        run(&mut c, 0x5ea1b820);
        assert_eq!(c.v[0], [(-7i32) as u32 as u64, 0]);
    }

    #[test]
    fn precision_conversion_immediates_and_unary_forms() {
        let mut c = cpu();
        set64(&mut c, 1, 0.1);
        run(&mut c, 0x1e624020); // fcvt s0, d1
        assert_eq!(get32(&c, 0), 0.1f32);
        set32(&mut c, 1, 0.5);
        run(&mut c, 0x1e22c020); // fcvt d0, s1
        assert_eq!(get64(&c, 0), 0.5);
        run(&mut c, 0x1e6e1000); // fmov d0, #1.0
        assert_eq!(get64(&c, 0), 1.0);
        run(&mut c, 0x1e309001); // fmov s1, #-2.5
        assert_eq!(get32(&c, 1), -2.5);
        set64(&mut c, 1, -16.0);
        run(&mut c, 0x1e614020); // fneg d0, d1
        assert_eq!(get64(&c, 0), 16.0);
        set64(&mut c, 1, 16.0);
        run(&mut c, 0x1e61c020); // fsqrt d0, d1
        assert_eq!(get64(&c, 0), 4.0);
        set32(&mut c, 1, -3.0);
        run(&mut c, 0x1e20c020); // fabs s0, s1
        assert_eq!(get32(&c, 0), 3.0);
        // sqrt of a negative is the default NaN.
        set64(&mut c, 1, -1.0);
        run(&mut c, 0x1e61c020);
        assert_eq!(c.v[0][0], 0x7ff8_0000_0000_0000);
    }

    #[test]
    fn vector_fneg_and_fabs_flip_every_lane_and_q_clear_zeroes_the_upper_half() {
        let mut c = cpu();
        c.v[1] = [0x3ff0_0000_0000_0000, 0xc000_0000_0000_0000];
        run(&mut c, 0x6ee0f820); // fneg v0.2d, v1.2d
        assert_eq!(c.v[0], [0xbff0_0000_0000_0000, 0x4000_0000_0000_0000]);
        c.v[1] = [0xbf80_0000_c000_0000, 0x3f80_0000_3f80_0000];
        run(&mut c, 0x4ea0f820); // fabs v0.4s, v1.4s
        assert_eq!(c.v[0], [0x3f80_0000_4000_0000, 0x3f80_0000_3f80_0000]);
        run(&mut c, 0x2ea0f820); // fneg v0.2s, v1.2s
        assert_eq!(c.v[0], [0x3f80_0000_4000_0000, 0]);
    }

    #[test]
    fn unsupported_encodings_are_not_claimed() {
        // fmov d0, x1 and fmov x0, d1 are `SimdFmov`, not conversions.
        assert!(!is_supported(0x9e670020));
        assert!(!is_supported(0x9e660020));
        // Half precision (type 3) and reserved type 2.
        assert!(!is_supported(0x1ee22820));
        assert!(!is_supported(0x1ea22820));
        // fneg v0.1d is reserved.
        assert!(!is_supported(0x2ee0f820));
        assert!(is_supported(0x1e620820));
    }

    #[test]
    fn the_immediate_expansion_matches_the_architectural_pattern() {
        assert_eq!(expand_immediate(0x70, false), 0x3f80_0000);
        assert_eq!(expand_immediate(0x70, true), 0x3ff0_0000_0000_0000);
        assert_eq!(expand_immediate(0xf0, true), 0xbff0_0000_0000_0000);
    }
}
