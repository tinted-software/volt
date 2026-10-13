//! Scalar floating point (single and double precision) and the vector `fabs`/`fneg`
//! forms: the decoded shape of every word [`decode`] recognises.
//!
//! Half precision is not decoded here (`type` 3), nor is anything with a reserved
//! `type`; those words are not recognised.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    Single,
    Double,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unary {
    Move,
    Abs,
    Neg,
    Sqrt,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Binary {
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
pub enum Fused {
    Madd,
    Msub,
    Nmadd,
    Nmsub,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
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
    /// `fcmp`/`fcmpe`; no `rm` compares with zero. `signaling` is `fcmpe`.
    Compare {
        p: Precision,
        rn: u8,
        rm: Option<u8>,
        signaling: bool,
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

/// Decode `word` as a floating-point instruction, or `None` when it is not one.
pub fn decode(word: u32) -> Option<Op> {
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
            signaling: word & 0x10 != 0,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_encodings_are_not_claimed() {
        // fmov d0, x1 and fmov x0, d1 are `SimdFmov`, not conversions.
        assert_eq!(decode(0x9e670020), None);
        assert_eq!(decode(0x9e660020), None);
        // Half precision (type 3) and reserved type 2.
        assert_eq!(decode(0x1ee22820), None);
        assert_eq!(decode(0x1ea22820), None);
        // fneg v0.1d is reserved.
        assert_eq!(decode(0x2ee0f820), None);
        assert!(decode(0x1e620820).is_some());
    }

    #[test]
    fn compare_signaling() {
        let cmp = |word| match decode(word) {
            Some(Op::Compare { rm, signaling, .. }) => (rm, signaling),
            other => panic!("{other:?}"),
        };
        assert_eq!(cmp(0x1e222020), (Some(2), false)); // fcmp s1, s2
        assert_eq!(cmp(0x1e222030), (Some(2), true)); // fcmpe s1, s2
        assert_eq!(cmp(0x1e602008), (None, false)); // fcmp d0, #0.0
        assert_eq!(cmp(0x1e202038), (None, true)); // fcmpe s1, #0.0
    }
}
