//! Integer Advanced SIMD instructions the structured decoder has no form for: the
//! vector permutes (`uzp`, `zip`, `trn`), the two-register-miscellaneous group
//! (`rev`, `cnt`, `mvn`, compares against zero, `abs`/`neg`, the narrowing `xtn`
//! family and the pairwise-long adds), the three-registers-different long and wide
//! arithmetic (`uaddl`, `umull`, ...), immediate shifts, and the 64-bit scalar forms.
//!
//! [`decode`] classifies a word; executing it is the emulator's business.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Permute {
    Uzp1,
    Uzp2,
    Zip1,
    Zip2,
    Trn1,
    Trn2,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Zero {
    Gt,
    Ge,
    Le,
    Lt,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Narrow {
    Plain,
    SignedSat,
    UnsignedSat,
    SignedToUnsignedSat,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Long {
    Add,
    AddWide,
    Sub,
    SubWide,
    Abd,
    Mul,
    Mla,
    Mls,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scalar {
    Add,
    Sub,
    Gt,
    Hi,
    Ge,
    Hs,
    Eq,
    Tst,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Permute {
        kind: Permute,
        size: u32,
        q: bool,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// `rev16`/`rev32`/`rev64`: reverse the lanes inside each group of `group` bits.
    Reverse {
        group: u32,
        size: u32,
        q: bool,
        rd: u8,
        rn: u8,
    },
    Count {
        q: bool,
        rd: u8,
        rn: u8,
    },
    Not {
        q: bool,
        rd: u8,
        rn: u8,
    },
    ReverseBits {
        q: bool,
        rd: u8,
        rn: u8,
    },
    CompareZero {
        kind: Zero,
        size: u32,
        q: bool,
        rd: u8,
        rn: u8,
    },
    Abs {
        negate: bool,
        size: u32,
        q: bool,
        rd: u8,
        rn: u8,
    },
    /// `xtn`/`sqxtn`/`uqxtn`/`sqxtun`; `upper` is the `2` form (keeps the low half).
    Narrow {
        kind: Narrow,
        size: u32,
        upper: bool,
        rd: u8,
        rn: u8,
    },
    /// `saddlp`/`uaddlp` and, with `accumulate`, `sadalp`/`uadalp`.
    PairLong {
        signed: bool,
        accumulate: bool,
        size: u32,
        q: bool,
        rd: u8,
        rn: u8,
    },
    Long {
        kind: Long,
        signed: bool,
        size: u32,
        upper: bool,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    ScalarAddp {
        rd: u8,
        rn: u8,
    },
    ScalarDup {
        size: u32,
        index: u32,
        rd: u8,
        rn: u8,
    },
    ScalarThree {
        kind: Scalar,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// `shl`, and `sshr`/`ushr`, by an immediate.
    ShiftImm {
        right: bool,
        signed: bool,
        size: u32,
        q: bool,
        shift: u32,
        rd: u8,
        rn: u8,
    },
    /// `sshll`/`ushll` (`sxtl`/`uxtl` at shift 0); `upper` is the `2` form.
    ShiftLong {
        signed: bool,
        size: u32,
        shift: u32,
        upper: bool,
        rd: u8,
        rn: u8,
    },
    /// `shrn`/`shrn2`.
    ShiftNarrow {
        size: u32,
        shift: u32,
        upper: bool,
        rd: u8,
        rn: u8,
    },
}

/// Decode `word` as one of these SIMD instructions, or `None` when it is not one.
pub fn decode(word: u32) -> Option<Op> {
    let rd = (word & 31) as u8;
    let rn = ((word >> 5) & 31) as u8;
    let rm = ((word >> 16) & 31) as u8;
    let q = word & 0x4000_0000 != 0;
    let u = word & 0x2000_0000 != 0;
    let size = (word >> 22) & 3;
    // Permute: bit 21 clear, bits 11:10 = 10, bits 14:12 pick the operation.
    if word & 0xbf20_8c00 == 0x0e00_0800 {
        let kind = match (word >> 12) & 7 {
            1 => Permute::Uzp1,
            2 => Permute::Trn1,
            3 => Permute::Zip1,
            5 => Permute::Uzp2,
            6 => Permute::Trn2,
            7 => Permute::Zip2,
            _ => return None,
        };
        // 64-bit lanes need the full vector.
        if size == 3 && !q {
            return None;
        }
        return Some(Op::Permute {
            kind,
            size,
            q,
            rd,
            rn,
            rm,
        });
    }
    // Two-register miscellaneous: bits 21:17 = 10000, bits 11:10 = 10.
    if word & 0x9f3e_0c00 == 0x0e20_0800 {
        let opcode = (word >> 12) & 0x1f;
        let wide_needs_q = size != 3 || q;
        return match (u, opcode) {
            (false, 0) if size < 3 => Some(Op::Reverse {
                group: 64,
                size,
                q,
                rd,
                rn,
            }),
            (true, 0) if size < 2 => Some(Op::Reverse {
                group: 32,
                size,
                q,
                rd,
                rn,
            }),
            (false, 1) if size == 0 => Some(Op::Reverse {
                group: 16,
                size,
                q,
                rd,
                rn,
            }),
            (false | true, 2) if size < 3 => Some(Op::PairLong {
                signed: !u,
                accumulate: false,
                size,
                q,
                rd,
                rn,
            }),
            (false | true, 6) if size < 3 => Some(Op::PairLong {
                signed: !u,
                accumulate: true,
                size,
                q,
                rd,
                rn,
            }),
            (false, 5) if size == 0 => Some(Op::Count { q, rd, rn }),
            (true, 5) if size == 0 => Some(Op::Not { q, rd, rn }),
            (true, 5) if size == 1 => Some(Op::ReverseBits { q, rd, rn }),
            (false, 8) if wide_needs_q => Some(Op::CompareZero {
                kind: Zero::Gt,
                size,
                q,
                rd,
                rn,
            }),
            (true, 8) if wide_needs_q => Some(Op::CompareZero {
                kind: Zero::Ge,
                size,
                q,
                rd,
                rn,
            }),
            (true, 9) if wide_needs_q => Some(Op::CompareZero {
                kind: Zero::Le,
                size,
                q,
                rd,
                rn,
            }),
            (false, 10) if wide_needs_q => Some(Op::CompareZero {
                kind: Zero::Lt,
                size,
                q,
                rd,
                rn,
            }),
            (false, 11) if wide_needs_q => Some(Op::Abs {
                negate: false,
                size,
                q,
                rd,
                rn,
            }),
            (true, 11) if wide_needs_q => Some(Op::Abs {
                negate: true,
                size,
                q,
                rd,
                rn,
            }),
            (false, 0x12) if size < 3 => Some(Op::Narrow {
                kind: Narrow::Plain,
                size,
                upper: q,
                rd,
                rn,
            }),
            (true, 0x12) if size < 3 => Some(Op::Narrow {
                kind: Narrow::SignedToUnsignedSat,
                size,
                upper: q,
                rd,
                rn,
            }),
            (false, 0x14) if size < 3 => Some(Op::Narrow {
                kind: Narrow::SignedSat,
                size,
                upper: q,
                rd,
                rn,
            }),
            (true, 0x14) if size < 3 => Some(Op::Narrow {
                kind: Narrow::UnsignedSat,
                size,
                upper: q,
                rd,
                rn,
            }),
            // `cmeq #0` (U=0, opcode 9) is `SimdCompareZero`.
            _ => None,
        };
    }
    // Three registers different: bit 21 set, bits 11:10 = 00.
    if word & 0x9f20_0c00 == 0x0e20_0000 && size < 3 {
        let kind = match (word >> 12) & 15 {
            0 => Long::Add,
            1 => Long::AddWide,
            2 => Long::Sub,
            3 => Long::SubWide,
            7 => Long::Abd,
            8 => Long::Mla,
            10 => Long::Mls,
            12 => Long::Mul,
            _ => return None,
        };
        return Some(Op::Long {
            kind,
            signed: !u,
            size,
            upper: q,
            rd,
            rn,
            rm,
        });
    }
    // Shift by immediate: `immh` (bits 22:19) gives the element size and, with
    // `immb`, the shift. `immh == 0` is the modified-immediate class instead.
    if word & 0x9f80_0400 == 0x0f00_0400 && (word >> 19) & 15 != 0 {
        let immh = (word >> 19) & 15;
        let immhb = (word >> 16) & 0x7f;
        // The position of the top set bit of immh: 0 for bytes up to 3 for doublewords.
        let size = 31 - immh.leading_zeros();
        let esize = 8 << size;
        return match (u, (word >> 11) & 0x1f) {
            (false, 0b01010) if size < 3 || q => Some(Op::ShiftImm {
                right: false,
                signed: false,
                size,
                q,
                shift: immhb - esize,
                rd,
                rn,
            }),
            (_, 0b00000) if size < 3 || q => Some(Op::ShiftImm {
                right: true,
                signed: !u,
                size,
                q,
                shift: 2 * esize - immhb,
                rd,
                rn,
            }),
            (_, 0b10100) if size < 3 => Some(Op::ShiftLong {
                signed: !u,
                size,
                shift: immhb - esize,
                upper: q,
                rd,
                rn,
            }),
            (false, 0b10000) if size < 3 => Some(Op::ShiftNarrow {
                size,
                shift: 2 * esize - immhb,
                upper: q,
                rd,
                rn,
            }),
            _ => None,
        };
    }
    // Scalar pairwise `addp Dd, Vn.2D`.
    if word & 0xffff_fc00 == 0x5ef1_b800 {
        return Some(Op::ScalarAddp { rd, rn });
    }
    // Scalar `dup`: `mov Bd/Hd/Sd/Dd, Vn.T[index]`.
    if word & 0xffe0_fc00 == 0x5e00_0400 {
        let imm5 = (word >> 16) & 0x1f;
        if imm5 == 0 || imm5.trailing_zeros() > 3 {
            return None;
        }
        let size = imm5.trailing_zeros();
        return Some(Op::ScalarDup {
            size,
            index: imm5 >> (size + 1),
            rd,
            rn,
        });
    }
    // Scalar three-same, 64-bit only.
    if word & 0xdf20_0400 == 0x5e20_0400 && size == 3 {
        let kind = match (u, (word >> 11) & 0x1f) {
            (false, 0x10) => Scalar::Add,
            (true, 0x10) => Scalar::Sub,
            (false, 0x06) => Scalar::Gt,
            (true, 0x06) => Scalar::Hi,
            (false, 0x07) => Scalar::Ge,
            (true, 0x07) => Scalar::Hs,
            (true, 0x11) => Scalar::Eq,
            (false, 0x11) => Scalar::Tst,
            _ => return None,
        };
        return Some(Op::ScalarThree { kind, rd, rn, rm });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classification_declines_what_it_does_not_implement() {
        // cmeq #0 is `SimdCompareZero`; half-width lanes need Q for 64-bit lanes;
        // `ins`, `umov` and the three-same group belong to other decoders.
        assert_eq!(decode(0x0e209820), None);
        assert_eq!(decode(0x0ee0b820), None, "abs .1d is reserved");
        assert_eq!(decode(0x2ee21c20), None, "bif is three-same");
        assert_eq!(decode(0x4e083c20), None, "umov");
        assert!(decode(0x4e821820).is_some());
    }
}
