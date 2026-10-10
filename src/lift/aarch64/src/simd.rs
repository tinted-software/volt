//! Register-only SIMD semantics: immediates, moves, lane insert/extract/dup,
//! `ext`, compare-against-zero, table lookup, `fmov` and the integer ALU.
//!
//! Every function works on the 128-bit vector registers in guest state as two
//! little-endian doublewords at [`Builder::v_at`].
use crate::Builder;
use alloc::vec::Vec;
use volt_ir::function::{BinOp as B, CmpOp as C, Value};
use volt_isa_aarch64::decode::*;
pub(crate) fn imm(b: &Builder, a: SimdImmediate) {
    let at = b.v_at(a.rd);
    match a.combine {
        ImmCombine::Set | ImmCombine::SetInverted => {
            b.store(at, b.k(b.i64, a.low));
            b.store(at + 8, b.k(b.i64, a.high));
        }
        ImmCombine::Or | ImmCombine::AndNot => {
            // A 64-bit form (`q == false`) also clears the upper half.
            let (op, imm) = match a.combine {
                ImmCombine::Or => (B::BitOr, a.low),
                _ => (B::BitAnd, !a.low),
            };
            b.store(at, b.imm(b.i64, op, b.load(b.i64, at), imm));
            b.store(
                at + 8,
                if a.q {
                    b.imm(b.i64, op, b.load(b.i64, at + 8), imm)
                } else {
                    b.k(b.i64, 0)
                },
            );
        }
    }
}
pub(crate) fn insert(b: &Builder, a: SimdInsert) {
    // Replace one lane of the destination with the low bits of a GPR.
    let ebits = (1u64 << a.esize) * 8;
    let emask = if ebits == 64 {
        u64::MAX
    } else {
        (1u64 << ebits) - 1
    };
    let at_d = b.v_at(a.rd);
    let per_dw = (64 / ebits) as usize;
    let word = a.index as usize / per_dw;
    let shift = (a.index as usize % per_dw) as u64 * ebits;
    let old = b.load(b.i64, at_d + word * 8);
    let cleared = b.bin(b.i64, B::BitAnd, old, b.k(b.i64, !(emask << shift)));
    let value = b.bin(
        b.i64,
        B::BitAnd,
        b.reg(b.i64, a.rn, true),
        b.k(b.i64, emask),
    );
    b.store(
        at_d + word * 8,
        b.bin(b.i64, B::BitOr, cleared, b.imm(b.i64, B::Shl, value, shift)),
    );
}
pub(crate) fn insert_element(b: &Builder, a: SimdInsertElement) {
    // Copy one lane of Vn over one lane of Vd. The source is read before
    // the store, so `rd == rn` is fine.
    let ebits = (1u64 << a.esize) * 8;
    let emask = if ebits == 64 {
        u64::MAX
    } else {
        (1u64 << ebits) - 1
    };
    let per_dw = (64 / ebits) as usize;
    let at_d = b.v_at(a.rd);
    let at_n = b.v_at(a.rn);
    let (src_word, src_shift) = (
        a.src as usize / per_dw,
        (a.src as usize % per_dw) as u64 * ebits,
    );
    let (dst_word, dst_shift) = (
        a.dst as usize / per_dw,
        (a.dst as usize % per_dw) as u64 * ebits,
    );
    let source = b.load(b.i64, at_n + src_word * 8);
    let value = b.imm(
        b.i64,
        B::BitAnd,
        b.imm(b.i64, B::Shr, source, src_shift),
        emask,
    );
    let old = b.load(b.i64, at_d + dst_word * 8);
    let cleared = b.imm(b.i64, B::BitAnd, old, !(emask << dst_shift));
    b.store(
        at_d + dst_word * 8,
        b.bin(
            b.i64,
            B::BitOr,
            cleared,
            b.imm(b.i64, B::Shl, value, dst_shift),
        ),
    );
}
pub(crate) fn extract(b: &Builder, a: SimdExtract) {
    // Move one lane into the low bits of a GPR, zero-extended.
    let ebits = (1u64 << a.esize) * 8;
    let emask = if ebits == 64 {
        u64::MAX
    } else {
        (1u64 << ebits) - 1
    };
    let at_n = b.v_at(a.rn);
    let per_dw = (64 / ebits) as usize;
    let word = a.index as usize / per_dw;
    let shift = (a.index as usize % per_dw) as u64 * ebits;
    let source = b.load(b.i64, at_n + word * 8);
    let value = b.imm(b.i64, B::BitAnd, b.imm(b.i64, B::Shr, source, shift), emask);
    b.put(a.rd, value);
}
pub(crate) fn dup_element(b: &Builder, a: SimdDupElement) {
    let ebits = (1u64 << a.esize) * 8;
    let emask = if ebits == 64 {
        u64::MAX
    } else {
        (1u64 << ebits) - 1
    };
    let at_n = b.v_at(a.rn);
    let vn_lo = b.load(b.i64, at_n);
    let vn_hi = b.load(b.i64, at_n + 8);
    // Locate the requested element across the 128-bit source.
    let lane = a.index as usize;
    let lanes_per_dw = (64 / ebits) as usize;
    let src = if lane / lanes_per_dw == 0 {
        vn_lo
    } else {
        vn_hi
    };
    let shift = (lane % lanes_per_dw) as u64 * ebits;
    let element = b.imm(b.i64, B::BitAnd, b.imm(b.i64, B::Shr, src, shift), emask);
    // Replicate the element across the destination lanes.
    let mut acc = b.k(b.i64, 0);
    let lanes = (64 / ebits) as usize;
    for lane in 0..lanes {
        acc = b.bin(
            b.i64,
            B::BitOr,
            acc,
            b.imm(b.i64, B::Shl, element, lane as u64 * ebits),
        );
    }
    let at_d = b.v_at(a.rd);
    b.store(at_d, acc);
    b.store(at_d + 8, if a.q { acc } else { b.k(b.i64, 0) });
}
pub(crate) fn dup(b: &Builder, rd: u8, rn: u8, esize: u8, q: bool) {
    let r = b.reg(b.i64, rn, true);
    let val64 = match esize {
        0 => {
            let byte = b.imm(b.i64, B::BitAnd, r, 0xff);
            b.bin(b.i64, B::Mul, byte, b.k(b.i64, 0x0101_0101_0101_0101))
        }
        1 => {
            let h = b.imm(b.i64, B::BitAnd, r, 0xffff);
            b.bin(b.i64, B::Mul, h, b.k(b.i64, 0x0001_0001_0001_0001))
        }
        2 => {
            let w = b.imm(b.i64, B::BitAnd, r, 0xffff_ffff);
            b.bin(b.i64, B::BitOr, w, b.imm(b.i64, B::Shl, w, 32))
        }
        _ => r,
    };
    let at = b.v_at(rd);
    b.store(at, val64);
    let hi = if q { val64 } else { b.k(b.i64, 0) };
    b.store(at + 8, hi);
}
pub(crate) fn ext(b: &Builder, a: SimdExt) {
    let at_d = b.v_at(a.rd);
    let at_n = b.v_at(a.rn);
    let at_m = b.v_at(a.rm);
    let vn_lo = b.load(b.i64, at_n);
    let vn_hi = b.load(b.i64, at_n + 8);
    let vm_lo = b.load(b.i64, at_m);
    let vm_hi = b.load(b.i64, at_m + 8);
    let (lo, hi) = if !a.q {
        // `.8b`: a byte window over the low halves of `rm:rn`; the upper half clears.
        let lo = if a.imm == 0 {
            vn_lo
        } else {
            let s = a.imm as u64 * 8;
            b.bin(
                b.i64,
                B::BitOr,
                b.imm(b.i64, B::Shr, vn_lo, s),
                b.imm(b.i64, B::Shl, vm_lo, 64 - s),
            )
        };
        (lo, b.k(b.i64, 0))
    } else if a.imm == 0 {
        (vn_lo, vn_hi)
    } else if a.imm == 8 {
        (vn_hi, vm_lo)
    } else if a.imm < 8 {
        let s = a.imm as u64 * 8;
        let r = (8 - a.imm as u64) * 8;
        let lo = b.bin(
            b.i64,
            B::BitOr,
            b.imm(b.i64, B::Shr, vn_lo, s),
            b.imm(b.i64, B::Shl, vn_hi, r),
        );
        let hi = b.bin(
            b.i64,
            B::BitOr,
            b.imm(b.i64, B::Shr, vn_hi, s),
            b.imm(b.i64, B::Shl, vm_lo, r),
        );
        (lo, hi)
    } else {
        let shift = a.imm as u64 - 8;
        let s = shift * 8;
        let r = (8 - shift) * 8;
        let lo = b.bin(
            b.i64,
            B::BitOr,
            b.imm(b.i64, B::Shr, vn_hi, s),
            b.imm(b.i64, B::Shl, vm_lo, r),
        );
        let hi = b.bin(
            b.i64,
            B::BitOr,
            b.imm(b.i64, B::Shr, vm_lo, s),
            b.imm(b.i64, B::Shl, vm_hi, r),
        );
        (lo, hi)
    };
    b.store(at_d, lo);
    b.store(at_d + 8, hi);
}
pub(crate) fn compare_zero(b: &Builder, a: SimdCompareZero) {
    let at_n = b.v_at(a.rn);
    let at_d = b.v_at(a.rd);
    let vn_lo = b.load(b.i64, at_n);
    let vn_hi = b.load(b.i64, at_n + 8);
    let ebits = (1u64 << a.size) * 8;
    let emask = if ebits == 64 {
        u64::MAX
    } else {
        (1u64 << ebits) - 1
    };
    let halves = if a.q { 2 } else { 1 };
    for half in 0..2 {
        if half >= halves {
            b.store(at_d + half * 8, b.k(b.i64, 0));
            continue;
        }
        let v = if half == 0 { vn_lo } else { vn_hi };
        let mut acc = b.k(b.i64, 0);
        let lanes = 8 >> a.size;
        for lane in 0..lanes {
            let shift = lane as u64 * ebits;
            let field = b.imm(b.i64, B::BitAnd, b.imm(b.i64, B::Shr, v, shift), emask);
            let is_zero = b.cmp(C::Eq, field, b.k(b.i64, 0));
            let bits = b.sel(b.i64, is_zero, b.k(b.i64, emask), b.k(b.i64, 0));
            acc = b.bin(b.i64, B::BitOr, acc, b.imm(b.i64, B::Shl, bits, shift));
        }
        b.store(at_d + half * 8, acc);
    }
}
pub(crate) fn table(b: &Builder, a: SimdTable) {
    // Table bytes live in `len + 1` consecutive registers. Each index
    // byte selects a table byte; out-of-range indices leave zero for
    // `tbl` and keep the destination byte for `tbx`.
    let at_m = b.v_at(a.rm);
    let vm_lo = b.load(b.i64, at_m);
    let vm_hi = b.load(b.i64, at_m + 8);
    let regs = a.len as usize + 1;
    let mut table = Vec::with_capacity(regs * 2);
    for i in 0..regs {
        let reg = (a.rn as usize + i) % 32;
        let at = b.v_at(reg as u8);
        table.push(b.load(b.i64, at));
        table.push(b.load(b.i64, at + 8));
    }
    let at_d = b.v_at(a.rd);
    let old_lo = if a.extend {
        Some(b.load(b.i64, at_d))
    } else {
        None
    };
    let old_hi = if a.extend {
        Some(b.load(b.i64, at_d + 8))
    } else {
        None
    };
    let halves = if a.q { 2 } else { 1 };
    let limit = (16 * regs) as u64;
    for half in 0..2 {
        if half >= halves {
            b.store(at_d + half * 8, b.k(b.i64, 0));
            continue;
        }
        let indices = if half == 0 { vm_lo } else { vm_hi };
        let old = if half == 0 { old_lo } else { old_hi };
        let mut acc = b.k(b.i64, 0);
        for byte in 0..8u64 {
            let shift = byte * 8;
            let index = b.imm(b.i64, B::BitAnd, b.imm(b.i64, B::Shr, indices, shift), 0xff);
            let in_range = b.cmp(C::Ult, index, b.k(b.i64, limit));
            let mut value = b.k(b.i64, 0);
            let sel = |c: Value, x: Value, y: Value| b.sel(b.i64, c, x, y);
            for (slot, entry) in table.iter().enumerate() {
                let here = b.cmp(
                    C::Eq,
                    b.bin(b.i64, B::Shr, index, b.k(b.i64, 3)),
                    b.k(b.i64, slot as u64),
                );
                let byte_at = b.imm(
                    b.i64,
                    B::BitAnd,
                    b.bin(
                        b.i64,
                        B::Shr,
                        *entry,
                        b.bin(
                            b.i64,
                            B::Shl,
                            b.imm(b.i64, B::BitAnd, index, 7),
                            b.k(b.i64, 3),
                        ),
                    ),
                    0xff,
                );
                value = sel(b.bin(b.boolean, B::BitAnd, in_range, here), byte_at, value);
            }
            if a.extend {
                value = sel(
                    in_range,
                    value,
                    b.imm(
                        b.i64,
                        B::BitAnd,
                        b.imm(b.i64, B::Shr, old.unwrap(), shift),
                        0xff,
                    ),
                );
            }
            acc = b.bin(b.i64, B::BitOr, acc, b.imm(b.i64, B::Shl, value, shift));
        }
        b.store(at_d + half * 8, acc);
    }
}
pub(crate) fn fmov(b: &Builder, a: SimdFmov) {
    // Pure bit move; no FP arithmetic involved.
    // `fmov Vd.D[1], Xn` writes only the upper doubleword; `fmov Xd, Vn.D[1]` reads it.
    let half = if a.upper { 8 } else { 0 };
    if a.to_fp {
        let v = b.reg(b.i64, a.rn, true);
        let bits = if a.double {
            v
        } else {
            b.imm(b.i64, B::BitAnd, v, 0xffff_ffff)
        };
        let at = b.v_at(a.rd);
        b.store(at + half, bits);
        if !a.upper {
            b.store(at + 8, b.k(b.i64, 0));
        }
    } else {
        let at = b.v_at(a.rn);
        let v = b.load(b.i64, at + half);
        let bits = if a.double {
            v
        } else {
            b.imm(b.i64, B::BitAnd, v, 0xffff_ffff)
        };
        b.put(a.rd, bits);
    }
}
pub(crate) fn alu(b: &Builder, a: SimdAlu) {
    // Bitwise logical operations work on whole halves, lane width free.
    if matches!(
        a.op,
        SimdAluOp::AddV | SimdAluOp::UMinV | SimdAluOp::UMaxV | SimdAluOp::UAddLV
    ) {
        // Horizontal add, minimum, maximum or widening add: reduce every
        // lane of the source and store the result in lane 0, zeroing the
        // rest. `UAddLV` keeps twice the lane width.
        let ebits = (1u64 << a.size) * 8;
        let mask_of = |bits: u64| {
            if bits >= 64 {
                u64::MAX
            } else {
                (1u64 << bits) - 1
            }
        };
        let emask = mask_of(ebits);
        let result_mask = if a.op == SimdAluOp::UAddLV {
            mask_of(ebits * 2)
        } else {
            emask
        };
        let at_n = b.v_at(a.rn);
        let at_d = b.v_at(a.rd);
        let vn_lo = b.load(b.i64, at_n);
        let vn_hi = b.load(b.i64, at_n + 8);
        let halves = if a.q { 2 } else { 1 };
        let mut total = if a.op == SimdAluOp::UMinV {
            b.k(b.i64, emask)
        } else {
            b.k(b.i64, 0)
        };
        for half in 0..halves {
            let v = if half == 0 { vn_lo } else { vn_hi };
            for lane in 0..(8 >> a.size) {
                let shift = lane as u64 * ebits;
                let field = b.imm(b.i64, B::BitAnd, b.imm(b.i64, B::Shr, v, shift), emask);
                total = match a.op {
                    SimdAluOp::UMinV => b.sel(b.i64, b.cmp(C::Ult, field, total), field, total),
                    SimdAluOp::UMaxV => b.sel(b.i64, b.cmp(C::Ugt, field, total), field, total),
                    _ => b.bin(b.i64, B::Add, total, field),
                };
            }
        }
        b.store(
            at_d,
            b.bin(b.i64, B::BitAnd, total, b.k(b.i64, result_mask)),
        );
        b.store(at_d + 8, b.k(b.i64, 0));
    } else if matches!(
        a.op,
        SimdAluOp::And
            | SimdAluOp::Bic
            | SimdAluOp::Orr
            | SimdAluOp::Orn
            | SimdAluOp::Eor
            | SimdAluOp::Bsl
            | SimdAluOp::Bit
            | SimdAluOp::Bif
    ) {
        let at_n = b.v_at(a.rn);
        let at_m = b.v_at(a.rm);
        let at_d = b.v_at(a.rd);
        for half in 0..2 {
            if half == 1 && !a.q {
                b.store(at_d + 8, b.k(b.i64, 0));
                continue;
            }
            let x = b.load(b.i64, at_n + half * 8);
            let y = b.load(b.i64, at_m + half * 8);
            let not = |v| b.bin(b.i64, B::BitXor, v, b.k(b.i64, u64::MAX));
            let r = match a.op {
                SimdAluOp::And => b.bin(b.i64, B::BitAnd, x, y),
                SimdAluOp::Bic => b.bin(b.i64, B::BitAnd, x, not(y)),
                SimdAluOp::Orr => b.bin(b.i64, B::BitOr, x, y),
                SimdAluOp::Orn => b.bin(b.i64, B::BitOr, x, not(y)),
                SimdAluOp::Eor => b.bin(b.i64, B::BitXor, x, y),
                // The destination is both a source mask and the
                // merge base for BSL/BIT/BIF.
                _ => {
                    let d = b.load(b.i64, at_d + half * 8);
                    match a.op {
                        SimdAluOp::Bsl => b.bin(
                            b.i64,
                            B::BitOr,
                            b.bin(b.i64, B::BitAnd, x, d),
                            b.bin(b.i64, B::BitAnd, y, not(d)),
                        ),
                        _ => {
                            let selected = match a.op {
                                SimdAluOp::Bit => {
                                    b.bin(b.i64, B::BitAnd, b.bin(b.i64, B::BitXor, d, x), y)
                                }
                                _ => b.bin(b.i64, B::BitAnd, b.bin(b.i64, B::BitXor, d, x), not(y)),
                            };
                            b.bin(b.i64, B::BitXor, d, selected)
                        }
                    }
                }
            };
            b.store(at_d + half * 8, r);
        }
    } else {
        let at_n = b.v_at(a.rn);
        let at_m = b.v_at(a.rm);
        let at_d = b.v_at(a.rd);
        let vn_lo = b.load(b.i64, at_n);
        let vn_hi = b.load(b.i64, at_n + 8);
        let vm_lo = b.load(b.i64, at_m);
        let vm_hi = b.load(b.i64, at_m + 8);
        let need_d = matches!(a.op, SimdAluOp::Mla | SimdAluOp::Mls);
        // Pairwise forms consume adjacent lanes of the full source pair:
        // lane 0/1 of Vn and Vm, then lane 2/3, and so on.
        if matches!(
            a.op,
            SimdAluOp::UMaxP
                | SimdAluOp::SMaxP
                | SimdAluOp::UMinP
                | SimdAluOp::SMinP
                | SimdAluOp::AddP
        ) {
            let ebits = (1u64 << a.size) * 8;
            let emask = if ebits == 64 {
                u64::MAX
            } else {
                (1u64 << ebits) - 1
            };
            let at_n = b.v_at(a.rn);
            let at_m = b.v_at(a.rm);
            let at_d = b.v_at(a.rd);
            let ssel = |c: Value, x: Value, y: Value| b.sel(b.i64, c, x, y);
            let sources = [
                b.load(b.i64, at_n),
                b.load(b.i64, at_n + 8),
                b.load(b.i64, at_m),
                b.load(b.i64, at_m + 8),
            ];
            // Each source contributes `per_source` lanes (a 64-bit vector for
            // the D form, a 128-bit one for the Q form). Lane `2k` pairs with
            // lane `2k + 1` and the result lands in result lane
            // `which * reduced + k`: the reduction of Vn fills the low half of
            // the result and Vm the high half. For the D form the whole result
            // is 64 bits, so Vm's lanes land in bits 32..64.
            let per_source = ((if a.q { 128 } else { 64 }) / ebits) as usize;
            let reduced_lanes = per_source / 2;
            let signed = matches!(a.op, SimdAluOp::SMaxP | SimdAluOp::SMinP);
            let mut out = [b.k(b.i64, 0); 2];
            for (which, half) in [(0usize, 0usize), (2usize, 1usize)] {
                for k in 0..reduced_lanes {
                    let lanes_per_dw = (64 / ebits) as usize;
                    let read = |lane: usize| {
                        let double = lane / lanes_per_dw;
                        let shift = (lane % lanes_per_dw) as u64 * ebits;
                        let raw = b.imm(
                            b.i64,
                            B::BitAnd,
                            b.imm(b.i64, B::Shr, sources[which + double], shift),
                            emask,
                        );
                        if signed {
                            // Ordered as signed numbers: sign-extend the lane.
                            let up = 64 - ebits;
                            b.imm(b.i64, B::Sar, b.imm(b.i64, B::Shl, raw, up), up)
                        } else {
                            raw
                        }
                    };
                    let lo = read(2 * k);
                    let hi = read(2 * k + 1);
                    let r = match a.op {
                        SimdAluOp::AddP => b.bin(b.i64, B::Add, lo, hi),
                        SimdAluOp::UMaxP => ssel(b.cmp(C::Ugt, lo, hi), lo, hi),
                        SimdAluOp::UMinP => ssel(b.cmp(C::Ult, lo, hi), lo, hi),
                        SimdAluOp::SMaxP => ssel(b.cmp(C::Sgt, lo, hi), lo, hi),
                        _ => ssel(b.cmp(C::Slt, lo, hi), lo, hi),
                    };
                    let masked = b.bin(b.i64, B::BitAnd, r, b.k(b.i64, emask));
                    // Result lane `index` sits at bit `index * ebits` of the
                    // 128-bit result.
                    let index = half * reduced_lanes + k;
                    let bit = index as u64 * ebits;
                    let slot = (bit / 64) as usize;
                    out[slot] = b.bin(
                        b.i64,
                        B::BitOr,
                        out[slot],
                        b.imm(b.i64, B::Shl, masked, bit % 64),
                    );
                }
            }
            b.store(at_d, out[0]);
            b.store(at_d + 8, out[1]);
        } else {
            let vd_lo = if need_d {
                Some(b.load(b.i64, at_d))
            } else {
                None
            };
            let vd_hi = if need_d {
                Some(b.load(b.i64, at_d + 8))
            } else {
                None
            };
            let ebits = (1u64 << a.size) * 8;
            let emask = if ebits == 64 {
                u64::MAX
            } else {
                (1u64 << ebits) - 1
            };
            let smax = u64::MAX >> (65 - ebits);
            let smin = !smax;
            let halves = if a.q { 2 } else { 1 };
            for half in 0..2 {
                if half >= halves {
                    b.store(at_d + half * 8, b.k(b.i64, 0));
                    continue;
                }
                let vn = if half == 0 { vn_lo } else { vn_hi };
                let vm = if half == 0 { vm_lo } else { vm_hi };
                let vd = if half == 0 { vd_lo } else { vd_hi };
                let mut acc = b.k(b.i64, 0);
                let lanes = 8 >> a.size;
                for lane in 0..lanes {
                    let shift = lane as u64 * ebits;
                    let fld = |v| b.imm(b.i64, B::BitAnd, b.imm(b.i64, B::Shr, v, shift), emask);
                    let ext = |v: Value| {
                        let up = b.bin(b.i64, B::Shl, v, b.k(b.i64, 64 - ebits));
                        b.imm(b.i64, B::Sar, up, 64 - ebits)
                    };
                    let an = fld(vn);
                    let bn = fld(vm);
                    let sa = ext(an);
                    let sb = ext(bn);
                    let dn = vd.map(fld);
                    let ssel = |c: Value, x: Value, y: Value| b.sel(b.i64, c, x, y);
                    let ssat = |v: Value| {
                        let over = b.cmp(C::Sgt, v, b.k(b.i64, smax));
                        let under = b.cmp(C::Slt, v, b.k(b.i64, smin));
                        ssel(over, b.k(b.i64, smax), ssel(under, b.k(b.i64, smin), v))
                    };
                    let r = match a.op {
                        SimdAluOp::Add => b.bin(b.i64, B::Add, an, bn),
                        SimdAluOp::Sub => b.bin(b.i64, B::Sub, an, bn),
                        SimdAluOp::Mul => b.bin(b.i64, B::Mul, an, bn),
                        SimdAluOp::Mla => {
                            b.bin(b.i64, B::Add, dn.unwrap(), b.bin(b.i64, B::Mul, an, bn))
                        }
                        SimdAluOp::Mls => {
                            b.bin(b.i64, B::Sub, dn.unwrap(), b.bin(b.i64, B::Mul, an, bn))
                        }
                        SimdAluOp::CmEq => {
                            ssel(b.cmp(C::Eq, an, bn), b.k(b.i64, emask), b.k(b.i64, 0))
                        }
                        SimdAluOp::CmGt => {
                            ssel(b.cmp(C::Sgt, sa, sb), b.k(b.i64, emask), b.k(b.i64, 0))
                        }
                        SimdAluOp::CmGe => {
                            ssel(b.cmp(C::Sge, sa, sb), b.k(b.i64, emask), b.k(b.i64, 0))
                        }
                        SimdAluOp::CmHi => {
                            ssel(b.cmp(C::Ugt, an, bn), b.k(b.i64, emask), b.k(b.i64, 0))
                        }
                        SimdAluOp::CmHs => {
                            ssel(b.cmp(C::Uge, an, bn), b.k(b.i64, emask), b.k(b.i64, 0))
                        }
                        SimdAluOp::CmTst => ssel(
                            b.cmp(C::Ne, b.bin(b.i64, B::BitAnd, an, bn), b.k(b.i64, 0)),
                            b.k(b.i64, emask),
                            b.k(b.i64, 0),
                        ),
                        SimdAluOp::UMin => ssel(b.cmp(C::Ult, an, bn), an, bn),
                        SimdAluOp::UMax => ssel(b.cmp(C::Ugt, an, bn), an, bn),
                        SimdAluOp::SMin => ssel(b.cmp(C::Slt, sa, sb), an, bn),
                        SimdAluOp::SMax => ssel(b.cmp(C::Sgt, sa, sb), an, bn),
                        SimdAluOp::UAbd => {
                            let d = b.bin(b.i64, B::Sub, an, bn);
                            ssel(
                                b.cmp(C::Uge, an, bn),
                                d,
                                b.bin(b.i64, B::Sub, b.k(b.i64, 0), d),
                            )
                        }
                        SimdAluOp::SAbd => {
                            let d = b.bin(b.i64, B::Sub, sa, sb);
                            let neg = b.bin(b.i64, B::Sub, b.k(b.i64, 0), d);
                            ssel(b.cmp(C::Sge, sa, sb), d, neg)
                        }
                        SimdAluOp::SQAdd => ssat(b.bin(b.i64, B::Add, sa, sb)),
                        SimdAluOp::UQAdd => {
                            let s = b.bin(b.i64, B::Add, an, bn);
                            let carry = b.cmp(C::Ult, s, an);
                            let big = b.cmp(C::Ugt, s, b.k(b.i64, emask));
                            ssel(b.bin(b.i64, B::BitOr, carry, big), b.k(b.i64, emask), s)
                        }
                        SimdAluOp::SQSub => ssat(b.bin(b.i64, B::Sub, sa, sb)),
                        SimdAluOp::UQSub => {
                            let borrow = b.cmp(C::Ugt, bn, an);
                            ssel(borrow, b.k(b.i64, 0), b.bin(b.i64, B::Sub, an, bn))
                        }
                        SimdAluOp::SHAdd
                        | SimdAluOp::UHAdd
                        | SimdAluOp::SHSub
                        | SimdAluOp::UHSub
                        | SimdAluOp::SRHAdd
                        | SimdAluOp::URHAdd => {
                            let signed_op = matches!(
                                a.op,
                                SimdAluOp::SHAdd | SimdAluOp::SHSub | SimdAluOp::SRHAdd
                            );
                            // Signed forms operate on sign-extended lanes, which
                            // keeps the halving shift meaningful for negatives.
                            let (op_a, op_b) = if signed_op { (sa, sb) } else { (an, bn) };
                            let shr = |v: Value| {
                                if signed_op {
                                    b.imm(b.i64, B::Sar, v, 1)
                                } else {
                                    b.imm(b.i64, B::Shr, v, 1)
                                }
                            };
                            let k = b.bin(b.i64, B::Sub, shr(op_a), shr(op_b));
                            let k = if matches!(
                                a.op,
                                SimdAluOp::SHAdd
                                    | SimdAluOp::UHAdd
                                    | SimdAluOp::SRHAdd
                                    | SimdAluOp::URHAdd
                            ) {
                                b.bin(b.i64, B::Add, shr(op_a), shr(op_b))
                            } else {
                                k
                            };
                            let a0 = b.bin(b.i64, B::BitAnd, op_a, b.k(b.i64, 1));
                            let b0 = b.bin(b.i64, B::BitAnd, op_b, b.k(b.i64, 1));
                            let corr = match a.op {
                                SimdAluOp::SHAdd | SimdAluOp::UHAdd => {
                                    b.bin(b.i64, B::BitAnd, a0, b0)
                                }
                                SimdAluOp::SRHAdd | SimdAluOp::URHAdd => {
                                    b.bin(b.i64, B::BitOr, a0, b0)
                                }
                                _ => b.bin(
                                    b.i64,
                                    B::BitAnd,
                                    b.bin(b.i64, B::BitXor, a0, b.k(b.i64, 1)),
                                    b0,
                                ),
                            };
                            if matches!(
                                a.op,
                                SimdAluOp::SHAdd
                                    | SimdAluOp::UHAdd
                                    | SimdAluOp::SRHAdd
                                    | SimdAluOp::URHAdd
                            ) {
                                b.bin(b.i64, B::Add, k, corr)
                            } else {
                                b.bin(b.i64, B::Sub, k, corr)
                            }
                        }
                        SimdAluOp::SSHl
                        | SimdAluOp::USHl
                        | SimdAluOp::SQShl
                        | SimdAluOp::UQShl
                        | SimdAluOp::SQRShl
                        | SimdAluOp::UQRShl => {
                            let signed_op = matches!(
                                a.op,
                                SimdAluOp::SSHl | SimdAluOp::SQShl | SimdAluOp::SQRShl
                            );
                            let rounding = matches!(a.op, SimdAluOp::SQRShl | SimdAluOp::UQRShl);
                            let saturating = matches!(
                                a.op,
                                SimdAluOp::SQShl
                                    | SimdAluOp::UQShl
                                    | SimdAluOp::SQRShl
                                    | SimdAluOp::UQRShl
                            );
                            // Shift amount: the lane's low byte, sign-extended to
                            // 64 bits (negative means a right shift).
                            let amt = b.bin(
                                b.i64,
                                B::Shl,
                                b.bin(b.i64, B::BitAnd, bn, b.k(b.i64, 0xff)),
                                b.k(b.i64, 56),
                            );
                            let amt = b.bin(b.i64, B::Sar, amt, b.k(b.i64, 56));
                            let left = b.cmp(C::Sge, amt, b.k(b.i64, 0));
                            let n = ssel(left, amt, b.bin(b.i64, B::Sub, b.k(b.i64, 0), amt));
                            let shl = |v: Value, k: Value| b.bin(b.i64, B::Shl, v, k);
                            let sar = |v: Value, k: Value| b.bin(b.i64, B::Sar, v, k);
                            let shr = |v: Value, k: Value| b.bin(b.i64, B::Shr, v, k);
                            // Beyond the lane width: a plain right shift keeps the
                            // sign fill (or zero), a rounding shift collapses to zero,
                            // and a saturating left shift pins to the lane bound.
                            let sign_fill = if signed_op {
                                ssel(
                                    b.cmp(C::Slt, sa, b.k(b.i64, 0)),
                                    b.k(b.i64, emask),
                                    b.k(b.i64, 0),
                                )
                            } else {
                                b.k(b.i64, 0)
                            };
                            let sat_fill = if signed_op {
                                ssel(
                                    b.cmp(C::Slt, sa, b.k(b.i64, 0)),
                                    b.k(b.i64, smin),
                                    b.k(b.i64, smax),
                                )
                            } else {
                                b.k(b.i64, emask)
                            };
                            let far = ssel(
                                left,
                                if saturating { sat_fill } else { b.k(b.i64, 0) },
                                if rounding { b.k(b.i64, 0) } else { sign_fill },
                            );
                            // Shift when the amount stays inside the lane width.
                            // Clamping to `ebits - 1` keeps the host shift defined.
                            let kc = ssel(
                                b.cmp(C::Uge, n, b.k(b.i64, ebits)),
                                b.k(b.i64, ebits - 1),
                                n,
                            );
                            let rnd = if rounding {
                                shl(b.k(b.i64, 1), b.bin(b.i64, B::Sub, kc, b.k(b.i64, 1)))
                            } else {
                                b.k(b.i64, 0)
                            };
                            let opnd = if signed_op { sa } else { an };
                            let t = if rounding {
                                // The rounding constant must not leak out of the lane.
                                b.bin(
                                    b.i64,
                                    B::BitAnd,
                                    b.bin(b.i64, B::Add, opnd, rnd),
                                    b.k(b.i64, emask),
                                )
                            } else {
                                opnd
                            };
                            let right = if rounding {
                                if signed_op { sar(t, kc) } else { shr(t, kc) }
                            } else if signed_op {
                                sar(opnd, kc)
                            } else {
                                shr(opnd, kc)
                            };
                            let shifted = shl(opnd, kc);
                            let moved = ssel(left, shifted, right);
                            let moved = ssel(b.cmp(C::Uge, n, b.k(b.i64, ebits)), far, moved);
                            if !saturating {
                                moved
                            } else if signed_op {
                                // Saturation only matters for left shifts; the
                                // right-shift result already fits the lane. Below
                                // the 64-bit lane the shifted value is exact, so the
                                // lane bounds test it directly; a 64-bit lane needs
                                // the round-trip check because the shift wraps.
                                let ov = if ebits == 64 {
                                    b.cmp(C::Ne, sar(shifted, kc), opnd)
                                } else {
                                    b.bin(
                                        b.boolean,
                                        B::BitOr,
                                        b.cmp(C::Sgt, shifted, b.k(b.i64, smax)),
                                        b.cmp(C::Slt, shifted, b.k(b.i64, smin)),
                                    )
                                };
                                let sat = ssel(
                                    b.cmp(C::Slt, sa, b.k(b.i64, 0)),
                                    b.k(b.i64, smin),
                                    b.k(b.i64, smax),
                                );
                                ssel(left, ssel(ov, sat, moved), moved)
                            } else {
                                let ov = if ebits == 64 {
                                    b.cmp(C::Ne, shr(shifted, kc), an)
                                } else {
                                    b.cmp(C::Ugt, shifted, b.k(b.i64, emask))
                                };
                                ssel(left, ssel(ov, b.k(b.i64, emask), moved), moved)
                            }
                        }
                        _ => b.k(b.i64, 0),
                    };
                    let masked = b.bin(b.i64, B::BitAnd, r, b.k(b.i64, emask));
                    acc = b.bin(b.i64, B::BitOr, acc, b.imm(b.i64, B::Shl, masked, shift));
                }
                b.store(at_d + half * 8, acc);
            }
        }
    }
}
