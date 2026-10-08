//! Integer Advanced SIMD instructions executed on the host from the raw word:
//! the vector permutes (`uzp`, `zip`, `trn`), the two-register-miscellaneous
//! group (`rev`, `cnt`, `mvn`, compares against zero, `abs`/`neg`, the narrowing
//! `xtn` family and the pairwise-long adds), the three-registers-different long and
//! wide arithmetic (`uaddl`, `umull`, ...), and the 64-bit scalar forms.
//!
//! Each register is a `u128`, each lane a bit field of it, so one lane loop serves
//! every element size. Decode only classifies a word with [`is_supported`]; the
//! JIT traps and the machine (or user mode) calls [`run`].

use super::Cpu;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Permute {
    Uzp1,
    Uzp2,
    Zip1,
    Zip2,
    Trn1,
    Trn2,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Zero {
    Gt,
    Ge,
    Le,
    Lt,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Narrow {
    Plain,
    SignedSat,
    UnsignedSat,
    SignedToUnsignedSat,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Long {
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
enum Scalar {
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
enum Op {
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

fn parse(word: u32) -> Option<Op> {
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

/// Whether [`run`] implements `word`.
pub fn is_supported(word: u32) -> bool {
    parse(word).is_some()
}

fn read(cpu: &Cpu, reg: u8) -> u128 {
    let [lo, hi] = cpu.v[usize::from(reg)];
    u128::from(lo) | u128::from(hi) << 64
}
/// Write `value`; without `full` the upper 64 bits are cleared.
fn write(cpu: &mut Cpu, reg: u8, value: u128, full: bool) {
    cpu.v[usize::from(reg)] = [value as u64, if full { (value >> 64) as u64 } else { 0 }];
}
fn mask(bits: u32) -> u128 {
    if bits >= 128 {
        u128::MAX
    } else {
        (1 << bits) - 1
    }
}
fn get(v: u128, bits: u32, index: u32) -> u128 {
    (v >> (bits * index)) & mask(bits)
}
fn put(v: &mut u128, bits: u32, index: u32, value: u128) {
    let field = mask(bits) << (bits * index);
    *v = (*v & !field) | ((value << (bits * index)) & field);
}
/// The low `bits` bits of `x` as a signed number.
fn signed(x: u128, bits: u32) -> i128 {
    let shift = 128 - bits;
    ((x << shift) as i128) >> shift
}
fn extend(x: u128, bits: u32, signed_lane: bool) -> i128 {
    if signed_lane {
        signed(x, bits)
    } else {
        x as i128
    }
}
fn lanes(size: u32, q: bool) -> u32 {
    (if q { 128 } else { 64 }) / (8 << size)
}

/// Run the instruction `word` on `cpu`. `word` must satisfy [`is_supported`].
pub fn run(cpu: &mut Cpu, word: u32) {
    let op = parse(word).expect("an unsupported SIMD word reached the executor");
    match op {
        Op::Permute {
            kind,
            size,
            q,
            rd,
            rn,
            rm,
        } => {
            let (a, b) = (read(cpu, rn), read(cpu, rm));
            let (bits, n) = (8 << size, lanes(size, q));
            let half = n / 2;
            let mut out = 0;
            for i in 0..n {
                let value = match kind {
                    Permute::Uzp1 if i < half => get(a, bits, 2 * i),
                    Permute::Uzp1 => get(b, bits, 2 * (i - half)),
                    Permute::Uzp2 if i < half => get(a, bits, 2 * i + 1),
                    Permute::Uzp2 => get(b, bits, 2 * (i - half) + 1),
                    Permute::Zip1 => get(if i % 2 == 0 { a } else { b }, bits, i / 2),
                    Permute::Zip2 => get(if i % 2 == 0 { a } else { b }, bits, half + i / 2),
                    Permute::Trn1 => get(if i % 2 == 0 { a } else { b }, bits, i - i % 2),
                    Permute::Trn2 => get(if i % 2 == 0 { a } else { b }, bits, i | 1),
                };
                put(&mut out, bits, i, value);
            }
            write(cpu, rd, out, q);
        }
        Op::Reverse {
            group,
            size,
            q,
            rd,
            rn,
        } => {
            let a = read(cpu, rn);
            let bits = 8 << size;
            let per_group = group / bits;
            let mut out = 0;
            for i in 0..lanes(size, q) {
                let (g, j) = (i / per_group, i % per_group);
                put(
                    &mut out,
                    bits,
                    i,
                    get(a, bits, g * per_group + (per_group - 1 - j)),
                );
            }
            write(cpu, rd, out, q);
        }
        Op::Count { q, rd, rn } => {
            let a = read(cpu, rn);
            let mut out = 0;
            for i in 0..lanes(0, q) {
                put(
                    &mut out,
                    8,
                    i,
                    u128::from((get(a, 8, i) as u8).count_ones()),
                );
            }
            write(cpu, rd, out, q);
        }
        Op::Not { q, rd, rn } => {
            let a = !read(cpu, rn);
            write(cpu, rd, a, q);
        }
        Op::ReverseBits { q, rd, rn } => {
            let a = read(cpu, rn);
            let mut out = 0;
            for i in 0..lanes(0, q) {
                put(
                    &mut out,
                    8,
                    i,
                    u128::from((get(a, 8, i) as u8).reverse_bits()),
                );
            }
            write(cpu, rd, out, q);
        }
        Op::CompareZero {
            kind,
            size,
            q,
            rd,
            rn,
        } => {
            let a = read(cpu, rn);
            let bits = 8 << size;
            let mut out = 0;
            for i in 0..lanes(size, q) {
                let x = signed(get(a, bits, i), bits);
                let holds = match kind {
                    Zero::Gt => x > 0,
                    Zero::Ge => x >= 0,
                    Zero::Le => x <= 0,
                    Zero::Lt => x < 0,
                };
                put(&mut out, bits, i, if holds { mask(bits) } else { 0 });
            }
            write(cpu, rd, out, q);
        }
        Op::Abs {
            negate,
            size,
            q,
            rd,
            rn,
        } => {
            let a = read(cpu, rn);
            let bits = 8 << size;
            let mut out = 0;
            for i in 0..lanes(size, q) {
                let x = signed(get(a, bits, i), bits);
                // The most negative value keeps its pattern in both operations.
                let r = if negate {
                    x.wrapping_neg()
                } else {
                    x.wrapping_abs()
                };
                put(&mut out, bits, i, r as u128 & mask(bits));
            }
            write(cpu, rd, out, q);
        }
        Op::Narrow {
            kind,
            size,
            upper,
            rd,
            rn,
        } => {
            let a = read(cpu, rn);
            let (narrow, wide) = (8 << size, 16 << size);
            let mut out = 0;
            for i in 0..64 / narrow {
                let x = get(a, wide, i);
                let r = match kind {
                    Narrow::Plain => x,
                    Narrow::SignedSat => {
                        let s = signed(x, wide);
                        let (lo, hi) = (-(1i128 << (narrow - 1)), (1i128 << (narrow - 1)) - 1);
                        s.clamp(lo, hi) as u128
                    }
                    Narrow::UnsignedSat => x.min(mask(narrow)),
                    Narrow::SignedToUnsignedSat => {
                        signed(x, wide).clamp(0, mask(narrow) as i128) as u128
                    }
                };
                put(&mut out, narrow, i, r & mask(narrow));
            }
            let value = if upper {
                (read(cpu, rd) & mask(64)) | out << 64
            } else {
                out
            };
            write(cpu, rd, value, true);
        }
        Op::PairLong {
            signed: signed_lane,
            accumulate,
            size,
            q,
            rd,
            rn,
        } => {
            let a = read(cpu, rn);
            let (bits, wide) = (8 << size, 16 << size);
            let old = read(cpu, rd);
            let mut out = 0;
            for i in 0..lanes(size, q) / 2 {
                let pair = extend(get(a, bits, 2 * i), bits, signed_lane)
                    + extend(get(a, bits, 2 * i + 1), bits, signed_lane);
                let sum = if accumulate {
                    pair.wrapping_add(get(old, wide, i) as i128)
                } else {
                    pair
                };
                put(&mut out, wide, i, sum as u128 & mask(wide));
            }
            write(cpu, rd, out, q);
        }
        Op::Long {
            kind,
            signed: signed_lane,
            size,
            upper,
            rd,
            rn,
            rm,
        } => {
            let (a, b, old) = (read(cpu, rn), read(cpu, rm), read(cpu, rd));
            let (bits, wide) = (8 << size, 16 << size);
            // The narrow operand comes from the low or high half of the register.
            let shift = if upper { 64 } else { 0 };
            let mut out = 0;
            for i in 0..64 / bits {
                let y = extend(get(b >> shift, bits, i), bits, signed_lane);
                let x = match kind {
                    Long::AddWide | Long::SubWide => extend(get(a, wide, i), wide, signed_lane),
                    _ => extend(get(a >> shift, bits, i), bits, signed_lane),
                };
                let accumulator = get(old, wide, i) as i128;
                let r = match kind {
                    Long::Add | Long::AddWide => x + y,
                    Long::Sub | Long::SubWide => x - y,
                    Long::Abd => (x - y).abs(),
                    Long::Mul => x * y,
                    Long::Mla => accumulator.wrapping_add(x * y),
                    Long::Mls => accumulator.wrapping_sub(x * y),
                };
                put(&mut out, wide, i, r as u128 & mask(wide));
            }
            write(cpu, rd, out, true);
        }
        Op::ScalarAddp { rd, rn } => {
            let [lo, hi] = cpu.v[usize::from(rn)];
            cpu.v[usize::from(rd)] = [lo.wrapping_add(hi), 0];
        }
        Op::ScalarDup {
            size,
            index,
            rd,
            rn,
        } => {
            let value = get(read(cpu, rn), 8 << size, index);
            write(cpu, rd, value, false);
        }
        Op::ShiftImm {
            right,
            signed: signed_lane,
            size,
            q,
            shift,
            rd,
            rn,
        } => {
            let a = read(cpu, rn);
            let bits = 8 << size;
            let mut out = 0;
            for i in 0..lanes(size, q) {
                let x = get(a, bits, i);
                let r = if !right {
                    x << shift
                } else if signed_lane {
                    (signed(x, bits) >> shift) as u128
                } else {
                    x >> shift
                };
                put(&mut out, bits, i, r & mask(bits));
            }
            write(cpu, rd, out, q);
        }
        Op::ShiftLong {
            signed: signed_lane,
            size,
            shift,
            upper,
            rd,
            rn,
        } => {
            let a = read(cpu, rn) >> if upper { 64 } else { 0 };
            let (bits, wide) = (8 << size, 16 << size);
            let mut out = 0;
            for i in 0..64 / bits {
                let x = extend(get(a, bits, i), bits, signed_lane);
                put(&mut out, wide, i, (x << shift) as u128 & mask(wide));
            }
            write(cpu, rd, out, true);
        }
        Op::ShiftNarrow {
            size,
            shift,
            upper,
            rd,
            rn,
        } => {
            let a = read(cpu, rn);
            let narrow = 8 << size;
            let mut out = 0;
            for i in 0..64 / narrow {
                put(
                    &mut out,
                    narrow,
                    i,
                    (get(a, 2 * narrow, i) >> shift) & mask(narrow),
                );
            }
            let value = if upper {
                (read(cpu, rd) & mask(64)) | out << 64
            } else {
                out
            };
            write(cpu, rd, value, true);
        }
        Op::ScalarThree { kind, rd, rn, rm } => {
            let (a, b) = (cpu.v[usize::from(rn)][0], cpu.v[usize::from(rm)][0]);
            let flag = |holds: bool| if holds { u64::MAX } else { 0 };
            let r = match kind {
                Scalar::Add => a.wrapping_add(b),
                Scalar::Sub => a.wrapping_sub(b),
                Scalar::Gt => flag((a as i64) > (b as i64)),
                Scalar::Hi => flag(a > b),
                Scalar::Ge => flag((a as i64) >= (b as i64)),
                Scalar::Hs => flag(a >= b),
                Scalar::Eq => flag(a == b),
                Scalar::Tst => flag(a & b != 0),
            };
            cpu.v[usize::from(rd)] = [r, 0];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A register whose lane `i` of `bits` bits is `f(i)`.
    fn pack(bits: u32, count: u32, f: impl Fn(u32) -> u128) -> [u64; 2] {
        let mut v = 0u128;
        for i in 0..count {
            put(&mut v, bits, i, f(i));
        }
        [v as u64, (v >> 64) as u64]
    }
    fn unpack(r: [u64; 2], bits: u32, count: u32) -> Vec<u128> {
        let v = u128::from(r[0]) | u128::from(r[1]) << 64;
        (0..count).map(|i| get(v, bits, i)).collect()
    }
    /// A CPU whose v1 holds bytes 0..16 and v2 bytes 16..32.
    fn permute_cpu() -> Cpu {
        let mut cpu = Cpu::default();
        cpu.v[1] = pack(8, 16, |i| u128::from(i));
        cpu.v[2] = pack(8, 16, |i| u128::from(16 + i));
        cpu
    }
    fn run_lanes(word: u32, cpu: &mut Cpu, bits: u32, count: u32) -> Vec<u128> {
        run(cpu, word);
        unpack(cpu.v[0], bits, count)
    }

    #[test]
    fn permutes_pick_the_architectural_lanes_for_every_size_and_q() {
        let mut c = permute_cpu();
        // Word lanes of v1 are w1_i = bytes 4i..4i+3; of v2 likewise from 16.
        let a = |i: u128| 0x0302_0100 + 0x0404_0404 * i;
        let b = |i: u128| 0x1312_1110 + 0x0404_0404 * i;
        assert_eq!(
            run_lanes(0x4e821820, &mut c, 32, 4),
            [a(0), a(2), b(0), b(2)]
        ); // uzp1.4s
        assert_eq!(
            run_lanes(0x4e825820, &mut c, 32, 4),
            [a(1), a(3), b(1), b(3)]
        ); // uzp2.4s
        assert_eq!(
            run_lanes(0x4e822820, &mut c, 32, 4),
            [a(0), b(0), a(2), b(2)]
        ); // trn1.4s
        // zip1.8b interleaves the low bytes and clears the upper half.
        let mut c = permute_cpu();
        run(&mut c, 0x0e023820);
        assert_eq!(unpack(c.v[0], 8, 8), [0, 16, 1, 17, 2, 18, 3, 19]);
        assert_eq!(c.v[0][1], 0);
        // zip2.16b interleaves the high bytes.
        let mut c = permute_cpu();
        run(&mut c, 0x4e027820);
        let expected: Vec<u128> = (8..16).flat_map(|i| [i, 16 + i]).collect();
        assert_eq!(unpack(c.v[0], 8, 16), expected);
        // trn2.8h takes the odd halfwords.
        let mut c = permute_cpu();
        run(&mut c, 0x4e426820);
        let half = |base: u128, i: u128| (base + 2 * i) | (base + 2 * i + 1) << 8;
        assert_eq!(
            unpack(c.v[0], 16, 8),
            [
                half(0, 1),
                half(16, 1),
                half(0, 3),
                half(16, 3),
                half(0, 5),
                half(16, 5),
                half(0, 7),
                half(16, 7)
            ]
        );
        // zip1.2d: the low doublewords.
        let mut c = permute_cpu();
        run(&mut c, 0x4ec23820);
        assert_eq!(c.v[0], [c.v[1][0], c.v[2][0]]);
    }

    #[test]
    fn reversals_count_and_bitwise_not() {
        let mut c = Cpu::default();
        c.v[1] = pack(8, 16, |i| u128::from(i));
        // rev32.16b reverses the bytes inside each word.
        run(&mut c, 0x6e200820);
        assert_eq!(
            unpack(c.v[0], 8, 16),
            [3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12]
        );
        // rev16.8b swaps byte pairs and clears the upper half.
        c.v[0] = [u64::MAX; 2];
        run(&mut c, 0x0e201820);
        assert_eq!(unpack(c.v[0], 8, 8), [1, 0, 3, 2, 5, 4, 7, 6]);
        assert_eq!(c.v[0][1], 0);
        // rev64.4s swaps the words of each doubleword.
        c.v[1] = pack(32, 4, |i| u128::from(10 + i));
        run(&mut c, 0x4ea00820);
        assert_eq!(unpack(c.v[0], 32, 4), [11, 10, 13, 12]);
        // cnt counts the bits of every byte; mvn inverts; rbit reverses each byte.
        c.v[1] = pack(8, 16, |i| {
            [
                0xff, 0, 0x0f, 1, 0x80, 0xaa, 0x55, 3, 0, 0, 0, 0, 0, 0, 0, 0xfe,
            ][i as usize]
        });
        run(&mut c, 0x4e205820);
        assert_eq!(unpack(c.v[0], 8, 8), [8, 0, 4, 1, 1, 4, 4, 2]);
        assert_eq!(unpack(c.v[0], 8, 16)[15], 7);
        run(&mut c, 0x6e205820);
        assert_eq!(unpack(c.v[0], 8, 4), [0, 0xff, 0xf0, 0xfe]);
        c.v[0] = [u64::MAX; 2];
        run(&mut c, 0x2e605820); // rbit.8b
        assert_eq!(unpack(c.v[0], 8, 4), [0xff, 0, 0xf0, 0x80]);
        assert_eq!(c.v[0][1], 0);
    }

    #[test]
    fn compares_against_zero_abs_and_neg_are_signed_and_wrap() {
        let mut c = Cpu::default();
        let words = [-1i32 as u32 as u128, 0, 1, 0x8000_0000];
        c.v[1] = pack(32, 4, |i| words[i as usize]);
        let ones = u128::from(u32::MAX);
        run(&mut c, 0x4ea08820); // cmgt.4s #0
        assert_eq!(unpack(c.v[0], 32, 4), [0, 0, ones, 0]);
        // cmle.8h #0 over halfwords of the same register.
        run(&mut c, 0x6e609820);
        let h = unpack(c.v[0], 16, 8);
        assert_eq!(h[0], 0xffff, "0xffff is -1");
        assert_eq!(h[1], 0xffff, "0xffff is -1 (upper half of the first word)");
        assert_eq!(h[4], 0, "1 is positive");
        // cmlt.2d #0 looks at the sign of 64-bit lanes.
        c.v[1] = [u64::MAX, 1];
        run(&mut c, 0x4ee0a820);
        assert_eq!(c.v[0], [u64::MAX, 0]);
        // abs.4s keeps the most negative value; neg.2d wraps likewise.
        c.v[1] = pack(32, 4, |i| {
            [(-5i32) as u32 as u128, 5, 0x8000_0000, 0][i as usize]
        });
        run(&mut c, 0x4ea0b820);
        assert_eq!(unpack(c.v[0], 32, 4), [5, 5, 0x8000_0000, 0]);
        c.v[1] = [1, 1 << 63];
        run(&mut c, 0x6ee0b820);
        assert_eq!(c.v[0], [u64::MAX, 1 << 63]);
        // cmge.16b #0 with both halves.
        c.v[1] = pack(8, 16, |i| if i % 2 == 0 { 0x80 } else { 0x7f });
        run(&mut c, 0x6e208820);
        assert_eq!(unpack(c.v[0], 8, 4), [0, 0xff, 0, 0xff]);
    }

    #[test]
    fn narrowing_truncates_or_saturates_and_the_2_form_keeps_the_low_half() {
        let mut c = Cpu::default();
        // xtn.2s from two doublewords; the stale upper half is cleared.
        c.v[1] = [0x1_0000_0002, 0x3_0000_0004];
        c.v[0] = [u64::MAX; 2];
        run(&mut c, 0x0ea12820);
        assert_eq!(c.v[0], [0x0000_0004_0000_0002, 0]);
        // xtn2.4s writes the upper half and keeps the lower.
        c.v[0] = [0x1111, 0x2222];
        run(&mut c, 0x4ea12820);
        assert_eq!(c.v[0], [0x1111, 0x0000_0004_0000_0002]);
        // uqxtn.8b saturates unsigned.
        c.v[1] = pack(16, 8, |i| [0x1ff, 5, 0xff, 0x100, 0, 1, 2, 3][i as usize]);
        run(&mut c, 0x2e214820);
        assert_eq!(unpack(c.v[0], 8, 8), [0xff, 5, 0xff, 0xff, 0, 1, 2, 3]);
        // sqxtn.4h saturates signed; sqxtun.8b clamps negatives to zero.
        c.v[1] = pack(32, 4, |i| {
            [
                70_000u128,
                (-70_000i32) as u32 as u128,
                5,
                (-5i32) as u32 as u128,
            ][i as usize]
        });
        run(&mut c, 0x0e614820);
        assert_eq!(unpack(c.v[0], 16, 4), [0x7fff, 0x8000, 5, 0xfffb]);
        c.v[1] = pack(16, 8, |i| {
            [(-5i16) as u16 as u128, 300, 7, 255, 256, 0, 0, 0][i as usize]
        });
        run(&mut c, 0x2e212820);
        assert_eq!(unpack(c.v[0], 8, 5), [0, 255, 7, 255, 255]);
    }

    #[test]
    fn pairwise_long_adds_widen_and_accumulate() {
        let mut c = Cpu::default();
        // uaddlp.4h from bytes.
        c.v[1] = pack(8, 8, |i| [255, 255, 1, 2, 0, 0, 128, 128][i as usize]);
        run(&mut c, 0x2e202820);
        assert_eq!(unpack(c.v[0], 16, 4), [510, 3, 0, 256]);
        // saddlp.2d from words: -1 + -1 = -2.
        c.v[1] = pack(32, 4, |i| {
            [u128::from(u32::MAX), u128::from(u32::MAX), 3, 4][i as usize]
        });
        run(&mut c, 0x4ea02820);
        assert_eq!(c.v[0], [(-2i64) as u64, 7]);
        // uadalp.2d accumulates into the destination.
        c.v[0] = [10, 20];
        run(&mut c, 0x6ea06820);
        assert_eq!(c.v[0], [u64::from(u32::MAX) * 2 + 10, 27]);
        // sadalp.8h sign-extends the bytes before adding.
        c.v[1] = pack(8, 16, |i| if i % 2 == 0 { 0x80 } else { 1 });
        c.v[0] = pack(16, 8, |_| 100);
        run(&mut c, 0x4e206820);
        assert_eq!(unpack(c.v[0], 16, 8)[0], (100 - 128 + 1) as u16 as u128);
    }

    #[test]
    fn long_and_wide_arithmetic_uses_the_requested_half_and_extension() {
        let mut c = Cpu::default();
        c.v[1] = pack(8, 16, |i| if i < 8 { 255 } else { 3 });
        c.v[2] = pack(8, 16, |i| if i < 8 { 255 } else { 5 });
        // uaddl.8h adds the low bytes of both operands as 16-bit lanes.
        run(&mut c, 0x2e220020);
        assert_eq!(unpack(c.v[0], 16, 8), [510; 8]);
        // usubl.4s over halfwords: 3 - 5 wraps in 32 bits.
        c.v[1] = pack(16, 8, |i| if i < 4 { 3 } else { 9 });
        c.v[2] = pack(16, 8, |i| if i < 4 { 5 } else { 4 });
        run(&mut c, 0x2e622020);
        assert_eq!(unpack(c.v[0], 32, 4), [0xffff_fffe; 4]);
        // uabdl2.2d takes the high words and the absolute difference.
        c.v[1] = pack(32, 4, |i| [0, 0, 10, 100][i as usize]);
        c.v[2] = pack(32, 4, |i| [0, 0, 40, 7][i as usize]);
        run(&mut c, 0x6ea27020);
        assert_eq!(c.v[0], [30, 93]);
        // uaddw.2d adds the narrow low words to the wide Vn lanes (no extension of Vn).
        c.v[1] = [1 << 40, 2];
        c.v[2] = pack(32, 4, |i| [7, 9, 0, 0][i as usize]);
        run(&mut c, 0x2ea21020);
        assert_eq!(c.v[0], [(1 << 40) + 7, 11]);
        // saddw.8h sign-extends the bytes of Vm.
        c.v[1] = pack(16, 8, |_| 1000);
        c.v[2] = pack(8, 16, |i| if i == 0 { 0xff } else { 1 });
        run(&mut c, 0x0e221020);
        assert_eq!(unpack(c.v[0], 16, 2), [999, 1001]);
        // usubw2.8h: Vn wide minus the upper bytes of Vm.
        c.v[1] = pack(16, 8, |_| 300);
        c.v[2] = pack(8, 16, |i| if i >= 8 { 2 } else { 99 });
        run(&mut c, 0x6e223020);
        assert_eq!(unpack(c.v[0], 16, 8), [298; 8]);
        // umull.4s: 0xffff * 0xffff; smull.8h: -128 * -128.
        c.v[1] = pack(16, 8, |_| 0xffff);
        c.v[2] = pack(16, 8, |_| 0xffff);
        run(&mut c, 0x2e62c020);
        assert_eq!(unpack(c.v[0], 32, 4), [0xfffe_0001; 4]);
        c.v[1] = pack(8, 16, |_| 0x80);
        c.v[2] = pack(8, 16, |_| 0x80);
        run(&mut c, 0x0e22c020);
        assert_eq!(unpack(c.v[0], 16, 8), [16384; 8]);
        // umull2.2d uses the upper words; umlal.4s accumulates; umlsl2.8h subtracts.
        c.v[1] = pack(32, 4, |i| [1, 1, 100_000, 3][i as usize]);
        c.v[2] = pack(32, 4, |i| [1, 1, 100_000, 5][i as usize]);
        run(&mut c, 0x6ea2c020);
        assert_eq!(c.v[0], [10_000_000_000, 15]);
        c.v[1] = pack(16, 8, |_| 3);
        c.v[2] = pack(16, 8, |_| 4);
        c.v[0] = pack(32, 4, |_| 1);
        run(&mut c, 0x2e628020);
        assert_eq!(unpack(c.v[0], 32, 4), [13; 4]);
        c.v[1] = pack(8, 16, |_| 2);
        c.v[2] = pack(8, 16, |_| 3);
        c.v[0] = pack(16, 8, |_| 10);
        run(&mut c, 0x6e22a020);
        assert_eq!(unpack(c.v[0], 16, 8), [4; 8]);
    }

    #[test]
    fn scalar_forms_work_on_64_bits_and_clear_the_rest_of_the_register() {
        let mut c = Cpu::default();
        c.v[1] = [5, 7];
        c.v[0] = [u64::MAX; 2];
        run(&mut c, 0x5ef1b820); // addp d0, v1.2d
        assert_eq!(c.v[0], [12, 0]);
        c.v[0] = [u64::MAX; 2];
        run(&mut c, 0x5e180420); // mov d0, v1.d[1]
        assert_eq!(c.v[0], [7, 0]);
        c.v[1] = pack(8, 16, |i| u128::from(0x10 + i));
        run(&mut c, 0x5e070420); // mov b0, v1.b[3]
        assert_eq!(c.v[0], [0x13, 0]);
        c.v[1] = [2, 0];
        c.v[2] = [3, 0];
        run(&mut c, 0x5ee28420); // add d0, d1, d2
        assert_eq!(c.v[0][0], 5);
        run(&mut c, 0x7ee28420); // sub d0, d1, d2
        assert_eq!(c.v[0][0], u64::MAX);
        // cmhi d4, d2, d1 is unsigned: 3 > 2.
        run(&mut c, 0x7ee13444);
        assert_eq!(c.v[4][0], u64::MAX);
        c.v[1] = [u64::MAX, 0];
        run(&mut c, 0x7ee13444); // d2 (3) > d1 (max)? no
        assert_eq!(c.v[4][0], 0);
        c.v[1] = [4, 0];
        c.v[2] = [4, 0];
        run(&mut c, 0x7ee28c20); // cmeq d0, d1, d2
        assert_eq!(c.v[0][0], u64::MAX);
    }

    #[test]
    fn shifts_by_immediate_widen_narrow_and_keep_lane_boundaries() {
        let mut c = Cpu::default();
        // ushll.8h v1, v0.8b, #0 is `uxtl`: zero-extend the low eight bytes.
        c.v[0] = pack(8, 16, |i| u128::from(0xf0 + i));
        run(&mut c, 0x2f08a401);
        assert_eq!(
            unpack(c.v[1], 16, 8),
            (0..8).map(|i| 0xf0 + i).collect::<Vec<u128>>()
        );
        // ushll2.4s #5 widens the upper four halfwords and shifts left.
        c.v[1] = pack(16, 8, |i| u128::from(i + 1));
        run(&mut c, 0x6f15a420);
        assert_eq!(unpack(c.v[0], 32, 4), [5 << 5, 6 << 5, 7 << 5, 8 << 5]);
        // sshll.2d #31 sign-extends the low two words.
        c.v[1] = pack(32, 4, |i| [u128::from(u32::MAX), 1, 9, 9][i as usize]);
        run(&mut c, 0x0f3fa420);
        assert_eq!(c.v[0], [0xffff_ffff_8000_0000, 1 << 31]);
        // sshll2.8h #7 takes the high bytes.
        c.v[1] = pack(8, 16, |i| if i >= 8 { 0x80 } else { 1 });
        run(&mut c, 0x4f0fa420);
        assert_eq!(
            unpack(c.v[0], 16, 8),
            [0xc000; 8],
            "-128 << 7 in sixteen bits"
        );
        // shl.4s #5, shl.2d #63, and shl.8b #1 which clears the upper half.
        c.v[1] = pack(32, 4, |i| u128::from(i + 1) | 0x8000_0000);
        run(&mut c, 0x4f255420);
        assert_eq!(
            unpack(c.v[0], 32, 4),
            [32, 64, 96, 128],
            "bits shifted out are dropped"
        );
        c.v[1] = [3, 2];
        run(&mut c, 0x4f7f5420);
        assert_eq!(c.v[0], [1 << 63, 0]);
        c.v[1] = [0x8040_2010_0804_0281, u64::MAX];
        c.v[0] = [u64::MAX; 2];
        run(&mut c, 0x0f095420);
        assert_eq!(c.v[0], [0x0080_4020_1008_0402, 0]);
        // ushr.2d #1, and ushr.4s #32 which shifts everything out.
        c.v[1] = [u64::MAX, 2];
        run(&mut c, 0x6f7f0420);
        assert_eq!(c.v[0], [u64::MAX >> 1, 1]);
        run(&mut c, 0x6f200420);
        assert_eq!(c.v[0], [0, 0]);
        // sshr.16b #7 and sshr.8h #16 fill with the sign bit.
        c.v[1] = pack(8, 16, |i| if i % 2 == 0 { 0x80 } else { 0x7f });
        run(&mut c, 0x4f090420);
        assert_eq!(unpack(c.v[0], 8, 4), [0xff, 0, 0xff, 0]);
        c.v[1] = pack(16, 8, |i| if i % 2 == 0 { 0x8000 } else { 0x7fff });
        run(&mut c, 0x4f100420);
        assert_eq!(unpack(c.v[0], 16, 4), [0xffff, 0, 0xffff, 0]);
        // shrn.8b #4 narrows halfwords; shrn2.4s #32 writes the upper half only.
        c.v[1] = pack(16, 8, |i| u128::from(i + 1) << 4 | 0xf000);
        c.v[0] = [u64::MAX; 2];
        run(&mut c, 0x0f0c8420);
        assert_eq!(unpack(c.v[0], 8, 8), [1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(c.v[0][1], 0);
        c.v[1] = [0x1_0000_0000_u64 * 7 | 5, 0x1_0000_0000_u64 * 9];
        c.v[0] = [0x1111, 0x2222];
        run(&mut c, 0x4f208420);
        assert_eq!(c.v[0], [0x1111, 0x0000_0009_0000_0007]);
        // shrn.2s #1 from doublewords.
        c.v[1] = [0x1_0000_0006, 0x2_0000_000a];
        run(&mut c, 0x0f3f8420);
        assert_eq!(c.v[0], [0x0000_0005_8000_0003, 0]);
    }
    #[test]
    fn classification_declines_what_it_does_not_implement() {
        // cmeq #0 is `SimdCompareZero`; half-width lanes need Q for 64-bit lanes;
        // `ins`, `umov` and the three-same group belong to other decoders.
        assert!(!is_supported(0x0e209820));
        assert!(!is_supported(0x0ee0b820), "abs .1d is reserved");
        assert!(!is_supported(0x2ee21c20), "bif is three-same");
        assert!(!is_supported(0x4e083c20), "umov");
        assert!(is_supported(0x4e821820));
    }
}
