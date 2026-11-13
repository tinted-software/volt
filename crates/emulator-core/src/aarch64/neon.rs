//! The rest of Advanced SIMD (NEON), executed on the host from the raw word.
//!
//! [`simd`](super::simd) and [`fp`](super::fp) cover the instructions the JIT
//! and the earlier fallback already handled. This module is the last resort for
//! every other vector or scalar Advanced SIMD data-processing word: the whole
//! three-same, two-register-miscellaneous, across-lanes, three-different,
//! shift-by-immediate and by-element classes, integer and floating point, in
//! their vector and scalar forms.
//!
//! Each register is a `u128` and each lane a bit field of it. One function per
//! encoding class decodes and executes in a single pass against a bare register
//! file, so [`is_supported`] simply dry-runs the word on scratch registers and
//! classification can never disagree with execution.
//!
//! Floating point follows [`fp`](super::fp): round to nearest even, no `FPCR`
//! modes, no exception flags, no half precision. Saturating instructions set
//! `FPSR.QC`.

use super::Cpu;
use super::fp::{Binary, Float, Fused, Unary, binary, fused, nan_or, unary};

type Regs = [[u64; 2]; 32];

/// Whether [`run`] implements `word`.
pub fn is_supported(word: u32) -> bool {
    let mut v = [[0u64; 2]; 32];
    let mut fpsr = 0;
    exec(&mut v, &mut fpsr, word)
}

/// Run `word` on `cpu`. `word` must satisfy [`is_supported`].
pub fn run(cpu: &mut Cpu, word: u32) {
    let done = exec(&mut cpu.v, &mut cpu.system.fpsr, word);
    debug_assert!(done, "an unsupported NEON word reached the executor");
}

#[derive(Clone, Copy)]
struct W {
    word: u32,
    rd: usize,
    rn: usize,
    rm: usize,
    q: bool,
    u: bool,
    size: u32,
}

struct Ctx<'a> {
    v: &'a mut Regs,
    qc: bool,
}

impl Ctx<'_> {
    fn get(&self, r: usize) -> u128 {
        u128::from(self.v[r][0]) | u128::from(self.v[r][1]) << 64
    }
    /// Write `x`; without `full` the upper 64 bits are cleared.
    fn put(&mut self, r: usize, x: u128, full: bool) {
        self.v[r] = [x as u64, if full { (x >> 64) as u64 } else { 0 }];
    }
    /// `x` clamped to a signed `bits`-bit lane, as lane bits.
    fn sat_s(&mut self, x: i128, bits: u32) -> u128 {
        let max = (1i128 << (bits - 1)) - 1;
        let r = x.clamp(-max - 1, max);
        self.qc |= r != x;
        r as u128 & mask(bits)
    }
    /// `x` clamped to an unsigned `bits`-bit lane, as lane bits.
    fn sat_u(&mut self, x: i128, bits: u32) -> u128 {
        let max = (1i128 << bits) - 1;
        let r = x.clamp(0, max);
        self.qc |= r != x;
        r as u128
    }
}

fn mask(bits: u32) -> u128 {
    if bits >= 128 {
        u128::MAX
    } else {
        (1 << bits) - 1
    }
}
fn lane(x: u128, bits: u32, i: u32) -> u128 {
    (x >> (bits * i)) & mask(bits)
}
fn set_lane(x: &mut u128, bits: u32, i: u32, value: u128) {
    let field = mask(bits) << (bits * i);
    *x = (*x & !field) | ((value << (bits * i)) & field);
}
fn sx(x: u128, bits: u32) -> i128 {
    let shift = 128 - bits;
    ((x << shift) as i128) >> shift
}
fn ext(x: u128, bits: u32, signed: bool) -> i128 {
    if signed { sx(x, bits) } else { x as i128 }
}
fn lanes(bits: u32, q: bool) -> u32 {
    (if q { 128 } else { 64 }) / bits
}
/// All ones when `condition`, as lane bits.
fn flag(condition: bool) -> u128 {
    if condition { u128::MAX } else { 0 }
}

fn exec(v: &mut Regs, fpsr: &mut u64, word: u32) -> bool {
    let w = W {
        word,
        rd: (word & 31) as usize,
        rn: ((word >> 5) & 31) as usize,
        rm: ((word >> 16) & 31) as usize,
        q: word & 0x4000_0000 != 0,
        u: word & 0x2000_0000 != 0,
        size: (word >> 22) & 3,
    };
    let mut c = Ctx { v, qc: false };
    let done = if word & 0x9f20_0400 == 0x0e20_0400 {
        three_same(&mut c, w, false)
    } else if word & 0x9f3e_0c00 == 0x0e20_0800 {
        misc(&mut c, w, false)
    } else if word & 0x9f3e_0c00 == 0x0e30_0800 {
        across(&mut c, w)
    } else if word & 0x9f20_0c00 == 0x0e20_0000 {
        three_diff(&mut c, w, false)
    } else if word & 0x9f80_0400 == 0x0f00_0400 {
        shift_imm(&mut c, w, false)
    } else if word & 0x9f00_0400 == 0x0f00_0000 {
        by_element(&mut c, w, false)
    } else if word & 0xdf20_0400 == 0x5e20_0400 {
        three_same(&mut c, w, true)
    } else if word & 0xdf3e_0c00 == 0x5e20_0800 {
        misc(&mut c, w, true)
    } else if word & 0xdf3e_0c00 == 0x5e30_0800 {
        pairwise_scalar(&mut c, w)
    } else if word & 0xdf20_0c00 == 0x5e20_0000 {
        three_diff(&mut c, w, true)
    } else if word & 0xdf80_0400 == 0x5f00_0400 {
        shift_imm(&mut c, w, true)
    } else if word & 0xdf00_0400 == 0x5f00_0000 {
        by_element(&mut c, w, true)
    } else {
        false
    };
    if c.qc {
        *fpsr |= 1 << 27;
    }
    done
}

// ---- Three registers, same type ------------------------------------------------

/// One lane of a shift by a register: `sh` is the signed shift (negative shifts
/// right), with optional rounding and saturation.
fn shift_lane(
    c: &mut Ctx,
    a: u128,
    sh: i32,
    bits: u32,
    signed: bool,
    round: bool,
    sat: bool,
) -> u128 {
    let x = ext(a, bits, signed);
    if sh >= 0 {
        let sh = sh as u32;
        if x == 0 {
            return 0;
        }
        if !sat {
            return if sh >= bits {
                0
            } else {
                (x << sh) as u128 & mask(bits)
            };
        }
        let wide = if sh >= 64 {
            if x < 0 { -(1i128 << 100) } else { 1i128 << 100 }
        } else {
            x << sh
        };
        return if signed {
            c.sat_s(wide, bits)
        } else {
            c.sat_u(wide, bits)
        };
    }
    let n = (-sh).min(100) as u32;
    let r = if round { x + (1i128 << (n - 1)) } else { x };
    (r >> n) as u128 & mask(bits)
}

fn sqdmulh(c: &mut Ctx, a: u128, b: u128, bits: u32, round: bool) -> u128 {
    let mut p = 2 * sx(a, bits) * sx(b, bits);
    if round {
        p += 1i128 << (bits - 1);
    }
    c.sat_s(p >> bits, bits)
}

fn int_lane(c: &mut Ctx, op: u32, u: bool, a: u128, b: u128, d: u128, bits: u32) -> u128 {
    let (sa, sb) = (sx(a, bits), sx(b, bits));
    let (ua, ub) = (a as i128, b as i128);
    let sh = (b & 0xff) as u8 as i8 as i32;
    match (op, u) {
        (0, false) => ((sa + sb) >> 1) as u128,
        (0, true) => ((ua + ub) >> 1) as u128,
        (1, false) => c.sat_s(sa + sb, bits),
        (1, true) => c.sat_u(ua + ub, bits),
        (2, false) => ((sa + sb + 1) >> 1) as u128,
        (2, true) => ((ua + ub + 1) >> 1) as u128,
        (4, false) => ((sa - sb) >> 1) as u128,
        (4, true) => ((ua - ub) >> 1) as u128,
        (5, false) => c.sat_s(sa - sb, bits),
        (5, true) => c.sat_u(ua - ub, bits),
        (6, false) => flag(sa > sb),
        (6, true) => flag(a > b),
        (7, false) => flag(sa >= sb),
        (7, true) => flag(a >= b),
        (8, _) => shift_lane(c, a, sh, bits, !u, false, false),
        (9, _) => shift_lane(c, a, sh, bits, !u, false, true),
        (10, _) => shift_lane(c, a, sh, bits, !u, true, false),
        (11, _) => shift_lane(c, a, sh, bits, !u, true, true),
        (12, false) => sa.max(sb) as u128,
        (12, true) => ua.max(ub) as u128,
        (13, false) => sa.min(sb) as u128,
        (13, true) => ua.min(ub) as u128,
        (14, false) => (sa - sb).unsigned_abs(),
        (14, true) => (ua - ub).unsigned_abs(),
        (15, false) => d.wrapping_add((sa - sb).unsigned_abs()),
        (15, true) => d.wrapping_add((ua - ub).unsigned_abs()),
        (16, false) => a.wrapping_add(b),
        (16, true) => a.wrapping_sub(b),
        (17, false) => flag(a & b != 0),
        (17, true) => flag(a == b),
        (18, false) => d.wrapping_add(a.wrapping_mul(b)),
        (18, true) => d.wrapping_sub(a.wrapping_mul(b)),
        (19, false) => a.wrapping_mul(b),
        (19, true) => (0..8).fold(0, |r, k| if (b >> k) & 1 != 0 { r ^ (a << k) } else { r }),
        (22, false) => sqdmulh(c, a, b, bits, false),
        (22, true) => sqdmulh(c, a, b, bits, true),
        _ => 0,
    }
}

fn three_same(c: &mut Ctx, w: W, scalar: bool) -> bool {
    let op = (w.word >> 11) & 31;
    if op >= 0x18 {
        return fp_three(c, w, scalar, op);
    }
    let (u, size) = (w.u, w.size);
    let bits = 8 << size;
    if op == 3 {
        return !scalar && logic(c, w);
    }
    let wide_ok = matches!(op, 1 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 16 | 17 | 23);
    if size == 3 && (!wide_ok || (!scalar && !w.q)) {
        return false;
    }
    if scalar {
        if !matches!(op, 1 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 16 | 17 | 22) {
            return false;
        }
        if matches!(op, 6 | 7 | 8 | 10 | 16 | 17) && size != 3 {
            return false;
        }
    }
    match op {
        22 if size == 0 || size == 3 => return false,
        23 if u => return false,
        19 if u && size != 0 => return false,
        _ => {}
    }
    let n = if scalar { 1 } else { lanes(bits, w.q) };
    let (ra, rb, rd) = (c.get(w.rn), c.get(w.rm), c.get(w.rd));
    let mut out = 0;
    if matches!(op, 20 | 21 | 23) {
        let cat = |j: u32| {
            if j < n {
                lane(ra, bits, j)
            } else {
                lane(rb, bits, j - n)
            }
        };
        for i in 0..n {
            let (x, y) = (cat(2 * i), cat(2 * i + 1));
            let r = match (op, u) {
                (23, _) => x.wrapping_add(y),
                (20, false) => sx(x, bits).max(sx(y, bits)) as u128,
                (20, true) => x.max(y),
                (21, false) => sx(x, bits).min(sx(y, bits)) as u128,
                _ => x.min(y),
            };
            set_lane(&mut out, bits, i, r);
        }
    } else {
        for i in 0..n {
            let (a, b, d) = (lane(ra, bits, i), lane(rb, bits, i), lane(rd, bits, i));
            let r = int_lane(c, op, u, a, b, d, bits);
            set_lane(&mut out, bits, i, r);
        }
    }
    c.put(w.rd, out, !scalar && w.q);
    true
}

fn logic(c: &mut Ctx, w: W) -> bool {
    let (a, b, d) = (c.get(w.rn), c.get(w.rm), c.get(w.rd));
    let r = match (w.u, w.size) {
        (false, 0) => a & b,
        (false, 1) => a & !b,
        (false, 2) => a | b,
        (false, _) => a | !b,
        (true, 0) => a ^ b,
        (true, 1) => (d & a) | (!d & b),
        (true, 2) => (d & !b) | (a & b),
        (true, _) => (d & b) | (a & !b),
    };
    c.put(w.rd, r, w.q);
    true
}

// ---- Floating point lanes --------------------------------------------------------

trait Fl: Float {
    /// Exponent and fraction widths.
    const E: u32;
    const F: u32;
    fn rint(self) -> Self;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn trunc(self) -> Self;
    fn round(self) -> Self;
    fn to_f64(self) -> f64;
    fn from_i128(v: i128) -> Self;
    fn from_u128(v: u128) -> Self;
    fn ldexp(self, n: i32) -> Self;
    fn abs(self) -> Self {
        Self::from_u64(self.to_u64() & !Self::SIGN)
    }
    fn is_zero(self) -> bool {
        self.to_u64() & !Self::SIGN == 0
    }
    fn from_f64_value(v: f64) -> Self;
}
impl Fl for f32 {
    const E: u32 = 8;
    const F: u32 = 23;
    fn rint(self) -> Self {
        libm::rintf(self)
    }
    fn floor(self) -> Self {
        libm::floorf(self)
    }
    fn ceil(self) -> Self {
        libm::ceilf(self)
    }
    fn trunc(self) -> Self {
        libm::truncf(self)
    }
    fn round(self) -> Self {
        libm::roundf(self)
    }
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    fn from_i128(v: i128) -> Self {
        v as f32
    }
    fn from_u128(v: u128) -> Self {
        v as f32
    }
    fn ldexp(self, n: i32) -> Self {
        libm::ldexpf(self, n)
    }
    fn from_f64_value(v: f64) -> Self {
        v as f32
    }
}
impl Fl for f64 {
    const E: u32 = 11;
    const F: u32 = 52;
    fn rint(self) -> Self {
        libm::rint(self)
    }
    fn floor(self) -> Self {
        libm::floor(self)
    }
    fn ceil(self) -> Self {
        libm::ceil(self)
    }
    fn trunc(self) -> Self {
        libm::trunc(self)
    }
    fn round(self) -> Self {
        libm::round(self)
    }
    fn to_f64(self) -> f64 {
        self
    }
    fn from_i128(v: i128) -> Self {
        v as f64
    }
    fn from_u128(v: u128) -> Self {
        v as f64
    }
    fn ldexp(self, n: i32) -> Self {
        libm::ldexp(self, n)
    }
    fn from_f64_value(v: f64) -> Self {
        v
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum F3 {
    MaxNm,
    MinNm,
    Max,
    Min,
    Add,
    Sub,
    Mul,
    Div,
    Mla,
    Mls,
    Mulx,
    Abd,
    Recps,
    Rsqrts,
    Eq,
    Ge,
    Gt,
    Acge,
    Acgt,
}

fn is_inf<T: Fl>(x: T) -> bool {
    let e = (x.to_u64() >> T::F) & ((1 << T::E) - 1);
    e == (1 << T::E) - 1 && x.to_u64() & ((1 << T::F) - 1) == 0
}
fn two<T: Fl>(sign_of: T) -> T {
    // 2.0 with the sign of `sign_of`, built from the exponent field.
    let bias = (1u64 << (T::E - 1)) - 1;
    T::from_u64((sign_of.to_u64() & T::SIGN) | ((bias + 1) << T::F))
}
fn f3<T: Fl>(k: F3, a: T, b: T, d: T) -> u64 {
    let lane_mask = |c: bool| if c { u64::MAX } else { 0 };
    let r = match k {
        F3::Add => binary(Binary::Add, a, b),
        F3::Sub => binary(Binary::Sub, a, b),
        F3::Mul => binary(Binary::Mul, a, b),
        F3::Div => binary(Binary::Div, a, b),
        F3::Max => binary(Binary::Max, a, b),
        F3::Min => binary(Binary::Min, a, b),
        F3::MaxNm => binary(Binary::MaxNum, a, b),
        F3::MinNm => binary(Binary::MinNum, a, b),
        F3::Mla => fused(Fused::Madd, a, b, d),
        F3::Mls => fused(Fused::Msub, a, b, d),
        F3::Abd => unary(Unary::Abs, binary(Binary::Sub, a, b)),
        F3::Mulx => {
            if (a.is_zero() && is_inf(b)) || (is_inf(a) && b.is_zero()) {
                two(T::from_u64((a.to_u64() ^ b.to_u64()) & T::SIGN))
            } else {
                binary(Binary::Mul, a, b)
            }
        }
        F3::Recps => {
            if (a.is_zero() && is_inf(b)) || (is_inf(a) && b.is_zero()) {
                two(T::from_u64(0))
            } else {
                fused(Fused::Msub, a, b, two(T::from_u64(0)))
            }
        }
        F3::Rsqrts => {
            // (3 - a*b) / 2 with the halving done before the single rounding, so
            // a huge product does not overflow: 1.5 - (a/2)*b, halving the larger
            // factor to keep that exact. NaNs are processed with `a` negated.
            if a.is_nan() || b.is_nan() {
                return nan_or(T::default_nan(), &[a.neg(), b]).to_u64();
            }
            if (a.is_zero() && is_inf(b)) || (is_inf(a) && b.is_zero()) {
                T::from_f64_value(1.5)
            } else if a.abs() >= b.abs() {
                fused(Fused::Msub, a.ldexp(-1), b, T::from_f64_value(1.5))
            } else {
                fused(Fused::Msub, a, b.ldexp(-1), T::from_f64_value(1.5))
            }
        }
        F3::Eq => return lane_mask(a == b),
        F3::Ge => return lane_mask(a >= b),
        F3::Gt => return lane_mask(a > b),
        F3::Acge => return lane_mask(a.abs() >= b.abs()),
        F3::Acgt => return lane_mask(a.abs() > b.abs()),
    };
    r.to_u64()
}

fn fp3_lane(k: F3, double: bool, a: u64, b: u64, d: u64) -> u64 {
    if double {
        f3(k, f64::from_u64(a), f64::from_u64(b), f64::from_u64(d))
    } else {
        f3(k, f32::from_u64(a), f32::from_u64(b), f32::from_u64(d))
    }
}

fn fp_three(c: &mut Ctx, w: W, scalar: bool, op: u32) -> bool {
    let (a_bit, double) = (w.size >> 1, w.size & 1 == 1);
    if double && !scalar && !w.q {
        return false;
    }
    let (k, pair) = match (w.u, op, a_bit) {
        (false, 0x18, 0) => (F3::MaxNm, false),
        (false, 0x18, _) => (F3::MinNm, false),
        (false, 0x19, 0) => (F3::Mla, false),
        (false, 0x19, _) => (F3::Mls, false),
        (false, 0x1a, 0) => (F3::Add, false),
        (false, 0x1a, _) => (F3::Sub, false),
        (false, 0x1b, 0) => (F3::Mulx, false),
        (false, 0x1c, 0) => (F3::Eq, false),
        (false, 0x1e, 0) => (F3::Max, false),
        (false, 0x1e, _) => (F3::Min, false),
        (false, 0x1f, 0) => (F3::Recps, false),
        (false, 0x1f, _) => (F3::Rsqrts, false),
        (true, 0x18, 0) => (F3::MaxNm, true),
        (true, 0x18, _) => (F3::MinNm, true),
        (true, 0x1a, 0) => (F3::Add, true),
        (true, 0x1a, _) => (F3::Abd, false),
        (true, 0x1b, 0) => (F3::Mul, false),
        (true, 0x1c, 0) => (F3::Ge, false),
        (true, 0x1c, _) => (F3::Gt, false),
        (true, 0x1d, 0) => (F3::Acge, false),
        (true, 0x1d, _) => (F3::Acgt, false),
        (true, 0x1e, 0) => (F3::Max, true),
        (true, 0x1e, _) => (F3::Min, true),
        (true, 0x1f, 0) => (F3::Div, false),
        _ => return false,
    };
    if scalar
        && !matches!(
            k,
            F3::Mulx
                | F3::Recps
                | F3::Rsqrts
                | F3::Eq
                | F3::Ge
                | F3::Gt
                | F3::Acge
                | F3::Acgt
                | F3::Abd
        )
    {
        return false;
    }
    let bits = if double { 64 } else { 32 };
    let n = if scalar { 1 } else { lanes(bits, w.q) };
    let (ra, rb, rd) = (c.get(w.rn), c.get(w.rm), c.get(w.rd));
    let mut out = 0;
    for i in 0..n {
        let r = if pair {
            let cat = |j: u32| {
                if j < n {
                    lane(ra, bits, j)
                } else {
                    lane(rb, bits, j - n)
                }
            };
            fp3_lane(k, double, cat(2 * i) as u64, cat(2 * i + 1) as u64, 0)
        } else {
            let (a, b, d) = (lane(ra, bits, i), lane(rb, bits, i), lane(rd, bits, i));
            fp3_lane(k, double, a as u64, b as u64, d as u64)
        };
        set_lane(&mut out, bits, i, u128::from(r));
    }
    c.put(w.rd, out, !scalar && w.q);
    true
}

// ---- Floating point estimates and conversions ----------------------------------------

fn recip_estimate(a: u64) -> u64 {
    let a = a * 2 + 1;
    let b = (1 << 19) / a;
    b.div_ceil(2)
}
fn rsqrt_estimate(a: u64) -> u64 {
    let a = if a < 256 {
        a * 2 + 1
    } else {
        (((a >> 1) << 1) + 1) * 2
    };
    let mut b = 512;
    while a * (b + 1) * (b + 1) < 1 << 28 {
        b += 1;
    }
    // `(b + 1) DIV 2`, rounding to nearest.
    b.div_ceil(2)
}

fn frecpe<T: Fl>(x: T) -> u64 {
    let (e, f) = (T::E, T::F);
    let bits = x.to_u64();
    let sign = bits & T::SIGN;
    let exp = (bits >> f) & ((1 << e) - 1);
    let frac = bits & ((1 << f) - 1);
    let max = (1u64 << e) - 1;
    if exp == max && frac != 0 {
        return bits | T::QUIET;
    }
    if exp == max {
        return sign;
    }
    if exp == 0 && frac >> (f - 2) == 0 {
        return sign | (max << f);
    }
    let (mut exp_i, mut frac_i) = (exp as i64, frac);
    if exp == 0 {
        if frac >> (f - 1) == 0 {
            exp_i = -1;
            frac_i = (frac << 2) & ((1 << f) - 1);
        } else {
            frac_i = (frac << 1) & ((1 << f) - 1);
        }
    }
    let scaled = (1 << 8) | (frac_i >> (f - 8));
    let bias = (1i64 << (e - 1)) - 1;
    let mut result_exp = 2 * bias - 1 - exp_i;
    let mut fraction = (recip_estimate(scaled) & 0xff) << (f - 8);
    if result_exp == 0 {
        fraction = (1 << (f - 1)) | (fraction >> 1);
    } else if result_exp == -1 {
        fraction = (1 << (f - 2)) | (fraction >> 2);
        result_exp = 0;
    }
    sign | ((result_exp as u64) << f) | fraction
}

fn frsqrte<T: Fl>(x: T) -> u64 {
    let (e, f) = (T::E, T::F);
    let bits = x.to_u64();
    let sign = bits & T::SIGN;
    let exp = (bits >> f) & ((1 << e) - 1);
    let frac = bits & ((1 << f) - 1);
    let max = (1u64 << e) - 1;
    if exp == max && frac != 0 {
        return bits | T::QUIET;
    }
    if exp == 0 && frac == 0 {
        return sign | (max << f);
    }
    if sign != 0 {
        return T::default_nan().to_u64();
    }
    if exp == max {
        return 0;
    }
    let (mut exp_i, mut frac_i) = (exp as i64, frac);
    if exp == 0 {
        while frac_i >> (f - 1) == 0 {
            frac_i = (frac_i << 1) & ((1 << f) - 1);
            exp_i -= 1;
        }
        frac_i = (frac_i << 1) & ((1 << f) - 1);
    }
    let scaled = if exp_i & 1 == 0 {
        (1 << 8) | (frac_i >> (f - 8))
    } else {
        (1 << 7) | (frac_i >> (f - 7))
    };
    let bias = (1i64 << (e - 1)) - 1;
    let result_exp = (3 * bias - 1 - exp_i) >> 1;
    ((result_exp as u64) << f) | ((rsqrt_estimate(scaled) & 0xff) << (f - 8))
}

fn frecpx<T: Fl>(x: T) -> u64 {
    let (e, f) = (T::E, T::F);
    let bits = x.to_u64();
    let exp = (bits >> f) & ((1 << e) - 1);
    if exp == (1 << e) - 1 && bits & ((1 << f) - 1) != 0 {
        return bits | T::QUIET;
    }
    let result_exp = if exp == 0 {
        (1 << e) - 2
    } else {
        !exp & ((1 << e) - 1)
    };
    (bits & T::SIGN) | (result_exp << f)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Round {
    Nearest,
    Down,
    Up,
    Zero,
    Away,
}

/// `x` rounded as `mode` says, in the float's own precision.
fn round_float<T: Fl>(x: T, mode: Round) -> T {
    if x.is_nan() {
        return x.quiet();
    }
    match mode {
        Round::Nearest => x.rint(),
        Round::Down => x.floor(),
        Round::Up => x.ceil(),
        Round::Zero => x.trunc(),
        Round::Away => x.round(),
    }
}

/// `x * 2^fbits` rounded as `mode`, saturated to a `bits`-bit integer; NaN is 0.
fn float_to_int<T: Fl>(x: T, mode: Round, signed: bool, bits: u32, fbits: u32) -> u64 {
    if x.is_nan() {
        return 0;
    }
    let scaled = libm::ldexp(x.to_f64(), fbits as i32);
    let r = match mode {
        Round::Nearest => libm::rint(scaled),
        Round::Down => libm::floor(scaled),
        Round::Up => libm::ceil(scaled),
        Round::Zero => libm::trunc(scaled),
        Round::Away => libm::round(scaled),
    };
    let v = r as i128;
    let v = if signed {
        v.clamp(-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1)
    } else {
        v.clamp(0, (1i128 << bits) - 1)
    };
    v as u64
}

fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let frac = b & 0x7f_ffff;
    if exp == 0xff {
        return sign
            | 0x7c00
            | if frac != 0 {
                0x200 | (frac >> 13) as u16
            } else {
                0
            };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        // Subnormal half: the value in units of 2^-24 is (1.frac << 23) >> s.
        let m = if exp == 0 { frac } else { frac | 0x80_0000 };
        let s = if exp == 0 { 25 } else { (14 - e) as u32 };
        if s >= 25 {
            return sign;
        }
        let (q, r, half) = (m >> s, m & ((1 << s) - 1), 1u32 << (s - 1));
        let q = if r > half || (r == half && q & 1 == 1) {
            q + 1
        } else {
            q
        };
        return sign | q as u16;
    }
    let (m, rem) = ((frac >> 13) as u16, frac & 0x1fff);
    let mut r = ((e as u16) << 10) | m;
    if rem > 0x1000 || (rem == 0x1000 && m & 1 == 1) {
        r += 1;
    }
    sign | r
}
fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let exp = u32::from((h >> 10) & 0x1f);
    let frac = u32::from(h & 0x3ff);
    if exp == 0 {
        let v = frac as f32 * libm::ldexpf(1.0, -24);
        return if sign != 0 { -v } else { v };
    }
    if exp == 31 {
        let nan = if frac != 0 { 0x40_0000 } else { 0 };
        return f32::from_bits(sign | 0x7f80_0000 | nan | (frac << 13));
    }
    f32::from_bits(sign | ((exp + 112) << 23) | (frac << 13))
}
/// `f64` to `f32`, rounding to odd (`fcvtxn`).
fn f64_to_f32_odd(x: f64) -> f32 {
    if x.is_nan() || x.is_infinite() {
        return x as f32;
    }
    let r = x as f32;
    if f64::from(r) == x {
        return r;
    }
    let mut bits = r.to_bits();
    if libm::fabs(f64::from(r)) > libm::fabs(x) {
        bits -= 1;
    }
    f32::from_bits(bits | 1)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum F1 {
    Abs,
    Neg,
    Sqrt,
    Rint(Round),
    Rintx,
    Recpe,
    Rsqrte,
    Recpx,
    Scvtf,
    Ucvtf,
    ToInt(Round, bool),
    CmpGt,
    CmpGe,
    CmpEq,
    CmpLe,
    CmpLt,
}

fn f1<T: Fl>(k: F1, x: T, bits: u32, fbits: u32) -> u64 {
    let lane_mask = |c: bool| if c { u64::MAX } else { 0 };
    let zero = T::from_u64(0);
    match k {
        F1::Abs => unary(Unary::Abs, x).to_u64(),
        F1::Neg => unary(Unary::Neg, x).to_u64(),
        F1::Sqrt => unary(Unary::Sqrt, x).to_u64(),
        F1::Rint(m) => round_float(x, m).to_u64(),
        F1::Rintx => round_float(x, Round::Nearest).to_u64(),
        F1::Recpe => frecpe(x),
        F1::Rsqrte => frsqrte(x),
        F1::Recpx => frecpx(x),
        F1::Scvtf => T::from_i128(sx(u128::from(x.to_u64()), bits))
            .ldexp(-(fbits as i32))
            .to_u64(),
        F1::Ucvtf => T::from_u128(u128::from(x.to_u64()))
            .ldexp(-(fbits as i32))
            .to_u64(),
        F1::ToInt(m, signed) => float_to_int(x, m, signed, bits, fbits),
        F1::CmpGt => lane_mask(x > zero),
        F1::CmpGe => lane_mask(x >= zero),
        F1::CmpEq => lane_mask(x == zero),
        F1::CmpLe => lane_mask(x <= zero),
        F1::CmpLt => lane_mask(x < zero),
    }
}

fn fp1_lane(k: F1, double: bool, x: u64, fbits: u32) -> u64 {
    if double {
        f1(k, f64::from_u64(x), 64, fbits)
    } else {
        f1(k, f32::from_u64(x), 32, fbits)
    }
}

// ---- Two-register miscellaneous ----------------------------------------------------------

fn misc(c: &mut Ctx, w: W, scalar: bool) -> bool {
    let op = (w.word >> 12) & 31;
    let (u, size) = (w.u, w.size);
    let bits = 8 << size;
    match op {
        0x0c..=0x0f | 0x16..=0x1f => return fp_misc(c, w, scalar, op),
        _ => {}
    }
    let (ra, rd) = (c.get(w.rn), c.get(w.rd));
    let full = !scalar && w.q;
    let n = if scalar { 1 } else { lanes(bits, w.q) };
    let wide_ok = matches!(op, 3 | 7 | 8 | 9 | 10 | 11);
    if size == 3 && (!wide_ok || (!scalar && !w.q)) {
        return false;
    }
    if scalar {
        match op {
            3 | 7 => {}
            8..=11 if size == 3 && !(op == 10 && u) => {}
            0x12 | 0x14 if size < 3 && !(op == 0x12 && !u) => {}
            _ => return false,
        }
    }
    let mut out = 0u128;
    match op {
        0 | 1 => {
            let group = match (u, op) {
                (false, 0) if size < 3 => 64,
                (true, 0) if size < 2 => 32,
                (false, 1) if size == 0 => 16,
                _ => return false,
            };
            let gl = group / bits;
            for i in 0..n {
                let (g, p) = (i / gl, i % gl);
                set_lane(&mut out, bits, g * gl + (gl - 1 - p), lane(ra, bits, i));
            }
        }
        2 | 6 => {
            if size == 3 {
                return false;
            }
            let wb = bits * 2;
            for i in 0..n / 2 {
                let (x, y) = (lane(ra, bits, 2 * i), lane(ra, bits, 2 * i + 1));
                let mut r = (ext(x, bits, !u) + ext(y, bits, !u)) as u128;
                if op == 6 {
                    r = r.wrapping_add(lane(rd, wb, i));
                }
                set_lane(&mut out, wb, i, r);
            }
        }
        3 => {
            for i in 0..n {
                let (a, d) = (lane(ra, bits, i), lane(rd, bits, i));
                let r = if u {
                    // usqadd: unsigned accumulator plus signed operand.
                    c.sat_u(d as i128 + sx(a, bits), bits)
                } else {
                    // suqadd: signed accumulator plus unsigned operand.
                    c.sat_s(sx(d, bits) + a as i128, bits)
                };
                set_lane(&mut out, bits, i, r);
            }
        }
        4 => {
            if size == 3 {
                return false;
            }
            for i in 0..n {
                let a = lane(ra, bits, i);
                let r = if u {
                    (a.leading_zeros() - (128 - bits)) as u128
                } else {
                    let s = sx(a, bits);
                    let y = if s < 0 { !s } else { s } as u128;
                    (y.leading_zeros() - (128 - bits) - 1) as u128
                };
                set_lane(&mut out, bits, i, r);
            }
        }
        5 => {
            let q = w.q;
            match (u, size) {
                (false, 0) => {
                    for i in 0..lanes(8, q) {
                        set_lane(&mut out, 8, i, u128::from(lane(ra, 8, i).count_ones()));
                    }
                }
                (true, 0) => out = !ra,
                (true, 1) => {
                    for i in 0..lanes(8, q) {
                        set_lane(
                            &mut out,
                            8,
                            i,
                            u128::from((lane(ra, 8, i) as u8).reverse_bits()),
                        );
                    }
                }
                _ => return false,
            }
        }
        7 => {
            for i in 0..n {
                let s = sx(lane(ra, bits, i), bits);
                let r = c.sat_s(if u { -s } else { s.abs() }, bits);
                set_lane(&mut out, bits, i, r);
            }
        }
        8..=11 => {
            for i in 0..n {
                let (a, s) = (lane(ra, bits, i), sx(lane(ra, bits, i), bits));
                let r = match (op, u) {
                    (8, false) => flag(s > 0),
                    (8, true) => flag(s >= 0),
                    (9, false) => flag(s == 0),
                    (9, true) => flag(s <= 0),
                    (10, false) => flag(s < 0),
                    (11, false) => s.unsigned_abs(),
                    (11, true) => a.wrapping_neg(),
                    _ => return false,
                };
                set_lane(&mut out, bits, i, r);
            }
        }
        0x12 | 0x14 => {
            // Narrowing: `size` is the destination lane size.
            if size == 3 {
                return false;
            }
            let wb = bits * 2;
            let count = if scalar { 1 } else { 64 / bits };
            for i in 0..count {
                let x = lane(ra, wb, i);
                let r = match (op, u) {
                    (0x12, false) => x & mask(bits),
                    (0x12, true) => c.sat_u(sx(x, wb), bits),
                    (_, false) => c.sat_s(sx(x, wb), bits),
                    (_, true) => c.sat_u(x as i128, bits),
                };
                set_lane(&mut out, bits, i, r);
            }
            if !scalar && w.q {
                out = (rd & mask(64)) | (out << 64);
                c.put(w.rd, out, true);
            } else {
                c.put(w.rd, out, false);
            }
            return true;
        }
        0x13 => {
            // shll/shll2: widen and shift left by the source lane size.
            if !u || size == 3 {
                return false;
            }
            let src = if w.q { ra >> 64 } else { ra };
            for i in 0..64 / bits {
                set_lane(&mut out, bits * 2, i, lane(src, bits, i) << bits);
            }
            c.put(w.rd, out, true);
            return true;
        }
        _ => return false,
    }
    // Unsigned pairwise and the accumulating forms write full width.
    c.put(w.rd, out, full);
    true
}

fn fp_misc(c: &mut Ctx, w: W, scalar: bool, op: u32) -> bool {
    let (u, a_bit, double) = (w.u, w.size >> 1, w.size & 1 == 1);
    let bits = if double { 64 } else { 32 };
    let ra = c.get(w.rn);
    // Conversions between widths and the integer-only estimates have their own shape.
    match (op, u, a_bit) {
        (0x16, false, 0) | (0x16, true, 0) | (0x17, false, 0) => {
            return fp_convert_width(c, w, scalar, op);
        }
        (0x1c, _, 1) => return urecpe(c, w, scalar, op),
        _ => {}
    }
    if double && !scalar && !w.q {
        return false;
    }
    let k = match (op, u, a_bit) {
        (0x0c, false, 1) => F1::CmpGt,
        (0x0c, true, 1) => F1::CmpGe,
        (0x0d, false, 1) => F1::CmpEq,
        (0x0d, true, 1) => F1::CmpLe,
        (0x0e, false, 1) => F1::CmpLt,
        (0x0f, false, 1) if !scalar => F1::Abs,
        (0x0f, true, 1) if !scalar => F1::Neg,
        (0x18, false, 0) if !scalar => F1::Rint(Round::Nearest),
        (0x18, false, 1) if !scalar => F1::Rint(Round::Up),
        (0x18, true, 0) if !scalar => F1::Rint(Round::Away),
        (0x19, false, 0) if !scalar => F1::Rint(Round::Down),
        (0x19, false, 1) if !scalar => F1::Rint(Round::Zero),
        (0x19, true, 0) if !scalar => F1::Rintx,
        (0x19, true, 1) if !scalar => F1::Rint(Round::Nearest),
        (0x1a, _, 0) => F1::ToInt(Round::Nearest, !u),
        (0x1a, _, _) => F1::ToInt(Round::Up, !u),
        (0x1b, _, 0) => F1::ToInt(Round::Down, !u),
        (0x1b, _, _) => F1::ToInt(Round::Zero, !u),
        (0x1c, _, 0) => F1::ToInt(Round::Away, !u),
        (0x1d, false, 0) => F1::Scvtf,
        (0x1d, true, 0) => F1::Ucvtf,
        (0x1d, false, 1) => F1::Recpe,
        (0x1d, true, 1) => F1::Rsqrte,
        (0x1f, true, 1) if !scalar => F1::Sqrt,
        (0x1f, false, 1) if scalar => F1::Recpx,
        _ => return false,
    };
    // The integer-source conversions read the lane as an integer, so they pass
    // through `f1` as raw bits; the compares and conversions need nothing extra.
    let n = if scalar { 1 } else { lanes(bits, w.q) };
    let mut out = 0;
    for i in 0..n {
        let x = lane(ra, bits, i) as u64;
        set_lane(&mut out, bits, i, u128::from(fp1_lane(k, double, x, 0)));
    }
    c.put(w.rd, out, !scalar && w.q);
    true
}

/// `fcvtn`, `fcvtxn` and `fcvtl`.
fn fp_convert_width(c: &mut Ctx, w: W, scalar: bool, op: u32) -> bool {
    let (u, double) = (w.u, w.size & 1 == 1);
    let (ra, rd) = (c.get(w.rn), c.get(w.rd));
    if op == 0x16 {
        // Narrow: sz = 0 is single to half, sz = 1 double to single.
        if scalar && !(u && double) {
            return false;
        }
        if u && !double {
            return false;
        }
        let (src, dst) = if double { (64, 32) } else { (32, 16) };
        let count = if scalar { 1 } else { 64 / dst };
        let mut out = 0u128;
        for i in 0..count {
            let x = lane(ra, src, i) as u64;
            let r = if double {
                let f = f64::from_bits(x);
                u128::from(if u { f64_to_f32_odd(f) } else { f as f32 }.to_bits())
            } else {
                u128::from(f32_to_f16(f32::from_bits(x as u32)))
            };
            set_lane(&mut out, dst, i, r);
        }
        if !scalar && w.q {
            out = (rd & mask(64)) | (out << 64);
            c.put(w.rd, out, true);
        } else {
            c.put(w.rd, out, false);
        }
        return true;
    }
    // fcvtl: sz = 0 is half to single, sz = 1 single to double.
    if scalar || u {
        return false;
    }
    let (src, dst) = if double { (32, 64) } else { (16, 32) };
    let from = if w.q { ra >> 64 } else { ra };
    let mut out = 0u128;
    for i in 0..64 / src {
        let x = lane(from, src, i);
        let r = if double {
            u128::from(f64::from(f32::from_bits(x as u32)).to_bits())
        } else {
            u128::from(f16_to_f32(x as u16).to_bits())
        };
        set_lane(&mut out, dst, i, r);
    }
    c.put(w.rd, out, true);
    true
}

/// `urecpe` and `ursqrte`.
fn urecpe(c: &mut Ctx, w: W, scalar: bool, _op: u32) -> bool {
    if scalar || w.size != 2 {
        return false;
    }
    let ra = c.get(w.rn);
    let mut out = 0;
    for i in 0..lanes(32, w.q) {
        let x = lane(ra, 32, i) as u32;
        let r = if w.u {
            if x >> 30 == 0 {
                0xffff_ffff
            } else {
                (rsqrt_estimate(u64::from(x >> 23)) & 0x1ff) << 23
            }
        } else if x >> 31 == 0 {
            0xffff_ffff
        } else {
            (recip_estimate(u64::from(x >> 23)) & 0x1ff) << 23
        };
        set_lane(&mut out, 32, i, u128::from(r));
    }
    c.put(w.rd, out, w.q);
    true
}

// ---- Across lanes and scalar pairwise ---------------------------------------------------------

fn across(c: &mut Ctx, w: W) -> bool {
    let op = (w.word >> 12) & 31;
    let (u, size) = (w.u, w.size);
    let ra = c.get(w.rn);
    if matches!(op, 0x0c | 0x0f) {
        // fmaxnmv, fminnmv, fmaxv, fminv: single precision, four lanes, U = 1.
        if !u || size & 1 != 0 || !w.q {
            return false;
        }
        let k = match (op, size >> 1) {
            (0x0c, 0) => F3::MaxNm,
            (0x0c, _) => F3::MinNm,
            (_, 0) => F3::Max,
            _ => F3::Min,
        };
        // The architecture reduces as a tree: (l0 . l1) . (l2 . l3).
        let l = |i| f32::from_u64(lane(ra, 32, i) as u64);
        let z = f32::from_u64(0);
        let (lo, hi) = (f3(k, l(0), l(1), z), f3(k, l(2), l(3), z));
        let acc = f3(k, f32::from_u64(lo), f32::from_u64(hi), z);
        c.put(w.rd, u128::from(acc), false);
        return true;
    }
    let bits = 8 << size;
    if size == 3 || (size == 2 && !w.q) {
        return false;
    }
    let n = lanes(bits, w.q);
    let (acc, rb) = match (op, u) {
        (0x03, _) => {
            let sum: i128 = (0..n).map(|i| ext(lane(ra, bits, i), bits, !u)).sum();
            (sum as u128, bits * 2)
        }
        (0x0a, _) | (0x1a, _) => {
            let mut best = ext(lane(ra, bits, 0), bits, !u);
            for i in 1..n {
                let x = ext(lane(ra, bits, i), bits, !u);
                best = if op == 0x0a { best.max(x) } else { best.min(x) };
            }
            (best as u128, bits)
        }
        (0x1b, false) => {
            let sum = (0..n).fold(0u128, |s, i| s.wrapping_add(lane(ra, bits, i)));
            (sum, bits)
        }
        _ => return false,
    };
    c.put(w.rd, acc & mask(rb), false);
    true
}

fn pairwise_scalar(c: &mut Ctx, w: W) -> bool {
    let op = (w.word >> 12) & 31;
    let ra = c.get(w.rn);
    if op == 0x1b {
        if w.u || w.size != 3 {
            return false;
        }
        c.put(
            w.rd,
            (lane(ra, 64, 0).wrapping_add(lane(ra, 64, 1))) & mask(64),
            false,
        );
        return true;
    }
    if !w.u {
        return false;
    }
    let (a_bit, double) = (w.size >> 1, w.size & 1 == 1);
    let k = match (op, a_bit) {
        (0x0c, 0) => F3::MaxNm,
        (0x0c, _) => F3::MinNm,
        (0x0d, 0) => F3::Add,
        (0x0f, 0) => F3::Max,
        (0x0f, _) => F3::Min,
        _ => return false,
    };
    let bits = if double { 64 } else { 32 };
    let r = fp3_lane(
        k,
        double,
        lane(ra, bits, 0) as u64,
        lane(ra, bits, 1) as u64,
        0,
    );
    c.put(w.rd, u128::from(r), false);
    true
}

// ---- Three registers, different sizes --------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum LongK {
    Mul,
    Mla,
    Mls,
    Dmull,
    Dmlal,
    Dmlsl,
}

/// One widening multiply lane: sources are `bits` wide, the result `2 * bits`.
fn long_lane(c: &mut Ctx, kind: LongK, signed: bool, a: u128, b: u128, d: u128, bits: u32) -> u128 {
    let (x, y) = (ext(a, bits, signed), ext(b, bits, signed));
    let wb = bits * 2;
    let product = (x * y) as u128;
    match kind {
        LongK::Mul => product & mask(wb),
        LongK::Mla => d.wrapping_add(product) & mask(wb),
        LongK::Mls => d.wrapping_sub(product) & mask(wb),
        LongK::Dmull | LongK::Dmlal | LongK::Dmlsl => {
            let p = sx(c.sat_s(2 * x * y, wb), wb);
            match kind {
                LongK::Dmull => p as u128 & mask(wb),
                LongK::Dmlal => c.sat_s(sx(d, wb) + p, wb),
                _ => c.sat_s(sx(d, wb) - p, wb),
            }
        }
    }
}

/// Carry-less multiply of `a` and `b`, `bits` wide, into `2 * bits`.
fn pmull(a: u128, b: u128, bits: u32) -> u128 {
    (0..bits).fold(0, |r, k| if (b >> k) & 1 != 0 { r ^ (a << k) } else { r })
}

fn three_diff(c: &mut Ctx, w: W, scalar: bool) -> bool {
    let op = (w.word >> 12) & 15;
    let (u, size) = (w.u, w.size);
    let bits = 8 << size;
    match op {
        0..=3 | 5 | 7 | 8 | 10 | 12 if size < 3 && !scalar => {}
        4 | 6 if size < 3 && !scalar => {}
        9 | 11 | 13 if !u && (size == 1 || size == 2) => {}
        14 if !u && !scalar && (size == 0 || size == 3) => {}
        _ => return false,
    }
    let (ra, rb, rd) = (c.get(w.rn), c.get(w.rm), c.get(w.rd));
    let mut out = 0u128;
    if op == 14 {
        // pmull/pmull2: 8x8 to 16 bits per lane, or 64x64 to 128.
        let (src, count) = if size == 0 { (8, 8) } else { (64, 1) };
        let (a, b) = if w.q { (ra >> 64, rb >> 64) } else { (ra, rb) };
        for i in 0..count {
            set_lane(
                &mut out,
                src * 2,
                i,
                pmull(lane(a, src, i), lane(b, src, i), src),
            );
        }
        c.put(w.rd, out, true);
        return true;
    }
    if matches!(op, 4 | 6) {
        // addhn/raddhn/subhn/rsubhn: high half of a wide sum, narrowed.
        let wb = bits * 2;
        for i in 0..64 / bits {
            let (x, y) = (lane(ra, wb, i), lane(rb, wb, i));
            let mut r = if op == 4 {
                x.wrapping_add(y)
            } else {
                x.wrapping_sub(y)
            };
            if u {
                // The rounding forms add half of the discarded low half.
                r = r.wrapping_add(1u128 << (bits - 1));
            }
            set_lane(&mut out, bits, i, (r & mask(wb)) >> bits);
        }
        if w.q {
            out = (rd & mask(64)) | (out << 64);
            c.put(w.rd, out, true);
        } else {
            c.put(w.rd, out, false);
        }
        return true;
    }
    let (ha, hb) = if scalar || !w.q {
        (ra, rb)
    } else {
        (ra >> 64, rb >> 64)
    };
    let wb = bits * 2;
    let count = if scalar { 1 } else { 64 / bits };
    for i in 0..count {
        let (a, b) = (lane(ha, bits, i), lane(hb, bits, i));
        let d = lane(rd, wb, i);
        let signed = !u;
        let r = match op {
            0 | 2 => {
                let (x, y) = (ext(a, bits, signed), ext(b, bits, signed));
                (if op == 0 { x + y } else { x - y }) as u128 & mask(wb)
            }
            1 | 3 => {
                let x = ext(lane(ra, wb, i), wb, signed);
                let y = ext(b, bits, signed);
                (if op == 1 { x + y } else { x - y }) as u128 & mask(wb)
            }
            5 | 7 => {
                let diff = (ext(a, bits, signed) - ext(b, bits, signed)).unsigned_abs();
                if op == 5 {
                    d.wrapping_add(diff) & mask(wb)
                } else {
                    diff
                }
            }
            8 => long_lane(c, LongK::Mla, signed, a, b, d, bits),
            9 => long_lane(c, LongK::Dmlal, true, a, b, d, bits),
            10 => long_lane(c, LongK::Mls, signed, a, b, d, bits),
            11 => long_lane(c, LongK::Dmlsl, true, a, b, d, bits),
            12 => long_lane(c, LongK::Mul, signed, a, b, d, bits),
            _ => long_lane(c, LongK::Dmull, true, a, b, d, bits),
        };
        set_lane(&mut out, wb, i, r);
    }
    c.put(w.rd, out, !scalar);
    true
}

// ---- Shift by immediate ----------------------------------------------------------------------------

fn shift_imm(c: &mut Ctx, w: W, scalar: bool) -> bool {
    let immh = (w.word >> 19) & 15;
    if immh == 0 {
        return false;
    }
    let immhb = (w.word >> 16) & 0x7f;
    let op = (w.word >> 11) & 31;
    let u = w.u;
    let size = 31 - immh.leading_zeros();
    let bits = 8 << size;
    let right = 2 * bits - immhb;
    let left = immhb.wrapping_sub(bits);
    let (ra, rd) = (c.get(w.rn), c.get(w.rd));
    let full = !scalar && w.q;
    if size == 3 && !scalar && !w.q {
        return false;
    }
    let n = if scalar { 1 } else { lanes(bits, w.q) };
    let mut out = 0u128;
    match op {
        0 | 2 | 4 | 6 => {
            if scalar && size != 3 {
                return false;
            }
            let (acc, round) = (op & 2 != 0, op & 4 != 0);
            for i in 0..n {
                let x = ext(lane(ra, bits, i), bits, !u);
                let r = if round { x + (1i128 << (right - 1)) } else { x };
                let mut r = (r >> right) as u128;
                if acc {
                    r = r.wrapping_add(lane(rd, bits, i));
                }
                set_lane(&mut out, bits, i, r);
            }
        }
        8 | 10 if u => {
            if scalar && size != 3 {
                return false;
            }
            for i in 0..n {
                let (a, d) = (lane(ra, bits, i), lane(rd, bits, i));
                let r = if op == 10 {
                    // sli: keep the low `left` bits of the destination.
                    (d & mask(left)) | (a << left)
                } else {
                    // sri: keep the high `right` bits of the destination.
                    let keep = if right >= bits {
                        0
                    } else {
                        mask(bits) >> right
                    };
                    (d & !keep) | ((a >> right) & keep)
                };
                set_lane(&mut out, bits, i, r);
            }
        }
        10 => {
            if scalar && size != 3 {
                return false;
            }
            for i in 0..n {
                set_lane(&mut out, bits, i, lane(ra, bits, i) << left);
            }
        }
        12 | 14 => {
            if op == 12 && !u {
                return false;
            }
            for i in 0..n {
                let a = lane(ra, bits, i);
                let r = if op == 12 {
                    c.sat_u(sx(a, bits) << left, bits)
                } else if u {
                    c.sat_u((a as i128) << left, bits)
                } else {
                    c.sat_s(sx(a, bits) << left, bits)
                };
                set_lane(&mut out, bits, i, r);
            }
        }
        0x10..=0x13 => {
            // Narrowing right shifts: `size` is the destination lane size.
            if size == 3 {
                return false;
            }
            let wb = bits * 2;
            let round = op & 1 != 0;
            let count = if scalar { 1 } else { 64 / bits };
            for i in 0..count {
                let a = lane(ra, wb, i);
                // Signed source except for `uqshrn`/`uqrshrn`.
                let unsigned_source = u && op >= 0x12;
                let x = ext(a, wb, !unsigned_source);
                let x = if round { x + (1i128 << (right - 1)) } else { x };
                let x = x >> right;
                let r = match (u, op >= 0x12) {
                    (false, false) => x as u128 & mask(bits),
                    (true, false) => c.sat_u(x, bits),
                    (false, true) => c.sat_s(x, bits),
                    (true, true) => c.sat_u(x, bits),
                };
                set_lane(&mut out, bits, i, r);
            }
            if full {
                out = (rd & mask(64)) | (out << 64);
                c.put(w.rd, out, true);
            } else {
                c.put(w.rd, out, false);
            }
            return true;
        }
        0x14 => {
            if scalar || size == 3 {
                return false;
            }
            let src = if w.q { ra >> 64 } else { ra };
            for i in 0..64 / bits {
                let x = ext(lane(src, bits, i), bits, !u);
                set_lane(&mut out, bits * 2, i, (x << left) as u128);
            }
            c.put(w.rd, out, true);
            return true;
        }
        0x1c | 0x1f => {
            // Fixed-point conversions: immh = 01xx is single, 1xxx double.
            let double = immh >= 8;
            if immh < 4 || (double && !scalar && !w.q) {
                return false;
            }
            let (fb, fbits) = if double {
                (64, 128 - immhb)
            } else {
                (32, 64 - immhb)
            };
            let k = match (op, u) {
                (0x1c, false) => F1::Scvtf,
                (0x1c, true) => F1::Ucvtf,
                (_, s) => F1::ToInt(Round::Zero, !s),
            };
            let count = if scalar { 1 } else { lanes(fb, w.q) };
            for i in 0..count {
                let x = lane(ra, fb, i) as u64;
                set_lane(&mut out, fb, i, u128::from(fp1_lane(k, double, x, fbits)));
            }
            c.put(w.rd, out, full);
            return true;
        }
        _ => return false,
    }
    c.put(w.rd, out, full);
    true
}

// ---- By element -------------------------------------------------------------------------------------------

fn by_element(c: &mut Ctx, w: W, scalar: bool) -> bool {
    let op = (w.word >> 12) & 15;
    let (u, size) = (w.u, w.size);
    let (l, m, h) = ((w.word >> 21) & 1, (w.word >> 20) & 1, (w.word >> 11) & 1);
    let rm4 = ((w.word >> 16) & 15) as usize;
    let (ra, rd) = (c.get(w.rn), c.get(w.rd));
    if matches!((u, op), (false, 1) | (false, 5) | (false, 9) | (true, 9)) {
        // Floating point: size 2 is single, 3 is double.
        if size < 2 || (size == 3 && l != 0) || (size == 3 && !scalar && !w.q) {
            return false;
        }
        let double = size == 3;
        let (index, reg) = if double {
            (h, ((m << 4) as usize) | rm4)
        } else {
            ((h << 1) | l, ((m << 4) as usize) | rm4)
        };
        let bits = if double { 64 } else { 32 };
        let b = lane(c.get(reg), bits, index) as u64;
        let k = match (u, op) {
            (false, 1) => F3::Mla,
            (false, 5) => F3::Mls,
            (false, _) => F3::Mul,
            _ => F3::Mulx,
        };
        let n = if scalar { 1 } else { lanes(bits, w.q) };
        let mut out = 0;
        for i in 0..n {
            let (a, d) = (lane(ra, bits, i) as u64, lane(rd, bits, i) as u64);
            set_lane(&mut out, bits, i, u128::from(fp3_lane(k, double, a, b, d)));
        }
        c.put(w.rd, out, !scalar && w.q);
        return true;
    }
    if size != 1 && size != 2 {
        return false;
    }
    let (index, reg) = if size == 1 {
        ((h << 2) | (l << 1) | m, rm4)
    } else {
        ((h << 1) | l, ((m << 4) as usize) | rm4)
    };
    let bits = 8 << size;
    let b = lane(c.get(reg), bits, index);
    let long = match (u, op) {
        (false, 2) | (true, 2) => Some(LongK::Mla),
        (false, 6) | (true, 6) => Some(LongK::Mls),
        (false, 10) | (true, 10) => Some(LongK::Mul),
        (false, 3) => Some(LongK::Dmlal),
        (false, 7) => Some(LongK::Dmlsl),
        (false, 11) => Some(LongK::Dmull),
        _ => None,
    };
    if let Some(kind) = long {
        if scalar && !matches!(kind, LongK::Dmlal | LongK::Dmlsl | LongK::Dmull) {
            return false;
        }
        let ha = if scalar || !w.q { ra } else { ra >> 64 };
        let wb = bits * 2;
        let mut out = 0;
        for i in 0..if scalar { 1 } else { 64 / bits } {
            let r = long_lane(c, kind, !u, lane(ha, bits, i), b, lane(rd, wb, i), bits);
            set_lane(&mut out, wb, i, r);
        }
        c.put(w.rd, out, !scalar);
        return true;
    }
    let valid = match (u, op) {
        (false, 8) | (true, 0) | (true, 4) => !scalar,
        (false, 12) | (false, 13) => true,
        _ => false,
    };
    if !valid {
        return false;
    }
    let n = if scalar { 1 } else { lanes(bits, w.q) };
    let mut out = 0;
    for i in 0..n {
        let (a, d) = (lane(ra, bits, i), lane(rd, bits, i));
        let r = match (u, op) {
            (false, 8) => a.wrapping_mul(b),
            (true, 0) => d.wrapping_add(a.wrapping_mul(b)),
            (true, 4) => d.wrapping_sub(a.wrapping_mul(b)),
            (_, 12) => sqdmulh(c, a, b, bits, false),
            _ => sqdmulh(c, a, b, bits, true),
        };
        set_lane(&mut out, bits, i, r);
    }
    c.put(w.rd, out, !scalar && w.q);
    true
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;

    /// Registers 1 and 2 loaded from 32-, 16- or 8-bit lanes, then `word` run.
    fn pack(bits: u32, lanes: &[u128]) -> [u64; 2] {
        let mut x = 0;
        for (i, &l) in lanes.iter().enumerate() {
            set_lane(&mut x, bits, i as u32, l);
        }
        [x as u64, (x >> 64) as u64]
    }
    fn unpack(v: [u64; 2], bits: u32, n: u32) -> Vec<u128> {
        let x = u128::from(v[0]) | u128::from(v[1]) << 64;
        (0..n).map(|i| lane(x, bits, i)).collect()
    }
    fn cpu(d: [u64; 2], n: [u64; 2], m: [u64; 2]) -> Cpu {
        let mut cpu = Cpu::default();
        cpu.v[0] = d;
        cpu.v[1] = n;
        cpu.v[2] = m;
        cpu
    }
    fn qc(cpu: &Cpu) -> bool {
        cpu.system.fpsr & (1 << 27) != 0
    }

    #[test]
    fn rounding_register_shifts_use_the_signed_low_byte() {
        // srshl.8h: the shift is the low byte only, negative shifts round.
        let mut c = cpu(
            [0; 2],
            pack(16, &[5, 0xfffb, 3, 0x7fff]),
            pack(16, &[0x01ff, 0xff, 2, 1]),
        );
        run(&mut c, 0x4e625420);
        assert_eq!(unpack(c.v[0], 16, 4), [3, 0xfffe, 12, 0xfffe]);
        assert!(!qc(&c));
    }

    #[test]
    fn absolute_difference_accumulate_and_doubling_multiplies() {
        // saba.4s: d + |n - m|.
        let mut c = cpu(
            pack(32, &[1, 1, 1, 1]),
            pack(32, &[10, 0xffff_fff6, 5, 0]),
            pack(32, &[3, 3, 5, 0xffff_fffc]),
        );
        run(&mut c, 0x4ea27c20);
        assert_eq!(unpack(c.v[0], 32, 4), [8, 14, 1, 5]);
        // sqrdmulh.8h: -1.0 * -1.0 saturates and sets QC; the others round.
        let mut c = cpu(
            [0; 2],
            pack(16, &[0x8000, 0x4000, 0xc000, 1]),
            pack(16, &[0x8000, 0x4000, 0x4000, 1]),
        );
        run(&mut c, 0x6e62b420);
        assert_eq!(unpack(c.v[0], 16, 4), [0x7fff, 0x2000, 0xe000, 0]);
        assert!(qc(&c));
        // sqdmull s0, h1, h2 saturates the same way.
        let mut c = cpu([0; 2], pack(16, &[0x8000]), pack(16, &[0x8000]));
        run(&mut c, 0x5e62d020);
        assert_eq!(c.v[0], [0x7fff_ffff, 0]);
        assert!(qc(&c));
    }

    #[test]
    fn narrowing_and_saturating_moves() {
        // sqshrn.8b #4 saturates to a signed byte.
        let mut c = cpu([0; 2], pack(16, &[0x7fff, 0x8000, 0x0123]), [0; 2]);
        run(&mut c, 0x0f0c9420);
        assert_eq!(unpack(c.v[0], 8, 4), [0x7f, 0x80, 0x12, 0]);
        assert!(qc(&c));
        // uqxtn b0, h1: 0x01ff saturates to 0xff, 0x00ff does not.
        let mut c = cpu([u64::MAX; 2], pack(16, &[0x01ff]), [0; 2]);
        run(&mut c, 0x7e214820);
        assert_eq!(c.v[0], [0xff, 0]);
        assert!(qc(&c));
        let mut c = cpu([0; 2], pack(16, &[0x00ff]), [0; 2]);
        run(&mut c, 0x7e214820);
        assert!(!qc(&c));
        // addhn keeps the high half; raddhn rounds first.
        let (n, m) = (pack(16, &[0x0180]), [0; 2]);
        let mut c = cpu([u64::MAX; 2], n, m);
        run(&mut c, 0x0e224020);
        assert_eq!(c.v[0], [0x01, 0]);
        let mut c = cpu([u64::MAX; 2], n, m);
        run(&mut c, 0x2e224020);
        assert_eq!(c.v[0], [0x02, 0]);
    }

    #[test]
    fn bit_insertion_polynomial_and_widening_multiplies() {
        // sli.4s #16 keeps the low 16 destination bits; sri.4s #3 keeps the high 3.
        let mut c = cpu(pack(32, &[0xaaaa_aaaa]), pack(32, &[0xbbbb]), [0; 2]);
        run(&mut c, 0x6f305420);
        assert_eq!(unpack(c.v[0], 32, 1), [0xbbbb_aaaa]);
        let mut c = cpu(pack(32, &[0xffff_ffff]), [0; 2], [0; 2]);
        run(&mut c, 0x6f3d4420);
        assert_eq!(unpack(c.v[0], 32, 1), [0xe000_0000]);
        // pmull.8h is a carry-less multiply: 3 * 3 = 5.
        let mut c = cpu([0; 2], pack(8, &[3]), pack(8, &[3]));
        run(&mut c, 0x0e22e020);
        assert_eq!(unpack(c.v[0], 16, 1), [5]);
        // mul.4s by element uses lane 1 of v2 for every lane.
        let mut c = cpu([0; 2], pack(32, &[1, 2, 3, 4]), pack(32, &[99, 10]));
        run(&mut c, 0x4fa28020);
        assert_eq!(unpack(c.v[0], 32, 4), [10, 20, 30, 40]);
        // smull.4s by element: signed halfword times h[1] = -32768.
        let mut c = cpu([0; 2], pack(16, &[0xffff, 2, 3, 4]), pack(16, &[0, 0x8000]));
        run(&mut c, 0x0f52a020);
        assert_eq!(
            unpack(c.v[0], 32, 4),
            [0x8000, 0xffff_0000, 0xfffe_8000, 0xfffe_0000]
        );
    }

    #[test]
    fn counting_negation_and_unsigned_estimates() {
        // cls.4s counts the redundant sign bits.
        let mut c = cpu([0; 2], pack(32, &[0xffff_ffff, 0, 1, 0x8000_0000]), [0; 2]);
        run(&mut c, 0x4ea04820);
        assert_eq!(unpack(c.v[0], 32, 4), [31, 31, 30, 0]);
        // sqneg.4s saturates the most negative value.
        let mut c = cpu([0; 2], pack(32, &[0x8000_0000, 5]), [0; 2]);
        run(&mut c, 0x6ea07820);
        assert_eq!(unpack(c.v[0], 32, 2), [0x7fff_ffff, 0xffff_fffb]);
        assert!(qc(&c));
        // urecpe.2s: the top bit must be set, else all ones.
        let mut c = cpu([0; 2], pack(32, &[0x8000_0000, 0x4000_0000]), [0; 2]);
        run(&mut c, 0x0ea1c820);
        assert_eq!(unpack(c.v[0], 32, 2), [0xff80_0000, 0xffff_ffff]);
    }

    #[test]
    fn float_estimates_and_special_values() {
        // frecpe.4s: 1/1, 1/2, 1/0 = inf, 1/-inf = -0.
        let mut c = cpu(
            [0; 2],
            pack(32, &[0x3f80_0000, 0x4000_0000, 0, 0xff80_0000]),
            [0; 2],
        );
        run(&mut c, 0x4ea1d820);
        assert_eq!(
            unpack(c.v[0], 32, 4),
            [0x3f7f_8000, 0x3eff_8000, 0x7f80_0000, 0x8000_0000]
        );
        // frsqrte.4s: 1, 4, 2, and 0 = inf.
        let mut c = cpu(
            [0; 2],
            pack(32, &[0x3f80_0000, 0x4080_0000, 0x4000_0000, 0]),
            [0; 2],
        );
        run(&mut c, 0x6ea1d820);
        assert_eq!(
            unpack(c.v[0], 32, 4),
            [0x3f7f_8000, 0x3eff_8000, 0x3f34_8000, 0x7f80_0000]
        );
        // fabs clears the sign of a NaN as well.
        let mut c = cpu([0; 2], pack(32, &[0xffc0_0000, 0xbf80_0000]), [0; 2]);
        run(&mut c, 0x4ea0f820);
        assert_eq!(unpack(c.v[0], 32, 2), [0x7fc0_0000, 0x3f80_0000]);
    }

    #[test]
    fn float_reductions_run_as_a_tree() {
        // fmaxnmv: (l0 . l1) . (l2 . l3). The two signaling NaNs combine into a
        // quiet NaN first, which then loses to the number.
        let mut c = cpu(
            [0; 2],
            pack(32, &[0xc3, 0, 0x7fa0_0000, 0x7fa0_0000]),
            [0; 2],
        );
        run(&mut c, 0x6e30c820);
        assert_eq!(c.v[0], [0xc3, 0]);
        let mut c = cpu(
            [0; 2],
            pack(32, &[0x3f80_0000, 0x4040_0000, 0x4000_0000, 0xbf80_0000]),
            [0; 2],
        );
        run(&mut c, 0x6e30c820);
        assert_eq!(c.v[0], [0x4040_0000, 0]);
    }

    #[test]
    fn float_conversions_between_widths_and_fixed_point() {
        // fcvtn.4h then fcvtl.4s round-trip exactly representable values.
        let singles = [0x3f80_0000, 0xc000_0000, 0x477f_e000, 0x7f80_0000];
        let mut c = cpu([0; 2], pack(32, &singles), [0; 2]);
        run(&mut c, 0x0e216820);
        assert_eq!(unpack(c.v[0], 16, 4), [0x3c00, 0xc000, 0x7bff, 0x7c00]);
        let narrow = c.v[0];
        let mut c = cpu([0; 2], narrow, [0; 2]);
        run(&mut c, 0x0e217820);
        assert_eq!(unpack(c.v[0], 32, 4), singles);
        // fcvtxn rounds to odd: an inexact result has its low bit set.
        let mut c = cpu(
            [0; 2],
            [0x3ff0_0000_0000_0001, 0x3ff0_0000_0000_0000],
            [0; 2],
        );
        run(&mut c, 0x2e616820);
        assert_eq!(unpack(c.v[0], 32, 2), [0x3f80_0001, 0x3f80_0000]);
        // fcvtzs.4s #3 scales by 8, truncates, saturates and maps NaN to 0.
        let mut c = cpu(
            [0; 2],
            pack(32, &[0x3fc0_0000, 0xbf10_0000, 0x5015_02f9, 0x7fc0_0000]),
            [0; 2],
        );
        run(&mut c, 0x4f3dfc20);
        assert_eq!(unpack(c.v[0], 32, 4), [12, 0xffff_fffc, 0x7fff_ffff, 0]);
        // scvtf.4s #3 divides by 8.
        let mut c = cpu([0; 2], pack(32, &[8, 0xffff_fff8, 1, 0]), [0; 2]);
        run(&mut c, 0x4f3de420);
        assert_eq!(
            unpack(c.v[0], 32, 4),
            [0x3f80_0000, 0xbf80_0000, 0x3e00_0000, 0]
        );
    }

    #[test]
    fn fused_multiply_add_by_element_and_nan_priority() {
        // fmla.4s by element: d + n * m[1] = 1 + 2 * 3.
        let mut c = cpu(
            pack(32, &[0x3f80_0000; 4]),
            pack(32, &[0x4000_0000; 4]),
            pack(32, &[0, 0x4040_0000]),
        );
        run(&mut c, 0x4fa21020);
        assert_eq!(unpack(c.v[0], 32, 4), [0x40e0_0000; 4]);
        // A signaling NaN addend wins over a quiet NaN factor, and is quietened.
        let mut c = cpu(
            pack(32, &[0x7fa0_0000]),
            pack(32, &[0x7fc0_0001]),
            pack(32, &[0, 0x3f80_0000]),
        );
        run(&mut c, 0x4fa21020);
        assert_eq!(unpack(c.v[0], 32, 1), [0x7fe0_0000]);
    }

    #[test]
    fn unallocated_words_are_not_claimed() {
        // pmull with size 1, three-same with size 3 and Q = 0 for add, and fadd.2d with Q = 0.
        for word in [0x0e62_e020, 0x0ee2_8420, 0x0e62_d420] {
            assert!(!is_supported(word), "{word:08x}");
        }
        assert!(is_supported(0x4e22_d420)); // fadd v0.4s, v1.4s, v2.4s
    }
}
