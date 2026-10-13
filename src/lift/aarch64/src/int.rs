//! Integer semantics: flags, conditions, shifts, extends, and the integer
//! instruction families too large to sit inline in the dispatcher.
use crate::Builder;
use volt_ir::{
    function::{BinOp as B, CmpOp as C, Value},
    types::Type,
};
use volt_isa_aarch64::decode::*;

impl Builder {
    pub(crate) fn not(&self, v: Value) -> Value {
        self.imm(self.boolean, B::BitXor, v, 1)
    }
    pub(crate) fn flag(&self, flags: Value, bit: u32) -> Value {
        self.cmp(
            C::Ne,
            self.imm(self.i32, B::BitAnd, flags, 1 << bit),
            self.k(self.i32, 0),
        )
    }
    pub(crate) fn condition(&self, c: Condition) -> Value {
        let f = self.load(self.i32, self.flags_at());
        let n = self.flag(f, 31);
        let z = self.flag(f, 30);
        let carry = self.flag(f, 29);
        let v = self.flag(f, 28);
        let changed = self.bin(self.boolean, B::BitXor, n, v);
        let ordered = self.not(changed);
        let positive = self.not(z);
        match c {
            Condition::Eq => z,
            Condition::Ne => positive,
            Condition::Cs => carry,
            Condition::Cc => self.not(carry),
            Condition::Mi => n,
            Condition::Pl => self.not(n),
            Condition::Vs => v,
            Condition::Vc => self.not(v),
            Condition::Hi => self.bin(self.boolean, B::BitAnd, carry, positive),
            Condition::Ls => self.not(self.bin(self.boolean, B::BitAnd, carry, positive)),
            Condition::Ge => ordered,
            Condition::Lt => changed,
            Condition::Gt => self.bin(self.boolean, B::BitAnd, positive, ordered),
            Condition::Le => self.not(self.bin(self.boolean, B::BitAnd, positive, ordered)),
            Condition::Al | Condition::Nv => self.k(self.boolean, 1),
        }
    }
    pub(crate) fn pack(&self, n: Value, z: Value, c: Value, v: Value) -> Value {
        let mut out = self.k(self.i32, 0);
        for (bit, shift) in [(n, 31), (z, 30), (c, 29), (v, 28)] {
            out = self.bin(
                self.i32,
                B::BitOr,
                out,
                self.imm(self.i32, B::Shl, self.cv(self.i32, bit), shift),
            );
        }
        out
    }
    pub(crate) fn flags(
        &self,
        w: Width,
        op: ArithOp,
        a: Value,
        b: Value,
        r: Value,
        carry: Option<Value>,
    ) -> Value {
        let t = self.ty(w);
        let zero = self.k(t, 0);
        let n = self.cmp(
            C::Ne,
            self.imm(t, B::BitAnd, r, 1u64 << ((w as u32) - 1)),
            zero,
        );
        let z = self.cmp(C::Eq, r, zero);
        let c = carry.unwrap_or_else(|| {
            if op == ArithOp::Sub {
                self.cmp(C::Uge, a, b)
            } else {
                self.cmp(C::Ult, r, a)
            }
        });
        let an = self.cmp(C::Slt, a, zero);
        let bn = self.cmp(C::Slt, b, zero);
        let rn = self.cmp(C::Slt, r, zero);
        let diff = self.bin(self.boolean, B::BitXor, an, bn);
        let flip = self.bin(self.boolean, B::BitXor, an, rn);
        let v = self.bin(
            self.boolean,
            B::BitAnd,
            if op == ArithOp::Sub {
                diff
            } else {
                self.not(diff)
            },
            flip,
        );
        self.pack(n, z, c, v)
    }
    pub(crate) fn logic_flags(&self, w: Width, r: Value) {
        let t = self.ty(w);
        let zero = self.k(t, 0);
        let n = self.cmp(
            C::Ne,
            self.imm(t, B::BitAnd, r, 1u64 << ((w as u32) - 1)),
            zero,
        );
        let z = self.cmp(C::Eq, r, zero);
        let no = self.k(self.boolean, 0);
        self.store(self.flags_at(), self.pack(n, z, no, no));
    }
    pub fn shift(&self, w: Width, v: Value, s: Shift, n: u64) -> Value {
        if n == 0 {
            return v;
        }
        let t = self.ty(w);
        match s {
            Shift::Lsl => self.imm(t, B::Shl, v, n),
            Shift::Lsr => self.imm(t, B::Shr, v, n),
            Shift::Asr => self.imm(t, B::Sar, v, n),
            Shift::Ror => self.bin(
                t,
                B::BitOr,
                self.imm(t, B::Shr, v, n),
                self.imm(t, B::Shl, v, w as u64 - n),
            ),
        }
    }
    pub fn extend(&self, v: Value, e: Extend) -> Value {
        match e {
            Extend::Uxtx | Extend::Sxtx => v,
            Extend::Uxtb => self.imm(self.i64, B::BitAnd, v, 0xff),
            Extend::Uxth => self.imm(self.i64, B::BitAnd, v, 0xffff),
            Extend::Uxtw => self.imm(self.i64, B::BitAnd, v, 0xffffffff),
            Extend::Sxtb | Extend::Sxth | Extend::Sxtw => {
                let n = match e {
                    Extend::Sxtb => 56,
                    Extend::Sxth => 48,
                    _ => 32,
                };
                let raised = self.imm(self.i64, B::Shl, v, n);
                self.shift(Width::X64, raised, Shift::Asr, n)
            }
        }
    }
}
pub(crate) fn arithmetic_op(op: ArithOp) -> B {
    if op == ArithOp::Sub { B::Sub } else { B::Add }
}
pub(crate) fn logic_op(op: LogicOp) -> B {
    match op {
        LogicOp::BitAnd => B::BitAnd,
        LogicOp::BitOr => B::BitOr,
        LogicOp::BitXor => B::BitXor,
    }
}
pub(crate) fn low_mask(n: u8) -> u64 {
    if n >= 64 { u64::MAX } else { (1u64 << n) - 1 }
}

pub(crate) fn variable(b: &Builder, a: Variable) {
    let t = b.ty(a.width);
    let lhs = b.reg(t, a.rn, true);
    let rhs = b.reg(t, a.rm, true);
    let bits = a.width as u64;
    let r = match a.op {
        VariableOp::Lsl | VariableOp::Lsr | VariableOp::Asr | VariableOp::Ror => {
            let distance = b.imm(t, B::BitAnd, rhs, bits - 1);
            match a.op {
                VariableOp::Lsl => b.bin(t, B::Shl, lhs, distance),
                VariableOp::Lsr => b.bin(t, B::Shr, lhs, distance),
                VariableOp::Asr => b.bin(t, B::Sar, lhs, distance),
                VariableOp::Ror => {
                    let back = b.imm(
                        t,
                        B::BitAnd,
                        b.bin(t, B::Sub, b.k(t, bits), distance),
                        bits - 1,
                    );
                    b.bin(
                        t,
                        B::BitOr,
                        b.bin(t, B::Shr, lhs, distance),
                        b.bin(t, B::Shl, lhs, back),
                    )
                }
                _ => unreachable!(),
            }
        }
        VariableOp::Udiv | VariableOp::Sdiv => {
            let zero = b.k(t, 0);
            let one = b.k(t, 1);
            let by_zero = b.cmp(C::Eq, rhs, zero);
            if a.op == VariableOp::Udiv {
                let safe = b.sel(t, by_zero, one, rhs);
                b.sel(t, by_zero, zero, b.bin(t, B::UDiv, lhs, safe))
            } else {
                let minus_one = b.cmp(C::Eq, rhs, b.k(t, u64::MAX));
                let safe = b.sel(t, minus_one, one, b.sel(t, by_zero, one, rhs));
                let quotient = b.bin(t, B::SDiv, lhs, safe);
                let negated = b.bin(t, B::Sub, zero, lhs);
                b.sel(t, by_zero, zero, b.sel(t, minus_one, negated, quotient))
            }
        }
    };
    if a.rd != 31 {
        b.put(a.rd, r)
    }
}
pub(crate) fn bitfield(b: &Builder, a: Bitfield) {
    let t = b.ty(a.width);
    let bits = a.width as u64;
    let len = a.len as u64;
    let lsb = a.lsb as u64;
    let source = b.reg(t, a.rn, true);
    let mask = low_mask(a.len);
    let width_mask = if a.width == Width::X64 {
        u64::MAX
    } else {
        u32::MAX as u64
    };
    let extract = |signed| {
        let raised = b.imm(t, B::Shl, source, bits - lsb - len);
        b.shift(
            a.width,
            raised,
            if signed { Shift::Asr } else { Shift::Lsr },
            bits - len,
        )
    };
    let r = match a.op {
        BitfieldOp::Unsigned => {
            if a.extract {
                extract(false)
            } else {
                b.imm(
                    t,
                    B::Shl,
                    b.imm(t, B::BitAnd, source, mask & width_mask),
                    lsb,
                )
            }
        }
        BitfieldOp::Signed => {
            if a.extract {
                extract(true)
            } else {
                let raised = b.imm(t, B::Shl, source, bits - len);
                let extended = b.shift(a.width, raised, Shift::Asr, bits - len);
                b.imm(t, B::Shl, extended, lsb)
            }
        }
        BitfieldOp::Insert => {
            let value = if a.extract {
                extract(false)
            } else {
                b.imm(
                    t,
                    B::Shl,
                    b.imm(t, B::BitAnd, source, mask & width_mask),
                    lsb,
                )
            };
            let place = if a.extract { 0 } else { lsb };
            let hole = (mask << place) & width_mask;
            let kept = b.imm(t, B::BitAnd, b.reg(t, a.rd, true), !hole & width_mask);
            b.bin(t, B::BitOr, kept, value)
        }
    };
    if a.rd != 31 {
        b.put(a.rd, r)
    }
}
fn repeated(pattern: u64, unit: u32, bits: u32) -> u64 {
    let mut r = 0;
    let mut at = 0;
    while at < bits {
        r |= pattern << at;
        at += unit
    }
    r
}
fn swap(b: &Builder, t: Type, v: Value, mask: u64, n: u64) -> Value {
    let low = b.imm(t, B::BitAnd, v, mask);
    let high = b.imm(t, B::BitAnd, b.imm(t, B::Shr, v, n), mask);
    b.bin(t, B::BitOr, b.imm(t, B::Shl, low, n), high)
}
pub(crate) fn unary(b: &Builder, a: Unary) {
    let t = b.ty(a.width);
    let bits = a.width as u32;
    let mut v = b.reg(t, a.rn, true);
    match a.op {
        UnaryOp::Rev16 => v = swap(b, t, v, repeated(0xff, 16, bits), 8),
        UnaryOp::Rev32 => {
            v = swap(b, t, v, repeated(0xff, 16, bits), 8);
            v = swap(b, t, v, repeated(0xffff, 32, bits), 16);
        }
        UnaryOp::Rev | UnaryOp::Rbit => {
            if a.op == UnaryOp::Rbit {
                for (mask, unit, n) in [(1, 2, 1), (3, 4, 2), (15, 8, 4)] {
                    v = swap(b, t, v, repeated(mask, unit, bits), n)
                }
            }
            v = swap(b, t, v, repeated(0xff, 16, bits), 8);
            v = swap(b, t, v, repeated(0xffff, 32, bits), 16);
            if bits == 64 {
                v = b.bin(
                    t,
                    B::BitOr,
                    b.imm(t, B::Shl, v, 32),
                    b.imm(t, B::Shr, v, 32),
                )
            }
        }
        UnaryOp::Clz => {
            let original = v;
            let mut count = b.k(t, 0);
            let mut step = bits / 2;
            while step >= 1 {
                let top = b.imm(t, B::Shr, v, (bits - step) as u64);
                let empty = b.cmp(C::Eq, top, b.k(t, 0));
                v = b.sel(t, empty, b.imm(t, B::Shl, v, step as u64), v);
                count = b.sel(t, empty, b.imm(t, B::Add, count, step as u64), count);
                step /= 2;
            }
            v = b.sel(
                t,
                b.cmp(C::Eq, original, b.k(t, 0)),
                b.k(t, bits as u64),
                count,
            );
        }
    }
    if a.rd != 31 {
        b.put(a.rd, v)
    }
}
pub(crate) fn high(b: &Builder, a: MultiplyHigh) {
    let t = b.i64;
    let x = b.reg(t, a.rn, true);
    let y = b.reg(t, a.rm, true);
    let xl = b.imm(t, B::BitAnd, x, 0xffffffff);
    let yl = b.imm(t, B::BitAnd, y, 0xffffffff);
    let xh = b.imm(t, B::Shr, x, 32);
    let yh = b.imm(t, B::Shr, y, 32);
    let ll = b.bin(t, B::Mul, xl, yl);
    let lh = b.bin(t, B::Mul, xl, yh);
    let hl = b.bin(t, B::Mul, xh, yl);
    let hh = b.bin(t, B::Mul, xh, yh);
    let middle = b.bin(
        t,
        B::Add,
        b.bin(
            t,
            B::Add,
            b.imm(t, B::Shr, ll, 32),
            b.imm(t, B::BitAnd, lh, 0xffffffff),
        ),
        b.imm(t, B::BitAnd, hl, 0xffffffff),
    );
    let mut high = b.bin(
        t,
        B::Add,
        b.bin(t, B::Add, hh, b.imm(t, B::Shr, lh, 32)),
        b.bin(
            t,
            B::Add,
            b.imm(t, B::Shr, hl, 32),
            b.imm(t, B::Shr, middle, 32),
        ),
    );
    if a.signed {
        let zero = b.k(t, 0);
        let xn = b.bin(t, B::Sub, zero, b.imm(t, B::Shr, x, 63));
        let yn = b.bin(t, B::Sub, zero, b.imm(t, B::Shr, y, 63));
        high = b.bin(t, B::Sub, high, b.bin(t, B::BitAnd, y, xn));
        high = b.bin(t, B::Sub, high, b.bin(t, B::BitAnd, x, yn));
    }
    if a.rd != 31 {
        b.put(a.rd, high)
    }
}
