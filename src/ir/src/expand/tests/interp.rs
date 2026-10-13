//! A small reference interpreter for the IR the expansions produce, used to check their
//! numerics against the host implementations. Values are bit patterns, held zero-extended at
//! their type's width (the backend convention); `Sext`, `Sar`, `SDiv`, ... re-read the bits as
//! signed on demand.

use alloc::vec::Vec;
use std::collections::HashMap;

use crate::function::{
    BinOp, Block, CmpOp, ConvertKind, Function, Opcode, Terminator, UnaryOp, Value,
};
use crate::types::{FloatKind, Type, TypeKind};

fn bits_of(func: &Function, ty: Type) -> u32 {
    match func.types.type_kind(ty) {
        TypeKind::Int(d) => u32::from(d.bits),
        TypeKind::Bool => 1,
        TypeKind::Ptr(_) => 64,
        TypeKind::Float(FloatKind::F16) => 16,
        TypeKind::Float(FloatKind::F32) => 32,
        TypeKind::Float(FloatKind::F64) => 64,
        other => panic!("interpreter: unsupported type {other:?}"),
    }
}

fn mask(bits: u32) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

fn sext(x: u64, bits: u32) -> i64 {
    if bits >= 64 {
        x as i64
    } else {
        ((x << (64 - bits)) as i64) >> (64 - bits)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Fp {
    F16,
    F32,
    F64,
}

fn fp_kind(func: &Function, ty: Type) -> Option<Fp> {
    match func.types.type_kind(ty) {
        TypeKind::Float(FloatKind::F16) => Some(Fp::F16),
        TypeKind::Float(FloatKind::F32) => Some(Fp::F32),
        TypeKind::Float(FloatKind::F64) => Some(Fp::F64),
        TypeKind::Float(FloatKind::F128) => panic!("interpreter: f128"),
        _ => None,
    }
}

fn f16_to_f64(bits: u64) -> f64 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((bits >> 10) & 0x1f) as i32;
    let frac = (bits & 0x3ff) as f64;
    match exp {
        0 => sign * frac * 2f64.powi(-24),
        31 if frac == 0.0 => sign * f64::INFINITY,
        31 => f64::NAN,
        _ => sign * (1.0 + frac / 1024.0) * 2f64.powi(exp - 15),
    }
}

/// The value of an IEEE half-precision bit pattern, exactly (every half is an f32).
pub fn f16_to_f32(bits: u16) -> f32 {
    f16_to_f64(u64::from(bits)) as f32
}

fn to_f64(kind: Fp, bits: u64) -> f64 {
    match kind {
        Fp::F16 => f16_to_f64(bits),
        Fp::F32 => f64::from(f32::from_bits(bits as u32)),
        Fp::F64 => f64::from_bits(bits),
    }
}

fn from_f64(kind: Fp, v: f64) -> u64 {
    match kind {
        Fp::F16 => panic!("interpreter: narrowing to f16"),
        Fp::F32 => u64::from((v as f32).to_bits()),
        Fp::F64 => v.to_bits(),
    }
}

fn int_arith(op: BinOp, l: u64, r: u64, bits: u32) -> u64 {
    let m = mask(bits);
    let (sl, sr) = (sext(l, bits), sext(r, bits));
    let raw = match op {
        BinOp::Add => l.wrapping_add(r),
        BinOp::Sub => l.wrapping_sub(r),
        BinOp::Mul => l.wrapping_mul(r),
        BinOp::BitAnd => l & r,
        BinOp::BitOr => l | r,
        BinOp::BitXor => l ^ r,
        BinOp::Shl => l.checked_shl(r as u32).unwrap_or(0),
        BinOp::Shr => l.checked_shr(r as u32).unwrap_or(0),
        BinOp::Sar => (sl >> r.min(63)) as u64,
        BinOp::UDiv => l / r,
        BinOp::URem => l % r,
        BinOp::SDiv => sl.wrapping_div(sr) as u64,
        BinOp::SRem => sl.wrapping_rem(sr) as u64,
        BinOp::UMulh => ((u128::from(l) * u128::from(r)) >> bits) as u64,
        BinOp::SMulh => ((i128::from(sl) * i128::from(sr)) >> bits) as u64,
        BinOp::Div | BinOp::Rem => panic!("interpreter: {op:?} on an integer"),
    };
    raw & m
}

fn float_arith(op: BinOp, kind: Fp, l: u64, r: u64) -> u64 {
    let (l, r) = (to_f64(kind, l), to_f64(kind, r));
    let v = match op {
        BinOp::Add => l + r,
        BinOp::Sub => l - r,
        BinOp::Mul => l * r,
        BinOp::Div => l / r,
        BinOp::Rem => l % r,
        other => panic!("interpreter: {other:?} on a float"),
    };
    from_f64(kind, v)
}

fn compare(op: CmpOp, kind: Option<Fp>, l: u64, r: u64, bits: u32) -> bool {
    if let Some(kind) = kind {
        let (l, r) = (to_f64(kind, l), to_f64(kind, r));
        return match op {
            CmpOp::Eq => l == r,
            CmpOp::Ne => l != r,
            CmpOp::Lt => l < r,
            CmpOp::Le => l <= r,
            CmpOp::Gt => l > r,
            CmpOp::Ge => l >= r,
            other => panic!("interpreter: {other:?} on a float"),
        };
    }
    let (sl, sr) = (sext(l, bits), sext(r, bits));
    match op {
        CmpOp::Eq => l == r,
        CmpOp::Ne => l != r,
        CmpOp::Slt => sl < sr,
        CmpOp::Sle => sl <= sr,
        CmpOp::Sgt => sl > sr,
        CmpOp::Sge => sl >= sr,
        CmpOp::Ult => l < r,
        CmpOp::Ule => l <= r,
        CmpOp::Ugt => l > r,
        CmpOp::Uge => l >= r,
        other => panic!("interpreter: {other:?} on an integer"),
    }
}

/// Runs functions over a flat little-endian memory.
pub struct Interp {
    pub mem: Vec<u8>,
}

impl Interp {
    pub fn new(mem: Vec<u8>) -> Self {
        Self { mem }
    }

    fn read(&self, addr: u64, bytes: u32) -> u64 {
        (0..bytes as usize).fold(0u64, |acc, i| {
            acc | u64::from(self.mem[addr as usize + i]) << (8 * i)
        })
    }

    fn write(&mut self, addr: u64, bytes: u32, value: u64) {
        for i in 0..bytes as usize {
            self.mem[addr as usize + i] = (value >> (8 * i)) as u8;
        }
    }

    /// Executes `func` from its entry block with `args` bound to the entry parameters and
    /// returns the returned values.
    pub fn run(&mut self, func: &Function, args: &[u64]) -> Vec<u64> {
        let mut vals: HashMap<Value, u64> = HashMap::new();
        let mut block = Block::from_u32(0);
        let entry_params = func.block_params(block).to_vec();
        assert_eq!(entry_params.len(), args.len());
        for (p, a) in entry_params.iter().zip(args) {
            vals.insert(*p, *a & mask(bits_of(func, func.value_type(*p))));
        }
        let mut steps = 0u64;
        'blocks: loop {
            for &inst in func.block_insts(block) {
                steps += 1;
                assert!(steps < 5_000_000, "interpreter: runaway loop");
                let result = func.inst_result(inst);
                let result_ty = result.map(|r| func.value_type(r));
                let out: Option<u64> = match func.opcode(inst) {
                    Opcode::Iconst(c) => Some(c as u64 & mask(bits_of(func, result_ty.unwrap()))),
                    Opcode::Fconst(f) => Some(match fp_kind(func, result_ty.unwrap()) {
                        Some(Fp::F32) => u64::from((f as f32).to_bits()),
                        Some(Fp::F64) => f.to_bits(),
                        other => panic!("interpreter: fconst {other:?}"),
                    }),
                    Opcode::Arith(a) => {
                        let ty = func.value_type(a.lhs);
                        let (l, r) = (vals[&a.lhs], vals[&a.rhs]);
                        Some(match fp_kind(func, ty) {
                            Some(kind) => float_arith(a.op, kind, l, r),
                            None => int_arith(a.op, l, r, bits_of(func, ty)),
                        })
                    }
                    Opcode::ArithImm(a) => {
                        let ty = func.value_type(a.lhs);
                        let bits = bits_of(func, ty);
                        Some(int_arith(
                            a.op,
                            vals[&a.lhs],
                            a.imm as u64 & mask(bits),
                            bits,
                        ))
                    }
                    Opcode::Icmp(c) => {
                        let ty = func.value_type(c.lhs);
                        Some(u64::from(compare(
                            c.op,
                            fp_kind(func, ty),
                            vals[&c.lhs],
                            vals[&c.rhs],
                            bits_of(func, ty),
                        )))
                    }
                    Opcode::Select(s) => Some(if vals[&s.cond] != 0 {
                        vals[&s.then]
                    } else {
                        vals[&s.else_]
                    }),
                    Opcode::Convert(cv) => {
                        let (src_ty, dst_ty) = (func.value_type(cv.value), result_ty.unwrap());
                        let (sb, db) = (bits_of(func, src_ty), bits_of(func, dst_ty));
                        let v = vals[&cv.value];
                        Some(match cv.kind {
                            ConvertKind::Trunc => {
                                assert!(db < sb);
                                v & mask(db)
                            }
                            ConvertKind::Zext => {
                                assert!(db > sb);
                                v
                            }
                            ConvertKind::Sext => {
                                assert!(db > sb);
                                sext(v, sb) as u64 & mask(db)
                            }
                            ConvertKind::SiToFp => {
                                from_f64(fp_kind(func, dst_ty).unwrap(), sext(v, sb) as f64)
                            }
                            ConvertKind::UiToFp => {
                                from_f64(fp_kind(func, dst_ty).unwrap(), v as f64)
                            }
                            ConvertKind::FpToSi => {
                                let f = to_f64(fp_kind(func, src_ty).unwrap(), v);
                                let (lo, hi) = if db >= 64 {
                                    (i64::MIN, i64::MAX)
                                } else {
                                    (-(1i64 << (db - 1)), (1i64 << (db - 1)) - 1)
                                };
                                (f as i64).clamp(lo, hi) as u64 & mask(db)
                            }
                            ConvertKind::FpToUi => {
                                let f = to_f64(fp_kind(func, src_ty).unwrap(), v);
                                (f as u64).min(mask(db))
                            }
                            ConvertKind::FpResize => {
                                let f = to_f64(fp_kind(func, src_ty).unwrap(), v);
                                from_f64(fp_kind(func, dst_ty).unwrap(), f)
                            }
                        })
                    }
                    Opcode::Unary(u) => {
                        assert_eq!(u.op, UnaryOp::Reinterpret);
                        let bits = bits_of(func, result_ty.unwrap());
                        assert_eq!(bits, bits_of(func, func.value_type(u.value)));
                        Some(vals[&u.value] & mask(bits))
                    }
                    Opcode::Load(l) => {
                        let bytes = bits_of(func, result_ty.unwrap()) / 8;
                        Some(self.read(vals[&l.ptr], bytes))
                    }
                    Opcode::Store(st) => {
                        let bytes = bits_of(func, func.value_type(st.value)) / 8;
                        self.write(vals[&st.ptr], bytes, vals[&st.value]);
                        None
                    }
                    Opcode::If(cf) => {
                        let jump = if vals[&cf.cond] != 0 {
                            cf.then
                        } else {
                            cf.else_
                        };
                        let moved: Vec<u64> =
                            func.block_args(jump).iter().map(|a| vals[a]).collect();
                        for (p, v) in func.block_params(jump.target).iter().zip(moved) {
                            vals.insert(*p, v);
                        }
                        block = jump.target;
                        continue 'blocks;
                    }
                    other => panic!("interpreter: unsupported opcode {other:?}"),
                };
                if let (Some(r), Some(v)) = (result, out) {
                    vals.insert(r, v);
                }
            }
            match func.terminator(block).expect("terminated block") {
                Terminator::Ret(r) => return r.slice().iter().map(|v| vals[v]).collect(),
                Terminator::Jump(j) => {
                    let moved: Vec<u64> = func.block_args(j).iter().map(|a| vals[a]).collect();
                    for (p, v) in func.block_params(j.target).iter().zip(moved) {
                        vals.insert(*p, v);
                    }
                    block = j.target;
                }
            }
        }
    }
}
