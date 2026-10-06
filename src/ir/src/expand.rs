extern crate alloc;

use alloc::vec::Vec;

use super::function::AttrTarget;
use super::function::{
    BinOp, Block, CmpOp, Function, Inst, LowFloatConvert, MatMul, MatMulType, NvFp4Convert, Opcode,
    Value,
};
use super::low_float::Format as LowFloatFormat;
use super::nvfp4::ScaleApplication;
use super::types::{FloatKind, IntDesc, Type, TypeKind};

/// Reason a `matmul` cannot lower on the current target.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Unsupported {
    Dtype,
    Tile,
    Quant,
}

pub fn matmul_unsupported(mm: &MatMul) -> Option<Unsupported> {
    match mm.dtype {
        MatMulType::Fp32 | MatMulType::Fp16 | MatMulType::Int8 | MatMulType::Uint8 => {}
    }
    if mm.m == 0 || mm.n == 0 || mm.k == 0 {
        return Some(Unsupported::Tile);
    }
    if mm.quant.is_some() && !matches!(mm.dtype, MatMulType::Int8 | MatMulType::Uint8) {
        return Some(Unsupported::Quant);
    }
    None
}

/// Replace `reduce`/`splat` with operations every backend lowers.
/// Returns whether anything changed.
pub fn expand_vector_lanes(func: &mut Function) -> bool {
    expand_vector_lanes_except(func, &|_, _| false)
}

/// Same rewrite, but leaves an instruction untouched when `keep` reports the
/// caller's own isel already lowers it.
pub fn expand_vector_lanes_except(
    func: &mut Function,
    keep: &dyn Fn(&Function, Inst) -> bool,
) -> bool {
    let mut changed = false;
    for bi in 0..func.block_count() {
        let block = Block::from_u32(bi as u32);
        let mut has = false;
        for inst in func.block_insts(block).to_vec() {
            match func.opcode(inst) {
                Opcode::Reduce(_) | Opcode::Splat(_) if !keep(func, inst) => {
                    has = true;
                    break;
                }
                _ => {}
            }
        }
        if !has {
            continue;
        }
        changed = true;
        let original = func.block_insts(block).to_vec();
        let mut out: Vec<Inst> = Vec::new();
        for inst in original {
            let is_target = matches!(func.opcode(inst), Opcode::Reduce(_) | Opcode::Splat(_))
                && !keep(func, inst);
            if !is_target {
                out.push(inst);
                continue;
            }
            match func.opcode(inst) {
                Opcode::Reduce(red) => {
                    let Some(result) = func.inst_result(inst) else {
                        continue;
                    };
                    let (len, elem) = match func.types.type_kind(func.value_type(red.vector)) {
                        TypeKind::Vector(v) => (v.len, v.elem),
                        _ => continue,
                    };
                    let mut acc: Option<Value> = None;
                    for lane in 0..len {
                        let e = func.create_inst(
                            elem,
                            Opcode::Extract(super::function::Extract {
                                aggregate: red.vector,
                                index: lane,
                            }),
                        );
                        out.push(func.defining_inst(e).unwrap());
                        acc = Some(match acc {
                            None => e,
                            Some(a) => {
                                let r = func.create_inst(
                                    elem,
                                    Opcode::Arith(super::function::Arith {
                                        op: red.op,
                                        lhs: a,
                                        rhs: e,
                                    }),
                                );
                                out.push(func.defining_inst(r).unwrap());
                                r
                            }
                        });
                    }
                    if let Some(final_v) = acc {
                        func.replace_all_uses(result, final_v);
                    }
                }
                Opcode::Splat(sp) => {
                    let Some(result) = func.inst_result(inst) else {
                        continue;
                    };
                    let res_ty = func.value_type(result);
                    let len = match func.types.type_kind(res_ty) {
                        TypeKind::Vector(v) => v.len,
                        _ => continue,
                    };
                    let fields: Vec<Value> = (0..len).map(|_| sp.scalar).collect();
                    let list = func.intern_values(&fields);
                    let v = func.create_inst(
                        res_ty,
                        Opcode::StructNew(super::function::StructNew { fields: list }),
                    );
                    out.push(func.defining_inst(v).unwrap());
                    func.replace_all_uses(result, v);
                }
                _ => {}
            }
        }
        func.set_block_insts(block, &out);
    }
    changed
}

// --- NumericBuilder ----------------------------------------------------------

struct NumericBuilder {
    u8_t: Type,
    u32_t: Type,
    f32_t: Type,
    bool_t: Type,
    out: Vec<Inst>,
}

/// The bounded operations a scalar expansion may emit.
impl NumericBuilder {
    fn common(_func: &mut Function, ct: &CommonTypes) -> Self {
        Self {
            u8_t: ct.u8_t,
            u32_t: ct.u32_t,
            f32_t: ct.f32_t,
            bool_t: ct.bool_t,
            out: Vec::new(),
        }
    }

    fn emit(&mut self, func: &mut Function, ty: Type, opcode: Opcode) -> Value {
        let value = func.create_inst(ty, opcode);
        self.out.push(func.defining_inst(value).unwrap());
        value
    }

    fn constant(&mut self, func: &mut Function, value: u32) -> Value {
        self.emit(func, self.u32_t, Opcode::Iconst(value as i64))
    }

    fn binary(&mut self, func: &mut Function, op: BinOp, lhs: Value, rhs: Value) -> Value {
        self.emit(
            func,
            self.u32_t,
            Opcode::Arith(super::function::Arith { op, lhs, rhs }),
        )
    }

    fn float_binary(&mut self, func: &mut Function, op: BinOp, lhs: Value, rhs: Value) -> Value {
        debug_assert!(op == BinOp::Mul || op == BinOp::Div);
        debug_assert_eq!(func.value_type(lhs), self.f32_t);
        debug_assert_eq!(func.value_type(rhs), self.f32_t);
        self.emit(
            func,
            self.f32_t,
            Opcode::Arith(super::function::Arith { op, lhs, rhs }),
        )
    }

    fn compare(&mut self, func: &mut Function, op: CmpOp, lhs: Value, rhs: Value) -> Value {
        self.emit(
            func,
            self.bool_t,
            Opcode::Icmp(super::function::Compare { op, lhs, rhs }),
        )
    }

    fn select(&mut self, func: &mut Function, cond: Value, then_v: Value, else_v: Value) -> Value {
        self.emit(
            func,
            self.u32_t,
            Opcode::Select(super::function::Select {
                cond,
                then: then_v,
                else_: else_v,
            }),
        )
    }

    fn convert(&mut self, func: &mut Function, ty: Type, value: Value) -> Value {
        self.emit(
            func,
            ty,
            Opcode::Convert(super::function::Convert { value }),
        )
    }

    fn reinterpret(&mut self, func: &mut Function, ty: Type, value: Value) -> Value {
        self.emit(
            func,
            ty,
            Opcode::Unary(super::function::Unary {
                op: super::function::UnaryOp::Reinterpret,
                value,
            }),
        )
    }

    fn masked(&mut self, func: &mut Function, value: Value, mask: u32) -> Value {
        let mask_v = self.constant(func, mask);
        self.binary(func, BinOp::BitAnd, value, mask_v)
    }

    fn shifted(&mut self, func: &mut Function, op: BinOp, value: Value, amount: u32) -> Value {
        let amount_v = self.constant(func, amount);
        self.binary(func, op, value, amount_v)
    }

    fn equal_constant(&mut self, func: &mut Function, value: Value, expected: u32) -> Value {
        let expected_v = self.constant(func, expected);
        self.compare(func, CmpOp::Eq, value, expected_v)
    }

    /// Division of `value` by 2^shift with round-to-nearest, odd-tie.
    fn round_right(&mut self, func: &mut Function, value: Value, shift: Value) -> Value {
        let one = self.constant(func, 1);
        let retained = self.binary(func, BinOp::Shr, value, shift);
        let shifted_one = self.binary(func, BinOp::Shl, one, shift);
        let mask = self.binary(func, BinOp::Sub, shifted_one, one);
        let discarded = self.binary(func, BinOp::BitAnd, value, mask);
        let shift_minus_one = self.binary(func, BinOp::Sub, shift, one);
        let halfway = self.binary(func, BinOp::Shl, one, shift_minus_one);
        let greater = self.compare(func, CmpOp::Gt, discarded, halfway);
        let equal = self.compare(func, CmpOp::Eq, discarded, halfway);
        let odd_bits = self.binary(func, BinOp::BitAnd, retained, one);
        let zero = self.constant(func, 0);
        let odd = self.compare(func, CmpOp::Ne, odd_bits, zero);
        let odd_as_int = self.select(func, odd, one, zero);
        let tie_and_odd = self.select(func, equal, odd_as_int, zero);
        let increment = self.select(func, greater, one, tie_and_odd);
        self.binary(func, BinOp::Add, retained, increment)
    }
}
// --- Shared per-block rebuild helpers ---------------------------------------
// --- Low float ---------------------------------------------------------------

/// Types used by the scalar builder passes, interned once per pass run.
struct CommonTypes {
    u8_t: Type,
    u16_t: Type,
    u32_t: Type,
    f32_t: Type,
    bool_t: Type,
}

fn intern_common(func: &mut Function) -> CommonTypes {
    let u8_t = func.types.intern(int_ty_kind(8, false));
    let u16_t = func.types.intern(int_ty_kind(16, false));
    let u32_t = func.types.intern(int_ty_kind(32, false));
    let f32_t = func.types.intern(TypeKind::Float(FloatKind::F32));
    let bool_t = func.types.intern(TypeKind::Bool);
    CommonTypes {
        u8_t,
        u16_t,
        u32_t,
        f32_t,
        bool_t,
    }
}

fn int_ty_kind(bits: u16, signed: bool) -> TypeKind {
    TypeKind::Int(IntDesc { signed, bits })
}
/// Replace all low-float storage conversions with target-independent integer
/// operations. Returns whether anything changed.
pub fn expand_low_float(func: &mut Function) -> bool {
    let mut has_low_float = false;
    for bi in 0..func.block_count() {
        let block = Block::from_u32(bi as u32);
        for inst in func.block_insts(block).to_vec() {
            if matches!(
                func.opcode(inst),
                Opcode::DecodeLowFloat(_) | Opcode::EncodeLowFloat(_)
            ) {
                has_low_float = true;
                break;
            }
        }
        if has_low_float {
            break;
        }
    }
    if !has_low_float {
        return false;
    }

    let ct = intern_common(func);
    let mut changed = false;
    for bi in 0..func.block_count() {
        let block = Block::from_u32(bi as u32);
        let contains_low_float = func.block_insts(block).iter().any(|i| {
            matches!(
                func.opcode(*i),
                Opcode::DecodeLowFloat(_) | Opcode::EncodeLowFloat(_)
            )
        });
        if !contains_low_float {
            continue;
        }
        changed = true;

        let original = func.block_insts(block).to_vec();
        let mut builder = NumericBuilder {
            out: Vec::new(),
            ..NumericBuilder::common(func, &ct)
        };
        let mut out: Vec<Inst> = core::mem::take(&mut builder.out);
        for inst in original {
            match func.opcode(inst).clone() {
                Opcode::DecodeLowFloat(conversion) => {
                    let Some(old_result) = func.inst_result(inst) else {
                        continue;
                    };
                    let expanded = decode_low_float(func, &mut builder, &conversion);
                    func.replace_all_uses(old_result, expanded);
                    func.retarget_attrs(AttrTarget::Value(old_result), AttrTarget::Value(expanded));
                    let defining = func.defining_inst(expanded).unwrap();
                    func.retarget_attrs(AttrTarget::Inst(inst), AttrTarget::Inst(defining));
                    out.append(&mut builder.out);
                }
                Opcode::EncodeLowFloat(conversion) => {
                    let Some(old_result) = func.inst_result(inst) else {
                        continue;
                    };
                    let expanded =
                        encode_low_float(func, &mut builder, &conversion, inst, ct.u16_t, ct.u8_t);
                    func.replace_all_uses(old_result, expanded);
                    func.retarget_attrs(AttrTarget::Value(old_result), AttrTarget::Value(expanded));
                    out.append(&mut builder.out);
                }
                _ => out.push(inst),
            }
        }
        func.set_block_insts(block, &out);
    }
    changed
}

fn decode_low_float(func: &mut Function, b: &mut NumericBuilder, conv: &LowFloatConvert) -> Value {
    let payload = b.convert(func, b.u32_t, conv.value);
    let bits = match conv.format {
        LowFloatFormat::Bf16 => b.shifted(func, BinOp::Shl, payload, 16),
        LowFloatFormat::F8E4M3 => decode_e4m3(func, b, payload),
        LowFloatFormat::F8E5M2 => decode_e5m2(func, b, payload),
    };
    b.reinterpret(func, b.f32_t, bits)
}

fn decode_e4m3(func: &mut Function, b: &mut NumericBuilder, payload: Value) -> Value {
    let hoist0 = b.masked(func, payload, 0x80);
    let sign = b.shifted(func, BinOp::Shl, hoist0, 24);
    let hoist1 = b.shifted(func, BinOp::Shr, payload, 3);
    let exponent = b.masked(func, hoist1, 0x0f);
    let fraction = b.masked(func, payload, 0x07);
    let hoist2 = b.constant(func, 120);
    let exponent_plus = b.binary(func, BinOp::Add, exponent, hoist2);
    let normal_exponent = b.shifted(func, BinOp::Shl, exponent_plus, 23);
    let normal_fraction = b.shifted(func, BinOp::Shl, fraction, 20);
    let normal_exp_fr = b.binary(func, BinOp::BitOr, normal_exponent, normal_fraction);
    let normal = b.binary(func, BinOp::BitOr, sign, normal_exp_fr);

    let mut subnormal = b.constant(func, 0);
    for (raw_fraction, bits) in [
        0x3b00_0000u32,
        0x3b80_0000,
        0x3bc0_0000,
        0x3c00_0000,
        0x3c20_0000,
        0x3c40_0000,
        0x3c60_0000,
    ]
    .into_iter()
    .enumerate()
    {
        let idx = raw_fraction as u32 + 1;
        let cond = b.equal_constant(func, fraction, idx);
        let hoist3 = b.constant(func, bits);
        subnormal = b.select(func, cond, hoist3, subnormal);
    }
    let subnormal = b.binary(func, BinOp::BitOr, sign, subnormal);
    let zero_cond = b.equal_constant(func, exponent, 0);
    let finite = b.select(func, zero_cond, subnormal, normal);
    let hoist4 = b.constant(func, 0x7fc0_0000);
    let nan = b.binary(func, BinOp::BitOr, sign, hoist4);
    let exponent_is_max = b.equal_constant(func, exponent, 0x0f);
    let fraction_is_nan = b.equal_constant(func, fraction, 0x07);
    let one = b.constant(func, 1);
    let zero = b.constant(func, 0);
    let fraction_is_nan_int = b.select(func, fraction_is_nan, one, zero);
    let is_nan_int = b.select(func, exponent_is_max, fraction_is_nan_int, zero);
    let is_nan = b.compare(func, CmpOp::Ne, is_nan_int, zero);
    b.select(func, is_nan, nan, finite)
}

fn decode_e5m2(func: &mut Function, b: &mut NumericBuilder, payload: Value) -> Value {
    let hoist5 = b.masked(func, payload, 0x80);
    let sign = b.shifted(func, BinOp::Shl, hoist5, 24);
    let hoist6 = b.shifted(func, BinOp::Shr, payload, 2);
    let exponent = b.masked(func, hoist6, 0x1f);
    let fraction = b.masked(func, payload, 0x03);
    let hoist7 = b.constant(func, 112);
    let exponent_plus = b.binary(func, BinOp::Add, exponent, hoist7);
    let normal_exponent = b.shifted(func, BinOp::Shl, exponent_plus, 23);
    let normal_fraction = b.shifted(func, BinOp::Shl, fraction, 21);
    let normal_exp_fr = b.binary(func, BinOp::BitOr, normal_exponent, normal_fraction);
    let normal = b.binary(func, BinOp::BitOr, sign, normal_exp_fr);

    let mut subnormal = b.constant(func, 0);
    for (raw_fraction, bits) in [0x3780_0000u32, 0x3800_0000, 0x3840_0000]
        .into_iter()
        .enumerate()
    {
        let idx = raw_fraction as u32 + 1;
        let cond = b.equal_constant(func, fraction, idx);
        let hoist8 = b.constant(func, bits);
        subnormal = b.select(func, cond, hoist8, subnormal);
    }
    let subnormal = b.binary(func, BinOp::BitOr, sign, subnormal);
    let zero_cond = b.equal_constant(func, exponent, 0);
    let finite = b.select(func, zero_cond, subnormal, normal);
    let hoist9 = b.constant(func, 2);
    let fraction_or_two = b.binary(func, BinOp::BitOr, fraction, hoist9);
    let quiet_fraction = b.shifted(func, BinOp::Shl, fraction_or_two, 21);
    let hoist10 = b.constant(func, 0x7f80_0000);
    let nan_exp = b.binary(func, BinOp::BitOr, hoist10, quiet_fraction);
    let nan = b.binary(func, BinOp::BitOr, sign, nan_exp);
    let hoist11 = b.constant(func, 0x7f80_0000);
    let infinity = b.binary(func, BinOp::BitOr, sign, hoist11);
    let fraction_zero = b.equal_constant(func, fraction, 0);
    let special = b.select(func, fraction_zero, infinity, nan);
    let exponent_special = b.equal_constant(func, exponent, 0x1f);
    b.select(func, exponent_special, special, finite)
}

fn encode_low_float(
    func: &mut Function,
    b: &mut NumericBuilder,
    conv: &LowFloatConvert,
    old_inst: Inst,
    u16_t: Type,
    u8_t: Type,
) -> Value {
    let bits = b.reinterpret(func, b.u32_t, conv.value);
    let defining = func.defining_inst(bits).unwrap();
    func.retarget_attrs(AttrTarget::Inst(old_inst), AttrTarget::Inst(defining));
    let payload = match conv.format {
        LowFloatFormat::Bf16 => encode_bf16(func, b, bits),
        LowFloatFormat::F8E4M3 => encode_fp8(func, b, bits, true),
        LowFloatFormat::F8E5M2 => encode_fp8(func, b, bits, false),
    };
    let narrow_ty = if conv.format == LowFloatFormat::Bf16 {
        u16_t
    } else {
        u8_t
    };
    b.convert(func, narrow_ty, payload)
}

fn encode_bf16(func: &mut Function, b: &mut NumericBuilder, bits: Value) -> Value {
    let exponent = b.masked(func, bits, 0x7f80_0000);
    let fraction = b.masked(func, bits, 0x007f_ffff);
    let retained = b.shifted(func, BinOp::Shr, bits, 16);
    let retained_lsb = b.masked(func, retained, 1);
    let hoist12 = b.constant(func, 0x7fff);
    let rounding_addend = b.binary(func, BinOp::Add, hoist12, retained_lsb);
    let sum = b.binary(func, BinOp::Add, bits, rounding_addend);
    let rounded = b.shifted(func, BinOp::Shr, sum, 16);
    let hoist13 = b.constant(func, 0x40);
    let nan = b.binary(func, BinOp::BitOr, retained, hoist13);
    let hoist14 = b.constant(func, 0);
    let fraction_ne = b.compare(func, CmpOp::Ne, fraction, hoist14);
    let special = b.select(func, fraction_ne, nan, retained);
    let exp_special = b.equal_constant(func, exponent, 0x7f80_0000);
    b.select(func, exp_special, special, rounded)
}

fn encode_fp8(func: &mut Function, b: &mut NumericBuilder, bits: Value, e4m3: bool) -> Value {
    let hoist15 = b.shifted(func, BinOp::Shr, bits, 24);
    let sign = b.masked(func, hoist15, 0x80);
    let hoist16 = b.shifted(func, BinOp::Shr, bits, 23);
    let source_exponent = b.masked(func, hoist16, 0xff);
    let source_fraction = b.masked(func, bits, 0x007f_ffff);
    let hoist0 = b.constant(func, 0x0080_0000);
    let significand = b.binary(func, BinOp::BitOr, source_fraction, hoist0);
    let threshold: u32 = if e4m3 { 121 } else { 113 };
    let shift_bias: u32 = if e4m3 { 141 } else { 134 };
    let bias_v = b.constant(func, shift_bias);
    let original_shift = b.binary(func, BinOp::Sub, bias_v, source_exponent);
    let hoist17 = b.constant(func, 24);
    let shift_too_large = b.compare(func, CmpOp::Gt, original_shift, hoist17);
    let hoist18 = b.constant(func, 24);
    let bounded_shift = b.select(func, shift_too_large, hoist18, original_shift);
    let hoist1 = b.constant(func, threshold);
    let subnormal_path = b.compare(func, CmpOp::Lt, source_exponent, hoist1);
    let hoist19 = b.constant(func, 24);
    let safe_shift = b.select(func, subnormal_path, bounded_shift, hoist19);
    let subnormal_rounded = b.round_right(func, significand, safe_shift);
    let hoist20 = b.constant(func, 0);
    let subnormal = b.select(func, shift_too_large, hoist20, subnormal_rounded);

    let normal_shift_const: u32 = if e4m3 { 20 } else { 21 };
    let normal_shift = b.constant(func, normal_shift_const);
    let normal_rounded = b.round_right(func, significand, normal_shift);
    let carry_value: u32 = if e4m3 { 16 } else { 8 };
    let retained_value: u32 = if e4m3 { 8 } else { 4 };
    let carry = b.equal_constant(func, normal_rounded, carry_value);
    let hoist21 = b.constant(func, retained_value);
    let rounded = b.select(func, carry, hoist21, normal_rounded);
    let exponent_base_v: u32 = if e4m3 { 120 } else { 112 };
    let hoist2 = b.constant(func, exponent_base_v);
    let exponent_base = b.binary(func, BinOp::Sub, source_exponent, hoist2);
    let one = b.constant(func, 1);
    let zero = b.constant(func, 0);
    let carry_increment = b.select(func, carry, one, zero);
    let target_exponent = b.binary(func, BinOp::Add, exponent_base, carry_increment);
    let scaled_exponent = b.shifted(func, BinOp::Shl, target_exponent, if e4m3 { 3 } else { 2 });
    let hoist3 = b.constant(func, retained_value);
    let rounded_minus = b.binary(func, BinOp::Sub, rounded, hoist3);
    let normal = b.binary(func, BinOp::Add, scaled_exponent, rounded_minus);
    let finite_magnitude = b.select(func, subnormal_path, subnormal, normal);
    let max_finite: u32 = if e4m3 { 0x7e } else { 0x7b };
    let hoist4 = b.constant(func, max_finite);
    let too_large = b.compare(func, CmpOp::Gt, finite_magnitude, hoist4);
    let hoist22 = b.constant(func, max_finite);
    let saturated = b.select(func, too_large, hoist22, finite_magnitude);
    let finite = b.binary(func, BinOp::BitOr, sign, saturated);

    let nan = if e4m3 {
        let hoist23 = b.constant(func, 0x7f);
        b.binary(func, BinOp::BitOr, sign, hoist23)
    } else {
        let hoist24 = b.shifted(func, BinOp::Shr, source_fraction, 21);
        let retained_nan = b.masked(func, hoist24, 0x03);
        let hoist25 = b.constant(func, 0x7e);
        let payload = b.binary(func, BinOp::BitOr, hoist25, retained_nan);
        b.binary(func, BinOp::BitOr, sign, payload)
    };
    let infinity_const: u32 = if e4m3 { 0x7e } else { 0x7c };
    let hoist26 = b.constant(func, infinity_const);
    let infinity = b.binary(func, BinOp::BitOr, sign, hoist26);
    let hoist27 = b.constant(func, 0);
    let fraction_ne = b.compare(func, CmpOp::Ne, source_fraction, hoist27);
    let special = b.select(func, fraction_ne, nan, infinity);
    let exponent_zero = b.equal_constant(func, source_exponent, 0);
    let non_special = b.select(func, exponent_zero, sign, finite);
    let exponent_ff = b.equal_constant(func, source_exponent, 0xff);
    b.select(func, exponent_ff, special, non_special)
}
// --- NVFP4 -------------------------------------------------------------------

struct NvFp4Expansion {
    result: Value,
    semantic_inst: Inst,
}

fn scale_op(application: ScaleApplication, inverse: bool) -> BinOp {
    match application {
        ScaleApplication::Multiply => {
            if inverse {
                BinOp::Div
            } else {
                BinOp::Mul
            }
        }
        ScaleApplication::Divide => {
            if inverse {
                BinOp::Mul
            } else {
                BinOp::Div
            }
        }
    }
}

/// Replace scalar NVFP4 conversions with integer classification and ordered
/// f32 scale operations. Returns whether anything changed.
pub fn expand_nv_fp4(func: &mut Function) -> bool {
    let mut has_nvfp4 = false;
    for bi in 0..func.block_count() {
        let block = Block::from_u32(bi as u32);
        for inst in func.block_insts(block).to_vec() {
            if matches!(
                func.opcode(inst),
                Opcode::DequantizeNvfp4(_) | Opcode::QuantizeNvfp4(_)
            ) {
                has_nvfp4 = true;
                break;
            }
        }
        if has_nvfp4 {
            break;
        }
    }
    if !has_nvfp4 {
        return false;
    }

    let ct = intern_common(func);
    for bi in 0..func.block_count() {
        let block = Block::from_u32(bi as u32);
        let contains_nvfp4 = func.block_insts(block).iter().any(|i| {
            matches!(
                func.opcode(*i),
                Opcode::DequantizeNvfp4(_) | Opcode::QuantizeNvfp4(_)
            )
        });
        if !contains_nvfp4 {
            continue;
        }

        let original = func.block_insts(block).to_vec();
        let mut builder = NumericBuilder {
            out: Vec::new(),
            ..NumericBuilder::common(func, &ct)
        };
        let mut out: Vec<Inst> = core::mem::take(&mut builder.out);
        for inst in original {
            match func.opcode(inst).clone() {
                Opcode::DequantizeNvfp4(conversion) => {
                    let Some(old_result) = func.inst_result(inst) else {
                        continue;
                    };
                    let expanded = dequantize_nvfp4(func, &mut builder, &conversion);
                    func.replace_all_uses(old_result, expanded.result);
                    let semantic = expanded.semantic_inst;
                    func.retarget_attrs(AttrTarget::Inst(inst), AttrTarget::Inst(semantic));
                    func.retarget_attrs(
                        AttrTarget::Value(old_result),
                        AttrTarget::Value(expanded.result),
                    );
                    out.append(&mut builder.out);
                }
                Opcode::QuantizeNvfp4(conversion) => {
                    let Some(old_result) = func.inst_result(inst) else {
                        continue;
                    };
                    let expanded = quantize_nvfp4(func, &mut builder, &conversion);
                    func.replace_all_uses(old_result, expanded.result);
                    let semantic = expanded.semantic_inst;
                    func.retarget_attrs(AttrTarget::Inst(inst), AttrTarget::Inst(semantic));
                    func.retarget_attrs(
                        AttrTarget::Value(old_result),
                        AttrTarget::Value(expanded.result),
                    );
                    out.append(&mut builder.out);
                }
                _ => out.push(inst),
            }
        }
        func.set_block_insts(block, &out);
    }
    true
}

fn dequantize_nvfp4(
    func: &mut Function,
    b: &mut NumericBuilder,
    conv: &NvFp4Convert,
) -> NvFp4Expansion {
    let carrier = b.convert(func, b.u32_t, conv.value);
    let nibble = b.masked(func, carrier, 0x0f);
    let sign_part = b.masked(func, nibble, 0x08);
    let sign = b.shifted(func, BinOp::Shl, sign_part, 28);
    let magnitude_index = b.masked(func, nibble, 0x07);
    let mut magnitude = b.constant(func, 0);
    for (index, bits) in [
        0x3f00_0000u32,
        0x3f80_0000,
        0x3fc0_0000,
        0x4000_0000,
        0x4040_0000,
        0x4080_0000,
        0x40c0_0000,
    ]
    .into_iter()
    .enumerate()
    {
        let cond = b.equal_constant(func, magnitude_index, index as u32 + 1);
        let hoist28 = b.constant(func, bits);
        magnitude = b.select(func, cond, hoist28, magnitude);
    }
    let value_bits = b.binary(func, BinOp::BitOr, sign, magnitude);
    let value = b.reinterpret(func, b.f32_t, value_bits);

    let block_payload = b.convert(func, b.u32_t, conv.block_scale);
    let block_bits = decode_e4m3(func, b, block_payload);
    let block_scale = b.reinterpret(func, b.f32_t, block_bits);
    let local = b.float_binary(
        func,
        scale_op(conv.block_application, false),
        value,
        block_scale,
    );
    let scaled = b.float_binary(
        func,
        scale_op(conv.global_application, false),
        local,
        conv.global_scale,
    );
    let semantic_inst = func.defining_inst(scaled).unwrap();

    // This bit round trip blocks a backend from fusing the final scale with an
    // original consumer.
    let scaled_bits = b.reinterpret(func, b.u32_t, scaled);
    let result = b.reinterpret(func, b.f32_t, scaled_bits);
    NvFp4Expansion {
        result,
        semantic_inst,
    }
}

fn quantize_nvfp4(
    func: &mut Function,
    b: &mut NumericBuilder,
    conv: &NvFp4Convert,
) -> NvFp4Expansion {
    let block_payload = b.convert(func, b.u32_t, conv.block_scale);
    let block_bits = decode_e4m3(func, b, block_payload);
    let block_scale = b.reinterpret(func, b.f32_t, block_bits);
    let global_unscaled = b.float_binary(
        func,
        scale_op(conv.global_application, true),
        conv.value,
        conv.global_scale,
    );
    let local_unscaled = b.float_binary(
        func,
        scale_op(conv.block_application, true),
        global_unscaled,
        block_scale,
    );
    let bits = b.reinterpret(func, b.u32_t, local_unscaled);
    let semantic_inst = func.defining_inst(bits).unwrap();

    let shifted_bits = b.shifted(func, BinOp::Shr, bits, 28);
    let sign = b.masked(func, shifted_bits, 0x08);
    let magnitude = b.masked(func, bits, 0x7fff_ffff);
    let mut payload = b.constant(func, 0x07);
    for (relation, threshold_bits, threshold_payload) in [
        (CmpOp::Le, 0x40a0_0000u32, 0x06u32),
        (CmpOp::Lt, 0x4060_0000, 0x05),
        (CmpOp::Le, 0x4020_0000, 0x04),
        (CmpOp::Lt, 0x3fe0_0000, 0x03),
        (CmpOp::Le, 0x3fa0_0000, 0x02),
        (CmpOp::Lt, 0x3f40_0000, 0x01),
        (CmpOp::Le, 0x3e80_0000, 0x00),
    ] {
        let hoist29 = b.constant(func, threshold_bits);
        let cond = b.compare(func, relation, magnitude, hoist29);
        let hoist30 = b.constant(func, threshold_payload);
        payload = b.select(func, cond, hoist30, payload);
    }
    let signed_payload = b.binary(func, BinOp::BitOr, sign, payload);
    let hoist5 = b.constant(func, 0x7f80_0000);
    let is_nan = b.compare(func, CmpOp::Gt, magnitude, hoist5);
    let hoist31 = b.constant(func, 0x07);
    let canonical = b.select(func, is_nan, hoist31, signed_payload);
    let result = b.convert(func, b.u8_t, canonical);
    NvFp4Expansion {
        result,
        semantic_inst,
    }
}
// --- F32 division ------------------------------------------------------------

struct NormalizedF32 {
    significand: Value,
    exponent: Value,
}

fn normalize_f32(func: &mut Function, b: &mut NumericBuilder, magnitude: Value) -> NormalizedF32 {
    let fraction = b.masked(func, magnitude, 0x007f_ffff);
    let shifted_exp = b.shifted(func, BinOp::Shr, magnitude, 23);
    let raw_exponent = b.masked(func, shifted_exp, 0xff);
    let is_subnormal = b.equal_constant(func, raw_exponent, 0);
    let hoist6 = b.constant(func, 0x0080_0000);
    let hidden_bit = b.binary(func, BinOp::BitOr, fraction, hoist6);
    let significand = b.select(func, is_subnormal, fraction, hidden_bit);
    let hoist32 = b.constant(func, 256);
    let exp_plus = b.binary(func, BinOp::Add, raw_exponent, hoist32);
    let hoist33 = b.constant(func, 257);
    let exponent = b.select(func, is_subnormal, hoist33, exp_plus);
    let mut significand = significand;
    let mut exponent = exponent;
    for shift in [16u32, 8, 4, 2, 1] {
        let threshold_v = b.constant(func, 0x0100_0000u32 >> shift);
        let needs_shift = b.compare(func, CmpOp::Lt, significand, threshold_v);
        let shifted_sig = b.shifted(func, BinOp::Shl, significand, shift);
        significand = b.select(func, needs_shift, shifted_sig, significand);
        let hoist7 = b.constant(func, shift as i64 as u32);
        let shifted_exp = b.binary(func, BinOp::Sub, exponent, hoist7);
        exponent = b.select(func, needs_shift, shifted_exp, exponent);
    }
    NormalizedF32 {
        significand,
        exponent,
    }
}

fn bool_as_u32(func: &mut Function, b: &mut NumericBuilder, condition: Value) -> Value {
    let one = b.constant(func, 1);
    let zero = b.constant(func, 0);
    b.select(func, condition, one, zero)
}

fn flag_set(func: &mut Function, b: &mut NumericBuilder, flag: Value) -> Value {
    let zero = b.constant(func, 0);
    b.compare(func, CmpOp::Ne, flag, zero)
}

fn round_division_quotient(
    func: &mut Function,
    b: &mut NumericBuilder,
    quotient: Value,
    remainder: Value,
    shift: Value,
) -> Value {
    let one = b.constant(func, 1);
    let retained = b.binary(func, BinOp::Shr, quotient, shift);
    let shifted_one = b.binary(func, BinOp::Shl, one, shift);
    let mask = b.binary(func, BinOp::Sub, shifted_one, one);
    let discarded = b.binary(func, BinOp::BitAnd, quotient, mask);
    let shift_minus_one = b.binary(func, BinOp::Sub, shift, one);
    let halfway = b.binary(func, BinOp::Shl, one, shift_minus_one);
    let discarded_gt = b.compare(func, CmpOp::Gt, discarded, halfway);
    let above_half = bool_as_u32(func, b, discarded_gt);
    let discarded_eq = b.compare(func, CmpOp::Eq, discarded, halfway);
    let at_half = bool_as_u32(func, b, discarded_eq);
    let zero = b.constant(func, 0);
    let rem_nonzero = b.compare(func, CmpOp::Ne, remainder, zero);
    let has_remainder = bool_as_u32(func, b, rem_nonzero);
    let odd_bits = b.binary(func, BinOp::BitAnd, retained, one);
    let odd_bits_nonzero = b.compare(func, CmpOp::Ne, odd_bits, zero);
    let odd = bool_as_u32(func, b, odd_bits_nonzero);
    let has_remainder_set = flag_set(func, b, has_remainder);
    let half_increment = b.select(func, has_remainder_set, one, odd);
    let at_half_set = flag_set(func, b, at_half);
    let tie_increment = b.select(func, at_half_set, half_increment, zero);
    let above_half_set = flag_set(func, b, above_half);
    let increment = b.select(func, above_half_set, one, tie_increment);
    b.binary(func, BinOp::Add, retained, increment)
}

fn expand_f32_division(
    func: &mut Function,
    b: &mut NumericBuilder,
    numerator: Value,
    denominator: Value,
) -> Value {
    let numerator_bits = b.reinterpret(func, b.u32_t, numerator);
    let denominator_bits = b.reinterpret(func, b.u32_t, denominator);
    let numerator_magnitude = b.masked(func, numerator_bits, 0x7fff_ffff);
    let denominator_magnitude = b.masked(func, denominator_bits, 0x7fff_ffff);
    let signs = b.binary(func, BinOp::BitXor, numerator_bits, denominator_bits);
    let sign = b.masked(func, signs, 0x8000_0000);
    let normalized_numerator = normalize_f32(func, b, numerator_magnitude);
    let normalized_denominator = normalize_f32(func, b, denominator_magnitude);

    let ratio_lt = b.compare(
        func,
        CmpOp::Lt,
        normalized_numerator.significand,
        normalized_denominator.significand,
    );
    let ratio_below_one = bool_as_u32(func, b, ratio_lt);
    let ratio_set = flag_set(func, b, ratio_below_one);
    let shifted_num = b.shifted(func, BinOp::Shl, normalized_numerator.significand, 1);
    let dividend = b.select(
        func,
        ratio_set,
        shifted_num,
        normalized_numerator.significand,
    );
    let exp_bias = b.constant(func, 383);
    let exp_diff = b.binary(
        func,
        BinOp::Sub,
        normalized_numerator.exponent,
        normalized_denominator.exponent,
    );
    let mut result_exponent = b.binary(func, BinOp::Add, exp_diff, exp_bias);
    result_exponent = b.binary(func, BinOp::Sub, result_exponent, ratio_below_one);

    let mut quotient = b.constant(func, 0);
    let mut remainder = dividend;
    for step in 0..27 {
        let bit = b.compare(
            func,
            CmpOp::Ge,
            remainder,
            normalized_denominator.significand,
        );
        let zero_sub = b.constant(func, 0);
        let subtrahend = b.select(func, bit, normalized_denominator.significand, zero_sub);
        let remainder_sub = b.binary(func, BinOp::Sub, remainder, subtrahend);
        remainder = remainder_sub;
        let bit_flag = bool_as_u32(func, b, bit);
        let shifted_q = b.shifted(func, BinOp::Shl, quotient, 1);
        quotient = b.binary(func, BinOp::BitOr, shifted_q, bit_flag);
        if step != 26 {
            remainder = b.shifted(func, BinOp::Shl, remainder, 1);
        }
    }

    let hh1 = b.constant(func, 256);
    let exp_le = b.compare(func, CmpOp::Le, result_exponent, hh1);
    let is_tiny = bool_as_u32(func, b, exp_le);
    let hh2 = b.constant(func, 260);
    let raw_tiny_shift = b.binary(func, BinOp::Sub, hh2, result_exponent);
    let hh3 = b.constant(func, 31);
    let shift_gt = b.compare(func, CmpOp::Gt, raw_tiny_shift, hh3);
    let shift_too_large = bool_as_u32(func, b, shift_gt);
    let shift_large_set = flag_set(func, b, shift_too_large);
    let hh4 = b.constant(func, 31);
    let tiny_shift = b.select(func, shift_large_set, hh4, raw_tiny_shift);
    let tiny_set = flag_set(func, b, is_tiny);
    let hh5 = b.constant(func, 3);
    let rounding_shift = b.select(func, tiny_set, tiny_shift, hh5);
    let rounded = round_division_quotient(func, b, quotient, remainder, rounding_shift);

    let hh6 = b.constant(func, 0x0100_0000);
    let carry_ge = b.compare(func, CmpOp::Ge, rounded, hh6);
    let rounded_carry = bool_as_u32(func, b, carry_ge);
    let carry_set = flag_set(func, b, rounded_carry);
    let halved = b.shifted(func, BinOp::Shr, rounded, 1);
    let normal_significand = b.select(func, carry_set, halved, rounded);
    let hh7 = b.constant(func, 256);
    let exp_minus = b.binary(func, BinOp::Sub, result_exponent, hh7);
    let normal_exponent = b.binary(func, BinOp::Add, exp_minus, rounded_carry);
    let shifted_exponent = b.shifted(func, BinOp::Shl, normal_exponent, 23);
    let masked_sig = b.masked(func, normal_significand, 0x007f_ffff);
    let normal_bits = b.binary(func, BinOp::BitOr, shifted_exponent, masked_sig);
    let tiny_set2 = flag_set(func, b, is_tiny);
    let finite_bits = b.select(func, tiny_set2, rounded, normal_bits);
    let hh8 = b.constant(func, 511);
    let overflow_before = b.compare(func, CmpOp::Ge, result_exponent, hh8);
    let overflow_before = bool_as_u32(func, b, overflow_before);
    let hh9 = b.constant(func, 255);
    let overflow_after = b.compare(func, CmpOp::Ge, normal_exponent, hh9);
    let overflow_after = bool_as_u32(func, b, overflow_after);
    let tiny_set3 = flag_set(func, b, is_tiny);
    let hh10 = b.constant(func, 0);
    let rounded_overflow = b.select(func, tiny_set3, hh10, overflow_after);
    let before_set = flag_set(func, b, overflow_before);
    let hh11 = b.constant(func, 1);
    let overflow = b.select(func, before_set, hh11, rounded_overflow);
    let one_v = b.constant(func, 1);
    let zero_v = b.constant(func, 0);
    let overflow_ne = b.compare(func, CmpOp::Ne, overflow, zero_v);
    let hh12 = b.constant(func, 0x7f80_0000);
    let mut result_bits = b.select(func, overflow_ne, hh12, finite_bits);

    let num_zero_eq = b.equal_constant(func, numerator_magnitude, 0);
    let numerator_zero = bool_as_u32(func, b, num_zero_eq);
    let den_zero_eq = b.equal_constant(func, denominator_magnitude, 0);
    let denominator_zero = bool_as_u32(func, b, den_zero_eq);
    let num_inf_eq = b.equal_constant(func, numerator_magnitude, 0x7f80_0000);
    let numerator_infinity = bool_as_u32(func, b, num_inf_eq);
    let den_inf_eq = b.equal_constant(func, denominator_magnitude, 0x7f80_0000);
    let denominator_infinity = bool_as_u32(func, b, den_inf_eq);
    let hh13 = b.constant(func, 0x7f80_0000);
    let num_nan_gt = b.compare(func, CmpOp::Gt, numerator_magnitude, hh13);
    let numerator_nan = bool_as_u32(func, b, num_nan_gt);
    let hh14 = b.constant(func, 0x7f80_0000);
    let den_nan_gt = b.compare(func, CmpOp::Gt, denominator_magnitude, hh14);
    let denominator_nan = bool_as_u32(func, b, den_nan_gt);
    let di_set = flag_set(func, b, denominator_infinity);
    let hh15 = b.constant(func, 0);
    result_bits = b.select(func, di_set, hh15, result_bits);
    let ni_set = flag_set(func, b, numerator_infinity);
    let hh16 = b.constant(func, 0x7f80_0000);
    result_bits = b.select(func, ni_set, hh16, result_bits);
    let dz_set = flag_set(func, b, denominator_zero);
    let hh17 = b.constant(func, 0x7f80_0000);
    result_bits = b.select(func, dz_set, hh17, result_bits);
    let nz_set = flag_set(func, b, numerator_zero);
    let hh18 = b.constant(func, 0);
    result_bits = b.select(func, nz_set, hh18, result_bits);
    result_bits = b.binary(func, BinOp::BitOr, sign, result_bits);

    let nz_set2 = flag_set(func, b, numerator_zero);
    let both_zero = b.select(func, nz_set2, denominator_zero, zero_v);
    let ni_set2 = flag_set(func, b, numerator_infinity);
    let both_infinite = b.select(func, ni_set2, denominator_infinity, zero_v);
    let nn_set = flag_set(func, b, numerator_nan);
    let invalid = b.select(func, nn_set, one_v, denominator_nan);
    let invalid0 = b.binary(func, BinOp::BitOr, invalid, both_zero);
    let invalid = b.binary(func, BinOp::BitOr, invalid0, both_infinite);
    let invalid_ne = b.compare(func, CmpOp::Ne, invalid, zero_v);
    let hh19 = b.constant(func, 0x7fc0_0000);
    result_bits = b.select(func, invalid_ne, hh19, result_bits);
    b.reinterpret(func, b.f32_t, result_bits)
}

/// Replace every scalar f32 division with target-independent u32 operations.
/// Returns whether anything changed.
pub fn expand_f32_div(func: &mut Function) -> bool {
    let mut has_division = false;
    for bi in 0..func.block_count() {
        let block = Block::from_u32(bi as u32);
        for inst in func.block_insts(block).to_vec() {
            if let Opcode::Arith(a) = func.opcode(inst) {
                if a.op != BinOp::Div {
                    continue;
                }
                let Some(result) = func.inst_result(inst) else {
                    continue;
                };
                if matches!(
                    func.types.type_kind(func.value_type(result)),
                    TypeKind::Float(FloatKind::F32)
                ) {
                    has_division = true;
                    break;
                }
            }
        }
        if has_division {
            break;
        }
    }
    if !has_division {
        return false;
    }

    let ct = intern_common(func);
    for bi in 0..func.block_count() {
        let block = Block::from_u32(bi as u32);
        let mut contains_division = false;
        for inst in func.block_insts(block).to_vec() {
            if let Opcode::Arith(a) = func.opcode(inst) {
                if a.op != BinOp::Div {
                    continue;
                }
                let Some(result) = func.inst_result(inst) else {
                    continue;
                };
                if func.value_type(result) == ct.f32_t {
                    contains_division = true;
                }
            }
        }
        if !contains_division {
            continue;
        }

        let original = func.block_insts(block).to_vec();
        let mut builder = NumericBuilder::common(func, &ct);
        let mut out: Vec<Inst> = core::mem::take(&mut builder.out);
        for inst in original {
            let mut replaced: Option<(Value, Value, Value)> = None;
            if let Opcode::Arith(a) = func.opcode(inst)
                && a.op == BinOp::Div
                && let Some(result) = func.inst_result(inst)
                && func.value_type(result) == ct.f32_t
            {
                replaced = Some((result, a.lhs, a.rhs));
            }
            match replaced {
                None => out.push(inst),
                Some((old_result, lhs, rhs)) => {
                    let expanded = expand_f32_division(func, &mut builder, lhs, rhs);
                    func.replace_all_uses(old_result, expanded);
                    func.retarget_attrs(AttrTarget::Value(old_result), AttrTarget::Value(expanded));
                    let defining = func.defining_inst(expanded).unwrap();
                    func.retarget_attrs(AttrTarget::Inst(inst), AttrTarget::Inst(defining));
                    out.append(&mut builder.out);
                }
            }
        }
        func.set_block_insts(block, &out);
    }
    true
}

// --- Mulh --------------------------------------------------------------------

fn is_mulh(func: &Function, inst: Inst) -> bool {
    matches!(func.opcode(inst), Opcode::Arith(a) if a.op == BinOp::Mulh)
}

/// Expand the high half of a full-width product into half-width limbs.
/// Returns whether anything changed.
pub fn expand_mulh(func: &mut Function) -> bool {
    let mut changed = false;
    for bi in 0..func.block_count() {
        let block = Block::from_u32(bi as u32);
        let mut has = false;
        for inst in func.block_insts(block).to_vec() {
            if is_mulh(func, inst) {
                has = true;
                break;
            }
        }
        if !has {
            continue;
        }
        changed = true;
        let original = func.block_insts(block).to_vec();
        let mut out: Vec<Inst> = Vec::new();
        for inst in original {
            if !is_mulh(func, inst) {
                out.push(inst);
                continue;
            }
            let Opcode::Arith(a) = func.opcode(inst).clone() else {
                continue;
            };
            let Some(result) = func.inst_result(inst) else {
                continue;
            };
            let ty = func.value_type(result);
            let high = emit_mulh_limbs(func, &mut out, a.lhs, a.rhs, ty);
            func.replace_all_uses(result, high);
        }
        func.set_block_insts(block, &out);
    }
    changed
}

/// Emit the limb sequence for `high half of (lhs * rhs)` at type `ty`,
/// appending each instruction to `out`, and return the high-half value.
fn emit_mulh_limbs(
    func: &mut Function,
    out: &mut Vec<Inst>,
    lhs: Value,
    rhs: Value,
    ty: Type,
) -> Value {
    let info = match func.types.type_kind(ty) {
        TypeKind::Int(i) => *i,
        _ => unreachable!("mulh is integer-only"),
    };
    let w = info.bits;
    let h = (w / 2) as i64;
    let mask_h: i64 = (1i64 << (w / 2)) - 1;

    let mut b = MulhBuilder { func, out, ty };

    let m = b.konst(mask_h);
    let hs = b.konst(h);

    let alo = b.op(BinOp::BitAnd, lhs, m);
    let ahi_s = b.op(BinOp::Shr, lhs, hs);
    let ahi = b.op(BinOp::BitAnd, ahi_s, m);
    let blo = b.op(BinOp::BitAnd, rhs, m);
    let bhi_s = b.op(BinOp::Shr, rhs, hs);
    let bhi = b.op(BinOp::BitAnd, bhi_s, m);

    let lolo = b.op(BinOp::Mul, alo, blo);
    let lohi = b.op(BinOp::Mul, alo, bhi);
    let hilo = b.op(BinOp::Mul, ahi, blo);
    let hihi = b.op(BinOp::Mul, ahi, bhi);

    let lolo_hi_s = b.op(BinOp::Shr, lolo, hs);
    let lolo_hi = b.op(BinOp::BitAnd, lolo_hi_s, m);
    let lohi_lo = b.op(BinOp::BitAnd, lohi, m);
    let hilo_lo = b.op(BinOp::BitAnd, hilo, m);
    let cross0 = b.op(BinOp::Add, lolo_hi, lohi_lo);
    let cross = b.op(BinOp::Add, cross0, hilo_lo);

    let lohi_hi_s = b.op(BinOp::Shr, lohi, hs);
    let lohi_hi = b.op(BinOp::BitAnd, lohi_hi_s, m);
    let hilo_hi_s = b.op(BinOp::Shr, hilo, hs);
    let hilo_hi = b.op(BinOp::BitAnd, hilo_hi_s, m);
    let cross_hi_s = b.op(BinOp::Shr, cross, hs);
    let cross_hi = b.op(BinOp::BitAnd, cross_hi_s, m);

    let s0 = b.op(BinOp::Add, hihi, lohi_hi);
    let s1 = b.op(BinOp::Add, s0, hilo_hi);
    let unsigned_high = b.op(BinOp::Add, s1, cross_hi);
    if !info.signed {
        return unsigned_high;
    }

    let wm1 = b.konst((w - 1) as i64);
    let amask = b.op(BinOp::Shr, lhs, wm1);
    let bmask = b.op(BinOp::Shr, rhs, wm1);
    let ca = b.op(BinOp::BitAnd, amask, rhs);
    let cb = b.op(BinOp::BitAnd, bmask, lhs);
    let c0 = b.op(BinOp::Sub, unsigned_high, ca);
    b.op(BinOp::Sub, c0, cb)
}

struct MulhBuilder<'a> {
    func: &'a mut Function,
    out: &'a mut Vec<Inst>,
    ty: Type,
}

impl MulhBuilder<'_> {
    fn konst(&mut self, c: i64) -> Value {
        let v = self.func.create_inst(self.ty, Opcode::Iconst(c));
        self.out.push(self.func.defining_inst(v).unwrap());
        v
    }
    fn op(&mut self, op: BinOp, x: Value, y: Value) -> Value {
        let v = self.func.create_inst(
            self.ty,
            Opcode::Arith(super::function::Arith { op, lhs: x, rhs: y }),
        );
        self.out.push(self.func.defining_inst(v).unwrap());
        v
    }
}

// --- Matmul ------------------------------------------------------------------

struct Shape {
    a_elem: Type,
    b_elem: Type,
    acc: Type,
    elem_bytes: i64,
    convert: bool,
    acc_is_float: bool,
}

/// C is always a 32-bit element.
const C_ELEM_BYTES: i64 = 4;

fn shape_of(func: &mut Function, mm: &MatMul) -> Shape {
    let f32_t = func.types.intern(TypeKind::Float(FloatKind::F32));
    let i32_t = func.types.intern(int_ty_kind(32, true));
    match mm.dtype {
        MatMulType::Fp32 => Shape {
            a_elem: f32_t,
            b_elem: f32_t,
            acc: f32_t,
            elem_bytes: 4,
            convert: false,
            acc_is_float: true,
        },
        MatMulType::Int8 | MatMulType::Uint8 => {
            // `input_signs`, when set, is AUTHORITATIVE per operand and `dtype`
            // then only names the hardware element type. Without it the dtype
            // decides both operands.
            let default_unsigned = mm.dtype == MatMulType::Uint8;
            let (a_unsigned, b_unsigned) = match mm.input_signs {
                Some(s) => (s.a_unsigned, s.b_unsigned),
                None => (default_unsigned, default_unsigned),
            };
            Shape {
                a_elem: func.types.intern(int_ty_kind(8, !a_unsigned)),
                b_elem: func.types.intern(int_ty_kind(8, !b_unsigned)),
                acc: i32_t,
                elem_bytes: 1,
                convert: true,
                acc_is_float: false,
            }
        }
        // fp16 loads half-width elements and widens each to the f32
        // accumulator. C stays 32-bit, so only A and B narrow.
        MatMulType::Fp16 => Shape {
            a_elem: func.types.intern(TypeKind::Float(FloatKind::F16)),
            b_elem: func.types.intern(TypeKind::Float(FloatKind::F16)),
            acc: func.types.intern(TypeKind::Float(FloatKind::F32)),
            elem_bytes: 2,
            convert: true,
            acc_is_float: true,
        },
    }
}

struct Site {
    block: Block,
    index: usize,
    mm: MatMul,
}

fn find_site(func: &Function) -> Option<Site> {
    for bi in 0..func.block_count() {
        let block = Block::from_u32(bi as u32);
        let mut branched = false;
        for (index, inst) in func.block_insts(block).iter().enumerate() {
            match func.opcode(*inst) {
                // An `if` does not terminate its block, so any instruction
                // after it is unreachable. A matmul behind one is left alone.
                Opcode::If(_) => branched = true,
                Opcode::Matmul(mm) => {
                    if branched {
                        continue;
                    }
                    if matmul_unsupported(&mm).is_some() {
                        continue;
                    }
                    return Some(Site {
                        block,
                        index,
                        mm: mm.clone(),
                    });
                }
                _ => {}
            }
        }
    }
    None
}

/// Whether any attribute is keyed by a block. `reorder_blocks` does not remap a
/// block id held in an attribute payload, so such a function has no safe
/// rewrite here.
fn has_block_attribute(func: &Function) -> bool {
    func.attribute_entries()
        .iter()
        .any(|entry| matches!(entry.target, AttrTarget::Block(_)))
}

/// Rewrite every supported `matmul` in `func` into a scalar loop nest over
/// plain loads, multiplies, adds and one store. Returns whether anything
/// changed.
pub fn expand_matmul(func: &mut Function) -> bool {
    if has_block_attribute(func) {
        return false;
    }
    let mut changed = false;
    while let Some(site) = find_site(func) {
        expand_matmul_site(func, &site);
        changed = true;
    }
    changed
}

struct Nest {
    i_head: Block,
    j_head: Block,
    j_body: Block,
    p_head: Block,
    p_body: Block,
    j_latch: Block,
    i_latch: Block,
    cont: Block,
}

fn expand_matmul_site(func: &mut Function, site: &Site) {
    let mm = site.mm.clone();
    let shape = shape_of(func, &mm);
    let old_block_count = func.block_count();

    let ptr_t = func.types.ptr_global();
    let bool_t = func.types.intern(TypeKind::Bool);
    let i32_t = func.types.intern(int_ty_kind(32, true));

    // Snapshot the block's instruction list: `set_block_insts` clears the list
    // it would then read from, so the two halves must be copied out first.
    let original = func.block_insts(site.block).to_vec();
    let before: Vec<Inst> = original[..site.index].to_vec();
    let after: Vec<Inst> = original[site.index + 1..].to_vec();

    let nest = Nest {
        i_head: func.append_block(),
        j_head: func.append_block(),
        j_body: func.append_block(),
        p_head: func.append_block(),
        p_body: func.append_block(),
        j_latch: func.append_block(),
        i_latch: func.append_block(),
        cont: func.append_block(),
    };

    let term = func.terminator(site.block);
    func.set_block_insts(nest.cont, &after);
    func.set_terminator(nest.cont, term.unwrap());
    func.set_block_insts(site.block, &before);
    *func.terminator_ptr(site.block) = None;

    // The preheader: the loop bounds and the fresh accumulator.
    let zero = func.append_inst(site.block, i32_t, Opcode::Iconst(0));
    let m_bound = func.append_inst(site.block, i32_t, Opcode::Iconst(mm.m as i64));
    let n_bound = func.append_inst(site.block, i32_t, Opcode::Iconst(mm.n as i64));
    let k_bound = func.append_inst(site.block, i32_t, Opcode::Iconst(mm.k as i64));
    let acc_zero = if shape.acc_is_float {
        func.append_inst(site.block, shape.acc, Opcode::Fconst(0.0))
    } else {
        func.append_inst(site.block, shape.acc, Opcode::Iconst(0))
    };
    func.set_jump(site.block, nest.i_head, &[zero, mm.a, mm.c]);

    // The i-loop carries the row of A and the running write pointer into C.
    let i = func.append_block_param(nest.i_head, i32_t);
    let a_row = func.append_block_param(nest.i_head, ptr_t);
    let c_row = func.append_block_param(nest.i_head, ptr_t);
    let i_lt = func.append_inst(
        nest.i_head,
        bool_t,
        Opcode::Icmp(super::function::Compare {
            op: CmpOp::Lt,
            lhs: i,
            rhs: m_bound,
        }),
    );
    func.append_if(
        nest.i_head,
        i_lt,
        super::function::EdgeDesc {
            target: nest.j_head,
            args: alloc::vec![zero, mm.b, c_row],
        },
        super::function::EdgeDesc::bare(nest.cont),
    );

    // The j-loop carries the column pointer into B and the element pointer
    // into C. The A row is invariant across j.
    let j = func.append_block_param(nest.j_head, i32_t);
    let b_col = func.append_block_param(nest.j_head, ptr_t);
    let c_elem = func.append_block_param(nest.j_head, ptr_t);
    let j_lt = func.append_inst(
        nest.j_head,
        bool_t,
        Opcode::Icmp(super::function::Compare {
            op: CmpOp::Lt,
            lhs: j,
            rhs: n_bound,
        }),
    );
    func.append_if(
        nest.j_head,
        j_lt,
        super::function::EdgeDesc::bare(nest.j_body),
        super::function::EdgeDesc::bare(nest.i_latch),
    );

    // The `accumulate` load lives HERE and not in the j-header, because the
    // j-header also runs with `j == n`, where `c_elem` is one past the end of
    // the C row.
    let acc_init = if mm.accumulate {
        func.append_inst(
            nest.j_body,
            shape.acc,
            Opcode::Load(super::function::Load {
                ptr: c_elem,
                volatile: false,
            }),
        )
    } else {
        acc_zero
    };
    func.set_jump(nest.j_body, nest.p_head, &[zero, acc_init, a_row, b_col]);

    // The p-loop: the reduction.
    let p = func.append_block_param(nest.p_head, i32_t);
    let acc = func.append_block_param(nest.p_head, shape.acc);
    let a_elem = func.append_block_param(nest.p_head, ptr_t);
    let b_elem = func.append_block_param(nest.p_head, ptr_t);
    let p_lt = func.append_inst(
        nest.p_head,
        bool_t,
        Opcode::Icmp(super::function::Compare {
            op: CmpOp::Lt,
            lhs: p,
            rhs: k_bound,
        }),
    );
    func.append_if(
        nest.p_head,
        p_lt,
        super::function::EdgeDesc::bare(nest.p_body),
        super::function::EdgeDesc::bare(nest.j_latch),
    );

    let a_val = func.append_inst(
        nest.p_body,
        shape.a_elem,
        Opcode::Load(super::function::Load {
            ptr: a_elem,
            volatile: false,
        }),
    );
    let b_val = func.append_inst(
        nest.p_body,
        shape.b_elem,
        Opcode::Load(super::function::Load {
            ptr: b_elem,
            volatile: false,
        }),
    );
    let a_op = if shape.convert {
        func.append_inst(
            nest.p_body,
            shape.acc,
            Opcode::Convert(super::function::Convert { value: a_val }),
        )
    } else {
        a_val
    };
    let b_op = if shape.convert {
        func.append_inst(
            nest.p_body,
            shape.acc,
            Opcode::Convert(super::function::Convert { value: b_val }),
        )
    } else {
        b_val
    };
    let product = func.append_inst(
        nest.p_body,
        shape.acc,
        Opcode::Arith(super::function::Arith {
            op: BinOp::Mul,
            lhs: a_op,
            rhs: b_op,
        }),
    );
    let next_acc = func.append_inst(
        nest.p_body,
        shape.acc,
        Opcode::Arith(super::function::Arith {
            op: BinOp::Add,
            lhs: acc,
            rhs: product,
        }),
    );
    let next_p = func.append_arith_imm(nest.p_body, i32_t, BinOp::Add, p, 1);
    let next_a_elem =
        func.append_arith_imm(nest.p_body, ptr_t, BinOp::Add, a_elem, shape.elem_bytes);
    let next_b_elem = func.append_arith_imm(
        nest.p_body,
        ptr_t,
        BinOp::Add,
        b_elem,
        (mm.n as i64) * shape.elem_bytes,
    );
    func.set_jump(
        nest.p_body,
        nest.p_head,
        &[next_p, next_acc, next_a_elem, next_b_elem],
    );

    // The j-latch: write the element, then step j, the B column, and C.
    func.append_store(nest.j_latch, acc, c_elem);
    let next_j = func.append_arith_imm(nest.j_latch, i32_t, BinOp::Add, j, 1);
    let next_b_col =
        func.append_arith_imm(nest.j_latch, ptr_t, BinOp::Add, b_col, shape.elem_bytes);
    let next_c_elem = func.append_arith_imm(nest.j_latch, ptr_t, BinOp::Add, c_elem, C_ELEM_BYTES);
    func.set_jump(
        nest.j_latch,
        nest.j_head,
        &[next_j, next_b_col, next_c_elem],
    );

    // The i-latch: step i and advance A by one k-element row.
    let next_i = func.append_arith_imm(nest.i_latch, i32_t, BinOp::Add, i, 1);
    let next_a_row = func.append_arith_imm(
        nest.i_latch,
        ptr_t,
        BinOp::Add,
        a_row,
        (mm.k as i64) * shape.elem_bytes,
    );
    func.set_jump(nest.i_latch, nest.i_head, &[next_i, next_a_row, c_elem]);

    lay_out_nest(func, site.block.index(), old_block_count, &nest);
}

/// Put the blocks in an order where every block follows its immediate
/// dominator. The nest is spliced in where its preheader already was, then the
/// blocks that followed the preheader.
fn lay_out_nest(func: &mut Function, preheader_index: usize, old_block_count: usize, nest: &Nest) {
    let mut order: Vec<Block> = Vec::with_capacity(func.block_count());
    for bi in 0..preheader_index + 1 {
        order.push(Block::from_u32(bi as u32));
    }
    order.extend([
        nest.i_head,
        nest.j_head,
        nest.j_body,
        nest.p_head,
        nest.p_body,
        nest.j_latch,
        nest.i_latch,
        nest.cont,
    ]);
    for bi in preheader_index + 1..old_block_count {
        order.push(Block::from_u32(bi as u32));
    }
    func.reorder_blocks(&order);
}

#[cfg(test)]
mod tests;
