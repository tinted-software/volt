//! Tests for the low-float, NVFP4, mulh and f32-division expansions.

use alloc::string::ToString;
use alloc::vec::Vec;

use super::super::*;
use crate::attribute::{AttrValue, Attribute, Custom};
use crate::expand::{expand_low_float, expand_mulh, expand_nv_fp4};
use crate::function::AttrTarget;
use crate::function::Function;
use crate::function::{Arith, Ret, Terminator, Unary, UnaryOp};
use crate::types::{IntDesc, Type, TypeKind};

fn unsigned_int(func: &mut Function, bits: u16) -> Type {
    func.types.intern(TypeKind::Int(IntDesc {
        signed: false,
        bits,
    }))
}

fn f32_ty(func: &mut Function) -> Type {
    func.types.intern(TypeKind::Float(FloatKind::F32))
}

fn u8_ty(func: &mut Function) -> Type {
    unsigned_int(func, 8)
}

fn u16_ty(func: &mut Function) -> Type {
    unsigned_int(func, 16)
}

fn u32_ty(func: &mut Function) -> Type {
    unsigned_int(func, 32)
}

fn custom_attr(key: &str) -> Attribute {
    Attribute::Custom(Custom {
        namespace: "debug".to_string(),
        key: key.to_string(),
        value: AttrValue::Flag,
    })
}

// --- Low float ---------------------------------------------------------------

#[test]
fn expand_low_float_no_op_without_trigger() {
    let mut func = Function::new();
    let block = func.append_block();
    let u32t = u32_ty(&mut func);
    let v = func.append_inst(block, u32t, Opcode::Iconst(41));
    let inst = func.defining_inst(v).unwrap();
    func.add_attr(AttrTarget::Value(v), custom_attr("unchanged"));
    func.set_terminator(block, Terminator::Ret(Ret::one(v)));
    let before = func.clone_func();
    // serialize a marker: still add the attr AFTER clone for the clone check
    func.add_attr(AttrTarget::Inst(inst), custom_attr("line"));
    let _ = before;
    assert!(!expand_low_float(&mut func));
    assert!(matches!(func.opcode(inst), Opcode::Iconst(_)));
}

#[test]
fn expand_low_float_rewrites_and_migrates_attrs() {
    let mut func = Function::new();
    let payload_t = u16_ty(&mut func);
    let f32t = f32_ty(&mut func);
    let block = func.append_block();
    let payload = func.append_block_param(block, payload_t);
    let decoded = func.append_inst(
        block,
        f32t,
        Opcode::DecodeLowFloat(crate::function::LowFloatConvert {
            value: payload,
            format: crate::low_float::Format::Bf16,
        }),
    );
    let orig_inst = func.defining_inst(decoded).unwrap();
    func.add_attr(AttrTarget::Value(decoded), custom_attr("semantic"));
    func.add_attr(AttrTarget::Inst(orig_inst), custom_attr("line"));
    func.set_terminator(block, Terminator::Ret(Ret::one(decoded)));

    assert!(expand_low_float(&mut func));
    assert!(!expand_low_float(&mut func));
    // No decode op survives.
    for inst in func.block_insts(block).to_vec() {
        assert!(!matches!(func.opcode(inst), Opcode::DecodeLowFloat(_)));
    }
    // The value attr landed on the final reinterpret (the return value's
    // definition), the inst attr migrated too, and the originals are gone.
    let ret_v = match func.terminator(block) {
        Some(Terminator::Ret(r)) => r.values[0],
        _ => panic!("ret"),
    };
    let found_value_attr = func
        .attributes_of(AttrTarget::Value(ret_v))
        .next_attr()
        .is_some();
    assert!(found_value_attr);
    assert!(
        func.attributes_of(AttrTarget::Value(decoded))
            .next_attr()
            .is_none()
    );
    assert!(
        func.attributes_of(AttrTarget::Inst(orig_inst))
            .next_attr()
            .is_none()
    );
    let mut migrated_inst_attr = false;
    for inst in func.block_insts(block).to_vec() {
        if func
            .attributes_of(AttrTarget::Inst(inst))
            .next_attr()
            .is_some()
        {
            migrated_inst_attr = true;
        }
    }
    assert!(migrated_inst_attr);
}

#[test]
fn expand_low_float_encode_fp8_moves_attrs_to_f32_boundary() {
    let mut func = Function::new();
    let f32t = f32_ty(&mut func);
    let u8t = u8_ty(&mut func);
    let block = func.append_block();
    let source = func.append_block_param(block, f32t);
    let encoded = func.append_inst(
        block,
        u8t,
        Opcode::EncodeLowFloat(crate::function::LowFloatConvert {
            value: source,
            format: crate::low_float::Format::F8E5M2,
        }),
    );
    let orig_inst = func.defining_inst(encoded).unwrap();
    func.add_attr(AttrTarget::Inst(orig_inst), custom_attr("line"));
    func.set_terminator(block, Terminator::Ret(Ret::one(encoded)));

    assert!(expand_low_float(&mut func));
    // The encoders retarget their inst attr onto the final reinterpret of the
    // f32 source (the leading unary reinterpret).
    let mut boundary: Option<crate::function::Inst> = None;
    for inst in func.block_insts(block).to_vec() {
        if let Opcode::Unary(Unary {
            op: UnaryOp::Reinterpret,
            value,
        }) = func.opcode(inst)
            && value == source
        {
            boundary = Some(inst);
        }
    }
    assert!(boundary.is_some());
    assert!(
        func.attributes_of(AttrTarget::Inst(boundary.unwrap()))
            .next_attr()
            .is_some()
    );
    assert!(
        func.attributes_of(AttrTarget::Inst(orig_inst))
            .next_attr()
            .is_none()
    );
    assert!(!expand_low_float(&mut func));
}

// --- NVFP4 -------------------------------------------------------------------

fn nvfp4_fixture(func: &mut Function, dequant: bool) -> (Block, Value) {
    let u8t = u8_ty(func);
    let f32t = f32_ty(func);
    let block = func.append_block();
    let payload = func.append_block_param(block, u8t);
    let block_scale = func.append_block_param(block, u8t);
    let value = func.append_block_param(block, f32t);
    let global_scale = func.append_block_param(block, f32t);
    let conv = crate::function::NvFp4Convert {
        value: if dequant { payload } else { value },
        block_scale,
        global_scale,
        block_application: crate::nvfp4::ScaleApplication::Multiply,
        global_application: crate::nvfp4::ScaleApplication::Divide,
    };
    let result = if dequant {
        func.append_inst(block, f32t, Opcode::DequantizeNvfp4(conv))
    } else {
        func.append_inst(block, u8t, Opcode::QuantizeNvfp4(conv))
    };
    func.set_terminator(block, Terminator::Ret(Ret::one(result)));
    (block, result)
}

#[test]
fn expand_nv_fp4_no_op_without_trigger() {
    let mut func = Function::new();
    let block = func.append_block();
    let u32t = u32_ty(&mut func);
    let v = func.append_inst(block, u32t, Opcode::Iconst(43));
    func.set_terminator(block, Terminator::Ret(Ret::one(v)));
    assert!(!expand_nv_fp4(&mut func));
    assert!(matches!(
        func.opcode(func.defining_inst(v).unwrap()),
        Opcode::Iconst(_)
    ));
}

#[test]
fn expand_nv_fp4_dequantize_has_two_f32_ops_and_reinterpret_barriers() {
    let mut func = Function::new();
    let (block, _original) = nvfp4_fixture(&mut func, true);
    assert!(expand_nv_fp4(&mut func));
    assert!(!expand_nv_fp4(&mut func));
    // Exactly two f32 arith ops in program order, and the return value goes
    // through reinterpret(u32)->reinterpret(f32) barrier chain whose source is
    // the second f32 op.
    let mut f32_ops: Vec<Inst> = Vec::new();
    for inst in func.block_insts(block).to_vec() {
        if let Opcode::Arith(Arith { op, .. }) = func.opcode(inst)
            && matches!(op, BinOp::Mul | BinOp::Div)
        {
            f32_ops.push(inst);
        }
    }
    assert_eq!(f32_ops.len(), 2);
    let _first = func.inst_result(f32_ops[0]).unwrap();
    let second = func.inst_result(f32_ops[1]).unwrap();
    // result chain: ret <- reinterpret f32 <- reinterpret u32(scaled=second)
    let ret_v = match func.terminator(block) {
        Some(Terminator::Ret(r)) => r.values[0],
        _ => panic!("ret"),
    };
    let final_reinterpret = func.defining_inst(ret_v).unwrap();
    match func.opcode(final_reinterpret) {
        Opcode::Unary(Unary {
            op: UnaryOp::Reinterpret,
            value,
        }) => {
            let bits = func.defining_inst(value).unwrap();
            match func.opcode(bits) {
                Opcode::Unary(Unary {
                    op: UnaryOp::Reinterpret,
                    value,
                }) => {
                    assert_eq!(value, second);
                }
                _ => panic!("expected reinterpret(u32)"),
            }
        }
        _ => panic!("expected reinterpret(f32)"),
    }
    // No dequantize op survives anywhere in the function.
    for inst in func.block_insts(block).to_vec() {
        assert!(!matches!(
            func.opcode(inst),
            Opcode::DequantizeNvfp4(_) | Opcode::QuantizeNvfp4(_)
        ));
    }
}

#[test]
fn expand_nv_fp4_quantize_has_two_inverse_f32_ops() {
    let mut func = Function::new();
    let (block, _original) = nvfp4_fixture(&mut func, false);
    assert!(expand_nv_fp4(&mut func));
    assert!(!expand_nv_fp4(&mut func));
    let mut f32_ops: Vec<BinOp> = Vec::new();
    for inst in func.block_insts(block).to_vec() {
        if let Opcode::Arith(Arith { op, .. }) = func.opcode(inst)
            && matches!(op, BinOp::Mul | BinOp::Div)
        {
            f32_ops.push(op);
        }
    }
    // global (Divide, inverse=true => mul) then block (Multiply, inverse=true
    // => div).
    assert_eq!(f32_ops, alloc::vec![BinOp::Mul, BinOp::Div]);
}

#[test]
fn expand_nv_fp4_migrates_attrs_to_semantic_boundary() {
    let mut func = Function::new();
    let (block, original) = nvfp4_fixture(&mut func, true);
    let orig_inst = func.defining_inst(original).unwrap();
    func.add_attr(AttrTarget::Inst(orig_inst), custom_attr("line"));
    func.add_attr(AttrTarget::Value(original), custom_attr("semantic"));
    assert!(expand_nv_fp4(&mut func));
    assert!(
        func.attributes_of(AttrTarget::Value(original))
            .next_attr()
            .is_none()
    );
    assert!(
        func.attributes_of(AttrTarget::Inst(orig_inst))
            .next_attr()
            .is_none()
    );
    let mut boundary_attr = false;
    for inst in func.block_insts(block).to_vec() {
        if let Opcode::Arith(Arith { op: BinOp::Div, .. }) = func.opcode(inst)
            && func
                .types
                .type_kind(func.value_type(func.inst_result(inst).unwrap()))
                == &TypeKind::Float(FloatKind::F32)
            && func
                .attributes_of(AttrTarget::Inst(inst))
                .next_attr()
                .is_some()
        {
            boundary_attr = true;
        }
    }
    assert!(boundary_attr);
}

// --- Mulh --------------------------------------------------------------------

fn mulh_const_fixture(func: &mut Function, bits: u16, signed: bool) -> (Block, Inst) {
    let ty = func.types.intern(TypeKind::Int(IntDesc { signed, bits }));
    let block = func.append_block();
    let a = func.append_inst(block, ty, Opcode::Iconst(0x1234_5678_9abcu64 as i64));
    let b = func.append_inst(block, ty, Opcode::Iconst(0xfeed_beefu64 as i64));
    let r = func.append_inst(
        block,
        ty,
        Opcode::Arith(Arith {
            op: BinOp::Mulh,
            lhs: a,
            rhs: b,
        }),
    );
    func.set_terminator(block, Terminator::Ret(Ret::one(r)));
    (block, func.defining_inst(r).unwrap())
}

#[test]
fn expand_mulh_no_mulh_survives_and_idempotent() {
    let mut func = Function::new();
    let (block, _) = mulh_const_fixture(&mut func, 64, true);
    assert!(expand_mulh(&mut func));
    assert!(!expand_mulh(&mut func));
    for inst in func.block_insts(block).to_vec() {
        assert!(!matches!(
            func.opcode(inst),
            Opcode::Arith(Arith {
                op: BinOp::Mulh,
                ..
            })
        ));
    }
}

#[test]
fn expand_mulh_false_without_trigger() {
    let mut func = Function::new();
    let ty = u32_ty(&mut func);
    let block = func.append_block();
    let a = func.append_inst(block, ty, Opcode::Iconst(3));
    let b = func.append_inst(block, ty, Opcode::Iconst(7));
    let r = func.append_inst(
        block,
        ty,
        Opcode::Arith(Arith {
            op: BinOp::Mul,
            lhs: a,
            rhs: b,
        }),
    );
    func.set_terminator(block, Terminator::Ret(Ret::one(r)));
    assert!(!expand_mulh(&mut func));
}

#[test]
fn expand_mulh_limb_math_matches_oracle_signed() {
    let mut func = Function::new();
    let (block, _) = mulh_const_fixture(&mut func, 64, true);
    assert!(expand_mulh(&mut func));
    // Evaluate constant dataflow at declared width: high = (a*b as i128) >> 64
    let ret_v = match func.terminator(block) {
        Some(Terminator::Ret(r)) => r.values[0],
        _ => panic!("ret"),
    };
    let got = eval_const(&func, ret_v);
    let a = 0x1234_5678_9abcu64 as i64;
    let b = 0xfeed_beefu64 as i64;
    let oracle = ((a as i128) * (b as i128)) >> 64;
    assert_eq!(got, oracle as i64);
}

#[test]
fn expand_mulh_limb_math_matches_oracle_unsigned() {
    let mut func = Function::new();
    let (block, _) = mulh_const_fixture(&mut func, 64, false);
    assert!(expand_mulh(&mut func));
    let ret_v = match func.terminator(block) {
        Some(Terminator::Ret(r)) => r.values[0],
        _ => panic!("ret"),
    };
    let got = eval_const(&func, ret_v) as u64;
    let a = 0x1234_5678_9abcu64;
    let b = 0xfeed_beefu64;
    let oracle = ((a as u128) * (b as u128)) >> 64;
    assert_eq!(got, oracle as u64);
}

#[test]
fn expand_mulh_works_at_32_bit_width() {
    let mut func = Function::new();
    let ty = signed_int32(&mut func);
    let block = func.append_block();
    let a = func.append_inst(block, ty, Opcode::Iconst(100_000));
    let b = func.append_inst(block, ty, Opcode::Iconst(100_000));
    let r = func.append_inst(
        block,
        ty,
        Opcode::Arith(Arith {
            op: BinOp::Mulh,
            lhs: a,
            rhs: b,
        }),
    );
    func.set_terminator(block, Terminator::Ret(Ret::one(r)));
    assert!(expand_mulh(&mut func));
    let ret_v = match func.terminator(block) {
        Some(Terminator::Ret(r)) => r.values[0],
        _ => panic!("ret"),
    };
    let got = eval_const(&func, ret_v);
    let oracle = ((100_000i128) * (100_000i128)) >> 32;
    assert_eq!(got, oracle as i64);
}

fn signed_int32(func: &mut Function) -> Type {
    func.types.intern(TypeKind::Int(IntDesc {
        signed: true,
        bits: 32,
    }))
}

/// Evaluate a constants-only dataflow at the result type's width.
fn eval_const(func: &Function, v: Value) -> i64 {
    let info = match func.types.type_kind(func.value_type(v)) {
        TypeKind::Int(i) => *i,
        _ => panic!("int expected"),
    };
    let wrap = |x: i64| -> i64 {
        let bits = info.bits;
        if bits >= 64 {
            return x;
        }
        let mask: u64 = if bits >= 64 { !0 } else { (1u64 << bits) - 1 };
        let low = (x as u64) & mask;
        if info.signed {
            let sign = 1u64 << (bits - 1);
            if low & sign != 0 {
                (low | !mask) as i64
            } else {
                low as i64
            }
        } else {
            low as i64
        }
    };
    let inst = func.defining_inst(v).unwrap();
    match func.opcode(inst) {
        Opcode::Iconst(c) => wrap(c),
        Opcode::Arith(a) => {
            let lhs = eval_const(func, a.lhs);
            let rhs = eval_const(func, a.rhs);
            let raw: i64 = match a.op {
                BinOp::Add => lhs.wrapping_add(rhs),
                BinOp::Sub => lhs.wrapping_sub(rhs),
                BinOp::Mul => lhs.wrapping_mul(rhs),
                BinOp::BitAnd => lhs & rhs,
                BinOp::BitOr => lhs | rhs,
                BinOp::Shr => lhs.wrapping_shr(rhs as u32),
                BinOp::Shl => lhs.wrapping_shl(rhs as u32),
                _ => panic!("unexpected op"),
            };
            wrap(raw)
        }
        Opcode::Select(sel) => {
            let cond = eval_const(func, sel.cond);
            if cond != 0 {
                eval_const(func, sel.then)
            } else {
                eval_const(func, sel.else_)
            }
        }
        Opcode::Convert(cv) => eval_const(func, cv.value),
        _ => panic!("unexpected opcode"),
    }
}
