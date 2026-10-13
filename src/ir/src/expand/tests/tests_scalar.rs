//! Tests for the low-float, NVFP4, mulh and f32-division expansions.

use alloc::string::ToString;
use alloc::vec::Vec;

use super::super::*;
use super::interp::Interp;
use crate::attribute::{AttrValue, Attribute, Custom};
use crate::expand::{expand_f32_div, expand_low_float, expand_mulh, expand_nv_fp4};
use crate::function::AttrTarget;
use crate::function::Function;
use crate::function::{Arith, Ret, Terminator, Unary, UnaryOp};
use crate::low_float::{self, Format};
use crate::nvfp4::{self, ScaleApplication};
use crate::types::{IntDesc, Type, TypeKind};
use crate::verify::{Profile, verify};

fn int_ty(func: &mut Function, bits: u16) -> Type {
    func.types.intern(TypeKind::Int(IntDesc { bits }))
}

fn assert_verifies(func: &Function) {
    let diags = verify(func, Profile::High);
    assert!(diags.ok(), "{:?}", diags.items());
}

fn f32_ty(func: &mut Function) -> Type {
    func.types.intern(TypeKind::Float(FloatKind::F32))
}

fn i8_ty(func: &mut Function) -> Type {
    int_ty(func, 8)
}

fn i16_ty(func: &mut Function) -> Type {
    int_ty(func, 16)
}

fn i32_ty(func: &mut Function) -> Type {
    int_ty(func, 32)
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
    let i32t = i32_ty(&mut func);
    let v = func.append_inst(block, i32t, Opcode::Iconst(41));
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
    let payload_t = i16_ty(&mut func);
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
    let i8t = i8_ty(&mut func);
    let block = func.append_block();
    let source = func.append_block_param(block, f32t);
    let encoded = func.append_inst(
        block,
        i8t,
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
    let i8t = i8_ty(func);
    let f32t = f32_ty(func);
    let block = func.append_block();
    let payload = func.append_block_param(block, i8t);
    let block_scale = func.append_block_param(block, i8t);
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
        func.append_inst(block, i8t, Opcode::QuantizeNvfp4(conv))
    };
    func.set_terminator(block, Terminator::Ret(Ret::one(result)));
    (block, result)
}

#[test]
fn expand_nv_fp4_no_op_without_trigger() {
    let mut func = Function::new();
    let block = func.append_block();
    let i32t = i32_ty(&mut func);
    let v = func.append_inst(block, i32t, Opcode::Iconst(43));
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

fn mulh_const_fixture(func: &mut Function, bits: u16, op: BinOp) -> (Block, Inst) {
    let ty = int_ty(func, bits);
    let block = func.append_block();
    let a = func.append_inst(block, ty, Opcode::Iconst(0x1234_5678_9abcu64 as i64));
    let b = func.append_inst(block, ty, Opcode::Iconst(0xfeed_beefu64 as i64));
    let r = func.append_inst(block, ty, Opcode::Arith(Arith { op, lhs: a, rhs: b }));
    func.set_terminator(block, Terminator::Ret(Ret::one(r)));
    (block, func.defining_inst(r).unwrap())
}

#[test]
fn expand_mulh_no_mulh_survives_and_idempotent() {
    for op in [BinOp::UMulh, BinOp::SMulh] {
        let mut func = Function::new();
        let (block, _) = mulh_const_fixture(&mut func, 64, op);
        assert!(expand_mulh(&mut func));
        assert!(!expand_mulh(&mut func));
        for inst in func.block_insts(block).to_vec() {
            assert!(!matches!(
                func.opcode(inst),
                Opcode::Arith(Arith {
                    op: BinOp::UMulh | BinOp::SMulh,
                    ..
                })
            ));
        }
        assert_verifies(&func);
    }
}

#[test]
fn expand_mulh_false_without_trigger() {
    let mut func = Function::new();
    let ty = i32_ty(&mut func);
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

/// The value `fn() { ret <lhs op rhs> }` returns, computed by the expanded limb math.
fn expanded_mulh(bits: u16, op: BinOp, lhs: u64, rhs: u64) -> u64 {
    let mut func = Function::new();
    let ty = int_ty(&mut func, bits);
    let block = func.append_block();
    let a = func.append_inst(block, ty, Opcode::Iconst(lhs as i64));
    let b = func.append_inst(block, ty, Opcode::Iconst(rhs as i64));
    let r = func.append_inst(block, ty, Opcode::Arith(Arith { op, lhs: a, rhs: b }));
    func.set_terminator(block, Terminator::Ret(Ret::one(r)));
    assert!(expand_mulh(&mut func));
    assert_verifies(&func);
    Interp::new(Vec::new()).run(&func, &[])[0]
}

#[test]
fn expand_mulh_limb_math_matches_oracle_signed() {
    let a = 0x1234_5678_9abcu64 as i64;
    let b = 0xfeed_beefu64 as i64;
    let oracle = ((a as i128) * (b as i128)) >> 64;
    assert_eq!(
        expanded_mulh(64, BinOp::SMulh, a as u64, b as u64),
        oracle as u64
    );
}

#[test]
fn expand_mulh_limb_math_matches_oracle_unsigned() {
    let a = 0x1234_5678_9abcu64;
    let b = 0xfeed_beefu64;
    let oracle = ((a as u128) * (b as u128)) >> 64;
    assert_eq!(expanded_mulh(64, BinOp::UMulh, a, b), oracle as u64);
}

#[test]
fn expand_mulh_works_at_32_bit_width() {
    let oracle = ((100_000i128) * (100_000i128)) >> 32;
    assert_eq!(
        expanded_mulh(32, BinOp::SMulh, 100_000, 100_000),
        oracle as u64
    );
    assert_eq!(
        expanded_mulh(32, BinOp::UMulh, 100_000, 100_000),
        oracle as u64
    );
}

#[test]
fn expand_mulh_matches_the_oracle_at_every_width_for_both_signs() {
    let samples: [u64; 9] = [
        0,
        1,
        2,
        0x7f,
        0x80,
        0xff,
        0x8000_0001,
        0x1234_5678_9abc_def0,
        u64::MAX,
    ];
    for bits in [8u16, 16, 32, 64] {
        let mask = if bits == 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };
        let signed = |x: u64| {
            if bits == 64 {
                x as i64
            } else {
                ((x << (64 - bits)) as i64) >> (64 - bits)
            }
        };
        for &a in &samples {
            for &b in &samples {
                let (a, b) = (a & mask, b & mask);
                let unsigned_want = ((u128::from(a) * u128::from(b)) >> bits) as u64 & mask;
                let signed_want =
                    ((i128::from(signed(a)) * i128::from(signed(b))) >> bits) as u64 & mask;
                assert_eq!(
                    expanded_mulh(bits, BinOp::UMulh, a, b),
                    unsigned_want,
                    "umulh i{bits} {a:#x} {b:#x}"
                );
                assert_eq!(
                    expanded_mulh(bits, BinOp::SMulh, a, b),
                    signed_want,
                    "smulh i{bits} {a:#x} {b:#x}"
                );
            }
        }
    }
}

// --- Numerics of the scalar expansions ----------------------------------------

/// Equal bit patterns, or both NaN (the expansions need not preserve a NaN payload).
fn same_f32(got: u32, want: u32) -> bool {
    got == want || (f32::from_bits(got).is_nan() && f32::from_bits(want).is_nan())
}

fn sample_floats() -> Vec<f32> {
    let mut v = alloc::vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        1.5,
        -2.5,
        0.1,
        -0.1,
        3.7,
        1e-3,
        1e-6,
        1e-10,
        0.001_953_125,
        0.000_976_562_5,
        6.0,
        5.0,
        0.75,
        0.25,
        448.0,
        449.0,
        464.0,
        57344.0,
        61440.0,
        65504.0,
        1e10,
        -1e10,
        f32::MAX,
        f32::MIN_POSITIVE,
        f32::from_bits(1),
        f32::from_bits(0x007f_ffff),
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    let mut s = 0x1234_5678_9abc_def0u64;
    for _ in 0..1500 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        v.push(f32::from_bits(s as u32));
        // Values near 1 exercise the rounding paths of the narrow formats.
        v.push(f32::from_bits(0x3f00_0000 + (s >> 40) as u32 % 0x0100_0000));
    }
    v
}

#[test]
fn expanded_low_float_decode_matches_the_host_decoder() {
    for format in [Format::Bf16, Format::F8E4M3, Format::F8E5M2] {
        let mut func = Function::new();
        let payload_t = int_ty(&mut func, format.payload_bits());
        let f32t = f32_ty(&mut func);
        let block = func.append_block();
        let payload = func.append_block_param(block, payload_t);
        let decoded = func.append_inst(
            block,
            f32t,
            Opcode::DecodeLowFloat(crate::function::LowFloatConvert {
                value: payload,
                format,
            }),
        );
        func.set_terminator(block, Terminator::Ret(Ret::one(decoded)));
        assert!(expand_low_float(&mut func));
        assert_verifies(&func);
        let payloads: Vec<u16> = if format == Format::Bf16 {
            alloc::vec![
                0, 1, 0x3f80, 0x8000, 0x7f80, 0xff80, 0x7fc0, 0xc2f7, 0x0001, 0x007f, 0x8001,
                0xffff
            ]
        } else {
            (0..=255).collect()
        };
        for p in payloads {
            let got = Interp::new(Vec::new()).run(&func, &[u64::from(p)])[0] as u32;
            let want = low_float::decode(format, p).unwrap().to_bits();
            assert!(
                same_f32(got, want),
                "{format:?} {p:#x}: {got:#x} != {want:#x}"
            );
        }
    }
}

#[test]
fn expanded_low_float_encode_matches_the_host_encoder() {
    for format in [Format::Bf16, Format::F8E4M3, Format::F8E5M2] {
        let mut func = Function::new();
        let payload_t = int_ty(&mut func, format.payload_bits());
        let f32t = f32_ty(&mut func);
        let block = func.append_block();
        let value = func.append_block_param(block, f32t);
        let encoded = func.append_inst(
            block,
            payload_t,
            Opcode::EncodeLowFloat(crate::function::LowFloatConvert { value, format }),
        );
        func.set_terminator(block, Terminator::Ret(Ret::one(encoded)));
        assert!(expand_low_float(&mut func));
        assert_verifies(&func);
        for v in sample_floats() {
            let got = Interp::new(Vec::new()).run(&func, &[u64::from(v.to_bits())])[0];
            let want = u64::from(low_float::encode(format, v));
            assert_eq!(got, want, "{format:?} {v:e} ({:#x})", v.to_bits());
        }
    }
}

const APPLICATIONS: [ScaleApplication; 2] = [ScaleApplication::Multiply, ScaleApplication::Divide];

fn nvfp4_function(
    dequant: bool,
    block_application: ScaleApplication,
    global_application: ScaleApplication,
) -> Function {
    let mut func = Function::new();
    let i8t = i8_ty(&mut func);
    let f32t = f32_ty(&mut func);
    let block = func.append_block();
    let payload = func.append_block_param(block, i8t);
    let value = func.append_block_param(block, f32t);
    let block_scale = func.append_block_param(block, i8t);
    let global_scale = func.append_block_param(block, f32t);
    let conv = crate::function::NvFp4Convert {
        value: if dequant { payload } else { value },
        block_scale,
        global_scale,
        block_application,
        global_application,
    };
    let result = if dequant {
        func.append_inst(block, f32t, Opcode::DequantizeNvfp4(conv))
    } else {
        func.append_inst(block, i8t, Opcode::QuantizeNvfp4(conv))
    };
    func.set_terminator(block, Terminator::Ret(Ret::one(result)));
    func
}

#[test]
fn expanded_nvfp4_dequantize_matches_the_host() {
    for block_application in APPLICATIONS {
        for global_application in APPLICATIONS {
            let mut func = nvfp4_function(true, block_application, global_application);
            assert!(expand_nv_fp4(&mut func));
            assert_verifies(&func);
            for payload in 0..16u8 {
                for scale in [0x00u8, 0x01, 0x08, 0x38, 0x40, 0x7e, 0x7f, 0xb8, 0xff] {
                    for global in [1.0f32, 0.5, 3.0, -2.0] {
                        let got = Interp::new(Vec::new()).run(
                            &func,
                            &[
                                u64::from(payload),
                                0,
                                u64::from(scale),
                                u64::from(global.to_bits()),
                            ],
                        )[0] as u32;
                        let want = nvfp4::dequantize(
                            payload,
                            scale,
                            global,
                            block_application,
                            global_application,
                        )
                        .to_bits();
                        assert!(
                            same_f32(got, want),
                            "{payload:#x} {scale:#x} {global}: {got:#x} != {want:#x}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn expanded_nvfp4_quantize_matches_the_host() {
    for block_application in APPLICATIONS {
        for global_application in APPLICATIONS {
            let mut func = nvfp4_function(false, block_application, global_application);
            assert!(expand_nv_fp4(&mut func));
            assert_verifies(&func);
            for value in sample_floats().into_iter().take(400) {
                for scale in [0x38u8, 0x40, 0x7e] {
                    let got = Interp::new(Vec::new()).run(
                        &func,
                        &[
                            0,
                            u64::from(value.to_bits()),
                            u64::from(scale),
                            u64::from(2.0f32.to_bits()),
                        ],
                    )[0];
                    let want = u64::from(nvfp4::quantize(
                        value,
                        scale,
                        2.0,
                        block_application,
                        global_application,
                    ));
                    assert_eq!(got, want, "{value:e} {scale:#x}");
                }
            }
        }
    }
}

#[test]
fn expanded_f32_division_matches_the_hardware() {
    let mut func = Function::new();
    let f32t = f32_ty(&mut func);
    let block = func.append_block();
    let a = func.append_block_param(block, f32t);
    let b = func.append_block_param(block, f32t);
    let q = func.append_inst(
        block,
        f32t,
        Opcode::Arith(Arith {
            op: BinOp::Div,
            lhs: a,
            rhs: b,
        }),
    );
    func.set_terminator(block, Terminator::Ret(Ret::one(q)));
    assert!(expand_f32_div(&mut func));
    assert!(!expand_f32_div(&mut func));
    assert_verifies(&func);
    for inst in func.block_insts(block).to_vec() {
        assert!(!matches!(
            func.opcode(inst),
            Opcode::Arith(Arith { op: BinOp::Div, .. })
        ));
    }
    let samples = sample_floats();
    for (i, &x) in samples.iter().enumerate() {
        for &y in samples.iter().skip(i % 7).step_by(23).take(60) {
            let got = Interp::new(Vec::new())
                .run(&func, &[u64::from(x.to_bits()), u64::from(y.to_bits())])[0]
                as u32;
            let want = (x / y).to_bits();
            assert!(same_f32(got, want), "{x:e} / {y:e}: {got:#x} != {want:#x}");
        }
    }
}
