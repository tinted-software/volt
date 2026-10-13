//! Tests for `expand_matmul`.

use alloc::string::ToString;

use super::interp::{Interp, f16_to_f32};
use crate::attribute::Attribute;
use crate::expand::expand_matmul;
use crate::function::{
    Arith, AttrTarget, ConvertKind, Function, InputSigns, MatMulQuant, MatMulQuantOut, MatMulScale,
    MatMulType, Opcode, Ret, Terminator,
};
use crate::types::{FloatKind, Type, TypeKind};
use crate::verify::{Profile, verify};
use alloc::{vec, vec::Vec};

fn f32_ty(func: &mut Function) -> Type {
    func.types.intern(TypeKind::Float(FloatKind::F32))
}

fn custom_attr(key: &str) -> Attribute {
    Attribute::Custom(crate::attribute::Custom {
        namespace: "debug".to_string(),
        key: key.to_string(),
        value: crate::attribute::AttrValue::Flag,
    })
}

fn matmul_fixture(
    func: &mut Function,
    dtype: MatMulType,
    quant: bool,
) -> Vec<crate::function::Inst> {
    let ptr_t = func.types.ptr_global();
    let block = func.append_block();
    let a = func.append_inst(block, ptr_t, Opcode::Iconst(0));
    let b = func.append_inst(block, ptr_t, Opcode::Iconst(0));
    let c = func.append_inst(block, ptr_t, Opcode::Iconst(0));
    func.append_matmul(block, a, b, c, 2, 2, 2, dtype, false);
    let matmul_inst = func.block_insts(block)[func.block_insts(block).len() - 1];
    if quant && let Opcode::Matmul(mm) = func.opcode_mut(matmul_inst) {
        mm.quant = Some(MatMulQuant {
            scale: MatMulScale::Scalar(1),
            relu: false,
            out: MatMulQuantOut::I8,
            bias: None,
            zero_point: 0,
        });
    }
    let _f32t = f32_ty(func);
    func.set_terminator(block, Terminator::Ret(Ret::none()));
    vec![matmul_inst]
}

fn count(func: &Function, pred: impl Fn(&Opcode) -> bool) -> usize {
    let mut n = 0;
    for bi in 0..func.block_count() {
        for inst in func
            .block_insts(crate::function::Block::from_u32(bi as u32))
            .to_vec()
        {
            if pred(&func.opcode(inst)) {
                n += 1;
            }
        }
    }
    n
}

#[test]
fn expand_matmul_rewrites_fp32_into_scalar_nest() {
    let mut func = Function::new();
    let _matmul = matmul_fixture(&mut func, MatMulType::Fp32, false);
    assert!(expand_matmul(&mut func), "supported matmul must expand");
    assert!(!expand_matmul(&mut func), "second run must be a no-op");
    assert_eq!(count(&func, |op| matches!(op, Opcode::Matmul(_))), 0);
    // Statically the nest holds one A load and one B load in the p-body and
    // one store in the j-latch.
    assert_eq!(count(&func, |op| matches!(op, Opcode::Load(_))), 2);
    assert_eq!(count(&func, |op| matches!(op, Opcode::Store(_))), 1);
}

#[test]
fn expand_matmul_leaves_unsupported_quant_in_place() {
    let mut func = Function::new();
    let _matmul = matmul_fixture(&mut func, MatMulType::Fp32, true);
    assert!(!expand_matmul(&mut func));
    assert_eq!(count(&func, |op| matches!(op, Opcode::Matmul(_))), 1);
}

#[test]
fn expand_matmul_continuation_moves_tail_and_terminator() {
    let mut func = Function::new();
    let ptr_t = func.types.ptr_global();
    let f32t = f32_ty(&mut func);
    let entry = func.append_block();
    let a = func.append_inst(entry, ptr_t, Opcode::Iconst(0));
    let b = func.append_inst(entry, ptr_t, Opcode::Iconst(0));
    let c = func.append_inst(entry, ptr_t, Opcode::Iconst(0));
    let marker = func.append_inst(entry, f32t, Opcode::Fconst(1.5));
    func.append_matmul(entry, a, b, c, 1, 1, 1, MatMulType::Fp32, false);
    let addend = func.append_inst(entry, f32t, Opcode::Fconst(2.5));
    let sum = func.append_inst(
        entry,
        f32t,
        Opcode::Arith(Arith {
            op: crate::function::BinOp::Add,
            lhs: marker,
            rhs: addend,
        }),
    );
    func.set_terminator(entry, Terminator::Ret(Ret::one(sum)));

    assert!(expand_matmul(&mut func));
    assert!(!expand_matmul(&mut func));
    // The marker (before the matmul) stays in the entry; the tail (after it)
    // and the terminator moved to the continuation.
    let entry_insts = func.block_insts(crate::function::Block(0)).to_vec();
    assert!(
        entry_insts
            .iter()
            .any(|i| matches!(func.opcode(*i), Opcode::Fconst(v) if v == 1.5))
    );
    // The instruction after the matmul moved out of the entry.
    assert!(
        !entry_insts
            .iter()
            .any(|i| matches!(func.opcode(*i), Opcode::Fconst(v) if v == 2.5))
    );
    let mut addend_lives_somewhere_else = false;
    for bi in 1..func.block_count() {
        for inst in func
            .block_insts(crate::function::Block::from_u32(bi as u32))
            .to_vec()
        {
            if matches!(func.opcode(inst), Opcode::Fconst(v) if v == 2.5) {
                addend_lives_somewhere_else = true;
            }
        }
    }
    assert!(addend_lives_somewhere_else);
    // A whole-function walk: the `sum` inst still exists somewhere, and no
    // matmul does.
    let mut has_sum = false;
    for bi in 0..func.block_count() {
        for inst in func
            .block_insts(crate::function::Block::from_u32(bi as u32))
            .to_vec()
        {
            if func.defining_inst(marker).map(|d| d == inst) == Some(true) {
                continue;
            }
            if let Opcode::Arith(Arith { .. }) = func.opcode(inst) {
                has_sum = true;
            }
        }
    }
    assert!(has_sum);
    assert_eq!(count(&func, |op| matches!(op, Opcode::Matmul(_))), 0);
    // The entry ends in an If into the nest (it had return before the rewrite,
    // now the nest handles flow); some block still carries the original ret.
    assert_eq!(
        count(&func, |op| matches!(op, Opcode::If(_))),
        3,
        "i/j/p headers each hold an if"
    );
}

#[test]
fn expand_matmul_skips_block_keyed_attributes() {
    let mut func = Function::new();
    let _matmul = matmul_fixture(&mut func, MatMulType::Fp32, false);
    func.add_attr(
        AttrTarget::Block(crate::function::Block(0)),
        custom_attr("line"),
    );
    assert!(!expand_matmul(&mut func));
}

/// Builds `fn() { matmul c, a, b [m x n x k] }` over the flat memory layout A, B, C (in that
/// order, `a_len` and `b_len` bytes long), expands it, checks it still verifies and runs it
/// on `mem`, returning the final memory.
#[allow(clippy::too_many_arguments)]
fn run_matmul(
    dtype: MatMulType,
    (m, n, k): (u16, u16, u16),
    signs: Option<InputSigns>,
    accumulate: bool,
    (a_len, b_len): (usize, usize),
    mem: Vec<u8>,
) -> (Function, Vec<u8>) {
    let mut func = Function::new();
    let ptr_t = func.types.ptr_global();
    let block = func.append_block();
    let a = func.append_inst(block, ptr_t, Opcode::Iconst(0));
    let b = func.append_inst(block, ptr_t, Opcode::Iconst(a_len as i64));
    let c = func.append_inst(block, ptr_t, Opcode::Iconst((a_len + b_len) as i64));
    func.append_matmul_embedded(block, a, b, c, m, n, k, dtype, accumulate, signs);
    func.set_terminator(block, Terminator::Ret(Ret::none()));
    assert!(expand_matmul(&mut func));
    let diags = verify(&func, Profile::High);
    assert!(diags.ok(), "{:?}", diags.items());
    let mut interp = Interp::new(mem);
    interp.run(&func, &[]);
    (func, interp.mem)
}

fn pseudo_random_bytes(len: usize, mut seed: u64) -> Vec<u8> {
    (0..len)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 24) as u8
        })
        .collect()
}

/// Runs an int8 matmul and compares C with the plain host computation, reading each operand
/// as unsigned or signed as `a_unsigned` / `b_unsigned` say.
fn check_int8(
    dtype: MatMulType,
    signs: Option<InputSigns>,
    a_unsigned: bool,
    b_unsigned: bool,
    accumulate: bool,
) {
    let (m, n, k) = (3usize, 4usize, 5usize);
    let a = pseudo_random_bytes(m * k, 0x9e37_79b9_7f4a_7c15);
    let b = pseudo_random_bytes(k * n, 0x1234_5678_9abc_def1);
    let c_init = pseudo_random_bytes(m * n * 4, 0x0fed_cba9_8765_4321);
    let mut mem = a.clone();
    mem.extend_from_slice(&b);
    mem.extend_from_slice(&c_init);
    let (func, out) = run_matmul(
        dtype,
        (m as u16, n as u16, k as u16),
        signs,
        accumulate,
        (a.len(), b.len()),
        mem,
    );
    let widen = |x: u8, unsigned: bool| {
        if unsigned {
            i32::from(x)
        } else {
            i32::from(x as i8)
        }
    };
    for i in 0..m {
        for j in 0..n {
            let at = 4 * (i * n + j);
            let mut want = if accumulate {
                i32::from_le_bytes(c_init[at..at + 4].try_into().unwrap())
            } else {
                0
            };
            for p in 0..k {
                want = want.wrapping_add(
                    widen(a[i * k + p], a_unsigned) * widen(b[p * n + j], b_unsigned),
                );
            }
            let base = a.len() + b.len() + at;
            let got = i32::from_le_bytes(out[base..base + 4].try_into().unwrap());
            assert_eq!(
                got, want,
                "{dtype:?} {signs:?} acc={accumulate} c[{i}][{j}]"
            );
        }
    }
    // The widening is the explicit part of the expansion: one `Sext` or `Zext` per operand.
    let widens = |kind: ConvertKind| {
        count(
            &func,
            |op| matches!(op, Opcode::Convert(c) if c.kind == kind),
        )
    };
    let (zexts, sexts) = (widens(ConvertKind::Zext), widens(ConvertKind::Sext));
    assert_eq!(zexts, usize::from(a_unsigned) + usize::from(b_unsigned));
    assert_eq!(sexts, usize::from(!a_unsigned) + usize::from(!b_unsigned));
}

#[test]
fn expand_matmul_int8_reads_both_operands_signed() {
    check_int8(MatMulType::Int8, None, false, false, false);
    check_int8(MatMulType::Int8, None, false, false, true);
}

#[test]
fn expand_matmul_uint8_reads_both_operands_unsigned() {
    check_int8(MatMulType::Uint8, None, true, true, false);
    check_int8(MatMulType::Uint8, None, true, true, true);
}

#[test]
fn expand_matmul_input_signs_override_the_dtype_per_operand() {
    let mixed = Some(InputSigns {
        a_unsigned: true,
        b_unsigned: false,
    });
    check_int8(MatMulType::Int8, mixed, true, false, false);
    check_int8(MatMulType::Uint8, mixed, true, false, true);
    let other = Some(InputSigns {
        a_unsigned: false,
        b_unsigned: true,
    });
    check_int8(MatMulType::Uint8, other, false, true, false);
    check_int8(MatMulType::Int8, other, false, true, true);
}

#[test]
fn expand_matmul_fp16_widens_with_fp_resize() {
    let (m, n, k) = (2usize, 3usize, 4usize);
    // Finite halves, positive and negative: 1.0, -2.5, 0.375, 100.0, -0.0625, ...
    let halves: [u16; 8] = [
        0x3c00, 0xc100, 0x3600, 0x5640, 0xac00, 0x4200, 0xbd00, 0x0400,
    ];
    let pick = |i: usize| halves[i % halves.len()];
    let a: Vec<u16> = (0..m * k).map(|i| pick(i * 3 + 1)).collect();
    let b: Vec<u16> = (0..k * n).map(|i| pick(i * 5 + 2)).collect();
    let mut mem: Vec<u8> = Vec::new();
    for h in a.iter().chain(&b) {
        mem.extend_from_slice(&h.to_le_bytes());
    }
    let (a_len, b_len) = (a.len() * 2, b.len() * 2);
    mem.extend(core::iter::repeat_n(0u8, m * n * 4));
    let (func, out) = run_matmul(
        MatMulType::Fp16,
        (m as u16, n as u16, k as u16),
        None,
        false,
        (a_len, b_len),
        mem,
    );
    for i in 0..m {
        for j in 0..n {
            let mut want = 0.0f32;
            for p in 0..k {
                want += f16_to_f32(a[i * k + p]) * f16_to_f32(b[p * n + j]);
            }
            let base = a_len + b_len + 4 * (i * n + j);
            let got = f32::from_le_bytes(out[base..base + 4].try_into().unwrap());
            assert_eq!(got.to_bits(), want.to_bits(), "c[{i}][{j}]");
        }
    }
    assert_eq!(
        count(
            &func,
            |op| matches!(op, Opcode::Convert(c) if c.kind == ConvertKind::FpResize)
        ),
        2
    );
}

#[test]
fn expand_matmul_fp32_accumulates_in_k_order() {
    let (m, n, k) = (2usize, 2usize, 3usize);
    let a = [1.5f32, -2.0, 0.25, 3.0, 0.5, -1.0];
    let b = [2.0f32, 4.0, -1.0, 0.5, 8.0, 3.0];
    let mut mem: Vec<u8> = Vec::new();
    for x in a.iter().chain(&b) {
        mem.extend_from_slice(&x.to_le_bytes());
    }
    let c_init = [10.0f32, 20.0, 30.0, 40.0];
    for x in &c_init {
        mem.extend_from_slice(&x.to_le_bytes());
    }
    let (func, out) = run_matmul(
        MatMulType::Fp32,
        (m as u16, n as u16, k as u16),
        None,
        true,
        (a.len() * 4, b.len() * 4),
        mem,
    );
    for i in 0..m {
        for j in 0..n {
            let mut want = c_init[i * n + j];
            for p in 0..k {
                want += a[i * k + p] * b[p * n + j];
            }
            let base = (a.len() + b.len()) * 4 + 4 * (i * n + j);
            let got = f32::from_le_bytes(out[base..base + 4].try_into().unwrap());
            assert_eq!(got.to_bits(), want.to_bits(), "c[{i}][{j}]");
        }
    }
    assert_eq!(count(&func, |op| matches!(op, Opcode::Convert(_))), 0);
}
