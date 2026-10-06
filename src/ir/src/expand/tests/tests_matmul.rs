//! Tests for `expand_matmul`.

use alloc::string::ToString;

use crate::attribute::Attribute;
use crate::expand::expand_matmul;
use crate::function::{
    Arith, AttrTarget, Function, MatMulQuant, MatMulQuantOut, MatMulScale, MatMulType, Opcode, Ret,
    Terminator,
};
use crate::types::{FloatKind, Type, TypeKind};
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
