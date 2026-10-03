extern crate alloc;

use alloc::vec::Vec;

use super::function::{Block, Function, Inst, MatMul, Opcode, Value};

/// Reason a `matmul` cannot lower on the current target.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Unsupported {
    Dtype,
    Tile,
    Quant,
}

pub fn matmul_unsupported(mm: &MatMul) -> Option<Unsupported> {
    match mm.dtype {
        super::function::MatMulType::Fp32
        | super::function::MatMulType::Fp16
        | super::function::MatMulType::Int8
        | super::function::MatMulType::Uint8 => {}
    }
    if mm.m == 0 || mm.n == 0 || mm.k == 0 {
        return Some(Unsupported::Tile);
    }
    if mm.quant.is_some()
        && !matches!(
            mm.dtype,
            super::function::MatMulType::Int8 | super::function::MatMulType::Uint8
        )
    {
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
                        super::types::TypeKind::Vector(v) => (v.len, v.elem),
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
                        super::types::TypeKind::Vector(v) => v.len,
                        _ => continue,
                    };
                    let fields = alloc::vec![sp.scalar; len as usize];
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

// --- Deferred expansions ---------------------------------------------------
// The scalar-evaluation expansions below (low-float bit manipulation,
// NVFP4 classification, mulh limb decomposition, matmul loop nests, f32
// division) are large numeric lowering passes (~1500 lines in the Zig
// original). They are deferred to the parser/bitcode follow-up: every
// backend today rejects the unexpanded ops in its isel pre-pass, so
// returning `false` (no-op) preserves current behavior exactly.

/// Deferred: replace low-float storage conversions with integer operations.
pub fn expand_low_float(_func: &mut Function) -> bool {
    false
}

/// Deferred: replace scalar NVFP4 conversions with integer classification.
pub fn expand_nv_fp4(_func: &mut Function) -> bool {
    false
}

/// Deferred: expand the high half of a full-width product into half-width limbs.
pub fn expand_mulh(_func: &mut Function) -> bool {
    false
}

/// Deferred: rewrite a tensor-tile `matmul` into a scalar loop nest.
pub fn expand_matmul(_func: &mut Function) -> bool {
    false
}

/// Deferred: expand f32 division into a reciprocal-multiply sequence.
pub fn expand_f32_div(_func: &mut Function) -> bool {
    false
}
