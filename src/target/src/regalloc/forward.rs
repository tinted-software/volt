//! Single-pass allocator for forward-only, parameter-free control flow.
//!
//! The general scan in [`super::wimmer`] handles loops, block parameters,
//! calls, live-range splitting and spilling, and pays for all of that on every
//! function. A dynamic translator compiles huge numbers of tiny functions that
//! need none of it: a chain of blocks whose edges only go forward and carry no
//! arguments, over scalar integer values. This module covers exactly that
//! shape and declines (returns `None`) for anything else, so the caller can
//! fall back to the general allocator.
//!
//! Positions are numbered exactly as `build_intervals` numbers them. Because
//! every edge goes to a higher block index, every path from a definition to a
//! use stays between the two positions, so the single interval
//! `[def, last_use]` is a safe (if conservative) description of liveness and
//! no dataflow fixpoint is needed. When more values are live at once than the
//! register pool holds, the function is declined rather than spilled.

extern crate alloc;

use alloc::vec::Vec;

use super::ir::{Func, visit_inst_operands, visit_term_operands};
use super::wimmer::{Allocation, MAX_FUSED_OPERANDS, RegDescription, UsedSaved};
use volt_ir::function::{Block, Opcode, Terminator, Value};

/// Allocate integer-class registers for `func` from `pool`, in preference
/// order, or `None` if `func` is not a forward-only chain of parameter-free
/// blocks or does not fit in `pool`.
///
/// The caller owns the choice of `pool`: every register in it must be free for
/// the allocator's exclusive use (not a scratch register of the emitter and not
/// holding an incoming argument other than the single entry parameter).
pub fn allocate_forward<F: Func, D: RegDescription<F>>(
    func: &F,
    desc: &D,
    pool: &[u16],
) -> Option<Allocation> {
    if !desc.call_sites().is_empty() || !desc.entry_fixed().is_empty() || desc.classes().is_empty()
    {
        return None;
    }
    let nblocks = func.block_count();
    if nblocks == 0 || func.block_params(Block::from_u32(0)).len() > 1 {
        return None;
    }
    let scalar = |v: Value| {
        desc.class_of(func, v) == 0 && {
            let w = desc.reg_width(func, v);
            w.regs == 1 && w.alignment == 1
        }
    };

    // Pass 1: number positions, record definitions in position order, and the
    // exclusive end of each value's interval.
    let mut end = alloc::vec![0u32; func.value_count()];
    let mut defs: Vec<(u32, Value)> = Vec::with_capacity(func.value_count());
    let mut pos = 0u32;
    for bi in 0..nblocks {
        let block = Block::from_u32(bi as u32);
        let params = func.block_params(block);
        if bi != 0 && !params.is_empty() {
            return None;
        }
        for p in params {
            if !scalar(*p) {
                return None;
            }
            defs.push((pos, *p));
            end[p.index()] = pos + 1;
        }
        pos += 1;
        for inst in func.block_insts(block) {
            let inst = *inst;
            let mut usable = true;
            let mut visit = |v: Value, edge: bool| {
                if edge || desc.late_read(func, inst, v) {
                    usable = false;
                } else {
                    let e = &mut end[v.index()];
                    *e = (*e).max(pos + 1);
                }
            };
            let mut fused = [Value::from_u32(0); MAX_FUSED_OPERANDS];
            match desc.fused_operands(func, inst, &mut fused) {
                Some(n) => {
                    for v in &fused[..n as usize] {
                        visit(*v, false);
                    }
                }
                None => visit_inst_operands(func, inst, &mut visit),
            }
            if !usable {
                return None;
            }
            // Only opcodes whose emission uses no registers beyond the
            // allocated ones and the emitter's fixed scratch set.
            match func.opcode(inst) {
                Opcode::If(cf) => {
                    if cf.then.target.index() <= bi || cf.else_.target.index() <= bi {
                        return None;
                    }
                }
                Opcode::Iconst(_)
                | Opcode::Arith(_)
                | Opcode::ArithImm(_)
                | Opcode::Icmp(_)
                | Opcode::Select(_)
                | Opcode::Convert(_)
                | Opcode::Load(_)
                | Opcode::Store(_) => {}
                _ => return None,
            }
            if let Some(r) = func.inst_result(inst) {
                if !scalar(r) {
                    return None;
                }
                defs.push((pos, r));
                end[r.index()] = end[r.index()].max(pos + 1);
            }
            pos += 1;
        }
        if let Some(term) = func.terminator(block) {
            let mut usable = true;
            visit_term_operands(func, &term, &mut |v: Value, edge: bool| {
                if edge {
                    usable = false;
                } else {
                    let e = &mut end[v.index()];
                    *e = (*e).max(pos + 1);
                }
            });
            if !usable {
                return None;
            }
            if let Terminator::Jump(j) = term
                && j.target.index() <= bi
            {
                return None;
            }
        }
        pos += 1;
    }

    // Pass 2: linear scan in definition order. `busy[i]` is the exclusive end
    // of the interval currently holding `pool[i]`.
    let callee_saved = &desc.classes()[0].callee_saved;
    let mut busy = alloc::vec![0u32; pool.len()];
    let mut used = alloc::vec![false; pool.len()];
    let mut result = Allocation::default();
    let mut regs = alloc::vec![u16::MAX; func.value_count()];
    for (at, v) in defs {
        let slot = busy.iter().position(|b| *b <= at)?;
        busy[slot] = end[v.index()];
        used[slot] = true;
        regs[v.index()] = pool[slot];
    }
    result.single_regs = regs;
    result.slot_count_per_class = alloc::vec![0; desc.classes().len()];
    for (slot, reg) in pool.iter().enumerate() {
        if used[slot] && callee_saved.contains(reg) {
            result.used_callee_saved.push(UsedSaved {
                class: 0,
                reg: *reg,
            });
        }
    }
    Some(result)
}
