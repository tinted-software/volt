extern crate alloc;

use super::function::{Block, Function, Opcode};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Side {
    Then,
    Else,
}

pub fn split_critical_edges(func: &mut Function) {
    let original_count = func.block_count();
    if original_count == 0 {
        return;
    }
    let mut pred_count = alloc::vec![0u32; original_count];
    for bi in 0..original_count {
        let block = Block::from_u32(bi as u32);
        for inst in func.block_insts(block).to_vec() {
            if let Opcode::If(cf) = func.opcode(inst) {
                pred_count[cf.then.target.index()] += 1;
                pred_count[cf.else_.target.index()] += 1;
            }
        }
        if let Some(super::function::Terminator::Jump(j)) = func.terminator(block) {
            pred_count[j.target.index()] += 1;
        }
    }
    for bi in 0..original_count {
        let block = Block::from_u32(bi as u32);
        for inst in func.block_insts(block).to_vec() {
            if func.opcode(inst).is_if() {
                split_edge(func, inst, &pred_count, Side::Then);
                split_edge(func, inst, &pred_count, Side::Else);
            }
        }
    }
}

fn split_edge(func: &mut Function, inst: super::function::Inst, pred_count: &[u32], side: Side) {
    let edge = match func.opcode(inst) {
        Opcode::If(ref cf) => match side {
            Side::Then => cf.then,
            Side::Else => cf.else_,
        },
        _ => return,
    };
    if pred_count[edge.target.index()] <= 1 {
        return;
    }
    let orig_args = func.block_args(edge).to_vec();
    let fwd = func.append_block();
    func.set_jump(fwd, edge.target, &orig_args);
    let empty = func.intern_value_list(&[]);
    if let Opcode::If(cf) = func.opcode_mut(inst) {
        match side {
            Side::Then => {
                cf.then = super::function::Jump {
                    target: fwd,
                    args: empty,
                }
            }
            Side::Else => {
                cf.else_ = super::function::Jump {
                    target: fwd,
                    args: empty,
                }
            }
        }
    }
}
