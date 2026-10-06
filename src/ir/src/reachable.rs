extern crate alloc;

use super::function::{Block, Function, Opcode};

pub fn neutralize_unreachable(func: &mut Function) -> alloc::vec::Vec<bool> {
    let nblocks = func.block_count();
    let mut reachable = alloc::vec![false; nblocks];
    if nblocks > 0 {
        reachable[0] = true;
        let mut stack: alloc::vec::Vec<u32> = alloc::vec![0];
        while let Some(bi) = stack.pop() {
            let block = Block::from_u32(bi);
            for inst in func.block_insts(block).to_vec() {
                if let Opcode::If(cf) = func.opcode(inst) {
                    visit(&mut reachable, &mut stack, cf.then.target);
                    visit(&mut reachable, &mut stack, cf.else_.target);
                }
            }
            if let Some(super::function::Terminator::Jump(j)) = func.terminator(block) {
                visit(&mut reachable, &mut stack, j.target);
            }
        }
    }
    for (bi, is_reach) in reachable.iter().enumerate() {
        if *is_reach {
            continue;
        }
        let dead = Block::from_u32(bi as u32);
        func.set_block_params(dead, &[]);
        func.block_insts_mut(dead).clear();
        *func.terminator_ptr(dead) = None;
    }
    reachable
}

fn visit(reachable: &mut [bool], stack: &mut alloc::vec::Vec<u32>, target: Block) {
    let ti = target.index();
    if reachable[ti] {
        return;
    }
    reachable[ti] = true;
    stack.push(ti as u32);
}
