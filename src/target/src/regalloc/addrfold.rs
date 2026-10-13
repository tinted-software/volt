//! Shared, target-independent address-mode-folding analysis. Recognizes when
//! a load or store's pointer is a foldable `arith_imm.add(base, imm)` (a base
//! register plus a constant byte offset) and computes which of those adds
//! become dead once every one of their uses is folded away. Pure analysis: no
//! IR is rewritten here. Each backend later consumes this analysis to emit a
//! base+offset addressing mode directly and to drop the now-dead add.

extern crate alloc;

use alloc::vec::Vec;

use crate::regalloc::ir::{Func, visit_inst_operands, visit_term_operands};
use volt_ir::function::{BinOp, Block, Inst, Opcode, Value};

/// A folded address: a base value plus a constant byte displacement.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Addr {
    pub base: Value,
    pub off: i64,
}

/// The result of folding one function: which mem ops folded, and which adds
/// died as a result.
#[derive(Clone, Debug, Default)]
pub struct Analysis {
    /// The load/store instructions that folded, and the base+offset they fold to.
    pub folds: Vec<(Inst, Addr)>,
    /// `arith_imm.add` instructions whose result is used only by folded mem
    /// ops. They become dead once the fold is applied and the backend stops
    /// reading the add's result.
    pub dead_adds: Vec<Inst>,
    /// Dense lookup tables indexed by `Inst` index so per-instruction queries
    /// are O(1) instead of scanning `folds` / `dead_adds`.
    fold_slot: Vec<u32>,
    dead: Vec<bool>,
}

impl Analysis {
    /// A no-fold analysis: nothing folds, no add is dead. `base_of` returns
    /// the raw ptr, `off_of` returns 0, and `is_dead_add` is false for every
    /// instruction.
    pub const fn empty() -> Self {
        Self {
            folds: Vec::new(),
            dead_adds: Vec::new(),
            fold_slot: Vec::new(),
            dead: Vec::new(),
        }
    }

    fn index(folds: Vec<(Inst, Addr)>, dead_adds: Vec<Inst>) -> Self {
        let max = folds
            .iter()
            .map(|(i, _)| i.0)
            .chain(dead_adds.iter().map(|i| i.0))
            .max();
        let len = max.map_or(0, |m| m as usize + 1);
        let mut fold_slot = alloc::vec![u32::MAX; if folds.is_empty() { 0 } else { len }];
        for (n, (inst, _)) in folds.iter().enumerate() {
            // First fold wins, matching the previous linear search.
            if fold_slot[inst.0 as usize] == u32::MAX {
                fold_slot[inst.0 as usize] = n as u32;
            }
        }
        let mut dead = alloc::vec![false; if dead_adds.is_empty() { 0 } else { len }];
        for inst in &dead_adds {
            dead[inst.0 as usize] = true;
        }
        Self {
            folds,
            dead_adds,
            fold_slot,
            dead,
        }
    }

    fn fold_of(&self, mem_inst: Inst) -> Option<Addr> {
        match self.fold_slot.get(mem_inst.0 as usize) {
            Some(&n) if n != u32::MAX => Some(self.folds[n as usize].1),
            _ => None,
        }
    }

    /// The pointer VALUE the mem op addresses after folding: the fold base if
    /// folded, else the raw ptr operand.
    pub fn base_of<F: Func>(&self, func: &F, mem_inst: Inst) -> Value {
        if let Some(addr) = self.fold_of(mem_inst) {
            return addr.base;
        }
        raw_ptr(func, mem_inst)
    }

    /// The byte displacement: the fold offset if folded, else 0.
    pub fn off_of(&self, mem_inst: Inst) -> i64 {
        if let Some(addr) = self.fold_of(mem_inst) {
            return addr.off;
        }
        0
    }

    /// Whether `inst` (an `arith_imm.add`) is dead because every one of its
    /// uses folded away.
    pub fn is_dead_add(&self, inst: Inst) -> bool {
        self.dead.get(inst.0 as usize).copied().unwrap_or(false)
    }
}

/// The raw, unfolded pointer operand of a load or store. Panics on any other
/// opcode. Callers only ever pass a mem_inst that came from `folds` or a
/// load/store scan.
fn raw_ptr<F: Func>(func: &F, mem_inst: Inst) -> Value {
    match func.opcode(mem_inst) {
        Opcode::Load(l) => l.ptr,
        Opcode::Store(st) => st.ptr,
        _ => panic!("raw_ptr on non-load/store"),
    }
}

/// Build the address-fold analysis for `func`. `fold_offset` is the target
/// predicate. Given a load or store instruction whose pointer is defined by an
/// `arith_imm.add`, it returns the byte offset to fold (always equal to the
/// add's imm) if that imm is within the target's addressing range for the
/// op's access size, else `None`.
pub fn analyze<F: Func>(func: &F, fold_offset: impl Fn(&F, Inst) -> Option<i64>) -> Analysis {
    let mut folds: Vec<(Inst, Addr)> = Vec::new();

    for bi in 0..func.block_count() {
        let block = Block::from_u32(bi as u32);
        for inst in func.block_insts(block) {
            let inst = *inst;
            let ptr: Value = match func.opcode(inst) {
                Opcode::Load(l) => l.ptr,
                Opcode::Store(st) => st.ptr,
                _ => continue,
            };
            let def = match func.defining_inst(ptr) {
                Some(d) => d,
                None => continue,
            };
            let add = match func.opcode(def) {
                Opcode::ArithImm(a) => a,
                _ => continue,
            };
            if add.op != BinOp::Add {
                continue;
            }
            if let Some(off) = fold_offset(func, inst) {
                folds.push((inst, Addr { base: add.lhs, off }));
            }
        }
    }

    let value_count = func.value_count();
    let mut total_uses = alloc::vec![0u32; value_count];
    let mut folded_ptr_uses = alloc::vec![0u32; value_count];

    count_uses(func, &mut total_uses);

    for (mem_inst, _) in &folds {
        let p = raw_ptr(func, *mem_inst);
        folded_ptr_uses[p.index()] += 1;
    }

    let mut dead_adds: Vec<Inst> = Vec::new();
    for bi in 0..func.block_count() {
        let block = Block::from_u32(bi as u32);
        for inst in func.block_insts(block) {
            let inst = *inst;
            let add = match func.opcode(inst) {
                Opcode::ArithImm(a) => a,
                _ => continue,
            };
            if add.op != BinOp::Add {
                continue;
            }
            // An arith_imm always defines a result; a missing one is a
            // programmer error.
            let result = func.inst_result(inst).expect("arith_imm without result");
            let rv = result.index();
            if total_uses[rv] > 0 && total_uses[rv] == folded_ptr_uses[rv] {
                dead_adds.push(inst);
            }
        }
    }

    Analysis::index(folds, dead_adds)
}

/// Count every use of every value across the whole function: every
/// Value-carrying operand of every instruction, plus every terminator edge.
fn count_uses<F: Func>(func: &F, uses: &mut [u32]) {
    for bi in 0..func.block_count() {
        let block = Block::from_u32(bi as u32);
        for inst in func.block_insts(block) {
            visit_inst_operands(func, *inst, &mut |v, _| {
                uses[v.index()] += 1;
            });
        }
        if let Some(term) = func.terminator(block) {
            visit_term_operands(func, &term, &mut |v, _| {
                uses[v.index()] += 1;
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use volt_ir::function::{ArithImm, BinOp, Function, Opcode, Ret, Terminator};
    use volt_ir::types::{IntDesc, TypeKind};

    fn i32_ty(func: &mut Function) -> volt_ir::types::Type {
        func.types.intern(TypeKind::Int(IntDesc { bits: 32 }))
    }

    fn fold_in_range<F: Func>(func: &F, mem_inst: Inst) -> Option<i64> {
        let ptr = raw_ptr(func, mem_inst);
        let def = func.defining_inst(ptr)?;
        let add = match func.opcode(def) {
            Opcode::ArithImm(a) => a,
            _ => return None,
        };
        if add.imm % 4 != 0 || add.imm < 0 || add.imm > 100 {
            return None;
        }
        Some(add.imm)
    }

    #[test]
    fn recognizes_fold_and_dead_add() {
        let mut func = Function::new();
        let ptr_t = func.types.ptr_global();
        let i32t = i32_ty(&mut func);
        let b = func.append_block();
        let base = func.append_block_param(b, ptr_t);
        let p = func.append_inst(
            b,
            ptr_t,
            Opcode::ArithImm(ArithImm {
                op: BinOp::Add,
                lhs: base,
                imm: 8,
            }),
        );
        let load = func.append_inst(
            b,
            i32t,
            Opcode::Load(volt_ir::function::Load {
                ptr: p,
                mem: volt_ir::function::MemFlags::new(),
            }),
        );
        func.set_terminator(b, Terminator::Ret(Ret::one(load)));
        let load_inst = func.defining_inst(load).unwrap();

        let a = analyze(&func, fold_in_range);
        assert_eq!(a.folds.len(), 1);
        let (_, addr) = a.folds.iter().find(|(i, _)| *i == load_inst).unwrap();
        assert_eq!(addr.base, base);
        assert_eq!(addr.off, 8);
        assert_eq!(a.base_of(&func, load_inst), base);
        assert_eq!(a.off_of(load_inst), 8);
        let add_inst = func.defining_inst(p).unwrap();
        assert!(a.is_dead_add(add_inst));
    }

    #[test]
    fn block_param_ptr_does_not_fold() {
        let mut func = Function::new();
        let ptr_t = func.types.ptr_global();
        let i32t = i32_ty(&mut func);
        let b = func.append_block();
        let base = func.append_block_param(b, ptr_t);
        let load = func.append_inst(
            b,
            i32t,
            Opcode::Load(volt_ir::function::Load {
                ptr: base,
                mem: volt_ir::function::MemFlags::new(),
            }),
        );
        func.set_terminator(b, Terminator::Ret(Ret::one(load)));
        let load_inst = func.defining_inst(load).unwrap();
        let a = analyze(&func, fold_in_range);
        assert!(a.folds.is_empty());
        assert_eq!(a.base_of(&func, load_inst), base);
        assert_eq!(a.off_of(load_inst), 0);
    }

    #[test]
    fn out_of_range_leaves_unfolded() {
        let mut func = Function::new();
        let ptr_t = func.types.ptr_global();
        let i32t = i32_ty(&mut func);
        let b = func.append_block();
        let base = func.append_block_param(b, ptr_t);
        let p = func.append_inst(
            b,
            ptr_t,
            Opcode::ArithImm(ArithImm {
                op: BinOp::Add,
                lhs: base,
                imm: 200,
            }),
        );
        let load = func.append_inst(
            b,
            i32t,
            Opcode::Load(volt_ir::function::Load {
                ptr: p,
                mem: volt_ir::function::MemFlags::new(),
            }),
        );
        func.set_terminator(b, Terminator::Ret(Ret::one(load)));
        let load_inst = func.defining_inst(load).unwrap();
        let a = analyze(&func, fold_in_range);
        assert!(a.folds.is_empty());
        assert_eq!(a.base_of(&func, load_inst), p);
        assert_eq!(a.off_of(load_inst), 0);
    }

    #[test]
    fn add_used_elsewhere_is_not_dead() {
        let mut func = Function::new();
        let ptr_t = func.types.ptr_global();
        let i32t = i32_ty(&mut func);
        let b = func.append_block();
        let base = func.append_block_param(b, ptr_t);
        let p = func.append_inst(
            b,
            ptr_t,
            Opcode::ArithImm(ArithImm {
                op: BinOp::Add,
                lhs: base,
                imm: 8,
            }),
        );
        let load = func.append_inst(
            b,
            i32t,
            Opcode::Load(volt_ir::function::Load {
                ptr: p,
                mem: volt_ir::function::MemFlags::new(),
            }),
        );
        let _ = load;
        func.set_terminator(b, Terminator::Ret(Ret::one(p)));
        let add_inst = func.defining_inst(p).unwrap();
        let a = analyze(&func, fold_in_range);
        assert!(!a.is_dead_add(add_inst));
    }

    #[test]
    fn atomic_address_never_folds() {
        let mut func = Function::new();
        let ptr_t = func.types.ptr_global();
        let i32t = i32_ty(&mut func);
        let b = func.append_block();
        let base = func.append_block_param(b, ptr_t);
        let v = func.append_block_param(b, i32t);
        let p = func.append_inst(
            b,
            ptr_t,
            Opcode::ArithImm(ArithImm {
                op: BinOp::Add,
                lhs: base,
                imm: 8,
            }),
        );
        func.append_atomic_rmw_stmt(
            b,
            volt_ir::function::AtomicRmw {
                op: volt_ir::function::AtomicOp::Add,
                ptr: p,
                value: v,
                compare: None,
                ordering: volt_ir::function::AtomicOrdering::Relaxed,
                scope: volt_ir::function::AtomicScope::Device,
            },
        );
        func.set_terminator(b, Terminator::Ret(Ret::none()));
        let add_inst = func.defining_inst(p).unwrap();
        let a = analyze(&func, fold_in_range);
        assert!(a.folds.is_empty());
        assert!(!a.is_dead_add(add_inst));
    }

    #[test]
    fn empty_analysis_is_neutral() {
        let a = Analysis::empty();
        assert!(a.folds.is_empty());
        assert!(!a.is_dead_add(Inst::from_u32(0)));
    }
}
