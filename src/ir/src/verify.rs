extern crate alloc;

use alloc::vec::Vec;

use super::function::{AttrTarget, Function, Opcode, Value};
use super::types::TypeKind;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Profile {
    High,
    Low,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Diagnostic {
    CompositeTypeInLowProfile(Value),
    ArgCountMismatch {
        target: super::function::Block,
        expected: u32,
        found: u32,
    },
    ArgTypeMismatch {
        target: super::function::Block,
        index: u32,
        expected: super::types::Type,
        found: super::types::Type,
    },
    NotDominated {
        value: Value,
        block: super::function::Block,
    },
    MisplacedEndian(AttrTarget),
    OperandTypeMismatch(Value),
}

#[derive(Debug, Default)]
pub struct Diagnostics {
    list: Vec<Diagnostic>,
}

impl Diagnostics {
    pub fn ok(&self) -> bool {
        self.list.is_empty()
    }

    pub fn count(&self) -> usize {
        self.list.len()
    }

    pub fn items(&self) -> &[Diagnostic] {
        &self.list
    }
}

pub fn verify(func: &Function, profile: Profile) -> Diagnostics {
    let mut diags = Diagnostics::default();
    check_edges(func, &mut diags);
    check_dominance(func, &mut diags);
    check_attributes(func, &mut diags);
    check_operand_types(func, &mut diags);
    if profile == Profile::Low {
        check_primitive_types(func, &mut diags);
    }
    diags
}

fn check_edges(func: &Function, diags: &mut Diagnostics) {
    for bi in 0..func.block_count() {
        let block = super::function::Block::from_u32(bi as u32);
        let check = |target: super::function::Block, args: &[Value], diags: &mut Diagnostics| {
            let params = func.block_params(target);
            if params.len() != args.len() {
                diags.list.push(Diagnostic::ArgCountMismatch {
                    target,
                    expected: params.len() as u32,
                    found: args.len() as u32,
                });
                return;
            }
            for (i, (p, a)) in params.iter().zip(args.iter()).enumerate() {
                if func.value_type(*p) != func.value_type(*a) {
                    diags.list.push(Diagnostic::ArgTypeMismatch {
                        target,
                        index: i as u32,
                        expected: func.value_type(*p),
                        found: func.value_type(*a),
                    });
                }
            }
        };
        for inst in func.block_insts(block).to_vec() {
            if let Opcode::If(cf) = func.opcode(inst) {
                check(cf.then.target, func.block_args(cf.then), diags);
                check(cf.else_.target, func.block_args(cf.else_), diags);
            }
        }
        if let Some(super::function::Terminator::Jump(j)) = func.terminator(block) {
            check(j.target, func.block_args(j), diags);
        }
        let _ = block;
    }
}

fn block_of_value(func: &Function, v: Value) -> Option<usize> {
    for bi in 0..func.block_count() {
        let b = super::function::Block::from_u32(bi as u32);
        if func.block_params(b).contains(&v) {
            return Some(bi);
        }
        for inst in func.block_insts(b).to_vec() {
            if func.inst_result(inst) == Some(v) {
                return Some(bi);
            }
        }
    }
    None
}

fn reachable_set(func: &Function) -> Vec<bool> {
    let n = func.block_count();
    let mut r = alloc::vec![false; n];
    if n == 0 {
        return r;
    }
    r[0] = true;
    let mut stack = alloc::vec![0usize];
    while let Some(bi) = stack.pop() {
        let b = super::function::Block::from_u32(bi as u32);
        let mut succ = Vec::new();
        for inst in func.block_insts(b).to_vec() {
            if let Opcode::If(cf) = func.opcode(inst) {
                succ.push(cf.then.target.index());
                succ.push(cf.else_.target.index());
            }
        }
        if let Some(super::function::Terminator::Jump(j)) = func.terminator(b) {
            succ.push(j.target.index());
        }
        for s in succ {
            if !r[s] {
                r[s] = true;
                stack.push(s);
            }
        }
    }
    r
}

fn check_dominance(func: &Function, diags: &mut Diagnostics) {
    // Dominance via iterative dataflow over reachable CFG.
    let n = func.block_count();
    if n == 0 {
        return;
    }
    let reach = reachable_set(func);
    let mut preds: Vec<Vec<usize>> = alloc::vec![Vec::new(); n];
    for (bi, is_reach) in reach.iter().enumerate() {
        if !*is_reach {
            continue;
        }
        let b = super::function::Block::from_u32(bi as u32);
        let mut succ = Vec::new();
        for inst in func.block_insts(b).to_vec() {
            if let Opcode::If(cf) = func.opcode(inst) {
                succ.push(cf.then.target.index());
                succ.push(cf.else_.target.index());
            }
        }
        if let Some(super::function::Terminator::Jump(j)) = func.terminator(b) {
            succ.push(j.target.index());
        }
        for s in succ {
            preds[s].push(bi);
        }
    }
    let mut dom = alloc::vec![alloc::vec![true; n]; n];
    dom[0] = alloc::vec![false; n];
    dom[0][0] = true;
    let mut changed = true;
    while changed {
        changed = false;
        for bi in 1..n {
            if !reach[bi] {
                continue;
            }
            let mut new_set = alloc::vec![true; n];
            if preds[bi].is_empty() {
                new_set = alloc::vec![false; n];
            } else {
                for (pi, p) in preds[bi].iter().enumerate() {
                    if pi == 0 {
                        new_set = dom[*p].clone();
                    } else {
                        for i in 0..n {
                            new_set[i] = new_set[i] && dom[*p][i];
                        }
                    }
                }
            }
            new_set[bi] = true;
            if new_set != dom[bi] {
                dom[bi] = new_set;
                changed = true;
            }
        }
    }
    // Every use must be dominated by its definition block.
    let uses = collect_uses(func);
    for (user_block, v) in uses {
        if let Some(def_block) = block_of_value(func, v)
            && reach[user_block]
            && reach[def_block]
            && !dom[user_block][def_block]
        {
            diags.list.push(Diagnostic::NotDominated {
                value: v,
                block: super::function::Block::from_u32(user_block as u32),
            });
        }
    }
}

fn collect_uses(func: &Function) -> Vec<(usize, Value)> {
    let mut out = Vec::new();
    for bi in 0..func.block_count() {
        let b = super::function::Block::from_u32(bi as u32);
        for inst in func.block_insts(b).to_vec() {
            for v in opcode_operands(func, inst) {
                out.push((bi, v));
            }
        }
        match func.terminator(b) {
            Some(super::function::Terminator::Ret(r)) => {
                for v in r.slice() {
                    out.push((bi, *v));
                }
            }
            Some(super::function::Terminator::Jump(j)) => {
                for v in func.block_args(j).to_vec() {
                    out.push((bi, v));
                }
            }
            None => {}
        }
    }
    out
}

fn opcode_operands(func: &Function, inst: super::function::Inst) -> Vec<Value> {
    let mut out = Vec::new();
    match func.opcode(inst) {
        Opcode::Arith(a) => out.extend_from_slice(&[a.lhs, a.rhs]),
        Opcode::ArithImm(a) => out.push(a.lhs),
        Opcode::Icmp(c) => out.extend_from_slice(&[c.lhs, c.rhs]),
        Opcode::Select(s) => out.extend_from_slice(&[s.cond, s.then, s.else_]),
        Opcode::StructNew(sn) => out.extend_from_slice(func.value_list(sn.fields)),
        Opcode::Extract(e) => out.push(e.aggregate),
        Opcode::Convert(cv) => out.push(cv.value),
        Opcode::DecodeLowFloat(cv) | Opcode::EncodeLowFloat(cv) => out.push(cv.value),
        Opcode::DequantizeNvfp4(cv) | Opcode::QuantizeNvfp4(cv) => {
            out.extend_from_slice(&[cv.value, cv.block_scale, cv.global_scale]);
        }
        Opcode::Unary(u) => out.push(u.value),
        Opcode::Load(l) => out.push(l.ptr),
        Opcode::Store(st) => out.extend_from_slice(&[st.value, st.ptr]),
        Opcode::Prefetch(pf) => out.push(pf.ptr),
        Opcode::VaStart(vs) => out.push(vs.list),
        Opcode::VaArg(va) => out.push(va.list),
        Opcode::VaEnd(ve) => out.push(ve.list),
        Opcode::Dot(d) => out.extend_from_slice(&[d.acc, d.a, d.b]),
        Opcode::Reduce(red) => out.push(red.vector),
        Opcode::Splat(sp) => out.push(sp.scalar),
        Opcode::Matmul(mm) => out.extend_from_slice(&[mm.a, mm.b, mm.c]),
        Opcode::AtomicRmw(a) => {
            out.extend_from_slice(&[a.ptr, a.value]);
            if let Some(c) = a.compare {
                out.push(c);
            }
        }
        Opcode::Call(c) => {
            if let Some(rd) = c.ret_dest {
                out.push(rd);
            }
            out.extend_from_slice(func.value_list(c.args));
        }
        Opcode::CallIndirect(c) => {
            out.push(c.target);
            if let Some(rd) = c.ret_dest {
                out.push(rd);
            }
            out.extend_from_slice(func.value_list(c.args));
        }
        Opcode::If(cf) => {
            out.push(cf.cond);
            out.extend_from_slice(func.value_list(cf.then.args));
            out.extend_from_slice(func.value_list(cf.else_.args));
        }
        Opcode::Iconst(_)
        | Opcode::Fconst(_)
        | Opcode::Fconst128(_)
        | Opcode::Alloca(_)
        | Opcode::GlobalAddr(_)
        | Opcode::Barrier(_) => {}
    }
    out
}

fn is_memory_op(op: &Opcode) -> bool {
    matches!(
        op,
        Opcode::Load(_) | Opcode::Store(_) | Opcode::Prefetch(_) | Opcode::Matmul(_)
    )
}

fn check_attributes(func: &Function, diags: &mut Diagnostics) {
    for entry in func.attribute_entries() {
        if let super::attribute::Attribute::Endian(_) = entry.attr {
            let ok = match entry.target {
                AttrTarget::Value(v) => func
                    .defining_inst(v)
                    .is_some_and(|i| is_memory_op(&func.opcode(i))),
                AttrTarget::Inst(i) => is_memory_op(&func.opcode(i)),
                AttrTarget::Func | AttrTarget::Block(_) => false,
            };
            if !ok {
                diags.list.push(Diagnostic::MisplacedEndian(entry.target));
            }
        }
    }
}

fn is_ptr(func: &Function, v: Value) -> bool {
    matches!(func.types.type_kind(func.value_type(v)), TypeKind::Ptr(_))
}

fn check_operand_types(func: &Function, diags: &mut Diagnostics) {
    for bi in 0..func.block_count() {
        let b = super::function::Block::from_u32(bi as u32);
        for inst in func.block_insts(b).to_vec() {
            let bad = match func.opcode(inst) {
                Opcode::Arith(a) => {
                    func.value_type(a.lhs) != func.value_type(a.rhs)
                        && !pointer_arith(func, a.lhs, a.rhs, a.op)
                }
                Opcode::Icmp(c) => func.value_type(c.lhs) != func.value_type(c.rhs),
                Opcode::Matmul(mm) => {
                    !(is_ptr(func, mm.a) && is_ptr(func, mm.b) && is_ptr(func, mm.c))
                }
                Opcode::AtomicRmw(a) => {
                    let mut bad = !is_ptr(func, a.ptr);
                    if !matches!(
                        func.types.type_kind(func.value_type(a.value)),
                        TypeKind::Int(_)
                    ) {
                        bad = true;
                    }
                    if let Some(r) = func.inst_result(inst)
                        && func.value_type(r) != func.value_type(a.value)
                    {
                        bad = true;
                    }
                    let wants = a.op == super::function::AtomicOp::CompareExchange;
                    if wants != a.compare.is_some() {
                        bad = true;
                    }
                    bad
                }
                Opcode::VaStart(vs) => !is_ptr(func, vs.list),
                Opcode::VaArg(va) => !is_ptr(func, va.list),
                Opcode::VaEnd(ve) => !is_ptr(func, ve.list),
                Opcode::Reduce(red) => {
                    let allowed = matches!(
                        red.op,
                        super::function::BinOp::Add
                            | super::function::BinOp::Mul
                            | super::function::BinOp::BitAnd
                            | super::function::BinOp::BitOr
                            | super::function::BinOp::BitXor
                    );
                    if !allowed {
                        true
                    } else {
                        match func.types.type_kind(func.value_type(red.vector)) {
                            TypeKind::Vector(v) => func
                                .inst_result(inst)
                                .is_none_or(|r| func.value_type(r) != v.elem),
                            _ => true,
                        }
                    }
                }
                Opcode::Splat(sp) => match func.inst_result(inst) {
                    Some(r) => match func.types.type_kind(func.value_type(r)) {
                        TypeKind::Vector(v) => func.value_type(sp.scalar) != v.elem,
                        _ => true,
                    },
                    None => true,
                },
                _ => false,
            };
            if bad {
                let v = func
                    .inst_result(inst)
                    .unwrap_or_else(|| match func.opcode(inst) {
                        Opcode::Matmul(mm) => mm.c,
                        Opcode::VaStart(vs) => vs.list,
                        Opcode::VaEnd(ve) => ve.list,
                        _ => Value::from_u32(u32::MAX),
                    });
                diags.list.push(Diagnostic::OperandTypeMismatch(v));
            }
        }
    }
}

fn pointer_arith(func: &Function, lhs: Value, rhs: Value, op: super::function::BinOp) -> bool {
    if !matches!(
        op,
        super::function::BinOp::Add | super::function::BinOp::Sub
    ) {
        return false;
    }
    let lt = func.types.type_kind(func.value_type(lhs));
    let rt = func.types.type_kind(func.value_type(rhs));
    (matches!(lt, TypeKind::Ptr(_)) && matches!(rt, TypeKind::Int(_)))
        || (matches!(rt, TypeKind::Ptr(_)) && matches!(lt, TypeKind::Int(_)))
}

fn check_primitive_types(func: &Function, diags: &mut Diagnostics) {
    for i in 0..func.value_count() {
        let v = Value::from_u32(i as u32);
        if block_of_value(func, v).is_none() {
            continue;
        }
        match func.types.type_kind(func.value_type(v)) {
            TypeKind::Struct(_) | TypeKind::Array(_) | TypeKind::Slice(_) => {
                diags.list.push(Diagnostic::CompositeTypeInLowProfile(v));
            }
            _ => {}
        }
    }
}
