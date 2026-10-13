extern crate alloc;

use alloc::vec::Vec;

use super::attribute::Endianness;
use super::function::{
    AtomicOrdering, BinOp, CmpOp, ConvertKind, Function, Inst, MemFlags, Opcode, Value,
};
use super::types::{Type, TypeKind};

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
    InvalidMemFlags(super::function::Inst),
    OperandTypeMismatch(Value),
    /// The operator is not defined for its operand (or result) types: integer division on
    /// floats, a signed compare on pointers, an ill-formed conversion, and so on. Carries
    /// the offending instruction.
    InvalidOperation(Inst),
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
    check_mem_flags(func, &mut diags);
    check_operand_types(func, &mut diags);
    check_operations(func, &mut diags);
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
        Opcode::Intrinsic(i) => out.extend_from_slice(func.value_list(i.args)),
        Opcode::Iconst(_)
        | Opcode::Fconst(_)
        | Opcode::Fconst128(_)
        | Opcode::Alloca(_)
        | Opcode::GlobalAddr(_)
        | Opcode::Barrier(_) => {}
    }
    out
}

/// Checks the `MemFlags` of every load and store against the accessed type.
fn check_mem_flags(func: &Function, diags: &mut Diagnostics) {
    for bi in 0..func.block_count() {
        let b = super::function::Block::from_u32(bi as u32);
        for &inst in func.block_insts(b) {
            let (mem, ty, is_store) = match func.opcode_ref(inst) {
                Opcode::Load(l) => match func.inst_result(inst) {
                    Some(r) => (l.mem, func.value_type(r), false),
                    None => {
                        diags.list.push(Diagnostic::InvalidMemFlags(inst));
                        continue;
                    }
                },
                Opcode::Store(st) => (st.mem, func.value_type(st.value), true),
                _ => continue,
            };
            if !mem_flags_valid(func, &mem, ty, is_store) {
                diags.list.push(Diagnostic::InvalidMemFlags(inst));
            }
        }
    }
}

/// Whether `mem` is a legal flag set for a load (`is_store == false`) or store of `ty`.
pub(crate) fn mem_flags_valid(func: &Function, mem: &MemFlags, ty: Type, is_store: bool) -> bool {
    if mem.align != 0 && !mem.align.is_power_of_two() {
        return false;
    }
    let bytes = func.types.scalar_mem_bytes(ty);
    if let Some(ordering) = mem.ordering {
        let allowed = if is_store {
            matches!(
                ordering,
                AtomicOrdering::Relaxed | AtomicOrdering::Release | AtomicOrdering::SeqCst
            )
        } else {
            matches!(
                ordering,
                AtomicOrdering::Relaxed | AtomicOrdering::Acquire | AtomicOrdering::SeqCst
            )
        };
        let sized = bytes.is_some_and(|n| n.is_power_of_two() && n <= 8);
        if !allowed || !sized {
            return false;
        }
    }
    if mem.endian != Endianness::Little && mem.endian != Endianness::Native {
        // A byte swap only has a meaning on a 1, 2, 4 or 8 byte int/float/ptr scalar
        // (one byte is a no-op).
        if !bytes.is_some_and(|n| matches!(n, 1 | 2 | 4 | 8)) {
            return false;
        }
    }
    true
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

/// The kind of a scalar type, or of the lanes of a vector type.
fn lane_kind(func: &Function, ty: Type) -> &TypeKind {
    match func.types.type_kind(ty) {
        TypeKind::Vector(v) => func.types.type_kind(v.elem),
        other => other,
    }
}

/// The lane count of a vector type; `None` for a scalar.
fn lane_count(func: &Function, ty: Type) -> Option<u32> {
    match func.types.type_kind(ty) {
        TypeKind::Vector(v) => Some(v.len),
        _ => None,
    }
}

/// The width in bits of an integer (or `bool`, one bit) lane.
fn lane_bits(kind: &TypeKind) -> Option<u16> {
    match kind {
        TypeKind::Int(i) => Some(i.bits),
        TypeKind::Bool => Some(1),
        _ => None,
    }
}

fn lane_is_int(kind: &TypeKind) -> bool {
    matches!(kind, TypeKind::Int(_))
}

fn lane_is_float(kind: &TypeKind) -> bool {
    matches!(kind, TypeKind::Float(_))
}

/// Whether `op` is defined for operands whose lanes are `lane`.
fn binop_accepts(op: BinOp, lane: &TypeKind) -> bool {
    match op {
        BinOp::Add | BinOp::Sub => {
            lane_is_int(lane) || lane_is_float(lane) || matches!(lane, TypeKind::Ptr(_))
        }
        BinOp::Mul => lane_is_int(lane) || lane_is_float(lane),
        BinOp::Div | BinOp::Rem => lane_is_float(lane),
        BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor => {
            lane_is_int(lane) || matches!(lane, TypeKind::Bool)
        }
        BinOp::Shl
        | BinOp::Shr
        | BinOp::Sar
        | BinOp::UDiv
        | BinOp::SDiv
        | BinOp::URem
        | BinOp::SRem
        | BinOp::UMulh
        | BinOp::SMulh => lane_is_int(lane),
    }
}

/// Whether `op` is defined for operands whose lanes are `lane`.
fn cmpop_accepts(op: CmpOp, lane: &TypeKind) -> bool {
    match op {
        CmpOp::Eq | CmpOp::Ne => matches!(
            lane,
            TypeKind::Bool | TypeKind::Int(_) | TypeKind::Float(_) | TypeKind::Ptr(_)
        ),
        CmpOp::Lt | CmpOp::Le | CmpOp::Gt | CmpOp::Ge => lane_is_float(lane),
        CmpOp::Slt | CmpOp::Sle | CmpOp::Sgt | CmpOp::Sge => lane_is_int(lane),
        CmpOp::Ult | CmpOp::Ule | CmpOp::Ugt | CmpOp::Uge => {
            lane_is_int(lane) || matches!(lane, TypeKind::Ptr(_))
        }
    }
}

/// Whether converting `src` to `dst` is the conversion `kind` (see [`ConvertKind`]).
fn convert_ok(func: &Function, kind: ConvertKind, src: Type, dst: Type) -> bool {
    if lane_count(func, src) != lane_count(func, dst) {
        return false;
    }
    let (s, d) = (lane_kind(func, src), lane_kind(func, dst));
    match kind {
        ConvertKind::Trunc => {
            lane_is_int(s)
                && lane_is_int(d)
                && lane_bits(d).unwrap_or(0) < lane_bits(s).unwrap_or(0)
        }
        ConvertKind::Zext => {
            lane_bits(s).is_some()
                && lane_is_int(d)
                && lane_bits(d).unwrap_or(0) > lane_bits(s).unwrap_or(0)
        }
        ConvertKind::Sext => {
            lane_is_int(s)
                && lane_is_int(d)
                && lane_bits(d).unwrap_or(0) > lane_bits(s).unwrap_or(0)
        }
        ConvertKind::SiToFp | ConvertKind::UiToFp => lane_is_int(s) && lane_is_float(d),
        ConvertKind::FpToSi | ConvertKind::FpToUi => lane_is_float(s) && lane_is_int(d),
        ConvertKind::FpResize => lane_is_float(s) && lane_is_float(d) && s != d,
    }
}

/// Whether the arithmetic `op` over `lhs` (and `rhs`) producing `result` is well typed.
/// Pointer arithmetic (`ptr +/- int`, `int + ptr`) mixes types, so only the operator rule is
/// checked for it; otherwise the result has the operand type.
fn arith_ok(
    func: &Function,
    op: BinOp,
    lhs: Value,
    rhs: Option<Value>,
    result: Option<Value>,
) -> bool {
    let lt = func.value_type(lhs);
    let lane = lane_kind(func, lt);
    if !binop_accepts(op, lane) {
        return false;
    }
    let mixed = rhs.is_some_and(|r| func.value_type(r) != lt);
    if mixed || matches!(lane, TypeKind::Ptr(_)) {
        return true;
    }
    result.is_none_or(|r| func.value_type(r) == lt)
}

/// Checks that every operator is defined for the types it is applied to: the signed /
/// unsigned / floating-point operator families (`BinOp`, `CmpOp`) and the legal type pairs
/// of each `ConvertKind`.
fn check_operations(func: &Function, diags: &mut Diagnostics) {
    for bi in 0..func.block_count() {
        let b = super::function::Block::from_u32(bi as u32);
        for &inst in func.block_insts(b) {
            let result = func.inst_result(inst);
            let ok = match func.opcode_ref(inst) {
                Opcode::Arith(a) => arith_ok(func, a.op, a.lhs, Some(a.rhs), result),
                Opcode::ArithImm(a) => arith_ok(func, a.op, a.lhs, None, result),
                Opcode::Icmp(c) => {
                    let lt = func.value_type(c.lhs);
                    let scalar_result = result.is_none_or(|r| {
                        lane_count(func, lt).is_some()
                            || matches!(func.types.type_kind(func.value_type(r)), TypeKind::Bool)
                    });
                    cmpop_accepts(c.op, lane_kind(func, lt)) && scalar_result
                }
                Opcode::Convert(cv) => result.is_some_and(|r| {
                    convert_ok(func, cv.kind, func.value_type(cv.value), func.value_type(r))
                }),
                _ => true,
            };
            if !ok {
                diags.list.push(Diagnostic::InvalidOperation(inst));
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

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::function::{Arith, Compare, Ret, Terminator};
    use crate::types::{FloatKind, IntDesc, VectorDesc};

    fn mem(ordering: Option<AtomicOrdering>, endian: Endianness, align: u32) -> MemFlags {
        MemFlags {
            volatile: false,
            align,
            ordering,
            endian,
        }
    }

    fn int(bits: u16) -> TypeKind {
        TypeKind::Int(IntDesc { bits })
    }

    /// Number of `InvalidMemFlags` diagnostics for a load and for a store of `kind` with `m`.
    fn bad_counts(kind: TypeKind, m: MemFlags) -> (usize, usize) {
        let count = |is_store: bool| {
            let mut func = Function::new();
            let ty = func.types.intern(kind.clone());
            let ptr_t = func.types.ptr_global();
            let entry = func.append_block();
            let p = func.append_block_param(entry, ptr_t);
            if is_store {
                let v = func.append_block_param(entry, ty);
                func.append_store_mem(entry, v, p, m);
                func.set_terminator(entry, Terminator::Ret(Ret::none()));
            } else {
                let v = func.append_load(entry, ty, p, m);
                func.set_terminator(entry, Terminator::Ret(Ret::one(v)));
            }
            let diags = verify(&func, Profile::High);
            assert!(
                diags
                    .items()
                    .iter()
                    .all(|d| matches!(d, Diagnostic::InvalidMemFlags(_))),
                "{:?}",
                diags.items()
            );
            diags.count()
        };
        (count(false), count(true))
    }

    fn ok(kind: TypeKind, m: MemFlags) -> bool {
        bad_counts(kind, m) == (0, 0)
    }

    const NATIVE: Endianness = Endianness::Native;

    #[test]
    fn default_and_plain_flags_verify() {
        assert!(ok(int(32), MemFlags::new()));
        assert!(ok(TypeKind::Bool, MemFlags::new()));
        assert!(ok(
            TypeKind::Struct(vec![]),
            mem(None, Endianness::Little, 0)
        ));
        let volatile = MemFlags {
            volatile: true,
            ..MemFlags::new()
        };
        assert!(ok(int(8), volatile));
    }

    #[test]
    fn rejects_an_ordering_invalid_for_the_access_kind() {
        use AtomicOrdering::*;
        for o in [Relaxed, Acquire, SeqCst] {
            let (load, _) = bad_counts(int(32), mem(Some(o), NATIVE, 0));
            assert_eq!(load, 0, "load {o:?}");
        }
        for o in [Release, AcqRel] {
            let (load, _) = bad_counts(int(32), mem(Some(o), NATIVE, 0));
            assert_eq!(load, 1, "load {o:?}");
        }
        for o in [Relaxed, Release, SeqCst] {
            let (_, store) = bad_counts(int(32), mem(Some(o), NATIVE, 0));
            assert_eq!(store, 0, "store {o:?}");
        }
        for o in [Acquire, AcqRel] {
            let (_, store) = bad_counts(int(32), mem(Some(o), NATIVE, 0));
            assert_eq!(store, 1, "store {o:?}");
        }
    }

    #[test]
    fn atomic_accesses_need_a_power_of_two_scalar_of_at_most_8_bytes() {
        let seq = Some(AtomicOrdering::SeqCst);
        for bits in [8, 16, 32, 64] {
            assert!(ok(int(bits), mem(seq, NATIVE, 0)), "i{bits}");
        }
        assert!(ok(TypeKind::Float(FloatKind::F32), mem(seq, NATIVE, 0)));
        assert!(ok(TypeKind::Float(FloatKind::F64), mem(seq, NATIVE, 0)));
        assert!(ok(TypeKind::Float(FloatKind::F16), mem(seq, NATIVE, 0)));
        assert!(ok(
            TypeKind::Ptr(crate::types::AddressSpace::Global),
            mem(seq, NATIVE, 0)
        ));
        assert_eq!(bad_counts(int(128), mem(seq, NATIVE, 0)), (1, 1), "i128");
        assert_eq!(bad_counts(int(24), mem(seq, NATIVE, 0)), (1, 1), "i24");
        assert_eq!(bad_counts(int(1), mem(seq, NATIVE, 0)), (1, 1), "i1");
        assert_eq!(bad_counts(TypeKind::Bool, mem(seq, NATIVE, 0)), (1, 1));
        assert_eq!(
            bad_counts(TypeKind::Float(FloatKind::F128), mem(seq, NATIVE, 0)),
            (1, 1)
        );
        assert_eq!(
            bad_counts(TypeKind::Struct(vec![]), mem(seq, NATIVE, 0)),
            (1, 1)
        );
    }

    #[test]
    fn big_endian_needs_a_1_2_4_or_8_byte_scalar() {
        let big = Endianness::Big;
        for bits in [8, 16, 32, 64] {
            assert!(ok(int(bits), mem(None, big, 0)), "i{bits}");
        }
        assert!(ok(TypeKind::Float(FloatKind::F32), mem(None, big, 0)));
        assert!(ok(TypeKind::Float(FloatKind::F16), mem(None, big, 0)));
        assert!(ok(
            TypeKind::Ptr(crate::types::AddressSpace::Shared),
            mem(None, big, 0)
        ));
        assert_eq!(bad_counts(int(128), mem(None, big, 0)), (1, 1));
        assert_eq!(bad_counts(int(24), mem(None, big, 0)), (1, 1));
        assert_eq!(bad_counts(TypeKind::Bool, mem(None, big, 0)), (1, 1));
        assert_eq!(
            bad_counts(TypeKind::Float(FloatKind::F128), mem(None, big, 0)),
            (1, 1)
        );
        assert_eq!(
            bad_counts(TypeKind::Struct(vec![]), mem(None, big, 0)),
            (1, 1)
        );
        let mut func = Function::new();
        let elem = func.types.intern(int(32));
        let vector = TypeKind::Vector(VectorDesc { len: 4, elem });
        assert_eq!(bad_counts(vector.clone(), mem(None, big, 0)), (1, 1));
        assert!(ok(vector, mem(None, Endianness::Little, 0)));
    }

    #[test]
    fn rejects_an_align_that_is_not_a_power_of_two() {
        for align in [0, 1, 2, 16, 4096] {
            assert!(ok(int(32), mem(None, NATIVE, align)), "align {align}");
        }
        for align in [3, 6, 12, 4095] {
            assert_eq!(
                bad_counts(int(32), mem(None, NATIVE, align)),
                (1, 1),
                "align {align}"
            );
        }
    }

    #[test]
    fn a_load_with_no_result_is_rejected() {
        let mut func = Function::new();
        let ptr_t = func.types.ptr_global();
        let entry = func.append_block();
        let p = func.append_block_param(entry, ptr_t);
        func.append_stmt_raw(
            entry,
            Opcode::Load(crate::function::Load {
                ptr: p,
                mem: MemFlags::new(),
            }),
        );
        func.set_terminator(entry, Terminator::Ret(Ret::none()));
        assert_eq!(verify(&func, Profile::High).count(), 1);
    }

    #[test]
    fn intrinsic_operands_are_dominance_checked() {
        let mut func = Function::new();
        let i32_t = func.types.intern(int(32));
        let entry = func.append_block();
        let a = func.append_block_param(entry, i32_t);
        let r = func.append_intrinsic(entry, i32_t, "t.op", &[a], 1, false);
        func.append_intrinsic_void(entry, "t.sink", &[r, a], 0, true);
        func.set_terminator(entry, Terminator::Ret(Ret::one(r)));
        assert!(verify(&func, Profile::High).ok());
    }

    // --- Operator / type rules ---------------------------------------------------

    /// The type shapes the operator tables are checked against; vectors have 4 lanes.
    #[derive(Clone, Copy, Debug)]
    enum K {
        Bool,
        I(u16),
        F(FloatKind),
        Ptr,
        VI(u16),
        VF(FloatKind),
    }

    fn make(func: &mut Function, k: K) -> Type {
        let scalar = |func: &mut Function, kind: TypeKind| func.types.intern(kind);
        match k {
            K::Bool => scalar(func, TypeKind::Bool),
            K::I(bits) => scalar(func, int(bits)),
            K::F(f) => scalar(func, TypeKind::Float(f)),
            K::Ptr => func.types.ptr_global(),
            K::VI(bits) => {
                let elem = scalar(func, int(bits));
                scalar(func, TypeKind::Vector(VectorDesc { len: 4, elem }))
            }
            K::VF(f) => {
                let elem = scalar(func, TypeKind::Float(f));
                scalar(func, TypeKind::Vector(VectorDesc { len: 4, elem }))
            }
        }
    }

    /// Asserts every diagnostic is an `InvalidOperation` and returns how many there are.
    fn invalid_operations(func: &Function) -> usize {
        let diags = verify(func, Profile::High);
        assert!(
            diags
                .items()
                .iter()
                .all(|d| matches!(d, Diagnostic::InvalidOperation(_))),
            "{:?}",
            diags.items()
        );
        diags.count()
    }

    /// `fn(a: k, b: k) { ret a <op> b }`.
    fn arith_bad(k: K, op: BinOp) -> usize {
        let mut func = Function::new();
        let ty = make(&mut func, k);
        let entry = func.append_block();
        let a = func.append_block_param(entry, ty);
        let b = func.append_block_param(entry, ty);
        let r = func.append_inst(entry, ty, Opcode::Arith(Arith { op, lhs: a, rhs: b }));
        func.set_terminator(entry, Terminator::Ret(Ret::one(r)));
        invalid_operations(&func)
    }

    /// `fn(a: k) { ret a <op> 1 }`.
    fn arith_imm_bad(k: K, op: BinOp) -> usize {
        let mut func = Function::new();
        let ty = make(&mut func, k);
        let entry = func.append_block();
        let a = func.append_block_param(entry, ty);
        let r = func.append_arith_imm(entry, ty, op, a, 1);
        func.set_terminator(entry, Terminator::Ret(Ret::one(r)));
        invalid_operations(&func)
    }

    /// `fn(a: k, b: k) { ret a <op> b }` for a comparison; the result is `bool` (a `bool`
    /// vector for vector operands).
    fn cmp_bad(k: K, op: CmpOp) -> usize {
        let mut func = Function::new();
        let ty = make(&mut func, k);
        let bool_t = func.types.intern(TypeKind::Bool);
        let result_t = match k {
            K::VI(_) | K::VF(_) => func.types.intern(TypeKind::Vector(VectorDesc {
                len: 4,
                elem: bool_t,
            })),
            _ => bool_t,
        };
        let entry = func.append_block();
        let a = func.append_block_param(entry, ty);
        let b = func.append_block_param(entry, ty);
        let r = func.append_inst(
            entry,
            result_t,
            Opcode::Icmp(Compare { op, lhs: a, rhs: b }),
        );
        func.set_terminator(entry, Terminator::Ret(Ret::one(r)));
        invalid_operations(&func)
    }

    /// `fn(a: src) { ret convert <kind> dst, a }`.
    fn convert_bad(kind: ConvertKind, src: K, dst: K) -> usize {
        let mut func = Function::new();
        let (s, d) = (make(&mut func, src), make(&mut func, dst));
        let entry = func.append_block();
        let a = func.append_block_param(entry, s);
        let r = func.append_convert(entry, d, kind, a);
        func.set_terminator(entry, Terminator::Ret(Ret::one(r)));
        invalid_operations(&func)
    }

    const F32: K = K::F(FloatKind::F32);
    const F64: K = K::F(FloatKind::F64);

    #[test]
    fn arithmetic_operators_are_split_by_integer_and_float_domain() {
        use BinOp::*;
        // (operator, defined on integers, defined on floats)
        let table = [
            (Add, true, true),
            (Sub, true, true),
            (Mul, true, true),
            (Div, false, true),
            (Rem, false, true),
            (UDiv, true, false),
            (SDiv, true, false),
            (URem, true, false),
            (SRem, true, false),
            (Shl, true, false),
            (Shr, true, false),
            (Sar, true, false),
            (UMulh, true, false),
            (SMulh, true, false),
            (BitAnd, true, false),
            (BitOr, true, false),
            (BitXor, true, false),
        ];
        for (op, on_int, on_float) in table {
            for k in [K::I(8), K::I(32), K::I(64), K::VI(32)] {
                assert_eq!(arith_bad(k, op), usize::from(!on_int), "{op:?} on {k:?}");
                assert_eq!(
                    arith_imm_bad(k, op),
                    usize::from(!on_int),
                    "{op:?} imm {k:?}"
                );
            }
            for k in [F32, F64, K::F(FloatKind::F16), K::VF(FloatKind::F32)] {
                assert_eq!(arith_bad(k, op), usize::from(!on_float), "{op:?} on {k:?}");
                assert_eq!(
                    arith_imm_bad(k, op),
                    usize::from(!on_float),
                    "{op:?} imm {k:?}"
                );
            }
        }
    }

    #[test]
    fn bitwise_operators_also_accept_bool_but_nothing_else_does() {
        for op in [BinOp::BitAnd, BinOp::BitOr, BinOp::BitXor] {
            assert_eq!(arith_bad(K::Bool, op), 0, "{op:?}");
        }
        for op in [BinOp::Add, BinOp::Mul, BinOp::Shl, BinOp::UDiv, BinOp::Div] {
            assert_eq!(arith_bad(K::Bool, op), 1, "{op:?} on bool");
        }
        for op in [BinOp::Mul, BinOp::Shr, BinOp::SDiv, BinOp::BitAnd] {
            assert_eq!(arith_bad(K::Ptr, op), 1, "{op:?} on ptr");
        }
    }

    #[test]
    fn pointer_arithmetic_still_mixes_pointer_and_integer_operands() {
        let mut func = Function::new();
        let ptr_t = func.types.ptr_global();
        let i64_t = func.types.intern(int(64));
        let entry = func.append_block();
        let p = func.append_block_param(entry, ptr_t);
        let n = func.append_block_param(entry, i64_t);
        let q = func.append_inst(
            entry,
            ptr_t,
            Opcode::Arith(Arith {
                op: BinOp::Add,
                lhs: p,
                rhs: n,
            }),
        );
        let r = func.append_arith_imm(entry, ptr_t, BinOp::Sub, q, 8);
        func.set_terminator(entry, Terminator::Ret(Ret::one(r)));
        assert!(verify(&func, Profile::High).ok());
    }

    #[test]
    fn an_arithmetic_result_must_have_the_operand_type() {
        let mut func = Function::new();
        let i32_t = func.types.intern(int(32));
        let i64_t = func.types.intern(int(64));
        let entry = func.append_block();
        let a = func.append_block_param(entry, i32_t);
        let b = func.append_block_param(entry, i32_t);
        let r = func.append_inst(
            entry,
            i64_t,
            Opcode::Arith(Arith {
                op: BinOp::Add,
                lhs: a,
                rhs: b,
            }),
        );
        func.set_terminator(entry, Terminator::Ret(Ret::one(r)));
        assert_eq!(invalid_operations(&func), 1);
    }

    #[test]
    fn comparison_operators_are_split_by_domain() {
        use CmpOp::*;
        // (operator, ints, floats, bools, pointers)
        let table = [
            (Eq, true, true, true, true),
            (Ne, true, true, true, true),
            (Lt, false, true, false, false),
            (Le, false, true, false, false),
            (Gt, false, true, false, false),
            (Ge, false, true, false, false),
            (Slt, true, false, false, false),
            (Sle, true, false, false, false),
            (Sgt, true, false, false, false),
            (Sge, true, false, false, false),
            (Ult, true, false, false, true),
            (Ule, true, false, false, true),
            (Ugt, true, false, false, true),
            (Uge, true, false, false, true),
        ];
        for (op, on_int, on_float, on_bool, on_ptr) in table {
            for k in [K::I(8), K::I(32), K::I(64), K::VI(32)] {
                assert_eq!(cmp_bad(k, op), usize::from(!on_int), "{op:?} on {k:?}");
            }
            for k in [F32, F64, K::VF(FloatKind::F32)] {
                assert_eq!(cmp_bad(k, op), usize::from(!on_float), "{op:?} on {k:?}");
            }
            assert_eq!(
                cmp_bad(K::Bool, op),
                usize::from(!on_bool),
                "{op:?} on bool"
            );
            assert_eq!(cmp_bad(K::Ptr, op), usize::from(!on_ptr), "{op:?} on ptr");
        }
    }

    #[test]
    fn a_scalar_comparison_yields_bool() {
        let mut func = Function::new();
        let i32_t = func.types.intern(int(32));
        let entry = func.append_block();
        let a = func.append_block_param(entry, i32_t);
        let b = func.append_block_param(entry, i32_t);
        let r = func.append_inst(
            entry,
            i32_t,
            Opcode::Icmp(Compare {
                op: CmpOp::Ult,
                lhs: a,
                rhs: b,
            }),
        );
        func.set_terminator(entry, Terminator::Ret(Ret::one(r)));
        assert_eq!(invalid_operations(&func), 1);
    }

    #[test]
    fn conversions_accept_exactly_their_type_pairs() {
        use ConvertKind::*;
        let (i8, i16, i32, i64) = (K::I(8), K::I(16), K::I(32), K::I(64));
        let f16 = K::F(FloatKind::F16);
        let f128 = K::F(FloatKind::F128);
        let good: &[(ConvertKind, K, K)] = &[
            (Trunc, i32, i8),
            (Trunc, i64, i32),
            (Trunc, K::VI(32), K::VI(16)),
            (Zext, K::Bool, i8),
            (Zext, K::Bool, i32),
            (Zext, i8, i32),
            (Zext, i32, i64),
            (Zext, K::VI(8), K::VI(32)),
            (Sext, i8, i16),
            (Sext, i32, i64),
            (Sext, K::VI(16), K::VI(32)),
            (SiToFp, i32, F32),
            (SiToFp, i64, F64),
            (SiToFp, K::VI(32), K::VF(FloatKind::F32)),
            (UiToFp, i8, F32),
            (UiToFp, i64, f128),
            (FpToSi, F32, i32),
            (FpToSi, F64, i8),
            (FpToSi, K::VF(FloatKind::F32), K::VI(32)),
            (FpToUi, F32, i32),
            (FpToUi, f128, i64),
            (FpResize, F32, F64),
            (FpResize, F64, F32),
            (FpResize, f16, f128),
            (FpResize, K::VF(FloatKind::F32), K::VF(FloatKind::F64)),
        ];
        for &(kind, src, dst) in good {
            assert_eq!(
                convert_bad(kind, src, dst),
                0,
                "{kind:?} {src:?} -> {dst:?}"
            );
        }
        let bad: &[(ConvertKind, K, K)] = &[
            // Trunc must narrow, and between ints.
            (Trunc, i8, i32),
            (Trunc, i32, i32),
            (Trunc, F64, F32),
            (Trunc, i32, K::Bool),
            // Zext / Sext must widen; Sext does not take bool.
            (Zext, i32, i8),
            (Zext, i32, i32),
            (Zext, F32, i64),
            (Sext, i32, i8),
            (Sext, i32, i32),
            (Sext, K::Bool, i32),
            (Sext, F32, F64),
            // Int -> float and float -> int only.
            (SiToFp, F32, F64),
            (SiToFp, K::Bool, F32),
            (UiToFp, F32, i32),
            (FpToSi, i32, F32),
            (FpToUi, i32, i64),
            (FpToUi, F32, F64),
            // FpResize needs two different float types.
            (FpResize, F32, F32),
            (FpResize, i32, F32),
            (FpResize, F32, i32),
            // Lane counts must agree.
            (Zext, i8, K::VI(32)),
            (SiToFp, K::VI(32), F32),
            (FpResize, F32, K::VF(FloatKind::F64)),
        ];
        for &(kind, src, dst) in bad {
            assert_eq!(
                convert_bad(kind, src, dst),
                1,
                "{kind:?} {src:?} -> {dst:?}"
            );
        }
    }

    #[test]
    fn pointers_are_never_converted_pointer_integer_casts_are_reinterprets() {
        for kind in [
            ConvertKind::Trunc,
            ConvertKind::Zext,
            ConvertKind::Sext,
            ConvertKind::SiToFp,
            ConvertKind::UiToFp,
            ConvertKind::FpToSi,
            ConvertKind::FpToUi,
            ConvertKind::FpResize,
        ] {
            assert_eq!(convert_bad(kind, K::Ptr, K::I(64)), 1, "{kind:?}");
            assert_eq!(convert_bad(kind, K::I(32), K::Ptr), 1, "{kind:?}");
        }
    }

    #[test]
    fn a_convert_without_a_result_is_rejected() {
        let mut func = Function::new();
        let i32_t = func.types.intern(int(32));
        let entry = func.append_block();
        let a = func.append_block_param(entry, i32_t);
        func.append_stmt_raw(
            entry,
            Opcode::Convert(crate::function::Convert {
                value: a,
                kind: ConvertKind::Zext,
            }),
        );
        func.set_terminator(entry, Terminator::Ret(Ret::none()));
        assert_eq!(invalid_operations(&func), 1);
    }
}
