extern crate alloc;

use alloc::vec::Vec;

use super::function::{BinOp, Block, Function, Opcode, Value};
use super::types::TypeKind;

pub fn legalize(func: &mut Function) {
    split_struct_params(func);
    let mut subst: Vec<(Value, Value)> = Vec::new();
    build_subst(func, &mut subst);
    apply_subst(func, &subst);
    fold_constants(func);
    fold_immediates(func);
    drop_dead(func);
}

fn split_struct_params(func: &mut Function) {
    // Split struct-typed block params into scalar field params; rewrite
    // jump/if edges by extracting fields from the struct argument.
    for bi in 0..func.block_count() {
        let block = Block::from_u32(bi as u32);
        let old: Vec<Value> = func.block_params(block).to_vec();
        let mut need = false;
        for p in &old {
            if matches!(
                func.types.type_kind(func.value_type(*p)),
                TypeKind::Struct(_)
            ) {
                need = true;
                break;
            }
        }
        if !need {
            continue;
        }
        // Record field types, create new params.
        let mut new_params: Vec<Value> = Vec::new();
        let mut mapping: Vec<(Value, Vec<Value>)> = Vec::new();
        for p in &old {
            match func.types.type_kind(func.value_type(*p)).clone() {
                TypeKind::Struct(fields) => {
                    let mut fps = Vec::new();
                    for ft in &fields {
                        fps.push(func.new_param(block, *ft));
                    }
                    new_params.extend_from_slice(&fps);
                    mapping.push((*p, fps));
                }
                _ => new_params.push(*p),
            }
        }
        func.set_block_params(block, &new_params);
        // Rewrite predecessor edges targeting this block.
        rewrite_edges_to(func, block, &old, &mapping);
    }
}

fn rewrite_edges_to(
    func: &mut Function,
    target: Block,
    old: &[Value],
    mapping: &[(Value, Vec<Value>)],
) {
    let is_split = |p: Value| {
        mapping
            .iter()
            .find(|(o, _)| *o == p)
            .map(|(_, f)| f.clone())
    };
    for bi in 0..func.block_count() {
        let src = Block::from_u32(bi as u32);
        // Jump terminators.
        if let Some(super::function::Terminator::Jump(j)) = func.terminator(src).clone()
            && j.target == target
        {
            let args = func.block_args(j).to_vec();
            let mut new_args = Vec::new();
            let mut changed = false;
            for (op_, arg) in old.iter().zip(args.iter()) {
                if let Some(flds) = is_split(*op_) {
                    changed = true;
                    for (i, fp) in flds.iter().enumerate() {
                        let e = func.append_inst(
                            src,
                            func.value_type(*fp),
                            Opcode::Extract(super::function::Extract {
                                aggregate: *arg,
                                index: i as u32,
                            }),
                        );
                        new_args.push(e);
                    }
                } else {
                    new_args.push(*arg);
                }
            }
            if changed {
                func.set_jump(src, target, &new_args);
            }
        }
        // If edges: rebuild block inst list with extracts spliced before.
        let insts = func.block_insts(src).to_vec();
        let mut new_list: Vec<super::function::Inst> = Vec::new();
        let mut changed = false;
        for inst in insts {
            let op = func.opcode(inst);
            if let Opcode::If(mut cf) = op {
                let mut extracts: Vec<super::function::Inst> = Vec::new();
                for side in 0..2 {
                    let edge = if side == 0 { cf.then } else { cf.else_ };
                    if edge.target != target {
                        continue;
                    }
                    let args = func.block_args(edge).to_vec();
                    let mut new_args = Vec::new();
                    let mut side_changed = false;
                    for (op_, arg) in old.iter().zip(args.iter()) {
                        if let Some(flds) = is_split(*op_) {
                            side_changed = true;
                            for (i, fp) in flds.iter().enumerate() {
                                let v = func.create_inst(
                                    func.value_type(*fp),
                                    Opcode::Extract(super::function::Extract {
                                        aggregate: *arg,
                                        index: i as u32,
                                    }),
                                );
                                extracts.push(func.defining_inst(v).unwrap());
                                new_args.push(v);
                            }
                        } else {
                            new_args.push(*arg);
                        }
                    }
                    if side_changed {
                        changed = true;
                        let list = func.intern_values(&new_args);
                        if side == 0 {
                            cf.then.args = list;
                        } else {
                            cf.else_.args = list;
                        }
                    }
                }
                if !extracts.is_empty() {
                    new_list.extend_from_slice(&extracts);
                    *func.opcode_mut(inst) = Opcode::If(cf);
                }
            }
            new_list.push(inst);
        }
        if changed {
            func.set_block_insts(src, &new_list);
        }
    }
}

fn resolve(subst: &[(Value, Value)], v: Value) -> Value {
    let mut cur = v;
    loop {
        match subst.iter().find(|(f, _)| *f == cur) {
            Some((_, t)) => cur = *t,
            None => return cur,
        }
    }
}

fn build_subst(func: &Function, subst: &mut Vec<(Value, Value)>) {
    for i in 0..func.inst_count() {
        let inst = super::function::Inst::from_u32(i as u32);
        if let Opcode::Extract(ex) = func.opcode(inst) {
            let agg = resolve(subst, ex.aggregate);
            let Some(agg_inst) = func.defining_inst(agg) else {
                continue;
            };
            if let Opcode::StructNew(sn) = func.opcode(agg_inst) {
                let fields = func.value_list(sn.fields).to_vec();
                if let Some(field) = fields.get(ex.index as usize)
                    && let Some(r) = func.inst_result(inst)
                {
                    subst.push((r, resolve(subst, *field)));
                }
            }
        }
    }
}

fn sub(subst: &[(Value, Value)], v: Value) -> Value {
    subst
        .iter()
        .find(|(f, _)| *f == v)
        .map(|(_, t)| *t)
        .unwrap_or(v)
}

fn apply_subst(func: &mut Function, subst: &[(Value, Value)]) {
    if subst.is_empty() {
        return;
    }
    for i in 0..func.inst_count() {
        let inst = super::function::Inst::from_u32(i as u32);
        // Read value-list runs up front (immutable), then rewrite (mutable).
        let snapshot = func.opcode(inst);
        let rewritten = match snapshot {
            Opcode::Arith(mut a) => {
                a.lhs = sub(subst, a.lhs);
                a.rhs = sub(subst, a.rhs);
                Opcode::Arith(a)
            }
            Opcode::ArithImm(mut a) => {
                a.lhs = sub(subst, a.lhs);
                Opcode::ArithImm(a)
            }
            Opcode::Icmp(mut c) => {
                c.lhs = sub(subst, c.lhs);
                c.rhs = sub(subst, c.rhs);
                Opcode::Icmp(c)
            }
            Opcode::Select(mut s) => {
                s.cond = sub(subst, s.cond);
                s.then = sub(subst, s.then);
                s.else_ = sub(subst, s.else_);
                Opcode::Select(s)
            }
            Opcode::Extract(mut e) => {
                e.aggregate = sub(subst, e.aggregate);
                Opcode::Extract(e)
            }
            Opcode::Convert(mut cv) => {
                cv.value = sub(subst, cv.value);
                Opcode::Convert(cv)
            }
            Opcode::DecodeLowFloat(mut cv) => {
                cv.value = sub(subst, cv.value);
                Opcode::DecodeLowFloat(cv)
            }
            Opcode::EncodeLowFloat(mut cv) => {
                cv.value = sub(subst, cv.value);
                Opcode::EncodeLowFloat(cv)
            }
            Opcode::DequantizeNvfp4(mut cv) => {
                cv.value = sub(subst, cv.value);
                cv.block_scale = sub(subst, cv.block_scale);
                cv.global_scale = sub(subst, cv.global_scale);
                Opcode::DequantizeNvfp4(cv)
            }
            Opcode::QuantizeNvfp4(mut cv) => {
                cv.value = sub(subst, cv.value);
                cv.block_scale = sub(subst, cv.block_scale);
                cv.global_scale = sub(subst, cv.global_scale);
                Opcode::QuantizeNvfp4(cv)
            }
            Opcode::Unary(mut u) => {
                u.value = sub(subst, u.value);
                Opcode::Unary(u)
            }
            Opcode::Load(mut l) => {
                l.ptr = sub(subst, l.ptr);
                Opcode::Load(l)
            }
            Opcode::Store(mut st) => {
                st.value = sub(subst, st.value);
                st.ptr = sub(subst, st.ptr);
                Opcode::Store(st)
            }
            Opcode::Prefetch(mut pf) => {
                pf.ptr = sub(subst, pf.ptr);
                Opcode::Prefetch(pf)
            }
            Opcode::VaStart(mut vs) => {
                vs.list = sub(subst, vs.list);
                Opcode::VaStart(vs)
            }
            Opcode::VaArg(mut va) => {
                va.list = sub(subst, va.list);
                Opcode::VaArg(va)
            }
            Opcode::VaEnd(mut ve) => {
                ve.list = sub(subst, ve.list);
                Opcode::VaEnd(ve)
            }
            Opcode::Dot(mut d) => {
                d.acc = sub(subst, d.acc);
                d.a = sub(subst, d.a);
                d.b = sub(subst, d.b);
                Opcode::Dot(d)
            }
            Opcode::Reduce(mut red) => {
                red.vector = sub(subst, red.vector);
                Opcode::Reduce(red)
            }
            Opcode::Splat(mut sp) => {
                sp.scalar = sub(subst, sp.scalar);
                Opcode::Splat(sp)
            }
            Opcode::Matmul(mut mm) => {
                mm.a = sub(subst, mm.a);
                mm.b = sub(subst, mm.b);
                mm.c = sub(subst, mm.c);
                Opcode::Matmul(mm)
            }
            Opcode::AtomicRmw(mut a) => {
                a.ptr = sub(subst, a.ptr);
                a.value = sub(subst, a.value);
                if let Some(c) = a.compare {
                    a.compare = Some(sub(subst, c));
                }
                Opcode::AtomicRmw(a)
            }
            Opcode::Call(mut c) => {
                if let Some(rd) = c.ret_dest {
                    c.ret_dest = Some(sub(subst, rd));
                }
                let vals = func.value_list(c.args).to_vec();
                let out: Vec<Value> = vals.iter().map(|v| sub(subst, *v)).collect();
                c.args = func.intern_values(&out);
                Opcode::Call(c)
            }
            Opcode::CallIndirect(mut c) => {
                c.target = sub(subst, c.target);
                if let Some(rd) = c.ret_dest {
                    c.ret_dest = Some(sub(subst, rd));
                }
                let vals = func.value_list(c.args).to_vec();
                let out: Vec<Value> = vals.iter().map(|v| sub(subst, *v)).collect();
                c.args = func.intern_values(&out);
                Opcode::CallIndirect(c)
            }
            Opcode::If(mut cf) => {
                cf.cond = sub(subst, cf.cond);
                let t = func.value_list(cf.then.args).to_vec();
                let e = func.value_list(cf.else_.args).to_vec();
                let ot: Vec<Value> = t.iter().map(|v| sub(subst, *v)).collect();
                let oe: Vec<Value> = e.iter().map(|v| sub(subst, *v)).collect();
                cf.then.args = func.intern_values(&ot);
                cf.else_.args = func.intern_values(&oe);
                Opcode::If(cf)
            }
            Opcode::StructNew(sn) => {
                let vals = func.value_list(sn.fields).to_vec();
                let out: Vec<Value> = vals.iter().map(|v| sub(subst, *v)).collect();
                Opcode::StructNew(super::function::StructNew {
                    fields: func.intern_values(&out),
                })
            }
            Opcode::Iconst(_)
            | Opcode::Fconst(_)
            | Opcode::Fconst128(_)
            | Opcode::Alloca(_)
            | Opcode::GlobalAddr(_)
            | Opcode::Barrier(_) => continue,
        };
        *func.opcode_mut(inst) = rewritten;
    }
    for bi in 0..func.block_count() {
        let b = Block::from_u32(bi as u32);
        match func.terminator(b).clone() {
            Some(super::function::Terminator::Ret(r)) => {
                let mut nr = r.clone();
                for vv in &mut nr.values[0..nr.count as usize] {
                    *vv = sub(subst, *vv);
                }
                func.set_terminator(b, super::function::Terminator::Ret(nr));
            }
            Some(super::function::Terminator::Jump(j)) => {
                let vals = func.block_args(j).to_vec();
                let out: Vec<Value> = vals.iter().map(|v| sub(subst, *v)).collect();
                func.set_jump(b, j.target, &out);
            }
            None => {}
        }
    }
}

fn const_value(func: &Function, v: Value) -> Option<i64> {
    let inst = func.defining_inst(v)?;
    match func.opcode(inst) {
        Opcode::Iconst(c) => Some(c),
        _ => None,
    }
}

fn fold_arith(op: BinOp, lhs: i64, rhs: i64) -> Option<i64> {
    match op {
        BinOp::Add => Some(lhs.wrapping_add(rhs)),
        BinOp::Sub => Some(lhs.wrapping_sub(rhs)),
        BinOp::Mul => Some(lhs.wrapping_mul(rhs)),
        BinOp::Div => {
            if rhs == 0 {
                None
            } else {
                Some(lhs.wrapping_div(rhs))
            }
        }
        BinOp::Rem => {
            if rhs == 0 {
                None
            } else {
                Some(lhs.wrapping_rem(rhs))
            }
        }
        BinOp::BitAnd => Some(lhs & rhs),
        BinOp::BitOr => Some(lhs | rhs),
        BinOp::BitXor => Some(lhs ^ rhs),
        BinOp::Shl | BinOp::Shr | BinOp::Mulh => None,
    }
}

fn fold_constants(func: &mut Function) {
    for i in 0..func.inst_count() {
        let inst = super::function::Inst::from_u32(i as u32);
        let (op, lhs, rhs) = match func.opcode(inst) {
            Opcode::Arith(a) => (a.op, a.lhs, a.rhs),
            _ => continue,
        };
        let (Some(l), Some(r)) = (const_value(func, lhs), const_value(func, rhs)) else {
            continue;
        };
        if let Some(folded) = fold_arith(op, l, r) {
            *func.opcode_mut(inst) = Opcode::Iconst(folded);
        }
    }
}

fn is_unsigned(func: &Function, v: Value) -> bool {
    matches!(func.types.type_kind(func.value_type(v)), TypeKind::Int(i) if !i.signed)
}

fn power_of_two_shift(c: i64) -> Option<i64> {
    if c < 2 {
        return None;
    }
    let u = c as u64;
    if u & (u - 1) != 0 {
        return None;
    }
    Some(u.trailing_zeros() as i64)
}

fn imm_has_form(op: BinOp) -> bool {
    matches!(
        op,
        BinOp::Add
            | BinOp::Sub
            | BinOp::BitAnd
            | BinOp::BitOr
            | BinOp::BitXor
            | BinOp::Shl
            | BinOp::Shr
    )
}

fn imm_commutative(op: BinOp) -> bool {
    matches!(
        op,
        BinOp::Add | BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor
    )
}

fn imm_fits(op: BinOp, c: i64) -> bool {
    match op {
        BinOp::Shl | BinOp::Shr => (0..=63).contains(&c),
        BinOp::Sub => (-2047..=2048).contains(&c),
        _ => (-2048..=2047).contains(&c),
    }
}

fn fold_immediates(func: &mut Function) {
    for i in 0..func.inst_count() {
        let inst = super::function::Inst::from_u32(i as u32);
        let a = match func.opcode(inst) {
            Opcode::Arith(x) => x,
            _ => continue,
        };
        if a.op == BinOp::Mul {
            if let Some(c) = const_value(func, a.rhs) {
                if let Some(k) = power_of_two_shift(c) {
                    *func.opcode_mut(inst) = Opcode::ArithImm(super::function::ArithImm {
                        op: BinOp::Shl,
                        lhs: a.lhs,
                        imm: k,
                    });
                }
            } else if let Some(c) = const_value(func, a.lhs)
                && let Some(k) = power_of_two_shift(c)
            {
                *func.opcode_mut(inst) = Opcode::ArithImm(super::function::ArithImm {
                    op: BinOp::Shl,
                    lhs: a.rhs,
                    imm: k,
                });
            }
            continue;
        }
        if (a.op == BinOp::Div || a.op == BinOp::Rem) && is_unsigned(func, a.lhs) {
            if let Some(c) = const_value(func, a.rhs)
                && let Some(k) = power_of_two_shift(c)
            {
                if a.op == BinOp::Div {
                    *func.opcode_mut(inst) = Opcode::ArithImm(super::function::ArithImm {
                        op: BinOp::Shr,
                        lhs: a.lhs,
                        imm: k,
                    });
                } else if c <= 2048 {
                    *func.opcode_mut(inst) = Opcode::ArithImm(super::function::ArithImm {
                        op: BinOp::BitAnd,
                        lhs: a.lhs,
                        imm: c - 1,
                    });
                }
            }
            continue;
        }
        if !imm_has_form(a.op) {
            continue;
        }
        if let Some(c) = const_value(func, a.rhs) {
            if imm_fits(a.op, c) {
                *func.opcode_mut(inst) = Opcode::ArithImm(super::function::ArithImm {
                    op: a.op,
                    lhs: a.lhs,
                    imm: c,
                });
            }
        } else if imm_commutative(a.op)
            && let Some(c) = const_value(func, a.lhs)
            && imm_fits(a.op, c)
        {
            *func.opcode_mut(inst) = Opcode::ArithImm(super::function::ArithImm {
                op: a.op,
                lhs: a.rhs,
                imm: c,
            });
        }
    }
}

fn is_pure(op: &Opcode) -> bool {
    matches!(
        op,
        Opcode::Iconst(_)
            | Opcode::Fconst(_)
            | Opcode::Fconst128(_)
            | Opcode::Arith(_)
            | Opcode::ArithImm(_)
            | Opcode::Icmp(_)
            | Opcode::Select(_)
            | Opcode::StructNew(_)
            | Opcode::Extract(_)
            | Opcode::Convert(_)
            | Opcode::DecodeLowFloat(_)
            | Opcode::EncodeLowFloat(_)
            | Opcode::DequantizeNvfp4(_)
            | Opcode::QuantizeNvfp4(_)
            | Opcode::Unary(_)
            | Opcode::Alloca(_)
            | Opcode::GlobalAddr(_)
            | Opcode::Dot(_)
            | Opcode::Reduce(_)
            | Opcode::Splat(_)
    )
}

fn count_uses(func: &Function, uses: &mut [u32]) {
    for v in uses.iter_mut() {
        *v = 0;
    }
    let mut use_one = |v: Value| {
        if (v.0 as usize) < uses.len() {
            uses[v.0 as usize] += 1;
        }
    };
    for bi in 0..func.block_count() {
        let b = Block::from_u32(bi as u32);
        for inst in func.block_insts(b).to_vec() {
            match func.opcode(inst) {
                Opcode::Arith(a) => {
                    use_one(a.lhs);
                    use_one(a.rhs);
                }
                Opcode::ArithImm(a) => use_one(a.lhs),
                Opcode::Icmp(c) => {
                    use_one(c.lhs);
                    use_one(c.rhs);
                }
                Opcode::Select(s) => {
                    use_one(s.cond);
                    use_one(s.then);
                    use_one(s.else_);
                }
                Opcode::StructNew(sn) => {
                    for v in func.value_list(sn.fields).to_vec() {
                        use_one(v);
                    }
                }
                Opcode::Extract(e) => use_one(e.aggregate),
                Opcode::Convert(cv) => use_one(cv.value),
                Opcode::DecodeLowFloat(cv) | Opcode::EncodeLowFloat(cv) => use_one(cv.value),
                Opcode::DequantizeNvfp4(cv) | Opcode::QuantizeNvfp4(cv) => {
                    use_one(cv.value);
                    use_one(cv.block_scale);
                    use_one(cv.global_scale);
                }
                Opcode::Unary(u) => use_one(u.value),
                Opcode::Load(l) => use_one(l.ptr),
                Opcode::Store(st) => {
                    use_one(st.value);
                    use_one(st.ptr);
                }
                Opcode::Prefetch(pf) => use_one(pf.ptr),
                Opcode::VaStart(vs) => use_one(vs.list),
                Opcode::VaArg(va) => use_one(va.list),
                Opcode::VaEnd(ve) => use_one(ve.list),
                Opcode::Dot(d) => {
                    use_one(d.acc);
                    use_one(d.a);
                    use_one(d.b);
                }
                Opcode::Reduce(red) => use_one(red.vector),
                Opcode::Splat(sp) => use_one(sp.scalar),
                Opcode::Matmul(mm) => {
                    use_one(mm.a);
                    use_one(mm.b);
                    use_one(mm.c);
                }
                Opcode::AtomicRmw(a) => {
                    use_one(a.ptr);
                    use_one(a.value);
                    if let Some(c) = a.compare {
                        use_one(c);
                    }
                }
                Opcode::Call(c) => {
                    if let Some(rd) = c.ret_dest {
                        use_one(rd);
                    }
                    for v in func.value_list(c.args).to_vec() {
                        use_one(v);
                    }
                }
                Opcode::CallIndirect(c) => {
                    use_one(c.target);
                    if let Some(rd) = c.ret_dest {
                        use_one(rd);
                    }
                    for v in func.value_list(c.args).to_vec() {
                        use_one(v);
                    }
                }
                Opcode::If(cf) => {
                    use_one(cf.cond);
                    for v in func.value_list(cf.then.args).to_vec() {
                        use_one(v);
                    }
                    for v in func.value_list(cf.else_.args).to_vec() {
                        use_one(v);
                    }
                }
                Opcode::Iconst(_)
                | Opcode::Fconst(_)
                | Opcode::Fconst128(_)
                | Opcode::Alloca(_)
                | Opcode::GlobalAddr(_)
                | Opcode::Barrier(_) => {}
            }
        }
        match func.terminator(b).clone() {
            Some(super::function::Terminator::Ret(r)) => {
                for v in r.slice() {
                    use_one(*v);
                }
            }
            Some(super::function::Terminator::Jump(j)) => {
                for v in func.block_args(j).to_vec() {
                    use_one(v);
                }
            }
            None => {}
        }
    }
}

fn drop_dead(func: &mut Function) {
    let mut uses = alloc::vec![0u32; func.value_count()];
    loop {
        count_uses(func, &mut uses);
        let mut removed = false;
        for bi in 0..func.block_count() {
            let b = Block::from_u32(bi as u32);
            let insts = func.block_insts(b).to_vec();
            let mut kept = Vec::new();
            for inst in insts {
                let dead = is_pure(&func.opcode(inst))
                    && func
                        .inst_result(inst)
                        .is_some_and(|r| uses.get(r.0 as usize).copied().unwrap_or(0) == 0);
                if dead {
                    removed = true;
                } else {
                    kept.push(inst);
                }
            }
            if kept.len() != func.block_insts(b).len() {
                func.set_block_insts(b, &kept);
            }
        }
        if !removed {
            break;
        }
    }
}
