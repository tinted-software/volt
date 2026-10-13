extern crate alloc;

use alloc::vec::Vec;

use super::attribute::Endianness;
use super::function::{
    Arith, BinOp, Block, Function, Inst, Load, MemFlags, Opcode, Store, Unary, UnaryOp, Value,
};
use super::types::{IntDesc, Type, TypeKind};
use super::verify::mem_flags_valid;

/// Lowers the IR to what the native backends select instructions for.
///
/// Besides struct-param splitting and folding, this removes every big-endian `Load`/`Store`
/// (afterwards `mem.endian` is in {Native, Little}) and every atomic-ordered float access.
/// Atomic-ordered int and pointer accesses stay as `Load`/`Store` with `mem.ordering` set: the
/// backends lower those natively. See [`legalize_memory`] for the rewrites and the
/// little-endian host assumption.
///
/// Integer constants are signless, so `-1` and `0xffff_ffff` are the same `i32`; this rewrites
/// every sub-64-bit `Iconst` to the canonical zero-extended payload (the form the backends hold
/// values in) and folds constants and immediates width-aware, with the signedness taken from
/// the operation (`UDiv` vs `SDiv`, ...), never from a type.
pub fn legalize(func: &mut Function) {
    split_struct_params(func);
    let mut subst: Vec<(Value, Value)> = Vec::new();
    build_subst(func, &mut subst);
    apply_subst(func, &subst);
    legalize_memory(func);
    canonicalize_constants(func);
    fold_constants(func);
    fold_immediates(func);
    drop_dead(func);
}

// --- Memory legalization -------------------------------------------------------

/// Rewrites big-endian loads/stores and atomic-ordered float accesses into forms the backends
/// select instructions for.
///
/// ASSUMES A LITTLE-ENDIAN HOST: `Native` and `Little` are treated as identical, and
/// `Big` is realised as a byte swap around a native access.
///
/// - Atomic int/pointer access -> unchanged: it stays a `Load`/`Store` carrying `mem.ordering`
///   (pointers are 8-byte scalars to the backends, so they pass through like a `u64`).
/// - Atomic float access -> the same-width int access (still atomic) plus
///   `Unary::Reinterpret`, since the backends only order integer-register accesses.
/// - Big endian   -> byte swap (shifts, masks, ors in the same int type) after a load /
///   before a store; composed around the (possibly atomic) access when both are requested.
///   Float and pointer accesses go through the same-width int with
///   `Unary::Reinterpret`.
///
/// `volatile`, `align` and `ordering` stay on the resulting access. Accesses the verifier
/// rejects are left untouched so backends report them.
fn legalize_memory(func: &mut Function) {
    for bi in 0..func.block_count() {
        let block = Block::from_u32(bi as u32);
        let insts = func.block_insts(block).to_vec();
        let mut out: Vec<Inst> = Vec::with_capacity(insts.len());
        let mut changed = false;
        for inst in insts {
            match func.opcode(inst) {
                Opcode::Load(l) if needs_lowering(&l.mem) => {
                    changed |= lower_load(func, &mut out, inst, l);
                }
                Opcode::Store(st) if needs_lowering(&st.mem) => {
                    changed |= lower_store(func, &mut out, inst, st);
                }
                _ => {}
            }
            out.push(inst);
        }
        if changed {
            func.set_block_insts(block, &out);
        }
    }
}

fn needs_lowering(mem: &MemFlags) -> bool {
    mem.ordering.is_some() || mem.endian == Endianness::Big
}

/// The integer type an access of `ty` is performed in (`ty` itself when it is an int).
fn access_int_type(func: &mut Function, ty: Type, bytes: u32) -> Type {
    match func.types.type_kind(ty) {
        TypeKind::Int(_) => ty,
        _ => func.types.intern(TypeKind::Int(IntDesc {
            bits: (bytes * 8) as u16,
        })),
    }
}

fn emit(func: &mut Function, out: &mut Vec<Inst>, ty: Type, op: Opcode) -> Value {
    let value = func.create_inst(ty, op);
    out.push(func.defining_inst(value).unwrap());
    value
}

/// An integer constant of `ty`, wrapped to the type's width and zero-extended (the canonical
/// payload, see [`canonicalize_constants`]).
fn int_const(func: &mut Function, out: &mut Vec<Inst>, ty: Type, value: u64) -> Value {
    let wrapped = match func.types.type_kind(ty) {
        TypeKind::Int(d) => zero_extend(value as i64, u32::from(d.bits)),
        _ => value as i64,
    };
    emit(func, out, ty, Opcode::Iconst(wrapped))
}

fn arith(
    func: &mut Function,
    out: &mut Vec<Inst>,
    ty: Type,
    op: BinOp,
    lhs: Value,
    rhs: Value,
) -> Value {
    emit(func, out, ty, Opcode::Arith(Arith { op, lhs, rhs }))
}

/// Byte `i` of `x` moved to byte position `bytes - 1 - i`.
fn swapped_byte(
    func: &mut Function,
    out: &mut Vec<Inst>,
    x: Value,
    ty: Type,
    bytes: u32,
    i: u32,
) -> Value {
    let j = bytes - 1 - i;
    if i < j {
        let mask = int_const(func, out, ty, 0xffu64 << (8 * i));
        let low = arith(func, out, ty, BinOp::BitAnd, x, mask);
        let amount = int_const(func, out, ty, u64::from(8 * (j - i)));
        arith(func, out, ty, BinOp::Shl, low, amount)
    } else {
        let amount = int_const(func, out, ty, u64::from(8 * (i - j)));
        let shifted = arith(func, out, ty, BinOp::Shr, x, amount);
        let mask = int_const(func, out, ty, 0xffu64 << (8 * j));
        arith(func, out, ty, BinOp::BitAnd, shifted, mask)
    }
}

/// Emits a byte swap of the `bytes`-wide (2, 4 or 8) int `x` of type `ty` and returns its
/// FINAL operation unemitted, so the caller can place it onto the instruction it replaces.
/// Every term is masked after a right shift, so the (logical) `Shr` is all that is needed.
fn byte_swap(func: &mut Function, out: &mut Vec<Inst>, x: Value, ty: Type, bytes: u32) -> Opcode {
    let mut acc = swapped_byte(func, out, x, ty, bytes, 0);
    for i in 1..bytes - 1 {
        let term = swapped_byte(func, out, x, ty, bytes, i);
        acc = arith(func, out, ty, BinOp::BitOr, acc, term);
    }
    let term = swapped_byte(func, out, x, ty, bytes, bytes - 1);
    Opcode::Arith(Arith {
        op: BinOp::BitOr,
        lhs: acc,
        rhs: term,
    })
}

fn reinterpret(value: Value) -> Opcode {
    Opcode::Unary(Unary {
        op: UnaryOp::Reinterpret,
        value,
    })
}

/// `mem` with the byte order lowered away (the byte swap is emitted separately).
fn native_mem(mem: MemFlags) -> MemFlags {
    MemFlags {
        endian: Endianness::Native,
        ..mem
    }
}

fn is_float(func: &Function, ty: Type) -> bool {
    matches!(func.types.type_kind(ty), TypeKind::Float(_))
}

fn lower_load(func: &mut Function, out: &mut Vec<Inst>, inst: Inst, l: Load) -> bool {
    let Some(result) = func.inst_result(inst) else {
        return false;
    };
    let ty = func.value_type(result);
    if !mem_flags_valid(func, &l.mem, ty, false) {
        return false;
    }
    let Some(bytes) = func.types.scalar_mem_bytes(ty) else {
        return false;
    };
    let big = l.mem.endian == Endianness::Big && bytes > 1;
    let atomic_float = l.mem.ordering.is_some() && is_float(func, ty);
    if !big && !atomic_float {
        // Nothing to swap or reinterpret: an ordered int/ptr load is native, a one-byte
        // big-endian one is just a native load.
        if l.mem.endian == Endianness::Big {
            *func.opcode_mut(inst) = Opcode::Load(Load {
                mem: native_mem(l.mem),
                ..l
            });
            return true;
        }
        return false;
    }
    let int_ty = access_int_type(func, ty, bytes);

    // `pending` always yields the access in `int_ty`; the last step lands on `inst`.
    let mut pending = Opcode::Load(Load {
        ptr: l.ptr,
        mem: native_mem(l.mem),
    });
    if big {
        let raw = emit(func, out, int_ty, pending);
        pending = byte_swap(func, out, raw, int_ty, bytes);
    }
    if int_ty != ty {
        let int_value = emit(func, out, int_ty, pending);
        pending = reinterpret(int_value);
    }
    *func.opcode_mut(inst) = pending;
    true
}

fn lower_store(func: &mut Function, out: &mut Vec<Inst>, inst: Inst, st: Store) -> bool {
    let ty = func.value_type(st.value);
    if !mem_flags_valid(func, &st.mem, ty, true) {
        return false;
    }
    let Some(bytes) = func.types.scalar_mem_bytes(ty) else {
        return false;
    };
    let big = st.mem.endian == Endianness::Big && bytes > 1;
    let atomic_float = st.mem.ordering.is_some() && is_float(func, ty);
    if !big && !atomic_float {
        if st.mem.endian == Endianness::Big {
            *func.opcode_mut(inst) = Opcode::Store(Store {
                mem: native_mem(st.mem),
                ..st
            });
            return true;
        }
        return false;
    }
    let int_ty = access_int_type(func, ty, bytes);

    let mut value = st.value;
    if int_ty != ty {
        value = emit(func, out, int_ty, reinterpret(value));
    }
    if big {
        let swap = byte_swap(func, out, value, int_ty, bytes);
        value = emit(func, out, int_ty, swap);
    }
    *func.opcode_mut(inst) = Opcode::Store(Store {
        value,
        ptr: st.ptr,
        mem: native_mem(st.mem),
    });
    true
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
            Opcode::Intrinsic(mut i) => {
                let vals = func.value_list(i.args).to_vec();
                let out: Vec<Value> = vals.iter().map(|v| sub(subst, *v)).collect();
                i.args = func.intern_values(&out);
                Opcode::Intrinsic(i)
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

/// `value` wrapped to `bits` and zero-extended: the canonical payload of a `bits`-wide integer.
fn zero_extend(value: i64, bits: u32) -> i64 {
    if bits == 0 || bits >= 64 {
        value
    } else {
        (value as u64 & ((1u64 << bits) - 1)) as i64
    }
}

/// The low `bits` of `value` read as a two's complement signed integer.
fn sign_extend(value: i64, bits: u32) -> i64 {
    if bits == 0 || bits >= 64 {
        value
    } else {
        (((value as u64) << (64 - bits)) as i64) >> (64 - bits)
    }
}

/// The width of a constant of `v`'s type: an int's own, a pointer's 64, a bool's 1.
fn value_bits(func: &Function, v: Value) -> Option<u32> {
    match func.types.type_kind(func.value_type(v)) {
        TypeKind::Int(d) => Some(u32::from(d.bits)),
        TypeKind::Ptr(_) => Some(64),
        TypeKind::Bool => Some(1),
        _ => None,
    }
}

/// Integers are signless, so a constant is only its low `bits`: rewrite every sub-64-bit
/// integer `Iconst` to exactly that, zero-extended. The backends hold a value narrower than 64
/// bits zero-extended and need no further case analysis for a constant.
fn canonicalize_constants(func: &mut Function) {
    for i in 0..func.inst_count() {
        let inst = super::function::Inst::from_u32(i as u32);
        let Opcode::Iconst(c) = func.opcode(inst) else {
            continue;
        };
        let Some(result) = func.inst_result(inst) else {
            continue;
        };
        if let TypeKind::Int(d) = func.types.type_kind(func.value_type(result)) {
            let canonical = zero_extend(c, u32::from(d.bits));
            if canonical != c {
                *func.opcode_mut(inst) = Opcode::Iconst(canonical);
            }
        }
    }
}

/// The value of `lhs op rhs` on `bits`-wide constants, in canonical (zero-extended) form, or
/// `None` when the operation is left to run time: shifts and high multiplies are not folded,
/// and neither is a division by zero or the overflowing `MIN / -1`, so that the backend's own
/// behavior for them is kept.
fn fold_arith(op: BinOp, lhs: i64, rhs: i64, bits: u32) -> Option<i64> {
    let (ul, ur) = (zero_extend(lhs, bits) as u64, zero_extend(rhs, bits) as u64);
    let (sl, sr) = (sign_extend(lhs, bits), sign_extend(rhs, bits));
    let min = if bits == 0 || bits >= 64 {
        i64::MIN
    } else {
        -(1i64 << (bits - 1))
    };
    let folded = match op {
        BinOp::Add => lhs.wrapping_add(rhs),
        BinOp::Sub => lhs.wrapping_sub(rhs),
        BinOp::Mul => lhs.wrapping_mul(rhs),
        BinOp::BitAnd => lhs & rhs,
        BinOp::BitOr => lhs | rhs,
        BinOp::BitXor => lhs ^ rhs,
        BinOp::UDiv => ul.checked_div(ur)? as i64,
        BinOp::URem => ul.checked_rem(ur)? as i64,
        BinOp::SDiv | BinOp::SRem if sl == min && sr == -1 => return None,
        BinOp::SDiv => sl.checked_div(sr)?,
        BinOp::SRem => sl.checked_rem(sr)?,
        // `Div` and `Rem` are floating point; the rest is not folded.
        BinOp::Div
        | BinOp::Rem
        | BinOp::Shl
        | BinOp::Shr
        | BinOp::Sar
        | BinOp::UMulh
        | BinOp::SMulh => return None,
    };
    Some(zero_extend(folded, bits))
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
        let Some(bits) = value_bits(func, lhs) else {
            continue;
        };
        if let Some(folded) = fold_arith(op, l, r, bits) {
            *func.opcode_mut(inst) = Opcode::Iconst(folded);
        }
    }
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
            | BinOp::Sar
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
        BinOp::Shl | BinOp::Shr | BinOp::Sar => (0..=63).contains(&c),
        BinOp::Sub => (-2047..=2048).contains(&c),
        _ => (-2048..=2047).contains(&c),
    }
}

/// The immediate `op` takes for the `bits`-wide constant `c`, if it has an immediate form that
/// fits. Addition and subtraction wrap, so a constant is as well written sign-extended:
/// `x + 0xffff_ffff` on an `i32` is `x + -1`.
fn imm_for(op: BinOp, c: i64, bits: u32) -> Option<i64> {
    if imm_fits(op, c) {
        return Some(c);
    }
    let signed = sign_extend(c, bits);
    (matches!(op, BinOp::Add | BinOp::Sub) && imm_fits(op, signed)).then_some(signed)
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
        if matches!(a.op, BinOp::UDiv | BinOp::URem) {
            if let Some(c) = const_value(func, a.rhs)
                && let Some(k) = power_of_two_shift(c)
            {
                if a.op == BinOp::UDiv {
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
        let bits = value_bits(func, a.lhs).unwrap_or(64);
        if let Some(c) = const_value(func, a.rhs) {
            if let Some(imm) = imm_for(a.op, c, bits) {
                *func.opcode_mut(inst) = Opcode::ArithImm(super::function::ArithImm {
                    op: a.op,
                    lhs: a.lhs,
                    imm,
                });
            }
        } else if imm_commutative(a.op)
            && let Some(c) = const_value(func, a.lhs)
            && let Some(imm) = imm_for(a.op, c, bits)
        {
            *func.opcode_mut(inst) = Opcode::ArithImm(super::function::ArithImm {
                op: a.op,
                lhs: a.rhs,
                imm,
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
    ) || matches!(op, Opcode::Intrinsic(i) if !i.side_effects)
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
                Opcode::Intrinsic(i) => {
                    for v in func.value_list(i.args).to_vec() {
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

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;
    use std::collections::HashMap;

    use super::*;
    use crate::function::{AtomicOrdering, Ret, Terminator};
    use crate::types::{AddressSpace, FloatKind};
    use crate::verify::{Profile, verify};

    /// Straight-line interpreter over one block, enough to execute legalized memory
    /// accesses. Values are raw bits masked to their type's width; the address of the
    /// pointer parameter is 0. Asserts it never sees an access `legalize` must remove.
    struct Machine {
        mem: Vec<u8>,
        values: HashMap<Value, u64>,
    }

    fn bits_of(func: &Function, v: Value) -> u32 {
        match func.types.type_kind(func.value_type(v)) {
            TypeKind::Int(d) => u32::from(d.bits),
            TypeKind::Float(FloatKind::F16) => 16,
            TypeKind::Float(FloatKind::F32) => 32,
            TypeKind::Float(FloatKind::F64) => 64,
            TypeKind::Ptr(_) => 64,
            other => panic!("unexpected type {other:?}"),
        }
    }

    fn mask(bits: u32) -> u64 {
        if bits >= 64 { !0 } else { (1u64 << bits) - 1 }
    }

    impl Machine {
        fn new(mem: &[u8]) -> Self {
            Self {
                mem: mem.to_vec(),
                values: HashMap::new(),
            }
        }

        fn read(&self, addr: u64, bytes: u32) -> u64 {
            let a = addr as usize;
            (0..bytes as usize).fold(0u64, |acc, i| acc | u64::from(self.mem[a + i]) << (8 * i))
        }

        fn write(&mut self, addr: u64, bytes: u32, value: u64) {
            for i in 0..bytes as usize {
                self.mem[addr as usize + i] = (value >> (8 * i)) as u8;
            }
        }

        fn binary(func: &Function, lhs: Value, op: BinOp, l: u64, r: u64) -> u64 {
            let bits = bits_of(func, lhs);
            let raw = match op {
                BinOp::BitAnd => l & r,
                BinOp::BitOr => l | r,
                BinOp::Shl => l << r,
                BinOp::Shr => l >> r,
                other => panic!("unexpected op {other:?}"),
            };
            raw & mask(bits)
        }

        fn run(&mut self, func: &Function, block: Block, args: &[(Value, u64)]) {
            self.values.extend(args.iter().copied());
            for &inst in func.block_insts(block) {
                let result = func.inst_result(inst);
                let out = match func.opcode(inst) {
                    Opcode::Iconst(c) => Some((c as u64) & mask(bits_of(func, result.unwrap()))),
                    Opcode::Arith(a) => {
                        let (l, r) = (self.values[&a.lhs], self.values[&a.rhs]);
                        Some(Self::binary(func, a.lhs, a.op, l, r))
                    }
                    Opcode::ArithImm(a) => {
                        let l = self.values[&a.lhs];
                        Some(Self::binary(func, a.lhs, a.op, l, a.imm as u64))
                    }
                    Opcode::Unary(u) => {
                        assert_eq!(u.op, UnaryOp::Reinterpret);
                        let bits = bits_of(func, result.unwrap());
                        assert_eq!(bits, bits_of(func, u.value));
                        Some(self.values[&u.value] & mask(bits))
                    }
                    // On a single thread an ordered access is an ordinary one.
                    Opcode::Load(l) => {
                        assert!(l.mem.endian != Endianness::Big);
                        let bytes = bits_of(func, result.unwrap()) / 8;
                        Some(self.read(self.values[&l.ptr], bytes))
                    }
                    Opcode::Store(st) => {
                        assert!(st.mem.endian != Endianness::Big);
                        let bytes = bits_of(func, st.value) / 8;
                        self.write(self.values[&st.ptr], bytes, self.values[&st.value]);
                        None
                    }
                    other => panic!("unexpected opcode {other:?}"),
                };
                if let (Some(r), Some(v)) = (result, out) {
                    self.values.insert(r, v);
                }
            }
        }
    }

    fn int_ty(func: &mut Function, bits: u16) -> Type {
        func.types.intern(TypeKind::Int(IntDesc { bits }))
    }

    fn mem(ordering: Option<AtomicOrdering>, endian: Endianness) -> MemFlags {
        MemFlags {
            ordering,
            endian,
            ..MemFlags::new()
        }
    }

    /// `fn(p: ptr) -> ty { ret load.<m> p }`; returns the function, entry block and loaded value.
    fn load_fn(ty: TypeKind, m: MemFlags) -> (Function, Block, Value, Value) {
        let mut func = Function::new();
        let ty = func.types.intern(ty);
        let ptr_t = func.types.ptr_global();
        let entry = func.append_block();
        let p = func.append_block_param(entry, ptr_t);
        let v = func.append_load(entry, ty, p, m);
        func.set_terminator(entry, Terminator::Ret(Ret::one(v)));
        (func, entry, p, v)
    }

    /// `fn(p: ptr, x: ty) { store.<m> x, p }`.
    fn store_fn(ty: TypeKind, m: MemFlags) -> (Function, Block, Value, Value) {
        let mut func = Function::new();
        let ty = func.types.intern(ty);
        let ptr_t = func.types.ptr_global();
        let entry = func.append_block();
        let p = func.append_block_param(entry, ptr_t);
        let x = func.append_block_param(entry, ty);
        func.append_store_mem(entry, x, p, m);
        func.set_terminator(entry, Terminator::Ret(Ret::none()));
        (func, entry, p, x)
    }

    fn assert_legal(func: &Function) {
        let diags = verify(func, Profile::High);
        assert!(diags.ok(), "{:?}", diags.items());
        for bi in 0..func.block_count() {
            for &inst in func.block_insts(Block::from_u32(bi as u32)) {
                match func.opcode(inst) {
                    Opcode::Load(l) => {
                        assert!(l.mem.endian != Endianness::Big);
                        if l.mem.ordering.is_some() {
                            assert!(!is_float(
                                func,
                                func.value_type(func.inst_result(inst).unwrap())
                            ));
                        }
                    }
                    Opcode::Store(st) => {
                        assert!(st.mem.endian != Endianness::Big);
                        if st.mem.ordering.is_some() {
                            assert!(!is_float(func, func.value_type(st.value)));
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    fn ops(func: &Function) -> Vec<Opcode> {
        func.block_insts(Block::from_u32(0))
            .iter()
            .map(|&i| func.opcode(i))
            .collect()
    }

    fn int(bits: u16) -> TypeKind {
        TypeKind::Int(IntDesc { bits })
    }

    #[test]
    fn big_endian_loads_become_a_native_load_and_a_byte_swap() {
        let image: [u8; 8] = [0xf1, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        for kind in [int(16), int(32), int(64)] {
            let bytes = match &kind {
                TypeKind::Int(d) => usize::from(d.bits / 8),
                _ => unreachable!(),
            };
            let (mut func, entry, p, v) = load_fn(kind.clone(), mem(None, Endianness::Big));
            legalize(&mut func);
            assert_legal(&func);
            let loads = ops(&func)
                .into_iter()
                .filter(|o| matches!(o, Opcode::Load(_)))
                .count();
            assert_eq!(loads, 1, "{kind:?}");
            let mut m = Machine::new(&image);
            m.run(&func, entry, &[(p, 0)]);
            let want = image[..bytes]
                .iter()
                .fold(0u64, |acc, &b| acc << 8 | u64::from(b));
            assert_eq!(m.values[&v], want, "{kind:?}");
        }
    }

    #[test]
    fn big_endian_float_loads_swap_in_an_int_and_reinterpret() {
        let (mut func, entry, p, v) =
            load_fn(TypeKind::Float(FloatKind::F32), mem(None, Endianness::Big));
        legalize(&mut func);
        assert_legal(&func);
        assert!(matches!(
            ops(&func).last(),
            Some(Opcode::Unary(u)) if u.op == UnaryOp::Reinterpret
        ));
        let mut m = Machine::new(&1.5f32.to_bits().to_be_bytes());
        m.run(&func, entry, &[(p, 0)]);
        assert_eq!(m.values[&v], u64::from(1.5f32.to_bits()));
    }

    #[test]
    fn big_endian_stores_swap_before_a_native_store() {
        for (kind, value, want) in [
            (int(16), 0x0102u64, vec![0x01u8, 0x02]),
            (int(32), 0x0102_0304, vec![1, 2, 3, 4]),
            (int(64), 0x0102_0304_0506_0708, vec![1, 2, 3, 4, 5, 6, 7, 8]),
        ] {
            let (mut func, entry, p, x) = store_fn(kind, mem(None, Endianness::Big));
            legalize(&mut func);
            assert_legal(&func);
            let mut m = Machine::new(&[0xee; 8]);
            m.run(&func, entry, &[(p, 0), (x, value)]);
            assert_eq!(&m.mem[..want.len()], &want[..]);
            assert!(m.mem[want.len()..].iter().all(|&b| b == 0xee));
        }
        let (mut func, entry, p, x) =
            store_fn(TypeKind::Float(FloatKind::F32), mem(None, Endianness::Big));
        legalize(&mut func);
        assert_legal(&func);
        let mut m = Machine::new(&[0; 4]);
        m.run(&func, entry, &[(p, 0), (x, u64::from(2.5f32.to_bits()))]);
        assert_eq!(m.mem, 2.5f32.to_bits().to_be_bytes());
    }

    #[test]
    fn volatile_and_align_stay_on_the_plain_access() {
        let flags = MemFlags {
            volatile: true,
            align: 4,
            ordering: None,
            endian: Endianness::Big,
        };
        let (mut func, ..) = load_fn(int(32), flags);
        legalize(&mut func);
        let plain: Vec<MemFlags> = ops(&func)
            .into_iter()
            .filter_map(|o| match o {
                Opcode::Load(l) => Some(l.mem),
                _ => None,
            })
            .collect();
        assert_eq!(
            plain,
            [MemFlags {
                volatile: true,
                align: 4,
                ordering: None,
                endian: Endianness::Native,
            }]
        );
        let (mut func, ..) = store_fn(int(32), flags);
        legalize(&mut func);
        assert!(ops(&func).iter().any(|o| matches!(
            o,
            Opcode::Store(st) if st.mem.volatile && st.mem.align == 4
                && st.mem.endian == Endianness::Native
        )));
    }

    #[test]
    fn one_byte_big_endian_needs_no_swap() {
        let (mut func, entry, p, v) = load_fn(int(8), mem(None, Endianness::Big));
        legalize(&mut func);
        assert_legal(&func);
        assert_eq!(ops(&func).len(), 1);
        let mut m = Machine::new(&[0x7f]);
        m.run(&func, entry, &[(p, 0)]);
        assert_eq!(m.values[&v], 0x7f);
    }

    #[test]
    fn native_and_little_accesses_are_left_alone() {
        for endian in [Endianness::Native, Endianness::Little] {
            let flags = MemFlags {
                volatile: true,
                align: 8,
                endian,
                ..MemFlags::new()
            };
            let (mut func, ..) = load_fn(int(64), flags);
            let before = ops(&func);
            legalize(&mut func);
            assert_eq!(ops(&func), before);
        }
    }

    #[test]
    fn atomic_int_loads_stay_ordered_loads() {
        for ordering in [
            AtomicOrdering::Relaxed,
            AtomicOrdering::Acquire,
            AtomicOrdering::SeqCst,
        ] {
            for kind in [int(8), int(16), int(32), int(64)] {
                let flags = mem(Some(ordering), Endianness::Native);
                let (mut func, entry, p, v) = load_fn(kind.clone(), flags);
                let before = ops(&func);
                legalize(&mut func);
                assert_legal(&func);
                assert_eq!(ops(&func), before, "{kind:?} {ordering:?}");
                assert!(matches!(
                    ops(&func)[..],
                    [Opcode::Load(l)] if l.mem.ordering == Some(ordering)
                ));
                let mut m = Machine::new(&[0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]);
                m.run(&func, entry, &[(p, 0)]);
                let bytes = match &kind {
                    TypeKind::Int(d) => u32::from(d.bits / 8),
                    _ => unreachable!(),
                };
                assert_eq!(m.values[&v], m.read(0, bytes), "{kind:?}");
                assert_eq!(
                    &m.mem[..],
                    &[0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]
                );
            }
        }
    }

    #[test]
    fn atomic_float_loads_go_through_an_atomic_int_load() {
        let seq = Some(AtomicOrdering::SeqCst);
        let (mut func, entry, p, v) = load_fn(
            TypeKind::Float(FloatKind::F64),
            mem(seq, Endianness::Native),
        );
        legalize(&mut func);
        assert_legal(&func);
        let ops = ops(&func);
        assert_eq!(ops.len(), 2);
        assert!(matches!(
            ops[0],
            Opcode::Load(l) if l.mem.ordering == seq
        ));
        assert!(matches!(ops[1], Opcode::Unary(u) if u.op == UnaryOp::Reinterpret));
        let mut m = Machine::new(&(-4.25f64).to_bits().to_le_bytes());
        m.run(&func, entry, &[(p, 0)]);
        assert_eq!(m.values[&v], (-4.25f64).to_bits());
    }

    #[test]
    fn atomic_pointer_accesses_pass_through_unchanged() {
        let seq = Some(AtomicOrdering::SeqCst);
        let (mut func, entry, p, v) = load_fn(
            TypeKind::Ptr(AddressSpace::Global),
            mem(seq, Endianness::Native),
        );
        let before = ops(&func);
        legalize(&mut func);
        assert_legal(&func);
        assert_eq!(ops(&func), before);
        let mut m = Machine::new(&0x0123_4567_89ab_cdefu64.to_le_bytes());
        m.run(&func, entry, &[(p, 0)]);
        assert_eq!(m.values[&v], 0x0123_4567_89ab_cdef);

        let (mut func, ..) = store_fn(
            TypeKind::Ptr(AddressSpace::Global),
            mem(Some(AtomicOrdering::Release), Endianness::Native),
        );
        let before = ops(&func);
        legalize(&mut func);
        assert_eq!(ops(&func), before);
    }

    #[test]
    fn atomic_int_stores_stay_ordered_stores() {
        for ordering in [
            AtomicOrdering::Relaxed,
            AtomicOrdering::Release,
            AtomicOrdering::SeqCst,
        ] {
            let (mut func, entry, p, x) =
                store_fn(int(32), mem(Some(ordering), Endianness::Native));
            let before = ops(&func);
            legalize(&mut func);
            assert_legal(&func);
            assert_eq!(ops(&func), before);
            assert!(matches!(
                ops(&func)[..],
                [Opcode::Store(s)] if s.mem.ordering == Some(ordering) && s.ptr == p && s.value == x
            ));
            let mut m = Machine::new(&[0; 4]);
            m.run(&func, entry, &[(p, 0), (x, 0xdead_beef)]);
            assert_eq!(m.mem, 0xdead_beefu32.to_le_bytes());
        }
    }

    #[test]
    fn atomic_float_stores_reinterpret_into_an_atomic_int_store() {
        let release = Some(AtomicOrdering::Release);
        let (mut func, entry, p, x) = store_fn(
            TypeKind::Float(FloatKind::F32),
            mem(release, Endianness::Native),
        );
        legalize(&mut func);
        assert_legal(&func);
        let ops = ops(&func);
        assert_eq!(ops.len(), 2);
        assert!(matches!(ops[0], Opcode::Unary(u) if u.op == UnaryOp::Reinterpret));
        assert!(matches!(
            ops[1],
            Opcode::Store(s) if s.mem.ordering == release
        ));
        let mut m = Machine::new(&[0; 4]);
        m.run(&func, entry, &[(p, 0), (x, u64::from(0.5f32.to_bits()))]);
        assert_eq!(m.mem, 0.5f32.to_bits().to_le_bytes());
    }

    #[test]
    fn atomic_and_big_endian_compose() {
        let flags = mem(Some(AtomicOrdering::Acquire), Endianness::Big);
        let (mut func, entry, p, v) = load_fn(int(32), flags);
        legalize(&mut func);
        assert_legal(&func);
        let mut m = Machine::new(&[0xde, 0xad, 0xbe, 0xef]);
        m.run(&func, entry, &[(p, 0)]);
        assert_eq!(m.values[&v], 0xdead_beef);

        let flags = mem(Some(AtomicOrdering::Release), Endianness::Big);
        let (mut func, entry, p, x) = store_fn(int(32), flags);
        legalize(&mut func);
        assert_legal(&func);
        assert!(ops(&func).iter().any(|o| matches!(
            o,
            Opcode::Store(s) if s.mem.ordering == Some(AtomicOrdering::Release)
                && s.mem.endian == Endianness::Native
        )));
        let mut m = Machine::new(&[0; 4]);
        m.run(&func, entry, &[(p, 0), (x, 0xdead_beef)]);
        assert_eq!(m.mem, [0xde, 0xad, 0xbe, 0xef]);

        let flags = mem(Some(AtomicOrdering::SeqCst), Endianness::Big);
        let (mut func, entry, p, v) = load_fn(TypeKind::Float(FloatKind::F32), flags);
        legalize(&mut func);
        assert_legal(&func);
        let mut m = Machine::new(&1.25f32.to_bits().to_be_bytes());
        m.run(&func, entry, &[(p, 0)]);
        assert_eq!(m.values[&v], u64::from(1.25f32.to_bits()));
    }

    #[test]
    fn every_valid_flag_combination_legalizes_to_verifiable_plain_ops() {
        use AtomicOrdering::*;
        let mut func = Function::new();
        let i32_t = int_ty(&mut func, 32);
        let ptr_t = func.types.ptr_global();
        let entry = func.append_block();
        let p = func.append_block_param(entry, ptr_t);
        let mut last = None;
        for volatile in [false, true] {
            for align in [0, 4] {
                for endian in [Endianness::Native, Endianness::Little, Endianness::Big] {
                    for (load_o, store_o) in [
                        (None, None),
                        (Some(Relaxed), Some(Relaxed)),
                        (Some(Acquire), Some(Release)),
                        (Some(SeqCst), Some(SeqCst)),
                    ] {
                        let l = MemFlags {
                            volatile,
                            align,
                            ordering: load_o,
                            endian,
                        };
                        let s = MemFlags {
                            volatile,
                            align,
                            ordering: store_o,
                            endian,
                        };
                        let v = func.append_load(entry, i32_t, p, l);
                        func.append_store_mem(entry, v, p, s);
                        last = Some(v);
                    }
                }
            }
        }
        func.set_terminator(entry, Terminator::Ret(Ret::one(last.unwrap())));
        assert!(verify(&func, Profile::High).ok());
        legalize(&mut func);
        assert_legal(&func);
    }

    #[test]
    fn accesses_the_verifier_rejects_are_left_untouched() {
        // A release load: there is no sound lowering, so a backend must see it and refuse.
        let (mut func, ..) = load_fn(
            int(32),
            mem(Some(AtomicOrdering::Release), Endianness::Native),
        );
        let before = ops(&func);
        legalize(&mut func);
        assert_eq!(ops(&func), before);
        // A big-endian aggregate.
        let (mut func, ..) = load_fn(TypeKind::Struct(vec![]), mem(None, Endianness::Big));
        let before = ops(&func);
        legalize(&mut func);
        assert_eq!(ops(&func), before);
    }

    #[test]
    fn pure_unused_intrinsics_are_dropped_and_effectful_ones_kept() {
        let mut func = Function::new();
        let i32_t = int_ty(&mut func, 32);
        let entry = func.append_block();
        let a = func.append_block_param(entry, i32_t);
        let _dead = func.append_intrinsic(entry, i32_t, "t.pure", &[a], 0, false);
        let _kept = func.append_intrinsic(entry, i32_t, "t.effect", &[a], 0, true);
        func.set_terminator(entry, Terminator::Ret(Ret::one(a)));
        legalize(&mut func);
        let names: Vec<&str> = ops(&func)
            .iter()
            .map(|o| match o {
                Opcode::Intrinsic(i) => func.symbol_name(i.symbol),
                _ => panic!("unexpected op"),
            })
            .collect();
        assert_eq!(names, ["t.effect"]);
    }

    /// The defining op of the value `fn() { ret <lhs op rhs> }` returns after `legalize`, over
    /// constants of `bits` width.
    fn folded(bits: u16, op: BinOp, lhs: i64, rhs: i64) -> Opcode {
        let mut func = Function::new();
        let ty = int_ty(&mut func, bits);
        let entry = func.append_block();
        let a = func.append_inst(entry, ty, Opcode::Iconst(lhs));
        let b = func.append_inst(entry, ty, Opcode::Iconst(rhs));
        let v = func.append_inst(entry, ty, Opcode::Arith(Arith { op, lhs: a, rhs: b }));
        func.set_terminator(entry, Terminator::Ret(Ret::one(v)));
        legalize(&mut func);
        func.opcode(func.defining_inst(v).unwrap())
    }

    /// The defining op of the value `fn(x: i32) { ret x <op> c }` returns after `legalize`.
    fn with_constant(op: BinOp, c: i64) -> Opcode {
        let mut func = Function::new();
        let ty = int_ty(&mut func, 32);
        let entry = func.append_block();
        let x = func.append_block_param(entry, ty);
        let k = func.append_inst(entry, ty, Opcode::Iconst(c));
        let v = func.append_inst(entry, ty, Opcode::Arith(Arith { op, lhs: x, rhs: k }));
        func.set_terminator(entry, Terminator::Ret(Ret::one(v)));
        legalize(&mut func);
        func.opcode(func.defining_inst(v).unwrap())
    }

    #[test]
    fn integer_constants_become_their_zero_extended_bits() {
        let mut func = Function::new();
        let (i8_t, i32_t, i64_t) = (
            int_ty(&mut func, 8),
            int_ty(&mut func, 32),
            int_ty(&mut func, 64),
        );
        let entry = func.append_block();
        let minus_one = func.append_inst(entry, i32_t, Opcode::Iconst(-1));
        let wide = func.append_inst(entry, i8_t, Opcode::Iconst(0x1_0000_00ff));
        let all_ones = func.append_inst(entry, i64_t, Opcode::Iconst(-1));
        let min = func.append_inst(entry, i8_t, Opcode::Iconst(-128));
        func.set_terminator(
            entry,
            Terminator::Ret(Ret::many(&[minus_one, wide, all_ones, min])),
        );
        legalize(&mut func);
        let payload = |v: Value| match func.opcode(func.defining_inst(v).unwrap()) {
            Opcode::Iconst(c) => c,
            other => panic!("expected a constant, got {other:?}"),
        };
        assert_eq!(payload(minus_one), 0xffff_ffff);
        assert_eq!(payload(wide), 0xff);
        assert_eq!(payload(all_ones), -1);
        assert_eq!(payload(min), 0x80);
    }

    #[test]
    fn folding_reads_the_operands_as_the_operation_says() {
        use BinOp::*;
        // (bits, operator, lhs, rhs, the canonical zero-extended result)
        let cases: &[(u16, BinOp, i64, i64, i64)] = &[
            (32, UDiv, -1, 2, 0x7fff_ffff),
            (32, SDiv, -1, 2, 0),
            (32, SDiv, -6, 2, 0xffff_fffd),
            (32, URem, -1, 10, 5),
            (32, SRem, -7, 3, 0xffff_ffff),
            (8, UDiv, 200, 3, 66),
            (8, SDiv, 200, 3, 0xee),
            (8, SRem, 200, 3, 0xfe),
            (8, Add, 200, 100, 44),
            (8, Sub, 1, 2, 0xff),
            (8, Mul, 16, 16, 0),
            (16, BitXor, 0xffff, 0x00ff, 0xff00),
            (64, UDiv, -1, 2, i64::MAX),
            (64, SDiv, -1, 2, 0),
            (64, URem, -1, 10, 5),
        ];
        for &(bits, op, lhs, rhs, want) in cases {
            match folded(bits, op, lhs, rhs) {
                Opcode::Iconst(got) => {
                    assert_eq!(got, want, "i{bits} {op:?} {lhs} {rhs}");
                }
                other => panic!("i{bits} {op:?} {lhs} {rhs} did not fold: {other:?}"),
            }
        }
    }

    #[test]
    fn folding_leaves_what_the_backend_defines_to_run_time() {
        use BinOp::*;
        let cases: &[(u16, BinOp, i64, i64)] = &[
            (32, UDiv, 5, 0),
            (32, URem, 5, 0),
            (32, SDiv, 5, 0),
            (32, SRem, 5, 0),
            (32, SDiv, i64::from(i32::MIN), -1),
            (32, SRem, i64::from(i32::MIN), -1),
            (8, SDiv, 0x80, 0xff),
            (64, SDiv, i64::MIN, -1),
            (64, SRem, i64::MIN, -1),
            (32, UMulh, 3, 5),
            (32, SMulh, 3, 5),
        ];
        for &(bits, op, lhs, rhs) in cases {
            assert!(
                matches!(folded(bits, op, lhs, rhs), Opcode::Arith(a) if a.op == op),
                "i{bits} {op:?} {lhs} {rhs}"
            );
        }
    }

    #[test]
    fn immediates_follow_the_operation() {
        use BinOp::*;
        let imm = |op, c| match with_constant(op, c) {
            Opcode::ArithImm(a) => {
                assert_eq!(a.lhs, Value::from_u32(0), "{op:?} {c}");
                Some((a.op, a.imm))
            }
            Opcode::Arith(a) => {
                assert_eq!(a.op, op, "{op:?} {c}");
                None
            }
            other => panic!("unexpected {other:?}"),
        };
        // Unsigned division and remainder by a power of two are a shift and a mask ...
        assert_eq!(imm(UDiv, 8), Some((Shr, 3)));
        assert_eq!(imm(URem, 8), Some((BitAnd, 7)));
        assert_eq!(imm(URem, 4096), None);
        // ... the signed forms and the high multiplies are not.
        assert_eq!(imm(SDiv, 8), None);
        assert_eq!(imm(SRem, 8), None);
        assert_eq!(imm(UMulh, 2), None);
        assert_eq!(imm(SMulh, 2), None);
        // Both right shifts and the left shift have an immediate form.
        assert_eq!(imm(Shr, 3), Some((Shr, 3)));
        assert_eq!(imm(Sar, 3), Some((Sar, 3)));
        assert_eq!(imm(Shl, 3), Some((Shl, 3)));
        assert_eq!(imm(Sar, 64), None);
        assert_eq!(imm(Mul, 16), Some((Shl, 4)));
        // A constant that is -1 read signed is still `-1` for add / sub, which wrap.
        assert_eq!(imm(Add, -1), Some((Add, -1)));
        assert_eq!(imm(Sub, -1), Some((Sub, -1)));
        assert_eq!(imm(Add, 0x7ff), Some((Add, 0x7ff)));
        assert_eq!(imm(Add, 0x800), None);
        assert_eq!(imm(BitAnd, 0xff), Some((BitAnd, 0xff)));
    }
}
