// Ported and modified from Vulcan, Copyright 2026 LilithSemi (Apache-2.0).
// Rust Function-based lowering and executable/object integration modifications.
use super::super::{encode as e, peephole};
use super::{Abi, Compiled, Error, Fixup, LineEntry, ModelCaps, Reloc, RelocKind};
use crate::regalloc::wimmer::{
    self, Allocation, CallSite, ClassRegs, FixedAssign, Location, Move, RegClass, RegDescription,
    UseKind,
};
use alloc::{vec, vec::Vec};
use e::{Cond, FKind, Reg};
use volt_ir::attribute::{AttrValue, Attribute};
use volt_ir::function::{
    AttrTarget, BinOp, Block, CmpOp, Function, Inst, Jump, Opcode, Ret, Terminator, UnaryOp, Value,
};
use volt_ir::types::{FloatKind, Type, TypeKind};

fn reg(n: u16) -> Reg {
    Reg::from_u5(n as u32)
}
fn kind(f: &Function, v: Value) -> &TypeKind {
    f.types.type_kind(f.value_type(v))
}
fn class(f: &Function, v: Value) -> u16 {
    u16::from(matches!(
        kind(f, v),
        TypeKind::Float(_) | TypeKind::Vector(_)
    ))
}
fn half(f: &Function, v: Value) -> bool {
    matches!(kind(f, v), TypeKind::Float(FloatKind::F16))
}
fn double(f: &Function, v: Value) -> bool {
    matches!(kind(f, v), TypeKind::Float(FloatKind::F64))
}
fn quad(f: &Function, v: Value) -> bool {
    matches!(kind(f, v), TypeKind::Float(FloatKind::F128))
}
fn vector(f: &Function, v: Value) -> bool {
    matches!(kind(f, v), TypeKind::Vector(_))
}
fn signed(f: &Function, v: Value) -> bool {
    matches!(kind(f,v), TypeKind::Int(i) if i.signed)
}
fn bits(f: &Function, v: Value) -> u16 {
    match kind(f, v) {
        TypeKind::Int(i) => i.bits,
        TypeKind::Bool => 1,
        _ => 64,
    }
}
fn wide(f: &Function, v: Value) -> bool {
    bits(f, v) > 32
}
fn fkind(f: &Function, v: Value, caps: &ModelCaps) -> FKind {
    if half(f, v) && caps.fp16 {
        FKind::Half
    } else if double(f, v) {
        FKind::Double
    } else {
        FKind::Single
    }
}
fn align(n: usize, a: usize) -> usize {
    (n + a - 1) & !(a - 1)
}
fn layout(f: &Function, ty: Type) -> Result<(usize, usize), Error> {
    Ok(match f.types.type_kind(ty) {
        TypeKind::Bool => (1, 1),
        TypeKind::Int(i) if i.bits > 0 && i.bits <= 64 => {
            let n = (i.bits as usize).div_ceil(8);
            (n, n.next_power_of_two())
        }
        TypeKind::Float(k) => match k {
            FloatKind::F16 => (2, 2),
            FloatKind::F32 => (4, 4),
            FloatKind::F64 => (8, 8),
            FloatKind::F128 => (16, 16),
        },
        TypeKind::Ptr(_) => (8, 8),
        TypeKind::Vector(v) => {
            let (s, _) = layout(f, v.elem)?;
            let n = s.checked_mul(v.len as usize).ok_or(Error::Unsupported)?;
            (n, n.next_power_of_two().min(16))
        }
        TypeKind::Array(a) => {
            let (s, al) = layout(f, a.elem)?;
            (s.checked_mul(a.len as usize).ok_or(Error::Unsupported)?, al)
        }
        TypeKind::Struct(fields) => {
            let mut n = 0;
            let mut a = 1;
            for t in fields {
                let (s, al) = layout(f, *t)?;
                n = align(n, al) + s;
                a = a.max(al);
            }
            (align(n, a), a)
        }
        TypeKind::Slice(_) => (16, 8),
        _ => return Err(Error::Unsupported),
    })
}
// The source's native NEON lane operations are four f32/i32 lanes only.
fn lane(f: &Function, v: Value) -> Result<bool, Error> {
    match kind(f, v) {
        TypeKind::Vector(x) if x.len == 4 => match f.types.type_kind(x.elem) {
            TypeKind::Float(FloatKind::F32) => Ok(true),
            TypeKind::Int(i) if i.bits == 32 => Ok(false),
            _ => Err(Error::Unsupported),
        },
        _ => Err(Error::Unsupported),
    }
}

#[derive(Clone, Copy)]
enum Fusion {
    Float {
        a: Value,
        b: Value,
        c: Value,
        shape: u8,
    },
    Shift {
        a: Value,
        b: Value,
        shift: u8,
        sub: bool,
    },
    Compare {
        lhs: Value,
        rhs: Value,
        op: CmpOp,
    },
}
struct Model {
    classes: Vec<RegClass>,
    calls: Vec<CallSite>,
    fused: Vec<(Inst, Fusion)>,
    skipped: Vec<Inst>,
}
impl Model {
    fn new(f: &Function, caps: &ModelCaps) -> Self {
        let classes = vec![
            RegClass {
                name: "gpr",
                allocatable: (9..13).chain(19..29).collect(),
                callee_saved: (19..29).collect(),
                slot_bytes: 16,
                narrow_preserved: vec![],
            },
            RegClass {
                name: "fpr",
                allocatable: (16..24).chain(28..32).chain(8..16).collect(),
                callee_saved: (8..16).collect(),
                slot_bytes: 16,
                narrow_preserved: (8..16).collect(),
            },
        ];
        let mut calls = Vec::new();
        let mut pos = 0;
        for bi in 0..f.block_count() {
            pos += 1;
            for i in f.block_insts(Block(bi as u32)) {
                if matches!(f.opcode(*i), Opcode::Call(_) | Opcode::CallIndirect(_)) {
                    calls.push(CallSite {
                        pos,
                        clobbered: vec![
                            ClassRegs {
                                class: 0,
                                regs: (0..18).collect(),
                            },
                            ClassRegs {
                                class: 1,
                                regs: (0..32).collect(),
                            },
                        ],
                    });
                }
                pos += 1;
            }
            pos += 1;
        }
        let mut uses = vec![0usize; f.value_count()];
        for bi in 0..f.block_count() {
            let b = Block(bi as u32);
            for i in f.block_insts(b) {
                crate::regalloc::ir::visit_inst_operands(f, *i, &mut |v, _| uses[v.index()] += 1);
            }
            if let Some(t) = f.terminator(b) {
                crate::regalloc::ir::visit_term_operands(f, &t, &mut |v, _| uses[v.index()] += 1);
            }
        }
        let mut fused = Vec::new();
        let mut skipped = Vec::new();
        for bi in 0..f.block_count() {
            for pair in f.block_insts(Block(bi as u32)).windows(2) {
                let p = pair[0];
                let i = pair[1];
                let Some(v) = f.inst_result(p) else {
                    continue;
                };
                if caps.fuse_cmp_arith && uses[v.index()] == 1 {
                    if let (Opcode::Icmp(c), Opcode::If(cf)) = (f.opcode(p), f.opcode(i)) {
                        if cf.cond == v
                            && class(f, c.lhs) == 0
                            && f.block_args(cf.then).len() + f.block_args(cf.else_).len() <= 2
                        {
                            skipped.push(p);
                            fused.push((
                                i,
                                Fusion::Compare {
                                    lhs: c.lhs,
                                    rhs: c.rhs,
                                    op: c.op,
                                },
                            ));
                            continue;
                        }
                    }
                }
                let Opcode::Arith(a) = f.opcode(i) else {
                    continue;
                };
                if uses[v.index()] != 1
                    || !matches!(a.op, BinOp::Add | BinOp::Sub)
                    || (a.lhs != v && a.rhs != v)
                {
                    continue;
                }
                let fusion = match f.opcode(p) {
                    Opcode::Arith(m)
                        if m.op == BinOp::Mul && class(f, v) == 1 && !half(f, v) && !quad(f, v) =>
                    {
                        let shape = if a.op == BinOp::Add {
                            0
                        } else if a.lhs == v {
                            1
                        } else {
                            2
                        };
                        if vector(f, v) && (shape == 1 || lane(f, v) != Ok(true)) {
                            continue;
                        }
                        Fusion::Float {
                            a: m.lhs,
                            b: m.rhs,
                            c: if a.lhs == v { a.rhs } else { a.lhs },
                            shape,
                        }
                    }
                    Opcode::ArithImm(s)
                        if caps.fuse_shift_add
                            && s.op == BinOp::Shl
                            && class(f, v) == 0
                            && s.imm >= 0
                            && s.imm < bits(f, v) as i64 =>
                    {
                        if a.op == BinOp::Sub && a.lhs == v {
                            continue;
                        }
                        Fusion::Shift {
                            a: if a.lhs == v { a.rhs } else { a.lhs },
                            b: s.lhs,
                            shift: s.imm as u8,
                            sub: a.op == BinOp::Sub,
                        }
                    }
                    _ => continue,
                };
                skipped.push(p);
                fused.push((i, fusion));
            }
        }
        Self {
            classes,
            calls,
            fused,
            skipped,
        }
    }
}
impl RegDescription<Function> for Model {
    fn classes(&self) -> &[RegClass] {
        &self.classes
    }
    fn class_of(&self, f: &Function, v: Value) -> u16 {
        class(f, v)
    }
    fn use_kind(&self, _: &Function, _: Inst, _: Value) -> UseKind {
        UseKind::ShouldHaveRegister
    }
    fn entry_fixed(&self) -> &[FixedAssign] {
        &[]
    }
    fn call_sites(&self) -> &[CallSite] {
        &self.calls
    }
    fn scratch(&self) -> &[u16] {
        &[17, 27]
    }
    fn scratch2(&self) -> &[u16] {
        &[16, 26]
    }
    fn is_narrow(&self, f: &Function, v: Value) -> bool {
        class(f, v) == 1 && !quad(f, v) && !vector(f, v)
    }
    fn coalesce_block_params(&self) -> bool {
        true
    }
    fn coalesce_spill_slots(&self) -> bool {
        true
    }
    fn fused_operands(
        &self,
        f: &Function,
        i: Inst,
        out: &mut [Value; wimmer::MAX_FUSED_OPERANDS],
    ) -> Option<u8> {
        if self.skipped.contains(&i) {
            return Some(0);
        }
        match self.fused.iter().find(|(inst, _)| *inst == i)?.1 {
            Fusion::Float { a, b, c, .. } => {
                out[0] = a;
                out[1] = b;
                out[2] = c;
                Some(3)
            }
            Fusion::Shift { a, b, .. } => {
                out[0] = a;
                out[1] = b;
                Some(2)
            }
            Fusion::Compare { lhs, rhs, .. } => {
                out[0] = lhs;
                out[1] = rhs;
                let mut n = 2;
                if let Opcode::If(cf) = f.opcode(i) {
                    for v in f.block_args(cf.then).iter().chain(f.block_args(cf.else_)) {
                        out[n] = *v;
                        n += 1;
                    }
                }
                Some(n as u8)
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Arg {
    Reg(u16),
    Stack(usize),
}
fn arg_locations(
    f: &Function,
    values: &[Value],
    sret: bool,
    abi: Abi,
) -> Result<(Vec<Arg>, usize), Error> {
    let mut counts = [0u16; 2];
    let mut stack = 0;
    let mut out = Vec::with_capacity(values.len());
    for (i, v) in values.iter().enumerate() {
        if sret && i == 0 {
            out.push(Arg::Reg(8));
            continue;
        }
        let c = class(f, *v) as usize;
        let n = counts[c];
        counts[c] += 1;
        if n < 8 {
            out.push(Arg::Reg(n));
        } else {
            let (size, a) = layout(f, f.value_type(*v))?;
            let (size, a) = if abi == Abi::Apple {
                (size, a)
            } else {
                (align(size, 8), a.max(8))
            };
            stack = align(stack, a);
            out.push(Arg::Stack(stack));
            stack += size;
        }
    }
    Ok((out, align(stack, 16)))
}

struct Emitter<'a> {
    f: &'a Function,
    caps: &'a ModelCaps,
    native: bool,
    model: &'a Model,
    alloc: Allocation,
    code: Vec<u32>,
    relocs: Vec<Reloc>,
    lines: Vec<LineEntry>,
    labels: Vec<usize>,
    fixups: Vec<Fixup>,
    pos: u32,
    block: Block,
    slot_base: usize,
    frame: usize,
    saved: Vec<(u16, u16, usize)>,
    alloca: Vec<Option<usize>>,
    gr_save: usize,
    vr_save: usize,
    fold: crate::regalloc::addrfold::Analysis,
}
impl Emitter<'_> {
    fn push(&mut self, w: u32) {
        self.code.push(w);
    }
    fn constant(&mut self, r: Reg, v: u64) {
        self.push(e::movz64(r, v as u16, 0));
        for k in 1..4 {
            let p = (v >> (k * 16)) as u16;
            if p != 0 {
                self.push(e::movk64(r, p, k));
            }
        }
    }
    fn frame_imm(&mut self, sub: bool, rd: Reg, rn: Reg, amount: usize) -> Result<(), Error> {
        if amount >= 1 << 24 {
            return Err(Error::Unsupported);
        }
        let hi = (amount >> 12) as u16;
        let lo = (amount & 4095) as u16;
        let mut src = rn;
        if hi != 0 {
            self.push(if sub {
                e::sub_imm64_shift(rd, src, hi)
            } else {
                e::add_imm64_shift(rd, src, hi)
            });
            src = rd;
        }
        if lo != 0 || hi == 0 {
            self.push(if sub {
                e::sub_imm64(rd, src, lo)
            } else {
                e::add_imm64(rd, src, lo)
            });
        }
        Ok(())
    }
    fn slot_offset(&self, c: u16, s: u32) -> usize {
        self.slot_base
            + 16 * (s as usize
                + if c == 1 {
                    self.alloc.slot_count_per_class[0] as usize
                } else {
                    0
                })
    }
    // Spill traffic always preserves the complete register representation (including
    // emulated half's exact f32 widening and all 128 bits of vector/quad values).
    fn raw_mem(&mut self, load: bool, c: u16, r: Reg, base: Reg, off: usize) -> Result<(), Error> {
        let scale = if c == 1 { 16 } else { 8 };
        let (base, off) = self.address(base, off, scale)?;
        self.push(if c == 1 {
            if load {
                e::ldr_q(r, base, off)
            } else {
                e::str_q(r, base, off)
            }
        } else if load {
            e::ldr_off(r, base, off)
        } else {
            e::str_off(r, base, off)
        });
        Ok(())
    }
    fn address(&mut self, base: Reg, off: usize, scale: usize) -> Result<(Reg, u16), Error> {
        if off % scale == 0 && off / scale <= 4095 && off <= u16::MAX as usize {
            return Ok((base, off as u16));
        }
        // x8 is outside both allocation and edge-move scratch pools.
        self.frame_imm(false, Reg::X8, base, off)?;
        Ok((Reg::X8, 0))
    }
    fn loc(&self, v: Value) -> Result<Location, Error> {
        let s = self.alloc.segments_of(v).ok_or(Error::Unsupported)?;
        s.iter()
            .rev()
            .find(|s| s.from <= self.pos)
            .or_else(|| s.first())
            .map(|s| s.loc)
            .ok_or(Error::Unsupported)
    }
    fn read(&mut self, v: Value, scratch: Reg) -> Result<Reg, Error> {
        match self.loc(v)? {
            Location::Reg(r) => Ok(reg(r)),
            Location::Slot(s) => {
                self.raw_mem(
                    true,
                    class(self.f, v),
                    scratch,
                    Reg::Zr,
                    self.slot_offset(class(self.f, v), s),
                )?;
                Ok(scratch)
            }
        }
    }
    fn result(&self, v: Value) -> Result<Reg, Error> {
        Ok(match self.loc(v)? {
            Location::Reg(r) => reg(r),
            Location::Slot(_) => {
                if class(self.f, v) == 1 {
                    reg(26)
                } else {
                    reg(15)
                }
            }
        })
    }
    fn finish(&mut self, v: Value, r: Reg) -> Result<(), Error> {
        if class(self.f, v) == 0 {
            self.normalize(v, r);
        }
        if let Location::Slot(s) = self.loc(v)? {
            self.raw_mem(
                false,
                class(self.f, v),
                r,
                Reg::Zr,
                self.slot_offset(class(self.f, v), s),
            )?;
        }
        Ok(())
    }
    fn normalize(&mut self, v: Value, r: Reg) {
        let n = bits(self.f, v);
        if n < 64 {
            self.push(if signed(self.f, v) {
                e::sbfm(r, r, 0, (n - 1) as u8)
            } else {
                e::ubfm(r, r, 0, (n - 1) as u8)
            });
        }
    }
    fn move_one(&mut self, m: Move) -> Result<(), Error> {
        if m.src == m.dst {
            return Ok(());
        }
        match (m.src, m.dst) {
            (Location::Reg(s), Location::Reg(d)) => self.push(if m.class == 1 {
                e::mov_vec(reg(d), reg(s))
            } else {
                e::mov(reg(d), reg(s))
            }),
            (Location::Slot(s), Location::Reg(d)) => {
                self.raw_mem(true, m.class, reg(d), Reg::Zr, self.slot_offset(m.class, s))?
            }
            (Location::Reg(s), Location::Slot(d)) => self.raw_mem(
                false,
                m.class,
                reg(s),
                Reg::Zr,
                self.slot_offset(m.class, d),
            )?,
            (Location::Slot(s), Location::Slot(d)) => {
                let r = if m.class == 1 { reg(27) } else { reg(17) };
                self.raw_mem(true, m.class, r, Reg::Zr, self.slot_offset(m.class, s))?;
                self.raw_mem(false, m.class, r, Reg::Zr, self.slot_offset(m.class, d))?;
            }
        }
        Ok(())
    }
    fn actions(&mut self, cursor: &mut usize) -> Result<(), Error> {
        while *cursor < self.alloc.actions.len() && self.alloc.actions[*cursor].at <= self.pos {
            let a = self.alloc.actions[*cursor];
            self.move_one(Move {
                src: a.src,
                dst: a.dst,
                class: a.class,
                value: a.value,
            })?;
            *cursor += 1;
        }
        Ok(())
    }
    fn local_label(&mut self) -> u32 {
        let n = self.labels.len();
        self.labels.push(0);
        n as u32
    }
    fn bind(&mut self, label: u32) {
        self.labels[label as usize] = self.code.len();
    }
    fn branch(&mut self, target: u32, word: u32) {
        self.fixups.push(Fixup {
            at: self.code.len(),
            target,
        });
        self.push(word);
    }
    fn edge(&mut self, j: Jump) -> Result<(), Error> {
        if let Some(em) = self
            .alloc
            .edge_moves
            .iter()
            .find(|em| em.pred == self.block && em.succ == j.target)
        {
            let moves = em.moves.clone();
            for m in moves {
                self.move_one(m)?;
            }
        }
        self.branch(j.target.0, e::b(0));
        Ok(())
    }
    fn half_round(&mut self, r: Reg) {
        self.push(e::fcvt_hfrom_s(r, r));
        self.push(e::fcvt_sfrom_h(r, r));
    }
    fn memory(&mut self, load: bool, v: Value, r: Reg, base: Reg, off: usize) -> Result<(), Error> {
        let (size, _) = layout(self.f, self.f.value_type(v))?;
        let (base, off) = self.address(base, off, size.next_power_of_two())?;
        let word = if vector(self.f, v) || quad(self.f, v) {
            if size != 16 {
                return Err(Error::Unsupported);
            }
            if load {
                e::ldr_q(r, base, off)
            } else {
                e::str_q(r, base, off)
            }
        } else if half(self.f, v) {
            if load {
                e::ldr_hfp(r, base, off)
            } else {
                let src = if self.caps.fp16 {
                    r
                } else {
                    self.push(e::fcvt_hfrom_s(reg(27), r));
                    reg(27)
                };
                e::str_hfp(src, base, off)
            }
        } else if class(self.f, v) == 1 {
            if load {
                e::ldr_fp(r, base, off, double(self.f, v))
            } else {
                e::str_fp(r, base, off, double(self.f, v))
            }
        } else {
            match (load, size, signed(self.f, v)) {
                (true, 1, true) => e::ldrsb_off(r, base, off),
                (true, 1, false) => e::ldrb_off(r, base, off),
                (true, 2, true) => e::ldrsh_off(r, base, off),
                (true, 2, false) => e::ldrh_off(r, base, off),
                (true, 3..=4, true) => e::ldrsw(r, base, off),
                (true, 3..=4, false) => e::ldr_w(r, base, off),
                (true, 5..=8, _) => e::ldr_off(r, base, off),
                (false, 1, _) => e::strb_off(r, base, off),
                (false, 2, _) => e::strh_off(r, base, off),
                (false, 3..=4, _) => e::str_w(r, base, off),
                (false, 5..=8, _) => e::str_off(r, base, off),
                _ => return Err(Error::Unsupported),
            }
        };
        self.push(word);
        if load && half(self.f, v) && !self.caps.fp16 {
            self.push(e::fcvt_sfrom_h(r, r));
        }
        Ok(())
    }
    fn integer_binary(
        &mut self,
        v: Value,
        op: BinOp,
        rd: Reg,
        a: Reg,
        b: Reg,
    ) -> Result<(), Error> {
        let sf = if wide(self.f, v) { 1 << 31 } else { 0 };
        let si = signed(self.f, v);
        let w = match op {
            BinOp::Add => e::add(rd, a, b),
            BinOp::Sub => e::sub(rd, a, b),
            BinOp::Mul => e::mul(rd, a, b),
            BinOp::Mulh => {
                if bits(self.f, v) != 64 {
                    return Err(Error::Unsupported);
                }
                if si {
                    e::smulh(rd, a, b)
                } else {
                    e::umulh(rd, a, b)
                }
            }
            BinOp::Div => {
                if si {
                    e::sdiv(rd, a, b)
                } else {
                    e::udiv(rd, a, b)
                }
            }
            BinOp::Rem => {
                self.push(
                    sf | if si {
                        e::sdiv(reg(16), a, b)
                    } else {
                        e::udiv(reg(16), a, b)
                    },
                );
                self.push(sf | e::msub(rd, reg(16), b, a));
                return Ok(());
            }
            BinOp::BitAnd => e::andr(rd, a, b),
            BinOp::BitOr => e::orr(rd, a, b),
            BinOp::BitXor => e::eor(rd, a, b),
            BinOp::Shl => e::lslv(rd, a, b),
            BinOp::Shr => {
                if si {
                    e::asrv(rd, a, b)
                } else {
                    e::lsrv(rd, a, b)
                }
            }
        };
        self.push(sf | w);
        Ok(())
    }
    fn binary(&mut self, v: Value, op: BinOp, lhs: Value, rhs: Value) -> Result<(), Error> {
        let fp = class(self.f, v) == 1;
        let a = self.read(lhs, reg(if fp { 24 } else { 13 }))?;
        let b = self.read(rhs, reg(if fp { 25 } else { 14 }))?;
        let rd = self.result(v)?;
        if quad(self.f, v) {
            return Err(Error::Unsupported);
        }
        if vector(self.f, v) {
            if !lane(self.f, v)? {
                return Err(Error::Unsupported);
            }
            self.push(match op {
                BinOp::Add => e::fadd_vec(rd, a, b),
                BinOp::Sub => e::fsub_vec(rd, a, b),
                BinOp::Mul => e::fmul_vec(rd, a, b),
                BinOp::Div => e::fdiv_vec(rd, a, b),
                _ => return Err(Error::Unsupported),
            });
        } else if fp {
            let k = fkind(self.f, v, self.caps);
            self.push(match op {
                BinOp::Add => e::fadd(rd, a, b, k),
                BinOp::Sub => e::fsub(rd, a, b, k),
                BinOp::Mul => e::fmul(rd, a, b, k),
                BinOp::Div => e::fdiv(rd, a, b, k),
                _ => return Err(Error::Unsupported),
            });
            if half(self.f, v) && !self.caps.fp16 {
                self.half_round(rd);
            }
        } else {
            self.integer_binary(v, op, rd, a, b)?;
        }
        self.finish(v, rd)
    }
    fn fusion(&mut self, v: Value, fusion: Fusion) -> Result<(), Error> {
        let rd = self.result(v)?;
        match fusion {
            Fusion::Float { a, b, c, shape } => {
                let a = self.read(a, reg(24))?;
                let b = self.read(b, reg(25))?;
                let c = self.read(c, reg(26))?;
                if vector(self.f, v) {
                    self.push(e::mov_vec(reg(27), c));
                    self.push(if shape == 0 {
                        e::fmla_vec(reg(27), a, b)
                    } else {
                        e::fmls_vec(reg(27), a, b)
                    });
                    self.push(e::mov_vec(rd, reg(27)));
                } else {
                    let d = double(self.f, v);
                    self.push(match shape {
                        0 => e::fmadd(rd, a, b, c, d),
                        1 => e::fnmsub(rd, a, b, c, d),
                        _ => e::fmsub(rd, a, b, c, d),
                    });
                }
            }
            Fusion::Shift { a, b, shift, sub } => {
                let a = self.read(a, reg(13))?;
                let b = self.read(b, reg(14))?;
                self.push(
                    (if sub {
                        e::sub_shifted(rd, a, b, shift)
                    } else {
                        e::add_shifted(rd, a, b, shift)
                    }) | if wide(self.f, v) { 1 << 31 } else { 0 },
                );
            }
            Fusion::Compare { .. } => return Err(Error::Unsupported),
        }
        self.finish(v, rd)
    }
    fn convert(&mut self, v: Value, src: Value) -> Result<(), Error> {
        if quad(self.f, v) || quad(self.f, src) || vector(self.f, v) || vector(self.f, src) {
            return Err(Error::Unsupported);
        }
        let sc = class(self.f, src);
        let dc = class(self.f, v);
        let a = self.read(src, reg(if sc == 1 { 24 } else { 13 }))?;
        let rd = self.result(v)?;
        match (sc, dc) {
            (0, 1) => {
                self.push(e::cvt_int_to_float(
                    rd,
                    a,
                    fkind(self.f, v, self.caps),
                    signed(self.f, src),
                ));
                if half(self.f, v) && !self.caps.fp16 {
                    self.half_round(rd);
                }
            }
            (1, 0) => self.push(e::cvt_float_to_int(
                rd,
                a,
                fkind(self.f, src, self.caps),
                signed(self.f, v),
            )),
            (1, 1) => {
                if self.caps.fp16 && (half(self.f, src) || half(self.f, v)) {
                    self.push(if half(self.f, src) && half(self.f, v) {
                        e::fmov_reg(rd, a)
                    } else if half(self.f, v) {
                        if double(self.f, src) {
                            e::fcvt_hfrom_d(rd, a)
                        } else {
                            e::fcvt_hfrom_s(rd, a)
                        }
                    } else if double(self.f, v) {
                        e::fcvt_dfrom_h(rd, a)
                    } else {
                        e::fcvt_sfrom_h(rd, a)
                    });
                } else if half(self.f, v) {
                    self.push(if double(self.f, src) {
                        e::fcvt_hfrom_d(rd, a)
                    } else {
                        e::fcvt_hfrom_s(rd, a)
                    });
                    self.push(e::fcvt_sfrom_h(rd, rd));
                } else if double(self.f, src) == double(self.f, v) {
                    self.push(e::fmov_reg(rd, a));
                } else {
                    self.push(e::fcvt(rd, a, double(self.f, v)));
                }
            }
            _ => {
                let n = bits(self.f, src);
                if bits(self.f, v) > n && n < 64 {
                    self.push(if signed(self.f, src) {
                        e::sbfm(rd, a, 0, (n - 1) as u8)
                    } else {
                        e::ubfm(rd, a, 0, (n - 1) as u8)
                    });
                } else {
                    self.push(e::mov(rd, a));
                }
            }
        }
        self.finish(v, rd)
    }
    fn setup_args(&mut self, args: &[Value], sret: bool) -> Result<(), Error> {
        let (locs, _) = arg_locations(self.f, args, sret, self.caps.abi)?;
        // Allocation excludes all ABI argument registers. Sequential homing is
        // therefore a parallel move without register cycles.
        for (v, l) in args.iter().zip(locs) {
            let c = class(self.f, *v);
            let a = self.read(*v, reg(if c == 1 { 24 } else { 13 }))?;
            match l {
                Arg::Reg(r) => {
                    if half(self.f, *v) && !self.caps.fp16 {
                        self.push(e::fcvt_hfrom_s(reg(r), a));
                    } else {
                        self.push(if c == 1 {
                            e::mov_vec(reg(r), a)
                        } else {
                            e::mov(reg(r), a)
                        });
                    }
                }
                Arg::Stack(o) => self.memory(false, *v, a, Reg::Zr, o)?,
            }
        }
        Ok(())
    }
    fn symbol(&self, name: &str) -> Result<u64, Error> {
        #[cfg(unix)]
        {
            let name = alloc::ffi::CString::new(name).map_err(|_| Error::Unsupported)?;
            // SAFETY: the symbol string is terminated and RTLD_DEFAULT denotes
            // the process-global symbol scope. The returned address is emitted,
            // not dereferenced by this compiler.
            let p = unsafe { libc::dlsym(libc::RTLD_DEFAULT, name.as_ptr()) };
            if p.is_null() {
                Err(Error::Unsupported)
            } else {
                Ok(p as usize as u64)
            }
        }
        #[cfg(not(unix))]
        {
            let _ = name;
            Err(Error::Unsupported)
        }
    }
    fn call_result(
        &mut self,
        result: Option<Value>,
        dest: Option<Value>,
        pieces: &[volt_ir::function::RetPiece],
        count: u8,
    ) -> Result<(), Error> {
        if let Some(v) = result {
            let r = self.result(v)?;
            if half(self.f, v) && !self.caps.fp16 {
                self.push(e::fcvt_sfrom_h(r, reg(0)));
            } else {
                self.push(if class(self.f, v) == 1 {
                    e::mov_vec(r, reg(0))
                } else {
                    e::mov(r, reg(0))
                });
            }
            self.finish(v, r)?;
        }
        if let Some(v) = dest {
            let off = self.alloca[v.index()].ok_or(Error::Unsupported)?;
            self.frame_imm(false, reg(13), Reg::Zr, off)?;
            let mut bank = [0u16; 2];
            for p in &pieces[..count as usize] {
                let c = u16::from(p.fp);
                let r = reg(bank[c as usize]);
                bank[c as usize] += 1;
                let (base, offset) = self.address(reg(13), p.offset as usize, p.bytes as usize)?;
                self.push(if p.fp {
                    e::str_fp(r, base, offset, p.bytes != 4)
                } else {
                    match p.bytes {
                        1 => e::strb_off(r, base, offset),
                        2 => e::strh_off(r, base, offset),
                        4 => e::str_w(r, base, offset),
                        8 => e::str_off(r, base, offset),
                        _ => return Err(Error::Unsupported),
                    }
                });
            }
        }
        Ok(())
    }
    fn va_start(&mut self, list: Value) -> Result<(), Error> {
        let lp = self.read(list, reg(13))?;
        for (field, off) in [
            (0, self.frame),
            (8, self.gr_save + 64),
            (16, self.vr_save + 128),
        ] {
            self.frame_imm(false, reg(16), Reg::Zr, off)?;
            self.push(e::str_off(reg(16), lp, field));
        }
        let mut named = [0usize; 2];
        for v in self
            .f
            .block_params(Block(0))
            .iter()
            .take(self.f.num_fixed_params as usize)
        {
            named[class(self.f, *v) as usize] += 1;
        }
        for (c, field, stride) in [(0, 24, 8), (1, 28, 16)] {
            let n = -((stride * (8 - named[c].min(8))) as i64);
            self.constant(reg(16), n as u64);
            self.push(e::str_w(reg(16), lp, field));
        }
        Ok(())
    }
    fn va_arg(&mut self, list: Value, v: Value) -> Result<(), Error> {
        let fp = class(self.f, v) == 1;
        let lp = self.read(list, reg(13))?;
        let field = if fp { 28 } else { 24 };
        let top = if fp { 16 } else { 8 };
        let stride = if fp { 16 } else { 8 };
        self.push(e::ldrsw(reg(16), lp, field));
        self.push(e::cmp64(reg(16), Reg::Zr));
        let stack = self.local_label();
        let load = self.local_label();
        self.branch(stack, e::bcc(Cond::Ge, 0));
        self.push(e::ldr_off(reg(14), lp, top));
        self.push(e::add64(reg(17), reg(14), reg(16)));
        self.push(e::add_imm64(reg(16), reg(16), stride));
        self.push(e::str_w(reg(16), lp, field));
        self.branch(load, e::b(0));
        self.bind(stack);
        self.push(e::ldr_off(reg(17), lp, 0));
        self.push(e::add_imm64(reg(14), reg(17), 8));
        self.push(e::str_off(reg(14), lp, 0));
        self.bind(load);
        let rd = self.result(v)?;
        self.memory(true, v, rd, reg(17), 0)?;
        self.finish(v, rd)
    }
    fn instruction(&mut self, inst: Inst) -> Result<bool, Error> {
        let result = self.f.inst_result(inst);
        if self.model.skipped.contains(&inst) {
            return Ok(false);
        }
        if let Some(fusion) = self
            .model
            .fused
            .iter()
            .find(|(i, _)| *i == inst)
            .map(|(_, f)| *f)
        {
            if let Fusion::Compare { lhs, rhs, op } = fusion {
                let a = self.read(lhs, reg(13))?;
                let b = self.read(rhs, reg(14))?;
                self.push(if wide(self.f, lhs) {
                    e::cmp64(a, b)
                } else {
                    e::cmp(a, b)
                });
                let Opcode::If(cf) = self.f.opcode(inst) else {
                    return Err(Error::Unsupported);
                };
                let other = self.local_label();
                self.branch(
                    other,
                    e::bcc(e::invert(int_cond(op, signed(self.f, lhs))), 0),
                );
                self.edge(cf.then)?;
                self.bind(other);
                self.edge(cf.else_)?;
                return Ok(true);
            }
            self.fusion(result.ok_or(Error::Unsupported)?, fusion)?;
            return Ok(false);
        }
        match self.f.opcode(inst) {
            Opcode::Iconst(n) => {
                let v = result.ok_or(Error::Unsupported)?;
                let rd = self.result(v)?;
                if class(self.f, v) == 1 {
                    self.constant(reg(16), n as u64);
                    self.push(e::fmov_from_gpr(rd, reg(16), double(self.f, v)));
                } else {
                    self.constant(rd, n as u64);
                }
                self.finish(v, rd)?;
            }
            Opcode::Fconst(n) => {
                let v = result.ok_or(Error::Unsupported)?;
                let rd = self.result(v)?;
                if half(self.f, v) {
                    self.constant(reg(16), n.to_bits());
                    self.push(e::fmov_from_gpr(rd, reg(16), true));
                    self.push(e::fcvt_hfrom_d(rd, rd));
                    if !self.caps.fp16 {
                        self.push(e::fcvt_sfrom_h(rd, rd));
                    }
                } else {
                    self.constant(
                        reg(16),
                        if double(self.f, v) {
                            n.to_bits()
                        } else {
                            (n as f32).to_bits() as u64
                        },
                    );
                    self.push(e::fmov_from_gpr(rd, reg(16), double(self.f, v)));
                }
                self.finish(v, rd)?;
            }
            Opcode::Fconst128(n) => {
                let v = result.ok_or(Error::Unsupported)?;
                let rd = self.result(v)?;
                self.constant(reg(16), n as u64);
                self.push(e::fmov_from_gpr(rd, reg(16), true));
                self.constant(reg(16), (n >> 64) as u64);
                self.push(e::ins_d1_from_gpr(rd, reg(16)));
                self.finish(v, rd)?;
            }
            Opcode::Arith(a) => {
                self.binary(result.ok_or(Error::Unsupported)?, a.op, a.lhs, a.rhs)?
            }
            Opcode::ArithImm(a) => {
                let v = result.ok_or(Error::Unsupported)?;
                if class(self.f, v) != 0 {
                    return Err(Error::Unsupported);
                }
                let r = self.read(a.lhs, reg(13))?;
                let rd = self.result(v)?;
                if matches!(a.op, BinOp::Add | BinOp::Sub) && a.imm.unsigned_abs() <= 4095 {
                    let sub = (a.op == BinOp::Sub) ^ (a.imm < 0);
                    let sf = if wide(self.f, v) { 1 << 31 } else { 0 };
                    self.push(
                        sf | if sub {
                            e::sub_imm(rd, r, a.imm.unsigned_abs() as u16)
                        } else {
                            e::add_imm(rd, r, a.imm.unsigned_abs() as u16)
                        },
                    );
                } else {
                    self.constant(reg(14), a.imm as u64);
                    self.integer_binary(v, a.op, rd, r, reg(14))?;
                }
                self.finish(v, rd)?;
            }
            Opcode::Icmp(c) => {
                let v = result.ok_or(Error::Unsupported)?;
                let fp = class(self.f, c.lhs) == 1;
                let a = self.read(c.lhs, reg(if fp { 24 } else { 13 }))?;
                let b = self.read(c.rhs, reg(if fp { 25 } else { 14 }))?;
                let rd = self.result(v)?;
                if vector(self.f, c.lhs) {
                    if !lane(self.f, c.lhs)? {
                        return Err(Error::Unsupported);
                    }
                    self.push(match c.op {
                        CmpOp::Eq | CmpOp::Ne => e::fcmeq_vec(rd, a, b),
                        CmpOp::Lt => e::fcmgt_vec(rd, b, a),
                        CmpOp::Le => e::fcmge_vec(rd, b, a),
                        CmpOp::Gt => e::fcmgt_vec(rd, a, b),
                        CmpOp::Ge => e::fcmge_vec(rd, a, b),
                    });
                    if c.op == CmpOp::Ne {
                        self.push(e::mvn_vec(rd, rd));
                    }
                } else if fp {
                    if quad(self.f, c.lhs) {
                        return Err(Error::Unsupported);
                    }
                    self.push(e::fcmp(a, b, fkind(self.f, c.lhs, self.caps)));
                    self.push(e::cset(rd, float_cond(c.op)));
                } else {
                    self.push(if wide(self.f, c.lhs) {
                        e::cmp64(a, b)
                    } else {
                        e::cmp(a, b)
                    });
                    self.push(e::cset(rd, int_cond(c.op, signed(self.f, c.lhs))));
                }
                self.finish(v, rd)?;
            }
            Opcode::Select(s) => {
                let v = result.ok_or(Error::Unsupported)?;
                let rd = self.result(v)?;
                if vector(self.f, v) {
                    let c = self.read(s.cond, reg(24))?;
                    let a = self.read(s.then, reg(25))?;
                    let b = self.read(s.else_, reg(26))?;
                    self.push(e::mov_vec(reg(27), c));
                    self.push(e::bsl_vec(reg(27), a, b));
                    self.push(e::mov_vec(rd, reg(27)));
                } else {
                    if quad(self.f, v) {
                        return Err(Error::Unsupported);
                    }
                    let c = self.read(s.cond, reg(13))?;
                    let fp = class(self.f, v) == 1;
                    let a = self.read(s.then, reg(if fp { 24 } else { 14 }))?;
                    let b = self.read(s.else_, reg(if fp { 25 } else { 16 }))?;
                    self.push(e::cmp(c, Reg::Zr));
                    self.push(if fp {
                        e::fcsel(rd, a, b, Cond::Ne, fkind(self.f, v, self.caps))
                    } else {
                        e::csel(rd, a, b, Cond::Ne) | if wide(self.f, v) { 1 << 31 } else { 0 }
                    });
                }
                self.finish(v, rd)?;
            }
            Opcode::Convert(c) => self.convert(result.ok_or(Error::Unsupported)?, c.value)?,
            Opcode::Unary(u) => {
                let v = result.ok_or(Error::Unsupported)?;
                let rd = self.result(v)?;
                let sc = class(self.f, u.value);
                let dc = class(self.f, v);
                let a = self.read(u.value, reg(if sc == 1 { 24 } else { 13 }))?;
                if quad(self.f, v) || quad(self.f, u.value) {
                    return Err(Error::Unsupported);
                }
                if vector(self.f, v) {
                    if u.op != UnaryOp::Sqrt || !lane(self.f, v)? {
                        return Err(Error::Unsupported);
                    }
                    self.push(e::fsqrt_vec(rd, a));
                } else if u.op == UnaryOp::Reinterpret {
                    if half(self.f, v) && sc == 0 {
                        self.push(e::fmov_hfrom_gpr(rd, a));
                        if !self.caps.fp16 {
                            self.push(e::fcvt_sfrom_h(rd, rd));
                        }
                    } else if half(self.f, u.value) && dc == 0 {
                        let src = if self.caps.fp16 {
                            a
                        } else {
                            self.push(e::fcvt_hfrom_s(reg(27), a));
                            reg(27)
                        };
                        self.push(e::fmov_to_gpr(rd, src, false));
                    } else {
                        self.push(match (sc, dc) {
                            (0, 1) => e::fmov_from_gpr(rd, a, double(self.f, v)),
                            (1, 0) => e::fmov_to_gpr(rd, a, double(self.f, u.value)),
                            (1, 1) => e::fmov_reg(rd, a),
                            _ => e::mov(rd, a),
                        });
                    }
                } else {
                    if half(self.f, v) || dc != 1 {
                        return Err(Error::Unsupported);
                    }
                    let d = double(self.f, v);
                    self.push(match u.op {
                        UnaryOp::Sqrt => e::fsqrt(rd, a, d),
                        UnaryOp::Ceil => e::frintp(rd, a, d),
                        UnaryOp::Floor => e::frintm(rd, a, d),
                        UnaryOp::Trunc => e::frintz(rd, a, d),
                        UnaryOp::Nearest => e::frintn(rd, a, d),
                        _ => return Err(Error::Unsupported),
                    });
                }
                self.finish(v, rd)?;
            }
            Opcode::Alloca(_) => {
                let v = result.ok_or(Error::Unsupported)?;
                let rd = self.result(v)?;
                self.frame_imm(
                    false,
                    rd,
                    Reg::Zr,
                    self.alloca[v.index()].ok_or(Error::Unsupported)?,
                )?;
                self.finish(v, rd)?;
            }
            Opcode::Load(l) => {
                let v = result.ok_or(Error::Unsupported)?;
                let a = self.read(l.ptr, reg(13))?;
                let rd = self.result(v)?;
                self.memory(true, v, rd, a, self.fold.off_of(inst) as usize)?;
                self.finish(v, rd)?;
            }
            Opcode::Store(s) => {
                let a = self.read(
                    s.value,
                    reg(if class(self.f, s.value) == 1 { 24 } else { 13 }),
                )?;
                let p = self.read(s.ptr, reg(14))?;
                self.memory(false, s.value, a, p, self.fold.off_of(inst) as usize)?;
            }
            Opcode::Prefetch(p) => {
                let a = self.read(p.ptr, reg(13))?;
                self.push(e::prfm(a));
            }
            Opcode::GlobalAddr(g) => {
                let v = result.ok_or(Error::Unsupported)?;
                let rd = self.result(v)?;
                let name = self.f.symbol_name(g.symbol);
                if self.native {
                    let addr = self.symbol(name)?;
                    self.constant(rd, addr);
                } else {
                    self.relocs.push(Reloc {
                        offset: self.code.len(),
                        symbol: name.into(),
                        kind: if g.via_got {
                            RelocKind::GotPg
                        } else {
                            RelocKind::AdrpPg
                        },
                    });
                    self.push(e::adrp(rd, 0));
                    self.relocs.push(Reloc {
                        offset: self.code.len(),
                        symbol: name.into(),
                        kind: if g.via_got {
                            RelocKind::GotLo12
                        } else {
                            RelocKind::AddPgOff
                        },
                    });
                    self.push(if g.via_got {
                        e::ldr_off(rd, rd, 0)
                    } else {
                        e::add_imm64(rd, rd, 0)
                    });
                }
                self.finish(v, rd)?;
            }
            Opcode::Call(c) => {
                self.setup_args(self.f.value_list(c.args), c.sret)?;
                let name = self.f.symbol_name(c.symbol);
                if self.native {
                    let addr = self.symbol(name)?;
                    self.constant(reg(16), addr);
                    self.push(e::blr(reg(16)));
                } else {
                    self.relocs.push(Reloc::call(self.code.len(), name));
                    self.push(e::bl(0));
                }
                self.call_result(result, c.ret_dest, &c.ret_pieces, c.ret_regs)?;
            }
            Opcode::CallIndirect(c) => {
                // Keep the target out of both the ABI registers and the argument
                // reload scratch until BLR has consumed it.
                let a = self.read(c.target, reg(17))?;
                self.push(e::mov(reg(17), a));
                self.setup_args(self.f.value_list(c.args), c.sret)?;
                self.push(e::blr(reg(17)));
                self.call_result(result, c.ret_dest, &c.ret_pieces, c.ret_regs)?;
            }
            Opcode::Extract(x) => {
                let v = result.ok_or(Error::Unsupported)?;
                let rd = self.result(v)?;
                let fp = lane(self.f, x.aggregate)?;
                if x.index >= 4 {
                    return Err(Error::Unsupported);
                }
                let a = self.read(x.aggregate, reg(24))?;
                self.push(if fp {
                    e::dup_lane(rd, a, x.index as u8)
                } else {
                    e::umov_lane(rd, a, x.index as u8)
                });
                self.finish(v, rd)?;
            }
            Opcode::StructNew(s) => {
                let v = result.ok_or(Error::Unsupported)?;
                let fp = lane(self.f, v)?;
                let fields = self.f.value_list(s.fields);
                if fields.len() != 4 {
                    return Err(Error::Unsupported);
                }
                let rd = self.result(v)?;
                for n in 0..4 {
                    let a = self.read(fields[n], reg(if fp { 24 } else { 13 }))?;
                    self.push(if fp {
                        e::ins_lane(reg(27), n as u8, a)
                    } else {
                        e::ins_lane_from_gpr(reg(27), n as u8, a)
                    });
                }
                self.push(e::mov_vec(rd, reg(27)));
                self.finish(v, rd)?;
            }
            Opcode::Splat(s) => {
                let v = result.ok_or(Error::Unsupported)?;
                let fp = lane(self.f, v)?;
                let a = self.read(s.scalar, reg(if fp { 24 } else { 13 }))?;
                let rd = self.result(v)?;
                self.push(if fp {
                    e::dup_vec_lane(rd, a, 0)
                } else {
                    e::dup_from_gpr(rd, a)
                });
                self.finish(v, rd)?;
            }
            Opcode::Reduce(r) => {
                let v = result.ok_or(Error::Unsupported)?;
                if r.op != BinOp::Add {
                    return Err(Error::Unsupported);
                }
                let fp = lane(self.f, r.vector)?;
                let a = self.read(r.vector, reg(24))?;
                let rd = self.result(v)?;
                if fp {
                    self.push(e::faddp_vec(reg(25), a, a));
                    self.push(e::faddp_scalar(rd, reg(25)));
                } else {
                    self.push(e::addv(reg(25), a));
                    self.push(e::umov_lane(rd, reg(25), 0));
                }
                self.finish(v, rd)?;
            }
            Opcode::Dot(d) => {
                let v = result.ok_or(Error::Unsupported)?;
                let a = self.read(d.acc, reg(24))?;
                let b = self.read(d.a, reg(25))?;
                let c = self.read(d.b, reg(26))?;
                self.push(e::mov_vec(reg(27), a));
                let si = match kind(self.f, d.a) {
                    TypeKind::Vector(x) => {
                        matches!(self.f.types.type_kind(x.elem),TypeKind::Int(i) if i.signed)
                    }
                    _ => return Err(Error::Unsupported),
                };
                self.push(if si {
                    e::sdot(reg(27), b, c)
                } else {
                    e::udot(reg(27), b, c)
                });
                let rd = self.result(v)?;
                self.push(e::mov_vec(rd, reg(27)));
                self.finish(v, rd)?;
            }
            Opcode::VaStart(v) => self.va_start(v.list)?,
            Opcode::VaArg(a) => self.va_arg(a.list, result.ok_or(Error::Unsupported)?)?,
            Opcode::VaEnd(_) => {}
            Opcode::If(cf) => {
                let c = self.read(cf.cond, reg(13))?;
                let other = self.local_label();
                self.branch(other, e::cbz(c, 0));
                self.edge(cf.then)?;
                self.bind(other);
                self.edge(cf.else_)?;
                return Ok(true);
            }
            // GPU-only memory synchronization/atomics and unexpanded matrix ops
            // have no source AArch64 lowering, and fail closed here too.
            _ => return Err(Error::Unsupported),
        }
        Ok(false)
    }
    fn ret(&mut self, r: &Ret) -> Result<(), Error> {
        let mut bank = [0u16; 2];
        let mut moves = Vec::new();
        for v in r.slice() {
            let c = class(self.f, *v);
            let n = bank[c as usize];
            bank[c as usize] += 1;
            if (c == 0 && n >= 2) || (c == 1 && n >= 4) {
                return Err(Error::Unsupported);
            }
            moves.push(Move {
                src: self.loc(*v)?,
                dst: Location::Reg(n),
                class: c,
                value: *v,
            });
        }
        for m in wimmer::order_moves::<Function, Model>(&moves, self.model) {
            self.move_one(m)?;
        }
        for v in r.slice() {
            if half(self.f, *v) && !self.caps.fp16 {
                let n = r
                    .slice()
                    .iter()
                    .take_while(|x| *x != v)
                    .filter(|x| class(self.f, **x) == 1)
                    .count() as u16;
                self.push(e::fcvt_hfrom_s(reg(n), reg(n)));
            }
        }
        let saved = self.saved.clone();
        for (c, r, o) in saved {
            if c == 1 {
                let (b, o) = self.address(Reg::Zr, o, 8)?;
                self.push(e::ldr_fp(reg(r), b, o, true));
            } else {
                let (b, o) = self.address(Reg::Zr, o, 8)?;
                self.push(e::ldr_off(reg(r), b, o));
            }
        }
        if self.frame != 0 {
            self.frame_imm(false, Reg::Zr, Reg::Zr, self.frame)?;
        }
        self.push(e::ret());
        Ok(())
    }
}
fn int_cond(op: CmpOp, si: bool) -> Cond {
    match op {
        CmpOp::Eq => Cond::Eq,
        CmpOp::Ne => Cond::Ne,
        CmpOp::Lt => {
            if si {
                Cond::Lt
            } else {
                Cond::Lo
            }
        }
        CmpOp::Le => {
            if si {
                Cond::Le
            } else {
                Cond::Ls
            }
        }
        CmpOp::Gt => {
            if si {
                Cond::Gt
            } else {
                Cond::Hi
            }
        }
        CmpOp::Ge => {
            if si {
                Cond::Ge
            } else {
                Cond::Hs
            }
        }
    }
}
fn float_cond(op: CmpOp) -> Cond {
    match op {
        CmpOp::Eq => Cond::Eq,
        CmpOp::Ne => Cond::Ne,
        CmpOp::Lt => Cond::Mi,
        CmpOp::Le => Cond::Ls,
        CmpOp::Gt => Cond::Gt,
        CmpOp::Ge => Cond::Ge,
    }
}

pub(super) fn compile(input: &Function, caps: &ModelCaps, native: bool) -> Result<Compiled, Error> {
    if input.block_count() == 0 || volt_ir::function::function_uses_composite_f16(input) {
        return Err(Error::Unsupported);
    }
    let mut f = input.clone_func();
    volt_ir::expand::expand_nv_fp4(&mut f);
    volt_ir::expand::expand_low_float(&mut f);
    volt_ir::expand::expand_vector_lanes_except(&mut f, &|f, i| match f.opcode(i) {
        Opcode::Reduce(r) => r.op == BinOp::Add && lane(f, r.vector).is_ok(),
        Opcode::Splat(_) => f.inst_result(i).is_some_and(|v| lane(f, v).is_ok()),
        _ => false,
    });
    volt_ir::softfp::lower(&mut f);
    volt_ir::critical_edge::split_critical_edges(&mut f);
    volt_ir::reachable::neutralize_unreachable(&mut f);
    let fold = crate::regalloc::addrfold::analyze(&f, |f, i| {
        let (p, v) = match f.opcode(i) {
            Opcode::Load(l) => (l.ptr, f.inst_result(i)?),
            Opcode::Store(s) => (s.ptr, s.value),
            _ => return None,
        };
        let add = match f.opcode(f.defining_inst(p)?) {
            Opcode::ArithImm(a) => a,
            _ => return None,
        };
        let (sz, _) = layout(f, f.value_type(v)).ok()?;
        let scale = sz.next_power_of_two();
        if add.imm >= 0
            && add.imm as usize % scale == 0
            && add.imm as usize / scale <= 4095
            && add.imm <= u16::MAX as i64
        {
            Some(add.imm)
        } else {
            None
        }
    });
    for (i, a) in &fold.folds {
        match f.opcode_mut(*i) {
            Opcode::Load(l) => l.ptr = a.base,
            Opcode::Store(s) => s.ptr = a.base,
            _ => {}
        }
    }
    for bi in 0..f.block_count() {
        f.block_insts_mut(Block(bi as u32))
            .retain(|i| !fold.is_dead_add(*i));
    }
    let model = Model::new(&f, caps);
    let alloc = wimmer::allocate(&f, &model).map_err(|_| Error::Unsupported)?;
    let mut outgoing = 0;
    let mut leaf = true;
    for bi in 0..f.block_count() {
        for i in f.block_insts(Block(bi as u32)) {
            match f.opcode(*i) {
                Opcode::Call(c) => {
                    leaf = false;
                    outgoing =
                        outgoing.max(arg_locations(&f, f.value_list(c.args), c.sret, caps.abi)?.1);
                }
                Opcode::CallIndirect(c) => {
                    leaf = false;
                    outgoing =
                        outgoing.max(arg_locations(&f, f.value_list(c.args), c.sret, caps.abi)?.1);
                }
                _ => {}
            }
        }
    }
    let mut saved = Vec::new();
    let mut cursor = outgoing;
    for r in &alloc.used_callee_saved {
        saved.push((r.class, r.reg, cursor));
        cursor += 8;
    }
    if !leaf {
        saved.push((0, 30, cursor));
        cursor += 8;
    }
    let gr_save = align(cursor, 8);
    let vr_save = align(gr_save + if f.is_variadic { 64 } else { 0 }, 16);
    cursor = vr_save + if f.is_variadic { 128 } else { 0 };
    let slot_base = align(cursor, 16);
    cursor = slot_base
        + 16 * alloc
            .slot_count_per_class
            .iter()
            .map(|n| *n as usize)
            .sum::<usize>();
    let mut alloca = vec![None; f.value_count()];
    for bi in 0..f.block_count() {
        for i in f.block_insts(Block(bi as u32)) {
            if let Opcode::Alloca(a) = f.opcode(*i) {
                let (sz, al) = layout(&f, a.elem)?;
                cursor = align(cursor, al);
                alloca[f.inst_result(*i).ok_or(Error::Unsupported)?.index()] = Some(cursor);
                cursor += sz;
            }
        }
    }
    let frame = align(cursor, 16);
    let mut em = Emitter {
        f: &f,
        caps,
        native,
        model: &model,
        alloc,
        code: Vec::new(),
        relocs: Vec::new(),
        lines: Vec::new(),
        labels: vec![0; f.block_count()],
        fixups: Vec::new(),
        pos: 0,
        block: Block(0),
        slot_base,
        frame,
        saved,
        alloca,
        gr_save,
        vr_save,
        fold,
    };
    if frame != 0 {
        em.frame_imm(true, Reg::Zr, Reg::Zr, frame)?;
    }
    for (c, r, o) in em.saved.clone() {
        let (b, o) = em.address(Reg::Zr, o, 8)?;
        em.push(if c == 1 {
            e::str_fp(reg(r), b, o, true)
        } else {
            e::str_off(reg(r), b, o)
        });
    }
    if f.is_variadic {
        for n in 0..8 {
            em.raw_mem(false, 0, reg(n), Reg::Zr, gr_save + 8 * n as usize)?;
            em.raw_mem(false, 1, reg(n), Reg::Zr, vr_save + 16 * n as usize)?;
        }
    }
    let params = f.block_params(Block(0));
    let (locs, _) = arg_locations(&f, params, f.sret, caps.abi)?;
    for (v, l) in params.iter().zip(locs) {
        let rd = em.result(*v)?;
        let c = class(&f, *v);
        match l {
            Arg::Reg(r) => {
                if half(&f, *v) && !caps.fp16 {
                    em.push(e::fcvt_sfrom_h(rd, reg(r)));
                } else {
                    em.push(if c == 1 {
                        e::mov_vec(rd, reg(r))
                    } else {
                        e::mov(rd, reg(r))
                    });
                }
            }
            Arg::Stack(o) => em.memory(true, *v, rd, Reg::Zr, frame + o)?,
        }
        em.finish(*v, rd)?;
    }
    let mut action = 0;
    for bi in 0..f.block_count() {
        let b = Block(bi as u32);
        em.block = b;
        if f.block_insts(b).is_empty() && f.terminator(b).is_none() {
            em.labels[bi] = em.code.len();
            em.pos += 2;
            continue;
        }
        if caps.fetch_align > 4 {
            let mut header = false;
            for pi in bi..f.block_count() {
                let pb = Block(pi as u32);
                for i in f.block_insts(pb) {
                    if let Opcode::If(cf) = f.opcode(*i) {
                        header |= cf.then.target == b || cf.else_.target == b;
                    }
                }
                if let Some(Terminator::Jump(j)) = f.terminator(pb) {
                    header |= j.target == b;
                }
            }
            if header {
                while em.code.len() * 4 % caps.fetch_align as usize != 0 {
                    em.push(e::nop());
                }
            }
        }
        em.labels[bi] = em.code.len();
        em.pos += 1;
        let mut terminated = false;
        for i in f.block_insts(b) {
            em.actions(&mut action)?;
            let mut attrs = f.attributes_of(AttrTarget::Inst(*i));
            while let Some(a) = attrs.next_attr() {
                if let Attribute::Custom(c) = a {
                    if c.namespace == "debug" && c.key == "line" {
                        if let AttrValue::Int(line) = c.value {
                            if line >= 0 && em.lines.last().is_none_or(|l| l.line != line as u32) {
                                em.lines.push(LineEntry {
                                    offset: (em.code.len() * 4) as u32,
                                    line: line as u32,
                                });
                            }
                        }
                    }
                }
            }
            terminated |= em.instruction(*i)?;
            em.pos += 1;
        }
        em.actions(&mut action)?;
        if !terminated {
            match f.terminator(b).unwrap_or(Terminator::Ret(Ret::none())) {
                Terminator::Ret(r) => em.ret(&r)?,
                Terminator::Jump(j) => em.edge(j)?,
            }
        }
        em.pos += 1;
    }
    peephole::pair_memory(
        &mut em.code,
        &mut em.labels,
        &mut em.fixups,
        &mut em.relocs,
        &mut em.lines,
    );
    for fix in em.fixups {
        let delta = (em.labels[fix.target as usize] as i64 - fix.at as i64) * 4;
        let word = em.code[fix.at];
        em.code[fix.at] = if word & 0x7c00_0000 == 0x1400_0000 {
            if !(-(1 << 27)..1 << 27).contains(&delta) {
                return Err(Error::Unsupported);
            }
            e::b(delta as i32)
        } else {
            if !(-(1 << 20)..1 << 20).contains(&delta) {
                return Err(Error::Unsupported);
            }
            (word & !0x00ff_ffe0) | (((delta >> 2) as u32 & 0x7ffff) << 5)
        };
    }
    Ok(Compiled {
        code: em.code,
        relocs: em.relocs,
        lines: em.lines,
    })
}
