// Ported and modified from Vulcan, Copyright 2026 LilithSemi (Apache-2.0).
// Rust Function-based lowering and executable/object integration modifications.
use super::super::{encode as e, peephole};
use super::{Abi, Compiled, Error, Fixup, LineEntry, ModelCaps, Reloc, RelocKind};
use crate::regalloc::forward;
use crate::regalloc::wimmer::{
    self, Allocation, CallSite, ClassRegs, FixedAssign, Location, Move, RegClass, RegDescription,
    UseKind,
};
use alloc::{vec, vec::Vec};
use e::{Cond, FKind, Reg};
use volt_ir::attribute::{AttrValue, Attribute, Endianness};
use volt_ir::function::{
    AtomicOrdering, AttrTarget, BinOp, Block, CmpOp, ConvertKind, Function, Inst, Jump, MemFlags,
    Opcode, Ret, Terminator, UnaryOp, Value,
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
/// Integer values of fewer than 64 bits are held zero-extended in registers; signedness is a
/// property of the operation, which sign-extends its operands on demand (`signed_operand`).
fn bits(f: &Function, v: Value) -> u16 {
    match kind(f, v) {
        TypeKind::Int(i) => i.bits,
        TypeKind::Bool => 1,
        _ => 64,
    }
}
/// `n` as `Emitter::normalize` would leave a value of `v`'s type in a register.
fn canonical_constant(f: &Function, v: Value, n: u64) -> u64 {
    let bits = u32::from(bits(f, v));
    if bits >= 64 {
        n
    } else {
        n & ((1u64 << bits) - 1)
    }
}
/// Width in bytes of an `ldar`/`stlr` access to a value of `v`'s type: whole-byte integers up to
/// 64 bits that are a power of two wide, and pointers. Other types have no ordered form.
fn ordered_size(f: &Function, v: Value) -> Option<u8> {
    match kind(f, v) {
        TypeKind::Int(i) if matches!(i.bits, 8 | 16 | 32 | 64) => Some((i.bits / 8) as u8),
        TypeKind::Ptr(_) => Some(8),
        _ => None,
    }
}
/// Whether the access is lowered to `ldar`/`stlr`, which only take a bare `[xn]` address, so no
/// offset may be folded into it. A relaxed access is an ordinary `ldr`/`str` and folds freely.
fn needs_bare_address(mem: &MemFlags) -> bool {
    matches!(mem.ordering, Some(o) if o != AtomicOrdering::Relaxed)
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
    /// Instruction index -> position in `fused` (`u32::MAX` if not fused).
    fused_at: Vec<u32>,
    /// Instruction index -> true when a fusion absorbed it and nothing is emitted.
    skipped: Vec<bool>,
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
                if matches!(f.opcode_ref(*i), Opcode::Call(_) | Opcode::CallIndirect(_)) {
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
                    if let (Opcode::Icmp(c), Opcode::If(cf)) = (f.opcode_ref(p), f.opcode_ref(i)) {
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
                let Opcode::Arith(a) = f.opcode_ref(i) else {
                    continue;
                };
                if uses[v.index()] != 1
                    || !matches!(a.op, BinOp::Add | BinOp::Sub)
                    || (a.lhs != v && a.rhs != v)
                {
                    continue;
                }
                let fusion = match f.opcode_ref(p) {
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
        let mut fused_at = vec![u32::MAX; f.inst_count()];
        for (n, (i, _)) in fused.iter().enumerate() {
            // First fusion for an instruction wins.
            if fused_at[i.index()] == u32::MAX {
                fused_at[i.index()] = n as u32;
            }
        }
        let mut skipped_mask = vec![false; f.inst_count()];
        for i in &skipped {
            skipped_mask[i.index()] = true;
        }
        Self {
            classes,
            calls,
            fused,
            fused_at,
            skipped: skipped_mask,
        }
    }
    fn is_skipped(&self, i: Inst) -> bool {
        self.skipped.get(i.index()).copied().unwrap_or(false)
    }
    fn fusion(&self, i: Inst) -> Option<Fusion> {
        match self.fused_at.get(i.index()) {
            Some(&n) if n != u32::MAX => Some(self.fused[n as usize].1),
            _ => None,
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
        if self.is_skipped(i) {
            return Some(0);
        }
        match self.fusion(i)? {
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
                if let Opcode::If(cf) = f.opcode_ref(i) {
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
        let chunk = |x: u64, k: u32| (x >> (k * 16)) as u16;
        let nonzero = |x: u64| (0..4).filter(|k| chunk(x, *k) != 0).count();
        if nonzero(v) > 1 {
            // One instruction instead of a movz/movk run: a lone non-0xffff
            // chunk (movn) or a repeated, rotated bit run (logical immediate).
            if nonzero(!v) == 1 {
                let k = (0..4).find(|k| chunk(!v, *k) != 0).unwrap_or(0);
                self.push(e::movn64(r, chunk(!v, k), k as u8));
                return;
            }
            if let Some(encoded) = e::logical_imm(v, true) {
                self.push(e::orr_imm(r, Reg::Zr, encoded, true));
                return;
            }
        }
        self.push(e::movz64(r, v as u16, 0));
        for k in 1..4 {
            let p = chunk(v, k);
            if p != 0 {
                self.push(e::movk64(r, p, k as u8));
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
    /// The base register for an `ldar`/`stlr` of `[base + off]`: these have no offset form, so
    /// a non-zero `off` is added into the x8 scratch first.
    fn bare_base(&mut self, base: Reg, off: usize) -> Result<Reg, Error> {
        if off == 0 {
            return Ok(base);
        }
        self.frame_imm(false, Reg::X8, base, off)?;
        Ok(Reg::X8)
    }
    fn loc(&self, v: Value) -> Result<Location, Error> {
        if let Some(&r) = self.alloc.single_regs.get(v.index())
            && r != u16::MAX
        {
            return Ok(Location::Reg(r));
        }
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
        self.spill_result(v, r)
    }
    /// Like `finish`, for a result already in the canonical form `normalize`
    /// would produce (so the extension instruction is skipped).
    fn finish_normal(&mut self, v: Value, r: Reg) -> Result<(), Error> {
        debug_assert!(class(self.f, v) == 0);
        self.spill_result(v, r)
    }
    fn spill_result(&mut self, v: Value, r: Reg) -> Result<(), Error> {
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
    /// Whether `v = lhs op rhs` (rhs a constant when `imm` is given) is already
    /// in canonical form, given canonical operands. Values narrower than 64 bits
    /// are kept zero-extended to the full register. Operations on 32-bit or
    /// narrower types run in 32-bit form, which zeroes the upper half, so a
    /// 32-bit result is always canonical (`SMulh` excepted: it works in 64-bit
    /// form); a narrower result only is if the operation cannot set bits above
    /// the type.
    fn keeps_canonical_form(&self, v: Value, op: BinOp, imm: Option<i64>) -> bool {
        let n = bits(self.f, v);
        if class(self.f, v) != 0 || n >= 64 || wide(self.f, v) || op == BinOp::SMulh {
            return false;
        }
        if n == 32 {
            return true;
        }
        let in_range = imm.is_none_or(|k| k >= 0 && (k as u64) >> n == 0);
        matches!(
            op,
            BinOp::BitAnd
                | BinOp::BitOr
                | BinOp::BitXor
                | BinOp::Shr
                | BinOp::UDiv
                | BinOp::URem
                | BinOp::UMulh
        ) && in_range
    }
    /// `rd = rn op imm` as a single instruction, if the operation has an
    /// immediate form for this operand width.
    fn immediate_form(&self, v: Value, op: BinOp, rd: Reg, rn: Reg, imm: i64) -> Option<u32> {
        let is64 = wide(self.f, v);
        let width = if is64 { 64 } else { 32 };
        match op {
            BinOp::Shl | BinOp::Shr | BinOp::Sar if (0..width).contains(&imm) => {
                let n = bits(self.f, v);
                Some(match op {
                    BinOp::Shl => e::lsl_imm(rd, rn, imm as u32, is64),
                    BinOp::Shr => e::lsr_imm(rd, rn, imm as u32, is64),
                    _ if n == 32 || n == 64 => e::asr_imm(rd, rn, imm as u32, is64),
                    // A zero-extended narrow value: extract its bitfield from `imm` up,
                    // sign-extended from bit `n - 1`.
                    _ => e::sbfm(rd, rn, imm.min(i64::from(n) - 1) as u8, (n - 1) as u8),
                })
            }
            BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor => {
                let value = if is64 {
                    imm as u64
                } else {
                    imm as u64 & 0xffff_ffff
                };
                let encoded = e::logical_imm(value, is64)?;
                Some(match op {
                    BinOp::BitAnd => e::and_imm(rd, rn, encoded, is64),
                    BinOp::BitOr => e::orr_imm(rd, rn, encoded, is64),
                    _ => e::eor_imm(rd, rn, encoded, is64),
                })
            }
            _ => None,
        }
    }
    fn normalize(&mut self, v: Value, r: Reg) {
        let n = bits(self.f, v);
        if n < 64 {
            self.push(e::ubfm(r, r, 0, (n - 1) as u8));
        }
    }
    /// Whether `signed_operand` has to emit an extension for a value of `v`'s
    /// type. Narrow operations read only the low 32 bits of a register, so a
    /// 32-bit value needs none unless the whole register is read (`full`).
    fn needs_sign_extension(&self, v: Value, full: bool) -> bool {
        let n = bits(self.f, v);
        n < 64 && (n != 32 || full)
    }
    /// `r`, which holds `v` zero-extended, as a sign-extended value for an
    /// operation that reads its sign: `r` itself when no extension is needed,
    /// else `scratch` after an extension into it.
    fn signed_operand(&mut self, v: Value, r: Reg, scratch: Reg, full: bool) -> Reg {
        if !self.needs_sign_extension(v, full) {
            return r;
        }
        self.push(e::sbfm(scratch, r, 0, (bits(self.f, v) - 1) as u8));
        scratch
    }
    /// Sets the flags for an integer compare of `a` and `b` (holding `lhs`'s
    /// type) and returns the condition that `op` tests.
    fn compare_flags(&mut self, op: CmpOp, lhs: Value, a: Reg, b: Reg) -> Result<Cond, Error> {
        let cond = int_cond(op)?;
        let (a, b) = if matches!(op, CmpOp::Slt | CmpOp::Sle | CmpOp::Sgt | CmpOp::Sge) {
            (
                self.signed_operand(lhs, a, reg(16), false),
                self.signed_operand(lhs, b, reg(17), false),
            )
        } else {
            (a, b)
        };
        self.push(if wide(self.f, lhs) {
            e::cmp64(a, b)
        } else {
            e::cmp(a, b)
        });
        Ok(cond)
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
            match (load, size) {
                (true, 1) => e::ldrb_off(r, base, off),
                (true, 2) => e::ldrh_off(r, base, off),
                (true, 3..=4) => e::ldr_w(r, base, off),
                (true, 5..=8) => e::ldr_off(r, base, off),
                (false, 1) => e::strb_off(r, base, off),
                (false, 2) => e::strh_off(r, base, off),
                (false, 3..=4) => e::str_w(r, base, off),
                (false, 5..=8) => e::str_off(r, base, off),
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
        let w = match op {
            BinOp::Add => e::add(rd, a, b),
            BinOp::Sub => e::sub(rd, a, b),
            BinOp::Mul => e::mul(rd, a, b),
            BinOp::UMulh | BinOp::SMulh => {
                let n = bits(self.f, v);
                if n == 64 {
                    self.push(if op == BinOp::SMulh {
                        e::smulh(rd, a, b)
                    } else {
                        e::umulh(rd, a, b)
                    });
                    return Ok(());
                }
                // The product of two values of up to 32 bits fits a register.
                if n > 32 {
                    return Err(Error::Unsupported);
                }
                if op == BinOp::SMulh {
                    let a = self.signed_operand(v, a, reg(16), true);
                    let b = self.signed_operand(v, b, reg(17), true);
                    self.push(e::mul64(rd, a, b));
                    self.push(e::asr_imm(rd, rd, u32::from(n), true));
                } else {
                    self.push(e::mul64(rd, a, b));
                    self.push(e::lsr_imm(rd, rd, u32::from(n), true));
                }
                return Ok(());
            }
            BinOp::UDiv => e::udiv(rd, a, b),
            BinOp::SDiv => {
                let a = self.signed_operand(v, a, reg(16), false);
                let b = self.signed_operand(v, b, reg(17), false);
                e::sdiv(rd, a, b)
            }
            BinOp::URem => {
                self.push(sf | e::udiv(reg(16), a, b));
                self.push(sf | e::msub(rd, reg(16), b, a));
                return Ok(());
            }
            BinOp::SRem => {
                // Extended operands live in x16/x17, so the quotient can use `rd`.
                let extend = self.needs_sign_extension(v, false);
                let a = self.signed_operand(v, a, reg(16), false);
                let b = self.signed_operand(v, b, reg(17), false);
                let q = if extend { rd } else { reg(16) };
                self.push(sf | e::sdiv(q, a, b));
                self.push(sf | e::msub(rd, q, b, a));
                return Ok(());
            }
            BinOp::BitAnd => e::andr(rd, a, b),
            BinOp::BitOr => e::orr(rd, a, b),
            BinOp::BitXor => e::eor(rd, a, b),
            BinOp::Shl => e::lslv(rd, a, b),
            BinOp::Shr => e::lsrv(rd, a, b),
            BinOp::Sar => {
                let a = self.signed_operand(v, a, reg(16), false);
                e::asrv(rd, a, b)
            }
            // Floating-point only.
            BinOp::Div | BinOp::Rem => return Err(Error::Unsupported),
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
        if !fp && !vector(self.f, v) && self.keeps_canonical_form(v, op, None) {
            return self.finish_normal(v, rd);
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
    fn convert(&mut self, v: Value, src: Value, kind: ConvertKind) -> Result<(), Error> {
        if quad(self.f, v) || quad(self.f, src) || vector(self.f, v) || vector(self.f, src) {
            return Err(Error::Unsupported);
        }
        let sc = class(self.f, src);
        let dc = class(self.f, v);
        let a = self.read(src, reg(if sc == 1 { 24 } else { 13 }))?;
        let rd = self.result(v)?;
        match kind {
            ConvertKind::SiToFp | ConvertKind::UiToFp => {
                if (sc, dc) != (0, 1) {
                    return Err(Error::Unsupported);
                }
                let si = kind == ConvertKind::SiToFp;
                let a = if si {
                    self.signed_operand(src, a, reg(16), false)
                } else {
                    a
                };
                let sf = if wide(self.f, src) { 1 << 31 } else { 0 };
                self.push(sf | e::cvt_int_to_float(rd, a, fkind(self.f, v, self.caps), si));
                if half(self.f, v) && !self.caps.fp16 {
                    self.half_round(rd);
                }
            }
            ConvertKind::FpToSi | ConvertKind::FpToUi => {
                if (sc, dc) != (1, 0) {
                    return Err(Error::Unsupported);
                }
                // A destination narrower than 32 bits keeps the low bits of the 32-bit
                // saturating conversion (`finish` truncates).
                let sf = if wide(self.f, v) { 1 << 31 } else { 0 };
                self.push(
                    sf | e::cvt_float_to_int(
                        rd,
                        a,
                        fkind(self.f, src, self.caps),
                        kind == ConvertKind::FpToSi,
                    ),
                );
            }
            ConvertKind::FpResize => {
                if (sc, dc) != (1, 1) {
                    return Err(Error::Unsupported);
                }
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
            ConvertKind::Trunc | ConvertKind::Zext | ConvertKind::Sext => {
                if (sc, dc) != (0, 0) {
                    return Err(Error::Unsupported);
                }
                let n = bits(self.f, src);
                let m = bits(self.f, v);
                if kind == ConvertKind::Sext && m > n && n < 64 {
                    let word = e::sbfm(rd, a, 0, (n - 1) as u8);
                    if m == 32 {
                        // The 32-bit form clears the upper half: already canonical.
                        self.push(word & !(1 << 31 | 1 << 22));
                        return self.finish_normal(v, rd);
                    }
                    self.push(word);
                } else {
                    // A zero-extension of a canonical value, a truncation (`finish`
                    // clears the bits above `m`) or a same-width copy.
                    self.push(e::mov(rd, a));
                    if kind == ConvertKind::Zext {
                        return self.finish_normal(v, rd);
                    }
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
        if self.model.is_skipped(inst) {
            return Ok(false);
        }
        if let Some(fusion) = self.model.fusion(inst) {
            if let Fusion::Compare { lhs, rhs, op } = fusion {
                let a = self.read(lhs, reg(13))?;
                let b = self.read(rhs, reg(14))?;
                let cond = self.compare_flags(op, lhs, a, b)?;
                let Opcode::If(cf) = self.f.opcode_ref(inst) else {
                    return Err(Error::Unsupported);
                };
                let other = self.local_label();
                self.branch(other, e::bcc(e::invert(cond), 0));
                self.edge(cf.then)?;
                self.bind(other);
                self.edge(cf.else_)?;
                return Ok(true);
            }
            self.fusion(result.ok_or(Error::Unsupported)?, fusion)?;
            return Ok(false);
        }
        match self.f.opcode_ref(inst) {
            Opcode::Iconst(n) => {
                let v = result.ok_or(Error::Unsupported)?;
                let rd = self.result(v)?;
                if class(self.f, v) == 1 {
                    self.constant(reg(16), *n as u64);
                    self.push(e::fmov_from_gpr(rd, reg(16), double(self.f, v)));
                    self.finish(v, rd)?;
                } else {
                    // Canonicalize at compile time instead of emitting an extension.
                    self.constant(rd, canonical_constant(self.f, v, *n as u64));
                    self.finish_normal(v, rd)?;
                }
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
                            (*n as f32).to_bits() as u64
                        },
                    );
                    self.push(e::fmov_from_gpr(rd, reg(16), double(self.f, v)));
                }
                self.finish(v, rd)?;
            }
            Opcode::Fconst128(n) => {
                let v = result.ok_or(Error::Unsupported)?;
                let rd = self.result(v)?;
                self.constant(reg(16), *n as u64);
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
                } else if let Some(word) = self.immediate_form(v, a.op, rd, r, a.imm) {
                    self.push(word);
                } else {
                    self.constant(reg(14), canonical_constant(self.f, v, a.imm as u64));
                    self.integer_binary(v, a.op, rd, r, reg(14))?;
                }
                if self.keeps_canonical_form(v, a.op, Some(a.imm)) {
                    self.finish_normal(v, rd)?;
                } else {
                    self.finish(v, rd)?;
                }
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
                        _ => return Err(Error::Unsupported),
                    });
                    if c.op == CmpOp::Ne {
                        self.push(e::mvn_vec(rd, rd));
                    }
                } else if fp {
                    if quad(self.f, c.lhs) {
                        return Err(Error::Unsupported);
                    }
                    let cond = float_cond(c.op)?;
                    self.push(e::fcmp(a, b, fkind(self.f, c.lhs, self.caps)));
                    self.push(e::cset(rd, cond));
                } else {
                    let cond = self.compare_flags(c.op, c.lhs, a, b)?;
                    self.push(e::cset(rd, cond));
                }
                // `cset` already yields 0 or 1.
                if !fp && !vector(self.f, c.lhs) && bits(self.f, v) == 1 {
                    self.finish_normal(v, rd)?;
                } else {
                    self.finish(v, rd)?;
                }
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
                // `csel` copies one of two canonical operands, so the result
                // needs no extension.
                if class(self.f, v) == 0 && !vector(self.f, v) {
                    self.finish_normal(v, rd)?;
                } else {
                    self.finish(v, rd)?;
                }
            }
            Opcode::Convert(c) => {
                self.convert(result.ok_or(Error::Unsupported)?, c.value, c.kind)?
            }
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
                if l.mem.endian == Endianness::Big {
                    return Err(Error::Unsupported);
                }
                let v = result.ok_or(Error::Unsupported)?;
                let size = ordered_size(self.f, v);
                let acquire = match l.mem.ordering {
                    None => false,
                    // Relaxed: an ordinary, naturally aligned load is single-copy atomic.
                    Some(AtomicOrdering::Relaxed) => {
                        if size.is_none() {
                            return Err(Error::Unsupported);
                        }
                        false
                    }
                    Some(AtomicOrdering::Acquire | AtomicOrdering::SeqCst) => true,
                    Some(AtomicOrdering::Release | AtomicOrdering::AcqRel) => {
                        return Err(Error::Unsupported);
                    }
                };
                let a = self.read(l.ptr, reg(13))?;
                let rd = self.result(v)?;
                if acquire {
                    let size = size.ok_or(Error::Unsupported)?;
                    let base = self.bare_base(a, self.fold.off_of(inst) as usize)?;
                    self.push(e::ldar(rd, base, size));
                } else {
                    self.memory(true, v, rd, a, self.fold.off_of(inst) as usize)?;
                }
                // `ldrb`/`ldrh`/`ldr w`/`ldar*` already zero-extend a whole-byte value.
                let n = bits(self.f, v);
                if class(self.f, v) == 0 && matches!(n, 8 | 16 | 32 | 64) {
                    self.finish_normal(v, rd)?;
                } else {
                    self.finish(v, rd)?;
                }
            }
            Opcode::Store(s) => {
                if s.mem.endian == Endianness::Big {
                    return Err(Error::Unsupported);
                }
                let size = ordered_size(self.f, s.value);
                let release = match s.mem.ordering {
                    None => false,
                    // Relaxed: an ordinary, naturally aligned store is single-copy atomic.
                    Some(AtomicOrdering::Relaxed) => {
                        if size.is_none() {
                            return Err(Error::Unsupported);
                        }
                        false
                    }
                    Some(AtomicOrdering::Release | AtomicOrdering::SeqCst) => true,
                    Some(AtomicOrdering::Acquire | AtomicOrdering::AcqRel) => {
                        return Err(Error::Unsupported);
                    }
                };
                let a = self.read(
                    s.value,
                    reg(if class(self.f, s.value) == 1 { 24 } else { 13 }),
                )?;
                let p = self.read(s.ptr, reg(14))?;
                if release {
                    let size = size.ok_or(Error::Unsupported)?;
                    let base = self.bare_base(p, self.fold.off_of(inst) as usize)?;
                    self.push(e::stlr(a, base, size));
                } else {
                    self.memory(false, s.value, a, p, self.fold.off_of(inst) as usize)?;
                }
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
                if fp {
                    self.finish(v, rd)?;
                } else {
                    // `umov` to a w register zero-extends the 32-bit lane.
                    self.finish_normal(v, rd)?;
                }
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
                if !vector(self.f, d.a) {
                    return Err(Error::Unsupported);
                }
                let si = d.signed;
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
/// The condition code that holds after comparing the operands of an integer
/// (or pointer) `op`; the signed forms need sign-extended operands.
fn int_cond(op: CmpOp) -> Result<Cond, Error> {
    Ok(match op {
        CmpOp::Eq => Cond::Eq,
        CmpOp::Ne => Cond::Ne,
        CmpOp::Slt => Cond::Lt,
        CmpOp::Sle => Cond::Le,
        CmpOp::Sgt => Cond::Gt,
        CmpOp::Sge => Cond::Ge,
        CmpOp::Ult => Cond::Lo,
        CmpOp::Ule => Cond::Ls,
        CmpOp::Ugt => Cond::Hi,
        CmpOp::Uge => Cond::Hs,
        CmpOp::Lt | CmpOp::Le | CmpOp::Gt | CmpOp::Ge => return Err(Error::Unsupported),
    })
}
fn float_cond(op: CmpOp) -> Result<Cond, Error> {
    Ok(match op {
        CmpOp::Eq => Cond::Eq,
        CmpOp::Ne => Cond::Ne,
        CmpOp::Lt => Cond::Mi,
        CmpOp::Le => Cond::Ls,
        CmpOp::Gt => Cond::Gt,
        CmpOp::Ge => Cond::Ge,
        _ => return Err(Error::Unsupported),
    })
}

/// True for a function made only of scalar integer, boolean and pointer
/// operations with no block parameters past the entry block: what a dynamic
/// translator emits. None of the float, vector and quantization lowerings or
/// the critical-edge split (which only matters for edge moves) can change it.
fn is_plain_integer(f: &Function) -> bool {
    for bi in 0..f.block_count() {
        let block = Block(bi as u32);
        if bi != 0 && !f.block_params(block).is_empty() {
            return false;
        }
        for i in f.block_insts(block) {
            if !matches!(
                f.opcode_ref(*i),
                Opcode::Iconst(_)
                    | Opcode::Arith(_)
                    | Opcode::ArithImm(_)
                    | Opcode::Icmp(_)
                    | Opcode::Select(_)
                    | Opcode::Convert(_)
                    | Opcode::Load(_)
                    | Opcode::Store(_)
                    | Opcode::If(_)
            ) {
                return false;
            }
            let mem = match f.opcode_ref(*i) {
                Opcode::Load(l) => Some(&l.mem),
                Opcode::Store(s) => Some(&s.mem),
                _ => None,
            };
            if mem.is_some_and(|m| !m.is_plain()) {
                return false;
            }
        }
    }
    (0..f.value_count()).all(|v| match kind(f, Value(v as u32)) {
        TypeKind::Int(i) => i.bits <= 64,
        TypeKind::Bool | TypeKind::Ptr(_) => true,
        _ => false,
    })
}

pub(super) fn compile(input: &Function, caps: &ModelCaps, native: bool) -> Result<Compiled, Error> {
    if input.block_count() == 0 || volt_ir::function::function_uses_composite_f16(input) {
        return Err(Error::Unsupported);
    }
    compile_owned(input.clone_func(), caps, native)
}

/// Like `compile`, consuming the function so no copy is needed.
pub(super) fn compile_owned(
    mut f: Function,
    caps: &ModelCaps,
    native: bool,
) -> Result<Compiled, Error> {
    if f.block_count() == 0 {
        return Err(Error::Unsupported);
    }
    let plain = is_plain_integer(&f);
    if !plain {
        if volt_ir::function::function_uses_composite_f16(&f) {
            return Err(Error::Unsupported);
        }
        volt_ir::expand::expand_nv_fp4(&mut f);
        volt_ir::expand::expand_low_float(&mut f);
        volt_ir::expand::expand_vector_lanes_except(&mut f, &|f, i| match f.opcode_ref(i) {
            Opcode::Reduce(r) => r.op == BinOp::Add && lane(f, r.vector).is_ok(),
            Opcode::Splat(_) => f.inst_result(i).is_some_and(|v| lane(f, v).is_ok()),
            _ => false,
        });
        volt_ir::softfp::lower(&mut f);
        // Legalize rewrites byte-swapped and atomic float accesses into forms selected below;
        // ordered int/pointer accesses become `ldar`/`stlr` (or plain accesses when relaxed).
        volt_ir::legalize::legalize(&mut f);
        volt_ir::critical_edge::split_critical_edges(&mut f);
        volt_ir::reachable::neutralize_unreachable(&mut f);
    }
    let fold = crate::regalloc::addrfold::analyze(&f, |f, i| {
        let (p, v) = match f.opcode_ref(i) {
            Opcode::Load(l) if !needs_bare_address(&l.mem) => (l.ptr, f.inst_result(i)?),
            Opcode::Store(s) if !needs_bare_address(&s.mem) => (s.ptr, s.value),
            _ => return None,
        };
        let add = match f.opcode_ref(f.defining_inst(p)?) {
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
    let mut model = Model::new(&f, caps);
    // Straight-line translator output (forward edges, no block parameters,
    // integer values) skips the general scan. Every register here is free for
    // allocation in a function without calls: x0 holds the one incoming
    // argument until the prologue homes it, x8 and x13..x17 are emitter
    // scratch, and x19..x28 are callee-saved (tried last, saved if used).
    const FORWARD_POOL: [u16; 21] = [
        1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
    ];
    let forward = if f.is_variadic || f.sret {
        None
    } else {
        forward::allocate_forward(&f, &model, &FORWARD_POOL)
    };
    let alloc = match forward {
        Some(alloc) => alloc,
        None => {
            if plain {
                // The general allocator wants the CFG normalized first.
                volt_ir::critical_edge::split_critical_edges(&mut f);
                volt_ir::reachable::neutralize_unreachable(&mut f);
                model = Model::new(&f, caps);
            }
            wimmer::allocate(&f, &model).map_err(|_| Error::Unsupported)?
        }
    };
    let mut outgoing = 0;
    let mut leaf = true;
    for bi in 0..f.block_count() {
        for i in f.block_insts(Block(bi as u32)) {
            match f.opcode_ref(*i) {
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
            if let Opcode::Alloca(a) = f.opcode_ref(*i) {
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
                    if let Opcode::If(cf) = f.opcode_ref(*i) {
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
