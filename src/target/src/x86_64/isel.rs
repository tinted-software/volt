// SPDX-License-Identifier: Apache-2.0
// Copyright LilithSemi (Vulcan). Modified for the Volt Rust port.
//! SSA instruction selection and System V AMD64 ABI lowering.
use super::{
    Error,
    encode::{Encoder, *},
};
use crate::regalloc::wimmer::{
    self, Allocation, CallSite, ClassRegs, FixedAssign, Location, Move, RegClass, RegDescription,
    UseKind,
};
use alloc::{vec, vec::Vec};
use volt_ir::function::*;
use volt_ir::types::{FloatKind, Type, TypeKind};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelocKind {
    Call,
    Absolute,
    PcRelative,
    GotPcRelative,
}
#[derive(Clone, Debug)]
pub struct Reloc {
    pub offset: usize,
    pub symbol: alloc::string::String,
    pub kind: RelocKind,
    pub addend: i64,
}
#[derive(Clone, Debug, Default)]
pub struct Compiled {
    pub code: Vec<u8>,
    pub relocs: Vec<Reloc>,
}
struct Description {
    classes: Vec<RegClass>,
    calls: Vec<CallSite>,
}
impl RegDescription<Function> for Description {
    fn classes(&self) -> &[RegClass] {
        &self.classes
    }
    fn class_of(&self, f: &Function, v: Value) -> u16 {
        if matches!(
            f.types.type_kind(f.value_type(v)),
            TypeKind::Float(_) | TypeKind::Vector(_)
        ) {
            1
        } else {
            0
        }
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
        &[11, 15]
    }
    fn scratch2(&self) -> &[u16] {
        &[10, 14]
    }
    fn hosts_critical_edge_moves(&self) -> bool {
        true
    }
}
fn shape(f: &Function, t: Type) -> Result<(usize, bool), Error> {
    Ok(match f.types.type_kind(t) {
        TypeKind::Bool => (1, false),
        TypeKind::Ptr(_) => (8, false),
        TypeKind::Int(i) if i.bits > 0 && i.bits <= 64 => (((i.bits as usize) + 7) / 8, false),
        TypeKind::Float(k) => (
            match k {
                FloatKind::F16 => 2,
                FloatKind::F32 => 4,
                FloatKind::F64 => 8,
                FloatKind::F128 => 16,
            },
            true,
        ),
        TypeKind::Vector(v)
            if (v.len == 4 || v.len == 8)
                && matches!(
                    f.types.type_kind(v.elem),
                    TypeKind::Float(FloatKind::F32)
                        | TypeKind::Int(volt_ir::types::IntDesc { bits: 32, .. })
                ) =>
        {
            ((v.len * 4) as usize, true)
        }
        _ => return Err(Error::Unsupported("type")),
    })
}
fn size(f: &Function, t: Type) -> Result<usize, Error> {
    match f.types.type_kind(t) {
        TypeKind::Array(a) => Ok(size(f, a.elem)?
            .checked_mul(a.len as usize)
            .ok_or(Error::InvalidFunction)?),
        TypeKind::Struct(fields) => {
            let mut n = 0;
            for &t in fields {
                let s = size(f, t)?;
                let a = s.next_power_of_two().min(16);
                n = (n + a - 1) & !(a - 1);
                n += s;
            }
            Ok((n + 7) & !7)
        }
        _ => Ok(shape(f, t)?.0),
    }
}
struct Ctx<'a> {
    f: &'a Function,
    e: Encoder,
    a: Allocation,
    locations: Vec<Vec<wimmer::Segment>>,
    pos: u32,
    saved: Vec<u8>,
    slot_base: [usize; 2],
    temp: usize,
    rsa: usize,
    sret_home: Option<usize>,
    stack_alignment: usize,
    allocas: Vec<Option<usize>>,
    frame: i32,
    fixups: Vec<(usize, Block)>,
    relocs: Vec<Reloc>,
    fold: crate::regalloc::addrfold::Analysis,
}
impl Ctx<'_> {
    fn ty(&self, v: Value) -> &TypeKind {
        self.f.types.type_kind(self.f.value_type(v))
    }
    fn fp(&self, v: Value) -> bool {
        matches!(self.ty(v), TypeKind::Float(_) | TypeKind::Vector(_))
    }
    fn bytes(&self, v: Value) -> usize {
        shape(self.f, self.f.value_type(v))
            .expect("validated type")
            .0
    }
    fn bits(&self, v: Value) -> u16 {
        match self.ty(v) {
            TypeKind::Int(i) => i.bits,
            TypeKind::Bool => 8,
            _ => 64,
        }
    }
    fn signed(&self, v: Value) -> bool {
        matches!(self.ty(v),TypeKind::Int(i) if i.signed)
    }
    fn wide(&self, v: Value) -> bool {
        self.bytes(v) == 32
    }
    fn half(&self, v: Value) -> bool {
        matches!(self.ty(v), TypeKind::Float(FloatKind::F16))
    }
    fn double(&self, v: Value) -> bool {
        matches!(self.ty(v), TypeKind::Float(FloatKind::F64))
    }
    fn disp(&self, off: usize) -> i32 {
        -((self.saved.len() * 8 + off + 32) as i32)
    }
    fn slot(&self, class: u16, s: u32) -> i32 {
        self.disp(self.slot_base[class as usize] + s as usize * 32)
    }
    fn loc(&self, v: Value) -> Location {
        self.locations[v.index()]
            .iter()
            .rev()
            .find(|s| s.from <= self.pos)
            .map_or(Location::Reg(if self.fp(v) { 15 } else { 10 }), |s| s.loc)
    }
    fn get(&mut self, v: Value, r: u8) {
        match self.loc(v) {
            Location::Reg(s) => self.e.mov(r, s as u8),
            Location::Slot(s) => {
                let d = self.slot(0, s);
                self.e.load(r, RBP, d, 64, false);
            }
        }
    }
    fn put(&mut self, v: Value, r: u8) {
        match self.loc(v) {
            Location::Reg(d) => self.e.mov(d as u8, r),
            Location::Slot(s) => {
                let d = self.slot(0, s);
                self.e.store(RBP, d, r, 64);
            }
        }
    }
    fn xget(&mut self, v: Value, r: u8) {
        let wide = self.wide(v);
        match self.loc(v) {
            Location::Reg(s) => self.e.xmm_mov(r, s as u8, wide),
            Location::Slot(s) => {
                let d = self.slot(1, s);
                self.e.xmm_mem(r, RBP, d, if wide { 32 } else { 16 }, false);
            }
        }
    }
    fn xput(&mut self, v: Value, r: u8) {
        let wide = self.wide(v);
        match self.loc(v) {
            Location::Reg(d) => self.e.xmm_mov(d as u8, r, wide),
            Location::Slot(s) => {
                let d = self.slot(1, s);
                self.e.xmm_mem(r, RBP, d, if wide { 32 } else { 16 }, true);
            }
        }
    }
    fn copy(&mut self, d: Value, s: Value) {
        if self.fp(d) {
            self.xget(s, 15);
            self.xput(d, 15);
        } else {
            self.get(s, R11);
            self.put(d, R11);
        }
    }
    fn machine_move(&mut self, m: Move) {
        let fp = m.class == 1;
        let wide = self.wide(m.value);
        match (m.src, m.dst) {
            (Location::Reg(s), Location::Reg(d)) => {
                if fp {
                    self.e.xmm_mov(d as u8, s as u8, wide)
                } else {
                    self.e.mov(d as u8, s as u8)
                }
            }
            (Location::Slot(s), Location::Reg(d)) => {
                let off = self.slot(m.class, s);
                if fp {
                    self.e
                        .xmm_mem(d as u8, RBP, off, if wide { 32 } else { 16 }, false)
                } else {
                    self.e.load(d as u8, RBP, off, 64, false)
                }
            }
            (Location::Reg(s), Location::Slot(d)) => {
                let off = self.slot(m.class, d);
                if fp {
                    self.e
                        .xmm_mem(s as u8, RBP, off, if wide { 32 } else { 16 }, true)
                } else {
                    self.e.store(RBP, off, s as u8, 64)
                }
            }
            (Location::Slot(s), Location::Slot(d)) => {
                let so = self.slot(m.class, s);
                let de = self.slot(m.class, d);
                if fp {
                    self.e
                        .xmm_mem(15, RBP, so, if wide { 32 } else { 16 }, false);
                    self.e
                        .xmm_mem(15, RBP, de, if wide { 32 } else { 16 }, true)
                } else {
                    self.e.load(R11, RBP, so, 64, false);
                    self.e.store(RBP, de, R11, 64)
                }
            }
        }
    }
    fn edge(&mut self, p: Block, j: Jump) {
        if let Some(i) = self
            .a
            .edge_moves
            .iter()
            .position(|e| e.pred == p && e.succ == j.target)
        {
            let moves = self.a.edge_moves[i].moves.clone();
            for m in moves {
                self.machine_move(m);
            }
        }
        let at = self.e.branch(None);
        self.fixups.push((at, j.target));
    }
    fn normalize(&mut self, r: u8, bits: u16, signed: bool) {
        if bits < 64 {
            let shift = (64 - bits) as u8;
            self.e.shift(4, r, Some(shift), true);
            self.e
                .shift(if signed { 7 } else { 5 }, r, Some(shift), true);
        }
    }
    fn int_binary_raw(&mut self, op: BinOp, wide: bool, signed: bool) {
        match op {
            BinOp::Add | BinOp::Sub | BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor => self.e.rr(
                None,
                &[match op {
                    BinOp::Add => 0x01,
                    BinOp::Sub => 0x29,
                    BinOp::BitAnd => 0x21,
                    BinOp::BitOr => 0x09,
                    _ => 0x31,
                }],
                R11,
                R10,
                wide,
            ),
            BinOp::Mul => self.e.rr(None, &[0x0f, 0xaf], R10, R11, wide),
            BinOp::Shl | BinOp::Shr => {
                self.e.mov(RCX, R11);
                self.e.shift(
                    if op == BinOp::Shl {
                        4
                    } else if signed {
                        7
                    } else {
                        5
                    },
                    R10,
                    None,
                    wide,
                );
            }
            BinOp::Div | BinOp::Rem => {
                self.e.mov(RAX, R10);
                if signed {
                    if wide {
                        self.e.bytes(&[0x48, 0x99]);
                    } else {
                        self.e.bytes(&[0x99]);
                    }
                } else {
                    self.e.rr(None, &[0x31], RDX, RDX, false);
                }
                self.e
                    .rr(None, &[0xf7], if signed { 7 } else { 6 }, R11, wide);
                self.e.mov(R10, if op == BinOp::Div { RAX } else { RDX });
            }
            BinOp::Mulh => {
                self.e.mov(RAX, R10);
                self.e
                    .rr(None, &[0xf7], if signed { 5 } else { 4 }, R11, wide);
                self.e.mov(R10, RDX);
            }
        }
    }
    fn int_binary(&mut self, op: BinOp, lhs: Value, result: Value) -> Result<(), Error> {
        let bits = self.bits(result);
        let signed = self.signed(lhs);
        if bits != 32 && bits != 64 {
            self.normalize(R10, bits, signed);
            self.normalize(R11, bits, signed);
        }
        if op == BinOp::Mulh && bits != 32 && bits != 64 {
            if bits < 32 {
                self.e.rr(None, &[0x0f, 0xaf], R10, R11, true);
                self.e
                    .shift(if signed { 7 } else { 5 }, R10, Some(bits as u8), true);
            } else {
                self.e.mov(RAX, R10);
                self.e
                    .rr(None, &[0xf7], if signed { 5 } else { 4 }, R11, true);
                self.e.mov(R10, RAX);
                self.e.shift(5, R10, Some(bits as u8), true);
                self.e.shift(4, RDX, Some((64 - bits) as u8), true);
                self.e.rr(None, &[0x09], RDX, R10, true);
            }
        } else {
            self.int_binary_raw(op, bits > 32, signed);
        }
        if bits != 32 && bits != 64 {
            self.normalize(R10, bits, self.signed(result));
        }
        self.put(result, R10);
        Ok(())
    }
    fn integer_vector(&self, v: Value) -> Option<bool> {
        match self.ty(v) {
            TypeKind::Vector(v) => match self.f.types.type_kind(v.elem) {
                TypeKind::Int(i) => Some(i.signed),
                _ => None,
            },
            _ => None,
        }
    }
    fn vector_integer_binary(
        &mut self,
        op: BinOp,
        result: Value,
        signed: bool,
    ) -> Result<(), Error> {
        let left = self.disp(self.temp);
        let right = self.disp(self.temp + 32);
        let out = self.disp(self.temp + 64);
        self.e.xmm_mem(13, RBP, left, self.bytes(result), true);
        self.e.xmm_mem(14, RBP, right, self.bytes(result), true);
        for lane in 0..self.bytes(result) / 4 {
            self.e.load(R10, RBP, left + lane as i32 * 4, 32, signed);
            self.e.load(R11, RBP, right + lane as i32 * 4, 32, signed);
            self.int_binary_raw(op, false, signed);
            self.e.store(RBP, out + lane as i32 * 4, R10, 32);
        }
        self.e.xmm_mem(13, RBP, out, self.bytes(result), false);
        self.xput(result, 13);
        Ok(())
    }
    fn float_binary(&mut self, op: BinOp, result: Value) -> Result<(), Error> {
        if let Some(signed) = self.integer_vector(result) {
            return self.vector_integer_binary(op, result, signed);
        }
        let opcode = match op {
            BinOp::Add => 0x58,
            BinOp::Sub => 0x5c,
            BinOp::Mul => 0x59,
            BinOp::Div => 0x5e,
            BinOp::BitAnd => 0x54,
            BinOp::BitOr => 0x56,
            BinOp::BitXor => 0x57,
            _ => return Err(Error::Unsupported("floating arithmetic")),
        };
        if self.bytes(result) == 16 && matches!(self.ty(result), TypeKind::Float(_)) {
            return Err(Error::Unsupported(
                "f128 arithmetic requires softfp legalization",
            ));
        }
        if self.wide(result) {
            self.e.vex_rr(1, 0, opcode, 13, 13, 14, true);
        } else {
            let prefix = if matches!(self.ty(result), TypeKind::Vector(_)) || opcode < 0x58 {
                None
            } else {
                Some(if self.double(result) { 0xf2 } else { 0xf3 })
            };
            self.e.rr(prefix, &[0x0f, opcode], 13, 14, false);
        }
        if self.half(result) {
            self.e.half_round(13);
        }
        self.xput(result, 13);
        Ok(())
    }
    fn memory_load(&mut self, result: Value, base: u8, off: i32) {
        if self.fp(result) {
            if self.half(result) {
                self.e.load(R10, base, off, 16, false);
                self.e.rr(Some(0x66), &[0x0f, 0x6e], 13, R10, false);
                self.e.vex_rr(2, 1, 0x13, 13, 0, 13, false);
            } else {
                self.e.xmm_mem(13, base, off, self.bytes(result), false);
            }
            self.xput(result, 13);
        } else {
            self.e
                .load(R10, base, off, self.bits(result), self.signed(result));
            self.put(result, R10);
        }
    }
    fn memory_store(&mut self, value: Value, base: u8, off: i32) {
        if self.fp(value) {
            self.xget(value, 13);
            if self.half(value) {
                self.e.vex_rr(3, 1, 0x1d, 13, 0, 15, false);
                self.e.code.push(0);
                self.e.rr(Some(0x66), &[0x0f, 0x7e], 15, R10, false);
                self.e.store(base, off, R10, 16);
            } else {
                self.e.xmm_mem(13, base, off, self.bytes(value), true);
            }
        } else {
            self.get(value, R10);
            self.e.store(base, off, R10, self.bits(value));
        }
    }
    fn integer_vector_compare(&mut self, c: Compare, r: Value, signed: bool) {
        let left = self.disp(self.temp);
        let right = self.disp(self.temp + 32);
        let out = self.disp(self.temp + 64);
        self.xget(c.lhs, 13);
        self.xget(c.rhs, 14);
        self.e.xmm_mem(13, RBP, left, self.bytes(c.lhs), true);
        self.e.xmm_mem(14, RBP, right, self.bytes(c.rhs), true);
        let cc = match c.op {
            CmpOp::Eq => 4,
            CmpOp::Ne => 5,
            CmpOp::Lt => {
                if signed {
                    12
                } else {
                    2
                }
            }
            CmpOp::Le => {
                if signed {
                    14
                } else {
                    6
                }
            }
            CmpOp::Gt => {
                if signed {
                    15
                } else {
                    7
                }
            }
            CmpOp::Ge => {
                if signed {
                    13
                } else {
                    3
                }
            }
        };
        for lane in 0..self.bytes(c.lhs) / 4 {
            self.e.load(R10, RBP, left + lane as i32 * 4, 32, false);
            self.e.load(R11, RBP, right + lane as i32 * 4, 32, false);
            self.e.rr(None, &[0x39], R11, R10, false);
            self.e.setcc(cc, R10);
            self.e.rr(None, &[0xf7], 3, R10, false);
            self.e.store(RBP, out + lane as i32 * 4, R10, 32);
        }
        self.e.xmm_mem(13, RBP, out, self.bytes(r), false);
        self.xput(r, 13);
    }
    fn comparison(&mut self, c: Compare, r: Value) -> Result<(), Error> {
        if let Some(signed) = self.integer_vector(c.lhs) {
            self.integer_vector_compare(c, r, signed);
            return Ok(());
        }
        if self.fp(c.lhs) {
            self.xget(c.lhs, 13);
            self.xget(c.rhs, 14);
            if matches!(self.ty(c.lhs), TypeKind::Vector(_)) {
                let swap = matches!(c.op, CmpOp::Gt | CmpOp::Ge);
                let (l, h) = if swap { (14, 13) } else { (13, 14) };
                if self.wide(c.lhs) {
                    self.e.vex_rr(1, 0, 0xc2, 15, l, h, true);
                } else {
                    self.e.xmm_mov(15, l, false);
                    self.e.rr(None, &[0x0f, 0xc2], 15, h, false);
                }
                self.e.code.push(match c.op {
                    CmpOp::Eq => 0,
                    CmpOp::Ne => 4,
                    CmpOp::Lt | CmpOp::Gt => 1,
                    _ => 2,
                });
                self.xput(r, 15);
                return Ok(());
            }
            if self.bytes(c.lhs) == 16 {
                return Err(Error::Unsupported(
                    "f128 compare requires softfp legalization",
                ));
            }
            self.e.rr(
                if self.double(c.lhs) { Some(0x66) } else { None },
                &[0x0f, 0x2e],
                13,
                14,
                false,
            );
            let cc = match c.op {
                CmpOp::Eq => 4,
                CmpOp::Ne => 5,
                CmpOp::Lt => 2,
                CmpOp::Le => 6,
                CmpOp::Gt => 7,
                CmpOp::Ge => 3,
            };
            self.e.setcc(cc, R10);
            if matches!(c.op, CmpOp::Eq | CmpOp::Lt | CmpOp::Le) {
                self.e.setcc(11, R11);
                self.e.rr(None, &[0x21], R11, R10, false);
            } else if c.op == CmpOp::Ne {
                self.e.setcc(10, R11);
                self.e.rr(None, &[0x09], R11, R10, false);
            }
            self.put(r, R10);
        } else {
            self.get(c.lhs, R10);
            self.get(c.rhs, R11);
            self.e.rr(None, &[0x39], R11, R10, self.bits(c.lhs) > 32);
            let signed = self.signed(c.lhs);
            let cc = match c.op {
                CmpOp::Eq => 4,
                CmpOp::Ne => 5,
                CmpOp::Lt => {
                    if signed {
                        12
                    } else {
                        2
                    }
                }
                CmpOp::Le => {
                    if signed {
                        14
                    } else {
                        6
                    }
                }
                CmpOp::Gt => {
                    if signed {
                        15
                    } else {
                        7
                    }
                }
                CmpOp::Ge => {
                    if signed {
                        13
                    } else {
                        3
                    }
                }
            };
            self.e.setcc(cc, R10);
            self.put(r, R10);
        }
        Ok(())
    }
    fn convert(&mut self, s: Value, r: Value) -> Result<(), Error> {
        match (self.fp(s), self.fp(r)) {
            (false, false) => {
                self.get(s, R10);
                if self.bits(r) > self.bits(s) {
                    self.normalize(R10, self.bits(s), self.signed(s));
                }
                self.normalize(R10, self.bits(r), self.signed(r));
                self.put(r, R10);
            }
            (true, true) => {
                if matches!(self.ty(s), TypeKind::Vector(_))
                    || matches!(self.ty(r), TypeKind::Vector(_))
                    || self.bytes(s) == 16
                    || self.bytes(r) == 16
                {
                    return Err(Error::Unsupported("composite conversion"));
                }
                self.xget(s, 13);
                if self.double(s) != self.double(r) {
                    self.e.rr(
                        Some(if self.double(s) { 0xf2 } else { 0xf3 }),
                        &[0x0f, 0x5a],
                        13,
                        13,
                        false,
                    );
                }
                if self.half(r) {
                    self.e.half_round(13);
                }
                self.xput(r, 13);
            }
            (false, true) => {
                self.get(s, R10);
                self.normalize(R10, self.bits(s), self.signed(s));
                let p = Some(if self.double(r) { 0xf2 } else { 0xf3 });
                if !self.signed(s) && self.bits(s) == 64 {
                    self.e.rr(None, &[0x85], R10, R10, true);
                    let normal = self.e.branch(Some(9));
                    self.e.mov(R11, R10);
                    self.e.shift(5, R10, Some(1), true);
                    self.e.alu_imm(4, R11, 1, true);
                    self.e.rr(None, &[0x09], R11, R10, true);
                    self.e.rr(p, &[0x0f, 0x2a], 13, R10, true);
                    self.e.rr(p, &[0x0f, 0x58], 13, 13, false);
                    let done = self.e.branch(None);
                    self.e.patch(normal, self.e.code.len())?;
                    self.e.rr(p, &[0x0f, 0x2a], 13, R10, true);
                    self.e.patch(done, self.e.code.len())?;
                } else {
                    self.e.rr(p, &[0x0f, 0x2a], 13, R10, true);
                }
                if self.half(r) {
                    self.e.half_round(13);
                }
                self.xput(r, 13);
            }
            (true, false) => {
                self.xget(s, 13);
                let prefix = Some(if self.double(s) { 0xf2 } else { 0xf3 });
                if !self.signed(r) && self.bits(r) == 64 {
                    self.e.imm(
                        R10,
                        if self.double(s) {
                            (9223372036854775808.0f64).to_bits()
                        } else {
                            (9223372036854775808.0f32).to_bits() as u64
                        },
                    );
                    self.e
                        .rr(Some(0x66), &[0x0f, 0x6e], 14, R10, self.double(s));
                    self.e.rr(
                        if self.double(s) { Some(0x66) } else { None },
                        &[0x0f, 0x2e],
                        13,
                        14,
                        false,
                    );
                    let low = self.e.branch(Some(2));
                    self.e.rr(prefix, &[0x0f, 0x5c], 13, 14, false);
                    self.e.rr(prefix, &[0x0f, 0x2c], R10, 13, true);
                    self.e.imm(R11, 1u64 << 63);
                    self.e.rr(None, &[0x31], R11, R10, true);
                    let end = self.e.branch(None);
                    self.e.patch(low, self.e.code.len())?;
                    self.e.rr(prefix, &[0x0f, 0x2c], R10, 13, true);
                    self.e.patch(end, self.e.code.len())?;
                } else {
                    self.e.rr(
                        prefix,
                        &[0x0f, 0x2c],
                        R10,
                        13,
                        self.bits(r) > 32 || !self.signed(r),
                    );
                }
                self.normalize(R10, self.bits(r), false);
                self.put(r, R10);
            }
        }
        Ok(())
    }
    fn call(
        &mut self,
        target: Option<Value>,
        symbol: Option<u32>,
        args: ValueList,
        result: Option<Value>,
        variadic: bool,
    ) -> Result<(), Error> {
        if let Some(v) = target {
            self.get(v, R10);
            self.e.store(RBP, self.disp(self.temp), R10, 64);
        }
        let mut gi = 0;
        let mut xi = 0;
        let mut stack: i32 = 0;
        let argv = self.f.value_list(args).to_vec();
        for (i, &v) in argv.iter().enumerate() {
            let off = self.disp(self.temp + 32 * (i + 1));
            self.memory_store(v, RBP, off);
        }
        for (i, &v) in argv.iter().enumerate() {
            let off = self.disp(self.temp + 32 * (i + 1));
            if self.fp(v) {
                if xi < 8 {
                    if self.half(v) {
                        self.e.load(R10, RBP, off, 16, false);
                        self.e.rr(Some(0x66), &[0x0f, 0x6e], xi, R10, false);
                    } else {
                        self.e.xmm_mem(xi, RBP, off, self.bytes(v), false);
                    }
                    xi += 1;
                } else {
                    let side = self.bytes(v).max(8) as i32;
                    // System V: each stack slot is aligned to its size (min 8).
                    stack = (stack + side - 1) / side * side;
                    self.e.xmm_mem(13, RBP, off, self.bytes(v), false);
                    self.e.xmm_mem(13, RSP, stack, self.bytes(v), true);
                    stack += side;
                }
            } else if gi < 6 {
                self.e.load(
                    [RDI, RSI, RDX, RCX, R8, R9][gi],
                    RBP,
                    off,
                    self.bits(v),
                    self.signed(v),
                );
                gi += 1;
            } else {
                self.e.load(R11, RBP, off, 64, false);
                self.e.store(RSP, stack, R11, 64);
                stack += 8;
            }
        }
        if variadic {
            self.e.imm(RAX, xi as u64);
        }
        if target.is_some() {
            self.e.load(R10, RBP, self.disp(self.temp), 64, false);
            self.e.rr(None, &[0xff], 2, R10, false);
        } else {
            self.e.code.push(0xe8);
            let off = self.e.code.len();
            self.e.bytes(&[0; 4]);
            self.relocs.push(Reloc {
                offset: off,
                symbol: self
                    .f
                    .symbol_name(symbol.ok_or(Error::InvalidFunction)?)
                    .into(),
                kind: RelocKind::Call,
                addend: -4,
            });
        }
        if let Some(r) = result {
            if self.fp(r) {
                if self.half(r) {
                    self.e.vex_rr(2, 1, 0x13, 0, 0, 0, false);
                }
                self.xput(r, 0);
            } else {
                self.put(r, RAX);
            }
        }
        Ok(())
    }
    fn returned_struct(
        &mut self,
        dest: Option<Value>,
        count: u8,
        pieces: [RetPiece; 4],
    ) -> Result<(), Error> {
        let Some(dest) = dest else { return Ok(()) };
        self.get(dest, R11);
        let mut gi = 0;
        let mut xi = 0;
        for piece in pieces.iter().take(count as usize) {
            if piece.fp {
                if xi >= 2 {
                    return Err(Error::Unsupported("aggregate FP returns"));
                }
                self.e
                    .xmm_mem(xi, R11, piece.offset as i32, piece.bytes as usize, true);
                xi += 1;
            } else {
                if gi >= 2 {
                    return Err(Error::Unsupported("aggregate integer returns"));
                }
                self.e.store(
                    R11,
                    piece.offset as i32,
                    [RAX, RDX][gi],
                    piece.bytes as u16 * 8,
                );
                gi += 1;
            }
        }
        Ok(())
    }
    fn va_start(&mut self, list: Value) {
        let mut gp = 0usize;
        let mut fp = 0usize;
        let mut stack = 0usize;
        for &v in self
            .f
            .block_params(Block(0))
            .iter()
            .take(self.f.num_fixed_params as usize)
        {
            if self.fp(v) {
                if fp < 8 {
                    fp += 1;
                } else {
                    stack += self.bytes(v).max(8);
                }
            } else if gp < 6 {
                gp += 1;
            } else {
                stack += 8;
            }
        }
        self.get(list, R11);
        self.e.imm(R10, (gp * 8) as u64);
        self.e.store(R11, 0, R10, 32);
        self.e.imm(R10, (48 + fp * 16) as u64);
        self.e.store(R11, 4, R10, 32);
        self.e
            .mem(None, &[0x8d], R10, RBP, 16 + stack as i32, true, false);
        self.e.store(R11, 8, R10, 64);
        self.e
            .mem(None, &[0x8d], R10, RBP, self.disp(self.rsa), true, false);
        self.e.store(R11, 16, R10, 64);
    }
    fn va_arg(&mut self, list: Value, result: Value) -> Result<(), Error> {
        if self.bytes(result) > 8 {
            return Err(Error::Unsupported("wide va_arg"));
        }
        let fp = self.fp(result);
        let field = if fp { 4 } else { 0 };
        let limit = if fp { 176 } else { 48 };
        let step = if fp { 16 } else { 8 };
        self.get(list, R11);
        self.e.load(R10, R11, field, 32, false);
        self.e.alu_imm(7, R10, limit, false);
        let overflow = self.e.branch(Some(3));
        self.e.mov(RCX, R10);
        self.e.load(RAX, R11, 16, 64, false);
        self.e.rr(None, &[0x01], R10, RAX, true);
        self.e.alu_imm(0, RCX, step, false);
        self.e.store(R11, field, RCX, 32);
        self.memory_load(result, RAX, 0);
        let end = self.e.branch(None);
        self.e.patch(overflow, self.e.code.len())?;
        self.e.load(RAX, R11, 8, 64, false);
        self.e.mov(RCX, RAX);
        self.e.alu_imm(0, RCX, 8, true);
        self.e.store(R11, 8, RCX, 64);
        self.memory_load(result, RAX, 0);
        self.e.patch(end, self.e.code.len())?;
        Ok(())
    }
    fn actions(&mut self) {
        for i in 0..self.a.actions.len() {
            let action = self.a.actions[i];
            if action.at == self.pos {
                self.machine_move(Move {
                    src: action.src,
                    dst: action.dst,
                    class: action.class,
                    value: action.value,
                });
            }
        }
    }
    fn inst(&mut self, i: Inst) -> Result<(), Error> {
        let result = self.f.inst_result(i);
        let op = self.f.opcode(i);
        match op {
            Opcode::Store(s) => {
                self.get(s.ptr, R11);
                self.memory_store(s.value, R11, self.fold.off_of(i) as i32);
            }
            Opcode::Prefetch(p) => {
                self.get(p.ptr, R11);
                self.e.mem(None, &[0x0f, 0x18], 1, R11, 0, false, false);
            }
            Opcode::Call(c) => {
                self.call(None, Some(c.symbol), c.args, result, c.is_variadic)?;
                self.returned_struct(c.ret_dest, c.ret_regs, c.ret_pieces)?;
            }
            Opcode::CallIndirect(c) => {
                self.call(Some(c.target), None, c.args, result, c.is_variadic)?;
                self.returned_struct(c.ret_dest, c.ret_regs, c.ret_pieces)?;
            }
            Opcode::VaStart(v) => self.va_start(v.list),
            Opcode::VaEnd(_) => {}
            _ => {
                let r = result.ok_or(Error::Unsupported("statement"))?;
                match op {
                    Opcode::Iconst(n) => {
                        self.e.imm(R10, n as u64);
                        self.normalize(R10, self.bits(r), self.signed(r));
                        self.put(r, R10);
                    }
                    Opcode::Fconst(n) => {
                        if self.bytes(r) == 16 {
                            return Err(Error::Unsupported("f128 constant must use Fconst128"));
                        }
                        let value = if self.double(r) {
                            n.to_bits()
                        } else {
                            (n as f32).to_bits() as u64
                        };
                        self.e.imm(R10, value);
                        self.e
                            .rr(Some(0x66), &[0x0f, 0x6e], 13, R10, self.double(r));
                        if self.half(r) {
                            self.e.half_round(13);
                        }
                        self.xput(r, 13);
                    }
                    Opcode::Fconst128(n) => {
                        let off = self.disp(self.temp);
                        self.e.imm(R10, n as u64);
                        self.e.store(RBP, off, R10, 64);
                        self.e.imm(R10, (n >> 64) as u64);
                        self.e.store(RBP, off + 8, R10, 64);
                        self.e.xmm_mem(13, RBP, off, 16, false);
                        self.xput(r, 13);
                    }
                    Opcode::Arith(a) => {
                        if self.fp(r) {
                            self.xget(a.lhs, 13);
                            self.xget(a.rhs, 14);
                            self.float_binary(a.op, r)?;
                        } else {
                            self.get(a.lhs, R10);
                            self.get(a.rhs, R11);
                            self.int_binary(a.op, a.lhs, r)?;
                        }
                    }
                    Opcode::ArithImm(a) => {
                        if self.fp(r) {
                            return Err(Error::Unsupported("floating immediate arithmetic"));
                        }
                        self.get(a.lhs, R10);
                        self.e.imm(R11, a.imm as u64);
                        self.int_binary(a.op, a.lhs, r)?;
                    }
                    Opcode::Icmp(c) => self.comparison(c, r)?,
                    Opcode::Select(s) => {
                        if matches!(self.ty(s.cond), TypeKind::Vector(_)) {
                            self.xget(s.cond, 13);
                            self.xget(s.then, 14);
                            self.xget(s.else_, 15);
                            if self.wide(r) {
                                self.e.vex_rr(1, 0, 0x54, 14, 14, 13, true);
                                self.e.vex_rr(1, 0, 0x55, 13, 13, 15, true);
                                self.e.vex_rr(1, 0, 0x56, 14, 14, 13, true);
                            } else {
                                self.e.rr(None, &[0x0f, 0x54], 14, 13, false);
                                self.e.rr(None, &[0x0f, 0x55], 13, 15, false);
                                self.e.rr(None, &[0x0f, 0x56], 14, 13, false);
                            }
                            self.xput(r, 14);
                        } else {
                            self.get(s.cond, R10);
                            self.e.rr(None, &[0x85], R10, R10, self.bits(s.cond) > 32);
                            let no = self.e.branch(Some(4));
                            self.copy(r, s.then);
                            let end = self.e.branch(None);
                            self.e.patch(no, self.e.code.len())?;
                            self.copy(r, s.else_);
                            self.e.patch(end, self.e.code.len())?;
                        }
                    }
                    Opcode::Convert(c) => self.convert(c.value, r)?,
                    Opcode::Unary(u) => {
                        if u.op == UnaryOp::Reinterpret {
                            if self.fp(u.value) == self.fp(r) {
                                self.copy(r, u.value);
                            } else if self.fp(r) {
                                self.get(u.value, R10);
                                self.e.rr(
                                    Some(0x66),
                                    &[0x0f, 0x6e],
                                    13,
                                    R10,
                                    self.bits(u.value) > 32,
                                );
                                self.xput(r, 13);
                            } else {
                                self.xget(u.value, 13);
                                self.e
                                    .rr(Some(0x66), &[0x0f, 0x7e], 13, R10, self.bits(r) > 32);
                                self.put(r, R10);
                            }
                        } else {
                            self.xget(u.value, 13);
                            if u.op == UnaryOp::Sqrt {
                                if self.wide(r) {
                                    self.e.vex_rr(1, 0, 0x51, 13, 0, 13, true);
                                } else {
                                    self.e.rr(
                                        if matches!(self.ty(r), TypeKind::Vector(_)) {
                                            None
                                        } else {
                                            Some(if self.double(r) { 0xf2 } else { 0xf3 })
                                        },
                                        &[0x0f, 0x51],
                                        13,
                                        13,
                                        false,
                                    );
                                }
                            } else {
                                let v = matches!(self.ty(r), TypeKind::Vector(_));
                                let opcode = if v {
                                    0x08
                                } else if self.double(r) {
                                    0x0b
                                } else {
                                    0x0a
                                };
                                if self.wide(r) {
                                    self.e.vex_rr(3, 1, opcode, 13, 0, 13, true);
                                } else {
                                    self.e.rr(Some(0x66), &[0x0f, 0x3a, opcode], 13, 13, false);
                                }
                                self.e.code.push(
                                    8 | match u.op {
                                        UnaryOp::Nearest => 0,
                                        UnaryOp::Floor => 1,
                                        UnaryOp::Ceil => 2,
                                        UnaryOp::Trunc => 3,
                                        _ => return Err(Error::Unsupported("unary")),
                                    },
                                );
                            }
                            if self.half(r) {
                                self.e.half_round(13);
                            }
                            self.xput(r, 13);
                        }
                    }
                    Opcode::Load(l) => {
                        self.get(l.ptr, R11);
                        self.memory_load(r, R11, self.fold.off_of(i) as i32);
                    }
                    Opcode::GlobalAddr(g) => {
                        self.e
                            .bytes(&[0x4c, if g.via_got { 0x8b } else { 0x8d }, 0x15]);
                        let offset = self.e.code.len();
                        self.e.bytes(&[0; 4]);
                        self.relocs.push(Reloc {
                            offset,
                            symbol: self.f.symbol_name(g.symbol).into(),
                            kind: if g.via_got {
                                RelocKind::GotPcRelative
                            } else {
                                RelocKind::PcRelative
                            },
                            addend: -4,
                        });
                        self.put(r, R10);
                    }
                    Opcode::Alloca(_) => {
                        let off = self.disp(self.allocas[r.index()].ok_or(Error::InvalidFunction)?);
                        self.e.mem(None, &[0x8d], R10, RBP, off, true, false);
                        self.put(r, R10);
                    }
                    Opcode::StructNew(s) => {
                        if !matches!(self.ty(r), TypeKind::Vector(_)) {
                            return Err(Error::Unsupported(
                                "aggregate construction requires legalization",
                            ));
                        }
                        let fields = self.f.value_list(s.fields).to_vec();
                        let off = self.disp(self.temp);
                        for (j, v) in fields.into_iter().enumerate() {
                            self.memory_store(v, RBP, off + (j * 4) as i32);
                        }
                        self.e.xmm_mem(13, RBP, off, self.bytes(r), false);
                        self.xput(r, 13);
                    }
                    Opcode::Extract(ex) => {
                        if !matches!(self.ty(ex.aggregate), TypeKind::Vector(_)) {
                            return Err(Error::Unsupported(
                                "aggregate extraction requires legalization",
                            ));
                        }
                        let off = self.disp(self.temp);
                        self.xget(ex.aggregate, 13);
                        self.e.xmm_mem(13, RBP, off, self.bytes(ex.aggregate), true);
                        self.memory_load(r, RBP, off + ex.index as i32 * 4);
                    }
                    Opcode::Splat(s) => {
                        let off = self.disp(self.temp);
                        for j in 0..self.bytes(r) / 4 {
                            self.memory_store(s.scalar, RBP, off + (j * 4) as i32);
                        }
                        self.e.xmm_mem(13, RBP, off, self.bytes(r), false);
                        self.xput(r, 13);
                    }
                    Opcode::Reduce(rd) => {
                        let off = self.disp(self.temp);
                        self.xget(rd.vector, 13);
                        self.e.xmm_mem(13, RBP, off, self.bytes(rd.vector), true);
                        if self.fp(r) {
                            self.e.xmm_mem(13, RBP, off, 4, false);
                            for j in 1..self.bytes(rd.vector) / 4 {
                                self.e.xmm_mem(14, RBP, off + (j * 4) as i32, 4, false);
                                self.float_binary(rd.op, r)?;
                            }
                            self.xput(r, 13);
                        } else {
                            self.e.load(R10, RBP, off, 32, false);
                            for j in 1..self.bytes(rd.vector) / 4 {
                                self.e.load(R11, RBP, off + (j * 4) as i32, 32, false);
                                self.int_binary(rd.op, r, r)?;
                            }
                            self.put(r, R10);
                        }
                    }
                    Opcode::Dot(d) => {
                        let off = self.disp(self.temp);
                        self.xget(d.a, 13);
                        self.xget(d.b, 14);
                        if self.wide(d.a) {
                            self.e.vex_rr(1, 0, 0x59, 13, 13, 14, true);
                        } else {
                            self.e.rr(None, &[0x0f, 0x59], 13, 14, false);
                        }
                        self.e.xmm_mem(13, RBP, off, self.bytes(d.a), true);
                        self.xget(d.acc, 13);
                        for j in 0..self.bytes(d.a) / 4 {
                            self.e.xmm_mem(14, RBP, off + (j * 4) as i32, 4, false);
                            self.e.rr(
                                Some(if self.double(r) { 0xf2 } else { 0xf3 }),
                                &[0x0f, 0x58],
                                13,
                                14,
                                false,
                            );
                        }
                        self.xput(r, 13);
                    }
                    Opcode::VaArg(v) => self.va_arg(v.list, r)?,
                    _ => return Err(Error::Unsupported("operation requires IR legalization")),
                }
            }
        }
        Ok(())
    }
    fn ret(&mut self, r: &Ret) -> Result<(), Error> {
        for (i, &v) in r.slice().iter().enumerate() {
            self.memory_store(v, RBP, self.disp(self.temp + 32 * i));
        }
        let mut gi = 0;
        let mut xi = 0;
        for (i, &v) in r.slice().iter().enumerate() {
            let off = self.disp(self.temp + 32 * i);
            if self.fp(v) {
                if xi >= 2 {
                    return Err(Error::Unsupported("too many FP returns"));
                }
                if self.half(v) {
                    self.e.load(R10, RBP, off, 16, false);
                    self.e.rr(Some(0x66), &[0x0f, 0x6e], xi, R10, false);
                } else {
                    self.e.xmm_mem(xi, RBP, off, self.bytes(v), false);
                }
                xi += 1;
            } else {
                if gi >= 2 {
                    return Err(Error::Unsupported("too many integer returns"));
                }
                self.e
                    .load([RAX, RDX][gi], RBP, off, self.bits(v), self.signed(v));
                gi += 1;
            }
        }
        if let Some(home) = self.sret_home {
            self.e.load(RAX, RBP, self.disp(home), 64, false);
        }
        if self.stack_alignment > 16 {
            self.e.mem(
                None,
                &[0x8d],
                RSP,
                RBP,
                -((self.saved.len() * 8) as i32),
                true,
                false,
            );
        } else if self.frame != 0 {
            self.e.alu_imm(0, RSP, self.frame, true);
        }
        for &s in self.saved.iter().rev() {
            self.e.pop(s);
        }
        self.e.pop(RBP);
        self.e.code.push(0xc3);
        Ok(())
    }
}

pub fn compile_object(function: &Function) -> Result<Compiled, Error> {
    if function_uses_composite_f16(function) {
        return Err(Error::Unsupported("composite f16"));
    }
    let mut work = function.clone_func();
    volt_ir::expand::expand_nv_fp4(&mut work);
    volt_ir::expand::expand_low_float(&mut work);
    volt_ir::expand::expand_matmul(&mut work);
    volt_ir::expand::expand_vector_lanes(&mut work);
    volt_ir::softfp::lower(&mut work);
    volt_ir::legalize::legalize(&mut work);
    volt_ir::critical_edge::split_critical_edges(&mut work);
    volt_ir::reachable::neutralize_unreachable(&mut work);
    let fold = crate::regalloc::addrfold::analyze(&work, |f, i| {
        let p = match f.opcode(i) {
            Opcode::Load(l) => l.ptr,
            Opcode::Store(s) => s.ptr,
            _ => return None,
        };
        match f.opcode(f.defining_inst(p)?) {
            Opcode::ArithImm(a) => i32::try_from(a.imm).ok().map(i64::from),
            _ => None,
        }
    });
    for &(i, addr) in &fold.folds {
        match work.opcode_mut(i) {
            Opcode::Load(l) => l.ptr = addr.base,
            Opcode::Store(s) => s.ptr = addr.base,
            _ => {}
        }
    }
    for bi in 0..work.block_count() {
        let b = Block(bi as u32);
        work.block_insts_mut(b).retain(|i| !fold.is_dead_add(*i));
    }
    compile_lowered(&work, fold)
}
fn compile_lowered(
    f: &Function,
    fold: crate::regalloc::addrfold::Analysis,
) -> Result<Compiled, Error> {
    if f.block_count() == 0 {
        return Err(Error::InvalidFunction);
    }
    for bi in 0..f.block_count() {
        let b = Block(bi as u32);
        for &v in f.block_params(b) {
            shape(f, f.value_type(v))?;
        }
        for &i in f.block_insts(b) {
            if let Some(v) = f.inst_result(i) {
                shape(f, f.value_type(v))?;
            }
        }
    }
    let mut calls = Vec::new();
    let mut pos = 0;
    let mut max_args = 8.max(f.block_params(Block(0)).len());
    for bi in 0..f.block_count() {
        pos += 1;
        for &i in f.block_insts(Block(bi as u32)) {
            match f.opcode(i) {
                Opcode::Call(c) => {
                    max_args = max_args.max(c.args.len as usize);
                    calls.push(CallSite {
                        pos,
                        clobbered: vec![ClassRegs {
                            class: 1,
                            regs: (0..13).collect(),
                        }],
                    });
                }
                Opcode::CallIndirect(c) => {
                    max_args = max_args.max(c.args.len as usize);
                    calls.push(CallSite {
                        pos,
                        clobbered: vec![ClassRegs {
                            class: 1,
                            regs: (0..13).collect(),
                        }],
                    });
                }
                _ => {}
            }
            pos += 1;
        }
        pos += 1;
    }
    let desc = Description {
        classes: vec![
            RegClass {
                name: "gpr",
                allocatable: vec![3, 12, 13, 14, 15],
                callee_saved: vec![3, 12, 13, 14, 15],
                slot_bytes: 32,
                narrow_preserved: vec![],
            },
            RegClass {
                name: "xmm",
                allocatable: (0..13).collect(),
                callee_saved: vec![],
                slot_bytes: 32,
                narrow_preserved: vec![],
            },
        ],
        calls,
    };
    let a = wimmer::allocate(f, &desc).map_err(|_| Error::Unsupported("register allocation"))?;
    let mut locations = vec![Vec::new(); f.value_count()];
    for (v, s) in &a.segments {
        locations[v.index()] = s.clone();
    }
    let saved: Vec<_> = a
        .used_callee_saved
        .iter()
        .filter(|s| s.class == 0)
        .map(|s| s.reg as u8)
        .collect();
    let slot_base = [0, a.slot_count_per_class[0] as usize * 32];
    let mut used = slot_base[1] + a.slot_count_per_class[1] as usize * 32;
    let sret_home = if f.sret {
        if f.block_params(Block(0)).first().is_none() {
            return Err(Error::InvalidFunction);
        }
        let home = used;
        used += 8;
        Some(home)
    } else {
        None
    };
    let mut stack_alignment = 16;
    for bi in 0..f.block_count() {
        for &i in f.block_insts(Block(bi as u32)) {
            let args = match f.opcode(i) {
                Opcode::Call(c) => c.args,
                Opcode::CallIndirect(c) => c.args,
                _ => continue,
            };
            let mut vectors = 0;
            for &v in f.value_list(args) {
                let (bytes, fp) = shape(f, f.value_type(v))?;
                if fp {
                    if vectors >= 8 {
                        stack_alignment = stack_alignment.max(bytes.max(8));
                    }
                    vectors += 1;
                }
            }
        }
    }
    let mut allocas = vec![None; f.value_count()];
    for bi in 0..f.block_count() {
        for &i in f.block_insts(Block(bi as u32)) {
            if let Opcode::Alloca(al) = f.opcode(i) {
                let v = f.inst_result(i).ok_or(Error::InvalidFunction)?;
                let sz = size(f, al.elem)?;
                allocas[v.index()] = Some(used + sz.saturating_sub(32));
                used += (sz + 31) & !31;
            }
        }
    }
    let rsa = used + 160;
    if f.is_variadic {
        used += 192;
    }
    let temp = used;
    used += (max_args + 1) * 32;
    let outgoing = max_args * 32;
    let total = used + outgoing;
    let frame = (((total + saved.len() * 8 + 15) & !15) - saved.len() * 8) as i32;
    let mut c = Ctx {
        f,
        e: Encoder::default(),
        a,
        locations,
        pos: 0,
        saved,
        slot_base,
        temp,
        rsa,
        sret_home,
        stack_alignment,
        allocas,
        frame,
        fixups: Vec::new(),
        relocs: Vec::new(),
        fold,
    };
    c.e.bytes(&[0xf3, 0x0f, 0x1e, 0xfa]);
    c.e.push(RBP);
    c.e.mov(RBP, RSP);
    for &s in &c.saved {
        c.e.push(s);
    }
    if frame != 0 {
        c.e.alu_imm(5, RSP, frame, true);
    }
    if stack_alignment > 16 {
        // Dynamic alignment: overflow 256-bit vector slots need >16 alignment.
        c.e.alu_imm(4, RSP, -32, true); // and rsp, imm32(=0xffffffe0)
    }
    if let Some(home) = sret_home {
        c.e.store(RBP, c.disp(home), RDI, 64);
    }
    if f.is_variadic {
        let base = c.disp(rsa);
        for (i, r) in [RDI, RSI, RDX, RCX, R8, R9].into_iter().enumerate() {
            c.e.store(RBP, base + i as i32 * 8, r, 64);
        }
        for i in 0..8 {
            c.e.xmm_mem(i, RBP, base + 48 + i as i32 * 16, 16, true);
        }
    }
    let entry = f.block_params(Block(0));
    let mut gi = 0;
    let mut xi = 0;
    let mut stack = 16;
    for (j, &v) in entry.iter().enumerate() {
        let off = c.disp(temp + j * 32);
        if c.fp(v) {
            if xi < 8 {
                if c.half(v) {
                    c.e.rr(Some(0x66), &[0x0f, 0x7e], xi, R10, false);
                    c.e.store(RBP, off, R10, 16);
                } else {
                    c.e.xmm_mem(xi, RBP, off, c.bytes(v), true);
                }
                xi += 1;
            } else {
                // Mirrors the caller's aligned outgoing slot layout.
                let side = c.bytes(v).max(8) as i32;
                let rel = stack - 16;
                let aligned = (rel + side - 1) / side * side;
                stack = 16 + aligned;
                c.e.xmm_mem(13, RBP, stack, c.bytes(v), false);
                c.e.xmm_mem(13, RBP, off, c.bytes(v), true);
                stack += side;
            }
        } else {
            if gi < 6 {
                c.e.store(RBP, off, [RDI, RSI, RDX, RCX, R8, R9][gi], 64);
                gi += 1;
            } else {
                c.e.load(R10, RBP, stack, 64, false);
                c.e.store(RBP, off, R10, 64);
                stack += 8;
            }
        }
    }
    for (j, &v) in entry.iter().enumerate() {
        c.memory_load(v, RBP, c.disp(temp + j * 32));
    }
    let mut starts = vec![0; f.block_count()];
    for bi in 0..f.block_count() {
        let block = Block(bi as u32);
        starts[bi] = c.e.code.len();
        c.pos += 1;
        let mut branched = false;
        for &i in f.block_insts(block) {
            c.actions();
            if let Opcode::If(cf) = f.opcode(i) {
                c.get(cf.cond, R10);
                c.e.rr(None, &[0x85], R10, R10, c.bits(cf.cond) > 32);
                let other = c.e.branch(Some(4));
                c.edge(block, cf.then);
                c.e.patch(other, c.e.code.len())?;
                c.edge(block, cf.else_);
                branched = true;
            } else {
                c.inst(i)?;
            }
            c.pos += 1;
        }
        c.actions();
        if !branched {
            match f.terminator(block) {
                Some(Terminator::Ret(r)) => c.ret(&r)?,
                Some(Terminator::Jump(j)) => c.edge(block, j),
                None if f.block_insts(block).is_empty() && f.block_params(block).is_empty() => {}
                None => return Err(Error::InvalidFunction),
            }
        }
        c.pos += 1;
    }
    for &(at, b) in &c.fixups {
        let target = *starts.get(b.index()).ok_or(Error::InvalidFunction)?;
        c.e.patch(at, target)?;
    }
    Ok(Compiled {
        code: c.e.code,
        relocs: c.relocs,
    })
}
