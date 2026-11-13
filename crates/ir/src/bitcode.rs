//! Binary IR serialization ("bitcode"): a compact, position-independent
//! encoding of a [`super::function::Function`] and a decoder that rebuilds an
//! equivalent one. The binary analog of the text printer/parser, used by
//! cross-module work (LTO/PGO). Ported from
//! `vulcan/libs/vulcan-ir/bitcode.zig`.
//!
//! Values are referenced by serial number (their order in a canonical walk:
//! per block, parameters then instruction results), not by raw handle, so the
//! decoder rebuilds the function with fresh handles and stays isomorphic.
//! Round-trip oracle: `print(decode(encode(f))) == print(f)`.

#[allow(unused_imports)]
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use super::attribute::{AttrValue, Attribute, Custom, Endianness};
use super::function::{
    Alloca, Arith, ArithImm, AtomicOp, AtomicOrdering, AtomicRmw, AtomicScope, AttrTarget, Barrier,
    BarrierScope, BinOp, Block, Call, CallIndirect, CmpOp, Compare, Convert, Dot, Extract,
    Function, GlobalAddr, InputSigns, Inst, Jump, Load, LowFloatConvert, MatMul, MatMulQuant,
    MatMulQuantOut, MatMulScale, MatMulType, NvFp4Convert, Opcode, Prefetch, Reduce, Ret, RetPiece,
    Select, Splat, Store, StructNew, Terminator, Unary, UnaryOp, VaArg, VaEnd, VaStart, Value,
};
use super::low_float::Format as LowFloatFormat;
use super::nvfp4::ScaleApplication;
use super::types::{
    AddressSpace, ArrayDesc, FloatKind, IntDesc, SliceDesc, Type, TypeKind, VectorDesc,
};

/// The stream is malformed: wrong magic, truncated, an unknown tag byte, an
/// out-of-range index or number. A recoverable fault, never an invalid enum.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MalformedSegment;

impl core::fmt::Display for MalformedSegment {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "malformed bitcode")
    }
}

/// Result alias with the error type as a defaulted parameter.
pub type Result<T, E = MalformedSegment> = core::result::Result<T, E>;

const MAGIC: &[u8; 4] = b"VBC1";

/// Bytes the stream header occupies before the type table: the magic, the
/// whole-function flag byte, and the fixed-parameter count.
#[cfg(test)]
const HEADER_LEN: usize = MAGIC.len() + 1 + 4;

/// Whole-function flag bits, packed into one header byte.
const FN_FLAG_VARIADIC: u8 = 1 << 0;
const FN_FLAG_SRET: u8 = 1 << 1;
const FN_FLAG_LOCAL: u8 = 1 << 2;

/// Extra-field flag bits of a `call` or `call_indirect` record.
const CALL_FLAG_VARIADIC: u8 = 1 << 0;
const CALL_FLAG_SRET: u8 = 1 << 1;
const CALL_FLAG_RET_DEST: u8 = 1 << 2;

/// Attribute target tags (stable on the wire).
const ATTR_TARGET_FUNC: u8 = 0;
const ATTR_TARGET_BLOCK: u8 = 1;
const ATTR_TARGET_INST: u8 = 2;
const ATTR_TARGET_VALUE: u8 = 3;

/// Attribute body tags (stable on the wire).
const ATTR_INLINE: u8 = 0;
const ATTR_NORETURN: u8 = 1;
const ATTR_COLD: u8 = 2;
const ATTR_ALIGN: u8 = 3;
const ATTR_ENDIAN: u8 = 4;
const ATTR_CUSTOM: u8 = 5;

/// Namespaced attribute payload tags (stable on the wire).
const ATTR_VALUE_FLAG: u8 = 0;
const ATTR_VALUE_INT: u8 = 1;
const ATTR_VALUE_STRING: u8 = 2;

/// A canonical number that names nothing. An instruction that no block holds,
/// or a value that no such instruction defines, has no place in the canonical
/// walk, so an attribute on it names no entity in the decoded function and is
/// not written.
const NO_SERIAL: u32 = u32::MAX;

// Opcode tags (stable on the wire).
const OP_ICONST: u8 = 0;
const OP_FCONST: u8 = 1;
const OP_ARITH: u8 = 2;
const OP_ARITH_IMM: u8 = 3;
const OP_ICMP: u8 = 4;
const OP_SELECT: u8 = 5;
const OP_STRUCT_NEW: u8 = 6;
const OP_EXTRACT: u8 = 7;
const OP_CONVERT: u8 = 8;
const OP_ALLOCA: u8 = 9;
const OP_CALL: u8 = 10;
const OP_LOAD: u8 = 11;
const OP_STORE: u8 = 12;
const OP_IF: u8 = 13;
const OP_GLOBAL_ADDR: u8 = 14;
const OP_UNARY: u8 = 15;
const OP_CALL_INDIRECT: u8 = 16;
const OP_PREFETCH: u8 = 17;
const OP_DOT: u8 = 18;
const OP_MATMUL: u8 = 19;
const OP_VA_START: u8 = 20;
const OP_VA_ARG: u8 = 21;
const OP_VA_END: u8 = 22;
const OP_FCONST128: u8 = 23;
const OP_BARRIER: u8 = 24;
const OP_ATOMIC_RMW: u8 = 25;
const OP_DECODE_LOW_FLOAT: u8 = 26;
const OP_ENCODE_LOW_FLOAT: u8 = 27;
const OP_DEQUANTIZE_NVFP4: u8 = 28;
const OP_QUANTIZE_NVFP4: u8 = 29;
const OP_REDUCE: u8 = 30;
const OP_SPLAT: u8 = 31;

/// An `atomic_rmw` record flag bit: the compare operand follows the two
/// ordinary operand slots. Written from the field and read back into it, so a
/// record round-trips whatever the opcode holds, rather than depending on the
/// op/compare agreement `verify` enforces.
const ATOMIC_FLAG_COMPARE: u8 = 1 << 0;

struct Writer {
    bytes: Vec<u8>,
}

impl Writer {
    fn new() -> Self {
        Self { bytes: Vec::new() }
    }
    fn u8(&mut self, v: u8) {
        self.bytes.push(v);
    }
    fn u16(&mut self, v: u16) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }
    fn slice(&mut self, s: &[u8]) {
        self.bytes.extend_from_slice(s);
    }
    /// A length-prefixed string, the same shape the symbol table uses.
    fn string(&mut self, s: &str) {
        self.u32(s.len() as u32);
        self.slice(s.as_bytes());
    }
}

/// Serialize `func` into bitcode. The caller owns the returned bytes.
pub fn encode(func: &Function) -> Result<Vec<u8>> {
    let mut w = Writer::new();

    // Canonical serial number for each value (params then results, block order)
    // and for each instruction (block order, then order within the block).
    // Both start at `NO_SERIAL`, so an entity the walk never reaches keeps a
    // number that names nothing.
    let mut serial = vec![NO_SERIAL; func.value_count()];
    let mut inst_serial = vec![NO_SERIAL; func.inst_count()];
    {
        let mut next: u32 = 0;
        let mut next_inst: u32 = 0;
        for bi in 0..func.block_count() {
            let block = Block(bi as u32);
            for p in func.block_params(block) {
                serial[p.0 as usize] = next;
                next += 1;
            }
            for inst in func.block_insts(block) {
                inst_serial[inst.0 as usize] = next_inst;
                next_inst += 1;
                if let Some(r) = func.inst_result(*inst) {
                    serial[r.0 as usize] = next;
                    next += 1;
                }
            }
        }
    }

    w.slice(MAGIC);

    // Whole-function metadata. Each of these changes how the function is
    // called or how its symbol binds, so a stream without them decodes to a
    // different function.
    let mut fn_flags = 0u8;
    if func.is_variadic {
        fn_flags |= FN_FLAG_VARIADIC;
    }
    if func.sret {
        fn_flags |= FN_FLAG_SRET;
    }
    if func.is_local {
        fn_flags |= FN_FLAG_LOCAL;
    }
    w.u8(fn_flags);
    w.u32(func.num_fixed_params);

    // Types (interned in dependency order, so a kind's nested types precede it).
    w.u32(func.types.count() as u32);
    for i in 0..func.types.count() {
        write_type(&mut w, func.types.type_kind(Type(i as u32)));
    }

    // Symbols.
    w.u32(func.symbol_count() as u32);
    for i in 0..func.symbol_count() {
        let name = func.symbol_name(i as u32);
        w.u32(name.len() as u32);
        w.slice(name.as_bytes());
    }

    // Blocks.
    w.u32(func.block_count() as u32);
    for bi in 0..func.block_count() {
        let block = Block(bi as u32);
        let params = func.block_params(block);
        w.u32(params.len() as u32);
        for p in params {
            w.u32(func.value_type(*p).0);
        }

        let insts = func.block_insts(block);
        w.u32(insts.len() as u32);
        for inst in insts {
            write_inst(&mut w, func, *inst, &serial);
        }

        write_term(&mut w, func, block, &serial);
    }

    // Attributes come last, because a target names a block, an instruction or
    // a value, and the decoder can only resolve those once it has rebuilt the
    // body.
    write_attrs(&mut w, func, &serial, &inst_serial);

    Ok(w.bytes)
}

/// Write the attribute list. An entry whose target is not in the canonical walk
/// names no entity in the decoded function, so it is dropped rather than
/// written with a number the decoder would have to reject.
fn write_attrs(w: &mut Writer, func: &Function, serial: &[u32], inst_serial: &[u32]) {
    let entries = func.attribute_entries();
    let mut count: u32 = 0;
    for entry in entries {
        if attr_target_serial(entry.target, serial, inst_serial).is_some() {
            count += 1;
        }
    }
    w.u32(count);
    for entry in entries {
        let number = match attr_target_serial(entry.target, serial, inst_serial) {
            Some(n) => n,
            None => continue,
        };
        match entry.target {
            AttrTarget::Func => w.u8(ATTR_TARGET_FUNC),
            AttrTarget::Block(_) => {
                w.u8(ATTR_TARGET_BLOCK);
                w.u32(number);
            }
            AttrTarget::Inst(_) => {
                w.u8(ATTR_TARGET_INST);
                w.u32(number);
            }
            AttrTarget::Value(_) => {
                w.u8(ATTR_TARGET_VALUE);
                w.u32(number);
            }
        }
        write_attr(w, &entry.attr);
    }
}

/// The canonical number an attribute target resolves to, or `None` when the
/// target is not part of the canonical walk. `Func` has no number and reports
/// 0.
fn attr_target_serial(target: AttrTarget, serial: &[u32], inst_serial: &[u32]) -> Option<u32> {
    match target {
        AttrTarget::Func => Some(0),
        AttrTarget::Block(b) => Some(b.0),
        AttrTarget::Inst(i) => {
            let n = inst_serial[i.0 as usize];
            if n == NO_SERIAL { None } else { Some(n) }
        }
        AttrTarget::Value(v) => {
            let n = serial[v.0 as usize];
            if n == NO_SERIAL { None } else { Some(n) }
        }
    }
}

/// Write one attribute body. String payloads use the length-prefixed form the
/// symbol table already uses.
fn write_attr(w: &mut Writer, attr: &Attribute) {
    match attr {
        Attribute::Inline => w.u8(ATTR_INLINE),
        Attribute::Noreturn => w.u8(ATTR_NORETURN),
        Attribute::Cold => w.u8(ATTR_COLD),
        Attribute::Align(a) => {
            w.u8(ATTR_ALIGN);
            w.u32(*a);
        }
        Attribute::Endian(e) => {
            w.u8(ATTR_ENDIAN);
            // The decoder maps this byte back with a range check, so pin the
            // tag values here. This follows the `float` and `ptr` arms of
            // `write_type`.
            debug_assert_eq!(Endianness::Little as u8, 0);
            debug_assert_eq!(Endianness::Big as u8, 1);
            debug_assert_eq!(Endianness::Native as u8, 2);
            w.u8(*e as u8);
        }
        Attribute::Custom(c) => {
            w.u8(ATTR_CUSTOM);
            w.string(&c.namespace);
            w.string(&c.key);
            match &c.value {
                AttrValue::Flag => w.u8(ATTR_VALUE_FLAG),
                AttrValue::Int(i) => {
                    w.u8(ATTR_VALUE_INT);
                    w.u64(*i as u64);
                }
                AttrValue::Str(s) => {
                    w.u8(ATTR_VALUE_STRING);
                    w.string(s);
                }
            }
        }
    }
}

/// Write the type-table record for a kind, in nested-first order.
fn write_type(w: &mut Writer, kind: &TypeKind) {
    match kind {
        TypeKind::Bool => w.u8(0),
        TypeKind::Int(i) => {
            w.u8(1);
            w.u8(if i.signed { 0 } else { 1 });
            w.u16(i.bits);
        }
        TypeKind::Float(f) => {
            w.u8(2);
            // Pin the tag values here: the decoder is a hardcoded switch, so a
            // future reorder would silently corrupt streams instead of failing
            // to build.
            debug_assert_eq!(FloatKind::F32 as u8, 0);
            debug_assert_eq!(FloatKind::F64 as u8, 1);
            debug_assert_eq!(FloatKind::F16 as u8, 2);
            debug_assert_eq!(FloatKind::F128 as u8, 3);
            w.u8(*f as u8);
        }
        TypeKind::Ptr(space) => {
            w.u8(3);
            // The decoder is a hardcoded switch on these values, so pin them
            // here. A future reorder would otherwise desync the two sides and
            // silently corrupt streams instead of failing to build.
            debug_assert_eq!(AddressSpace::Global as u8, 0);
            debug_assert_eq!(AddressSpace::Shared as u8, 1);
            debug_assert_eq!(AddressSpace::Private as u8, 2);
            debug_assert_eq!(AddressSpace::Constant as u8, 3);
            w.u8(*space as u8);
        }
        TypeKind::Vector(v) => {
            w.u8(4);
            w.u32(v.len);
            w.u32(v.elem.0);
        }
        TypeKind::Struct(fields) => {
            w.u8(5);
            w.u32(fields.len() as u32);
            for f in fields {
                w.u32(f.0);
            }
        }
        TypeKind::Array(a) => {
            w.u8(6);
            w.u64(a.len);
            w.u32(a.elem.0);
        }
        TypeKind::Slice(s) => {
            w.u8(7);
            w.u32(s.elem.0);
        }
    }
}

fn write_inst(w: &mut Writer, func: &Function, inst: Inst, serial: &[u32]) {
    let result = func.inst_result(inst);
    w.u8(if result.is_some() { 1 } else { 0 });
    if let Some(r) = result {
        w.u32(func.value_type(r).0);
    }

    match func.opcode(inst) {
        Opcode::Iconst(v) => {
            w.u8(OP_ICONST);
            w.u64(v as u64);
        }
        Opcode::Fconst(v) => {
            w.u8(OP_FCONST);
            w.u64(v.to_bits());
        }
        Opcode::Fconst128(v) => {
            w.u8(OP_FCONST128);
            w.u64(v as u64);
            w.u64((v >> 64) as u64);
        }
        Opcode::Arith(a) => {
            w.u8(OP_ARITH);
            w.u8(a.op as u8);
            w.u32(serial[a.lhs.0 as usize]);
            w.u32(serial[a.rhs.0 as usize]);
        }
        Opcode::ArithImm(a) => {
            w.u8(OP_ARITH_IMM);
            w.u8(a.op as u8);
            w.u32(serial[a.lhs.0 as usize]);
            w.u64(a.imm as u64);
        }
        Opcode::Icmp(c) => {
            w.u8(OP_ICMP);
            w.u8(c.op as u8);
            w.u32(serial[c.lhs.0 as usize]);
            w.u32(serial[c.rhs.0 as usize]);
        }
        Opcode::Select(s) => {
            w.u8(OP_SELECT);
            w.u32(serial[s.cond.0 as usize]);
            w.u32(serial[s.then.0 as usize]);
            w.u32(serial[s.else_.0 as usize]);
        }
        Opcode::StructNew(sn) => {
            w.u8(OP_STRUCT_NEW);
            let fields = func.value_list(sn.fields);
            w.u32(fields.len() as u32);
            for f in fields {
                w.u32(serial[f.0 as usize]);
            }
        }
        Opcode::Extract(e) => {
            w.u8(OP_EXTRACT);
            w.u32(serial[e.aggregate.0 as usize]);
            w.u32(e.index);
        }
        Opcode::Convert(c) => {
            w.u8(OP_CONVERT);
            w.u32(serial[c.value.0 as usize]);
        }
        Opcode::DecodeLowFloat(c) => {
            w.u8(OP_DECODE_LOW_FLOAT);
            debug_assert_eq!(LowFloatFormat::Bf16 as u8, 0);
            debug_assert_eq!(LowFloatFormat::F8E4M3 as u8, 1);
            debug_assert_eq!(LowFloatFormat::F8E5M2 as u8, 2);
            w.u8(c.format as u8);
            w.u32(serial[c.value.0 as usize]);
        }
        Opcode::EncodeLowFloat(c) => {
            w.u8(OP_ENCODE_LOW_FLOAT);
            w.u8(c.format as u8);
            w.u32(serial[c.value.0 as usize]);
        }
        Opcode::DequantizeNvfp4(c) | Opcode::QuantizeNvfp4(c) => {
            debug_assert_eq!(ScaleApplication::Multiply as u8, 0);
            debug_assert_eq!(ScaleApplication::Divide as u8, 1);
            let tag = match func.opcode(inst) {
                Opcode::DequantizeNvfp4(_) => OP_DEQUANTIZE_NVFP4,
                _ => OP_QUANTIZE_NVFP4,
            };
            w.u8(tag);
            w.u8(c.block_application as u8);
            w.u8(c.global_application as u8);
            w.u32(serial[c.value.0 as usize]);
            w.u32(serial[c.block_scale.0 as usize]);
            w.u32(serial[c.global_scale.0 as usize]);
        }
        Opcode::Unary(u) => {
            w.u8(OP_UNARY);
            w.u8(u.op as u8);
            w.u32(serial[u.value.0 as usize]);
        }
        Opcode::Alloca(a) => {
            w.u8(OP_ALLOCA);
            w.u32(a.elem.0);
        }
        Opcode::Call(c) => {
            w.u8(OP_CALL);
            w.u32(c.symbol);
            let args = func.value_list(c.args);
            w.u32(args.len() as u32);
            for a in args {
                w.u32(serial[a.0 as usize]);
            }
            write_call_extras(
                w,
                c.is_variadic,
                c.num_fixed,
                c.sret,
                c.ret_dest.map(|rd| serial[rd.0 as usize]),
                c.ret_regs,
                &c.ret_pieces,
            );
        }
        Opcode::CallIndirect(c) => {
            w.u8(OP_CALL_INDIRECT);
            w.u32(serial[c.target.0 as usize]);
            let args = func.value_list(c.args);
            w.u32(args.len() as u32);
            for a in args {
                w.u32(serial[a.0 as usize]);
            }
            write_call_extras(
                w,
                c.is_variadic,
                c.num_fixed,
                c.sret,
                c.ret_dest.map(|rd| serial[rd.0 as usize]),
                c.ret_regs,
                &c.ret_pieces,
            );
        }
        // `volatile` marks an access the optimizer must not remove, move or
        // merge. A stream that drops it turns an MMIO register access into an
        // ordinary one.
        Opcode::Load(l) => {
            w.u8(OP_LOAD);
            w.u32(serial[l.ptr.0 as usize]);
            w.u8(l.volatile as u8);
        }
        Opcode::Store(st) => {
            w.u8(OP_STORE);
            w.u32(serial[st.value.0 as usize]);
            w.u32(serial[st.ptr.0 as usize]);
            w.u8(st.volatile as u8);
        }
        Opcode::Prefetch(pf) => {
            w.u8(OP_PREFETCH);
            w.u32(serial[pf.ptr.0 as usize]);
        }
        // `VaArg.ty` is not written separately. It always equals the result
        // type already written generically above, so the decoder recovers it
        // from `rty` with nothing extra on the wire.
        Opcode::VaStart(vs) => {
            w.u8(OP_VA_START);
            w.u32(serial[vs.list.0 as usize]);
        }
        Opcode::VaArg(va) => {
            w.u8(OP_VA_ARG);
            w.u32(serial[va.list.0 as usize]);
        }
        Opcode::VaEnd(ve) => {
            w.u8(OP_VA_END);
            w.u32(serial[ve.list.0 as usize]);
        }
        Opcode::Barrier(bar) => {
            w.u8(OP_BARRIER);
            // The decoder maps this byte back with a range check, so pin the
            // tag values here. A future reorder of `BarrierScope` would
            // otherwise silently turn a workgroup barrier into a subgroup one.
            debug_assert_eq!(BarrierScope::Workgroup as u8, 0);
            debug_assert_eq!(BarrierScope::Subgroup as u8, 1);
            w.u8(bar.scope as u8);
        }
        Opcode::AtomicRmw(a) => {
            w.u8(OP_ATOMIC_RMW);
            // The decoder maps these three bytes back with range checks, so pin
            // the tag values here.
            debug_assert_eq!(AtomicOp::Add as u8, 0);
            debug_assert_eq!(AtomicOp::Min as u8, 1);
            debug_assert_eq!(AtomicOp::Max as u8, 2);
            debug_assert_eq!(AtomicOp::BitAnd as u8, 3);
            debug_assert_eq!(AtomicOp::BitOr as u8, 4);
            debug_assert_eq!(AtomicOp::BitXor as u8, 5);
            debug_assert_eq!(AtomicOp::Exchange as u8, 6);
            debug_assert_eq!(AtomicOp::CompareExchange as u8, 7);
            debug_assert_eq!(AtomicOrdering::Relaxed as u8, 0);
            debug_assert_eq!(AtomicOrdering::Acquire as u8, 1);
            debug_assert_eq!(AtomicOrdering::Release as u8, 2);
            debug_assert_eq!(AtomicOrdering::AcqRel as u8, 3);
            debug_assert_eq!(AtomicOrdering::SeqCst as u8, 4);
            debug_assert_eq!(AtomicScope::Workgroup as u8, 0);
            debug_assert_eq!(AtomicScope::Device as u8, 1);
            debug_assert_eq!(AtomicScope::System as u8, 2);
            w.u8(a.op as u8);
            w.u8(a.ordering as u8);
            w.u8(a.scope as u8);
            w.u8(if a.compare.is_some() {
                ATOMIC_FLAG_COMPARE
            } else {
                0
            });
            w.u32(serial[a.ptr.0 as usize]);
            w.u32(serial[a.value.0 as usize]);
            if let Some(c) = a.compare {
                w.u32(serial[c.0 as usize]);
            }
        }
        Opcode::Dot(d) => {
            w.u8(OP_DOT);
            w.u32(serial[d.acc.0 as usize]);
            w.u32(serial[d.a.0 as usize]);
            w.u32(serial[d.b.0 as usize]);
        }
        Opcode::Reduce(red) => {
            w.u8(OP_REDUCE);
            w.u8(red.op as u8);
            w.u32(serial[red.vector.0 as usize]);
        }
        Opcode::Splat(sp) => {
            w.u8(OP_SPLAT);
            w.u32(serial[sp.scalar.0 as usize]);
        }
        Opcode::Matmul(mm) => {
            w.u8(OP_MATMUL);
            w.u32(serial[mm.a.0 as usize]);
            w.u32(serial[mm.b.0 as usize]);
            w.u32(serial[mm.c.0 as usize]);
            w.u16(mm.m);
            w.u16(mm.n);
            w.u16(mm.k);
            w.u8(mm.dtype as u8); // MatMulType, widened to a byte
            w.u8(mm.accumulate as u8);
            w.u8(mm.embedded as u8); // self-contained (embedded) lowering flag
            if let Some(s) = mm.input_signs {
                w.u8(1);
                w.u8(s.a_unsigned as u8);
                w.u8(s.b_unsigned as u8);
            } else {
                w.u8(0);
            }
            write_matmul_quant(w, func, &mm.quant);
        }
        Opcode::If(cf) => {
            w.u8(OP_IF);
            w.u32(serial[cf.cond.0 as usize]);
            write_jump(w, func, cf.then, serial);
            write_jump(w, func, cf.else_, serial);
        }
        Opcode::GlobalAddr(ga) => {
            w.u8(OP_GLOBAL_ADDR);
            w.u32(ga.symbol);
            w.u8(ga.via_got as u8);
        }
    }
}

/// Write the quant epilogue, mirroring the Zig field order: relu, out,
/// zero_point, then bias, then scale.
fn write_matmul_quant(w: &mut Writer, func: &Function, q: &Option<MatMulQuant>) {
    if let Some(q) = q {
        w.u8(1);
        w.u8(q.relu as u8);
        w.u8(q.out as u8); // MatMulQuantOut (i8=0, u8=1)
        w.u32(q.zero_point as u32); // i32 zero-point as u32 bits
        if let Some(bh) = q.bias {
            w.u8(1);
            let bias = func.bias_list(bh);
            w.u32(bias.len() as u32);
            for v in bias {
                w.u32(*v as u32); // i32 as u32 bits
            }
        } else {
            w.u8(0);
        }
        match q.scale {
            MatMulScale::Scalar(bits) => {
                w.u8(0);
                w.u32(bits);
            }
            MatMulScale::PerColumn(h) => {
                w.u8(1);
                let scales = func.scale_list(h);
                w.u32(scales.len() as u32);
                for s in scales {
                    w.u32(*s);
                }
            }
        }
    } else {
        w.u8(0);
    }
}

/// Write the extra fields a call carries: the variadic marker with the
/// callee's fixed parameter count, the hidden-pointer struct return, and the
/// register-return destination with its pieces. `ret_dest_serial` is the
/// destination's canonical number, or `None` when the call has no register
/// return. It goes LAST, after the argument serials, and the decoder reads it
/// in the same place, so the operand fixup order stays fixed.
fn write_call_extras(
    w: &mut Writer,
    is_variadic: bool,
    num_fixed: u32,
    sret: bool,
    ret_dest_serial: Option<u32>,
    ret_regs: u8,
    ret_pieces: &[RetPiece; 4],
) {
    let mut flags = 0u8;
    if is_variadic {
        flags |= CALL_FLAG_VARIADIC;
    }
    if sret {
        flags |= CALL_FLAG_SRET;
    }
    if ret_dest_serial.is_some() {
        flags |= CALL_FLAG_RET_DEST;
    }
    w.u8(flags);
    w.u32(num_fixed);
    // Only `ret_pieces[0..ret_regs]` carries meaning, so only that part goes on
    // the wire and the decoder rebuilds the rest at its default.
    debug_assert!(ret_regs <= ret_pieces.len() as u8);
    w.u8(ret_regs);
    for p in ret_pieces.iter().take(ret_regs as usize) {
        w.u8(p.fp as u8);
        w.u8(p.offset);
        w.u8(p.bytes);
    }
    if let Some(s) = ret_dest_serial {
        w.u32(s);
    }
}

fn write_jump(w: &mut Writer, func: &Function, jump: Jump, serial: &[u32]) {
    w.u32(jump.target.0);
    let args = func.block_args(jump);
    w.u32(args.len() as u32);
    for a in args {
        w.u32(serial[a.0 as usize]);
    }
}

fn write_term(w: &mut Writer, func: &Function, block: Block, serial: &[u32]) {
    match func.terminator(block) {
        None => w.u8(0), // no terminator (implicit ret void)
        Some(Terminator::Ret(r)) => {
            w.u8(1);
            w.u8(r.count);
            for v in r.slice() {
                w.u32(serial[v.0 as usize]);
            }
        }
        Some(Terminator::Jump(j)) => {
            w.u8(2);
            write_jump(w, func, j, serial);
        }
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    /// Subtraction form: `pos <= len` is an invariant, so it never underflows,
    /// and unlike `pos + n > len` it cannot wrap.
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if n > self.bytes.len() - self.pos {
            return Err(MalformedSegment);
        }
        let s = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        let s = self.take(2)?;
        Ok(u16::from_le_bytes([s[0], s[1]]))
    }
    fn u32(&mut self) -> Result<u32> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn u64(&mut self) -> Result<u64> {
        let s = self.take(8)?;
        Ok(u64::from_le_bytes([
            s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
        ]))
    }
    fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }
}

/// Look up a type-table entry read from untrusted input. `valid` is the number
/// of entries decoded so far; an index at or beyond it is either out of range
/// or a forward reference into uninitialized memory. Both are malformed input.
fn map_type(type_map: &[Type], valid: usize, idx: u32) -> Result<Type> {
    if idx as usize >= valid {
        return Err(MalformedSegment);
    }
    Ok(type_map[idx as usize])
}

/// Convert an untrusted block number to a `Block`, rejecting out-of-range
/// values that would later index the block list out of bounds.
fn check_block(bnum: u32, block_count: usize) -> Result<Block> {
    if bnum as usize >= block_count {
        return Err(MalformedSegment);
    }
    Ok(Block(bnum))
}

/// Decode bitcode into an equivalent `Function`. The caller owns it.
pub fn decode(bytes: &[u8]) -> Result<Function> {
    let mut r = Reader { bytes, pos: 0 };
    if r.take(4)? != &MAGIC[..] {
        return Err(MalformedSegment);
    }

    let mut func = Function::new();

    // Whole-function metadata. Unknown flag bits are malformed input: the
    // stream carries no version, so a bit this build does not know is a stream
    // it cannot read.
    let fn_flags = r.u8()?;
    if fn_flags & !(FN_FLAG_VARIADIC | FN_FLAG_SRET | FN_FLAG_LOCAL) != 0 {
        return Err(MalformedSegment);
    }
    func.is_variadic = fn_flags & FN_FLAG_VARIADIC != 0;
    func.sret = fn_flags & FN_FLAG_SRET != 0;
    func.is_local = fn_flags & FN_FLAG_LOCAL != 0;
    func.num_fixed_params = r.u32()?;

    // Types (interned in order, nested references resolve to earlier handles).
    let type_count = r.u32()? as usize;
    if type_count > r.remaining() {
        // Each record occupies at least one byte; a count larger than the
        // remaining stream is malformed and must not drive a huge internal
        // allocation from an untrusted input.
        return Err(MalformedSegment);
    }
    let mut type_map: Vec<Type> = Vec::with_capacity(type_count);
    // Pass `i` as the valid-entry count so a nested type may only reference a
    // type decoded earlier, never a forward/uninitialized one.
    for i in 0..type_count {
        type_map.push(read_type(&mut r, &mut func, &type_map, i)?);
    }

    // Symbols.
    let sym_count = r.u32()?;
    for _ in 0..sym_count {
        let len = r.u32()? as usize;
        let bytes = r.take(len)?;
        let name = core::str::from_utf8(bytes).map_err(|_| MalformedSegment)?;
        func.intern_symbol(name);
    }

    // Blocks. Create them all first so jump/if targets resolve.
    let block_count = r.u32()? as usize;
    for _ in 0..block_count {
        func.append_block();
    }

    // The serial->Value table, filled as values are recreated in canonical
    // order.
    let mut serial: Vec<Value> = Vec::new();

    // Operand serials waiting to be patched, pooled for every fixup record.
    let mut fixups = FixupTable::new();

    // The instruction serial->Inst table, in the same canonical order, so an
    // attribute can name the instruction it rides on.
    let mut inst_serial: Vec<Inst> = Vec::new();

    for bi in 0..block_count {
        let block = Block(bi as u32);
        let param_count = r.u32()? as usize;
        if param_count > r.remaining() {
            return Err(MalformedSegment);
        }
        for _ in 0..param_count {
            let ty = map_type(&type_map, type_map.len(), r.u32()?)?;
            serial.push(func.append_block_param(block, ty));
        }
        let inst_count = r.u32()? as usize;
        if inst_count > r.remaining() {
            return Err(MalformedSegment);
        }
        for _ in 0..inst_count {
            read_inst(
                &mut r,
                &mut func,
                block,
                &type_map,
                block_count,
                &mut serial,
                &mut inst_serial,
                &mut fixups,
            )?;
        }
        read_term(&mut r, &mut func, block, block_count, &mut fixups)?;
    }

    // Every fixup slot is a serial number read from input; validate the whole
    // set against the recovered value table before applying, so `apply` can
    // index it without bounds checks and a bad serial is a recoverable fault,
    // not an OOB.
    for f in &fixups.fixups {
        for slot in f.slots(&fixups.slots) {
            if *slot as usize >= serial.len() {
                return Err(MalformedSegment);
            }
        }
    }
    for f in &fixups.fixups {
        match f.target {
            FixTarget::Inst(_) => f.apply_inst(&mut func, &fixups.slots, &serial),
            FixTarget::Terminator(_) => f.apply_term(&mut func, &fixups.slots, &serial),
        }
    }

    read_attrs(&mut r, &mut func, block_count, &serial, &inst_serial)?;

    Ok(func)
}

/// Read the attribute list written by `write_attrs` and reattach each entry.
/// Every target number comes off an UNTRUSTED stream, so one that names no
/// entity is malformed input.
fn read_attrs(
    r: &mut Reader<'_>,
    func: &mut Function,
    block_count: usize,
    serial: &[Value],
    inst_serial: &[Inst],
) -> Result<()> {
    let count = r.u32()?;
    for _ in 0..count {
        let kind = r.u8()?;
        let target = match kind {
            ATTR_TARGET_FUNC => AttrTarget::Func,
            ATTR_TARGET_BLOCK => AttrTarget::Block(check_block(r.u32()?, block_count)?),
            ATTR_TARGET_INST => {
                let n = r.u32()?;
                let inst = *inst_serial.get(n as usize).ok_or(MalformedSegment)?;
                AttrTarget::Inst(inst)
            }
            ATTR_TARGET_VALUE => {
                let n = r.u32()?;
                let v = *serial.get(n as usize).ok_or(MalformedSegment)?;
                AttrTarget::Value(v)
            }
            _ => return Err(MalformedSegment),
        };
        let attr = read_attr(r)?;
        func.add_attr(target, attr);
    }
    Ok(())
}

/// Read one attribute body.
fn read_attr(r: &mut Reader<'_>) -> Result<Attribute> {
    Ok(match r.u8()? {
        ATTR_INLINE => Attribute::Inline,
        ATTR_NORETURN => Attribute::Noreturn,
        ATTR_COLD => Attribute::Cold,
        ATTR_ALIGN => Attribute::Align(r.u32()?),
        ATTR_ENDIAN => {
            // The byte comes off an UNTRUSTED stream, so an unknown value is a
            // recoverable fault and never an invalid enum.
            let order = match r.u8()? {
                0 => Endianness::Little,
                1 => Endianness::Big,
                2 => Endianness::Native,
                _ => return Err(MalformedSegment),
            };
            Attribute::Endian(order)
        }
        ATTR_CUSTOM => {
            let namespace = read_string(r)?;
            let key = read_string(r)?;
            let value = match r.u8()? {
                ATTR_VALUE_FLAG => AttrValue::Flag,
                ATTR_VALUE_INT => AttrValue::Int(r.u64()? as i64),
                ATTR_VALUE_STRING => AttrValue::Str(read_string(r)?),
                _ => return Err(MalformedSegment),
            };
            Attribute::Custom(Custom {
                namespace,
                key,
                value,
            })
        }
        _ => return Err(MalformedSegment),
    })
}

/// Read a length-prefixed string, the same shape the symbol table uses.
fn read_string(r: &mut Reader<'_>) -> Result<String> {
    let len = r.u32()? as usize;
    let bytes = r.take(len)?;
    String::from_utf8(bytes.to_vec()).map_err(|_| MalformedSegment)
}

fn read_type(
    r: &mut Reader<'_>,
    func: &mut Function,
    type_map: &[Type],
    valid: usize,
) -> Result<Type> {
    Ok(match r.u8()? {
        0 => func.types.intern(TypeKind::Bool),
        1 => {
            let signed = r.u8()? == 0; // 0 = signed, 1 = unsigned
            let bits = r.u16()?;
            func.types.intern(TypeKind::Int(IntDesc { signed, bits }))
        }
        2 => {
            let kind = match r.u8()? {
                0 => FloatKind::F32,
                1 => FloatKind::F64,
                2 => FloatKind::F16,
                3 => FloatKind::F128,
                _ => return Err(MalformedSegment),
            };
            func.types.intern(TypeKind::Float(kind))
        }
        3 => {
            // The byte comes off an UNTRUSTED stream, so an unknown value is a
            // recoverable fault and never an invalid enum.
            let space = match r.u8()? {
                0 => AddressSpace::Global,
                1 => AddressSpace::Shared,
                2 => AddressSpace::Private,
                3 => AddressSpace::Constant,
                _ => return Err(MalformedSegment),
            };
            func.types.intern(TypeKind::Ptr(space))
        }
        4 => {
            let len = r.u32()?;
            let elem = map_type(type_map, valid, r.u32()?)?;
            func.types
                .intern(TypeKind::Vector(VectorDesc { len, elem }))
        }
        5 => {
            let n = r.u32()? as usize;
            if n > r.remaining() {
                return Err(MalformedSegment);
            }
            let mut fields = Vec::new();
            for _ in 0..n {
                fields.push(map_type(type_map, valid, r.u32()?)?);
            }
            func.types.intern(TypeKind::Struct(fields))
        }
        6 => {
            let len = r.u64()?;
            let elem = map_type(type_map, valid, r.u32()?)?;
            func.types.intern(TypeKind::Array(ArrayDesc { len, elem }))
        }
        7 => {
            let elem = map_type(type_map, valid, r.u32()?)?;
            func.types.intern(TypeKind::Slice(SliceDesc { elem }))
        }
        _ => return Err(MalformedSegment),
    })
}

/// A deferred operand fix: at decode time operands are recorded as slot
/// numbers, then patched to real values once all values exist.
struct Fixup {
    target: FixTarget,
    start: usize,
    len: usize,
}

/// Which entity a fixup's slots ride on.
enum FixTarget {
    Inst(Inst),
    Terminator(Block),
}

/// All pending fixups, with their serial slots pooled into one flat buffer.
struct FixupTable {
    fixups: Vec<Fixup>,
    slots: Vec<u32>,
}

impl FixupTable {
    fn new() -> Self {
        Self {
            fixups: Vec::new(),
            slots: Vec::new(),
        }
    }
    /// Begin a record: returns the offset its slots start at.
    fn begin(&mut self) -> usize {
        self.slots.len()
    }
    fn push_slot(&mut self, slot: u32) {
        self.slots.push(slot);
    }
    /// Commit the range started at `start` for `target`, if any slots were
    /// read (matches the Zig rule that a record with no slots fixes nothing).
    fn commit_inst(&mut self, target: Inst, start: usize) {
        if self.slots.len() > start {
            self.fixups.push(Fixup {
                target: FixTarget::Inst(target),
                start,
                len: self.slots.len() - start,
            });
        }
    }
    fn commit_term(&mut self, target: Block, start: usize) {
        if self.slots.len() > start {
            self.fixups.push(Fixup {
                target: FixTarget::Terminator(target),
                start,
                len: self.slots.len() - start,
            });
        }
    }
}

impl Fixup {
    fn slots<'a>(&self, slots: &'a [u32]) -> &'a [u32] {
        &slots[self.start..self.start + self.len]
    }

    /// Patch the operand slots of an instruction. Implemented as: take an owned
    /// copy of the opcode (a scalar record: operands and list ids only), apply
    /// the serial patches to that copy, patch value lists in place through the
    /// function, and write the opcode back — never holding two borrows at once.
    fn apply_inst(&self, func: &mut Function, slots: &[u32], serial: &[Value]) {
        let sl = self.slots(slots);
        let inst = match self.target {
            FixTarget::Inst(inst) => inst,
            _ => unreachable!(),
        };
        let mut i = 0usize;
        let mut next = || -> Value {
            let v = serial[sl[i] as usize];
            i += 1;
            v
        };
        let mut op = func.opcode(inst);
        match &mut op {
            // A barrier reserves no operand slot, because it carries no Value.
            Opcode::Iconst(_)
            | Opcode::Fconst(_)
            | Opcode::Fconst128(_)
            | Opcode::Alloca(_)
            | Opcode::GlobalAddr(_)
            | Opcode::Barrier(_) => {}
            Opcode::StructNew(sn) => {
                let fields = sn.fields;
                for f in func.value_list_mut(fields) {
                    *f = next();
                }
            }
            // The register-return destination is an operand, and it comes after
            // the arguments on the wire, so it is filled in that order here.
            Opcode::Call(c) => {
                let args = c.args;
                for a in func.value_list_mut(args) {
                    *a = next();
                }
                if let Some(rd) = c.ret_dest.as_mut() {
                    *rd = next();
                }
            }
            Opcode::CallIndirect(c) => {
                c.target = next();
                let args = c.args;
                for a in func.value_list_mut(args) {
                    *a = next();
                }
                if let Some(rd) = c.ret_dest.as_mut() {
                    *rd = next();
                }
            }
            Opcode::If(cf) => {
                cf.cond = next();
                let then_args = cf.then.args;
                let else_args = cf.else_.args;
                for a in func.value_list_mut(then_args) {
                    *a = next();
                }
                for a in func.value_list_mut(else_args) {
                    *a = next();
                }
            }
            Opcode::Arith(a) => {
                a.lhs = next();
                a.rhs = next();
            }
            Opcode::ArithImm(a) => a.lhs = next(),
            Opcode::Icmp(c) => {
                c.lhs = next();
                c.rhs = next();
            }
            Opcode::Select(s) => {
                s.cond = next();
                s.then = next();
                s.else_ = next();
            }
            Opcode::Extract(e) => e.aggregate = next(),
            Opcode::Convert(c) => c.value = next(),
            Opcode::DecodeLowFloat(c) | Opcode::EncodeLowFloat(c) => c.value = next(),
            Opcode::DequantizeNvfp4(c) | Opcode::QuantizeNvfp4(c) => {
                c.value = next();
                c.block_scale = next();
                c.global_scale = next();
            }
            Opcode::Unary(u) => u.value = next(),
            Opcode::Load(l) => l.ptr = next(),
            Opcode::Store(st) => {
                st.value = next();
                st.ptr = next();
            }
            // Slot order matches the write order above: address, operand, then
            // the compare operand when the record carried one.
            Opcode::AtomicRmw(a) => {
                a.ptr = next();
                a.value = next();
                if let Some(c) = a.compare.as_mut() {
                    *c = next();
                }
            }
            Opcode::Prefetch(pf) => pf.ptr = next(),
            Opcode::VaStart(vs) => vs.list = next(),
            Opcode::VaArg(va) => va.list = next(),
            Opcode::VaEnd(ve) => ve.list = next(),
            Opcode::Dot(d) => {
                d.acc = next();
                d.a = next();
                d.b = next();
            }
            Opcode::Reduce(red) => red.vector = next(),
            Opcode::Splat(sp) => sp.scalar = next(),
            Opcode::Matmul(mm) => {
                mm.a = next();
                mm.b = next();
                mm.c = next();
            }
        }
        *func.opcode_mut(inst) = op;
    }

    /// Patch the operand slots of a terminator record. Ret values are inline,
    /// so they patch through the terminator pointer; jump args ride a value
    /// list, patched in place. The two branches never borrow the function
    /// twice at once.
    fn apply_term(&self, func: &mut Function, slots: &[u32], serial: &[Value]) {
        let sl = self.slots(slots);
        let block = match self.target {
            FixTarget::Terminator(block) => block,
            _ => unreachable!(),
        };
        let mut i = 0usize;
        let jump_args = match func.terminator(block) {
            None => return,
            Some(Terminator::Ret(_)) => {
                let term_ptr = func.terminator_ptr(block);
                if let Some(Terminator::Ret(r)) = term_ptr.as_mut() {
                    for v in r.values.iter_mut().take(r.count as usize) {
                        *v = serial[sl[i] as usize];
                        i += 1;
                    }
                }
                return;
            }
            Some(Terminator::Jump(j)) => j.args,
        };
        for a in func.value_list_mut(jump_args) {
            *a = serial[sl[i] as usize];
            i += 1;
        }
    }
}
/// Read the extra fields `write_call_extras` wrote. The destination serial,
/// when present, is appended after the argument serials, matching the write
/// order and the fixup order.
fn read_call_extras(r: &mut Reader<'_>, slots: &mut FixupTable) -> Result<CallExtras> {
    let flags = r.u8()?;
    if flags & !(CALL_FLAG_VARIADIC | CALL_FLAG_SRET | CALL_FLAG_RET_DEST) != 0 {
        return Err(MalformedSegment);
    }
    let num_fixed = r.u32()?;
    let ret_regs = r.u8()?;
    // A call carries at most 4 return-register pieces. The count comes off an
    // UNTRUSTED stream, so a larger one is malformed input and never an
    // out-of-range array write.
    if ret_regs > 4 {
        return Err(MalformedSegment);
    }
    let mut ret_pieces: [RetPiece; 4] = [RetPiece::default(); 4];
    for p in ret_pieces.iter_mut().take(ret_regs as usize) {
        let fp = r.u8()? != 0;
        let offset = r.u8()?;
        let bytes = r.u8()?;
        *p = RetPiece { fp, offset, bytes };
    }
    let has_ret_dest = flags & CALL_FLAG_RET_DEST != 0;
    if has_ret_dest {
        slots.push_slot(r.u32()?);
    }
    Ok(CallExtras {
        is_variadic: flags & CALL_FLAG_VARIADIC != 0,
        num_fixed,
        has_ret_dest,
        ret_regs,
        ret_pieces,
        sret: flags & CALL_FLAG_SRET != 0,
    })
}

/// The decoded extra fields of a call. `has_ret_dest` says whether the operand
/// fixup must fill a destination value in.
struct CallExtras {
    is_variadic: bool,
    num_fixed: u32,
    has_ret_dest: bool,
    ret_regs: u8,
    ret_pieces: [RetPiece; 4],
    sret: bool,
}

fn read_jump(
    r: &mut Reader<'_>,
    func: &mut Function,
    block_count: usize,
    slots: &mut FixupTable,
    dummy: Value,
) -> Result<Jump> {
    let target = check_block(r.u32()?, block_count)?;
    let n = r.u32()? as usize;
    if n > r.remaining() {
        return Err(MalformedSegment);
    }
    for _ in 0..n {
        slots.push_slot(r.u32()?);
    }
    let dummies = alloc::vec![dummy; n];
    let args = func.intern_values(&dummies);
    Ok(Jump { target, args })
}

fn read_term(
    r: &mut Reader<'_>,
    func: &mut Function,
    block: Block,
    block_count: usize,
    fixups: &mut FixupTable,
) -> Result<()> {
    let start = fixups.begin();
    let dummy = Value(u32::MAX);
    match r.u8()? {
        0 => {} // no terminator
        1 => {
            let count = r.u8()?;
            if count > 4 {
                return Err(MalformedSegment);
            }
            let mut dummies: [Value; 4] = [Value(u32::MAX); 4];
            for slot in dummies.iter_mut().take(count as usize) {
                fixups.push_slot(r.u32()?);
                *slot = dummy;
            }
            func.set_terminator(
                block,
                Terminator::Ret(Ret::many(&dummies[..count as usize])),
            );
        }
        2 => {
            let jump = read_jump(r, func, block_count, fixups, dummy)?;
            func.set_terminator(block, Terminator::Jump(jump));
        }
        _ => return Err(MalformedSegment),
    }
    fixups.commit_term(block, start);
    Ok(())
}

/// Tags that cannot appear in a result-less record: their decoded value has
/// no meaning without a result slot to type.
fn requires_result(tag: u8) -> bool {
    !matches!(
        tag,
        OP_STORE
            | OP_PREFETCH
            | OP_VA_START
            | OP_VA_END
            | OP_BARRIER
            | OP_IF
            | OP_MATMUL
            | OP_CALL
            | OP_CALL_INDIRECT
            | OP_ATOMIC_RMW
    )
}

/// The result canonicality a record's own has-result type must satisfy: a
/// low-float decode yields f32, an encode yields the unsigned payload width;
/// anything else is a stream this build cannot read.
fn canonical_low_float_result(
    func: &Function,
    result_type: Type,
    format: LowFloatFormat,
    decode_direction: bool,
) -> bool {
    match func.types.type_kind(result_type) {
        TypeKind::Float(f) => decode_direction && *f == FloatKind::F32,
        TypeKind::Int(i) => !decode_direction && !i.signed && i.bits == format.payload_bits(),
        _ => false,
    }
}

/// Same rule for the NVFP4 records: dequantize yields f32, quantize yields u8.
fn canonical_nvfp4_result(func: &Function, result_type: Type, dequantize_direction: bool) -> bool {
    match func.types.type_kind(result_type) {
        TypeKind::Float(f) => dequantize_direction && *f == FloatKind::F32,
        TypeKind::Int(i) => !dequantize_direction && !i.signed && i.bits == 8,
        _ => false,
    }
}

/// Append an instruction that produces a result, recording the new value.
fn append_res(
    func: &mut Function,
    block: Block,
    serial: &mut Vec<Value>,
    rty: Type,
    op: Opcode,
) -> Inst {
    let v = func.append_inst(block, rty, op);
    serial.push(v);
    func.defining_inst(v).unwrap()
}

/// Build the instruction with placeholder operands, recording the serials.
fn read_inst(
    r: &mut Reader<'_>,
    func: &mut Function,
    block: Block,
    type_map: &[Type],
    block_count: usize,
    serial: &mut Vec<Value>,
    inst_serial: &mut Vec<Inst>,
    fixups: &mut FixupTable,
) -> Result<()> {
    let start = fixups.begin();
    let has_result = r.u8()? != 0;
    let rty: Type = if has_result {
        map_type(type_map, type_map.len(), r.u32()?)?
    } else {
        // Placeholder; unused for result-less records (whose tag cannot need
        // a result type - that is `requires_result`'s rule).
        Type(u32::MAX)
    };
    let tag = r.u8()?;
    if !has_result && requires_result(tag) {
        return Err(MalformedSegment);
    }

    let dummy = Value(u32::MAX);

    let inst: Inst = match tag {
        OP_ICONST => append_res(func, block, serial, rty, Opcode::Iconst(r.u64()? as i64)),
        OP_FCONST => append_res(
            func,
            block,
            serial,
            rty,
            Opcode::Fconst(f64::from_bits(r.u64()?)),
        ),
        OP_FCONST128 => {
            let lo = r.u64()? as u128;
            let hi = r.u64()? as u128;
            append_res(func, block, serial, rty, Opcode::Fconst128((hi << 64) | lo))
        }
        OP_ARITH => {
            let op = try_bin_op(r.u8()?)?;
            for _ in 0..2 {
                fixups.push_slot(r.u32()?);
            }
            append_res(
                func,
                block,
                serial,
                rty,
                Opcode::Arith(Arith {
                    op,
                    lhs: dummy,
                    rhs: dummy,
                }),
            )
        }
        OP_ARITH_IMM => {
            let op = try_bin_op(r.u8()?)?;
            fixups.push_slot(r.u32()?);
            let imm = r.u64()? as i64;
            append_res(
                func,
                block,
                serial,
                rty,
                Opcode::ArithImm(ArithImm {
                    op,
                    lhs: dummy,
                    imm,
                }),
            )
        }
        OP_ICMP => {
            let op = try_cmp_op(r.u8()?)?;
            for _ in 0..2 {
                fixups.push_slot(r.u32()?);
            }
            append_res(
                func,
                block,
                serial,
                rty,
                Opcode::Icmp(Compare {
                    op,
                    lhs: dummy,
                    rhs: dummy,
                }),
            )
        }
        OP_SELECT => {
            for _ in 0..3 {
                fixups.push_slot(r.u32()?);
            }
            append_res(
                func,
                block,
                serial,
                rty,
                Opcode::Select(Select {
                    cond: dummy,
                    then: dummy,
                    else_: dummy,
                }),
            )
        }
        OP_STRUCT_NEW => {
            let n = r.u32()? as usize;
            if n > r.remaining() {
                return Err(MalformedSegment);
            }
            for _ in 0..n {
                fixups.push_slot(r.u32()?);
            }
            let dummies = alloc::vec![dummy; n];
            let list = func.intern_values(&dummies);
            append_res(
                func,
                block,
                serial,
                rty,
                Opcode::StructNew(StructNew { fields: list }),
            )
        }
        OP_EXTRACT => {
            fixups.push_slot(r.u32()?);
            let index = r.u32()?;
            append_res(
                func,
                block,
                serial,
                rty,
                Opcode::Extract(Extract {
                    aggregate: dummy,
                    index,
                }),
            )
        }
        OP_CONVERT => {
            fixups.push_slot(r.u32()?);
            append_res(
                func,
                block,
                serial,
                rty,
                Opcode::Convert(Convert { value: dummy }),
            )
        }
        OP_DECODE_LOW_FLOAT | OP_ENCODE_LOW_FLOAT => {
            if !has_result {
                return Err(MalformedSegment);
            }
            let format = try_low_float_format(r.u8()?)?;
            if !canonical_low_float_result(func, rty, format, tag == OP_DECODE_LOW_FLOAT) {
                return Err(MalformedSegment);
            }
            fixups.push_slot(r.u32()?);
            let cv = LowFloatConvert {
                value: dummy,
                format,
            };
            if tag == OP_DECODE_LOW_FLOAT {
                append_res(func, block, serial, rty, Opcode::DecodeLowFloat(cv))
            } else {
                append_res(func, block, serial, rty, Opcode::EncodeLowFloat(cv))
            }
        }
        OP_DEQUANTIZE_NVFP4 | OP_QUANTIZE_NVFP4 => {
            // The stream is UNTRUSTED: an unknown policy byte is malformed
            // input, never an invalid enum.
            let block_application = try_scale_application(r.u8()?)?;
            let global_application = try_scale_application(r.u8()?)?;
            if !has_result {
                return Err(MalformedSegment);
            }
            let dequantize_direction = tag == OP_DEQUANTIZE_NVFP4;
            if !canonical_nvfp4_result(func, rty, dequantize_direction) {
                return Err(MalformedSegment);
            }
            for _ in 0..3 {
                fixups.push_slot(r.u32()?);
            }
            let cv = NvFp4Convert {
                value: dummy,
                block_scale: dummy,
                global_scale: dummy,
                block_application,
                global_application,
            };
            if dequantize_direction {
                append_res(func, block, serial, rty, Opcode::DequantizeNvfp4(cv))
            } else {
                append_res(func, block, serial, rty, Opcode::QuantizeNvfp4(cv))
            }
        }
        OP_UNARY => {
            let uop = try_unary_op(r.u8()?)?;
            fixups.push_slot(r.u32()?);
            append_res(
                func,
                block,
                serial,
                rty,
                Opcode::Unary(Unary {
                    op: uop,
                    value: dummy,
                }),
            )
        }
        OP_ALLOCA => {
            let elem = map_type(type_map, type_map.len(), r.u32()?)?;
            append_res(func, block, serial, rty, Opcode::Alloca(Alloca { elem }))
        }
        OP_CALL | OP_CALL_INDIRECT => {
            let is_indirect = tag == OP_CALL_INDIRECT;
            if is_indirect {
                fixups.push_slot(r.u32()?); // target serial
            }
            let symbol = if is_indirect { 0 } else { r.u32()? };
            let n = r.u32()? as usize;
            if n > r.remaining() {
                return Err(MalformedSegment);
            }
            for _ in 0..n {
                fixups.push_slot(r.u32()?);
            }
            let extras = read_call_extras(r, fixups)?;
            let dummies = alloc::vec![dummy; n];
            let list = func.intern_values(&dummies);
            let inst = if is_indirect {
                let op = Opcode::CallIndirect(CallIndirect {
                    target: dummy,
                    args: list,
                    is_variadic: extras.is_variadic,
                    num_fixed: extras.num_fixed,
                    ret_dest: if extras.has_ret_dest {
                        Some(dummy)
                    } else {
                        None
                    },
                    ret_regs: extras.ret_regs,
                    ret_pieces: extras.ret_pieces,
                    sret: extras.sret,
                });
                if has_result {
                    append_res(func, block, serial, rty, op)
                } else {
                    func.append_stmt_raw(block, op)
                }
            } else {
                let op = Opcode::Call(Call {
                    symbol,
                    args: list,
                    is_variadic: extras.is_variadic,
                    num_fixed: extras.num_fixed,
                    ret_dest: if extras.has_ret_dest {
                        Some(dummy)
                    } else {
                        None
                    },
                    ret_regs: extras.ret_regs,
                    ret_pieces: extras.ret_pieces,
                    sret: extras.sret,
                });
                if has_result {
                    append_res(func, block, serial, rty, op)
                } else {
                    func.append_stmt_raw(block, op)
                }
            };
            let _ = symbol;
            inst
        }
        OP_LOAD => {
            fixups.push_slot(r.u32()?);
            let is_volatile = r.u8()? != 0;
            append_res(
                func,
                block,
                serial,
                rty,
                Opcode::Load(Load {
                    ptr: dummy,
                    volatile: is_volatile,
                }),
            )
        }
        OP_STORE => {
            for _ in 0..2 {
                fixups.push_slot(r.u32()?);
            }
            let is_volatile = r.u8()? != 0;
            func.append_stmt_raw(
                block,
                Opcode::Store(Store {
                    value: dummy,
                    ptr: dummy,
                    volatile: is_volatile,
                }),
            )
        }
        OP_PREFETCH => {
            fixups.push_slot(r.u32()?);
            func.append_stmt_raw(block, Opcode::Prefetch(Prefetch { ptr: dummy }))
        }
        OP_VA_START => {
            fixups.push_slot(r.u32()?);
            func.append_stmt_raw(block, Opcode::VaStart(VaStart { list: dummy }))
        }
        // `rty` was already decoded above, so `VaArg.ty` recovers straight
        // from it - see `write_inst`'s matching comment.
        OP_VA_ARG => {
            if !has_result {
                return Err(MalformedSegment);
            }
            fixups.push_slot(r.u32()?);
            append_res(
                func,
                block,
                serial,
                rty,
                Opcode::VaArg(VaArg {
                    list: dummy,
                    ty: rty,
                }),
            )
        }
        OP_VA_END => {
            fixups.push_slot(r.u32()?);
            func.append_stmt_raw(block, Opcode::VaEnd(VaEnd { list: dummy }))
        }
        OP_BARRIER => {
            // The stream is UNTRUSTED. An unknown scope byte is malformed
            // bitcode, never an invalid enum.
            let scope = match r.u8()? {
                0 => BarrierScope::Workgroup,
                1 => BarrierScope::Subgroup,
                _ => return Err(MalformedSegment),
            };
            func.append_stmt_raw(block, Opcode::Barrier(Barrier { scope }))
        }
        OP_ATOMIC_RMW => {
            // The stream is UNTRUSTED, so each of the three selector bytes
            // maps back with a range check and an unknown value is malformed
            // bitcode, never an out-of-range tag every later exhaustive switch
            // reads as undefined behavior.
            let op = try_atomic_op(r.u8()?)?;
            let ordering = try_atomic_ordering(r.u8()?)?;
            let scope = try_atomic_scope(r.u8()?)?;
            // An unknown flag bit is a stream this build cannot read: the
            // record's operand count would be wrong and every later record
            // would decode from the wrong offset.
            let flags = r.u8()?;
            if flags & !ATOMIC_FLAG_COMPARE != 0 {
                return Err(MalformedSegment);
            }
            let has_compare = flags & ATOMIC_FLAG_COMPARE != 0;
            for _ in 0..2 {
                fixups.push_slot(r.u32()?);
            }
            if has_compare {
                fixups.push_slot(r.u32()?);
            }
            let rmw = AtomicRmw {
                op,
                ptr: dummy,
                value: dummy,
                compare: if has_compare { Some(dummy) } else { None },
                ordering,
                scope,
            };
            // The result is OPTIONAL, so the record's own `has_result` byte
            // decides which form is rebuilt, exactly as it does for a `call`.
            if has_result {
                append_res(func, block, serial, rty, Opcode::AtomicRmw(rmw))
            } else {
                func.append_stmt_raw(block, Opcode::AtomicRmw(rmw))
            }
        }
        OP_DOT => {
            for _ in 0..3 {
                fixups.push_slot(r.u32()?);
            }
            append_res(
                func,
                block,
                serial,
                rty,
                Opcode::Dot(Dot {
                    acc: dummy,
                    a: dummy,
                    b: dummy,
                }),
            )
        }
        OP_REDUCE => {
            // The stream is untrusted: an unknown op byte is malformed
            // bitcode, never an invalid enum.
            let red_op = try_bin_op(r.u8()?)?;
            fixups.push_slot(r.u32()?);
            append_res(
                func,
                block,
                serial,
                rty,
                Opcode::Reduce(Reduce {
                    vector: dummy,
                    op: red_op,
                }),
            )
        }
        OP_SPLAT => {
            fixups.push_slot(r.u32()?);
            append_res(
                func,
                block,
                serial,
                rty,
                Opcode::Splat(Splat { scalar: dummy }),
            )
        }
        OP_MATMUL => {
            for _ in 0..3 {
                fixups.push_slot(r.u32()?);
            }
            let m = r.u16()?;
            let n = r.u16()?;
            let k = r.u16()?;
            let dtype = try_matmul_type(r.u8()?)?;
            let accumulate = r.u8()? != 0;
            let embedded = r.u8()? != 0;
            let has_input_signs = r.u8()? != 0;
            let input_signs = if has_input_signs {
                let a_unsigned = r.u8()? != 0;
                let b_unsigned = r.u8()? != 0;
                Some(InputSigns {
                    a_unsigned,
                    b_unsigned,
                })
            } else {
                None
            };
            let has_quant = r.u8()? != 0;
            let quant = if has_quant {
                let relu = r.u8()? != 0;
                let out = try_quant_out(r.u8()?)?;
                let zero_point = r.u32()? as i32;
                let bias_present = r.u8()? != 0;
                let bias = if bias_present {
                    let count = r.u32()? as usize;
                    if count > r.remaining() {
                        return Err(MalformedSegment);
                    }
                    let mut tmp = Vec::new();
                    for _ in 0..count {
                        tmp.push(r.u32()? as i32);
                    }
                    Some(func.intern_bias(&tmp))
                } else {
                    None
                };
                let scale_kind = r.u8()?;
                let scale = match scale_kind {
                    0 => MatMulScale::Scalar(r.u32()?),
                    1 => {
                        let count = r.u32()? as usize;
                        if count > r.remaining() {
                            return Err(MalformedSegment);
                        }
                        let mut tmp = Vec::new();
                        for _ in 0..count {
                            tmp.push(r.u32()?);
                        }
                        MatMulScale::PerColumn(func.intern_scales(&tmp))
                    }
                    _ => return Err(MalformedSegment),
                };
                Some(MatMulQuant {
                    scale,
                    relu,
                    out,
                    bias,
                    zero_point,
                })
            } else {
                None
            };
            func.append_stmt_raw(
                block,
                Opcode::Matmul(MatMul {
                    a: dummy,
                    b: dummy,
                    c: dummy,
                    m,
                    n,
                    k,
                    dtype,
                    accumulate,
                    embedded,
                    quant,
                    input_signs,
                }),
            )
        }
        OP_IF => {
            fixups.push_slot(r.u32()?); // cond
            let then_j = read_jump(r, func, block_count, fixups, dummy)?;
            let else_j = read_jump(r, func, block_count, fixups, dummy)?;
            func.append_stmt_raw(
                block,
                Opcode::If(super::function::If {
                    cond: dummy,
                    then: then_j,
                    else_: else_j,
                }),
            )
        }
        OP_GLOBAL_ADDR => {
            let symbol = r.u32()?;
            let via_got = r.u8()? != 0;
            append_res(
                func,
                block,
                serial,
                rty,
                Opcode::GlobalAddr(GlobalAddr { symbol, via_got }),
            )
        }
        _ => return Err(MalformedSegment),
    };
    let _ = dummy;

    fixups.commit_inst(inst, start);
    inst_serial.push(inst);
    Ok(())
}

fn try_atomic_op(raw: u8) -> Result<AtomicOp> {
    Ok(match raw {
        0 => AtomicOp::Add,
        1 => AtomicOp::Min,
        2 => AtomicOp::Max,
        3 => AtomicOp::BitAnd,
        4 => AtomicOp::BitOr,
        5 => AtomicOp::BitXor,
        6 => AtomicOp::Exchange,
        7 => AtomicOp::CompareExchange,
        _ => return Err(MalformedSegment),
    })
}

fn try_atomic_ordering(raw: u8) -> Result<AtomicOrdering> {
    Ok(match raw {
        0 => AtomicOrdering::Relaxed,
        1 => AtomicOrdering::Acquire,
        2 => AtomicOrdering::Release,
        3 => AtomicOrdering::AcqRel,
        4 => AtomicOrdering::SeqCst,
        _ => return Err(MalformedSegment),
    })
}

fn try_atomic_scope(raw: u8) -> Result<AtomicScope> {
    Ok(match raw {
        0 => AtomicScope::Workgroup,
        1 => AtomicScope::Device,
        2 => AtomicScope::System,
        _ => return Err(MalformedSegment),
    })
}
// === tests ===

fn try_bin_op(raw: u8) -> Result<BinOp> {
    Ok(match raw {
        0 => BinOp::Add,
        1 => BinOp::Sub,
        2 => BinOp::Mul,
        3 => BinOp::Div,
        4 => BinOp::Rem,
        5 => BinOp::BitAnd,
        6 => BinOp::BitOr,
        7 => BinOp::BitXor,
        8 => BinOp::Shl,
        9 => BinOp::Shr,
        10 => BinOp::Mulh,
        _ => return Err(MalformedSegment),
    })
}

fn try_cmp_op(raw: u8) -> Result<CmpOp> {
    Ok(match raw {
        0 => CmpOp::Eq,
        1 => CmpOp::Ne,
        2 => CmpOp::Lt,
        3 => CmpOp::Le,
        4 => CmpOp::Gt,
        5 => CmpOp::Ge,
        _ => return Err(MalformedSegment),
    })
}

fn try_unary_op(raw: u8) -> Result<UnaryOp> {
    Ok(match raw {
        0 => UnaryOp::Reinterpret,
        1 => UnaryOp::Sqrt,
        2 => UnaryOp::Ceil,
        3 => UnaryOp::Floor,
        4 => UnaryOp::Trunc,
        5 => UnaryOp::Nearest,
        _ => return Err(MalformedSegment),
    })
}

fn try_low_float_format(raw: u8) -> Result<LowFloatFormat> {
    Ok(match raw {
        0 => LowFloatFormat::Bf16,
        1 => LowFloatFormat::F8E4M3,
        2 => LowFloatFormat::F8E5M2,
        _ => return Err(MalformedSegment),
    })
}

fn try_scale_application(raw: u8) -> Result<ScaleApplication> {
    Ok(match raw {
        0 => ScaleApplication::Multiply,
        1 => ScaleApplication::Divide,
        _ => return Err(MalformedSegment),
    })
}

fn try_matmul_type(raw: u8) -> Result<MatMulType> {
    Ok(match raw {
        0 => MatMulType::Fp32,
        1 => MatMulType::Fp16,
        2 => MatMulType::Int8,
        3 => MatMulType::Uint8,
        _ => return Err(MalformedSegment),
    })
}

fn try_quant_out(raw: u8) -> Result<MatMulQuantOut> {
    Ok(match raw {
        0 => MatMulQuantOut::I8,
        1 => MatMulQuantOut::U8,
        _ => return Err(MalformedSegment),
    })
}

mod tests {
    #![allow(dead_code, unused_macros)]
    /// Concatenate byte arrays / slices into one stream for hand-built malformed
    /// modules.
    macro_rules! bconcat {
    ($($part:expr),* $(,)?) => {{
        let mut out: Vec<u8> = Vec::new();
        $( out.extend_from_slice(&$part[..]); )*
        out
    }};
}

    use super::*;
    use alloc::format;

    /// The stream header a hand-built test module starts with: the magic, no
    /// whole-function flags, and a zero fixed-parameter count.
    const TEST_HEADER: &[u8] = b"VBC1\x00\x00\x00\x00\x00";

    /// A structural printer for the oracle: renders a function's blocks,
    /// params, instructions (operands numbered by canonical serial) and
    /// attributes. A pure function of structure, so `print(decode(encode(f)))
    /// == print(f)`.
    fn print_function(func: &Function) -> String {
        let mut serial = vec![NO_SERIAL; func.value_count()];
        {
            let mut next: u32 = 0;
            for bi in 0..func.block_count() {
                let block = Block(bi as u32);
                for p in func.block_params(block) {
                    serial[p.0 as usize] = next;
                    next += 1;
                }
                for inst in func.block_insts(block) {
                    if let Some(r) = func.inst_result(*inst) {
                        serial[r.0 as usize] = next;
                        next += 1;
                    }
                }
            }
        }
        let vn = |v: Value| format!("v{}", serial[v.0 as usize]);
        let mut out = String::new();
        let tn = |t: Type, out: &mut String| {
            func.types.render(t, out);
        };
        for bi in 0..func.block_count() {
            let block = Block(bi as u32);
            out.push_str(&format!("block{}(", bi));
            for (i, p) in func.block_params(block).iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str("param ");
                tn(func.value_type(*p), &mut out);
                out.push(' ');
                out.push_str(&vn(*p));
            }
            out.push_str("):\n");
            for inst in func.block_insts(block) {
                out.push_str("  ");
                if let Some(r) = func.inst_result(*inst) {
                    out.push_str(&vn(r));
                    out.push(':');
                    tn(func.value_type(r), &mut out);
                    out.push_str(" = ");
                }
                match func.opcode(*inst) {
                    Opcode::Arith(a) => out.push_str(&format!(
                        "arith {} {} {}",
                        a.op.symbol(),
                        vn(a.lhs),
                        vn(a.rhs)
                    )),
                    Opcode::ArithImm(a) => {
                        out.push_str(&format!("arith {} {}, {}", a.op.symbol(), vn(a.lhs), a.imm))
                    }
                    Opcode::Icmp(c) => out.push_str(&format!(
                        "icmp {} {} {}",
                        c.op.symbol(),
                        vn(c.lhs),
                        vn(c.rhs)
                    )),
                    Opcode::Convert(c) => out.push_str(&format!("convert {}", vn(c.value))),
                    Opcode::Call(c) => {
                        out.push_str(&format!("call \"{}\"", func.symbol_name(c.symbol)));
                        for a in func.value_list(c.args) {
                            out.push_str(&format!(" {}", vn(*a)));
                        }
                    }
                    Opcode::If(cf) => out.push_str(&format!(
                        "if {} -> block{}, else -> block{}",
                        vn(cf.cond),
                        cf.then.target.0,
                        cf.else_.target.0
                    )),
                    other => out.push_str(&format!(
                        "op<diagramatically {:?}>",
                        discriminant_of(&other)
                    )),
                }
                out.push('\n');
            }
            match func.terminator(block) {
                None => out.push_str("  ret\n"),
                Some(Terminator::Ret(r)) => {
                    out.push_str("  ret");
                    for v in r.slice() {
                        out.push_str(&format!(" {}", vn(*v)));
                    }
                    out.push('\n');
                }
                Some(Terminator::Jump(j)) => {
                    out.push_str(&format!("  jump block{}", j.target.0));
                    for a in func.block_args(j) {
                        out.push_str(&format!(" {}", vn(*a)));
                    }
                    out.push('\n');
                }
            }
        }
        for entry in func.attribute_entries() {
            out.push_str(&format!("attr {:?}\n", entry.target));
        }
        out
    }

    fn discriminant_of(op: &Opcode) -> core::mem::Discriminant<Opcode> {
        core::mem::discriminant(op)
    }

    /// Byte-identical no-op assertion: re-encoding what a decode built must
    /// reproduce the original stream bit for bit, and the print oracle must
    /// hold.
    fn expect_round_trip(func: &Function) -> Function {
        let bytes = encode(func).unwrap();
        let decoded = decode(&bytes).unwrap();
        let again = encode(&decoded).unwrap();
        assert_eq!(bytes, again, "re-encode of decoded function differs");
        assert_eq!(
            print_function(func),
            print_function(&decoded),
            "print(decode(encode(f))) != print(f)"
        );
        decoded
    }

    #[test]
    fn round_trips_a_function_through_bitcode() {
        // f(x, y): if x<y { jump m(x*y) } else { jump m(x+y) }, m(z): ret z + 1
        use super::super::function::EdgeDesc;
        let mut func = Function::new();
        let i32_t = func.types.intern(TypeKind::Int(IntDesc {
            signed: true,
            bits: 32,
        }));
        let bool_t = func.types.intern(TypeKind::Bool);
        let entry = func.append_block();
        let merge = func.append_block();
        let x = func.append_block_param(entry, i32_t);
        let y = func.append_block_param(entry, i32_t);
        let z = func.append_block_param(merge, i32_t);
        let c = func.append_inst(
            entry,
            bool_t,
            Opcode::Icmp(Compare {
                op: CmpOp::Lt,
                lhs: x,
                rhs: y,
            }),
        );
        let prod = func.append_inst(
            entry,
            i32_t,
            Opcode::Arith(Arith {
                op: BinOp::Mul,
                lhs: x,
                rhs: y,
            }),
        );
        let sum = func.append_inst(
            entry,
            i32_t,
            Opcode::Arith(Arith {
                op: BinOp::Add,
                lhs: x,
                rhs: y,
            }),
        );
        func.append_if(
            entry,
            c,
            EdgeDesc {
                target: merge,
                args: alloc::vec![prod],
            },
            EdgeDesc {
                target: merge,
                args: alloc::vec![sum],
            },
        );
        let r = func.append_inst(
            merge,
            i32_t,
            Opcode::ArithImm(ArithImm {
                op: BinOp::Add,
                lhs: z,
                imm: 1,
            }),
        );
        func.set_terminator(merge, Terminator::Ret(Ret::one(r)));

        let decoded = expect_round_trip(&func);
        let _ = decoded;
    }

    #[test]
    fn round_trips_a_prefetch_through_bitcode() {
        let mut func = Function::new();
        let ptr_t = func.types.ptr_global();
        let entry = func.append_block();
        let p = func.append_block_param(entry, ptr_t);
        func.append_prefetch(entry, p);
        func.set_terminator(entry, Terminator::Ret(Ret::none()));

        let bytes = encode(&func).unwrap();
        let decoded = decode(&bytes).unwrap();

        let insts = decoded.block_insts(entry);
        let op = decoded.opcode(insts[insts.len() - 1]);
        assert!(matches!(op, Opcode::Prefetch(_)));
        assert_eq!(p, op_prefetch_ptr(&op));

        let _ = expect_round_trip(&func);
    }

    fn op_prefetch_ptr(op: &Opcode) -> Value {
        match op {
            super::super::function::Opcode::Prefetch(pf) => pf.ptr,
            _ => unreachable!(),
        }
    }

    #[test]
    fn round_trips_a_barrier_and_its_scope() {
        let mut func = Function::new();
        let entry = func.append_block();
        func.append_barrier(entry, BarrierScope::Workgroup);
        func.append_barrier(entry, BarrierScope::Subgroup);
        func.set_terminator(entry, Terminator::Ret(Ret::none()));

        let bytes = encode(&func).unwrap();
        let decoded = decode(&bytes).unwrap();

        let insts = decoded.block_insts(entry);
        assert_eq!(
            BarrierScope::Workgroup,
            match decoded.opcode(insts[0]) {
                super::super::function::Opcode::Barrier(b) => b.scope,
                _ => unreachable!(),
            }
        );
        assert_eq!(
            BarrierScope::Subgroup,
            match decoded.opcode(insts[1]) {
                super::super::function::Opcode::Barrier(b) => b.scope,
                _ => unreachable!(),
            }
        );

        // An unknown barrier scope byte is rejected as malformed bitcode, not
        // turned into an out-of-range enum tag.
        let mut mutable = bytes.clone();
        let mut patched = false;
        for i in 0..mutable.len() - 1 {
            if mutable[i] == OP_BARRIER && mutable[i + 1] == 0 {
                mutable[i + 1] = 0xff;
                patched = true;
                break;
            }
        }
        assert!(patched);
        assert!(decode(&mutable).is_err());
    }

    #[test]
    fn round_trips_a_global_addr_via_got_flag_both_directions() {
        let mut func = Function::new();
        let ptr_t = func.types.ptr_global();
        let entry = func.append_block();
        func.append_global_addr(entry, ptr_t, "D");
        func.append_global_addr_got(entry, ptr_t, "G");
        func.set_terminator(entry, Terminator::Ret(Ret::none()));

        let bytes = encode(&func).unwrap();
        let decoded = decode(&bytes).unwrap();

        let insts = decoded.block_insts(entry);
        assert_eq!(
            Some(false),
            match decoded.opcode(insts[0]) {
                super::super::function::Opcode::GlobalAddr(g) => Some(g.via_got),
                _ => None,
            }
        );
        assert_eq!(
            Some(true),
            match decoded.opcode(insts[1]) {
                super::super::function::Opcode::GlobalAddr(g) => Some(g.via_got),
                _ => None,
            }
        );

        let _ = expect_round_trip(&func);
    }

    #[test]
    fn round_trips_an_atomic_read_modify_write_field_by_field() {
        // FIELD BY FIELD, not by comparing printed text: a printed comparison
        // hides a field the printer never prints.
        let mut func = Function::new();
        let i32_t = func.types.intern(TypeKind::Int(IntDesc {
            signed: true,
            bits: 32,
        }));
        let ptr_t = func.types.ptr_global();
        let entry = func.append_block();
        let p = func.append_block_param(entry, ptr_t);
        let v = func.append_block_param(entry, i32_t);
        let old = func.append_atomic_rmw(
            entry,
            AtomicRmw {
                op: AtomicOp::Min,
                ptr: p,
                value: v,
                compare: None,
                ordering: AtomicOrdering::AcqRel,
                scope: AtomicScope::Device,
            },
        );
        func.append_atomic_rmw_stmt(
            entry,
            AtomicRmw {
                op: AtomicOp::BitXor,
                ptr: p,
                value: v,
                compare: None,
                ordering: AtomicOrdering::Relaxed,
                scope: AtomicScope::System,
            },
        );
        func.set_terminator(entry, Terminator::Ret(Ret::one(old)));

        let bytes = encode(&func).unwrap();
        let decoded = decode(&bytes).unwrap();

        let insts = decoded.block_insts(entry);
        assert_eq!(2, insts.len());

        let reading = match decoded.opcode(insts[0]) {
            super::super::function::Opcode::AtomicRmw(a) => a,
            _ => unreachable!(),
        };
        assert_eq!(AtomicOp::Min, reading.op);
        assert_eq!(AtomicOrdering::AcqRel, reading.ordering);
        assert_eq!(AtomicScope::Device, reading.scope);
        assert_eq!(None, reading.compare);
        let params = decoded.block_params(entry);
        assert_eq!(params[0], reading.ptr);
        assert_eq!(params[1], reading.value);
        // The reading form keeps its result, with the operand's type.
        let result = decoded.inst_result(insts[0]).unwrap();
        assert_eq!(decoded.value_type(params[1]), decoded.value_type(result));

        let reduction = match decoded.opcode(insts[1]) {
            super::super::function::Opcode::AtomicRmw(a) => a,
            _ => unreachable!(),
        };
        assert_eq!(AtomicOp::BitXor, reduction.op);
        assert_eq!(AtomicOrdering::Relaxed, reduction.ordering);
        assert_eq!(AtomicScope::System, reduction.scope);
        assert_eq!(params[0], reduction.ptr);
        assert_eq!(params[1], reduction.value);
        // The reduction form comes back RESULT-LESS.
        assert_eq!(None, decoded.inst_result(insts[1]));

        let _ = expect_round_trip(&func);
    }

    #[test]
    fn round_trips_a_compare_exchange_and_its_compare_operand() {
        let mut func = Function::new();
        let i32_t = func.types.intern(TypeKind::Int(IntDesc {
            signed: true,
            bits: 32,
        }));
        let ptr_t = func.types.ptr_global();
        let entry = func.append_block();
        let p = func.append_block_param(entry, ptr_t);
        let desired = func.append_block_param(entry, i32_t);
        let expected = func.append_block_param(entry, i32_t);
        let old = func.append_atomic_rmw(
            entry,
            AtomicRmw {
                op: AtomicOp::CompareExchange,
                ptr: p,
                value: desired,
                compare: Some(expected),
                ordering: AtomicOrdering::SeqCst,
                scope: AtomicScope::Workgroup,
            },
        );
        func.set_terminator(entry, Terminator::Ret(Ret::one(old)));

        let bytes = encode(&func).unwrap();
        let decoded = decode(&bytes).unwrap();

        let params = decoded.block_params(entry);
        let cas = match decoded.opcode(decoded.block_insts(entry)[0]) {
            super::super::function::Opcode::AtomicRmw(a) => a,
            _ => unreachable!(),
        };
        assert_eq!(AtomicOp::CompareExchange, cas.op);
        assert_eq!(params[0], cas.ptr);
        assert_eq!(params[1], cas.value);
        // The compare operand is the field a two-slot record would have dropped.
        assert_eq!(params[2], cas.compare.unwrap());
        assert_eq!(AtomicOrdering::SeqCst, cas.ordering);
        assert_eq!(AtomicScope::Workgroup, cas.scope);
    }

    #[test]
    fn an_unknown_atomic_selector_byte_is_rejected() {
        let mut func = Function::new();
        let i32_t = func.types.intern(TypeKind::Int(IntDesc {
            signed: true,
            bits: 32,
        }));
        let ptr_t = func.types.ptr_global();
        let entry = func.append_block();
        let p = func.append_block_param(entry, ptr_t);
        let v = func.append_block_param(entry, i32_t);
        func.append_atomic_rmw_stmt(
            entry,
            AtomicRmw {
                op: AtomicOp::Add,
                ptr: p,
                value: v,
                compare: None,
                ordering: AtomicOrdering::Relaxed,
                scope: AtomicScope::Workgroup,
            },
        );
        func.set_terminator(entry, Terminator::Ret(Ret::one(v)));

        let bytes = encode(&func).unwrap();

        // The record is the tag byte followed by op, ordering, scope and the
        // flag byte, all four zero for the operation built above.
        let mut at: Option<usize> = None;
        for i in 0..bytes.len() - 4 {
            if bytes[i] == OP_ATOMIC_RMW
                && bytes[i + 1] == 0
                && bytes[i + 2] == 0
                && bytes[i + 3] == 0
                && bytes[i + 4] == 0
            {
                at = Some(i);
                break;
            }
        }
        assert!(at.is_some());

        // The unpatched stream decodes.
        let _ = decode(&bytes).unwrap();

        // Byte 4 is the flag byte; 0x02 leaves the record's length alone, so
        // only the unknown-bit check can refuse it.
        for (field, byte, _desc) in [
            (1usize, 0xffu8, 0),
            (2, 0xff, 0),
            (3, 0xff, 0),
            (4, 0x02, 0),
        ] {
            let mut mutable = bytes.clone();
            mutable[at.unwrap() + field] = byte;
            assert!(decode(&mutable).is_err());
        }
    }

    #[test]
    fn round_trips_a_matmul_per_column_quant_epilogue() {
        let mut func = Function::new();
        let ptr_t = func.types.ptr_global();
        let entry = func.append_block();
        let a_val = func.append_block_param(entry, ptr_t);
        let b_val = func.append_block_param(entry, ptr_t);
        let c_val = func.append_block_param(entry, ptr_t);
        let scales: Vec<u32> = vec![0x3F800000, 0x3F000000, 0x3E800000, 0x40000000];
        // Exercises .u8 here (the scalar round-trip covers the default .i8).
        func.append_matmul_quant_per_column(
            entry,
            a_val,
            b_val,
            c_val,
            8,
            4,
            4,
            MatMulType::Int8,
            true,
            true,
            MatMulQuantOut::U8,
            &scales,
        );
        func.set_terminator(entry, Terminator::Ret(Ret::none()));

        let bytes = encode(&func).unwrap();
        let decoded = decode(&bytes).unwrap();

        let insts = decoded.block_insts(entry);
        let mm = match decoded.opcode(insts[insts.len() - 1]) {
            super::super::function::Opcode::Matmul(m) => m,
            _ => unreachable!(),
        };
        let q = mm.quant.as_ref().unwrap();
        assert!(matches!(q.scale, MatMulScale::PerColumn(_)));
        assert_eq!(MatMulQuantOut::U8, q.out);
        assert_eq!(
            scales,
            decoded.scale_list(match q.scale {
                MatMulScale::PerColumn(h) => h,
                _ => unreachable!(),
            })
        );
        assert!(q.relu);

        let _ = expect_round_trip(&func);
    }

    #[test]
    fn rejects_truncated_bitcode() {
        assert!(decode(b"VBC1\x01").is_err());
        assert!(decode(b"nope").is_err());
    }

    #[test]
    fn round_trips_va_start_va_arg_va_end_through_bitcode() {
        let mut func = Function::new();
        let i32_t = func.types.intern(TypeKind::Int(IntDesc {
            signed: true,
            bits: 32,
        }));
        let ptr_t = func.types.ptr_global();
        let entry = func.append_block();
        let list = func.append_block_param(entry, ptr_t);
        func.append_va_start(entry, list);
        let v = func.append_va_arg(entry, list, i32_t);
        func.append_va_end(entry, list);
        func.set_terminator(entry, Terminator::Ret(Ret::one(v)));

        let bytes = encode(&func).unwrap();
        let decoded = decode(&bytes).unwrap();

        let insts = decoded.block_insts(entry);
        assert_eq!(
            Some(list),
            match decoded.opcode(insts[0]) {
                super::super::function::Opcode::VaStart(vs) => Some(vs.list),
                _ => None,
            }
        );
        let va = match decoded.opcode(insts[1]) {
            super::super::function::Opcode::VaArg(va) => va,
            _ => unreachable!(),
        };
        assert_eq!(list, va.list);
        assert_eq!(i32_t, va.ty);
        assert_eq!(
            Some(list),
            match decoded.opcode(insts[2]) {
                super::super::function::Opcode::VaEnd(ve) => Some(ve.list),
                _ => None,
            }
        );
        let _ = expect_round_trip(&func);
    }

    #[test]
    fn rejects_the_untrusted_bytes_and_out_of_range_indices() {
        // An unknown whole-function flag bit: the flag byte follows the magic.
        {
            let mut patched = TEST_HEADER.to_vec();
            patched[4] = 0xff;
            assert!(decode(&patched).is_err());
        }

        // A self-referential vector type: type 0's element index points at
        // type 0, not yet decoded.
        let self_ref = bconcat![
            TEST_HEADER,
            [1, 0, 0, 0], // type_count = 1
            [4],          // vector
            [1, 0, 0, 0], // len
            [0, 0, 0, 0], // elem = type 0 (not yet decoded)
        ];
        assert!(decode(&self_ref[..]).is_err());

        // Same, but an index past the whole table.
        let past_end = bconcat![
            TEST_HEADER,
            [1, 0, 0, 0],
            [4],
            [1, 0, 0, 0],
            [5, 0, 0, 0], // elem = type 5
        ];
        assert!(decode(&past_end[..]).is_err());

        // An unknown float-kind byte, no such FloatKind.
        let bad_float = bconcat![
            TEST_HEADER,
            [1, 0, 0, 0], // type_count = 1
            [2],          // float
            [3],          // float-kind byte 3 does not exist
        ];
        assert!(decode(&bad_float[..]).is_err());

        // An unknown arith operator byte, no such BinOp.
        let bad_arith = bconcat![
            TEST_HEADER,
            [1, 0, 0, 0],  // type_count = 1
            [1, 0, 32, 0], // int, signed, 32 bits
            [0, 0, 0, 0],  // sym_count = 0
            [1, 0, 0, 0],  // block_count = 1
            [1, 0, 0, 0],  // block 0 param_count = 1
            [0, 0, 0, 0],  // param 0 type 0
            [1, 0, 0, 0],  // inst_count = 1
            [1],           // has_result = 1
            [0, 0, 0, 0],  // result type 0
            [2],           // tag = op_arith
            [0xff],        // operator byte: no such BinOp
        ];
        assert!(decode(&bad_arith[..]).is_err());

        // One attribute target number that names no value.
        let bad_attr = bconcat![
            TEST_HEADER,
            [0, 0, 0, 0], // type_count = 0
            [0, 0, 0, 0], // sym_count = 0
            [1, 0, 0, 0], // block_count = 1
            [0, 0, 0, 0], // param_count = 0
            [0, 0, 0, 0], // inst_count = 0
            [0],          // no terminator
            [1, 0, 0, 0], // attr_count = 1
            [3],          // target kind = value
            [7, 0, 0, 0], // value number 7, which does not exist
            [0],          // attribute body = inline
        ];
        assert!(decode(&bad_attr[..]).is_err());

        // An unknown attribute body tag.
        let bad_body = bconcat![
            TEST_HEADER,
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0],
            [1, 0, 0, 0],
            [0],    // target kind = func
            [0x63], // body tag: no such attribute
        ];
        assert!(decode(&bad_body[..]).is_err());

        // An out-of-range endianness byte.
        let bad_endian = bconcat![
            TEST_HEADER,
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0],
            [1, 0, 0, 0],
            [0],    // target kind = func
            [4],    // attribute body = endian
            [0xff], // endianness byte, no such order
        ];
        assert!(decode(&bad_endian[..]).is_err());
    }

    #[test]
    fn the_header_len_names_the_first_type_record() {
        // The stream header is magic + flag byte + fixed-param count; with one
        // type (a pointer), that record lands exactly there.
        let mut func = Function::new();
        let _ = func.types.ptr_global();
        let b = func.append_block();
        func.set_terminator(b, Terminator::Ret(Ret::none()));

        let bytes = encode(&func).unwrap();
        assert_eq!(3u8, bytes[HEADER_LEN + 4]);

        // And its address-space byte follows the tag, if patched to an unknown
        // value the stream is malformed.
        let space_offset = HEADER_LEN + 4 + 1;
        assert_eq!(0, bytes[space_offset]);
        let patched = {
            let mut p = bytes.clone();
            p[space_offset] = 0xff;
            p
        };
        assert!(decode(&patched).is_err());
    }

    #[test]
    fn a_pointer_address_space_survives_a_round_trip() {
        // Every pointer decoded as global would silently change a shared one's
        // address space across a round trip.
        let mut func = Function::new();
        let s = func.types.intern(TypeKind::Ptr(AddressSpace::Shared));
        let g = func.types.ptr_global();
        let i32_t = func.types.intern(TypeKind::Int(IntDesc {
            signed: true,
            bits: 32,
        }));
        let b = func.append_block();
        let ps = func.append_block_param(b, s);
        let pg = func.append_block_param(b, g);
        let n = func.append_block_param(b, i32_t);
        func.set_terminator(b, Terminator::Ret(Ret::one(n)));

        let bytes = encode(&func).unwrap();
        let back = decode(&bytes).unwrap();
        assert!(matches!(
            back.types.type_kind(back.value_type(ps)),
            TypeKind::Ptr(AddressSpace::Shared)
        ));
        assert!(matches!(
            back.types.type_kind(back.value_type(pg)),
            TypeKind::Ptr(AddressSpace::Global)
        ));
        assert_ne!(back.value_type(ps), back.value_type(pg));
    }

    #[test]
    fn a_two_value_ret_round_trips_through_bitcode() {
        let mut func = Function::new();
        let i32_t = func.types.intern(TypeKind::Int(IntDesc {
            signed: true,
            bits: 32,
        }));
        let entry = func.append_block();
        let a = func.append_block_param(entry, i32_t);
        let b = func.append_block_param(entry, i32_t);
        func.set_terminator(entry, Terminator::Ret(Ret::many(&[a, b])));

        let bytes = encode(&func).unwrap();
        let decoded = decode(&bytes).unwrap();
        let decoded_ret = match decoded.terminator(entry).unwrap() {
            Terminator::Ret(r) => r,
            _ => unreachable!(),
        };
        assert_eq!(2, decoded_ret.count);
        let params = decoded.block_params(entry);
        assert_eq!(params[0], decoded_ret.values[0]);
        assert_eq!(params[1], decoded_ret.values[1]);
        let _ = expect_round_trip(&func);
    }
}
