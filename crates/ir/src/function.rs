extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use super::attribute::Attribute;
use super::low_float::Format as LowFloatFormat;
use super::nvfp4::ScaleApplication;
use super::types::{Type, TypeKind, TypeTable};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Block(pub u32);

impl Block {
    pub fn from_u32(i: u32) -> Self {
        Self(i)
    }
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Value(pub u32);

impl Value {
    pub fn from_u32(i: u32) -> Self {
        Self(i)
    }
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Inst(pub u32);

impl Inst {
    pub fn from_u32(i: u32) -> Self {
        Self(i)
    }
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    Mulh,
}

impl BinOp {
    pub fn symbol(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Rem => "%",
            BinOp::BitAnd => "&",
            BinOp::BitOr => "|",
            BinOp::BitXor => "^",
            BinOp::Shl => "<<",
            BinOp::Shr => ">>",
            BinOp::Mulh => "*h",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Arith {
    pub op: BinOp,
    pub lhs: Value,
    pub rhs: Value,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ArithImm {
    pub op: BinOp,
    pub lhs: Value,
    pub imm: i64,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl CmpOp {
    pub fn symbol(self) -> &'static str {
        match self {
            CmpOp::Eq => "==",
            CmpOp::Ne => "!=",
            CmpOp::Lt => "<",
            CmpOp::Le => "<=",
            CmpOp::Gt => ">",
            CmpOp::Ge => ">=",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Compare {
    pub op: CmpOp,
    pub lhs: Value,
    pub rhs: Value,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Select {
    pub cond: Value,
    pub then: Value,
    pub else_: Value,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct StructNew {
    pub fields: ValueList,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Extract {
    pub aggregate: Value,
    pub index: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Convert {
    pub value: Value,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LowFloatConvert {
    pub value: Value,
    pub format: LowFloatFormat,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct NvFp4Convert {
    pub value: Value,
    pub block_scale: Value,
    pub global_scale: Value,
    pub block_application: ScaleApplication,
    pub global_application: ScaleApplication,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum UnaryOp {
    Reinterpret,
    Sqrt,
    Ceil,
    Floor,
    Trunc,
    Nearest,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Unary {
    pub op: UnaryOp,
    pub value: Value,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Alloca {
    pub elem: Type,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct RetPiece {
    pub fp: bool,
    pub offset: u8,
    pub bytes: u8,
}

impl Default for RetPiece {
    fn default() -> Self {
        Self {
            fp: false,
            offset: 0,
            bytes: 8,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Call {
    pub symbol: u32,
    pub args: ValueList,
    pub is_variadic: bool,
    pub num_fixed: u32,
    pub ret_dest: Option<Value>,
    pub ret_regs: u8,
    pub ret_pieces: [RetPiece; 4],
    pub sret: bool,
}

impl Call {
    fn direct(symbol: u32, args: ValueList) -> Self {
        Self {
            symbol,
            args,
            is_variadic: false,
            num_fixed: 0,
            ret_dest: None,
            ret_regs: 0,
            ret_pieces: [RetPiece::default(); 4],
            sret: false,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CallIndirect {
    pub target: Value,
    pub args: ValueList,
    pub is_variadic: bool,
    pub num_fixed: u32,
    pub ret_dest: Option<Value>,
    pub ret_regs: u8,
    pub ret_pieces: [RetPiece; 4],
    pub sret: bool,
}

impl CallIndirect {
    fn direct(target: Value, args: ValueList) -> Self {
        Self {
            target,
            args,
            is_variadic: false,
            num_fixed: 0,
            ret_dest: None,
            ret_regs: 0,
            ret_pieces: [RetPiece::default(); 4],
            sret: false,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GlobalAddr {
    pub symbol: u32,
    pub via_got: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Load {
    pub ptr: Value,
    pub volatile: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Store {
    pub value: Value,
    pub ptr: Value,
    pub volatile: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Prefetch {
    pub ptr: Value,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct VaStart {
    pub list: Value,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct VaArg {
    pub list: Value,
    pub ty: Type,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct VaEnd {
    pub list: Value,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Dot {
    pub acc: Value,
    pub a: Value,
    pub b: Value,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Reduce {
    pub vector: Value,
    pub op: BinOp,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Splat {
    pub scalar: Value,
}

/// Bitcode-relevant: tag values pinned (fp32=0, fp16=1, int8=2, uint8=3).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[repr(u8)]
pub enum MatMulType {
    Fp32 = 0,
    Fp16 = 1,
    Int8 = 2,
    Uint8 = 3,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MatMulScale {
    Scalar(u32),
    PerColumn(ScaleList),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub enum MatMulQuantOut {
    #[default]
    I8,
    U8,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct MatMulQuant {
    pub scale: MatMulScale,
    pub relu: bool,
    pub out: MatMulQuantOut,
    pub bias: Option<BiasList>,
    pub zero_point: i32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct InputSigns {
    pub a_unsigned: bool,
    pub b_unsigned: bool,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct MatMul {
    pub a: Value,
    pub b: Value,
    pub c: Value,
    pub m: u16,
    pub n: u16,
    pub k: u16,
    pub dtype: MatMulType,
    pub accumulate: bool,
    pub embedded: bool,
    pub quant: Option<MatMulQuant>,
    pub input_signs: Option<InputSigns>,
}

/// Bitcode-relevant: encoded as one byte, tag values pinned.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[repr(u8)]
pub enum BarrierScope {
    Workgroup = 0,
    Subgroup = 1,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Barrier {
    pub scope: BarrierScope,
}

/// Bitcode-relevant: names map 1:1 onto NVIDIA encode ops, order pinned.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[repr(u8)]
pub enum AtomicOp {
    Add = 0,
    Min = 1,
    Max = 2,
    BitAnd = 3,
    BitOr = 4,
    BitXor = 5,
    Exchange = 6,
    CompareExchange = 7,
}

/// Bitcode-relevant: encoded as one byte, tag values pinned.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[repr(u8)]
pub enum AtomicOrdering {
    Relaxed = 0,
    Acquire = 1,
    Release = 2,
    AcqRel = 3,
    SeqCst = 4,
}

/// Bitcode-relevant: encoded as one byte, tag values pinned.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[repr(u8)]
pub enum AtomicScope {
    Workgroup = 0,
    Device = 1,
    System = 2,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AtomicRmw {
    pub op: AtomicOp,
    pub ptr: Value,
    pub value: Value,
    pub compare: Option<Value>,
    pub ordering: AtomicOrdering,
    pub scope: AtomicScope,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ValueList {
    pub start: u32,
    pub len: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ScaleList {
    pub start: u32,
    pub len: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BiasList {
    pub start: u32,
    pub len: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Jump {
    pub target: Block,
    pub args: ValueList,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct If {
    pub cond: Value,
    pub then: Jump,
    pub else_: Jump,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Ret {
    pub values: [Value; 4],
    pub count: u8,
}

impl Ret {
    pub fn none() -> Self {
        Self {
            values: [Value(u32::MAX); 4],
            count: 0,
        }
    }

    pub fn one(v: Value) -> Self {
        Self {
            values: [v, Value(u32::MAX), Value(u32::MAX), Value(u32::MAX)],
            count: 1,
        }
    }

    pub fn many(vals: &[Value]) -> Self {
        assert!(vals.len() <= 4);
        let mut r = Self::none();
        r.count = vals.len() as u8;
        for (i, v) in vals.iter().enumerate() {
            r.values[i] = *v;
        }
        r
    }

    pub fn slice(&self) -> &[Value] {
        &self.values[0..self.count as usize]
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Terminator {
    Ret(Ret),
    Jump(Jump),
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct EdgeDesc {
    pub target: Block,
    pub args: Vec<Value>,
}

impl EdgeDesc {
    pub fn bare(target: Block) -> Self {
        Self {
            target,
            args: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AttrTarget {
    Func,
    Block(Block),
    Inst(Inst),
    Value(Value),
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct AttrEntry {
    pub target: AttrTarget,
    pub attr: Attribute,
}

pub struct AttrIterator<'a> {
    entries: &'a [AttrEntry],
    target: AttrTarget,
    index: usize,
}

impl<'a> AttrIterator<'a> {
    pub fn next_attr(&mut self) -> Option<&'a Attribute> {
        while self.index < self.entries.len() {
            let entry = &self.entries[self.index];
            self.index += 1;
            if entry.target == self.target {
                return Some(&entry.attr);
            }
        }
        None
    }
}

#[derive(Clone, PartialEq, Debug)]
pub enum Opcode {
    Iconst(i64),
    Fconst(f64),
    Fconst128(u128),
    Arith(Arith),
    ArithImm(ArithImm),
    Icmp(Compare),
    Select(Select),
    StructNew(StructNew),
    Extract(Extract),
    Convert(Convert),
    DecodeLowFloat(LowFloatConvert),
    EncodeLowFloat(LowFloatConvert),
    DequantizeNvfp4(NvFp4Convert),
    QuantizeNvfp4(NvFp4Convert),
    Unary(Unary),
    Alloca(Alloca),
    Call(Call),
    CallIndirect(CallIndirect),
    GlobalAddr(GlobalAddr),
    Load(Load),
    Store(Store),
    Prefetch(Prefetch),
    VaStart(VaStart),
    VaArg(VaArg),
    VaEnd(VaEnd),
    Dot(Dot),
    Reduce(Reduce),
    Splat(Splat),
    Matmul(MatMul),
    Barrier(Barrier),
    AtomicRmw(AtomicRmw),
    If(If),
}

// f64 does not implement Eq; manual opcode equality is not needed. Provide a
// discriminant helper where passes need it.
impl Opcode {
    pub fn is_if(&self) -> bool {
        matches!(self, Opcode::If(_))
    }
}

#[derive(Clone, PartialEq, Debug)]
struct InstData {
    op: Opcode,
    result: Option<Value>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ValueDef {
    BlockParam { block: Block, index: u32 },
    InstResult(Inst),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct ValueData {
    ty: Type,
    def: ValueDef,
}

#[derive(Clone, PartialEq, Eq, Debug, Default)]
struct BlockData {
    params: Vec<Value>,
    insts: Vec<Inst>,
    term: Option<Terminator>,
}

#[derive(Clone, Debug, Default)]
pub struct Function {
    pub types: TypeTable,
    blocks: Vec<BlockData>,
    insts: Vec<InstData>,
    values: Vec<ValueData>,
    value_lists: Vec<Value>,
    scale_pool: Vec<u32>,
    bias_pool: Vec<i32>,
    attributes: Vec<AttrEntry>,
    symbols: Vec<String>,
    pub is_variadic: bool,
    pub num_fixed_params: u32,
    pub sret: bool,
    pub is_local: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ValuePair {
    pub old: Value,
    pub new: Value,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct InstPair {
    pub old: Inst,
    pub new: Inst,
}

#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct MatMulQuantSpec {
    pub scale_scalar: Option<u32>,
    pub scale_per_column: Option<Vec<u32>>,
    pub bias: Option<Vec<i32>>,
    pub zero_point: i32,
    pub relu: bool,
    pub out: Option<MatMulQuantOut>,
    pub input_signs: Option<InputSigns>,
}

impl Function {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reserve room for at least this many more blocks, instructions and values,
    /// so a builder that knows its rough size avoids repeated regrowth.
    pub fn reserve(&mut self, blocks: usize, insts: usize, values: usize) {
        self.blocks.reserve(blocks);
        self.insts.reserve(insts);
        self.values.reserve(values);
    }

    pub fn clone_func(&self) -> Self {
        self.clone()
    }

    pub fn intern_symbol(&mut self, name: &str) -> u32 {
        for (i, s) in self.symbols.iter().enumerate() {
            if s == name {
                return i as u32;
            }
        }
        self.symbols.push(String::from(name));
        self.symbols.len() as u32 - 1
    }

    pub fn symbol_name(&self, symbol: u32) -> &str {
        &self.symbols[symbol as usize]
    }

    pub fn symbol_count(&self) -> usize {
        self.symbols.len()
    }

    pub fn add_attr(&mut self, target: AttrTarget, attr: Attribute) {
        self.attributes.push(AttrEntry { target, attr });
    }

    pub fn attributes_of(&self, target: AttrTarget) -> AttrIterator<'_> {
        AttrIterator {
            entries: &self.attributes,
            target,
            index: 0,
        }
    }

    pub fn attribute_entries(&self) -> &[AttrEntry] {
        &self.attributes
    }

    pub fn retarget_attrs(&mut self, from: AttrTarget, to: AttrTarget) {
        for entry in &mut self.attributes {
            if entry.target == from {
                entry.target = to;
            }
        }
    }

    pub fn is_byte_order_tagged(&self, inst: Inst) -> bool {
        let mut on_inst = self.attributes_of(AttrTarget::Inst(inst));
        while let Some(attr) = on_inst.next_attr() {
            if matches!(attr, Attribute::Endian(_)) {
                return true;
            }
        }
        if let Some(result) = self.inst_result(inst) {
            let mut on_result = self.attributes_of(AttrTarget::Value(result));
            while let Some(attr) = on_result.next_attr() {
                if matches!(attr, Attribute::Endian(_)) {
                    return true;
                }
            }
        }
        false
    }

    pub fn clone_attrs(&mut self, values: &[ValuePair], insts: &[InstPair]) {
        let end = self.attributes.len();
        for i in 0..end {
            let entry = self.attributes[i].clone();
            match entry.target {
                AttrTarget::Value(v) => {
                    for pair in values {
                        if pair.old == v {
                            self.attributes.push(AttrEntry {
                                target: AttrTarget::Value(pair.new),
                                attr: entry.attr.clone(),
                            });
                        }
                    }
                }
                AttrTarget::Inst(n) => {
                    for pair in insts {
                        if pair.old == n {
                            self.attributes.push(AttrEntry {
                                target: AttrTarget::Inst(pair.new),
                                attr: entry.attr.clone(),
                            });
                        }
                    }
                }
                AttrTarget::Func | AttrTarget::Block(_) => {}
            }
        }
    }

    pub fn clone_attrs_from(&mut self, src: &Function, values: &[ValuePair], insts: &[InstPair]) {
        for entry in &src.attributes {
            match entry.target {
                AttrTarget::Value(v) => {
                    for pair in values {
                        if pair.old == v {
                            self.add_attr(AttrTarget::Value(pair.new), entry.attr.clone());
                        }
                    }
                }
                AttrTarget::Inst(n) => {
                    for pair in insts {
                        if pair.old == n {
                            self.add_attr(AttrTarget::Inst(pair.new), entry.attr.clone());
                        }
                    }
                }
                AttrTarget::Func | AttrTarget::Block(_) => {}
            }
        }
    }

    pub fn append_block(&mut self) -> Block {
        let index = self.blocks.len() as u32;
        self.blocks.push(BlockData::default());
        Block(index)
    }

    pub fn append_inst(&mut self, block: Block, ty: Type, op: Opcode) -> Value {
        let inst = Inst(self.insts.len() as u32);
        let value = Value(self.values.len() as u32);
        self.values.push(ValueData {
            ty,
            def: ValueDef::InstResult(inst),
        });
        self.insts.push(InstData {
            op,
            result: Some(value),
        });
        self.blocks[block.0 as usize].insts.push(inst);
        value
    }

    pub fn append_block_param(&mut self, block: Block, ty: Type) -> Value {
        let data = &mut self.blocks[block.0 as usize];
        let param_index = data.params.len() as u32;
        let value = Value(self.values.len() as u32);
        self.values.push(ValueData {
            ty,
            def: ValueDef::BlockParam {
                block,
                index: param_index,
            },
        });
        self.blocks[block.0 as usize].params.push(value);
        value
    }

    pub fn value_type(&self, value: Value) -> Type {
        self.values[value.0 as usize].ty
    }

    pub fn set_value_type(&mut self, value: Value, ty: Type) {
        self.values[value.0 as usize].ty = ty;
    }

    pub fn set_terminator(&mut self, block: Block, term: Terminator) {
        self.blocks[block.0 as usize].term = Some(term);
    }

    pub fn set_jump(&mut self, block: Block, target: Block, args: &[Value]) {
        let list = self.intern_values(args);
        self.set_terminator(block, Terminator::Jump(Jump { target, args: list }));
    }

    pub fn append_stmt_raw(&mut self, block: Block, op: Opcode) -> Inst {
        let inst = Inst(self.insts.len() as u32);
        self.insts.push(InstData { op, result: None });
        self.blocks[block.0 as usize].insts.push(inst);
        inst
    }

    fn append_stmt(&mut self, block: Block, op: Opcode) {
        let _ = self.append_stmt_raw(block, op);
    }

    pub fn append_if(
        &mut self,
        block: Block,
        cond: Value,
        then_edge: EdgeDesc,
        else_edge: EdgeDesc,
    ) {
        let then_jump = Jump {
            target: then_edge.target,
            args: self.intern_values(&then_edge.args),
        };
        let else_jump = Jump {
            target: else_edge.target,
            args: self.intern_values(&else_edge.args),
        };
        self.append_stmt(
            block,
            Opcode::If(If {
                cond,
                then: then_jump,
                else_: else_jump,
            }),
        );
    }

    pub fn append_store(&mut self, block: Block, value: Value, ptr: Value) {
        self.append_store_vol(block, value, ptr, false);
    }

    pub fn append_store_vol(&mut self, block: Block, value: Value, ptr: Value, volatile: bool) {
        self.append_stmt(
            block,
            Opcode::Store(Store {
                value,
                ptr,
                volatile,
            }),
        );
    }

    pub fn append_prefetch(&mut self, block: Block, ptr: Value) {
        self.append_stmt(block, Opcode::Prefetch(Prefetch { ptr }));
    }

    pub fn append_va_start(&mut self, block: Block, list: Value) {
        self.append_stmt(block, Opcode::VaStart(VaStart { list }));
    }

    pub fn append_va_arg(&mut self, block: Block, list: Value, ty: Type) -> Value {
        self.append_inst(block, ty, Opcode::VaArg(VaArg { list, ty }))
    }

    pub fn append_va_end(&mut self, block: Block, list: Value) {
        self.append_stmt(block, Opcode::VaEnd(VaEnd { list }));
    }

    pub fn append_barrier(&mut self, block: Block, scope: BarrierScope) {
        self.append_stmt(block, Opcode::Barrier(Barrier { scope }));
    }

    pub fn append_atomic_rmw(&mut self, block: Block, rmw: AtomicRmw) -> Value {
        let ty = self.value_type(rmw.value);
        self.append_inst(block, ty, Opcode::AtomicRmw(rmw))
    }

    pub fn append_atomic_rmw_stmt(&mut self, block: Block, rmw: AtomicRmw) {
        self.append_stmt(block, Opcode::AtomicRmw(rmw));
    }

    pub fn append_dot(&mut self, block: Block, acc: Value, a: Value, b: Value) -> Value {
        let ty = self.value_type(acc);
        self.append_inst(block, ty, Opcode::Dot(Dot { acc, a, b }))
    }

    pub fn append_reduce(&mut self, block: Block, op: BinOp, vector: Value) -> Value {
        let elem = match self.types.type_kind(self.value_type(vector)) {
            TypeKind::Vector(v) => v.elem,
            _ => panic!("append_reduce: vector must be a vector value"),
        };
        self.append_inst(block, elem, Opcode::Reduce(Reduce { vector, op }))
    }

    pub fn append_splat(&mut self, block: Block, vec_ty: Type, scalar: Value) -> Value {
        self.append_inst(block, vec_ty, Opcode::Splat(Splat { scalar }))
    }

    fn matmul_base(
        a: Value,
        b: Value,
        c: Value,
        m: u16,
        n: u16,
        k: u16,
        dtype: MatMulType,
        accumulate: bool,
    ) -> MatMul {
        MatMul {
            a,
            b,
            c,
            m,
            n,
            k,
            dtype,
            accumulate,
            embedded: false,
            quant: None,
            input_signs: None,
        }
    }

    pub fn append_matmul(
        &mut self,
        block: Block,
        a: Value,
        b: Value,
        c: Value,
        m: u16,
        n: u16,
        k: u16,
        dtype: MatMulType,
        accumulate: bool,
    ) {
        let mm = Self::matmul_base(a, b, c, m, n, k, dtype, accumulate);
        self.append_stmt(block, Opcode::Matmul(mm));
    }

    pub fn append_matmul_signed(
        &mut self,
        block: Block,
        a: Value,
        b: Value,
        c: Value,
        m: u16,
        n: u16,
        k: u16,
        dtype: MatMulType,
        accumulate: bool,
        input_signs: InputSigns,
    ) {
        let mut mm = Self::matmul_base(a, b, c, m, n, k, dtype, accumulate);
        mm.input_signs = Some(input_signs);
        self.append_stmt(block, Opcode::Matmul(mm));
    }

    pub fn append_matmul_embedded(
        &mut self,
        block: Block,
        a: Value,
        b: Value,
        c: Value,
        m: u16,
        n: u16,
        k: u16,
        dtype: MatMulType,
        accumulate: bool,
        input_signs: Option<InputSigns>,
    ) {
        let mut mm = Self::matmul_base(a, b, c, m, n, k, dtype, accumulate);
        mm.embedded = true;
        mm.input_signs = input_signs;
        self.append_stmt(block, Opcode::Matmul(mm));
    }

    pub fn append_matmul_quant(
        &mut self,
        block: Block,
        a: Value,
        b: Value,
        c: Value,
        m: u16,
        n: u16,
        k: u16,
        dtype: MatMulType,
        accumulate: bool,
        quant: MatMulQuant,
    ) {
        let mut mm = Self::matmul_base(a, b, c, m, n, k, dtype, accumulate);
        mm.quant = Some(quant);
        self.append_stmt(block, Opcode::Matmul(mm));
    }

    pub fn append_matmul_quant_per_column(
        &mut self,
        block: Block,
        a: Value,
        b: Value,
        c: Value,
        m: u16,
        n: u16,
        k: u16,
        dtype: MatMulType,
        accumulate: bool,
        relu: bool,
        out: MatMulQuantOut,
        scales: &[u32],
    ) {
        let h = self.intern_scales(scales);
        self.append_matmul_quant(
            block,
            a,
            b,
            c,
            m,
            n,
            k,
            dtype,
            accumulate,
            MatMulQuant {
                scale: MatMulScale::PerColumn(h),
                relu,
                out,
                bias: None,
                zero_point: 0,
            },
        );
    }

    pub fn append_matmul_quant_spec(
        &mut self,
        block: Block,
        a: Value,
        b: Value,
        c: Value,
        m: u16,
        n: u16,
        k: u16,
        dtype: MatMulType,
        accumulate: bool,
        spec: &MatMulQuantSpec,
    ) {
        assert!(spec.scale_scalar.is_some() != spec.scale_per_column.is_some());
        let scale = match spec.scale_scalar {
            Some(sb) => MatMulScale::Scalar(sb),
            None => MatMulScale::PerColumn(
                self.intern_scales(spec.scale_per_column.as_deref().unwrap_or(&[])),
            ),
        };
        let bias = spec.bias.as_deref().map(|bb| self.intern_bias(bb));
        self.append_matmul_quant(
            block,
            a,
            b,
            c,
            m,
            n,
            k,
            dtype,
            accumulate,
            MatMulQuant {
                scale,
                relu: spec.relu,
                out: spec.out.unwrap_or(MatMulQuantOut::I8),
                bias,
                zero_point: spec.zero_point,
            },
        );
        if let Some(signs) = spec.input_signs
            && let Opcode::Matmul(mm) =
                self.opcode_mut(self.blocks[block.0 as usize].insts.last().copied().unwrap())
        {
            mm.input_signs = Some(signs);
        }
    }

    pub fn append_arith_imm(
        &mut self,
        block: Block,
        ty: Type,
        op: BinOp,
        lhs: Value,
        imm: i64,
    ) -> Value {
        self.append_inst(block, ty, Opcode::ArithImm(ArithImm { op, lhs, imm }))
    }

    pub fn append_call(&mut self, block: Block, ty: Type, name: &str, args: &[Value]) -> Value {
        let symbol = self.intern_symbol(name);
        let list = self.intern_values(args);
        self.append_inst(block, ty, Opcode::Call(Call::direct(symbol, list)))
    }

    pub fn append_call_indirect(
        &mut self,
        block: Block,
        ty: Type,
        target: Value,
        args: &[Value],
    ) -> Value {
        let list = self.intern_values(args);
        self.append_inst(
            block,
            ty,
            Opcode::CallIndirect(CallIndirect::direct(target, list)),
        )
    }

    pub fn append_call_v(
        &mut self,
        block: Block,
        ty: Type,
        name: &str,
        args: &[Value],
        num_fixed: u32,
    ) -> Value {
        let symbol = self.intern_symbol(name);
        let list = self.intern_values(args);
        let mut c = Call::direct(symbol, list);
        c.is_variadic = true;
        c.num_fixed = num_fixed;
        self.append_inst(block, ty, Opcode::Call(c))
    }

    pub fn append_call_indirect_v(
        &mut self,
        block: Block,
        ty: Type,
        target: Value,
        args: &[Value],
        num_fixed: u32,
    ) -> Value {
        let list = self.intern_values(args);
        let mut c = CallIndirect::direct(target, list);
        c.is_variadic = true;
        c.num_fixed = num_fixed;
        self.append_inst(block, ty, Opcode::CallIndirect(c))
    }

    pub fn append_void_call_indirect(&mut self, block: Block, target: Value, args: &[Value]) {
        let list = self.intern_values(args);
        self.append_stmt(
            block,
            Opcode::CallIndirect(CallIndirect::direct(target, list)),
        );
    }

    pub fn append_void_call_indirect_v(
        &mut self,
        block: Block,
        target: Value,
        args: &[Value],
        num_fixed: u32,
    ) {
        let list = self.intern_values(args);
        let mut c = CallIndirect::direct(target, list);
        c.is_variadic = true;
        c.num_fixed = num_fixed;
        self.append_stmt(block, Opcode::CallIndirect(c));
    }

    pub fn append_global_addr(&mut self, block: Block, ty: Type, name: &str) -> Value {
        let symbol = self.intern_symbol(name);
        self.append_inst(
            block,
            ty,
            Opcode::GlobalAddr(GlobalAddr {
                symbol,
                via_got: false,
            }),
        )
    }

    pub fn append_global_addr_got(&mut self, block: Block, ty: Type, name: &str) -> Value {
        let symbol = self.intern_symbol(name);
        self.append_inst(
            block,
            ty,
            Opcode::GlobalAddr(GlobalAddr {
                symbol,
                via_got: true,
            }),
        )
    }

    pub fn append_void_call(&mut self, block: Block, name: &str, args: &[Value]) {
        let symbol = self.intern_symbol(name);
        let list = self.intern_values(args);
        self.append_stmt(block, Opcode::Call(Call::direct(symbol, list)));
    }

    pub fn append_void_call_v(&mut self, block: Block, name: &str, args: &[Value], num_fixed: u32) {
        let symbol = self.intern_symbol(name);
        let list = self.intern_values(args);
        let mut c = Call::direct(symbol, list);
        c.is_variadic = true;
        c.num_fixed = num_fixed;
        self.append_stmt(block, Opcode::Call(c));
    }

    fn ret_piece_array(pieces: &[RetPiece]) -> [RetPiece; 4] {
        assert!(pieces.len() <= 4);
        let mut out = [RetPiece::default(); 4];
        for (i, p) in pieces.iter().enumerate() {
            out[i] = *p;
        }
        out
    }

    pub fn append_call_struct_ret(
        &mut self,
        block: Block,
        name: &str,
        args: &[Value],
        ret_dest: Value,
        pieces: &[RetPiece],
    ) {
        let symbol = self.intern_symbol(name);
        let list = self.intern_values(args);
        let mut c = Call::direct(symbol, list);
        c.ret_dest = Some(ret_dest);
        c.ret_regs = pieces.len() as u8;
        c.ret_pieces = Self::ret_piece_array(pieces);
        self.append_stmt(block, Opcode::Call(c));
    }

    pub fn append_call_indirect_struct_ret(
        &mut self,
        block: Block,
        target: Value,
        args: &[Value],
        ret_dest: Value,
        pieces: &[RetPiece],
    ) {
        let list = self.intern_values(args);
        let mut c = CallIndirect::direct(target, list);
        c.ret_dest = Some(ret_dest);
        c.ret_regs = pieces.len() as u8;
        c.ret_pieces = Self::ret_piece_array(pieces);
        self.append_stmt(block, Opcode::CallIndirect(c));
    }

    pub fn append_call_sret(&mut self, block: Block, name: &str, args: &[Value]) {
        let symbol = self.intern_symbol(name);
        let list = self.intern_values(args);
        let mut c = Call::direct(symbol, list);
        c.sret = true;
        self.append_stmt(block, Opcode::Call(c));
    }

    pub fn append_call_indirect_sret(&mut self, block: Block, target: Value, args: &[Value]) {
        let list = self.intern_values(args);
        let mut c = CallIndirect::direct(target, list);
        c.sret = true;
        self.append_stmt(block, Opcode::CallIndirect(c));
    }

    pub fn append_struct_new(&mut self, block: Block, ty: Type, fields: &[Value]) -> Value {
        let list = self.intern_values(fields);
        self.append_inst(block, ty, Opcode::StructNew(StructNew { fields: list }))
    }

    pub fn block_insts(&self, block: Block) -> &[Inst] {
        &self.blocks[block.0 as usize].insts
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    pub fn block_params(&self, block: Block) -> &[Value] {
        &self.blocks[block.0 as usize].params
    }

    pub fn inst_result(&self, inst: Inst) -> Option<Value> {
        self.insts[inst.0 as usize].result
    }

    pub fn inst_count(&self) -> usize {
        self.insts.len()
    }

    pub fn opcode(&self, inst: Inst) -> Opcode {
        self.insts[inst.0 as usize].op.clone()
    }

    /// The opcode without copying it, for hot read-only walks.
    pub fn opcode_ref(&self, inst: Inst) -> &Opcode {
        &self.insts[inst.0 as usize].op
    }

    pub fn opcode_mut(&mut self, inst: Inst) -> &mut Opcode {
        &mut self.insts[inst.0 as usize].op
    }

    pub fn terminator_ptr(&mut self, block: Block) -> &mut Option<Terminator> {
        &mut self.blocks[block.0 as usize].term
    }

    pub fn value_list_mut(&mut self, list: ValueList) -> &mut [Value] {
        let start = list.start as usize;
        let len = list.len as usize;
        &mut self.value_lists[start..start + len]
    }

    pub fn replace_all_uses(&mut self, from: Value, to: Value) {
        fn repl(from: Value, to: Value, v: Value) -> Value {
            if v == from { to } else { v }
        }
        // Collect value-list indices first to satisfy the borrow checker.
        let mut lists: Vec<(u32, u32)> = Vec::new();
        for inst in &self.insts {
            match &inst.op {
                Opcode::StructNew(sn) => lists.push((sn.fields.start, sn.fields.len)),
                Opcode::Call(c) => lists.push((c.args.start, c.args.len)),
                Opcode::CallIndirect(c) => lists.push((c.args.start, c.args.len)),
                Opcode::If(cf) => {
                    lists.push((cf.then.args.start, cf.then.args.len));
                    lists.push((cf.else_.args.start, cf.else_.args.len));
                }
                _ => {}
            }
        }
        for inst in &mut self.insts {
            match &mut inst.op {
                Opcode::Iconst(_)
                | Opcode::Fconst(_)
                | Opcode::Fconst128(_)
                | Opcode::Alloca(_)
                | Opcode::GlobalAddr(_)
                | Opcode::Barrier(_) => {}
                Opcode::Arith(a) => {
                    a.lhs = repl(from, to, a.lhs);
                    a.rhs = repl(from, to, a.rhs);
                }
                Opcode::ArithImm(a) => a.lhs = repl(from, to, a.lhs),
                Opcode::Icmp(c) => {
                    c.lhs = repl(from, to, c.lhs);
                    c.rhs = repl(from, to, c.rhs);
                }
                Opcode::Select(s) => {
                    s.cond = repl(from, to, s.cond);
                    s.then = repl(from, to, s.then);
                    s.else_ = repl(from, to, s.else_);
                }
                Opcode::Extract(e) => e.aggregate = repl(from, to, e.aggregate),
                Opcode::Convert(cv) => cv.value = repl(from, to, cv.value),
                Opcode::DecodeLowFloat(cv) | Opcode::EncodeLowFloat(cv) => {
                    cv.value = repl(from, to, cv.value);
                }
                Opcode::DequantizeNvfp4(cv) | Opcode::QuantizeNvfp4(cv) => {
                    cv.value = repl(from, to, cv.value);
                    cv.block_scale = repl(from, to, cv.block_scale);
                    cv.global_scale = repl(from, to, cv.global_scale);
                }
                Opcode::Unary(u) => u.value = repl(from, to, u.value),
                Opcode::Load(l) => l.ptr = repl(from, to, l.ptr),
                Opcode::Store(st) => {
                    st.value = repl(from, to, st.value);
                    st.ptr = repl(from, to, st.ptr);
                }
                Opcode::Prefetch(pf) => pf.ptr = repl(from, to, pf.ptr),
                Opcode::VaStart(vs) => vs.list = repl(from, to, vs.list),
                Opcode::VaArg(va) => va.list = repl(from, to, va.list),
                Opcode::VaEnd(ve) => ve.list = repl(from, to, ve.list),
                Opcode::Dot(d) => {
                    d.acc = repl(from, to, d.acc);
                    d.a = repl(from, to, d.a);
                    d.b = repl(from, to, d.b);
                }
                Opcode::Reduce(red) => red.vector = repl(from, to, red.vector),
                Opcode::Splat(sp) => sp.scalar = repl(from, to, sp.scalar),
                Opcode::Matmul(mm) => {
                    mm.a = repl(from, to, mm.a);
                    mm.b = repl(from, to, mm.b);
                    mm.c = repl(from, to, mm.c);
                }
                Opcode::AtomicRmw(a) => {
                    a.ptr = repl(from, to, a.ptr);
                    a.value = repl(from, to, a.value);
                    if let Some(c) = &mut a.compare {
                        *c = repl(from, to, *c);
                    }
                }
                Opcode::StructNew(_)
                | Opcode::Call(_)
                | Opcode::CallIndirect(_)
                | Opcode::If(_) => {}
            }
        }
        // Rewrite value-list pool runs and call ret_dest/target fields.
        for inst in &mut self.insts {
            match &mut inst.op {
                Opcode::Call(c) => {
                    if let Some(rd) = &mut c.ret_dest {
                        *rd = repl(from, to, *rd);
                    }
                }
                Opcode::CallIndirect(c) => {
                    c.target = repl(from, to, c.target);
                    if let Some(rd) = &mut c.ret_dest {
                        *rd = repl(from, to, *rd);
                    }
                }
                Opcode::If(cf) => {
                    cf.cond = repl(from, to, cf.cond);
                }
                _ => {}
            }
        }
        for (start, len) in lists {
            for v in &mut self.value_lists[start as usize..(start + len) as usize] {
                *v = repl(from, to, *v);
            }
        }
        let (blocks, value_lists) = (&mut self.blocks, &mut self.value_lists);
        for block in blocks.iter_mut() {
            let term = block.term.clone();
            match term {
                Some(Terminator::Ret(r)) => {
                    let mut nr = r;
                    for vv in &mut nr.values[0..nr.count as usize] {
                        *vv = repl(from, to, *vv);
                    }
                    block.term = Some(Terminator::Ret(nr));
                }
                Some(Terminator::Jump(j)) => {
                    let (start, len) = (j.args.start as usize, j.args.len as usize);
                    for v in &mut value_lists[start..start + len] {
                        *v = repl(from, to, *v);
                    }
                }
                None => {}
            }
        }
    }

    pub fn block_insts_mut(&mut self, block: Block) -> &mut Vec<Inst> {
        &mut self.blocks[block.0 as usize].insts
    }

    pub fn new_param(&mut self, block: Block, ty: Type) -> Value {
        let value = Value(self.values.len() as u32);
        self.values.push(ValueData {
            ty,
            def: ValueDef::BlockParam { block, index: 0 },
        });
        value
    }

    pub fn set_block_params(&mut self, block: Block, params: &[Value]) {
        let data = &mut self.blocks[block.0 as usize];
        data.params.clear();
        data.params.extend_from_slice(params);
    }

    pub fn create_inst(&mut self, ty: Type, op: Opcode) -> Value {
        let inst = Inst(self.insts.len() as u32);
        let value = Value(self.values.len() as u32);
        self.values.push(ValueData {
            ty,
            def: ValueDef::InstResult(inst),
        });
        self.insts.push(InstData {
            op,
            result: Some(value),
        });
        value
    }

    pub fn set_block_insts(&mut self, block: Block, insts: &[Inst]) {
        let data = &mut self.blocks[block.0 as usize];
        data.insts.clear();
        data.insts.extend_from_slice(insts);
    }

    pub fn reorder_blocks(&mut self, order: &[Block]) {
        let n = self.block_count();
        assert_eq!(order.len(), n);
        assert_eq!(order[0], Block(0));
        let mut new_id = alloc::vec![0u32; n];
        for (new_index, old) in order.iter().enumerate() {
            new_id[old.0 as usize] = new_index as u32;
        }
        let mut tmp: Vec<BlockData> = Vec::with_capacity(n);
        for old in order {
            tmp.push(core::mem::take(&mut self.blocks[old.0 as usize]));
        }
        self.blocks = tmp;
        for bi in 0..n {
            let block = Block(bi as u32);
            if let Some(Terminator::Jump(j)) = self.terminator_ptr(block).as_mut() {
                j.target = Block(new_id[j.target.0 as usize]);
            }
            let insts = self.blocks[bi].insts.clone();
            for inst in insts {
                if let Opcode::If(cf) = self.opcode_mut(inst) {
                    cf.then.target = Block(new_id[cf.then.target.0 as usize]);
                    cf.else_.target = Block(new_id[cf.else_.target.0 as usize]);
                }
            }
        }
    }

    pub fn clone_block(&mut self, src: Block, map: &mut Vec<(Value, Value)>) -> Block {
        fn remap(map: &[(Value, Value)], v: Value) -> Value {
            map.iter()
                .find(|(old, _)| *old == v)
                .map(|(_, new)| *new)
                .unwrap_or(v)
        }
        let dst = self.append_block();
        let mut value_pairs: Vec<ValuePair> = Vec::new();
        let mut inst_pairs: Vec<InstPair> = Vec::new();
        for p in self.block_params(src).to_vec() {
            let ty = self.value_type(p);
            let np = self.append_block_param(dst, ty);
            map.push((p, np));
            value_pairs.push(ValuePair { old: p, new: np });
        }
        for inst in self.block_insts(src).to_vec() {
            let op = self.remap_opcode(self.opcode(inst).clone(), map);
            if let Some(result) = self.inst_result(inst) {
                let ty = self.value_type(result);
                let nr = self.append_inst(dst, ty, op);
                map.push((result, nr));
                value_pairs.push(ValuePair {
                    old: result,
                    new: nr,
                });
                inst_pairs.push(InstPair {
                    old: inst,
                    new: self.defining_inst(nr).unwrap(),
                });
            } else {
                let ni = self.append_stmt_raw(dst, op);
                inst_pairs.push(InstPair { old: inst, new: ni });
            }
        }
        self.clone_attrs(&value_pairs, &inst_pairs);
        if let Some(term) = self.terminator(src).clone() {
            let new_term = match term {
                Terminator::Ret(mut r) => {
                    for vv in &mut r.values[0..r.count as usize] {
                        *vv = remap(map, *vv);
                    }
                    Terminator::Ret(r)
                }
                Terminator::Jump(j) => Terminator::Jump(Jump {
                    target: j.target,
                    args: self.remap_value_list(j.args, map),
                }),
            };
            self.set_terminator(dst, new_term);
        }
        dst
    }

    fn remap_opcode(&mut self, op: Opcode, map: &[(Value, Value)]) -> Opcode {
        fn remap(map: &[(Value, Value)], v: Value) -> Value {
            map.iter()
                .find(|(old, _)| *old == v)
                .map(|(_, new)| *new)
                .unwrap_or(v)
        }
        match op {
            Opcode::Iconst(_)
            | Opcode::Fconst(_)
            | Opcode::Fconst128(_)
            | Opcode::Alloca(_)
            | Opcode::GlobalAddr(_)
            | Opcode::Barrier(_) => op,
            Opcode::Arith(a) => Opcode::Arith(Arith {
                op: a.op,
                lhs: remap(map, a.lhs),
                rhs: remap(map, a.rhs),
            }),
            Opcode::ArithImm(a) => Opcode::ArithImm(ArithImm {
                op: a.op,
                lhs: remap(map, a.lhs),
                imm: a.imm,
            }),
            Opcode::Icmp(c) => Opcode::Icmp(Compare {
                op: c.op,
                lhs: remap(map, c.lhs),
                rhs: remap(map, c.rhs),
            }),
            Opcode::Select(s) => Opcode::Select(Select {
                cond: remap(map, s.cond),
                then: remap(map, s.then),
                else_: remap(map, s.else_),
            }),
            Opcode::StructNew(sn) => Opcode::StructNew(StructNew {
                fields: self.remap_value_list(sn.fields, map),
            }),
            Opcode::Extract(ex) => Opcode::Extract(Extract {
                aggregate: remap(map, ex.aggregate),
                index: ex.index,
            }),
            Opcode::Convert(cv) => Opcode::Convert(Convert {
                value: remap(map, cv.value),
            }),
            Opcode::DecodeLowFloat(cv) => Opcode::DecodeLowFloat(LowFloatConvert {
                value: remap(map, cv.value),
                format: cv.format,
            }),
            Opcode::EncodeLowFloat(cv) => Opcode::EncodeLowFloat(LowFloatConvert {
                value: remap(map, cv.value),
                format: cv.format,
            }),
            Opcode::DequantizeNvfp4(cv) => Opcode::DequantizeNvfp4(NvFp4Convert {
                value: remap(map, cv.value),
                block_scale: remap(map, cv.block_scale),
                global_scale: remap(map, cv.global_scale),
                block_application: cv.block_application,
                global_application: cv.global_application,
            }),
            Opcode::QuantizeNvfp4(cv) => Opcode::QuantizeNvfp4(NvFp4Convert {
                value: remap(map, cv.value),
                block_scale: remap(map, cv.block_scale),
                global_scale: remap(map, cv.global_scale),
                block_application: cv.block_application,
                global_application: cv.global_application,
            }),
            Opcode::Unary(u) => Opcode::Unary(Unary {
                op: u.op,
                value: remap(map, u.value),
            }),
            Opcode::Call(mut c) => {
                c.args = self.remap_value_list(c.args, map);
                if let Some(rd) = c.ret_dest {
                    c.ret_dest = Some(remap(map, rd));
                }
                Opcode::Call(c)
            }
            Opcode::CallIndirect(mut c) => {
                c.target = remap(map, c.target);
                c.args = self.remap_value_list(c.args, map);
                if let Some(rd) = c.ret_dest {
                    c.ret_dest = Some(remap(map, rd));
                }
                Opcode::CallIndirect(c)
            }
            Opcode::Load(ld) => Opcode::Load(Load {
                ptr: remap(map, ld.ptr),
                volatile: ld.volatile,
            }),
            Opcode::Store(st) => Opcode::Store(Store {
                value: remap(map, st.value),
                ptr: remap(map, st.ptr),
                volatile: st.volatile,
            }),
            Opcode::Prefetch(pf) => Opcode::Prefetch(Prefetch {
                ptr: remap(map, pf.ptr),
            }),
            Opcode::VaStart(vs) => Opcode::VaStart(VaStart {
                list: remap(map, vs.list),
            }),
            Opcode::VaArg(va) => Opcode::VaArg(VaArg {
                list: remap(map, va.list),
                ty: va.ty,
            }),
            Opcode::VaEnd(ve) => Opcode::VaEnd(VaEnd {
                list: remap(map, ve.list),
            }),
            Opcode::Dot(d) => Opcode::Dot(Dot {
                acc: remap(map, d.acc),
                a: remap(map, d.a),
                b: remap(map, d.b),
            }),
            Opcode::Reduce(red) => Opcode::Reduce(Reduce {
                vector: remap(map, red.vector),
                op: red.op,
            }),
            Opcode::Splat(sp) => Opcode::Splat(Splat {
                scalar: remap(map, sp.scalar),
            }),
            Opcode::Matmul(mut mm) => {
                mm.a = remap(map, mm.a);
                mm.b = remap(map, mm.b);
                mm.c = remap(map, mm.c);
                Opcode::Matmul(mm)
            }
            Opcode::AtomicRmw(mut a) => {
                a.ptr = remap(map, a.ptr);
                a.value = remap(map, a.value);
                if let Some(c) = a.compare {
                    a.compare = Some(remap(map, c));
                }
                Opcode::AtomicRmw(a)
            }
            Opcode::If(cond) => Opcode::If(If {
                cond: remap(map, cond.cond),
                then: Jump {
                    target: cond.then.target,
                    args: self.remap_value_list(cond.then.args, map),
                },
                else_: Jump {
                    target: cond.else_.target,
                    args: self.remap_value_list(cond.else_.args, map),
                },
            }),
        }
    }

    fn remap_value_list(&mut self, list: ValueList, map: &[(Value, Value)]) -> ValueList {
        let src = self.value_list(list).to_vec();
        let mut out: Vec<Value> = Vec::with_capacity(src.len());
        for v in src {
            out.push(
                map.iter()
                    .find(|(old, _)| *old == v)
                    .map(|(_, new)| *new)
                    .unwrap_or(v),
            );
        }
        self.intern_values(&out)
    }

    pub fn intern_value_list(&mut self, vals: &[Value]) -> ValueList {
        self.intern_values(vals)
    }

    pub fn value_count(&self) -> usize {
        self.values.len()
    }

    pub fn block_args(&self, jump: Jump) -> &[Value] {
        self.value_list(jump.args)
    }

    pub fn intern_values(&mut self, vals: &[Value]) -> ValueList {
        let start = self.value_lists.len() as u32;
        self.value_lists.extend_from_slice(vals);
        ValueList {
            start,
            len: vals.len() as u32,
        }
    }

    pub fn value_list(&self, list: ValueList) -> &[Value] {
        &self.value_lists[list.start as usize..(list.start + list.len) as usize]
    }

    pub fn intern_scales(&mut self, scales: &[u32]) -> ScaleList {
        let start = self.scale_pool.len() as u32;
        self.scale_pool.extend_from_slice(scales);
        ScaleList {
            start,
            len: scales.len() as u32,
        }
    }

    pub fn scale_list(&self, list: ScaleList) -> &[u32] {
        &self.scale_pool[list.start as usize..(list.start + list.len) as usize]
    }

    pub fn intern_bias(&mut self, bias: &[i32]) -> BiasList {
        let start = self.bias_pool.len() as u32;
        self.bias_pool.extend_from_slice(bias);
        BiasList {
            start,
            len: bias.len() as u32,
        }
    }

    pub fn bias_list(&self, list: BiasList) -> &[i32] {
        &self.bias_pool[list.start as usize..(list.start + list.len) as usize]
    }

    pub fn terminator(&self, block: Block) -> Option<Terminator> {
        self.blocks[block.0 as usize].term.clone()
    }

    pub fn defining_inst(&self, value: Value) -> Option<Inst> {
        match self.values[value.0 as usize].def {
            ValueDef::InstResult(inst) => Some(inst),
            ValueDef::BlockParam { .. } => None,
        }
    }

    pub fn inst_op_tag_eq(&self, a: Inst, b: &Opcode) -> bool {
        core::mem::discriminant(&self.insts[a.0 as usize].op) == core::mem::discriminant(b)
    }
}

fn type_contains_f16(table: &TypeTable, ty: Type) -> bool {
    match table.type_kind(ty) {
        TypeKind::Bool | TypeKind::Int(_) | TypeKind::Ptr(_) => false,
        TypeKind::Float(f) => *f == super::types::FloatKind::F16,
        TypeKind::Vector(v) => type_contains_f16(table, v.elem),
        TypeKind::Array(a) => type_contains_f16(table, a.elem),
        TypeKind::Slice(s) => type_contains_f16(table, s.elem),
        TypeKind::Struct(fields) => fields.iter().any(|f| type_contains_f16(table, *f)),
    }
}

fn type_contains_f128(table: &TypeTable, ty: Type) -> bool {
    match table.type_kind(ty) {
        TypeKind::Bool | TypeKind::Int(_) | TypeKind::Ptr(_) => false,
        TypeKind::Float(f) => *f == super::types::FloatKind::F128,
        TypeKind::Vector(v) => type_contains_f128(table, v.elem),
        TypeKind::Array(a) => type_contains_f128(table, a.elem),
        TypeKind::Slice(s) => type_contains_f128(table, s.elem),
        TypeKind::Struct(fields) => fields.iter().any(|f| type_contains_f128(table, *f)),
    }
}

pub fn function_uses_f16(func: &Function) -> bool {
    (0..func.value_count())
        .map(|i| Value(i as u32))
        .any(|v| type_contains_f16(&func.types, func.value_type(v)))
}

pub fn function_uses_f128(func: &Function) -> bool {
    (0..func.value_count())
        .map(|i| Value(i as u32))
        .any(|v| type_contains_f128(&func.types, func.value_type(v)))
}

pub fn function_uses_composite_f16(func: &Function) -> bool {
    (0..func.value_count()).map(|i| Value(i as u32)).any(|v| {
        let ty = func.value_type(v);
        match func.types.type_kind(ty) {
            TypeKind::Vector(_) | TypeKind::Array(_) | TypeKind::Slice(_) | TypeKind::Struct(_) => {
                type_contains_f16(&func.types, ty)
            }
            _ => false,
        }
    })
}
