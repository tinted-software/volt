//! Every `volt` dialect operation.
//!
//! The op set mirrors the legacy [`Opcode`](crate::function::Opcode):
//! constants, arithmetic, control flow, memory, calls, vector/matrix and
//! atomics. Most ops use a declarative assembly format; [`CondBrOp`]
//! (two successors with partitioned operands) has a manual
//! printer/parser.

use alloc::string::String;
use alloc::vec::Vec;

use pliron::{
    builtin::{
        attributes::{BoolAttr, StringAttr, TypeAttr},
        op_interfaces::{
            IsTerminatorInterface, NOpdsInterface, NResultsInterface, OneResultInterface,
        },
    },
    combine::Parser,
    combine::parser::token::token,
    common_traits::Named,
    context::{Context, Ptr},
    derive::pliron_op,
    identifier::Identifier,
    input_err,
    irfmt::parsers::{
        block_opd_parser, delimited_list_parser, process_parsed_ssa_defs, spaced, ssa_opd_parser,
    },
    location::{Located, Location},
    op::{Op, OpObj},
    operation::Operation,
    parsable::{IntoParseResult, Parsable, ParseResult, StateStream},
    printable::{self, Printable},
    r#type::TypeHandle,
    value::Value,
};

use super::super::function::{AtomicOp, AtomicOrdering, AtomicScope, BarrierScope, BinOp, CmpOp};
use super::super::low_float::Format as LowFloatFormat;
use super::super::nvfp4::ScaleApplication;
use super::attrs::{
    AtomicOpAttr, AtomicOrderingAttr, AtomicScopeAttr, BarrierAttr, BinOpAttr, CallInfo,
    CallInfoAttr, CmpOpAttr, F128BitsAttr, FloatBitsAttr, ImmAttr, LowFloatAttr, MatMulAttr,
    MatMulDesc, ScaleAppAttr, UnaryOpAttr,
};
use pliron::basic_block::BasicBlock;

// Fetch a required attribute, panicking with the op name on absence.
// Attributes are always set by constructors; absence is a bug.
fn req_attr<T: Clone>(opt: Option<core::cell::Ref<'_, T>>, op: &str, attr: &str) -> T {
    opt.map(|r| T::clone(&r))
        .unwrap_or_else(|| panic!("{op} is missing required attribute {attr}"))
}

/// An integer constant: `%r = volt.iconst 42 : volt.int<true, 32>`.
#[pliron_op(
    name = "volt.iconst",
    format = "attr($iconst_value, $ImmAttr) ` : ` type($0)",
    interfaces = [NOpdsInterface<0>, OneResultInterface],
    attributes = (iconst_value: ImmAttr),
    verifier = "succ",
)]
pub struct ConstIntOp;

impl ConstIntOp {
    /// Build an integer constant.
    pub fn new(ctx: &mut Context, value: i64, ty: TypeHandle) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            alloc::vec![ty],
            Vec::new(),
            Vec::new(),
            0,
        );
        let op = Self { op };
        op.set_attr_iconst_value(ctx, ImmAttr::new(value));
        op
    }

    /// The constant value.
    pub fn value(&self, ctx: &Context) -> i64 {
        req_attr(
            self.get_attr_iconst_value(ctx),
            "volt.iconst",
            "iconst_value",
        )
        .value()
    }
}

/// A double-precision constant (stored as raw bits).
#[pliron_op(
    name = "volt.fconst",
    format = "attr($fconst_value, $FloatBitsAttr) ` : ` type($0)",
    interfaces = [NOpdsInterface<0>, OneResultInterface],
    attributes = (fconst_value: FloatBitsAttr),
    verifier = "succ",
)]
pub struct ConstFloatOp;

impl ConstFloatOp {
    /// Build a float constant.
    pub fn new(ctx: &mut Context, value: f64, ty: TypeHandle) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            alloc::vec![ty],
            Vec::new(),
            Vec::new(),
            0,
        );
        let op = Self { op };
        op.set_attr_fconst_value(ctx, FloatBitsAttr::new(value));
        op
    }

    /// The constant value.
    pub fn value(&self, ctx: &Context) -> f64 {
        req_attr(
            self.get_attr_fconst_value(ctx),
            "volt.fconst",
            "fconst_value",
        )
        .value()
    }
}

/// A quad-precision constant (stored as raw bits).
#[pliron_op(
    name = "volt.fconst128",
    format = "attr($fconst128_value, $F128BitsAttr) ` : ` type($0)",
    interfaces = [NOpdsInterface<0>, OneResultInterface],
    attributes = (fconst128_value: F128BitsAttr),
    verifier = "succ",
)]
pub struct ConstF128Op;

impl ConstF128Op {
    /// Build an f128 constant.
    pub fn new(ctx: &mut Context, value: u128, ty: TypeHandle) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            alloc::vec![ty],
            Vec::new(),
            Vec::new(),
            0,
        );
        let op = Self { op };
        op.set_attr_fconst128_value(ctx, F128BitsAttr::new(value));
        op
    }

    /// The constant value.
    pub fn value(&self, ctx: &Context) -> u128 {
        req_attr(
            self.get_attr_fconst128_value(ctx),
            "volt.fconst128",
            "fconst128_value",
        )
        .value()
    }
}

/// Binary arithmetic: `volt.arith add %lhs, %rhs : T`.
#[pliron_op(
    name = "volt.arith",
    format = "attr($arith_kind, $BinOpAttr) ` ` $0 `, ` $1 ` : ` type($0)",
    interfaces = [OneResultInterface],
    attributes = (arith_kind: BinOpAttr),
    verifier = "succ",
)]
pub struct ArithOp;

impl ArithOp {
    /// Build a binary arithmetic op.
    pub fn new(ctx: &mut Context, kind: BinOp, lhs: Value, rhs: Value, ty: TypeHandle) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            alloc::vec![ty],
            alloc::vec![lhs, rhs],
            Vec::new(),
            0,
        );
        let op = Self { op };
        op.set_attr_arith_kind(ctx, BinOpAttr::new(kind));
        op
    }

    /// The arithmetic operator.
    pub fn kind(&self, ctx: &Context) -> BinOp {
        req_attr(self.get_attr_arith_kind(ctx), "volt.arith", "arith_kind").value()
    }

    /// Left operand.
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Right operand.
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}

/// Arithmetic with an immediate: `volt.arith_imm add %lhs, 5 : T`.
#[pliron_op(
    name = "volt.arith_imm",
    format = "attr($arith_imm_kind, $BinOpAttr) ` ` $0 `, ` attr($arith_imm_imm, $ImmAttr) ` : ` type($0)",
    interfaces = [OneResultInterface],
    attributes = (arith_imm_kind: BinOpAttr, arith_imm_imm: ImmAttr),
    verifier = "succ",
)]
pub struct ArithImmOp;

impl ArithImmOp {
    /// Build an arithmetic-immediate op.
    pub fn new(ctx: &mut Context, kind: BinOp, lhs: Value, imm: i64, ty: TypeHandle) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            alloc::vec![ty],
            alloc::vec![lhs],
            Vec::new(),
            0,
        );
        let op = Self { op };
        op.set_attr_arith_imm_kind(ctx, BinOpAttr::new(kind));
        op.set_attr_arith_imm_imm(ctx, ImmAttr::new(imm));
        op
    }

    /// The arithmetic operator.
    pub fn kind(&self, ctx: &Context) -> BinOp {
        req_attr(
            self.get_attr_arith_imm_kind(ctx),
            "volt.arith_imm",
            "arith_imm_kind",
        )
        .value()
    }

    /// The value operand.
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// The immediate.
    pub fn imm(&self, ctx: &Context) -> i64 {
        req_attr(
            self.get_attr_arith_imm_imm(ctx),
            "volt.arith_imm",
            "arith_imm_imm",
        )
        .value()
    }
}

/// Integer comparison: `volt.icmp lt %lhs, %rhs : volt.bool`.
#[pliron_op(
    name = "volt.icmp",
    format = "attr($icmp_kind, $CmpOpAttr) ` ` $0 `, ` $1 ` : ` type($0)",
    interfaces = [OneResultInterface],
    attributes = (icmp_kind: CmpOpAttr),
    verifier = "succ",
)]
pub struct IcmpOp;

impl IcmpOp {
    /// Build a comparison.
    pub fn new(ctx: &mut Context, kind: CmpOp, lhs: Value, rhs: Value, ty: TypeHandle) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            alloc::vec![ty],
            alloc::vec![lhs, rhs],
            Vec::new(),
            0,
        );
        let op = Self { op };
        op.set_attr_icmp_kind(ctx, CmpOpAttr::new(kind));
        op
    }

    /// The comparison operator.
    pub fn kind(&self, ctx: &Context) -> CmpOp {
        req_attr(self.get_attr_icmp_kind(ctx), "volt.icmp", "icmp_kind").value()
    }

    /// Left operand.
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Right operand.
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}

/// Conditional select: `volt.select %c, %then, %else : T`.
#[pliron_op(
    name = "volt.select",
    format = "$0 `, ` $1 `, ` $2 ` : ` type($0)",
    interfaces = [OneResultInterface],
    verifier = "succ",
)]
pub struct SelectOp;

impl SelectOp {
    /// Build a select.
    pub fn new(ctx: &mut Context, cond: Value, then: Value, else_: Value, ty: TypeHandle) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            alloc::vec![ty],
            alloc::vec![cond, then, else_],
            Vec::new(),
            0,
        );
        Self { op }
    }

    /// Condition, then-value, else-value.
    pub fn operands3(&self, ctx: &Context) -> (Value, Value, Value) {
        let op = self.get_operation().deref(ctx);
        (op.get_operand(0), op.get_operand(1), op.get_operand(2))
    }
}

/// Struct construction from field values.
#[pliron_op(
    name = "volt.struct_new",
    format = "operands(CharSpace(`,`)) ` : ` type($0)",
    interfaces = [OneResultInterface],
    verifier = "succ",
)]
pub struct StructNewOp;

impl StructNewOp {
    /// Build a struct value.
    pub fn new(ctx: &mut Context, fields: Vec<Value>, ty: TypeHandle) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            alloc::vec![ty],
            fields,
            Vec::new(),
            0,
        );
        Self { op }
    }

    /// Field values.
    pub fn fields(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().deref(ctx).operands().collect()
    }
}

/// Struct field extraction: `volt.extract %agg 1 : T`.
#[pliron_op(
    name = "volt.extract",
    format = "$0 attr($extract_index, $ImmAttr) ` : ` type($0)",
    interfaces = [OneResultInterface],
    attributes = (extract_index: ImmAttr),
    verifier = "succ",
)]
pub struct ExtractOp;

impl ExtractOp {
    /// Build a field extraction.
    pub fn new(ctx: &mut Context, aggregate: Value, index: u32, ty: TypeHandle) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            alloc::vec![ty],
            alloc::vec![aggregate],
            Vec::new(),
            0,
        );
        let op = Self { op };
        op.set_attr_extract_index(ctx, ImmAttr::new(index as i64));
        op
    }

    /// The aggregate value.
    pub fn aggregate(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Field index.
    pub fn index(&self, ctx: &Context) -> u32 {
        req_attr(
            self.get_attr_extract_index(ctx),
            "volt.extract",
            "extract_index",
        )
        .value() as u32
    }
}

/// Type conversion (int/float resize, int<->float, bitcast flavors).
#[pliron_op(
    name = "volt.convert",
    format = "$0 ` : ` type($0)",
    interfaces = [OneResultInterface],
    verifier = "succ",
)]
pub struct ConvertOp;

impl ConvertOp {
    /// Build a conversion.
    pub fn new(ctx: &mut Context, value: Value, ty: TypeHandle) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            alloc::vec![ty],
            alloc::vec![value],
            Vec::new(),
            0,
        );
        Self { op }
    }

    /// The converted value.
    pub fn value(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}

/// Low-precision float decode: `volt.decode_low_float bf16 %v : T`.
#[pliron_op(
    name = "volt.decode_low_float",
    format = "attr($decode_format, $LowFloatAttr) ` ` $0 ` : ` type($0)",
    interfaces = [OneResultInterface],
    attributes = (decode_format: LowFloatAttr),
    verifier = "succ",
)]
pub struct DecodeLowFloatOp;

impl DecodeLowFloatOp {
    /// Build a decode.
    pub fn new(ctx: &mut Context, format: LowFloatFormat, value: Value, ty: TypeHandle) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            alloc::vec![ty],
            alloc::vec![value],
            Vec::new(),
            0,
        );
        let op = Self { op };
        op.set_attr_decode_format(ctx, LowFloatAttr::new(format));
        op
    }

    /// The source format.
    pub fn float_format(&self, ctx: &Context) -> LowFloatFormat {
        req_attr(
            self.get_attr_decode_format(ctx),
            "volt.decode_low_float",
            "decode_format",
        )
        .value()
    }

    /// The encoded value.
    pub fn value(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}

/// Low-precision float encode.
#[pliron_op(
    name = "volt.encode_low_float",
    format = "attr($encode_format, $LowFloatAttr) ` ` $0 ` : ` type($0)",
    interfaces = [OneResultInterface],
    attributes = (encode_format: LowFloatAttr),
    verifier = "succ",
)]
pub struct EncodeLowFloatOp;

impl EncodeLowFloatOp {
    /// Build an encode.
    pub fn new(ctx: &mut Context, format: LowFloatFormat, value: Value, ty: TypeHandle) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            alloc::vec![ty],
            alloc::vec![value],
            Vec::new(),
            0,
        );
        let op = Self { op };
        op.set_attr_encode_format(ctx, LowFloatAttr::new(format));
        op
    }

    /// The target format.
    pub fn float_format(&self, ctx: &Context) -> LowFloatFormat {
        req_attr(
            self.get_attr_encode_format(ctx),
            "volt.encode_low_float",
            "encode_format",
        )
        .value()
    }

    /// The decoded value.
    pub fn value(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}

/// NVFP4 dequantize: value + block/global scales.
#[pliron_op(
    name = "volt.dequantize_nvfp4",
    format = "attr($dequant_format, $LowFloatAttr) ` ` attr($dequant_block_app, $ScaleAppAttr) ` ` attr($dequant_global_app, $ScaleAppAttr) ` ` $0 `, ` $1 `, ` $2 ` : ` type($0)",
    interfaces = [OneResultInterface],
    attributes = (dequant_format: LowFloatAttr, dequant_block_app: ScaleAppAttr, dequant_global_app: ScaleAppAttr),
    verifier = "succ",
)]
pub struct DequantizeNvfp4Op;

impl DequantizeNvfp4Op {
    /// Build a dequantize.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: &mut Context,
        format: LowFloatFormat,
        value: Value,
        block_scale: Value,
        global_scale: Value,
        block_app: ScaleApplication,
        global_app: ScaleApplication,
        ty: TypeHandle,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            alloc::vec![ty],
            alloc::vec![value, block_scale, global_scale],
            Vec::new(),
            0,
        );
        let op = Self { op };
        op.set_attr_dequant_format(ctx, LowFloatAttr::new(format));
        op.set_attr_dequant_block_app(ctx, ScaleAppAttr::new(block_app));
        op.set_attr_dequant_global_app(ctx, ScaleAppAttr::new(global_app));
        op
    }

    /// Conversion parameters and operands (value, block scale, global scale).
    pub fn parts(
        &self,
        ctx: &Context,
    ) -> (
        LowFloatFormat,
        ScaleApplication,
        ScaleApplication,
        Value,
        Value,
        Value,
    ) {
        let op = self.get_operation();
        let format = req_attr(
            self.get_attr_dequant_format(ctx),
            "volt.dequantize_nvfp4",
            "dequant_format",
        )
        .value();
        let block_app = req_attr(
            self.get_attr_dequant_block_app(ctx),
            "volt.dequantize_nvfp4",
            "dequant_block_app",
        )
        .value();
        let global_app = req_attr(
            self.get_attr_dequant_global_app(ctx),
            "volt.dequantize_nvfp4",
            "dequant_global_app",
        )
        .value();
        let op = op.deref(ctx);
        (
            format,
            block_app,
            global_app,
            op.get_operand(0),
            op.get_operand(1),
            op.get_operand(2),
        )
    }
}

/// NVFP4 quantize: value + block/global scales.
#[pliron_op(
    name = "volt.quantize_nvfp4",
    format = "attr($quant_format, $LowFloatAttr) ` ` attr($quant_block_app, $ScaleAppAttr) ` ` attr($quant_global_app, $ScaleAppAttr) ` ` $0 `, ` $1 `, ` $2 ` : ` type($0)",
    interfaces = [OneResultInterface],
    attributes = (quant_format: LowFloatAttr, quant_block_app: ScaleAppAttr, quant_global_app: ScaleAppAttr),
    verifier = "succ",
)]
pub struct QuantizeNvfp4Op;

impl QuantizeNvfp4Op {
    /// Build a quantize.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: &mut Context,
        format: LowFloatFormat,
        value: Value,
        block_scale: Value,
        global_scale: Value,
        block_app: ScaleApplication,
        global_app: ScaleApplication,
        ty: TypeHandle,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            alloc::vec![ty],
            alloc::vec![value, block_scale, global_scale],
            Vec::new(),
            0,
        );
        let op = Self { op };
        op.set_attr_quant_format(ctx, LowFloatAttr::new(format));
        op.set_attr_quant_block_app(ctx, ScaleAppAttr::new(block_app));
        op.set_attr_quant_global_app(ctx, ScaleAppAttr::new(global_app));
        op
    }

    /// Conversion parameters and operands (value, block scale, global scale).
    pub fn parts(
        &self,
        ctx: &Context,
    ) -> (
        LowFloatFormat,
        ScaleApplication,
        ScaleApplication,
        Value,
        Value,
        Value,
    ) {
        let op = self.get_operation();
        let format = req_attr(
            self.get_attr_quant_format(ctx),
            "volt.quantize_nvfp4",
            "quant_format",
        )
        .value();
        let block_app = req_attr(
            self.get_attr_quant_block_app(ctx),
            "volt.quantize_nvfp4",
            "quant_block_app",
        )
        .value();
        let global_app = req_attr(
            self.get_attr_quant_global_app(ctx),
            "volt.quantize_nvfp4",
            "quant_global_app",
        )
        .value();
        let op = op.deref(ctx);
        (
            format,
            block_app,
            global_app,
            op.get_operand(0),
            op.get_operand(1),
            op.get_operand(2),
        )
    }
}

/// Unary float/int op: `volt.unary sqrt %v : T`.
#[pliron_op(
    name = "volt.unary",
    format = "attr($unary_kind, $UnaryOpAttr) ` ` $0 ` : ` type($0)",
    interfaces = [OneResultInterface],
    attributes = (unary_kind: UnaryOpAttr),
    verifier = "succ",
)]
pub struct UnaryOp;

impl UnaryOp {
    /// Build a unary op.
    pub fn new(
        ctx: &mut Context,
        kind: super::super::function::UnaryOp,
        value: Value,
        ty: TypeHandle,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            alloc::vec![ty],
            alloc::vec![value],
            Vec::new(),
            0,
        );
        let op = Self { op };
        op.set_attr_unary_kind(ctx, UnaryOpAttr::new(kind));
        op
    }

    /// The unary operator.
    pub fn kind(&self, ctx: &Context) -> super::super::function::UnaryOp {
        req_attr(self.get_attr_unary_kind(ctx), "volt.unary", "unary_kind").value()
    }

    /// The operand.
    pub fn value(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}

/// Stack allocation of one element.
#[pliron_op(
    name = "volt.alloca",
    format = "attr($alloca_elem, $TypeAttr) ` : ` type($0)",
    interfaces = [NOpdsInterface<0>, OneResultInterface],
    attributes = (alloca_elem: TypeAttr),
    verifier = "succ",
)]
pub struct AllocaOp;

impl AllocaOp {
    /// Build an alloca.
    pub fn new(ctx: &mut Context, elem: TypeHandle, ty: TypeHandle) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            alloc::vec![ty],
            Vec::new(),
            Vec::new(),
            0,
        );
        let op = Self { op };
        op.set_attr_alloca_elem(ctx, TypeAttr::new(elem));
        op
    }

    /// Element type.
    pub fn elem_type(&self, ctx: &Context) -> TypeHandle {
        TypeHandle::from(req_attr(
            self.get_attr_alloca_elem(ctx),
            "volt.alloca",
            "alloca_elem",
        ))
    }
}

/// Direct call: `volt.call @"sym"(%args) <info> : T`.
/// Operand 0.. are the call arguments; there are 0 or 1 results.
#[pliron_op(
    name = "volt.call",
    format = "`@` attr($call_callee, $StringAttr) `(` operands(CharSpace(`,`)) `)` ` ` attr($call_info, $CallInfoAttr) ` : ` types(CharSpace(`,`))",
    attributes = (call_callee: StringAttr, call_info: CallInfoAttr),
    verifier = "succ",
)]
pub struct CallOp;

impl CallOp {
    /// Build a direct call.
    pub fn new(
        ctx: &mut Context,
        callee: &str,
        args: Vec<Value>,
        results: Vec<TypeHandle>,
        info: CallInfo,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            results,
            args,
            Vec::new(),
            0,
        );
        let op = Self { op };
        op.set_attr_call_callee(ctx, StringAttr::new(String::from(callee)));
        op.set_attr_call_info(ctx, CallInfoAttr::new(info));
        op
    }

    /// Callee symbol name.
    pub fn callee(&self, ctx: &Context) -> String {
        req_attr(self.get_attr_call_callee(ctx), "volt.call", "call_callee")
            .as_str()
            .into()
    }

    /// Call ABI details.
    pub fn info(&self, ctx: &Context) -> CallInfo {
        req_attr(self.get_attr_call_info(ctx), "volt.call", "call_info")
            .info()
            .clone()
    }

    /// Call arguments.
    pub fn args(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().deref(ctx).operands().collect()
    }
}

/// Indirect call through a function pointer. Operand 0 is the target.
#[pliron_op(
    name = "volt.call_indirect",
    format = "operands(CharSpace(`,`)) ` ` attr($call_indirect_info, $CallInfoAttr) ` : ` types(CharSpace(`,`))",
    attributes = (call_indirect_info: CallInfoAttr),
    verifier = "succ",
)]
pub struct CallIndirectOp;

impl CallIndirectOp {
    /// Build an indirect call.
    pub fn new(
        ctx: &mut Context,
        target: Value,
        args: Vec<Value>,
        results: Vec<TypeHandle>,
        info: CallInfo,
    ) -> Self {
        let mut operands = alloc::vec![target];
        operands.extend(args);
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            results,
            operands,
            Vec::new(),
            0,
        );
        let op = Self { op };
        op.set_attr_call_indirect_info(ctx, CallInfoAttr::new(info));
        op
    }

    /// Call target.
    pub fn target(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Call ABI details.
    pub fn info(&self, ctx: &Context) -> CallInfo {
        req_attr(
            self.get_attr_call_indirect_info(ctx),
            "volt.call_indirect",
            "call_indirect_info",
        )
        .info()
        .clone()
    }

    /// Call arguments (excluding the target).
    pub fn args(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().deref(ctx).operands().skip(1).collect()
    }
}

/// Address of a global symbol.
#[pliron_op(
    name = "volt.global_addr",
    format = "`@` attr($global_symbol, $StringAttr) ` ` attr($global_via_got, $BoolAttr) ` : ` type($0)",
    interfaces = [NOpdsInterface<0>, OneResultInterface],
    attributes = (global_symbol: StringAttr, global_via_got: BoolAttr),
    verifier = "succ",
)]
pub struct GlobalAddrOp;

impl GlobalAddrOp {
    /// Build a global address.
    pub fn new(ctx: &mut Context, symbol: &str, via_got: bool, ty: TypeHandle) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            alloc::vec![ty],
            Vec::new(),
            Vec::new(),
            0,
        );
        let op = Self { op };
        op.set_attr_global_symbol(ctx, StringAttr::new(String::from(symbol)));
        op.set_attr_global_via_got(ctx, BoolAttr::new(via_got));
        op
    }

    /// Symbol name.
    pub fn symbol(&self, ctx: &Context) -> String {
        req_attr(
            self.get_attr_global_symbol(ctx),
            "volt.global_addr",
            "global_symbol",
        )
        .as_str()
        .into()
    }

    /// Route through the GOT.
    pub fn via_got(&self, ctx: &Context) -> bool {
        req_attr(
            self.get_attr_global_via_got(ctx),
            "volt.global_addr",
            "global_via_got",
        )
        .into()
    }
}

/// Memory load: `volt.load false %ptr : T`.
#[pliron_op(
    name = "volt.load",
    format = "attr($load_volatile, $BoolAttr) ` ` $0 ` : ` type($0)",
    interfaces = [OneResultInterface],
    attributes = (load_volatile: BoolAttr),
    verifier = "succ",
)]
pub struct LoadOp;

impl LoadOp {
    /// Build a load.
    pub fn new(ctx: &mut Context, ptr: Value, volatile: bool, ty: TypeHandle) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            alloc::vec![ty],
            alloc::vec![ptr],
            Vec::new(),
            0,
        );
        let op = Self { op };
        op.set_attr_load_volatile(ctx, BoolAttr::new(volatile));
        op
    }

    /// Pointer operand.
    pub fn ptr(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Volatile access.
    pub fn volatile(&self, ctx: &Context) -> bool {
        req_attr(
            self.get_attr_load_volatile(ctx),
            "volt.load",
            "load_volatile",
        )
        .into()
    }
}

/// Memory store: `volt.store false %value, %ptr`.
#[pliron_op(
    name = "volt.store",
    format = "attr($store_volatile, $BoolAttr) ` ` $0 `, ` $1",
    interfaces = [NOpdsInterface<2>, NResultsInterface<0>],
    attributes = (store_volatile: BoolAttr),
    verifier = "succ",
)]
pub struct StoreOp;

impl StoreOp {
    /// Build a store.
    pub fn new(ctx: &mut Context, value: Value, ptr: Value, volatile: bool) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            Vec::new(),
            alloc::vec![value, ptr],
            Vec::new(),
            0,
        );
        let op = Self { op };
        op.set_attr_store_volatile(ctx, BoolAttr::new(volatile));
        op
    }

    /// Stored value and pointer.
    pub fn parts(&self, ctx: &Context) -> (Value, Value) {
        let op = self.get_operation().deref(ctx);
        (op.get_operand(0), op.get_operand(1))
    }

    /// Volatile access.
    pub fn volatile(&self, ctx: &Context) -> bool {
        req_attr(
            self.get_attr_store_volatile(ctx),
            "volt.store",
            "store_volatile",
        )
        .into()
    }
}

/// Cache prefetch hint.
#[pliron_op(
    name = "volt.prefetch",
    format = "$0",
    interfaces = [NOpdsInterface<1>, NResultsInterface<0>],
    verifier = "succ",
)]
pub struct PrefetchOp;

impl PrefetchOp {
    /// Build a prefetch.
    pub fn new(ctx: &mut Context, ptr: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            Vec::new(),
            alloc::vec![ptr],
            Vec::new(),
            0,
        );
        Self { op }
    }

    /// Pointer operand.
    pub fn ptr(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}

/// Start of variadic argument access.
#[pliron_op(
    name = "volt.va_start",
    format = "$0",
    interfaces = [NOpdsInterface<1>, NResultsInterface<0>],
    verifier = "succ",
)]
pub struct VaStartOp;

impl VaStartOp {
    /// Build a va_start.
    pub fn new(ctx: &mut Context, list: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            Vec::new(),
            alloc::vec![list],
            Vec::new(),
            0,
        );
        Self { op }
    }

    /// Argument list pointer.
    pub fn list(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}

/// Fetch the next variadic argument.
#[pliron_op(
    name = "volt.va_arg",
    format = "attr($va_arg_ty, $TypeAttr) ` ` $0 ` : ` type($0)",
    interfaces = [OneResultInterface],
    attributes = (va_arg_ty: TypeAttr),
    verifier = "succ",
)]
pub struct VaArgOp;

impl VaArgOp {
    /// Build a va_arg.
    pub fn new(ctx: &mut Context, list: Value, ty: TypeHandle) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            alloc::vec![ty],
            alloc::vec![list],
            Vec::new(),
            0,
        );
        let op = Self { op };
        op.set_attr_va_arg_ty(ctx, TypeAttr::new(ty));
        op
    }

    /// Argument list pointer.
    pub fn list(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}

/// End of variadic argument access.
#[pliron_op(
    name = "volt.va_end",
    format = "$0",
    interfaces = [NOpdsInterface<1>, NResultsInterface<0>],
    verifier = "succ",
)]
pub struct VaEndOp;

impl VaEndOp {
    /// Build a va_end.
    pub fn new(ctx: &mut Context, list: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            Vec::new(),
            alloc::vec![list],
            Vec::new(),
            0,
        );
        Self { op }
    }

    /// Argument list pointer.
    pub fn list(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}

/// Fused dot product: `acc + dot(a, b)`.
#[pliron_op(
    name = "volt.dot",
    format = "$0 `, ` $1 `, ` $2 ` : ` type($0)",
    interfaces = [OneResultInterface],
    verifier = "succ",
)]
pub struct DotOp;

impl DotOp {
    /// Build a dot product.
    pub fn new(ctx: &mut Context, acc: Value, a: Value, b: Value, ty: TypeHandle) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            alloc::vec![ty],
            alloc::vec![acc, a, b],
            Vec::new(),
            0,
        );
        Self { op }
    }

    /// Accumulator and linear-algebra operands.
    pub fn operands3(&self, ctx: &Context) -> (Value, Value, Value) {
        let op = self.get_operation().deref(ctx);
        (op.get_operand(0), op.get_operand(1), op.get_operand(2))
    }
}

/// Vector reduction with a binary operator.
#[pliron_op(
    name = "volt.reduce",
    format = "attr($reduce_kind, $BinOpAttr) ` ` $0 ` : ` type($0)",
    interfaces = [OneResultInterface],
    attributes = (reduce_kind: BinOpAttr),
    verifier = "succ",
)]
pub struct ReduceOp;

impl ReduceOp {
    /// Build a reduction.
    pub fn new(ctx: &mut Context, kind: BinOp, vector: Value, ty: TypeHandle) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            alloc::vec![ty],
            alloc::vec![vector],
            Vec::new(),
            0,
        );
        let op = Self { op };
        op.set_attr_reduce_kind(ctx, BinOpAttr::new(kind));
        op
    }

    /// The reduction operator.
    pub fn kind(&self, ctx: &Context) -> BinOp {
        req_attr(self.get_attr_reduce_kind(ctx), "volt.reduce", "reduce_kind").value()
    }

    /// The reduced vector.
    pub fn vector(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}

/// Broadcast a scalar to a vector.
#[pliron_op(
    name = "volt.splat",
    format = "$0 ` : ` type($0)",
    interfaces = [OneResultInterface],
    verifier = "succ",
)]
pub struct SplatOp;

impl SplatOp {
    /// Build a splat.
    pub fn new(ctx: &mut Context, scalar: Value, ty: TypeHandle) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            alloc::vec![ty],
            alloc::vec![scalar],
            Vec::new(),
            0,
        );
        Self { op }
    }

    /// The broadcast scalar.
    pub fn scalar(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}

/// Matrix multiply-accumulate over three buffers.
#[pliron_op(
    name = "volt.matmul",
    format = "attr($matmul_config, $MatMulAttr) ` ` $0 `, ` $1 `, ` $2 ` : ` type($0)",
    interfaces = [OneResultInterface],
    attributes = (matmul_config: MatMulAttr),
    verifier = "succ",
)]
pub struct MatMulOp;

impl MatMulOp {
    /// Build a matmul.
    pub fn new(
        ctx: &mut Context,
        a: Value,
        b: Value,
        c: Value,
        config: MatMulDesc,
        ty: TypeHandle,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            alloc::vec![ty],
            alloc::vec![a, b, c],
            Vec::new(),
            0,
        );
        let op = Self { op };
        op.set_attr_matmul_config(ctx, MatMulAttr::new(config));
        op
    }

    /// Configuration.
    pub fn config(&self, ctx: &Context) -> MatMulDesc {
        req_attr(
            self.get_attr_matmul_config(ctx),
            "volt.matmul",
            "matmul_config",
        )
        .desc()
        .clone()
    }

    /// Buffer operands (a, b, c).
    pub fn operands3(&self, ctx: &Context) -> (Value, Value, Value) {
        let op = self.get_operation().deref(ctx);
        (op.get_operand(0), op.get_operand(1), op.get_operand(2))
    }
}

/// Execution barrier.
#[pliron_op(
    name = "volt.barrier",
    format = "attr($barrier_scope, $BarrierAttr)",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
    attributes = (barrier_scope: BarrierAttr),
    verifier = "succ",
)]
pub struct BarrierOp;

impl BarrierOp {
    /// Build a barrier.
    pub fn new(ctx: &mut Context, scope: BarrierScope) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            0,
        );
        let op = Self { op };
        op.set_attr_barrier_scope(ctx, BarrierAttr::new(scope));
        op
    }

    /// Barrier scope.
    pub fn scope(&self, ctx: &Context) -> BarrierScope {
        req_attr(
            self.get_attr_barrier_scope(ctx),
            "volt.barrier",
            "barrier_scope",
        )
        .value()
    }
}

/// Atomic read-modify-write. Operands are `ptr, value`, plus `compare`
/// for compare-exchange.
#[pliron_op(
    name = "volt.atomic_rmw",
    format = "attr($atomic_kind, $AtomicOpAttr) ` ` attr($atomic_ordering, $AtomicOrderingAttr) ` ` attr($atomic_scope, $AtomicScopeAttr) ` ` operands(CharSpace(`,`)) ` : ` type($0)",
    interfaces = [OneResultInterface],
    attributes = (atomic_kind: AtomicOpAttr, atomic_ordering: AtomicOrderingAttr, atomic_scope: AtomicScopeAttr),
    verifier = "succ",
)]
pub struct AtomicRmwOp;

impl AtomicRmwOp {
    /// Build an atomic RMW.
    pub fn new(
        ctx: &mut Context,
        kind: AtomicOp,
        ptr: Value,
        value: Value,
        compare: Option<Value>,
        ordering: AtomicOrdering,
        scope: AtomicScope,
        ty: TypeHandle,
    ) -> Self {
        let mut operands = alloc::vec![ptr, value];
        if let Some(c) = compare {
            operands.push(c);
        }
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            alloc::vec![ty],
            operands,
            Vec::new(),
            0,
        );
        let op = Self { op };
        op.set_attr_atomic_kind(ctx, AtomicOpAttr::new(kind));
        op.set_attr_atomic_ordering(ctx, AtomicOrderingAttr::new(ordering));
        op.set_attr_atomic_scope(ctx, AtomicScopeAttr::new(scope));
        op
    }

    /// The atomic operator.
    pub fn kind(&self, ctx: &Context) -> AtomicOp {
        req_attr(
            self.get_attr_atomic_kind(ctx),
            "volt.atomic_rmw",
            "atomic_kind",
        )
        .value()
    }

    /// Ordering.
    pub fn ordering(&self, ctx: &Context) -> AtomicOrdering {
        req_attr(
            self.get_attr_atomic_ordering(ctx),
            "volt.atomic_rmw",
            "atomic_ordering",
        )
        .value()
    }

    /// Scope.
    pub fn scope(&self, ctx: &Context) -> AtomicScope {
        req_attr(
            self.get_attr_atomic_scope(ctx),
            "volt.atomic_rmw",
            "atomic_scope",
        )
        .value()
    }

    /// Pointer, value, and optional compare operands.
    pub fn operands_rmw(&self, ctx: &Context) -> (Value, Value, Option<Value>) {
        let op = self.get_operation().deref(ctx);
        let n = op.get_num_operands();
        (
            op.get_operand(0),
            op.get_operand(1),
            if n > 2 { Some(op.get_operand(2)) } else { None },
        )
    }
}

/// Unconditional branch: `volt.jump ^target(%args)`.
#[pliron_op(
    name = "volt.jump",
    format = "succ($0) `(` operands(CharSpace(`,`)) `)`",
    interfaces = [IsTerminatorInterface, NResultsInterface<0>],
    verifier = "succ",
)]
pub struct JumpOp;

impl JumpOp {
    /// Build a jump.
    pub fn new(ctx: &mut Context, target: Ptr<BasicBlock>, args: Vec<Value>) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            Vec::new(),
            args,
            alloc::vec![target],
            0,
        );
        Self { op }
    }

    /// Jump target.
    pub fn target(&self, ctx: &Context) -> Ptr<BasicBlock> {
        self.get_operation().deref(ctx).get_successor(0)
    }

    /// Block arguments.
    pub fn args(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().deref(ctx).operands().collect()
    }
}

/// Conditional branch with per-successor operands.
///
/// Stored as one successor list `[then, else]`, one operand list
/// `[cond, ...then_args, ...else_args]`, and a `then_args` count
/// attribute splitting the two argument groups.
///
/// Syntax: `volt.cond_br %cond, ^then(%a), ^else(%b)`.
#[pliron_op(
    name = "volt.cond_br",
    interfaces = [IsTerminatorInterface, NResultsInterface<0>],
    attributes = (cond_then_args: ImmAttr),
    verifier = "succ",
)]
pub struct CondBrOp;

impl CondBrOp {
    /// Build a conditional branch.
    pub fn new(
        ctx: &mut Context,
        cond: Value,
        then: Ptr<BasicBlock>,
        then_args: Vec<Value>,
        else_: Ptr<BasicBlock>,
        else_args: Vec<Value>,
    ) -> Self {
        let n_then = then_args.len() as i64;
        let mut operands = alloc::vec![cond];
        operands.extend(then_args);
        operands.extend(else_args);
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            Vec::new(),
            operands,
            alloc::vec![then, else_],
            0,
        );
        let op = Self { op };
        op.set_attr_cond_then_args(ctx, ImmAttr::new(n_then));
        op
    }

    /// Branch condition.
    pub fn cond(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Number of then-successor arguments.
    fn n_then(&self, ctx: &Context) -> usize {
        req_attr(
            self.get_attr_cond_then_args(ctx),
            "volt.cond_br",
            "cond_then_args",
        )
        .value() as usize
    }

    /// Then destination.
    pub fn then_dest(&self, ctx: &Context) -> Ptr<BasicBlock> {
        self.get_operation().deref(ctx).get_successor(0)
    }

    /// Else destination.
    pub fn else_dest(&self, ctx: &Context) -> Ptr<BasicBlock> {
        self.get_operation().deref(ctx).get_successor(1)
    }

    /// Then arguments.
    pub fn then_args(&self, ctx: &Context) -> Vec<Value> {
        let n = self.n_then(ctx);
        self.get_operation()
            .deref(ctx)
            .operands()
            .skip(1)
            .take(n)
            .collect()
    }

    /// Else arguments.
    pub fn else_args(&self, ctx: &Context) -> Vec<Value> {
        let n = self.n_then(ctx);
        self.get_operation()
            .deref(ctx)
            .operands()
            .skip(1 + n)
            .collect()
    }
}

impl Printable for CondBrOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(
            f,
            "{} {}, ^{}({}), ^{}({})",
            self.get_opid().print(ctx, state),
            self.cond(ctx).print(ctx, state),
            self.then_dest(ctx).unique_name(ctx),
            comma_values(ctx, state, &self.then_args(ctx)),
            self.else_dest(ctx).unique_name(ctx),
            comma_values(ctx, state, &self.else_args(ctx)),
        )
    }
}

fn comma_values(ctx: &Context, state: &printable::State, values: &[Value]) -> String {
    let mut out = String::new();
    for (i, v) in values.iter().enumerate() {
        if i != 0 {
            out.push_str(", ");
        }
        out.push_str(&alloc::format!("{}", v.print(ctx, state)));
    }
    out
}

impl Parsable for CondBrOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let loc = state_stream.loc();
        if !results.is_empty() {
            input_err!(loc, "volt.cond_br cannot define SSA values")?;
        }
        let cond = ssa_opd_parser().parse_stream(state_stream).into_result()?.0;
        spaced(token(','))
            .parse_stream(state_stream)
            .into_result()?;
        let then = block_opd_parser()
            .parse_stream(state_stream)
            .into_result()?
            .0;
        let then_args = delimited_list_parser('(', ')', ',', ssa_opd_parser())
            .parse_stream(state_stream)
            .into_result()?
            .0;
        spaced(token(','))
            .parse_stream(state_stream)
            .into_result()?;
        let else_ = block_opd_parser()
            .parse_stream(state_stream)
            .into_result()?
            .0;
        let else_args = delimited_list_parser('(', ')', ',', ssa_opd_parser())
            .parse_stream(state_stream)
            .into_result()?
            .0;
        let op = Self::new(
            state_stream.state.ctx,
            cond,
            then,
            then_args,
            else_,
            else_args,
        );
        process_parsed_ssa_defs(state_stream, &results, op.get_operation())?;
        Ok(OpObj::new(op)).into_parse_result()
    }
}

/// Function return with up to N values.
#[pliron_op(
    name = "volt.ret",
    format = "operands(CharSpace(`,`))",
    interfaces = [IsTerminatorInterface, NResultsInterface<0>],
    verifier = "succ",
)]
pub struct RetOp;

impl RetOp {
    /// Build a return.
    pub fn new(ctx: &mut Context, values: Vec<Value>) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            Vec::new(),
            values,
            Vec::new(),
            0,
        );
        Self { op }
    }

    /// Returned values.
    pub fn values(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().deref(ctx).operands().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect::types::{BoolType, IntType};
    use pliron::{
        builtin::{op_interfaces::OneRegionInterface, ops::FuncOp},
        parsable::parse_from_str,
        result::ExpectOk,
    };

    #[test]
    fn arith_print_parse() {
        let ctx = &mut Context::new();
        let i32t = IntType::get_type(ctx, true, 32);
        let boolt = BoolType::get_type(ctx);
        let func_ty = pliron::builtin::types::FunctionType::get(ctx, vec![i32t, i32t], vec![boolt]);
        let func = FuncOp::new(ctx, pliron::ident!("f"), func_ty);
        let entry = func.get_entry_block(ctx);
        let a = entry.deref(ctx).get_argument(0);
        let b = entry.deref(ctx).get_argument(1);
        let add = ArithOp::new(ctx, BinOp::Add, a, b, i32t);
        add.get_operation().insert_at_back(entry, ctx);
        let text = alloc::format!(
            "{}",
            add.get_operation()
                .deref(ctx)
                .print(ctx, &Default::default())
        );
        assert!(text.contains("volt.arith"), "{text}");
        assert!(text.contains("add"), "{text}");
        let _ = boolt;
    }

    #[test]
    fn cond_br_round_trip() {
        let ctx = &mut Context::new();
        let boolt = BoolType::get_type(ctx);
        let func_ty = pliron::builtin::types::FunctionType::get(ctx, vec![boolt], vec![]);
        let func = FuncOp::new(ctx, pliron::ident!("f"), func_ty);
        let entry = func.get_entry_block(ctx);
        let region = func.get_region(ctx);
        let then_bb = BasicBlock::new(ctx, Some(pliron::ident!("then")), vec![]);
        then_bb.insert_at_back(region, ctx);
        let else_bb = BasicBlock::new(ctx, Some(pliron::ident!("else")), vec![]);
        else_bb.insert_at_back(region, ctx);
        let cond = entry.deref(ctx).get_argument(0);
        let br = CondBrOp::new(ctx, cond, then_bb, vec![], else_bb, vec![]);
        br.get_operation().insert_at_back(entry, ctx);
        let func_text = alloc::format!(
            "{}",
            func.get_operation()
                .deref(ctx)
                .print(ctx, &Default::default())
        );
        let parsed = parse_from_str(Operation::top_level_parser(), ctx, &func_text).expect_ok(ctx);
        let _ = parsed;
    }
}
