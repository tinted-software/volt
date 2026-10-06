//! Operation payload attributes for the `volt` dialect.
//!
//! Small enums reuse the existing volt definitions (from
//! [`function`](crate::function), [`low_float`](crate::low_float),
//! [`nvfp4`](crate::nvfp4), and [`attribute`](crate::attribute)); the
//! structs below wrap them as pliron attributes with keyword syntax.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use pliron::{
    combine::{
        Parser, choice,
        parser::{
            char::{spaces, string},
            token::{satisfy, token},
        },
    },
    context::Context,
    derive::pliron_attr,
    irfmt::parsers::int_parser,
    parsable::{Parsable, ParseResult, StateStream},
    printable::{self, Printable},
};

use super::super::attribute::{AttrValue, Attribute, Custom, Endianness};
use super::super::function::{
    AtomicOp, AtomicOrdering, AtomicScope, BarrierScope, BinOp, CmpOp, InputSigns, MatMulQuantOut,
    MatMulType, RetPiece, UnaryOp,
};
use super::super::low_float::Format as LowFloatFormat;
use super::super::nvfp4::ScaleApplication;

/// Parse one of several keywords.
/// Parse one of several keywords.
fn keyword<'a, T: Copy>(
    state_stream: &mut StateStream<'a>,
    options: &[(&'static str, T)],
) -> ParseResult<'a, T> {
    let mut parsers = Vec::new();
    for (word, value) in options {
        parsers.push(string(*word).map(move |_| *value));
    }
    spaces()
        .with(choice(&mut parsers[..]))
        .parse_stream(state_stream)
        .into()
}

macro_rules! keyword_attr {
    ($attr:ident, $inner:ty, $name:literal, [$($word:literal => $variant:path),* $(,)?]) => {
        #[doc = concat!("A `", $name, "` attribute.")]
        #[pliron_attr(name = $name, verifier = "succ")]
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
        pub struct $attr(pub $inner);

        impl $attr {
            /// Wrap a value.
            pub fn new(value: $inner) -> Self {
                Self(value)
            }

            /// The wrapped value.
            pub fn value(&self) -> $inner {
                self.0
            }
        }

        impl Printable for $attr {
            fn fmt(
                &self,
                _ctx: &Context,
                _state: &printable::State,
                f: &mut core::fmt::Formatter<'_>,
            ) -> core::fmt::Result {
                let word = match self.0 {
                    $($variant => $word,)*
                };
                f.write_str(word)
            }
        }

        impl Parsable for $attr {
            type Arg = ();
            type Parsed = Self;
            fn parse<'a>(
                state_stream: &mut StateStream<'a>,
                _arg: Self::Arg,
            ) -> ParseResult<'a, Self::Parsed> {
                keyword(state_stream, &[$(($word, $variant),)*])
                    .map(|(val, rest)| (Self(val), rest))
            }
        }
    };
}

keyword_attr!(
    BinOpAttr,
    BinOp,
    "volt.bin_op",
    [
        "add" => BinOp::Add, "sub" => BinOp::Sub, "mul" => BinOp::Mul,
        "div" => BinOp::Div, "rem" => BinOp::Rem, "and" => BinOp::BitAnd,
        "or" => BinOp::BitOr, "xor" => BinOp::BitXor, "shl" => BinOp::Shl,
        "shr" => BinOp::Shr, "mulh" => BinOp::Mulh,
    ]
);

keyword_attr!(
    CmpOpAttr,
    CmpOp,
    "volt.cmp_op",
    [
        "eq" => CmpOp::Eq, "ne" => CmpOp::Ne, "lt" => CmpOp::Lt,
        "le" => CmpOp::Le, "gt" => CmpOp::Gt, "ge" => CmpOp::Ge,
    ]
);

keyword_attr!(
    UnaryOpAttr,
    UnaryOp,
    "volt.unary_op",
    [
        "reinterpret" => UnaryOp::Reinterpret, "sqrt" => UnaryOp::Sqrt,
        "ceil" => UnaryOp::Ceil, "floor" => UnaryOp::Floor,
        "trunc" => UnaryOp::Trunc, "nearest" => UnaryOp::Nearest,
    ]
);

keyword_attr!(
    BarrierAttr,
    BarrierScope,
    "volt.barrier_scope",
    [
        "workgroup" => BarrierScope::Workgroup,
        "subgroup" => BarrierScope::Subgroup,
    ]
);

keyword_attr!(
    AtomicOpAttr,
    AtomicOp,
    "volt.atomic_op",
    [
        "add" => AtomicOp::Add, "min" => AtomicOp::Min,
        "max" => AtomicOp::Max, "and" => AtomicOp::BitAnd,
        "or" => AtomicOp::BitOr, "xor" => AtomicOp::BitXor,
        "exchange" => AtomicOp::Exchange,
        "compare_exchange" => AtomicOp::CompareExchange,
    ]
);

keyword_attr!(
    AtomicOrderingAttr,
    AtomicOrdering,
    "volt.atomic_ordering",
    [
        "relaxed" => AtomicOrdering::Relaxed,
        "acquire" => AtomicOrdering::Acquire,
        "release" => AtomicOrdering::Release,
        "acq_rel" => AtomicOrdering::AcqRel,
        "seq_cst" => AtomicOrdering::SeqCst,
    ]
);

keyword_attr!(
    AtomicScopeAttr,
    AtomicScope,
    "volt.atomic_scope",
    [
        "workgroup" => AtomicScope::Workgroup,
        "device" => AtomicScope::Device,
        "system" => AtomicScope::System,
    ]
);

keyword_attr!(
    LowFloatAttr,
    LowFloatFormat,
    "volt.low_float_format",
    [
        "bf16" => LowFloatFormat::Bf16,
        "f8e4m3" => LowFloatFormat::F8E4M3,
        "f8e5m2" => LowFloatFormat::F8E5M2,
    ]
);

keyword_attr!(
    ScaleAppAttr,
    ScaleApplication,
    "volt.scale_application",
    [
        "multiply" => ScaleApplication::Multiply,
        "divide" => ScaleApplication::Divide,
    ]
);

keyword_attr!(
    MatMulTypeAttr,
    MatMulType,
    "volt.matmul_dtype",
    [
        "fp32" => MatMulType::Fp32, "fp16" => MatMulType::Fp16,
        "int8" => MatMulType::Int8, "uint8" => MatMulType::Uint8,
    ]
);

keyword_attr!(
    MatMulQuantOutAttr,
    MatMulQuantOut,
    "volt.matmul_quant_out",
    [
        "i8" => MatMulQuantOut::I8,
        "u8" => MatMulQuantOut::U8,
    ]
);

/// A generic signed integer attribute: immediates, indices, counts.
#[pliron_attr(name = "volt.imm", verifier = "succ")]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ImmAttr(pub i64);

impl ImmAttr {
    /// Wrap a value.
    pub fn new(value: i64) -> Self {
        Self(value)
    }

    /// The wrapped value.
    pub fn value(&self) -> i64 {
        self.0
    }
}

impl Printable for ImmAttr {
    fn fmt(
        &self,
        _ctx: &Context,
        _state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Parsable for ImmAttr {
    type Arg = ();
    type Parsed = Self;
    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        spaces()
            .with(int_parser::<i64>())
            .map(Self)
            .parse_stream(state_stream)
            .into()
    }
}

/// Raw bits of an `f64` constant, printed as decimal.
#[pliron_attr(name = "volt.fbits", verifier = "succ")]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct FloatBitsAttr(pub u64);

impl FloatBitsAttr {
    /// Wrap an `f64` constant.
    pub fn new(value: f64) -> Self {
        Self(value.to_bits())
    }

    /// The constant value.
    pub fn value(&self) -> f64 {
        f64::from_bits(self.0)
    }
}

impl Printable for FloatBitsAttr {
    fn fmt(
        &self,
        _ctx: &Context,
        _state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Parsable for FloatBitsAttr {
    type Arg = ();
    type Parsed = Self;
    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        spaces()
            .with(int_parser::<u64>())
            .map(Self)
            .parse_stream(state_stream)
            .into()
    }
}

/// Raw bits of an `f128` constant, printed as decimal.
#[pliron_attr(name = "volt.f128bits", verifier = "succ")]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct F128BitsAttr(pub u128);

impl F128BitsAttr {
    /// Wrap a value.
    pub fn new(value: u128) -> Self {
        Self(value)
    }

    /// The wrapped value.
    pub fn value(&self) -> u128 {
        self.0
    }
}

impl Printable for F128BitsAttr {
    fn fmt(
        &self,
        _ctx: &Context,
        _state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Parsable for F128BitsAttr {
    type Arg = ();
    type Parsed = Self;
    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        spaces()
            .with(int_parser::<u128>())
            .map(Self)
            .parse_stream(state_stream)
            .into()
    }
}

/// Owned matrix-multiply configuration (mirrors [`MatMul`](crate::function::MatMul)).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct MatMulDesc {
    /// Rows of A / C.
    pub m: u16,
    /// Columns of B / C.
    pub n: u16,
    /// Reduction dimension.
    pub k: u16,
    /// Element flavor.
    pub dtype: MatMulType,
    /// Accumulate into C instead of overwriting.
    pub accumulate: bool,
    /// Embedded (in-register) operand layout.
    pub embedded: bool,
    /// Optional quantization parameters.
    pub quant: Option<QuantDesc>,
    /// Optional operand signedness overrides.
    pub input_signs: Option<InputSigns>,
}

/// Owned quantization parameters (mirrors [`MatMulQuant`](crate::function::MatMulQuant)).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct QuantDesc {
    /// Single shared scale, if any.
    pub scale_scalar: Option<u32>,
    /// Per-column scales, if any.
    pub scale_per_column: Option<Vec<u32>>,
    /// Apply ReLU to the output.
    pub relu: bool,
    /// Output flavor.
    pub out: MatMulQuantOut,
    /// Per-output bias, if any.
    pub bias: Option<Vec<i32>>,
    /// Zero point.
    pub zero_point: i32,
}

/// A `volt.matmul_config` attribute.
#[pliron_attr(name = "volt.matmul_config", verifier = "succ")]
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct MatMulAttr(pub MatMulDesc);

impl MatMulAttr {
    /// Wrap a descriptor.
    pub fn new(desc: MatMulDesc) -> Self {
        Self(desc)
    }

    /// The descriptor.
    pub fn desc(&self) -> &MatMulDesc {
        &self.0
    }
}

impl Printable for MatMulAttr {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        let d = &self.0;
        write!(
            f,
            "m={} n={} k={} dtype={} ",
            d.m,
            d.n,
            d.k,
            MatMulTypeAttr(d.dtype).print(ctx, state)
        )?;
        if d.accumulate {
            write!(f, "acc ")?;
        }
        if d.embedded {
            write!(f, "embedded ")?;
        }
        if let Some(q) = &d.quant {
            write!(f, "quant(")?;
            if let Some(s) = q.scale_scalar {
                write!(f, "scale={s} ")?;
            }
            if let Some(scales) = &q.scale_per_column {
                write!(f, "scales=[")?;
                for (i, s) in scales.iter().enumerate() {
                    if i != 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{s}")?;
                }
                write!(f, "] ")?;
            }
            if q.relu {
                write!(f, "relu ")?;
            }
            write!(f, "out={} ", MatMulQuantOutAttr(q.out).print(ctx, state))?;
            if let Some(bias) = &q.bias {
                write!(f, "bias=[")?;
                for (i, b) in bias.iter().enumerate() {
                    if i != 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{b}")?;
                }
                write!(f, "] ")?;
            }
            write!(f, "zp={} ) ", q.zero_point)?;
        }
        if let Some(signs) = &d.input_signs {
            write!(
                f,
                "signs(a_unsigned={} b_unsigned={}) ",
                signs.a_unsigned, signs.b_unsigned
            )?;
        }
        Ok(())
    }
}

impl Parsable for MatMulAttr {
    type Arg = ();
    type Parsed = Self;
    fn parse<'a>(
        _state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        // TODO: picks up in the parser port.
        unimplemented!("matmul config parsing")
    }
}

/// Call ABI details: variadic shape, struct-return, multi-register returns.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct CallInfo {
    /// Variadic call.
    pub is_variadic: bool,
    /// Number of fixed arguments (variadic calls).
    pub num_fixed: u32,
    /// Struct-return (first argument is the return slot).
    pub sret: bool,
    /// Number of registers used by the return value.
    pub ret_regs: u8,
    /// How the return value is split across registers.
    pub ret_pieces: Vec<RetPiece>,
}

impl Default for CallInfo {
    fn default() -> Self {
        Self {
            is_variadic: false,
            num_fixed: 0,
            sret: false,
            ret_regs: 0,
            ret_pieces: Vec::new(),
        }
    }
}

/// A `volt.call_info` attribute.
#[pliron_attr(name = "volt.call_info", verifier = "succ")]
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct CallInfoAttr(pub CallInfo);

impl CallInfoAttr {
    /// Wrap call info.
    pub fn new(info: CallInfo) -> Self {
        Self(info)
    }

    /// The call info.
    pub fn info(&self) -> &CallInfo {
        &self.0
    }
}

impl Printable for CallInfoAttr {
    fn fmt(
        &self,
        _ctx: &Context,
        _state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        let c = &self.0;
        write!(
            f,
            "variadic={} fixed={} sret={} regs={} pieces=[",
            c.is_variadic, c.num_fixed, c.sret, c.ret_regs
        )?;
        for (i, p) in c.ret_pieces.iter().enumerate() {
            if i != 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}:{}:{}", p.fp as u8, p.offset, p.bytes)?;
        }
        write!(f, "]")
    }
}

impl Parsable for CallInfoAttr {
    type Arg = ();
    type Parsed = Self;
    fn parse<'a>(
        _state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        // TODO: picks up in the parser port.
        unimplemented!("call info parsing")
    }
}

/// Function-level flags: variadic shape, struct-return, linkage.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct FuncInfo {
    /// Variadic function.
    pub is_variadic: bool,
    /// Number of fixed parameters (variadic functions).
    pub num_fixed_params: u32,
    /// Struct-return (first parameter is the return slot).
    pub sret: bool,
    /// Local (non-exported) linkage.
    pub is_local: bool,
}

/// A `volt.func_info` attribute.
#[pliron_attr(name = "volt.func_info", verifier = "succ")]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct FuncInfoAttr(pub FuncInfo);

impl FuncInfoAttr {
    /// Wrap function info.
    pub fn new(info: FuncInfo) -> Self {
        Self(info)
    }

    /// The function info.
    pub fn info(&self) -> FuncInfo {
        self.0
    }
}

impl Printable for FuncInfoAttr {
    fn fmt(
        &self,
        _ctx: &Context,
        _state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        let i = &self.0;
        write!(
            f,
            "variadic={} fixed={} sret={} local={}",
            i.is_variadic, i.num_fixed_params, i.sret, i.is_local
        )
    }
}

impl Parsable for FuncInfoAttr {
    type Arg = ();
    type Parsed = Self;
    fn parse<'a>(
        _state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        // TODO: picks up in the parser port.
        unimplemented!("func info parsing")
    }
}

/// A `volt.generic_attr` wrapping any legacy [`Attribute`], so existing
/// attribute flows (inline, align, endian, custom, ...) keep working.
/// Prints with the legacy syntax (`inline`, `align(16)`, `ns.key = 1`).
#[pliron_attr(name = "volt.generic_attr", verifier = "succ")]
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct VoltAttr(pub Attribute);

impl VoltAttr {
    /// Wrap a legacy attribute.
    pub fn new(attr: Attribute) -> Self {
        Self(attr)
    }

    /// The wrapped attribute.
    pub fn attr(&self) -> &Attribute {
        &self.0
    }
}

impl Printable for VoltAttr {
    fn fmt(
        &self,
        _ctx: &Context,
        _state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        match &self.0 {
            Attribute::Inline => f.write_str("inline"),
            Attribute::Noreturn => f.write_str("noreturn"),
            Attribute::Cold => f.write_str("cold"),
            Attribute::Align(a) => write!(f, "align({a})"),
            Attribute::Endian(e) => {
                let s = match e {
                    Endianness::Little => "little",
                    Endianness::Big => "big",
                    Endianness::Native => "native",
                };
                write!(f, "endian({s})")
            }
            Attribute::Custom(c) => {
                write!(f, "{}.{}", c.namespace, c.key)?;
                match &c.value {
                    AttrValue::Flag => Ok(()),
                    AttrValue::Int(i) => write!(f, " = {i}"),
                    AttrValue::Str(s) => write!(f, " = \"{s}\""),
                }
            }
        }
    }
}

fn quoted_string<'a>(state_stream: &mut StateStream<'a>) -> ParseResult<'a, String> {
    token('"')
        .with(pliron::combine::many(satisfy(|c: char| c != '"')))
        .skip(token('"'))
        .map(|chars: Vec<char>| chars.into_iter().collect::<String>())
        .parse_stream(state_stream)
        .into()
}

impl Parsable for VoltAttr {
    type Arg = ();
    type Parsed = Self;
    fn parse<'a>(
        _state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        // TODO: picks up in the parser port.
        unimplemented!("generic attribute parsing")
    }
}

/// Parse a legacy dotted-path custom attribute name (`a.b.key`).
#[allow(dead_code)]
pub fn parse_custom_path<'a>(state_stream: &mut StateStream<'a>) -> ParseResult<'a, String> {
    (
        pliron::combine::many1(satisfy(|c: char| c.is_ascii_alphanumeric() || c == '_'))
            .map(|w: Vec<char>| w.into_iter().collect::<String>()),
        pliron::combine::many(token('.').with(pliron::combine::many1(satisfy(|c: char| {
            c.is_ascii_alphanumeric() || c == '_'
        })))),
    )
        .map(|(first, rest): (String, Vec<Vec<char>>)| {
            let mut path = first;
            for seg in rest {
                path.push('.');
                path.extend(seg);
            }
            path
        })
        .parse_stream(state_stream)
        .into()
}

#[allow(dead_code)]
fn _use_quoted(state: &mut StateStream<'_>) {
    let _ = quoted_string(state);
    let _ = parse_custom_path(state);
}

impl Custom {
    /// Split a dotted path into namespace + key at the last dot.
    #[allow(dead_code)]
    pub fn split_path(path: String) -> Option<Custom> {
        let dot = path.rfind('.')?;
        Some(Custom {
            namespace: path[..dot].to_string(),
            key: path[dot + 1..].to_string(),
            value: AttrValue::Flag,
        })
    }
}
