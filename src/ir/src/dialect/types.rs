//! The volt type system as pliron types.
//!
//! Every variant of the legacy [`TypeKind`](crate::types::TypeKind) has a
//! corresponding pliron type in the `volt` dialect:
//!
//! | volt type              | example syntax                          |
//! |------------------------|-----------------------------------------|
//! | [`BoolType`]           | `volt.bool`                             |
//! | [`IntType`]            | `volt.int<true, 32>`                    |
//! | [`FloatType`]          | `volt.float<f32>`                       |
//! | [`PtrType`]            | `volt.ptr(global)`                      |
//! | [`VectorType`]         | `volt.vector<4 x volt.int<true, 32>>`   |
//! | [`ArrayType`]          | `volt.array[8 x volt.float<f32>]`       |
//! | [`SliceType`]          | `volt.slice[]volt.int<false, 8>`        |
//! | [`StructType`]         | `volt.struct{...}`                     |
//!
//! [`describe`] inspects any [`TypeHandle`] and returns an owned
//! [`VoltKind`], which is the main entry point for instruction selection.

use alloc::vec::Vec;
use core::hash::Hash;

use pliron::{
    combine::{
        Parser, attempt, choice,
        parser::char::{spaces, string},
        parser::token::token,
    },
    context::Context,
    derive::pliron_type,
    parsable::{Parsable, ParseResult, StateStream},
    printable::{self, Printable},
    r#type::TypeHandle,
};

use super::super::types::{AddressSpace, FloatKind};

/// A boolean value.
#[pliron_type(name = "volt.bool", generate_get = true, format, verifier = "succ")]
#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy, Default)]
pub struct BoolType;

impl BoolType {
    /// Intern (or fetch) the boolean type.
    pub fn get_type(ctx: &Context) -> TypeHandle {
        Self::get(ctx).into()
    }
}

/// A fixed-width integer. `signed` selects the `i`/`u` flavor.
#[pliron_type(
    name = "volt.int",
    generate_get = true,
    format = "`<` $signed `, ` $bits `>`",
    verifier = "succ"
)]
#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy)]
pub struct IntType {
    signed: bool,
    bits: u32,
}

impl IntType {
    /// Intern (or fetch) an integer type.
    pub fn get_type(ctx: &Context, signed: bool, bits: u32) -> TypeHandle {
        Self::get(ctx, signed, bits).into()
    }

    /// Is this the signed flavor?
    pub fn is_signed(&self) -> bool {
        self.signed
    }

    /// Bit width.
    pub fn bits(&self) -> u32 {
        self.bits
    }
}

/// A floating-point scalar. Prints with the legacy keyword (`f32`, ...).
#[pliron_type(
    name = "volt.float",
    generate_get = true,
    format = "`<` $kind `>`",
    verifier = "succ"
)]
#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy)]
pub struct FloatType {
    kind: FloatKind,
}

impl FloatType {
    /// Intern (or fetch) a float type.
    pub fn get_type(ctx: &Context, kind: FloatKind) -> TypeHandle {
        Self::get(ctx, kind).into()
    }

    /// The float flavor.
    pub fn kind(&self) -> FloatKind {
        self.kind
    }
}

/// A pointer into an address space.
#[pliron_type(
    name = "volt.ptr",
    generate_get = true,
    format = "`(` $space `)`",
    verifier = "succ"
)]
#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy)]
pub struct PtrType {
    space: AddressSpace,
}

impl PtrType {
    /// Intern (or fetch) a pointer type.
    pub fn get_type(ctx: &Context, space: AddressSpace) -> TypeHandle {
        Self::get(ctx, space).into()
    }

    /// The address space.
    pub fn space(&self) -> AddressSpace {
        self.space
    }
}

/// A fixed-length SIMD vector.
#[pliron_type(
    name = "volt.vector",
    generate_get = true,
    format = "`<` $len ` x ` $elem `>`",
    verifier = "succ"
)]
#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy)]
pub struct VectorType {
    len: u32,
    elem: TypeHandle,
}

impl VectorType {
    /// Intern (or fetch) a vector type.
    pub fn get_type(ctx: &Context, len: u32, elem: TypeHandle) -> TypeHandle {
        Self::get(ctx, len, elem).into()
    }

    /// Lane count.
    pub fn len(&self) -> u32 {
        self.len
    }

    /// Lane type.
    pub fn elem(&self) -> TypeHandle {
        self.elem
    }
}

/// A fixed-length array.
#[pliron_type(
    name = "volt.array",
    generate_get = true,
    format = "`[` $len ` x ` $elem `]`",
    verifier = "succ"
)]
#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy)]
pub struct ArrayType {
    len: u64,
    elem: TypeHandle,
}

impl ArrayType {
    /// Intern (or fetch) an array type.
    pub fn get_type(ctx: &Context, len: u64, elem: TypeHandle) -> TypeHandle {
        Self::get(ctx, len, elem).into()
    }

    /// Element count.
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Element type.
    pub fn elem(&self) -> TypeHandle {
        self.elem
    }
}

/// An unsized slice (fat pointer payload).
#[pliron_type(
    name = "volt.slice",
    generate_get = true,
    format = "`[` `]` $elem",
    verifier = "succ"
)]
#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy)]
pub struct SliceType {
    elem: TypeHandle,
}

impl SliceType {
    /// Intern (or fetch) a slice type.
    pub fn get_type(ctx: &Context, elem: TypeHandle) -> TypeHandle {
        Self::get(ctx, elem).into()
    }

    /// Element type.
    pub fn elem(&self) -> TypeHandle {
        self.elem
    }
}

/// A heterogeneous struct.
#[pliron_type(
    name = "volt.struct",
    generate_get = true,
    format = "`{` vec($fields, CharSpace(`,`)) `}`",
    verifier = "succ"
)]
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct StructType {
    fields: Vec<TypeHandle>,
}

impl StructType {
    /// Intern (or fetch) a struct type.
    pub fn get_type(ctx: &Context, fields: Vec<TypeHandle>) -> TypeHandle {
        Self::get(ctx, fields).into()
    }

    /// Field types.
    pub fn fields(&self) -> &[TypeHandle] {
        &self.fields
    }
}

/// An owned description of any volt type, mirroring the legacy `TypeKind`.
/// This is the main inspection entry point for instruction selection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VoltKind {
    /// `volt.bool`.
    Bool,
    /// `volt.int`.
    Int { signed: bool, bits: u32 },
    /// `volt.float`.
    Float(FloatKind),
    /// `volt.ptr`.
    Ptr(AddressSpace),
    /// `volt.vector`.
    Vector { len: u32, elem: TypeHandle },
    /// `volt.array`.
    Array { len: u64, elem: TypeHandle },
    /// `volt.slice`.
    Slice { elem: TypeHandle },
    /// `volt.struct`.
    Struct(Vec<TypeHandle>),
}

/// Describe a type handle as an owned [`VoltKind`]. Returns `None` for
/// non-volt types (e.g. `builtin.function`).
pub fn describe(ctx: &Context, ty: TypeHandle) -> Option<VoltKind> {
    let obj = ty.deref(ctx);
    if let Some(t) = obj.downcast_ref::<BoolType>() {
        let _ = t;
        return Some(VoltKind::Bool);
    }
    if let Some(t) = obj.downcast_ref::<IntType>() {
        return Some(VoltKind::Int {
            signed: t.is_signed(),
            bits: t.bits(),
        });
    }
    if let Some(t) = obj.downcast_ref::<FloatType>() {
        return Some(VoltKind::Float(t.kind()));
    }
    if let Some(t) = obj.downcast_ref::<PtrType>() {
        return Some(VoltKind::Ptr(t.space()));
    }
    if let Some(t) = obj.downcast_ref::<VectorType>() {
        return Some(VoltKind::Vector {
            len: t.len(),
            elem: t.elem(),
        });
    }
    if let Some(t) = obj.downcast_ref::<ArrayType>() {
        return Some(VoltKind::Array {
            len: t.len(),
            elem: t.elem(),
        });
    }
    if let Some(t) = obj.downcast_ref::<SliceType>() {
        return Some(VoltKind::Slice { elem: t.elem() });
    }
    if let Some(t) = obj.downcast_ref::<StructType>() {
        return Some(VoltKind::Struct(t.fields().to_vec()));
    }
    None
}

/// Intern an owned [`VoltKind`] description.
pub fn intern(ctx: &Context, kind: &VoltKind) -> TypeHandle {
    match kind {
        VoltKind::Bool => BoolType::get_type(ctx),
        VoltKind::Int { signed, bits } => IntType::get_type(ctx, *signed, *bits),
        VoltKind::Float(k) => FloatType::get_type(ctx, *k),
        VoltKind::Ptr(s) => PtrType::get_type(ctx, *s),
        VoltKind::Vector { len, elem } => VectorType::get_type(ctx, *len, *elem),
        VoltKind::Array { len, elem } => ArrayType::get_type(ctx, *len, *elem),
        VoltKind::Slice { elem } => SliceType::get_type(ctx, *elem),
        VoltKind::Struct(f) => StructType::get_type(ctx, f.clone()),
    }
}

// The `$kind` / `$space` format fields above need print/parse support for
// the legacy descriptor enums. These print with the legacy keywords.

impl Printable for FloatKind {
    fn fmt(
        &self,
        _ctx: &Context,
        _state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        let s = match self {
            FloatKind::F32 => "f32",
            FloatKind::F64 => "f64",
            FloatKind::F16 => "f16",
            FloatKind::F128 => "f128",
        };
        f.write_str(s)
    }
}

impl Printable for AddressSpace {
    fn fmt(
        &self,
        _ctx: &Context,
        _state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        let s = match self {
            AddressSpace::Global => "global",
            AddressSpace::Shared => "shared",
            AddressSpace::Private => "private",
            AddressSpace::Constant => "constant",
        };
        f.write_str(s)
    }
}

impl Parsable for FloatKind {
    type Arg = ();
    type Parsed = Self;
    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        spaces()
            .with(token('f').with(choice((
                string("32").map(|_| FloatKind::F32),
                string("64").map(|_| FloatKind::F64),
                attempt(string("128")).map(|_| FloatKind::F128),
                string("16").map(|_| FloatKind::F16),
            ))))
            .parse_stream(state_stream)
            .into()
    }
}

impl Parsable for AddressSpace {
    type Arg = ();
    type Parsed = Self;
    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        spaces()
            .with(choice((
                string("global").map(|_| AddressSpace::Global),
                string("shared").map(|_| AddressSpace::Shared),
                string("private").map(|_| AddressSpace::Private),
                string("constant").map(|_| AddressSpace::Constant),
            )))
            .parse_stream(state_stream)
            .into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pliron::{
        parsable::{Parsable, parse_from_str},
        result::ExpectOk,
    };

    fn ctx() -> Context {
        Context::new()
    }

    #[test]
    fn int_type_round_trip() {
        let ctx = &mut ctx();
        let ty = IntType::get_type(ctx, true, 32);
        assert_eq!(
            describe(ctx, ty),
            Some(VoltKind::Int {
                signed: true,
                bits: 32
            })
        );
        let parsed =
            parse_from_str(TypeHandle::parser(()), ctx, "volt.int<true, 32>").expect_ok(ctx);
        assert_eq!(parsed, ty);
    }

    #[test]
    fn composite_type_round_trip() {
        let ctx = &mut ctx();
        let i32t = IntType::get_type(ctx, true, 32);
        let v = VectorType::get_type(ctx, 4, i32t);
        let printed = alloc::format!("{}", v.print(ctx, &Default::default()));
        let back = parse_from_str(TypeHandle::parser(()), ctx, &printed).expect_ok(ctx);
        assert_eq!(back, v);

        let s = StructType::get_type(ctx, alloc::vec![i32t, BoolType::get_type(ctx)]);
        let printed = alloc::format!("{}", s.print(ctx, &Default::default()));
        let back = parse_from_str(TypeHandle::parser(()), ctx, &printed).expect_ok(ctx);
        assert_eq!(back, s);
    }

    #[test]
    fn float_ptr_kinds() {
        let ctx = &mut ctx();
        for (text, kind) in [
            ("volt.float<f32>", VoltKind::Float(FloatKind::F32)),
            ("volt.float<f128>", VoltKind::Float(FloatKind::F128)),
            ("volt.ptr(shared)", VoltKind::Ptr(AddressSpace::Shared)),
            (
                "volt.array[8 x volt.float<f64>]",
                VoltKind::Array {
                    len: 8,
                    elem: FloatType::get_type(ctx, FloatKind::F64),
                },
            ),
        ] {
            let parsed = parse_from_str(TypeHandle::parser(()), ctx, text).expect_ok(ctx);
            assert_eq!(describe(ctx, parsed), Some(kind), "{text}");
        }
    }
}
