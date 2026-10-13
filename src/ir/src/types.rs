extern crate alloc;

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Type(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum FloatKind {
    F32,
    F64,
    F16,
    F128,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct IntDesc {
    pub bits: u16,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct VectorDesc {
    pub len: u32,
    pub elem: Type,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct ArrayDesc {
    pub len: u64,
    pub elem: Type,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct SliceDesc {
    pub elem: Type,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum AddressSpace {
    Global,
    Shared,
    Private,
    Constant,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum TypeKind {
    Bool,
    Int(IntDesc),
    Float(FloatKind),
    Ptr(AddressSpace),
    Vector(VectorDesc),
    Struct(Vec<Type>),
    Array(ArrayDesc),
    Slice(SliceDesc),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ParseError;

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid type")
    }
}

#[derive(Clone, Debug, Default)]
pub struct TypeTable {
    kinds: Vec<TypeKind>,
}

impl TypeTable {
    pub fn new() -> Self {
        Self { kinds: Vec::new() }
    }

    pub fn intern(&mut self, kind: TypeKind) -> Type {
        for (i, k) in self.kinds.iter().enumerate() {
            if k == &kind {
                return Type(i as u32);
            }
        }
        let h = Type(self.kinds.len() as u32);
        self.kinds.push(kind);
        h
    }

    pub fn ptr_global(&mut self) -> Type {
        self.intern(TypeKind::Ptr(AddressSpace::Global))
    }

    pub fn type_kind(&self, h: Type) -> &TypeKind {
        &self.kinds[h.0 as usize]
    }

    /// Size in bytes of an int/float/ptr scalar as it sits in memory (pointers are 8 bytes);
    /// None for bool, vectors, aggregates and integers whose width is not a whole byte count.
    pub fn scalar_mem_bytes(&self, h: Type) -> Option<u32> {
        match self.type_kind(h) {
            TypeKind::Int(d) if d.bits % 8 == 0 && d.bits != 0 => Some(u32::from(d.bits) / 8),
            TypeKind::Float(FloatKind::F16) => Some(2),
            TypeKind::Float(FloatKind::F32) => Some(4),
            TypeKind::Float(FloatKind::F64) => Some(8),
            TypeKind::Float(FloatKind::F128) => Some(16),
            TypeKind::Ptr(_) => Some(8),
            _ => None,
        }
    }

    pub fn count(&self) -> usize {
        self.kinds.len()
    }

    pub fn kinds(&self) -> &[TypeKind] {
        &self.kinds
    }

    pub fn display<'a>(&'a self, ty: Type) -> TypeDisplay<'a> {
        TypeDisplay { table: self, ty }
    }

    pub fn parse_type(&mut self, text: &str) -> Result<Type, ParseError> {
        let mut p = Parser {
            src: text.as_bytes(),
            pos: 0,
        };
        let ty = self.parse_inner(&mut p)?;
        p.skip_ws();
        if p.pos != p.src.len() {
            return Err(ParseError);
        }
        Ok(ty)
    }

    fn parse_inner(&mut self, p: &mut Parser<'_>) -> Result<Type, ParseError> {
        p.skip_ws();
        let c = *p.src.get(p.pos).ok_or(ParseError)?;
        match c {
            b'<' => {
                p.pos += 1;
                p.skip_ws();
                let len = p.read_number()?;
                p.skip_ws();
                if p.src.get(p.pos) != Some(&b'x') {
                    return Err(ParseError);
                }
                p.pos += 1;
                let elem = self.parse_inner(p)?;
                p.skip_ws();
                if p.src.get(p.pos) != Some(&b'>') {
                    return Err(ParseError);
                }
                p.pos += 1;
                Ok(self.intern(TypeKind::Vector(VectorDesc {
                    len: len as u32,
                    elem,
                })))
            }
            b'[' => {
                p.pos += 1;
                if p.src.get(p.pos) == Some(&b']') {
                    p.pos += 1;
                    let elem = self.parse_inner(p)?;
                    return Ok(self.intern(TypeKind::Slice(SliceDesc { elem })));
                }
                p.skip_ws();
                let len = p.read_number()?;
                p.skip_ws();
                if p.src.get(p.pos) != Some(&b'x') {
                    return Err(ParseError);
                }
                p.pos += 1;
                let elem = self.parse_inner(p)?;
                p.skip_ws();
                if p.src.get(p.pos) != Some(&b']') {
                    return Err(ParseError);
                }
                p.pos += 1;
                Ok(self.intern(TypeKind::Array(ArrayDesc { len, elem })))
            }
            b'{' => {
                p.pos += 1;
                let mut fields: Vec<Type> = Vec::new();
                loop {
                    p.skip_ws();
                    if p.src.get(p.pos) == Some(&b'}') {
                        p.pos += 1;
                        break;
                    }
                    let f = self.parse_inner(p)?;
                    fields.push(f);
                    p.skip_ws();
                    if p.src.get(p.pos) == Some(&b',') {
                        p.pos += 1;
                        continue;
                    }
                    p.skip_ws();
                    if p.src.get(p.pos) == Some(&b'}') {
                        p.pos += 1;
                        break;
                    }
                    return Err(ParseError);
                }
                Ok(self.intern(TypeKind::Struct(fields)))
            }
            _ => {
                let w = p.read_word();
                if w.is_empty() {
                    return Err(ParseError);
                }
                match w {
                    "bool" => Ok(self.intern(TypeKind::Bool)),
                    "ptr" => {
                        if p.src.get(p.pos) != Some(&b'(') {
                            return Ok(self.intern(TypeKind::Ptr(AddressSpace::Global)));
                        }
                        p.pos += 1;
                        let sw = p.read_word().to_string();
                        if p.src.get(p.pos) != Some(&b')') {
                            return Err(ParseError);
                        }
                        p.pos += 1;
                        let space = match sw.as_str() {
                            "global" => AddressSpace::Global,
                            "shared" => AddressSpace::Shared,
                            "private" => AddressSpace::Private,
                            "constant" => AddressSpace::Constant,
                            _ => return Err(ParseError),
                        };
                        Ok(self.intern(TypeKind::Ptr(space)))
                    }
                    "f32" => Ok(self.intern(TypeKind::Float(FloatKind::F32))),
                    "f64" => Ok(self.intern(TypeKind::Float(FloatKind::F64))),
                    "f16" => Ok(self.intern(TypeKind::Float(FloatKind::F16))),
                    "f128" => Ok(self.intern(TypeKind::Float(FloatKind::F128))),
                    _ => {
                        let b = w.as_bytes();
                        if b.len() >= 2 && b[0] == b'i' {
                            let bits: u16 = w[1..].parse().map_err(|_| ParseError)?;
                            Ok(self.intern(TypeKind::Int(IntDesc { bits })))
                        } else {
                            Err(ParseError)
                        }
                    }
                }
            }
        }
    }

    pub fn render(&self, ty: Type, out: &mut String) {
        match &self.kinds[ty.0 as usize] {
            TypeKind::Bool => out.push_str("bool"),
            TypeKind::Int(i) => {
                out.push('i');
                out.push_str(&alloc::format!("{}", i.bits));
            }
            TypeKind::Float(f) => out.push_str(match f {
                FloatKind::F32 => "f32",
                FloatKind::F64 => "f64",
                FloatKind::F16 => "f16",
                FloatKind::F128 => "f128",
            }),
            TypeKind::Ptr(AddressSpace::Global) => out.push_str("ptr"),
            TypeKind::Ptr(s) => {
                out.push_str("ptr(");
                out.push_str(match s {
                    AddressSpace::Global => "global",
                    AddressSpace::Shared => "shared",
                    AddressSpace::Private => "private",
                    AddressSpace::Constant => "constant",
                });
                out.push(')');
            }
            TypeKind::Vector(v) => {
                out.push_str(&alloc::format!("<{} x ", v.len));
                self.render(v.elem, out);
                out.push('>');
            }
            TypeKind::Array(a) => {
                out.push_str(&alloc::format!("[{} x ", a.len));
                self.render(a.elem, out);
                out.push(']');
            }
            TypeKind::Slice(s) => {
                out.push_str("[]");
                self.render(s.elem, out);
            }
            TypeKind::Struct(fields) => {
                out.push_str("{ ");
                for (i, f) in fields.iter().enumerate() {
                    if i != 0 {
                        out.push_str(", ");
                    }
                    self.render(*f, out);
                }
                out.push_str(" }");
            }
        }
    }
}

pub struct TypeDisplay<'a> {
    table: &'a TypeTable,
    ty: Type,
}

impl fmt::Display for TypeDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = String::new();
        self.table.render(self.ty, &mut s);
        write!(f, "{s}")
    }
}

struct Parser<'a> {
    src: &'a [u8],
    pos: usize,
}

impl Parser<'_> {
    fn skip_ws(&mut self) {
        while self.src.get(self.pos) == Some(&b' ') {
            self.pos += 1;
        }
    }
    fn read_word(&mut self) -> &str {
        let start = self.pos;
        while let Some(&c) = self.src.get(self.pos) {
            if c.is_ascii_alphanumeric() || c == b'_' {
                self.pos += 1;
            } else {
                break;
            }
        }
        core::str::from_utf8(&self.src[start..self.pos]).unwrap_or("")
    }
    fn read_number(&mut self) -> Result<u64, ParseError> {
        let start = self.pos;
        while matches!(self.src.get(self.pos), Some(c) if c.is_ascii_digit()) {
            self.pos += 1;
        }
        if start == self.pos {
            return Err(ParseError);
        }
        core::str::from_utf8(&self.src[start..self.pos])
            .ok()
            .and_then(|s| s.parse().ok())
            .ok_or(ParseError)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalars_round_trip() {
        let mut t = TypeTable::new();
        let i32a = t.intern(TypeKind::Int(IntDesc { bits: 32 }));
        assert_eq!(i32a, t.parse_type("i32").unwrap());
        assert_eq!(t.parse_type("bool").unwrap(), t.intern(TypeKind::Bool));
        assert_eq!(t.ptr_global(), t.parse_type("ptr").unwrap());
        assert_eq!("i32", alloc::format!("{}", t.display(i32a)));
    }

    #[test]
    fn integer_types_are_signless_and_the_u_spelling_is_gone() {
        let mut t = TypeTable::new();
        let i8t = t.parse_type("i8").unwrap();
        assert_eq!(i8t, t.intern(TypeKind::Int(IntDesc { bits: 8 })));
        assert_eq!(i8t, t.parse_type(" i8 ").unwrap());
        for text in ["u8", "u32", "u64", "i", "i-1", "i8x", "i99999"] {
            assert!(t.parse_type(text).is_err(), "{text}");
        }
    }

    #[test]
    fn dedup_and_spaces() {
        let mut t = TypeTable::new();
        let g = t.intern(TypeKind::Ptr(AddressSpace::Global));
        let s = t.intern(TypeKind::Ptr(AddressSpace::Shared));
        assert_ne!(g, s);
        assert_eq!(g, t.ptr_global());
        assert_eq!(s, t.parse_type("ptr(shared)").unwrap());
        assert!(t.parse_type("ptr(nonsense)").is_err());
    }

    #[test]
    fn composites() {
        let mut t = TypeTable::new();
        let i32t = t.intern(TypeKind::Int(IntDesc { bits: 32 }));
        let f32t = t.intern(TypeKind::Float(FloatKind::F32));
        let v4 = t.intern(TypeKind::Vector(VectorDesc { len: 4, elem: i32t }));
        assert_eq!(v4, t.parse_type("<4 x i32>").unwrap());
        let st = t.intern(TypeKind::Struct(alloc::vec![i32t, f32t]));
        assert_eq!(st, t.parse_type("{ i32, f32 }").unwrap());
        assert_eq!("<4 x i32>", alloc::format!("{}", t.display(v4)));
    }
}
