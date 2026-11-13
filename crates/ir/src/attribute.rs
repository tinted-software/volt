extern crate alloc;

use alloc::string::String;

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub enum AttrValue {
    Flag,
    Int(i64),
    Str(String),
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct Custom {
    pub namespace: String,
    pub key: String,
    pub value: AttrValue,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Endianness {
    Little,
    Big,
    Native,
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub enum Attribute {
    Inline,
    Noreturn,
    Cold,
    Align(u32),
    Endian(Endianness),
    Custom(Custom),
}
