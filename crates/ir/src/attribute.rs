extern crate alloc;

use alloc::string::String;

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum AttrValue {
    Flag,
    Int(i64),
    Str(String),
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Custom {
    pub namespace: String,
    pub key: String,
    pub value: AttrValue,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Endianness {
    Little,
    Big,
    Native,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Attribute {
    Inline,
    Noreturn,
    Cold,
    Align(u32),
    Endian(Endianness),
    Custom(Custom),
}
