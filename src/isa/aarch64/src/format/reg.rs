//! Register and condition spelling shared by every family.

use crate::decode::{Condition, Width};
use core::fmt::{self, Formatter};

/// A general register where encoding 31 is the zero register: `x0`..`x30`, `xzr`,
/// `w0`..`w30`, `wzr`.
pub fn gpr(f: &mut Formatter<'_>, width: Width, r: u8) -> fmt::Result {
    let prefix = match width {
        Width::W32 => 'w',
        Width::X64 => 'x',
    };
    if r == 31 {
        write!(f, "{prefix}zr")
    } else {
        write!(f, "{prefix}{r}")
    }
}

/// A general register where encoding 31 is the stack pointer: `sp` / `wsp`.
pub fn gpr_sp(f: &mut Formatter<'_>, width: Width, r: u8) -> fmt::Result {
    match (r, width) {
        (31, Width::X64) => f.write_str("sp"),
        (31, Width::W32) => f.write_str("wsp"),
        _ => gpr(f, width, r),
    }
}

/// The condition code as binutils spells it (`cs`/`cc`, not `hs`/`lo`).
pub fn condition(c: Condition) -> &'static str {
    match c {
        Condition::Eq => "eq",
        Condition::Ne => "ne",
        Condition::Cs => "cs",
        Condition::Cc => "cc",
        Condition::Mi => "mi",
        Condition::Pl => "pl",
        Condition::Vs => "vs",
        Condition::Vc => "vc",
        Condition::Hi => "hi",
        Condition::Ls => "ls",
        Condition::Ge => "ge",
        Condition::Lt => "lt",
        Condition::Gt => "gt",
        Condition::Le => "le",
        Condition::Al => "al",
        Condition::Nv => "nv",
    }
}

/// A SIMD&FP scalar register `bytes` wide: `b`, `h`, `s`, `d` or `q` for 1, 2, 4, 8, 16.
pub fn fp_scalar(f: &mut Formatter<'_>, bytes: u8, r: u8) -> fmt::Result {
    let prefix = match bytes {
        1 => 'b',
        2 => 'h',
        4 => 's',
        8 => 'd',
        _ => 'q',
    };
    write!(f, "{prefix}{r}")
}

/// The element suffix of a vector arrangement: `size` is log2 of the element width
/// in bytes (0 = bytes .. 3 = doublewords), `q` selects 128 bits over 64.
/// `(0, true)` is `16b`, `(2, false)` is `2s`, `(3, false)` is `1d`.
pub fn arrangement(f: &mut Formatter<'_>, size: u8, q: bool) -> fmt::Result {
    let lanes = (if q { 16 } else { 8 }) >> size;
    let element = ['b', 'h', 's', 'd'][usize::from(size & 3)];
    write!(f, "{lanes}{element}")
}

/// A vector register with its arrangement: `v0.16b`.
pub fn vector(f: &mut Formatter<'_>, r: u8, size: u8, q: bool) -> fmt::Result {
    write!(f, "v{r}.")?;
    arrangement(f, size, q)
}
