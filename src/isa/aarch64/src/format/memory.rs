//! Loads and stores: scalar, pair, literal, exclusive, LSE atomics, prefetch and the
//! SIMD&FP and structure forms.
//!
//! Owns `Memory`, `Pair`, `Literal`, `SimdLiteral`, `Exclusive`, `Atomic`, `Prefetch`,
//! `PrefetchLiteral`, `SimdPair`, `SimdMemory` and `SimdStruct`.
//!
//! The decoded forms keep what distinguishes the instructions: `Memory::ordered` for
//! `ldar`/`stlr`/`ldapr`/`ldlar`/`stllr`, `MemoryOrder` for the `a`/`l` of exclusives and
//! atomics, `Addressing::Unscaled` for `ldur`/`prfum`, `Pair::non_temporal` for `ldnp` and
//! `SimdAccess::Ld1` for a one-register `ld1`/`st1`.

use super::Context;
use super::reg::{fp_scalar, gpr, gpr_sp, vector};
use crate::decode::{
    Addressing, Atomic, AtomicKind, Exclusive, Extend, Instruction, Literal, Memory, MemoryOp,
    MemoryOrder, Ordered, Pair, Prefetch, PrefetchLiteral, RangePrefetch, SignExtend, SimdAccess,
    SimdLiteral, SimdMemory, SimdPair, SimdStruct, Size, StructPost, Width,
};
use crate::simd_struct::{Shape, StructDesc};
use core::fmt::{self, Formatter};

pub(super) fn write(
    f: &mut Formatter<'_>,
    cx: &Context<'_>,
    instruction: &Instruction,
) -> fmt::Result {
    match instruction {
        Instruction::Memory(i) => memory(f, i),
        Instruction::Pair(i) => pair(f, i),
        Instruction::Literal(i) => literal(f, cx, i),
        Instruction::SimdLiteral(i) => simd_literal(f, cx, i),
        Instruction::Exclusive(i) => exclusive(f, i),
        Instruction::Atomic(i) => atomic(f, i),
        Instruction::Prefetch(i) => prefetch(f, i),
        Instruction::PrefetchLiteral(i) => prefetch_literal(f, cx, i),
        Instruction::RangePrefetch(i) => range_prefetch(f, i),
        Instruction::SimdPair(i) => simd_pair(f, i),
        Instruction::SimdMemory(i) => simd_memory(f, i),
        Instruction::SimdStruct(i) => simd_struct(f, i),
        _ => unreachable!("{instruction:?} is not a memory instruction"),
    }
}

fn width_of(size: Size) -> Width {
    if size == Size::Double {
        Width::X64
    } else {
        Width::W32
    }
}

/// `[xn]`, `[xn, #imm]`, `[xn, #imm]!`, `[xn], #imm` or `[xn, rm{, ext{ #n}}]`.
fn address(f: &mut Formatter<'_>, rn: u8, addressing: Addressing) -> fmt::Result {
    f.write_str("[")?;
    gpr_sp(f, Width::X64, rn)?;
    match addressing {
        Addressing::Offset(0) | Addressing::Unscaled(0) | Addressing::Unprivileged(0) => {
            f.write_str("]")
        }
        Addressing::Offset(offset)
        | Addressing::Unscaled(offset)
        | Addressing::Unprivileged(offset) => write!(f, ", #{offset}]"),
        Addressing::PreIndex(offset) => write!(f, ", #{offset}]!"),
        Addressing::PostIndex(offset) => write!(f, "], #{offset}"),
        Addressing::Register { rm, extend, amount } => {
            f.write_str(", ")?;
            let wide = matches!(extend, Extend::Uxtx | Extend::Sxtx);
            gpr(f, if wide { Width::X64 } else { Width::W32 }, rm)?;
            let name = match (extend, amount) {
                (Extend::Uxtx, None) => return f.write_str("]"),
                (Extend::Uxtx, Some(_)) => "lsl",
                (Extend::Uxtb, _) => "uxtb",
                (Extend::Uxth, _) => "uxth",
                (Extend::Uxtw, _) => "uxtw",
                (Extend::Sxtb, _) => "sxtb",
                (Extend::Sxth, _) => "sxth",
                (Extend::Sxtw, _) => "sxtw",
                (Extend::Sxtx, _) => "sxtx",
            };
            write!(f, ", {name}")?;
            if let Some(amount) = amount {
                write!(f, " #{amount}")?;
            }
            f.write_str("]")
        }
    }
}

fn memory(f: &mut Formatter<'_>, i: &Memory) -> fmt::Result {
    let (op, suffix) = match (i.op, i.size, i.signed) {
        (MemoryOp::Load, Size::Byte, SignExtend::None) => ("ld", "b"),
        (MemoryOp::Load, Size::Half, SignExtend::None) => ("ld", "h"),
        (MemoryOp::Load, Size::Byte, _) => ("ld", "sb"),
        (MemoryOp::Load, Size::Half, _) => ("ld", "sh"),
        (MemoryOp::Load, Size::Word, SignExtend::None) => ("ld", ""),
        (MemoryOp::Load, Size::Word, _) => ("ld", "sw"),
        (MemoryOp::Load, Size::Double, _) => ("ld", ""),
        (MemoryOp::Store, Size::Byte, _) => ("st", "b"),
        (MemoryOp::Store, Size::Half, _) => ("st", "h"),
        (MemoryOp::Store, _, _) => ("st", ""),
    };
    // The middle of the mnemonic: `ldr`, `ldur`, `ldtr`, `ldar`, `ldapr`, `ldapur`,
    // `ldlar`, `stlr`, `stlur`, `stllr`.
    let unscaled = matches!(i.addressing, Addressing::Unscaled(_));
    let stem = match i.ordered {
        None => match i.addressing {
            Addressing::Unscaled(_) => "ur",
            Addressing::Unprivileged(_) => "tr",
            _ => "r",
        },
        Some(Ordered::Acquire) => "ar",
        Some(Ordered::AcquirePc) if unscaled => "apur",
        Some(Ordered::AcquirePc) => "apr",
        Some(Ordered::LimitedAcquire) => "lar",
        Some(Ordered::Release) if unscaled => "lur",
        Some(Ordered::Release) => "lr",
        Some(Ordered::LimitedRelease) => "llr",
    };
    let width = if i.signed == SignExtend::To64 {
        Width::X64
    } else if i.signed == SignExtend::To32 {
        Width::W32
    } else {
        width_of(i.size)
    };
    write!(f, "{op}{stem}{suffix} ")?;
    gpr(f, width, i.rt)?;
    f.write_str(", ")?;
    address(f, i.rn, i.addressing)
}

fn pair(f: &mut Formatter<'_>, i: &Pair) -> fmt::Result {
    let op = match (i.op, i.signed, i.non_temporal) {
        (MemoryOp::Load, SignExtend::None, false) => "ldp",
        (MemoryOp::Load, SignExtend::None, true) => "ldnp",
        (MemoryOp::Load, _, _) => "ldpsw",
        (MemoryOp::Store, _, false) => "stp",
        (MemoryOp::Store, _, true) => "stnp",
    };
    let width = if i.signed != SignExtend::None {
        Width::X64
    } else {
        width_of(i.size)
    };
    write!(f, "{op} ")?;
    gpr(f, width, i.rt)?;
    f.write_str(", ")?;
    gpr(f, width, i.rt2)?;
    f.write_str(", ")?;
    address(f, i.rn, i.addressing)
}

fn literal(f: &mut Formatter<'_>, cx: &Context<'_>, i: &Literal) -> fmt::Result {
    if i.signed == SignExtend::None {
        f.write_str("ldr ")?;
        gpr(f, width_of(i.size), i.rt)?;
    } else {
        f.write_str("ldrsw ")?;
        gpr(f, Width::X64, i.rt)?;
    }
    f.write_str(", ")?;
    cx.target(f, i.offset)
}

fn simd_literal(f: &mut Formatter<'_>, cx: &Context<'_>, i: &SimdLiteral) -> fmt::Result {
    f.write_str("ldr ")?;
    fp_scalar(f, i.bytes, i.rt)?;
    f.write_str(", ")?;
    cx.target(f, i.offset)
}

fn exclusive(f: &mut Formatter<'_>, i: &Exclusive) -> fmt::Result {
    let (op, suffix) = (
        match (i.op, i.order) {
            (MemoryOp::Load, MemoryOrder { acquire: true, .. }) => "ldaxr",
            (MemoryOp::Load, _) => "ldxr",
            (MemoryOp::Store, MemoryOrder { release: true, .. }) => "stlxr",
            (MemoryOp::Store, _) => "stxr",
        },
        match i.size {
            Size::Byte => "b",
            Size::Half => "h",
            _ => "",
        },
    );
    write!(f, "{op}{suffix} ")?;
    if i.op == MemoryOp::Store {
        gpr(f, Width::W32, i.rs)?;
        f.write_str(", ")?;
    }
    gpr(f, width_of(i.size), i.rt)?;
    f.write_str(", ")?;
    address(f, i.rn, Addressing::Offset(0))
}

fn atomic(f: &mut Formatter<'_>, i: &Atomic) -> fmt::Result {
    let width = width_of(i.size);
    let suffix = match i.size {
        Size::Byte => "b",
        Size::Half => "h",
        _ => "",
    };
    let a = if i.order.acquire { "a" } else { "" };
    let l = if i.order.release { "l" } else { "" };
    let op = match i.kind {
        AtomicKind::Add => "add",
        AtomicKind::Clr => "clr",
        AtomicKind::Eor => "eor",
        AtomicKind::Set => "set",
        AtomicKind::SMax => "smax",
        AtomicKind::SMin => "smin",
        AtomicKind::UMax => "umax",
        AtomicKind::UMin => "umin",
        AtomicKind::Swap | AtomicKind::Cas | AtomicKind::Casp => "",
        AtomicKind::LoadPair | AtomicKind::StorePair => "",
    };
    let memory = Addressing::Offset(0);
    match i.kind {
        AtomicKind::Swap => {
            write!(f, "swp{a}{l}{suffix} ")?;
            gpr(f, width, i.rs)?;
            f.write_str(", ")?;
            gpr(f, width, i.rt)?;
        }
        AtomicKind::Cas => {
            write!(f, "cas{a}{l}{suffix} ")?;
            gpr(f, width, i.rs)?;
            f.write_str(", ")?;
            gpr(f, width, i.rt)?;
        }
        AtomicKind::Casp => {
            write!(f, "casp{a}{l} ")?;
            gpr(f, width, i.rs)?;
            f.write_str(", ")?;
            gpr(f, width, i.rs.wrapping_add(1) & 31)?;
            f.write_str(", ")?;
            gpr(f, width, i.rt)?;
            f.write_str(", ")?;
            gpr(f, width, i.rt.wrapping_add(1) & 31)?;
        }
        AtomicKind::LoadPair => {
            write!(f, "ld{a}xp ")?;
            gpr(f, width, i.rt)?;
            f.write_str(", ")?;
            gpr(f, width, i.rt2)?;
        }
        AtomicKind::StorePair => {
            write!(f, "st{l}xp ")?;
            gpr(f, Width::W32, i.rs)?;
            f.write_str(", ")?;
            gpr(f, width, i.rt)?;
            f.write_str(", ")?;
            gpr(f, width, i.rt2)?;
        }
        _ if i.rt == 31 && !i.order.acquire => {
            // `ldadd Rs, xzr, [Xn]` is `stadd Rs, [Xn]`; with acquire the load must
            // happen, so `ldadda Rs, xzr, [Xn]` keeps its spelling.
            write!(f, "st{op}{l}{suffix} ")?;
            gpr(f, width, i.rs)?;
        }
        _ => {
            write!(f, "ld{op}{a}{l}{suffix} ")?;
            gpr(f, width, i.rs)?;
            f.write_str(", ")?;
            gpr(f, width, i.rt)?;
        }
    }
    f.write_str(", ")?;
    address(f, i.rn, memory)
}

/// The operation operand of `prfm`: `pldl1keep`.. or `#0x..` when unnamed. Operation 24
/// is `ir` only for the scaled-offset form (`ir_form`); every other form spells it `#0x18`.
fn prefetch_operation(f: &mut Formatter<'_>, op: u8, ir_form: bool) -> fmt::Result {
    if op == 24 && ir_form {
        return f.write_str("ir");
    }
    let kind = match op >> 3 {
        0 => "pld",
        1 => "pli",
        2 => "pst",
        _ => return write!(f, "#{op:#x}"),
    };
    let target = ["l1", "l2", "l3", "slc"][usize::from((op >> 1) & 3)];
    let policy = if op & 1 == 0 { "keep" } else { "strm" };
    write!(f, "{kind}{target}{policy}")
}

fn prefetch(f: &mut Formatter<'_>, i: &Prefetch) -> fmt::Result {
    let u = if matches!(i.addressing, Addressing::Unscaled(_)) {
        "u"
    } else {
        ""
    };
    let ir_form = matches!(i.addressing, Addressing::Offset(_));
    write!(f, "prf{u}m ")?;
    prefetch_operation(f, i.op, ir_form)?;
    f.write_str(", ")?;
    address(f, i.rn, i.addressing)
}

fn prefetch_literal(f: &mut Formatter<'_>, cx: &Context<'_>, i: &PrefetchLiteral) -> fmt::Result {
    f.write_str("prfm ")?;
    prefetch_operation(f, i.op, false)?;
    f.write_str(", ")?;
    cx.target(f, i.offset)
}

/// `rprfm pldkeep, xm, [xn]`: the four named operations, `#n` for the rest.
fn range_prefetch(f: &mut Formatter<'_>, i: &RangePrefetch) -> fmt::Result {
    f.write_str("rprfm ")?;
    match i.op {
        0 => f.write_str("pldkeep")?,
        1 => f.write_str("pstkeep")?,
        4 => f.write_str("pldstrm")?,
        5 => f.write_str("pststrm")?,
        n => write!(f, "#{n}")?,
    }
    f.write_str(", ")?;
    gpr(f, Width::X64, i.rm)?;
    f.write_str(", ")?;
    address(f, i.rn, Addressing::Offset(0))
}

fn simd_pair(f: &mut Formatter<'_>, i: &SimdPair) -> fmt::Result {
    f.write_str(match (i.op, i.non_temporal) {
        (MemoryOp::Load, false) => "ldp ",
        (MemoryOp::Store, false) => "stp ",
        (MemoryOp::Load, true) => "ldnp ",
        (MemoryOp::Store, true) => "stnp ",
    })?;
    fp_scalar(f, i.bytes, i.rt)?;
    f.write_str(", ")?;
    fp_scalar(f, i.bytes, i.rt2)?;
    f.write_str(", ")?;
    address(f, i.rn, i.addressing)
}

fn simd_memory(f: &mut Formatter<'_>, i: &SimdMemory) -> fmt::Result {
    let op = match i.op {
        MemoryOp::Load => "ld",
        MemoryOp::Store => "st",
    };
    match i.access {
        SimdAccess::Register => {
            let u = if matches!(i.addressing, Addressing::Unscaled(_)) {
                "u"
            } else {
                ""
            };
            write!(f, "{op}{u}r ")?;
            fp_scalar(f, i.bytes, i.rt)?;
        }
        SimdAccess::Ld1 { esize } => {
            write!(f, "{op}1 {{")?;
            vector(f, i.rt, esize.trailing_zeros() as u8, i.bytes == 16)?;
            f.write_str("}")?;
        }
    }
    f.write_str(", ")?;
    address(f, i.rn, i.addressing)
}

fn element(esize: u8) -> char {
    match esize {
        1 => 'b',
        2 => 'h',
        4 => 's',
        _ => 'd',
    }
}

fn register_list(f: &mut Formatter<'_>, d: &StructDesc) -> fmt::Result {
    let size = (d.esize.max(1)).trailing_zeros() as u8;
    let last = d.rt.wrapping_add(d.registers.wrapping_sub(1)) & 31;
    f.write_str("{")?;
    for (n, r) in [d.rt, last].into_iter().enumerate() {
        if n == 1 {
            if d.registers <= 1 {
                break;
            }
            f.write_str("-")?;
        }
        match d.shape {
            Shape::Lane => write!(f, "v{r}.{}", element(d.esize))?,
            _ => vector(f, r, size, d.q)?,
        }
    }
    f.write_str("}")?;
    if d.shape == Shape::Lane {
        write!(f, "[{}]", d.lane)?;
    }
    Ok(())
}

fn simd_struct(f: &mut Formatter<'_>, i: &SimdStruct) -> fmt::Result {
    let d = &i.desc;
    let count = if d.shape == Shape::Multiple && !d.interleaved {
        1
    } else {
        d.registers
    };
    let op = if d.store { "st" } else { "ld" };
    let replicate = if d.shape == Shape::Replicate { "r" } else { "" };
    write!(f, "{op}{count}{replicate} ")?;
    register_list(f, d)?;
    f.write_str(", [")?;
    gpr_sp(f, Width::X64, i.rn)?;
    f.write_str("]")?;
    match i.post {
        StructPost::None => Ok(()),
        StructPost::Immediate => write!(f, ", #{}", d.bytes()),
        StructPost::Register(rm) => {
            f.write_str(", ")?;
            gpr(f, Width::X64, rm)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::decode::decode;
    use std::string::ToString;

    /// `(pc, word, text)`: text is what GNU objdump 2.46 prints for `word` at `pc`.
    const ROWS: &[(u64, u32, &str)] = &[
        (0x0, 0xf9400020, "ldr x0, [x1]"),
        (0x4, 0xf9400420, "ldr x0, [x1, #8]"),
        (0x8, 0xf97ffc20, "ldr x0, [x1, #32760]"),
        (0xc, 0xf94007e0, "ldr x0, [sp, #8]"),
        (0x10, 0xf85f8020, "ldur x0, [x1, #-8]"),
        (0x14, 0xf8401020, "ldur x0, [x1, #1]"),
        (0x18, 0xb97ffc20, "ldr w0, [x1, #16380]"),
        (0x1c, 0xf8408c20, "ldr x0, [x1, #8]!"),
        (0x20, 0xf8500fe0, "ldr x0, [sp, #-256]!"),
        (0x24, 0xf84ff420, "ldr x0, [x1], #255"),
        (0x28, 0xb85fc422, "ldr w2, [x1], #-4"),
        (0x2c, 0xf8626820, "ldr x0, [x1, x2]"),
        (0x30, 0xf8627820, "ldr x0, [x1, x2, lsl #3]"),
        (0x34, 0xf87f7820, "ldr x0, [x1, xzr, lsl #3]"),
        (0x38, 0xf8624820, "ldr x0, [x1, w2, uxtw]"),
        (0x3c, 0xf8625820, "ldr x0, [x1, w2, uxtw #3]"),
        (0x40, 0xf862c820, "ldr x0, [x1, w2, sxtw]"),
        (0x44, 0xf862d820, "ldr x0, [x1, w2, sxtw #3]"),
        (0x48, 0xf862e820, "ldr x0, [x1, x2, sxtx]"),
        (0x4c, 0xf862f820, "ldr x0, [x1, x2, sxtx #3]"),
        (0x50, 0xb8627820, "ldr w0, [x1, x2, lsl #2]"),
        (0x54, 0xb87fd820, "ldr w0, [x1, wzr, sxtw #2]"),
        (0x58, 0x397ffc20, "ldrb w0, [x1, #4095]"),
        (0x5c, 0x385ff020, "ldurb w0, [x1, #-1]"),
        (0x60, 0x38626820, "ldrb w0, [x1, x2]"),
        (0x64, 0x3862c820, "ldrb w0, [x1, w2, sxtw]"),
        (0x68, 0x38401420, "ldrb w0, [x1], #1"),
        (0x6c, 0x38401c3f, "ldrb wzr, [x1, #1]!"),
        (0x70, 0x797ffc20, "ldrh w0, [x1, #8190]"),
        (0x74, 0x78403020, "ldurh w0, [x1, #3]"),
        (0x78, 0x78627820, "ldrh w0, [x1, x2, lsl #1]"),
        (0x7c, 0x39c01420, "ldrsb w0, [x1, #5]"),
        (0x80, 0x39801420, "ldrsb x0, [x1, #5]"),
        (0x84, 0x389fb020, "ldursb x0, [x1, #-5]"),
        (0x88, 0x38e26820, "ldrsb w0, [x1, x2]"),
        (0x8c, 0x79c00c20, "ldrsh w0, [x1, #6]"),
        (0x90, 0x79800c20, "ldrsh x0, [x1, #6]"),
        (0x94, 0x78807020, "ldursh x0, [x1, #7]"),
        (0x98, 0x78a27820, "ldrsh x0, [x1, x2, lsl #1]"),
        (0x9c, 0xb9800420, "ldrsw x0, [x1, #4]"),
        (0xa0, 0xb89fc020, "ldursw x0, [x1, #-4]"),
        (0xa4, 0xb8804c20, "ldrsw x0, [x1, #4]!"),
        (0xa8, 0xb8804420, "ldrsw x0, [x1], #4"),
        (0xac, 0xb8a27820, "ldrsw x0, [x1, x2, lsl #2]"),
        (0xb0, 0xf9000020, "str x0, [x1]"),
        (0xb4, 0xf9000bff, "str xzr, [sp, #16]"),
        (0xb8, 0xb81fc03f, "stur wzr, [x1, #-4]"),
        (0xbc, 0xb81fcc20, "str w0, [x1, #-4]!"),
        (0xc0, 0xf8008420, "str x0, [x1], #8"),
        (0xc4, 0xf8227820, "str x0, [x1, x2, lsl #3]"),
        (0xc8, 0x39000420, "strb w0, [x1, #1]"),
        (0xcc, 0x381ff020, "sturb w0, [x1, #-1]"),
        (0xd0, 0x38226820, "strb w0, [x1, x2]"),
        (0xd4, 0x79000420, "strh w0, [x1, #2]"),
        (0xd8, 0x78001020, "sturh w0, [x1, #1]"),
        (0xdc, 0x781fe43f, "strh wzr, [x1], #-2"),
        (0xe0, 0xa9400440, "ldp x0, x1, [x2]"),
        (0xe4, 0xa9408440, "ldp x0, x1, [x2, #8]"),
        (0xe8, 0xa8c17bfd, "ldp x29, x30, [sp], #16"),
        (0xec, 0xa9bf7bfd, "stp x29, x30, [sp, #-16]!"),
        (0xf0, 0xa91ffbfd, "stp x29, x30, [sp, #504]"),
        (0xf4, 0xa9207bfd, "stp x29, x30, [sp, #-512]"),
        (0xf8, 0x295f8440, "ldp w0, w1, [x2, #252]"),
        (0xfc, 0x29600440, "ldp w0, w1, [x2, #-256]"),
        (0x100, 0x28817c40, "stp w0, wzr, [x2], #8"),
        (0x104, 0x69400440, "ldpsw x0, x1, [x2]"),
        (0x108, 0x697f8440, "ldpsw x0, x1, [x2, #-4]"),
        (0x10c, 0x68df8440, "ldpsw x0, x1, [x2], #252"),
        (0x110, 0x18000040, "ldr w0, 0x118"),
        (0x114, 0x58ffffc0, "ldr x0, 0x10c"),
        (0x118, 0x587fffff, "ldr xzr, 0x100114"),
        (0x11c, 0x58800000, "ldr x0, 0xfffffffffff0011c"),
        (0x120, 0x1c000040, "ldr s0, 0x128"),
        (0x124, 0x5c000040, "ldr d0, 0x12c"),
        (0x128, 0x9cffffdf, "ldr q31, 0x120"),
        (0x12c, 0xd8000040, "prfm pldl1keep, 0x134"),
        (0x130, 0xd8ffffd5, "prfm pstl3strm, 0x128"),
        (0x134, 0xd8000059, "prfm #0x19, 0x13c"),
        (0x138, 0xd8000058, "prfm #0x18, 0x140"),
        (0x13c, 0xf9800000, "prfm pldl1keep, [x0]"),
        (0x140, 0xf98007e6, "prfm pldslckeep, [sp, #8]"),
        (0x144, 0xf9bffc0b, "prfm plil2strm, [x0, #32760]"),
        (0x148, 0xf8a16810, "prfm pstl1keep, [x0, x1]"),
        (0x14c, 0xf8a17813, "prfm pstl2strm, [x0, x1, lsl #3]"),
        (0x150, 0xf8a14804, "prfm pldl3keep, [x0, w1, uxtw]"),
        (0x154, 0xf8a1d804, "prfm pldl3keep, [x0, w1, sxtw #3]"),
        (0x158, 0xf980041f, "prfm #0x1f, [x0, #8]"),
        (0x15c, 0xf980041f, "prfm #0x1f, [x0, #8]"),
        (0x160, 0xf9800418, "prfm ir, [x0, #8]"),
        (0x164, 0xf89f8000, "prfum pldl1keep, [x0, #-8]"),
        (0x168, 0xf8801011, "prfum pstl1strm, [x0, #1]"),
        (0x16c, 0xf89ff018, "prfum #0x18, [x0, #-1]"),
        (0x170, 0xf88ff3ff, "prfum #0x1f, [sp, #255]"),
        (0x174, 0x885f7c20, "ldxr w0, [x1]"),
        (0x178, 0xc85f7fe0, "ldxr x0, [sp]"),
        (0x17c, 0xc8027c20, "stxr w2, x0, [x1]"),
        (0x180, 0x88027fe0, "stxr w2, w0, [sp]"),
        (0x184, 0x085f7c20, "ldxrb w0, [x1]"),
        (0x188, 0x08027c20, "stxrb w2, w0, [x1]"),
        (0x18c, 0x485f7c20, "ldxrh w0, [x1]"),
        (0x190, 0x48027c20, "stxrh w2, w0, [x1]"),
        (0x194, 0xb8200041, "ldadd w0, w1, [x2]"),
        (0x198, 0xf82003e1, "ldadd x0, x1, [sp]"),
        (0x19c, 0x38200041, "ldaddb w0, w1, [x2]"),
        (0x1a0, 0x78200041, "ldaddh w0, w1, [x2]"),
        (0x1a4, 0xf8201041, "ldclr x0, x1, [x2]"),
        (0x1a8, 0xb8202041, "ldeor w0, w1, [x2]"),
        (0x1ac, 0xb8203041, "ldset w0, w1, [x2]"),
        (0x1b0, 0xb8204041, "ldsmax w0, w1, [x2]"),
        (0x1b4, 0xf8205041, "ldsmin x0, x1, [x2]"),
        (0x1b8, 0xb8206041, "ldumax w0, w1, [x2]"),
        (0x1bc, 0xf8207041, "ldumin x0, x1, [x2]"),
        (0x1c0, 0x38204041, "ldsmaxb w0, w1, [x2]"),
        (0x1c4, 0x78207041, "lduminh w0, w1, [x2]"),
        (0x1c8, 0xb820005f, "stadd w0, [x2]"),
        (0x1cc, 0xf820005f, "stadd x0, [x2]"),
        (0x1d0, 0x3820105f, "stclrb w0, [x2]"),
        (0x1d4, 0xb8208041, "swp w0, w1, [x2]"),
        (0x1d8, 0xf820805f, "swp x0, xzr, [x2]"),
        (0x1dc, 0x38208041, "swpb w0, w1, [x2]"),
        (0x1e0, 0x783f8041, "swph wzr, w1, [x2]"),
        (0x1e4, 0x88a07c41, "cas w0, w1, [x2]"),
        (0x1e8, 0xc8a07fe1, "cas x0, x1, [sp]"),
        (0x1ec, 0x08a07c41, "casb w0, w1, [x2]"),
        (0x1f0, 0x48a07c5f, "cash w0, wzr, [x2]"),
        (0x1f4, 0x08207c82, "casp w0, w1, w2, w3, [x4]"),
        (0x1f8, 0x48207c82, "casp x0, x1, x2, x3, [x4]"),
        (0x1fc, 0x483e7c82, "casp x30, xzr, x2, x3, [x4]"),
        (0x200, 0x48227c9e, "casp x2, x3, x30, xzr, [x4]"),
        (0x204, 0x887f0440, "ldxp w0, w1, [x2]"),
        (0x208, 0xc87f07e0, "ldxp x0, x1, [sp]"),
        (0x20c, 0x88200861, "stxp w0, w1, w2, [x3]"),
        (0x210, 0xc8200861, "stxp w0, x1, x2, [x3]"),
        (0x214, 0xc8207fff, "stxp w0, xzr, xzr, [sp]"),
        (0x218, 0x2d400440, "ldp s0, s1, [x2]"),
        (0x21c, 0x6d408440, "ldp d0, d1, [x2, #8]"),
        (0x220, 0xadc08440, "ldp q0, q1, [x2, #16]!"),
        (0x224, 0xacbf7c40, "stp q0, q31, [x2], #-32"),
        (0x228, 0x2d2007e0, "stp s0, s1, [sp, #-256]"),
        (0x22c, 0x6d1f805f, "stp d31, d0, [x2, #504]"),
        (0x230, 0xad1f8440, "stp q0, q1, [x2, #1008]"),
        (0x234, 0x3d400020, "ldr b0, [x1]"),
        (0x238, 0x7d400420, "ldr h0, [x1, #2]"),
        (0x23c, 0xbd400420, "ldr s0, [x1, #4]"),
        (0x240, 0xfd400420, "ldr d0, [x1, #8]"),
        (0x244, 0x3dc00420, "ldr q0, [x1, #16]"),
        (0x248, 0x3dfffc20, "ldr q0, [x1, #65520]"),
        (0x24c, 0x3c5ff020, "ldur b0, [x1, #-1]"),
        (0x250, 0x7c401020, "ldur h0, [x1, #1]"),
        (0x254, 0xbc402020, "ldur s0, [x1, #2]"),
        (0x258, 0xfc5f8020, "ldur d0, [x1, #-8]"),
        (0x25c, 0x3cdf0020, "ldur q0, [x1, #-16]"),
        (0x260, 0x3cc08020, "ldur q0, [x1, #8]"),
        (0x264, 0x3c9f0fe0, "str q0, [sp, #-16]!"),
        (0x268, 0xfc008420, "str d0, [x1], #8"),
        (0x26c, 0x3c001c20, "str b0, [x1, #1]!"),
        (0x270, 0x3ca27820, "str q0, [x1, x2, lsl #4]"),
        (0x274, 0x3ca26820, "str q0, [x1, x2]"),
        (0x278, 0xfc22d820, "str d0, [x1, w2, sxtw #3]"),
        (0x27c, 0xbc224820, "str s0, [x1, w2, uxtw]"),
        (0x280, 0x7c227820, "str h0, [x1, x2, lsl #1]"),
        (0x284, 0x3c226820, "str b0, [x1, x2]"),
        (0x288, 0x3c9f0020, "stur q0, [x1, #-16]"),
        (0x290, 0x4c40a000, "ld1 {v0.16b-v1.16b}, [x0]"),
        (0x294, 0x4cdfa000, "ld1 {v0.16b-v1.16b}, [x0], #32"),
        (0x298, 0x4cc1a000, "ld1 {v0.16b-v1.16b}, [x0], x1"),
        (0x29c, 0x4cc17400, "ld1 {v0.8h}, [x0], x1"),
        (0x2a0, 0x0cc17c00, "ld1 {v0.1d}, [x0], x1"),
        (0x2a4, 0x0cdf63e0, "ld1 {v0.8b-v2.8b}, [sp], #24"),
        (0x2a8, 0x4cdf2800, "ld1 {v0.4s-v3.4s}, [x0], #64"),
        (0x2ac, 0x4c40281f, "ld1 {v31.4s-v2.4s}, [x0]"),
        (0x2b0, 0x4cc32c1e, "ld1 {v30.2d-v1.2d}, [x0], x3"),
        (0x2b4, 0x0c40a800, "ld1 {v0.2s-v1.2s}, [x0]"),
        (0x2b8, 0x0cdf2400, "ld1 {v0.4h-v3.4h}, [x0], #32"),
        (0x2bc, 0x4c408800, "ld2 {v0.4s-v1.4s}, [x0]"),
        (0x2c0, 0x4cdf8000, "ld2 {v0.16b-v1.16b}, [x0], #32"),
        (0x2c4, 0x0cc28000, "ld2 {v0.8b-v1.8b}, [x0], x2"),
        (0x2c8, 0x4cdf4c00, "ld3 {v0.2d-v2.2d}, [x0], #48"),
        (0x2cc, 0x4c40441f, "ld3 {v31.8h-v1.8h}, [x0]"),
        (0x2d0, 0x4c400000, "ld4 {v0.16b-v3.16b}, [x0]"),
        (0x2d4, 0x0cc5001e, "ld4 {v30.8b-v1.8b}, [x0], x5"),
        (0x2d8, 0x0cdf07e0, "ld4 {v0.4h-v3.4h}, [sp], #32"),
        (0x2dc, 0x4c00a000, "st1 {v0.16b-v1.16b}, [x0]"),
        (0x2e0, 0x4c9f2800, "st1 {v0.4s-v3.4s}, [x0], #64"),
        (0x2e4, 0x0c812000, "st1 {v0.8b-v3.8b}, [x0], x1"),
        (0x2e8, 0x0c9f83e0, "st2 {v0.8b-v1.8b}, [sp], #16"),
        (0x2ec, 0x4c008c00, "st2 {v0.2d-v1.2d}, [x0]"),
        (0x2f0, 0x4c864800, "st3 {v0.4s-v2.4s}, [x0], x6"),
        (0x2f4, 0x4c9f001d, "st4 {v29.16b-v0.16b}, [x0], #64"),
        (0x2f8, 0x0d400000, "ld1 {v0.b}[0], [x0]"),
        (0x2fc, 0x4d401c00, "ld1 {v0.b}[15], [x0]"),
        (0x300, 0x0ddf1c00, "ld1 {v0.b}[7], [x0], #1"),
        (0x304, 0x4d405800, "ld1 {v0.h}[7], [x0]"),
        (0x308, 0x0ddf4000, "ld1 {v0.h}[0], [x0], #2"),
        (0x30c, 0x4dc29000, "ld1 {v0.s}[3], [x0], x2"),
        (0x310, 0x0ddf8000, "ld1 {v0.s}[0], [x0], #4"),
        (0x314, 0x4dc28400, "ld1 {v0.d}[1], [x0], x2"),
        (0x318, 0x0ddf841f, "ld1 {v31.d}[0], [x0], #8"),
        (0x31c, 0x0d600c00, "ld2 {v0.b-v1.b}[3], [x0]"),
        (0x320, 0x0dff5800, "ld2 {v0.h-v1.h}[3], [x0], #4"),
        (0x324, 0x0de49000, "ld2 {v0.s-v1.s}[1], [x0], x4"),
        (0x328, 0x4dff8400, "ld2 {v0.d-v1.d}[1], [x0], #16"),
        (0x32c, 0x0ddfb000, "ld3 {v0.s-v2.s}[1], [x0], #12"),
        (0x330, 0x4d40241f, "ld3 {v31.b-v1.b}[9], [x0]"),
        (0x334, 0x0d60a400, "ld4 {v0.d-v3.d}[0], [x0]"),
        (0x338, 0x0dff73e0, "ld4 {v0.h-v3.h}[2], [sp], #8"),
        (0x33c, 0x4d008000, "st1 {v0.s}[2], [x0]"),
        (0x340, 0x0d9f0400, "st1 {v0.b}[1], [x0], #1"),
        (0x344, 0x4da38400, "st2 {v0.d-v1.d}[1], [x0], x3"),
        (0x348, 0x4d9f6800, "st3 {v0.h-v2.h}[5], [x0], #6"),
        (0x34c, 0x4d20b000, "st4 {v0.s-v3.s}[3], [x0]"),
        (0x350, 0x4d40c000, "ld1r {v0.16b}, [x0]"),
        (0x354, 0x0ddfc000, "ld1r {v0.8b}, [x0], #1"),
        (0x358, 0x4ddfc800, "ld1r {v0.4s}, [x0], #4"),
        (0x35c, 0x4dc1cc00, "ld1r {v0.2d}, [x0], x1"),
        (0x360, 0x0ddfc400, "ld1r {v0.4h}, [x0], #2"),
        (0x364, 0x4d60c800, "ld2r {v0.4s-v1.4s}, [x0]"),
        (0x368, 0x4dffc400, "ld2r {v0.8h-v1.8h}, [x0], #4"),
        (0x36c, 0x0de3cc00, "ld2r {v0.1d-v1.1d}, [x0], x3"),
        (0x370, 0x0ddfe800, "ld3r {v0.2s-v2.2s}, [x0], #12"),
        (0x374, 0x4d40ec1f, "ld3r {v31.2d-v1.2d}, [x0]"),
        (0x378, 0x4d60ec00, "ld4r {v0.2d-v3.2d}, [x0]"),
        (0x37c, 0x4dffe3e0, "ld4r {v0.16b-v3.16b}, [sp], #4"),
        (0x380, 0x4de7e400, "ld4r {v0.8h-v3.8h}, [x0], x7"),
        (0x384, 0x98000040, "ldrsw x0, 0x38c"),
        (0x388, 0x98ffffc3, "ldrsw x3, 0x380"),
        (0x38c, 0x9800005f, "ldrsw xzr, 0x394"),
    ];

    #[test]
    fn matches_objdump() {
        for &(pc, word, expected) in ROWS {
            let instruction = decode(word).unwrap_or_else(|e| panic!("{word:#010x}: {e:?}"));
            assert_eq!(
                instruction.display_at(pc).to_string(),
                expected,
                "{word:#010x}"
            );
        }
    }

    /// `(word, text)`: GNU objdump 2.46's spelling of the words whose identity the decoded
    /// form must keep (acquire/release, unscaled and unprivileged offsets, non-temporal
    /// pairs, `ld1` arrangements, byte register offsets, `rprfm`).
    const IDENTITY_ROWS: &[(u32, &str)] = &[
        (0x88dffc20, "ldar w0, [x1]"),
        (0xc8dfffe0, "ldar x0, [sp]"),
        (0x08dffc20, "ldarb w0, [x1]"),
        (0x48dffc20, "ldarh w0, [x1]"),
        (0x88df7c20, "ldlar w0, [x1]"),
        (0xc8df7c20, "ldlar x0, [x1]"),
        (0x08df7c20, "ldlarb w0, [x1]"),
        (0x48df7c20, "ldlarh w0, [x1]"),
        (0x889f7c20, "stllr w0, [x1]"),
        (0xc89f7c20, "stllr x0, [x1]"),
        (0x089f7c20, "stllrb w0, [x1]"),
        (0x489f7c20, "stllrh w0, [x1]"),
        (0x889ffc20, "stlr w0, [x1]"),
        (0xc89fffe0, "stlr x0, [sp]"),
        (0x089ffc20, "stlrb w0, [x1]"),
        (0x489ffc20, "stlrh w0, [x1]"),
        (0x889ffc1f, "stlr wzr, [x0]"),
        (0xb8bfc020, "ldapr w0, [x1]"),
        (0xf8bfc020, "ldapr x0, [x1]"),
        (0x38bfc020, "ldaprb w0, [x1]"),
        (0x78bfc020, "ldaprh w0, [x1]"),
        (0x99404020, "ldapur w0, [x1, #4]"),
        (0xd95f8020, "ldapur x0, [x1, #-8]"),
        (0x19404020, "ldapurb w0, [x1, #4]"),
        (0x59400020, "ldapurh w0, [x1]"),
        (0x19c04020, "ldapursb w0, [x1, #4]"),
        (0x19804020, "ldapursb x0, [x1, #4]"),
        (0x59c04020, "ldapursh w0, [x1, #4]"),
        (0x59804020, "ldapursh x0, [x1, #4]"),
        (0x99804020, "ldapursw x0, [x1, #4]"),
        (0x99004020, "stlur w0, [x1, #4]"),
        (0xd9100020, "stlur x0, [x1, #-256]"),
        (0x19000020, "stlurb w0, [x1]"),
        (0x590ff020, "stlurh w0, [x1, #255]"),
        (0x885ffc20, "ldaxr w0, [x1]"),
        (0xc85fffe0, "ldaxr x0, [sp]"),
        (0x085ffc20, "ldaxrb w0, [x1]"),
        (0x485ffc20, "ldaxrh w0, [x1]"),
        (0x8802fc20, "stlxr w2, w0, [x1]"),
        (0xc802ffe0, "stlxr w2, x0, [sp]"),
        (0x0802fc20, "stlxrb w2, w0, [x1]"),
        (0x4802fc20, "stlxrh w2, w0, [x1]"),
        (0x887f8440, "ldaxp w0, w1, [x2]"),
        (0xc87f87e0, "ldaxp x0, x1, [sp]"),
        (0x88238440, "stlxp w3, w0, w1, [x2]"),
        (0xc8238440, "stlxp w3, x0, x1, [x2]"),
        (0x88e07c41, "casa w0, w1, [x2]"),
        (0x88a0fc41, "casl w0, w1, [x2]"),
        (0xc8e0fc41, "casal x0, x1, [x2]"),
        (0x08e07c41, "casab w0, w1, [x2]"),
        (0x48e0fc41, "casalh w0, w1, [x2]"),
        (0x08a0fc5f, "caslb w0, wzr, [x2]"),
        (0x08607c82, "caspa w0, w1, w2, w3, [x4]"),
        (0x4820fc82, "caspl x0, x1, x2, x3, [x4]"),
        (0x4860fc82, "caspal x0, x1, x2, x3, [x4]"),
        (0xb8a08041, "swpa w0, w1, [x2]"),
        (0xf8608041, "swpl x0, x1, [x2]"),
        (0xf8e0805f, "swpal x0, xzr, [x2]"),
        (0x38a08041, "swpab w0, w1, [x2]"),
        (0x78608041, "swplh w0, w1, [x2]"),
        (0x38e08041, "swpalb w0, w1, [x2]"),
        (0xb8a00041, "ldadda w0, w1, [x2]"),
        (0xb8600041, "ldaddl w0, w1, [x2]"),
        (0xf8e00041, "ldaddal x0, x1, [x2]"),
        (0xb8a0003f, "ldadda w0, wzr, [x1]"),
        (0xf8e0003f, "ldaddal x0, xzr, [x1]"),
        (0xb860003f, "staddl w0, [x1]"),
        (0x38a0003f, "ldaddab w0, wzr, [x1]"),
        (0x7860003f, "staddlh w0, [x1]"),
        (0x38e0003f, "ldaddalb w0, wzr, [x1]"),
        (0x3860003f, "staddlb w0, [x1]"),
        (0xb8a01041, "ldclra w0, w1, [x2]"),
        (0xb8602041, "ldeorl w0, w1, [x2]"),
        (0xb8a0305f, "ldseta w0, wzr, [x2]"),
        (0xf8e04041, "ldsmaxal x0, x1, [x2]"),
        (0xb8605041, "ldsminl w0, w1, [x2]"),
        (0x78e06041, "ldumaxalh w0, w1, [x2]"),
        (0x38a07041, "lduminab w0, w1, [x2]"),
        (0xb8400020, "ldur w0, [x1]"),
        (0xb8408020, "ldur w0, [x1, #8]"),
        (0xf8410020, "ldur x0, [x1, #16]"),
        (0xf85f8020, "ldur x0, [x1, #-8]"),
        (0x38408020, "ldurb w0, [x1, #8]"),
        (0x78404020, "ldurh w0, [x1, #4]"),
        (0x38808020, "ldursb x0, [x1, #8]"),
        (0x78c08020, "ldursh w0, [x1, #8]"),
        (0xb8808020, "ldursw x0, [x1, #8]"),
        (0xf8008020, "stur x0, [x1, #8]"),
        (0x38001020, "sturb w0, [x1, #1]"),
        (0xf9400420, "ldr x0, [x1, #8]"),
        (0xf9400020, "ldr x0, [x1]"),
        (0xb8404820, "ldtr w0, [x1, #4]"),
        (0xf8400820, "ldtr x0, [x1]"),
        (0x385ff820, "ldtrb w0, [x1, #-1]"),
        (0x78402820, "ldtrh w0, [x1, #2]"),
        (0x38c00820, "ldtrsb w0, [x1]"),
        (0x78900820, "ldtrsh x0, [x1, #-256]"),
        (0xb8804820, "ldtrsw x0, [x1, #4]"),
        (0xf8008820, "sttr x0, [x1, #8]"),
        (0x38000820, "sttrb w0, [x1]"),
        (0x780ff820, "sttrh w0, [x1, #255]"),
        (0xf9800020, "prfm pldl1keep, [x1]"),
        (0xf8800020, "prfum pldl1keep, [x1]"),
        (0xf8808020, "prfum pldl1keep, [x1, #8]"),
        (0xf9800420, "prfm pldl1keep, [x1, #8]"),
        (0xf89f8033, "prfum pstl2strm, [x1, #-8]"),
        (0xf8a26820, "prfm pldl1keep, [x1, x2]"),
        (0xf8a27820, "prfm pldl1keep, [x1, x2, lsl #3]"),
        (0xf8a24820, "prfm pldl1keep, [x1, w2, uxtw]"),
        (0xf8a2d820, "prfm pldl1keep, [x1, w2, sxtw #3]"),
        (0xf8a2e820, "prfm pldl1keep, [x1, x2, sxtx]"),
        (0xf8a24838, "rprfm pldkeep, x2, [x1]"),
        (0xf8a24839, "rprfm pstkeep, x2, [x1]"),
        (0xf8bf4bfc, "rprfm pldstrm, xzr, [sp]"),
        (0xf8a2483d, "rprfm pststrm, x2, [x1]"),
        (0xf8bf483c, "rprfm pldstrm, xzr, [x1]"),
        (0xf8bf4bfd, "rprfm pststrm, xzr, [sp]"),
        (0xf8a2483a, "rprfm #2, x2, [x1]"),
        (0x38627820, "ldrb w0, [x1, x2, lsl #0]"),
        (0x38626820, "ldrb w0, [x1, x2]"),
        (0x38625820, "ldrb w0, [x1, w2, uxtw #0]"),
        (0x3862d820, "ldrb w0, [x1, w2, sxtw #0]"),
        (0x3862c820, "ldrb w0, [x1, w2, sxtw]"),
        (0x3862f820, "ldrb w0, [x1, x2, sxtx #0]"),
        (0x38a27820, "ldrsb x0, [x1, x2, lsl #0]"),
        (0x38227820, "strb w0, [x1, x2, lsl #0]"),
        (0x3c207962, "str b2, [x11, x0, lsl #0]"),
        (0x3c206962, "str b2, [x11, x0]"),
        (0x3c605962, "ldr b2, [x11, w0, uxtw #0]"),
        (0x3ca07962, "str q2, [x11, x0, lsl #4]"),
        (0x3ca06962, "str q2, [x11, x0]"),
        (0x7c607962, "ldr h2, [x11, x0, lsl #1]"),
        (0xf8607962, "ldr x2, [x11, x0, lsl #3]"),
        (0xa8400440, "ldnp x0, x1, [x2]"),
        (0x28600440, "ldnp w0, w1, [x2, #-256]"),
        (0xa8008440, "stnp x0, x1, [x2, #8]"),
        (0x28010440, "stnp w0, w1, [x2, #8]"),
        (0x2c400440, "ldnp s0, s1, [x2]"),
        (0x6c408440, "ldnp d0, d1, [x2, #8]"),
        (0xac408440, "ldnp q0, q1, [x2, #16]"),
        (0x2c000440, "stnp s0, s1, [x2]"),
        (0x6c3f8440, "stnp d0, d1, [x2, #-8]"),
        (0xac3f0440, "stnp q0, q1, [x2, #-32]"),
        (0x4c407020, "ld1 {v0.16b}, [x1]"),
        (0x0c407020, "ld1 {v0.8b}, [x1]"),
        (0x4c407420, "ld1 {v0.8h}, [x1]"),
        (0x0c407420, "ld1 {v0.4h}, [x1]"),
        (0x4c407820, "ld1 {v0.4s}, [x1]"),
        (0x0c407820, "ld1 {v0.2s}, [x1]"),
        (0x4c407c20, "ld1 {v0.2d}, [x1]"),
        (0x0c407c20, "ld1 {v0.1d}, [x1]"),
        (0x4c4073ff, "ld1 {v31.16b}, [sp]"),
        (0x4cdf7020, "ld1 {v0.16b}, [x1], #16"),
        (0x0cdf7020, "ld1 {v0.8b}, [x1], #8"),
        (0x0cdf7c20, "ld1 {v0.1d}, [x1], #8"),
        (0x4cdf7820, "ld1 {v0.4s}, [x1], #16"),
        (0x0cc27420, "ld1 {v0.4h}, [x1], x2"),
        (0x4cc27020, "ld1 {v0.16b}, [x1], x2"),
        (0x4c007020, "st1 {v0.16b}, [x1]"),
        (0x4c007420, "st1 {v0.8h}, [x1]"),
        (0x0c9f7820, "st1 {v0.2s}, [x1], #8"),
        (0x4c9f73ff, "st1 {v31.16b}, [sp], #16"),
        (0x0c007c20, "st1 {v0.1d}, [x1]"),
        (0x3dc00020, "ldr q0, [x1]"),
        (0xfd400020, "ldr d0, [x1]"),
        (0x3cc10020, "ldur q0, [x1, #16]"),
        (0x3cc08020, "ldur q0, [x1, #8]"),
        (0xfc5f8020, "ldur d0, [x1, #-8]"),
        (0xfc400020, "ldur d0, [x1]"),
        (0xbc404020, "ldur s0, [x1, #4]"),
        (0x7c402020, "ldur h0, [x1, #2]"),
        (0x3c400020, "ldur b0, [x1]"),
        (0x3c401020, "ldur b0, [x1, #1]"),
        (0x3d400420, "ldr b0, [x1, #1]"),
        (0x3c810020, "stur q0, [x1, #16]"),
        (0xfc1f8020, "stur d0, [x1, #-8]"),
        (0xfd400420, "ldr d0, [x1, #8]"),
    ];

    #[test]
    fn keeps_instruction_identity() {
        for &(word, expected) in IDENTITY_ROWS {
            let instruction = decode(word).unwrap_or_else(|e| panic!("{word:#010x}: {e:?}"));
            assert_eq!(instruction.display().to_string(), expected, "{word:#010x}");
        }
    }

    #[test]
    fn without_pc_targets_are_relative() {
        assert_eq!(
            decode(0x18000040).unwrap().display().to_string(),
            "ldr w0, .+8"
        );
        assert_eq!(
            decode(0xd8ffffd5).unwrap().display().to_string(),
            "prfm pstl3strm, .-8"
        );
    }
}
