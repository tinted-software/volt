//! AArch64 (A64) disassembler: the inverse of [`super::encode`].
//!
//! It decodes exactly the instructions the encoder emits into a readable
//! listing, so codegen debugging does not mean hand-decoding hex. Field
//! positions mirror the encoder bit-for-bit. Anything outside the emitted
//! subset prints as `.word 0x<hex>` so nothing is silently mis-decoded.
//!
//! [`one`] renders a single 32-bit instruction word; [`format`] renders a
//! whole code buffer with byte addresses.
//!
//! This module is intentionally self-contained (it operates on raw words and
//! shares no types with the encoder) so it keeps compiling while sibling
//! modules evolve.

use alloc::string::{String, ToString as _};
use core::fmt::Write as _;

/// A source-line-table row: the byte offset where a source line's code begins.
pub struct SourceLine {
    pub offset: u32,
    pub line: u32,
}

/// A function's location in a linked image: its name and its word offset.
pub struct Sym<'a> {
    pub name: &'a str,
    pub word: usize,
}

/// A defined symbol by absolute address (a function or data object), for
/// resolving branch and `adrp` targets when disassembling an ELF image.
pub struct AddrSym<'a> {
    pub name: &'a str,
    pub addr: u64,
    pub size: u64,
    pub is_func: bool,
}

/// A decoded source-line row keyed by absolute address (from a `.debug_line`
/// program), for objdump `-S`-style annotation of a real ELF.
pub struct AddrLine {
    pub addr: u64,
    pub line: u32,
}

/// The context that makes a listing absolute: the current instruction's
/// address and the image's symbols. When present, PC-relative targets render
/// as `0x<addr> <sym+off>` (objdump style).
#[derive(Clone, Copy)]
struct Context<'a> {
    pc: u64,
    syms: &'a [AddrSym<'a>],
}

/// Render one instruction word as `mnemonic operands`.
pub fn one(word: u32) -> String {
    let mut out = String::new();
    decode(&mut out, word, None);
    out
}

/// Render a whole instruction stream (one word per line, `addr: hex  text`).
pub fn format(code: &[u32]) -> String {
    let mut out = String::new();
    for (i, &word) in code.iter().enumerate() {
        let _ = write!(out, "{:04x}: {:08x}  ", i * 4, word);
        decode(&mut out, word, None);
        out.push('\n');
    }
    out
}

/// Render a listing with source-line markers interleaved (objdump `-S`
/// style): a `; line N` header precedes the instructions belonging to that
/// source line. `lines` must be sorted by offset (as isel produces).
pub fn format_with_lines(code: &[u32], lines: &[SourceLine]) -> String {
    let mut out = String::new();
    let mut li: usize = 0;
    for (i, &word) in code.iter().enumerate() {
        let byte_off: u32 = (i * 4) as u32;
        while li < lines.len() && lines[li].offset == byte_off {
            let _ = writeln!(out, "; line {}", lines[li].line);
            li += 1;
        }
        let _ = write!(out, "{byte_off:04x}: {word:08x}  ");
        decode(&mut out, word, None);
        out.push('\n');
    }
    out
}

/// Render a whole linked module: each function gets a `name:` label at its
/// offset, and every resolved `bl` is annotated with the callee's name
/// (`<helper>`), so a linked image reads like a symbolized listing.
pub fn format_module(code: &[u32], syms: &[Sym<'_>]) -> String {
    let mut out = String::new();
    for (i, &word) in code.iter().enumerate() {
        for s in syms {
            if s.word == i {
                if i != 0 {
                    out.push('\n');
                }
                let _ = writeln!(out, "{}:", s.name);
            }
        }
        let _ = write!(out, "{:04x}: {:08x}  ", i * 4, word);
        decode(&mut out, word, None);
        // Annotate any PC-relative branch (call or local) with the function
        // it lands in, as `<sym>` or `<sym+0xN>`, the way objdump does.
        if let Some(target) = branch_target_word(word, i)
            && let Some(s) = containing_sym(syms, target)
        {
            let off = target - s.word as i64;
            if off == 0 {
                let _ = write!(out, "  <{}>", s.name);
            } else {
                let _ = write!(out, "  <{}+0x{:x}>", s.name, off as u64 * 4);
            }
        }
        out.push('\n');
    }
    out
}

/// Render an ELF code section as an objdump-style listing: absolute
/// addresses, function labels, and branch/`adrp` targets resolved to
/// `0x<addr> <sym+off>`. `base` is the section's load address; `syms` are all
/// defined symbols (functions and data).
pub fn format_elf(code: &[u32], base: u64, syms: &[AddrSym<'_>]) -> String {
    format_elf_impl(code, base, syms, &[])
}

/// Like [`format_elf`], but interleaves `; line N` markers from `lines`
/// (sorted by address) before the instruction at each address, so a listing
/// of a `-g` object reads with its source lines.
pub fn format_elf_with_lines(
    code: &[u32],
    base: u64,
    syms: &[AddrSym<'_>],
    lines: &[AddrLine],
) -> String {
    format_elf_impl(code, base, syms, lines)
}

fn format_elf_impl(code: &[u32], base: u64, syms: &[AddrSym<'_>], lines: &[AddrLine]) -> String {
    let mut out = String::new();
    let mut li: usize = 0;
    for (i, &word) in code.iter().enumerate() {
        let addr = base + i as u64 * 4;
        for s in syms {
            if s.is_func && s.addr == addr {
                if i != 0 {
                    out.push('\n');
                }
                let _ = writeln!(out, "{addr:016x} <{}>:", s.name);
            }
        }
        while li < lines.len() && lines[li].addr < addr {
            li += 1; // skip any before this insn
        }
        while li < lines.len() && lines[li].addr == addr {
            let _ = writeln!(out, "; line {}", lines[li].line);
            li += 1;
        }
        let _ = write!(out, "{addr:08x}: {word:08x}  ");
        decode(&mut out, word, Some(Context { pc: addr, syms }));
        out.push('\n');
    }
    out
}

/// Render a PC-relative code target `rel_bytes` from the current PC:
/// absolute `0x<addr> <sym+off>` when a context is present, else the
/// self-contained `.±N` form.
fn render_target(out: &mut String, ctx: Option<Context<'_>>, rel_bytes: i64) {
    if let Some(c) = ctx {
        // Wrapping add (like the adrp path): a disassembler must never panic
        // on the bytes it is given, including a branch whose displacement
        // runs below zero.
        let abs: u64 = (c.pc as i64).wrapping_add(rel_bytes) as u64;
        let _ = write!(out, "0x{abs:x}");
        annotate_sym(out, c.syms, abs);
    } else {
        out.push('.');
        out.push(if rel_bytes < 0 { '-' } else { '+' });
        out.push_str(&rel_bytes.unsigned_abs().to_string());
    }
}

/// Append ` <sym+off>` for the symbol that contains `addr`, if any.
fn annotate_sym(out: &mut String, syms: &[AddrSym<'_>], addr: u64) {
    if let Some((name, off)) = symbol_at(syms, addr) {
        if off == 0 {
            let _ = write!(out, " <{name}>");
        } else {
            let _ = write!(out, " <{name}+0x{off:x}>");
        }
    }
}

/// The symbol containing `addr` (the greatest-addressed symbol at or before
/// it): its name and the byte offset into it.
fn symbol_at<'a>(syms: &'a [AddrSym<'a>], addr: u64) -> Option<(&'a str, u64)> {
    let mut best: Option<&AddrSym<'_>> = None;
    for s in syms {
        if s.addr <= addr && best.is_none_or(|b| s.addr > b.addr) {
            best = Some(s);
        }
    }
    best.map(|s| (s.name, addr - s.addr))
}

/// The target word index of a PC-relative branch (b/bl/b.cond/cbz/cbnz/
/// tbz/tbnz) at `index`, or `None` if `w` is not such a branch.
fn branch_target_word(w: u32, index: usize) -> Option<i64> {
    let base: i64 = index as i64;
    if w & 0x7C00_0000 == 0x1400_0000 {
        // b / bl: imm26
        return Some(base + (((w << 6) as i32) >> 6) as i64);
    }
    if w & 0xFF00_0010 == 0x5400_0000 || w & 0x7E00_0000 == 0x3400_0000 {
        // b.cond / cbz / cbnz: imm19
        return Some(base + (((w << 8) as i32) >> 13) as i64);
    }
    if w & 0x7E00_0000 == 0x3600_0000 {
        // tbz / tbnz: imm14
        return Some(base + (((w << 13) as i32) >> 18) as i64);
    }
    None
}

/// The symbol whose range contains word `target` (the greatest symbol at or
/// before it), or `None`.
fn containing_sym<'a>(syms: &'a [Sym<'a>], target: i64) -> Option<&'a Sym<'a>> {
    if target < 0 {
        return None;
    }
    let mut best: Option<&'a Sym<'a>> = None;
    for s in syms {
        if s.word as i64 <= target && best.is_none_or(|b| s.word > b.word) {
            best = Some(s);
        }
    }
    best
}

fn rd(w: u32) -> u32 {
    w & 0x1F
}
fn rn(w: u32) -> u32 {
    (w >> 5) & 0x1F
}
fn rm(w: u32) -> u32 {
    (w >> 16) & 0x1F
}
fn ra(w: u32) -> u32 {
    (w >> 10) & 0x1F
}
fn sf(w: u32) -> bool {
    (w >> 31) & 1 != 0
}

/// A 32/64-bit general register (`w`/`x`), rendering 31 as the zero register.
fn gp(out: &mut String, x64: bool, r: u32) {
    let c: char = if x64 { 'x' } else { 'w' };
    if r == 31 {
        let _ = write!(out, "{c}zr");
    } else {
        let _ = write!(out, "{c}{r}");
    }
}

/// A general register in an address/stack context, rendering 31 as `sp`.
fn sp(out: &mut String, x64: bool, r: u32) {
    if r == 31 {
        out.push_str("sp");
    } else {
        gp(out, x64, r);
    }
}

fn cond_name(v: u32) -> &'static str {
    // binutils spelling (cs/cc rather than the hs/lo synonyms) so listings
    // match objdump.
    match v {
        0 => "eq",
        1 => "ne",
        2 => "cs",
        3 => "cc",
        4 => "mi",
        5 => "pl",
        6 => "vs",
        7 => "vc",
        8 => "hi",
        9 => "ls",
        10 => "ge",
        11 => "lt",
        12 => "gt",
        13 => "le",
        14 => "al",
        15 => "nv",
        _ => "??",
    }
}

/// Masks for the three-register data-processing form: clear rd, rn, rm; keep
/// the opcode and the op2/shift field in bits[10:15]. The `sf` bit is dropped
/// so one entry covers both the w and x forms.
const RRR_MASK: u32 = 0x7FE0_FC00;

/// `RRR_MASK` with the imm6 shift-amount field (bits[15:10]) dropped, for the
/// add/sub shifted-register form where that field is a real operand and not
/// part of the opcode.
const RRR_NO_IMM6_MASK: u32 = RRR_MASK & !0xFC00u32;

/// The 32-bit-base opcodes of the shared three-register form (sf dropped).
const RRR_TABLE: &[(u32, &str)] = &[
    (0x0B00_0000, "add"),
    (0x4B00_0000, "sub"),
    (0x0A00_0000, "and"),
    (0x6A00_0000, "ands"), // flag-setting and (shift 0)
    (0x2A00_0000, "orr"),
    (0x4A00_0000, "eor"),
    (0x1AC0_0C00, "sdiv"),
    (0x1AC0_0800, "udiv"),
    (0x1AC0_2000, "lsl"),
    (0x1AC0_2400, "lsr"),
    (0x1AC0_2800, "asr"),
];

/// `ret` (`ret x30`): the exact operandless encoding, matched first.
const RET_WORD: u32 = 0xD65F_0000 | (30 << 5);

fn decode(out: &mut String, w: u32, ctx: Option<Context<'_>>) {
    // Exact, operandless / special encodings first.
    if w == RET_WORD {
        out.push_str("ret");
        return;
    }
    if w & 0xFFE0_001F == 0xD400_0001 {
        let _ = write!(out, "svc #{}", (w >> 5) & 0xFFFF);
        return;
    }
    if w & 0xFFFF_FC1F == 0xD63F_0000 {
        out.push_str("blr ");
        sp(out, true, rn(w));
        return;
    }

    // `mov` (register) is `orr Xd, xzr, Xm` with rn == 31; render it as a move.
    if w & 0x7FE0_FFE0 == 0x2A00_03E0 {
        out.push_str("mov ");
        gp(out, sf(w), rd(w));
        out.push_str(", ");
        gp(out, sf(w), rm(w));
        return;
    }

    // neg Rd, Rm = sub Rd, RZR, Rm (rn == 31), the objdump alias.
    if w & 0x7FE0_FC00 == 0x4B00_0000 && rn(w) == 31 {
        out.push_str("neg ");
        gp(out, sf(w), rd(w));
        out.push_str(", ");
        gp(out, sf(w), rm(w));
        return;
    }

    // The shared three-register form.
    for &(m, mnem) in RRR_TABLE {
        if w & RRR_MASK == m {
            out.push_str(mnem);
            out.push(' ');
            gp(out, sf(w), rd(w));
            out.push_str(", ");
            gp(out, sf(w), rn(w));
            out.push_str(", ");
            gp(out, sf(w), rm(w));
            return;
        }
    }

    // add/sub, shifted-register form with a nonzero LSL shift amount (imm6 at
    // bits[15:10]). The shift-0 case is already handled by the table above.
    if w & RRR_NO_IMM6_MASK == 0x0B00_0000 || w & RRR_NO_IMM6_MASK == 0x4B00_0000 {
        let is_sub = w & RRR_NO_IMM6_MASK == 0x4B00_0000;
        out.push_str(if is_sub { "sub " } else { "add " });
        gp(out, sf(w), rd(w));
        out.push_str(", ");
        gp(out, sf(w), rn(w));
        out.push_str(", ");
        gp(out, sf(w), rm(w));
        let _ = write!(out, ", lsl #{}", (w >> 10) & 0x3F);
        return;
    }

    // madd/msub (mul is madd with ra == 31).
    if w & 0x7FE0_8000 == 0x1B00_0000 || w & 0x7FE0_8000 == 0x1B00_8000 {
        let is_sub = (w >> 15) & 1 != 0;
        if !is_sub && ra(w) == 31 {
            out.push_str("mul ");
            gp(out, sf(w), rd(w));
            out.push_str(", ");
            gp(out, sf(w), rn(w));
            out.push_str(", ");
            gp(out, sf(w), rm(w));
            return;
        }
        if is_sub && ra(w) == 31 {
            // mneg Rd, Rn, Rm = msub Rd, Rn, Rm, RZR
            out.push_str("mneg ");
            gp(out, sf(w), rd(w));
            out.push_str(", ");
            gp(out, sf(w), rn(w));
            out.push_str(", ");
            gp(out, sf(w), rm(w));
            return;
        }
        out.push_str(if is_sub { "msub " } else { "madd " });
        gp(out, sf(w), rd(w));
        out.push_str(", ");
        gp(out, sf(w), rn(w));
        out.push_str(", ");
        gp(out, sf(w), rm(w));
        out.push_str(", ");
        gp(out, sf(w), ra(w));
        return;
    }

    // add/sub immediate (bit30 = op is sub, bit29 = S sets flags, bit22 =
    // shift-by-12). When S is set and rd == 31 the alias is cmp / cmn. Rn/Rd
    // 31 otherwise mean sp.
    if w & 0x1F80_0000 == 0x1100_0000 {
        let is_sub = (w >> 30) & 1 != 0;
        let setflags = (w >> 29) & 1 != 0;
        let shift12 = (w >> 22) & 1 != 0;
        let imm: u32 = (w >> 10) & 0xFFF;
        if setflags && rd(w) == 31 {
            out.push_str(if is_sub { "cmp " } else { "cmn " });
            sp(out, sf(w), rn(w));
            let _ = write!(out, ", #{imm}");
            if shift12 {
                out.push_str(", lsl #12");
            }
            return;
        }
        // mov Rd, Rn (to/from SP) = add Rd, Rn, #0 with no shift.
        if !setflags && !is_sub && imm == 0 && !shift12 && (rd(w) == 31 || rn(w) == 31) {
            out.push_str("mov ");
            sp(out, sf(w), rd(w));
            out.push_str(", ");
            sp(out, sf(w), rn(w));
            return;
        }
        let mnem = if setflags {
            if is_sub { "subs" } else { "adds" }
        } else if is_sub {
            "sub"
        } else {
            "add"
        };
        out.push_str(mnem);
        out.push(' ');
        sp(out, sf(w), rd(w));
        out.push_str(", ");
        sp(out, sf(w), rn(w));
        let _ = write!(out, ", #{imm}");
        if shift12 {
            out.push_str(", lsl #12");
        }
        return;
    }

    // movn (move wide negated): rendered as `mov Rd, #<~imm>`, matching
    // objdump's alias.
    if w & 0x7F80_0000 == 0x1280_0000 {
        let imm: u64 = ((w >> 5) & 0xFFFF) as u64;
        let shift: u32 = ((w >> 21) & 3) * 16;
        let val = !(imm << shift);
        out.push_str("mov ");
        gp(out, sf(w), rd(w));
        if sf(w) {
            let _ = write!(out, ", #0x{val:x}");
        } else {
            let _ = write!(out, ", #0x{:x}", val & 0xFFFF_FFFF);
        }
        return;
    }

    // movz / movk (32/64-bit, with an optional lsl of the 16-bit immediate).
    // objdump aliases movz to `mov Rd, #<imm<<shift>>`, so render the
    // assembled value for movz.
    if w & 0x7F80_0000 == 0x5280_0000 || w & 0x7F80_0000 == 0x7280_0000 {
        let keep = (w >> 29) & 1 != 0; // opc 11 -> movk, 10 -> movz
        let imm: u64 = ((w >> 5) & 0xFFFF) as u64;
        let shift: u32 = ((w >> 21) & 3) * 16;
        if !keep {
            // movz -> mov Rd, #value
            let val = imm << shift;
            out.push_str("mov ");
            gp(out, sf(w), rd(w));
            if sf(w) {
                let _ = write!(out, ", #0x{val:x}");
            } else {
                let _ = write!(out, ", #0x{:x}", val & 0xFFFF_FFFF);
            }
            return;
        }
        out.push_str("movk ");
        gp(out, sf(w), rd(w));
        let _ = write!(out, ", #{imm}");
        if shift != 0 {
            let _ = write!(out, ", lsl #{shift}");
        }
        return;
    }

    // adds/subs (shifted register, shift 0), with cmn/cmp aliases when
    // rd == 31.
    if w & 0x7FE0_FC00 == 0x2B00_0000 || w & 0x7FE0_FC00 == 0x6B00_0000 {
        let is_sub = (w >> 30) & 1 != 0;
        if rd(w) == 31 {
            out.push_str(if is_sub { "cmp " } else { "cmn " });
            gp(out, sf(w), rn(w));
            out.push_str(", ");
            gp(out, sf(w), rm(w));
            return;
        }
        if is_sub && rn(w) == 31 {
            // negs Rd, Rm = subs Rd, RZR, Rm
            out.push_str("negs ");
            gp(out, sf(w), rd(w));
            out.push_str(", ");
            gp(out, sf(w), rm(w));
            return;
        }
        out.push_str(if is_sub { "subs " } else { "adds " });
        gp(out, sf(w), rd(w));
        out.push_str(", ");
        gp(out, sf(w), rn(w));
        out.push_str(", ");
        gp(out, sf(w), rm(w));
        return;
    }

    // cset (csinc Rd, zr, zr, invert(cond)). The mask excludes the cond field
    // [12:15].
    if w & 0x7FFF_0FE0 == 0x1A9F_07E0 {
        out.push_str("cset ");
        gp(out, sf(w), rd(w));
        let _ = write!(out, ", {}", cond_name(((w >> 12) & 0xF) ^ 1));
        return;
    }

    // csel Rd, Rn, Rm, cond.
    if w & 0x7FE0_0C00 == 0x1A80_0000 {
        out.push_str("csel ");
        gp(out, sf(w), rd(w));
        out.push_str(", ");
        gp(out, sf(w), rn(w));
        out.push_str(", ");
        gp(out, sf(w), rm(w));
        let _ = write!(out, ", {}", cond_name((w >> 12) & 0xF));
        return;
    }

    // cbz / cbnz Rt, label (bit24 selects nz).
    if w & 0x7E00_0000 == 0x3400_0000 {
        let nz = (w >> 24) & 1 != 0;
        let imm19: i32 = ((w << 8) as i32) >> 13; // sign-extend bits[5:23]
        out.push_str(if nz { "cbnz " } else { "cbz " });
        gp(out, sf(w), rd(w));
        out.push_str(", ");
        render_target(out, ctx, imm19 as i64 * 4);
        return;
    }

    // tbz / tbnz Rt, #bit, label (bit24 selects nz). Test-bit index is b5:b40.
    if w & 0x7E00_0000 == 0x3600_0000 {
        let nz = (w >> 24) & 1 != 0;
        let bit: u32 = (((w >> 31) & 1) << 5) | ((w >> 19) & 0x1F);
        let imm14: i32 = (((w << 13) as i32) >> 18) * 4; // bits[18:5], scaled
        out.push_str(if nz { "tbnz " } else { "tbz " });
        gp(out, (w >> 31) & 1 != 0, rd(w));
        let _ = write!(out, ", #{bit}, ");
        render_target(out, ctx, imm14 as i64);
        return;
    }

    // b.cond label (bit4 == 0 distinguishes it from the
    // consistent-conditional-branch forms).
    if w & 0xFF00_0010 == 0x5400_0000 {
        let imm19: i32 = (((w << 8) as i32) >> 13) * 4;
        let _ = write!(out, "b.{} ", cond_name(w & 0xF));
        render_target(out, ctx, imm19 as i64);
        return;
    }

    // b / bl label (imm26 << 2).
    if w & 0xFC00_0000 == 0x1400_0000 || w & 0xFC00_0000 == 0x9400_0000 {
        let link = (w >> 31) & 1 != 0;
        let imm26: i32 = (((w << 6) as i32) >> 6) << 2;
        out.push_str(if link { "bl " } else { "b " });
        render_target(out, ctx, imm26 as i64);
        return;
    }

    // stp/ldp pair, pre/post-index (64-bit).
    if w & 0xFFC0_0000 == 0xA980_0000 || w & 0xFFC0_0000 == 0xA8C0_0000 {
        let load = (w >> 22) & 1 != 0;
        let imm7: i32 = (((w << 10) as i32) >> 25) * 8; // bits[15:21], scaled by 8
        out.push_str(if load { "ldp " } else { "stp " });
        gp(out, true, rd(w));
        out.push_str(", ");
        gp(out, true, ra(w)); // rt2 is bits[10:15]
        out.push_str(", [");
        sp(out, true, rn(w));
        if load {
            // post-index: `[base], #imm`
            let _ = write!(out, "], #{imm7}");
        } else {
            // pre-index: `[base, #imm]!`
            let _ = write!(out, ", #{imm7}]!");
        }
        return;
    }

    // Logical immediate: and/orr/eor/ands, with mov (orr from zr) and tst
    // (ands to zr) aliases.
    if w & 0x1F80_0000 == 0x1200_0000 {
        let opc = (w >> 29) & 3;
        let nn: u32 = (w >> 22) & 1;
        let immr: u32 = (w >> 16) & 0x3F;
        let imms: u32 = (w >> 10) & 0x3F;
        if let Some(val) = decode_bitmask(sf(w), nn, imms, immr) {
            if opc == 1 && rn(w) == 31 {
                // mov Rd, #imm
                out.push_str("mov ");
                gp(out, sf(w), rd(w));
                let _ = write!(out, ", #0x{val:x}");
                return;
            }
            if opc == 3 && rd(w) == 31 {
                // tst Rn, #imm
                out.push_str("tst ");
                gp(out, sf(w), rn(w));
                let _ = write!(out, ", #0x{val:x}");
                return;
            }
            let mnem = match opc {
                0 => "and",
                1 => "orr",
                2 => "eor",
                _ => "ands",
            };
            out.push_str(mnem);
            out.push(' ');
            sp(out, sf(w), rd(w)); // and/orr/eor immediate can target sp
            out.push_str(", ");
            gp(out, sf(w), rn(w));
            let _ = write!(out, ", #0x{val:x}");
            return;
        }
    }

    // extr Rd, Rn, Rm, #lsb (ror when rn == rm). bit23 == 1 separates it from
    // the bitfield ops.
    if w & 0x7F80_0000 == 0x1380_0000 {
        let lsb: u32 = (w >> 10) & 0x3F;
        if rn(w) == rm(w) {
            out.push_str("ror ");
            gp(out, sf(w), rd(w));
            out.push_str(", ");
            gp(out, sf(w), rn(w));
            let _ = write!(out, ", #{lsb}");
        } else {
            out.push_str("extr ");
            gp(out, sf(w), rd(w));
            out.push_str(", ");
            gp(out, sf(w), rn(w));
            out.push_str(", ");
            gp(out, sf(w), rm(w));
            let _ = write!(out, ", #{lsb}");
        }
        return;
    }

    // STUR/LDUR: unscaled signed 9-bit offset (integer). size 00=b 01=h 10=w
    // 11=x; opc picks store / load / sign-extending load.
    if w & 0x3F20_0C00 == 0x3800_0000 {
        let size = w >> 30;
        let opc = (w >> 22) & 3;
        let imm9: i32 = ((w << 11) as i32) >> 23; // bits[20:12], sign-extended
        let load = opc != 0;
        let signed = opc >= 2;
        let to_x = size == 3 || opc == 2; // x-sized access, or sign-extend to 64-bit
        let suffix: &str = match size {
            0 => "b",
            1 => "h",
            2 if signed => "w",
            _ => "",
        };
        out.push_str(if load { "ldur" } else { "stur" });
        if signed {
            out.push('s');
        }
        out.push_str(suffix);
        out.push(' ');
        gp(out, to_x, rd(w));
        out.push_str(", [");
        sp(out, true, rn(w));
        if imm9 != 0 {
            let _ = write!(out, ", #{imm9}]");
        } else {
            out.push(']');
        }
        return;
    }

    // STUR/LDUR (FP/SIMD), unscaled signed 9-bit offset. Width is b/h/s/d by
    // size, or q (128-bit) when opc bit1 is set.
    if w & 0x3F20_0C00 == 0x3C00_0000 {
        let size = w >> 30;
        let opc = (w >> 22) & 3;
        let imm9: i32 = ((w << 11) as i32) >> 23;
        let load = opc & 1 != 0;
        let c: char = if opc & 2 != 0 {
            'q'
        } else {
            match size {
                0 => 'b',
                1 => 'h',
                2 => 's',
                _ => 'd',
            }
        };
        out.push_str(if load { "ldur " } else { "stur " });
        let _ = write!(out, "{c}{}, [", rd(w));
        sp(out, true, rn(w));
        if imm9 != 0 {
            let _ = write!(out, ", #{imm9}]");
        } else {
            out.push(']');
        }
        return;
    }

    // umulh / smulh (unsigned/signed 64x64 -> high 64). Ra field is 31.
    if w & 0xFFE0_FC00 == 0x9BC0_7C00 || w & 0xFFE0_FC00 == 0x9B40_7C00 {
        let uns = (w >> 23) & 1 != 0;
        let _ = write!(
            out,
            "{} x{}, x{}, x{}",
            if uns { "umulh" } else { "smulh" },
            rd(w),
            rn(w),
            rm(w)
        );
        return;
    }

    // Bitfield move (SBFM/UBFM/BFM) with the usual aliases.
    if w & 0x1F80_0000 == 0x1300_0000 && (w >> 29) & 3 != 3 {
        bitfield(out, w);
        return;
    }

    // stp/ldp signed offset (bit31 selects 32/64-bit, bit22 selects load).
    if w & 0x7FC0_0000 == 0x2900_0000 || w & 0x7FC0_0000 == 0x2940_0000 {
        let x64 = (w >> 31) & 1 != 0;
        let load = (w >> 22) & 1 != 0;
        let scale: u32 = if x64 { 3 } else { 2 };
        let imm7: i32 = (((w << 10) as i32) >> 25) << scale; // bits[21:15], scaled
        out.push_str(if load { "ldp " } else { "stp " });
        gp(out, x64, rd(w));
        out.push_str(", ");
        gp(out, x64, ra(w)); // rt2 is bits[14:10]
        out.push_str(", [");
        sp(out, true, rn(w));
        if imm7 != 0 {
            let _ = write!(out, ", #{imm7}]");
        } else {
            out.push(']');
        }
        return;
    }

    // adr / adrp Rd, imm (PC-relative address; adrp is page-scaled).
    if w & 0x1F00_0000 == 0x1000_0000 {
        let page = (w >> 31) & 1 != 0;
        let immlo: i64 = ((w >> 29) & 3) as i64;
        let immhi: i64 = ((((w & 0x00FF_FFE0) << 8) as i32) >> 13) as i64; // bits[23:5], sign-extended
        let mut imm: i64 = (immhi << 2) | immlo;
        if page {
            imm <<= 12;
        }
        out.push_str(if page { "adrp " } else { "adr " });
        gp(out, true, rd(w));
        if let Some(c) = ctx {
            // absolute target: adrp is page-based, adr is byte-based
            let pc_base: u64 = if page { c.pc & !0xFFF } else { c.pc };
            let abs: u64 = pc_base.wrapping_add(imm as u64);
            let _ = write!(out, ", 0x{abs:x}");
            annotate_sym(out, c.syms, abs);
        } else if imm < 0 {
            let _ = write!(out, ", .-0x{:x}", (-imm) as u64);
        } else {
            let _ = write!(out, ", .+0x{imm:x}");
        }
        return;
    }

    if ordered_mem(out, w) {
        return;
    }
    if mem_imm(out, w) {
        return;
    }
    if fp_decode(out, w) {
        return;
    }
    if vec_decode(out, w) {
        return;
    }

    // Unknown: emit the raw word so nothing is silently lost.
    let _ = write!(out, ".word 0x{w:08x}");
}

/// Decode a logical-immediate `(N:immr:imms)` to its 32/64-bit value, or
/// `None` if the encoding is reserved. This is ARM's `DecodeBitMasks`
/// (immediate result only): a run of `s+1` ones, rotated right by `r` within
/// an `esize`-bit element, replicated across the register.
fn decode_bitmask(x64: bool, n: u32, imms: u32, immr: u32) -> Option<u64> {
    let width: u32 = if x64 { 64 } else { 32 };
    // len = position of the highest set bit of (N : NOT(imms)), a 7-bit quantity.
    let combined: u32 = ((n & 1) << 6) | ((!imms) & 0x3F);
    if combined == 0 {
        return None;
    }
    let mut len: u32 = 6;
    while len > 0 && (combined & (1 << len)) == 0 {
        len -= 1;
    }
    if len == 0 {
        return None; // element size 1 is not a valid logical immediate
    }
    let esize: u32 = 1 << len;
    if esize > width {
        return None; // N == 1 is only valid for the 64-bit form
    }
    let levels: u32 = esize - 1;
    let s: u32 = imms & levels;
    let r: u32 = immr & levels;
    if s == levels {
        return None; // an all-ones element is reserved
    }
    let elem: u64 = (1u64 << (s + 1)) - 1;
    let rotated = ror_in_esize(elem, r, esize);
    let mut result: u64 = 0;
    let mut pos: u32 = 0;
    while pos < width {
        result |= rotated << pos;
        pos += esize;
    }
    Some(if x64 { result } else { result & 0xFFFF_FFFF })
}

/// Rotate the low `esize` bits of `v` right by `r`.
fn ror_in_esize(v: u64, r: u32, esize: u32) -> u64 {
    let mask: u64 = if esize == 64 {
        u64::MAX
    } else {
        (1u64 << esize) - 1
    };
    let rr = r % esize;
    if rr == 0 {
        return v & mask;
    }
    ((v >> rr) | (v << (esize - rr))) & mask
}

/// Load-acquire / store-release register (`ldar{,b,h}`, `stlr{,b,h}`): `[xn]` addressing only.
fn ordered_mem(out: &mut String, w: u32) -> bool {
    let mnem = match w & 0x3FFF_FC00 {
        0x08DF_FC00 => "ldar",
        0x089F_FC00 => "stlr",
        _ => return false,
    };
    let size = w >> 30;
    out.push_str(mnem);
    out.push_str(["b", "h", "", ""][size as usize]);
    out.push(' ');
    gp(out, size == 3, rd(w));
    out.push_str(", [");
    sp(out, true, rn(w));
    out.push(']');
    true
}

/// Scaled-unsigned-offset loads/stores (integer and FP). Returns whether it
/// matched.
fn mem_imm(out: &mut String, w: u32) -> bool {
    struct Entry {
        match_word: u32,
        mnem: &'static str,
        x64: bool,
        scale: u32,
        fp: bool,
        dbl: bool,
    }
    const TABLE: &[Entry] = &[
        Entry {
            match_word: 0xF900_0000,
            mnem: "str",
            x64: true,
            scale: 3,
            fp: false,
            dbl: false,
        },
        Entry {
            match_word: 0xF940_0000,
            mnem: "ldr",
            x64: true,
            scale: 3,
            fp: false,
            dbl: false,
        },
        Entry {
            match_word: 0xB900_0000,
            mnem: "str",
            x64: false,
            scale: 2,
            fp: false,
            dbl: false,
        },
        Entry {
            match_word: 0xB940_0000,
            mnem: "ldr",
            x64: false,
            scale: 2,
            fp: false,
            dbl: false,
        },
        Entry {
            match_word: 0x3900_0000,
            mnem: "strb",
            x64: false,
            scale: 0,
            fp: false,
            dbl: false,
        },
        Entry {
            match_word: 0x3940_0000,
            mnem: "ldrb",
            x64: false,
            scale: 0,
            fp: false,
            dbl: false,
        },
        Entry {
            match_word: 0x39C0_0000,
            mnem: "ldrsb",
            x64: false,
            scale: 0,
            fp: false,
            dbl: false,
        },
        Entry {
            match_word: 0x7900_0000,
            mnem: "strh",
            x64: false,
            scale: 1,
            fp: false,
            dbl: false,
        },
        Entry {
            match_word: 0x7940_0000,
            mnem: "ldrh",
            x64: false,
            scale: 1,
            fp: false,
            dbl: false,
        },
        Entry {
            match_word: 0x79C0_0000,
            mnem: "ldrsh",
            x64: false,
            scale: 1,
            fp: false,
            dbl: false,
        },
        Entry {
            match_word: 0xBD00_0000,
            mnem: "str",
            x64: false,
            scale: 2,
            fp: true,
            dbl: false,
        },
        Entry {
            match_word: 0xBD40_0000,
            mnem: "ldr",
            x64: false,
            scale: 2,
            fp: true,
            dbl: false,
        },
        Entry {
            match_word: 0xFD00_0000,
            mnem: "str",
            x64: false,
            scale: 3,
            fp: true,
            dbl: true,
        },
        Entry {
            match_word: 0xFD40_0000,
            mnem: "ldr",
            x64: false,
            scale: 3,
            fp: true,
            dbl: true,
        },
        Entry {
            match_word: 0x3D80_0000,
            mnem: "str",
            x64: false,
            scale: 4,
            fp: true,
            dbl: false,
        }, // Q
        Entry {
            match_word: 0x3DC0_0000,
            mnem: "ldr",
            x64: false,
            scale: 4,
            fp: true,
            dbl: false,
        }, // Q
        Entry {
            match_word: 0x7D00_0000,
            mnem: "str",
            x64: false,
            scale: 1,
            fp: true,
            dbl: false,
        }, // H (f16)
        Entry {
            match_word: 0x7D40_0000,
            mnem: "ldr",
            x64: false,
            scale: 1,
            fp: true,
            dbl: false,
        }, // H (f16)
    ];
    for m in TABLE {
        if w & 0xFFC0_0000 != m.match_word {
            continue;
        }
        let off: u32 = ((w >> 10) & 0xFFF) << m.scale;
        out.push_str(m.mnem);
        out.push(' ');
        if m.fp {
            // The FP view: Q (128-bit), D (64-bit), H (16-bit, the f16
            // boundary form), else S.
            let c: char = if m.scale == 4 {
                'q'
            } else if m.dbl {
                'd'
            } else if m.scale == 1 {
                'h'
            } else {
                's'
            };
            let _ = write!(out, "{c}{}", rd(w));
        } else {
            gp(out, m.x64, rd(w));
        }
        out.push_str(", [");
        sp(out, true, rn(w));
        if off != 0 {
            let _ = write!(out, ", #{off}]");
        } else {
            out.push(']');
        }
        return true;
    }
    // strb/ldrb/etc with a zero-offset `[xn]` form already covered (off field 0).
    false
}

/// Scalar floating-point instructions. Returns whether it matched.
fn fp_decode(out: &mut String, w: u32) -> bool {
    let dbl = (w >> 22) & 1 != 0;
    let fc: char = if dbl { 'd' } else { 's' };
    // Three-register scalar FP arithmetic.
    const FP3: &[(u32, &str)] = &[
        (0x1E20_2800, "fadd"),
        (0x1E20_3800, "fsub"),
        (0x1E20_0800, "fmul"),
        (0x1E20_1800, "fdiv"),
    ];
    for &(m, mnem) in FP3 {
        if w & 0xFFA0_FC00 == m {
            let _ = write!(out, "{mnem} {fc}{}, {fc}{}, {fc}{}", rd(w), rn(w), rm(w));
            return true;
        }
    }
    // Floating-point data-processing (3-source): Rd = op(Rn*Rm, Ra). Mask
    // excludes ftype (bit 22, read via `dbl` above) and the four 5-bit
    // register fields.
    const FP3A: &[(u32, &str)] = &[
        (0x1F00_0000, "fmadd"),
        (0x1F00_8000, "fmsub"),
        (0x1F20_0000, "fnmadd"),
        (0x1F20_8000, "fnmsub"),
    ];
    for &(m, mnem) in FP3A {
        if w & 0xFF20_8000 == m {
            let _ = write!(
                out,
                "{mnem} {fc}{}, {fc}{}, {fc}{}, {fc}{}",
                rd(w),
                rn(w),
                rm(w),
                ra(w)
            );
            return true;
        }
    }
    if w & 0xFFA0_0C00 == 0x1E20_0C00 {
        // fcsel (mask excludes the cond field [12:15])
        let _ = write!(
            out,
            "fcsel {fc}{}, {fc}{}, {fc}{}, {}",
            rd(w),
            rn(w),
            rm(w),
            cond_name((w >> 12) & 0xF)
        );
        return true;
    }
    if w & 0xFFA0_FC1F == 0x1E20_2000 {
        // fcmp
        let _ = write!(out, "fcmp {fc}{}, {fc}{}", rn(w), rm(w));
        return true;
    }
    if w & 0xFFFF_FC00 == 0x1E60_4000 {
        // fmov (reg, 64-bit view)
        let _ = write!(out, "fmov d{}, d{}", rd(w), rn(w));
        return true;
    }
    // fmov between a general register and an FP register (both directions,
    // s/d).
    struct Fmov {
        match_word: u32,
        to_gpr: bool,
        x64: bool,
    }
    const FMOV: &[Fmov] = &[
        Fmov {
            match_word: 0x1E27_0000,
            to_gpr: false,
            x64: false,
        },
        Fmov {
            match_word: 0x9E67_0000,
            to_gpr: false,
            x64: true,
        },
        Fmov {
            match_word: 0x1E26_0000,
            to_gpr: true,
            x64: false,
        },
        Fmov {
            match_word: 0x9E66_0000,
            to_gpr: true,
            x64: true,
        },
    ];
    for e in FMOV {
        if w & 0xFFFF_FC00 == e.match_word {
            let f: char = if e.x64 { 'd' } else { 's' };
            out.push_str("fmov ");
            if e.to_gpr {
                gp(out, e.x64, rd(w));
                let _ = write!(out, ", {f}{}", rn(w));
            } else {
                let _ = write!(out, "{f}{}, ", rd(w));
                gp(out, e.x64, rn(w));
            }
            return true;
        }
    }
    // Two-register scalar FP ops (sqrt, rounding, conversions).
    const FP2: &[(u32, &str)] = &[
        (0x1E21_C000, "fsqrt"),
        (0x1E24_4000, "frintn"),
        (0x1E24_C000, "frintp"),
        (0x1E25_4000, "frintm"),
        (0x1E25_C000, "frintz"),
    ];
    for &(m, mnem) in FP2 {
        if w & 0xFFBF_FC00 == m {
            let _ = write!(out, "{mnem} {fc}{}, {fc}{}", rd(w), rn(w));
            return true;
        }
    }
    if w & 0xFFBF_FC00 == 0x1E22_0000 {
        // scvtf sd, wn
        let _ = write!(out, "scvtf {fc}{}, ", rd(w));
        gp(out, false, rn(w));
        return true;
    }
    if w & 0xFFBF_FC00 == 0x1E23_0000 {
        // ucvtf
        let _ = write!(out, "ucvtf {fc}{}, ", rd(w));
        gp(out, false, rn(w));
        return true;
    }
    if w & 0xFFBF_FC00 == 0x1E38_0000 {
        // fcvtzs wd, sn
        out.push_str("fcvtzs ");
        gp(out, false, rd(w));
        let _ = write!(out, ", {fc}{}", rn(w));
        return true;
    }
    if w & 0xFFBF_FC00 == 0x1E39_0000 {
        // fcvtzu
        out.push_str("fcvtzu ");
        gp(out, false, rd(w));
        let _ = write!(out, ", {fc}{}", rn(w));
        return true;
    }
    if w & 0xFFFF_FC00 == 0x1E22_C000 {
        // fcvt d, s (s->d)
        let _ = write!(out, "fcvt d{}, s{}", rd(w), rn(w));
        return true;
    }
    if w & 0xFFFF_FC00 == 0x1E62_4000 {
        // fcvt s, d (d->s)
        let _ = write!(out, "fcvt s{}, d{}", rd(w), rn(w));
        return true;
    }
    if w & 0xFFFF_FC00 == 0x1EE2_4000 {
        // fcvt s, h (h->s, f16 widen)
        let _ = write!(out, "fcvt s{}, h{}", rd(w), rn(w));
        return true;
    }
    if w & 0xFFFF_FC00 == 0x1E23_C000 {
        // fcvt h, s (s->h, f16 narrow)
        let _ = write!(out, "fcvt h{}, s{}", rd(w), rn(w));
        return true;
    }
    if w & 0xFFFF_FC00 == 0x1E63_C000 {
        // fcvt h, d (d->h, f16 narrow, single round)
        let _ = write!(out, "fcvt h{}, d{}", rd(w), rn(w));
        return true;
    }
    false
}

/// NEON (128-bit vector) instructions. Returns whether it matched.
fn vec_decode(out: &mut String, w: u32) -> bool {
    // Three-register 4S / 16B forms.
    struct V3 {
        match_word: u32,
        mnem: &'static str,
        b16: bool,
    }
    const V3_TABLE: &[V3] = &[
        V3 {
            match_word: 0x4E20_D400,
            mnem: "fadd",
            b16: false,
        },
        V3 {
            match_word: 0x4EA0_D400,
            mnem: "fsub",
            b16: false,
        },
        V3 {
            match_word: 0x6E20_DC00,
            mnem: "fmul",
            b16: false,
        },
        V3 {
            match_word: 0x6E20_FC00,
            mnem: "fdiv",
            b16: false,
        },
        V3 {
            match_word: 0x4EA0_F400,
            mnem: "fmin",
            b16: false,
        },
        V3 {
            match_word: 0x4E20_F400,
            mnem: "fmax",
            b16: false,
        },
        V3 {
            match_word: 0x4E20_E400,
            mnem: "fcmeq",
            b16: false,
        },
        V3 {
            match_word: 0x6EA0_E400,
            mnem: "fcmgt",
            b16: false,
        },
        V3 {
            match_word: 0x6E20_E400,
            mnem: "fcmge",
            b16: false,
        },
        V3 {
            match_word: 0x6E60_1C00,
            mnem: "bsl",
            b16: true,
        },
        V3 {
            match_word: 0x4E20_CC00,
            mnem: "fmla",
            b16: false,
        },
        V3 {
            match_word: 0x4EA0_CC00,
            mnem: "fmls",
            b16: false,
        },
        V3 {
            match_word: 0x6E20_D400,
            mnem: "faddp",
            b16: false,
        },
    ];
    for e in V3_TABLE {
        if w == (e.match_word | (rm(w) << 16) | (rn(w) << 5) | rd(w)) {
            let s = if e.b16 { "16b" } else { "4s" };
            let _ = write!(
                out,
                "{} v{}.{s}, v{}.{s}, v{}.{s}",
                e.mnem,
                rd(w),
                rn(w),
                rm(w)
            );
            return true;
        }
    }
    // Two-register vector forms.
    if w & 0xFFFF_FC00 == 0x6EA0_F800 {
        return vec2(out, w, "fneg", "4s");
    }
    if w & 0xFFFF_FC00 == 0x6EA1_F800 {
        return vec2(out, w, "fsqrt", "4s");
    }
    if w & 0xFFFF_FC00 == 0x6E20_5800 {
        return vec2(out, w, "mvn", "16b");
    }
    // Horizontal reduce: addv sums the 4S lanes to a scalar S register;
    // faddpScalar finishes a pairwise float reduction from a 2S pair. Both
    // cross from the Vn.4S/2S view into Sd.
    if w & 0xFFFF_FC00 == 0x4EB1_B800 {
        let _ = write!(out, "addv s{}, v{}.4s", rd(w), rn(w));
        return true;
    }
    if w & 0xFFFF_FC00 == 0x7E30_D800 {
        let _ = write!(out, "faddp s{}, v{}.2s", rd(w), rn(w));
        return true;
    }
    // mov vd.16b, vn.16b is orr with rm == rn.
    if w & 0xFFE0_FC00 == 0x4EA0_1C00 && rm(w) == rn(w) {
        let _ = write!(out, "mov v{}.16b, v{}.16b", rd(w), rn(w));
        return true;
    }
    if w & 0xFFFF_FC00 == 0x4E04_0C00 {
        // dup Vd.4S, Wn
        let _ = write!(out, "dup v{}.4s, ", rd(w));
        gp(out, false, rn(w));
        return true;
    }
    // dup Vd.4S, Vn.s[i] / dup Sd, Vn.s[i] / ins Vd.s[i], Vn.s[0]: imm5 in
    // bits[16:20].
    let imm5 = (w >> 16) & 0x1F;
    let lane = (imm5 >> 3) & 3;
    if w & 0xFFE0_FC00 == 0x4E00_0400 {
        let _ = write!(out, "dup v{}.4s, v{}.s[{lane}]", rd(w), rn(w));
        return true;
    }
    if w & 0xFFE0_FC00 == 0x5E00_0400 {
        let _ = write!(out, "dup s{}, v{}.s[{lane}]", rd(w), rn(w));
        return true;
    }
    if w & 0xFFE0_8C00 == 0x6E00_0400 {
        let _ = write!(out, "ins v{}.s[{lane}], v{}.s[0]", rd(w), rn(w));
        return true;
    }
    if w & 0xFFE0_FC00 == 0x0E00_3C00 {
        // umov Wd, Vn.s[i]
        out.push_str("umov ");
        gp(out, false, rd(w));
        let _ = write!(out, ", v{}.s[{lane}]", rn(w));
        return true;
    }
    if w & 0xFFE0_FC00 == 0x4E00_1C00 {
        // ins Vd.s[i], Wn
        let _ = write!(out, "ins v{}.s[{lane}], ", rd(w));
        gp(out, false, rn(w));
        return true;
    }
    false
}

fn vec2(out: &mut String, w: u32, mnem: &str, suffix: &str) -> bool {
    let _ = write!(out, "{mnem} v{}.{suffix}, v{}.{suffix}", rd(w), rn(w));
    true
}

/// Render a bitfield-move (SBFM/UBFM/BFM) using the alias a debugger expects
/// (uxt*/sxt*/lsl/lsr/asr/ubfx/sbfx/bfi/bfxil). `opc` selects the family;
/// `immr`/`imms` drive the alias choice.
fn bitfield(out: &mut String, w: u32) {
    let opc = (w >> 29) & 3;
    let x64 = sf(w);
    let width: u32 = if x64 { 64 } else { 32 };
    let immr: u32 = (w >> 16) & 0x3F;
    let imms: u32 = (w >> 10) & 0x3F;

    match opc {
        2 => {
            // UBFM
            if imms + 1 == immr && imms != width - 1 {
                // lsl #(width-1-imms)
                bf_two(out, "lsl", x64, rd(w), rn(w));
                let _ = write!(out, ", #{}", width - 1 - imms);
            } else if imms == width - 1 {
                // lsr #immr
                bf_two(out, "lsr", x64, rd(w), rn(w));
                let _ = write!(out, ", #{immr}");
            } else if immr == 0 && imms == 7 && !x64 {
                bf_two(out, "uxtb", false, rd(w), rn(w));
            } else if immr == 0 && imms == 15 && !x64 {
                bf_two(out, "uxth", false, rd(w), rn(w));
            } else if imms < immr {
                // ubfiz Rd, Rn, #(width-immr), #(imms+1)
                bf_two(out, "ubfiz", x64, rd(w), rn(w));
                let _ = write!(out, ", #{}, #{}", width - immr, imms + 1);
            } else {
                // ubfx Rd, Rn, #immr, #(imms-immr+1)
                bf_two(out, "ubfx", x64, rd(w), rn(w));
                let _ = write!(out, ", #{immr}, #{}", imms - immr + 1);
            }
        }
        0 => {
            // SBFM
            if imms == width - 1 {
                // asr #immr
                bf_two(out, "asr", x64, rd(w), rn(w));
                let _ = write!(out, ", #{immr}");
            } else if immr == 0 && imms == 7 {
                bf_two(out, "sxtb", x64, rd(w), rn(w));
            } else if immr == 0 && imms == 15 {
                bf_two(out, "sxth", x64, rd(w), rn(w));
            } else if immr == 0 && imms == 31 && x64 {
                // sxtw Xd, Wn
                out.push_str("sxtw ");
                gp(out, true, rd(w));
                out.push_str(", ");
                gp(out, false, rn(w));
            } else if imms < immr {
                // sbfiz
                bf_two(out, "sbfiz", x64, rd(w), rn(w));
                let _ = write!(out, ", #{}, #{}", width - immr, imms + 1);
            } else {
                bf_two(out, "sbfx", x64, rd(w), rn(w));
                let _ = write!(out, ", #{immr}, #{}", imms - immr + 1);
            }
        }
        _ => {
            // BFM
            if imms < immr {
                // bfi Rd, Rn, #(width-immr), #(imms+1)
                bf_two(out, "bfi", x64, rd(w), rn(w));
                let _ = write!(out, ", #{}, #{}", width - immr, imms + 1);
            } else {
                // bfxil Rd, Rn, #immr, #(imms-immr+1)
                bf_two(out, "bfxil", x64, rd(w), rn(w));
                let _ = write!(out, ", #{immr}, #{}", imms - immr + 1);
            }
        }
    }
}

/// The common `mnem Rd, Rn` shape for bitfield aliases, at the register width.
fn bf_two(out: &mut String, mnem: &str, xw: bool, d: u32, n: u32) {
    out.push_str(mnem);
    out.push(' ');
    gp(out, xw, d);
    out.push_str(", ");
    gp(out, xw, n);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_basic_words() {
        // Hand-computed words (no dependency on the encoder module):
        // ret = 0xD65F0000 | (30 << 5); add w0,w1,w2; bl .+8; nop.
        assert_eq!(one(0xD65F_03C0), "ret");
        assert_eq!(one(0x0B02_0020), "add w0, w1, w2");
        assert_eq!(one(0x8B02_0020), "add x0, x1, x2");
        assert_eq!(one(0x9400_0002), "bl .+8");
        assert_eq!(one(0x1400_0002), "b .+8");
        // nop is outside the emitted-subset decode coverage: it must surface
        // as a raw word, never a mis-decode.
        assert_eq!(one(0xD503_201F), ".word 0xd503201f");
        // ldar/stlr in every width (GNU as output).
        assert_eq!(one(0x88df_fc20), "ldar w0, [x1]");
        assert_eq!(one(0xc8df_fc62), "ldar x2, [x3]");
        assert_eq!(one(0x08df_fca4), "ldarb w4, [x5]");
        assert_eq!(one(0x48df_fce6), "ldarh w6, [x7]");
        assert_eq!(one(0x889f_fc20), "stlr w0, [x1]");
        assert_eq!(one(0xc89f_fc62), "stlr x2, [x3]");
        assert_eq!(one(0x089f_fca4), "stlrb w4, [x5]");
        assert_eq!(one(0x489f_fce6), "stlrh w6, [x7]");
    }

    #[test]
    fn decodes_objdump_vectors() {
        // Spot checks from the Zig suite (verified against objdump there).
        assert_eq!(one(0xEB09_0108), "subs x8, x8, x9");
        assert_eq!(one(0x3400_00E8), "cbz w8, .+28");
        assert_eq!(one(0x5400_00C3), "b.cc .+24");
        assert_eq!(one(0xF85F_83A8), "ldur x8, [x29, #-8]");
        assert_eq!(one(0xA941_7BFD), "ldp x29, x30, [sp, #16]");
        assert_eq!(one(0x5300_3D08), "uxth w8, w8");
        assert_eq!(one(0x531F_7908), "lsl w8, w8, #1");
        assert_eq!(one(0x9340_7C00), "sxtw x0, w0");
        assert_eq!(one(0x9280_0AA1), "mov x1, #0xffffffffffffffaa");
    }

    #[test]
    fn format_with_lines_interleaves_markers() {
        let code = [0x0B02_0020, 0x1B01_7C00, 0xD65F_03C0];
        let lines = [
            SourceLine { offset: 0, line: 2 },
            SourceLine { offset: 4, line: 3 },
        ];
        assert_eq!(
            format_with_lines(&code, &lines),
            "; line 2\n\
             0000: 0b020020  add w0, w1, w2\n\
             ; line 3\n\
             0004: 1b017c00  mul w0, w0, w1\n\
             0008: d65f03c0  ret\n"
        );
    }

    #[test]
    fn format_module_annotates_branch_targets() {
        // fn f at word 0, fn g at word 3; b .+8 lands at words 2 and 4.
        let code = [
            0x1400_0002,
            0xD65F_03C0,
            0x1400_0002,
            0xD65F_03C0,
            0xD65F_03C0,
        ];
        let syms = [Sym { name: "f", word: 0 }, Sym { name: "g", word: 3 }];
        let text = format_module(&code, &syms);
        assert!(text.contains("b .+8  <f+0x8>"), "{text}");
        assert!(text.contains("b .+8  <g+0x4>"), "{text}");
    }

    #[test]
    fn format_elf_resolves_absolute_targets() {
        let base: u64 = 0x1000;
        let code = [0x9400_0002, 0xB000_0000, 0xD65F_03C0];
        let syms = [
            AddrSym {
                name: "start",
                addr: 0x1000,
                size: 8,
                is_func: true,
            },
            AddrSym {
                name: "callee",
                addr: 0x1008,
                size: 4,
                is_func: true,
            },
            AddrSym {
                name: "gv",
                addr: 0x2000,
                size: 16,
                is_func: false,
            },
        ];
        let text = format_elf(&code, base, &syms);
        assert!(text.contains("0000000000001000 <start>:"), "{text}");
        assert!(text.contains("0000000000001008 <callee>:"), "{text}");
        assert!(text.contains("bl 0x1008 <callee>"), "{text}");
        assert!(text.contains("adrp x0, 0x2000 <gv>"), "{text}");
    }
}
