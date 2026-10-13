//! Assembly listings of AArch64 code buffers, for debugging code generation.
//!
//! The text of each instruction comes from `volt_isa_aarch64::format`; this module
//! only lays words out with addresses, labels, source-line markers and symbol
//! annotations. A word the decoder does not recognise prints as `.word 0x<hex>`.

use alloc::string::{String, ToString as _};
use core::fmt::Write as _;
use volt_isa_aarch64::flow::flow;
use volt_isa_aarch64::format::{self, Symbols};

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
/// resolving branch and `adrp` targets when listing an ELF image.
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

/// The symbol containing an address: the greatest-addressed symbol at or before it.
struct AddrSyms<'a>(&'a [AddrSym<'a>]);

impl Symbols for AddrSyms<'_> {
    fn lookup(&self, addr: u64) -> Option<(&str, u64)> {
        let mut best: Option<&AddrSym<'_>> = None;
        for s in self.0 {
            if s.addr <= addr && best.is_none_or(|b| s.addr > b.addr) {
                best = Some(s);
            }
        }
        best.map(|s| (s.name, addr - s.addr))
    }
}

/// Render one instruction word as `mnemonic operands`.
pub fn one(word: u32) -> String {
    format::word(word).to_string()
}

/// Render a whole instruction stream (one word per line, `addr: hex  text`).
pub fn format(code: &[u32]) -> String {
    let mut out = String::new();
    for (i, &word) in code.iter().enumerate() {
        let _ = writeln!(out, "{:04x}: {word:08x}  {}", i * 4, format::word(word));
    }
    out
}

/// Render a listing with source-line markers interleaved (objdump `-S`
/// style): a `; line N` header precedes the instructions belonging to that
/// source line. `lines` must be sorted by offset (as isel produces).
pub fn format_with_lines(code: &[u32], lines: &[SourceLine]) -> String {
    let mut out = String::new();
    let mut next = 0;
    for (i, &word) in code.iter().enumerate() {
        let offset = (i * 4) as u32;
        while next < lines.len() && lines[next].offset == offset {
            let _ = writeln!(out, "; line {}", lines[next].line);
            next += 1;
        }
        let _ = writeln!(out, "{offset:04x}: {word:08x}  {}", format::word(word));
    }
    out
}

/// Render a whole linked module: each function gets a `name:` label at its
/// offset, and every PC-relative branch is annotated with the function it lands
/// in (`<callee>` or `<callee+0xN>`), so a linked image reads like a symbolized
/// listing.
pub fn format_module(code: &[u32], syms: &[Sym<'_>]) -> String {
    let mut out = String::new();
    for (i, &word) in code.iter().enumerate() {
        for s in syms.iter().filter(|s| s.word == i) {
            if i != 0 {
                out.push('\n');
            }
            let _ = writeln!(out, "{}:", s.name);
        }
        let _ = write!(out, "{:04x}: {word:08x}  {}", i * 4, format::word(word));
        let target = flow(word, (i * 4) as u64)
            .target()
            .filter(|&t| (t as i64) >= 0)
            .map(|t| (t / 4) as usize);
        if let Some(target) = target
            && let Some(s) = syms
                .iter()
                .filter(|s| s.word <= target)
                .max_by_key(|s| s.word)
        {
            if target == s.word {
                let _ = write!(out, "  <{}>", s.name);
            } else {
                let _ = write!(out, "  <{}+0x{:x}>", s.name, (target - s.word) * 4);
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
    format_elf_with_lines(code, base, syms, &[])
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
    let symbols = AddrSyms(syms);
    let mut out = String::new();
    let mut next = 0;
    for (i, &word) in code.iter().enumerate() {
        let addr = base + i as u64 * 4;
        for s in syms.iter().filter(|s| s.is_func && s.addr == addr) {
            if i != 0 {
                out.push('\n');
            }
            let _ = writeln!(out, "{addr:016x} <{}>:", s.name);
        }
        while next < lines.len() && lines[next].addr < addr {
            next += 1; // skip any before this insn
        }
        while next < lines.len() && lines[next].addr == addr {
            let _ = writeln!(out, "; line {}", lines[next].line);
            next += 1;
        }
        let _ = writeln!(
            out,
            "{addr:08x}: {word:08x}  {}",
            format::word_with(word, addr, &symbols)
        );
    }
    out
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
        assert_eq!(one(0xD503_201F), "nop");
        // A word the decoder does not recognise surfaces as a raw word.
        assert_eq!(
            one(0x0e21_4022 /* addhn: not decoded */),
            ".word 0x0e214022"
        );
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
