// SPDX-License-Identifier: Apache-2.0
// Copyright LilithSemi (Vulcan). Modified for the Volt Rust port.
//! Structural decoder for the x86-64 subset emitted by Vulcan/Volt.
//! Ported from Vulcan x86_64/disasm.zig (Apache-2.0). Unknown or truncated
//! instructions render as bytes; no external disassembler dependency is used.
use alloc::{format, string::String};
use core::fmt::Write;
#[derive(Clone, Debug)]
pub struct Sym {
    pub name: String,
    pub offset: usize,
}
#[derive(Clone, Copy, Debug)]
pub struct AddrLine {
    pub addr: u64,
    pub line: u32,
}
const GPR: [&str; 16] = [
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13",
    "r14", "r15",
];
const GPR32: [&str; 16] = [
    "eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi", "r8d", "r9d", "r10d", "r11d", "r12d",
    "r13d", "r14d", "r15d",
];
const GPR16: [&str; 16] = [
    "ax", "cx", "dx", "bx", "sp", "bp", "si", "di", "r8w", "r9w", "r10w", "r11w", "r12w", "r13w",
    "r14w", "r15w",
];
const GPR8: [&str; 16] = [
    "al", "cl", "dl", "bl", "spl", "bpl", "sil", "dil", "r8b", "r9b", "r10b", "r11b", "r12b",
    "r13b", "r14b", "r15b",
];
const CC: [&str; 16] = [
    "o", "no", "b", "ae", "e", "ne", "be", "a", "s", "ns", "p", "np", "l", "ge", "le", "g",
];
const ALU: [&str; 8] = ["add", "or", "adc", "sbb", "and", "sub", "xor", "cmp"];
struct D<'a> {
    code: &'a [u8],
    i: usize,
    mand: u8,
    w: bool,
    r: u8,
    x: u8,
    b: u8,
    vex: bool,
    wide: bool,
    v: u8,
    map: u8,
}
struct O {
    reg: u8,
    rm: u8,
    mem: bool,
    base: Option<u8>,
    rip: bool,
    index: Option<u8>,
    scale: u8,
    disp: i32,
}
impl D<'_> {
    fn byte(&mut self) -> Option<u8> {
        let b = *self.code.get(self.i)?;
        self.i += 1;
        Some(b)
    }
    fn peek(&self) -> Option<u8> {
        self.code.get(self.i).copied()
    }
    fn imm(&mut self, n: usize) -> Option<u64> {
        let mut out = 0;
        for j in 0..n {
            out |= (self.byte()? as u64) << (j * 8);
        }
        Some(out)
    }
    fn rel(&mut self, n: usize) -> Option<String> {
        let imm = self.imm(n)?;
        let v = if n == 1 {
            imm as i8 as i64
        } else {
            imm as i32 as i64
        };
        Some(format!(
            ".{}{vabs}",
            if v < 0 { "-" } else { "+" },
            vabs = v.unsigned_abs()
        ))
    }
    fn rv(&self, n: u8) -> &'static str {
        if self.w {
            GPR[n as usize]
        } else if self.mand == 0x66 {
            GPR16[n as usize]
        } else {
            GPR32[n as usize]
        }
    }
    fn rvq(&self, n: u8) -> &'static str {
        if self.w {
            GPR[n as usize]
        } else {
            GPR32[n as usize]
        }
    }
    fn hint(&self) -> &'static str {
        if self.w {
            "qword ptr "
        } else if self.mand == 0x66 {
            "word ptr "
        } else {
            "dword ptr "
        }
    }
    fn xr(&self, n: u8) -> String {
        format!("{}mm{n}", if self.wide { "y" } else { "x" })
    }
    fn modrm(&mut self) -> Option<O> {
        let m = self.byte()?;
        let mode = m >> 6;
        let reg = (self.r << 3) | ((m >> 3) & 7);
        let rm = (self.b << 3) | (m & 7);
        let mut o = O {
            reg,
            rm,
            mem: mode != 3,
            base: Some(rm),
            rip: false,
            index: None,
            scale: 1,
            disp: 0,
        };
        if mode == 3 {
            return Some(o);
        }
        if m & 7 == 4 {
            let sib = self.byte()?;
            let idx = (self.x << 3) | ((sib >> 3) & 7);
            if idx != 4 {
                o.index = Some(idx);
            }
            o.scale = 1 << (sib >> 6);
            o.base = Some((self.b << 3) | (sib & 7));
            if mode == 0 && sib & 7 == 5 {
                o.base = None;
                o.disp = self.imm(4)? as i32;
            }
        } else if mode == 0 && m & 7 == 5 {
            o.base = None;
            o.rip = true;
            o.disp = self.imm(4)? as i32;
        }
        if mode == 1 {
            o.disp = self.byte()? as i8 as i32;
        } else if mode == 2 {
            o.disp = self.imm(4)? as i32;
        }
        Some(o)
    }
    fn operand(&self, o: &O, hint: &str, register: &str) -> String {
        if !o.mem {
            return register.into();
        }
        let mut s = String::from(hint);
        s.push('[');
        let mut has = false;
        if o.rip {
            s.push_str("rip");
            has = true;
        } else if let Some(base) = o.base {
            s.push_str(GPR[base as usize]);
            has = true;
        }
        if let Some(i) = o.index {
            if has {
                s.push_str(" + ");
            }
            let _ = write!(s, "{}", GPR[i as usize]);
            if o.scale > 1 {
                let _ = write!(s, "*{}", o.scale);
            }
            has = true;
        }
        if o.disp != 0 || !has {
            if has {
                let _ = write!(
                    s,
                    " {} {}",
                    if o.disp < 0 { "-" } else { "+" },
                    o.disp.unsigned_abs()
                );
            } else {
                let _ = write!(s, "{}", o.disp);
            }
        }
        s.push(']');
        s
    }
    fn rm(&self, o: &O) -> String {
        self.operand(o, self.hint(), self.rv(o.rm))
    }
    fn xmm_rm(&self, o: &O) -> String {
        self.operand(
            o,
            if self.wide {
                "ymmword ptr "
            } else if self.mand == 0xf3 {
                "dword ptr "
            } else if self.mand == 0xf2 {
                "qword ptr "
            } else {
                "xmmword ptr "
            },
            &self.xr(o.rm),
        )
    }
    fn prefixes(&mut self) -> Option<()> {
        loop {
            match self.peek()? {
                0x66 | 0xf2 | 0xf3 => self.mand = self.byte()?,
                0x2e | 0x36 | 0x3e | 0x26 | 0x64 | 0x65 | 0x67 | 0xf0 => {
                    self.byte()?;
                }
                _ => break,
            }
        }
        if self.peek() == Some(0xc4) {
            self.byte()?;
            let b2 = self.byte()?;
            let b3 = self.byte()?;
            self.vex = true;
            self.r = ((b2 >> 7) & 1) ^ 1;
            self.x = ((b2 >> 6) & 1) ^ 1;
            self.b = ((b2 >> 5) & 1) ^ 1;
            self.map = b2 & 31;
            self.w = b3 & 0x80 != 0;
            self.wide = b3 & 4 != 0;
            self.v = (!(b3 >> 3)) & 15;
            self.mand = match b3 & 3 {
                1 => 0x66,
                2 => 0xf3,
                3 => 0xf2,
                _ => 0,
            };
        } else {
            if self.peek().is_some_and(|p| p & 0xf0 == 0x40) {
                let r = self.byte()?;
                self.w = r & 8 != 0;
                self.r = (r >> 2) & 1;
                self.x = (r >> 1) & 1;
                self.b = r & 1;
            }
            if self.peek() == Some(0x0f) {
                self.byte()?;
                self.map = 1;
                if matches!(self.peek(), Some(0x38 | 0x3a)) {
                    self.map = if self.byte()? == 0x38 { 2 } else { 3 };
                }
            }
        }
        Some(())
    }
    fn decode(&mut self) -> Option<String> {
        self.prefixes()?;
        let op = self.byte()?;
        if self.map != 0 {
            return self.extended(op);
        }
        Some(match op {
            0xc3 => "ret".into(),
            0x90 => "nop".into(),
            0x98 => if self.w { "cdqe" } else { "cwde" }.into(),
            0x99 => if self.w { "cqo" } else { "cdq" }.into(),
            0x50..=0x5f => format!(
                "{} {}",
                if op < 0x58 { "push" } else { "pop" },
                GPR[((self.b << 3) | (op & 7)) as usize]
            ),
            0xb8..=0xbf => {
                let r = (self.b << 3) | (op & 7);
                let n = self.imm(if self.w {
                    8
                } else if self.mand == 0x66 {
                    2
                } else {
                    4
                })?;
                format!("mov {}, {n}", self.rv(r))
            }
            0xe8 | 0xe9 | 0xeb => format!(
                "{} {}",
                if op == 0xe8 { "call" } else { "jmp" },
                self.rel(if op == 0xeb { 1 } else { 4 })?
            ),
            0x70..=0x7f => format!("j{} {}", CC[(op & 15) as usize], self.rel(1)?),
            0x81 | 0x83 => {
                let o = self.modrm()?;
                let n = if op == 0x83 {
                    self.byte()? as i8 as i64
                } else {
                    self.imm(if self.mand == 0x66 { 2 } else { 4 })? as i32 as i64
                };
                format!("{} {}, {n}", ALU[(o.reg & 7) as usize], self.rm(&o))
            }
            0xc1 | 0xd3 => {
                let o = self.modrm()?;
                let name = match o.reg & 7 {
                    4 => "shl",
                    5 => "shr",
                    7 => "sar",
                    _ => return None,
                };
                let n = if op == 0xc1 {
                    format!("{}", self.byte()?)
                } else {
                    "cl".into()
                };
                format!("{name} {}, {n}", self.rm(&o))
            }
            0xf7 => {
                let o = self.modrm()?;
                let name =
                    ["test", "?", "not", "neg", "mul", "imul", "div", "idiv"][(o.reg & 7) as usize];
                if o.reg & 7 == 0 {
                    format!("test {}, {}", self.rm(&o), self.imm(4)? as i32)
                } else {
                    format!("{name} {}", self.rm(&o))
                }
            }
            0xff => {
                let o = self.modrm()?;
                let name = match o.reg & 7 {
                    0 => "inc",
                    1 => "dec",
                    2 => "call",
                    4 => "jmp",
                    6 => "push",
                    _ => return None,
                };
                format!(
                    "{name} {}",
                    self.operand(&o, "qword ptr ", GPR[o.rm as usize])
                )
            }
            0x69 => {
                let o = self.modrm()?;
                format!(
                    "imul {}, {}, {}",
                    self.rv(o.reg),
                    self.rm(&o),
                    self.imm(4)? as i32
                )
            }
            0xc6 | 0xc7 => {
                let o = self.modrm()?;
                let n = self.imm(if op == 0xc6 {
                    1
                } else if self.mand == 0x66 {
                    2
                } else {
                    4
                })?;
                format!(
                    "mov {}, {n}",
                    if op == 0xc6 {
                        self.operand(&o, "byte ptr ", GPR8[o.rm as usize])
                    } else {
                        self.rm(&o)
                    }
                )
            }
            0x8b | 0x8d | 0x63 => {
                let o = self.modrm()?;
                format!(
                    "{} {}, {}",
                    match op {
                        0x8d => "lea",
                        0x63 => "movsxd",
                        _ => "mov",
                    },
                    self.rv(o.reg),
                    self.operand(
                        &o,
                        if op == 0x8d {
                            ""
                        } else if op == 0x63 {
                            "dword ptr "
                        } else {
                            self.hint()
                        },
                        if op == 0x63 {
                            GPR32[o.rm as usize]
                        } else {
                            self.rv(o.rm)
                        }
                    )
                )
            }
            0xa8 | 0xa9 => format!(
                "test {}, {}",
                if op == 0xa8 { "al" } else { self.rv(0) },
                self.imm(if op == 0xa8 { 1 } else { 4 })?
            ),
            _ if (op <= 0x3b && op & 7 <= 3) || matches!(op, 0x84 | 0x85 | 0x88 | 0x89 | 0x8a) => {
                let o = self.modrm()?;
                let byte = op & 1 == 0;
                let name = if op >= 0x88 {
                    "mov"
                } else if op >= 0x84 {
                    "test"
                } else {
                    ALU[((op >> 3) & 7) as usize]
                };
                let reg = if byte {
                    GPR8[o.reg as usize]
                } else {
                    self.rv(o.reg)
                };
                let rm = self.operand(
                    &o,
                    if byte { "byte ptr " } else { self.hint() },
                    if byte {
                        GPR8[o.rm as usize]
                    } else {
                        self.rv(o.rm)
                    },
                );
                if op & 2 != 0 {
                    format!("{name} {reg}, {rm}")
                } else {
                    format!("{name} {rm}, {reg}")
                }
            }
            _ if op <= 0x3d && matches!(op & 7, 4 | 5) => {
                let byte = op & 1 == 0;
                format!(
                    "{} {}, {}",
                    ALU[((op >> 3) & 7) as usize],
                    if byte { "al" } else { self.rv(0) },
                    self.imm(if byte { 1 } else { 4 })?
                )
            }
            _ => return None,
        })
    }
    fn extended(&mut self, op: u8) -> Option<String> {
        if self.map == 1 && !self.vex {
            match op {
                0x05 => return Some("syscall".into()),
                0x1e => {
                    self.byte()?;
                    return Some("endbr64".into());
                }
                0x1f => {
                    self.modrm()?;
                    return Some("nop".into());
                }
                0x80..=0x8f => {
                    return Some(format!("j{} {}", CC[(op & 15) as usize], self.rel(4)?));
                }
                0x90..=0x9f => {
                    let o = self.modrm()?;
                    return Some(format!(
                        "set{} {}",
                        CC[(op & 15) as usize],
                        self.operand(&o, "byte ptr ", GPR8[o.rm as usize])
                    ));
                }
                0x40..=0x4f => {
                    let o = self.modrm()?;
                    return Some(format!(
                        "cmov{} {}, {}",
                        CC[(op & 15) as usize],
                        self.rv(o.reg),
                        self.rm(&o)
                    ));
                }
                0xaf => {
                    let o = self.modrm()?;
                    return Some(format!("imul {}, {}", self.rv(o.reg), self.rm(&o)));
                }
                0xb6 | 0xb7 | 0xbe | 0xbf => {
                    let o = self.modrm()?;
                    let half = op & 1 != 0;
                    return Some(format!(
                        "{} {}, {}",
                        if op & 8 != 0 { "movsx" } else { "movzx" },
                        self.rv(o.reg),
                        self.operand(
                            &o,
                            if half { "word ptr " } else { "byte ptr " },
                            if half {
                                GPR16[o.rm as usize]
                            } else {
                                GPR8[o.rm as usize]
                            }
                        )
                    ));
                }
                0x18 => {
                    let o = self.modrm()?;
                    return Some(format!(
                        "prefetcht{} {}",
                        o.reg & 3,
                        self.operand(&o, "", GPR[o.rm as usize])
                    ));
                }
                _ => {}
            }
        }
        let o = self.modrm()?;
        if self.vex && self.map == 3 && op == 0x18 {
            let immediate = self.byte()?;
            return Some(format!(
                "vinsertf128 ymm{}, ymm{}, {}, {immediate}",
                o.reg,
                self.v,
                self.operand(&o, "xmmword ptr ", &format!("xmm{}", o.rm))
            ));
        }
        if self.vex && self.map == 3 && op == 0x19 {
            let immediate = self.byte()?;
            return Some(format!(
                "vextractf128 {}, ymm{}, {immediate}",
                self.operand(&o, "xmmword ptr ", &format!("xmm{}", o.rm)),
                o.reg
            ));
        }
        if self.vex && self.map == 3 && op == 0x1d {
            let immediate = self.byte()?;
            return Some(format!(
                "vcvtps2ph {}, {}, {immediate}",
                self.operand(
                    &o,
                    if self.wide {
                        "xmmword ptr "
                    } else {
                        "qword ptr "
                    },
                    &format!("xmm{}", o.rm)
                ),
                self.xr(o.reg)
            ));
        }
        if self.map == 1 && matches!(op, 0x6e | 0x7e) {
            let name = if self.w { "movq" } else { "movd" };
            return Some(if op == 0x6e {
                format!(
                    "{name} xmm{}, {}",
                    o.reg,
                    self.operand(&o, self.hint(), self.rvq(o.rm))
                )
            } else {
                format!(
                    "{name} {}, xmm{}",
                    self.operand(&o, self.hint(), self.rvq(o.rm)),
                    o.reg
                )
            });
        }
        if self.map == 1 && matches!(op, 0x2a | 0x2c) {
            let double = self.mand == 0xf2;
            return Some(if op == 0x2a {
                format!(
                    "cvtsi2{} xmm{}, {}",
                    if double { "sd" } else { "ss" },
                    o.reg,
                    self.operand(&o, self.hint(), self.rvq(o.rm))
                )
            } else {
                format!(
                    "cvtt{}2si {}, {}",
                    if double { "sd" } else { "ss" },
                    self.rvq(o.reg),
                    self.xmm_rm(&o)
                )
            });
        }
        let (name, immediate) = if self.map == 3 {
            (
                match op {
                    0x08 => "roundps",
                    0x09 => "roundpd",
                    0x0a => "roundss",
                    0x0b => "roundsd",
                    0x11 => {
                        if self.mand == 0xf2 {
                            "roundsd"
                        } else {
                            "roundss"
                        }
                    }
                    0x18 => "insertf128",
                    0x19 => "extractf128",
                    0x1d => "cvtps2ph",
                    0x21 => "insertps",
                    0x22 => "pinsrd",
                    _ => return None,
                },
                true,
            )
        } else if self.map == 2 {
            (
                match op {
                    0x13 => "cvtph2ps",
                    0x40 => "pmulld",
                    _ => return None,
                },
                false,
            )
        } else {
            (
                match op {
                    0x10 | 0x11 => match self.mand {
                        0xf3 => "movss",
                        0xf2 => "movsd",
                        _ => "movups",
                    },
                    0x28 | 0x29 => {
                        if self.mand == 0x66 {
                            "movapd"
                        } else {
                            "movaps"
                        }
                    }
                    0x6f | 0x7f => {
                        if self.mand == 0x66 {
                            "movdqa"
                        } else {
                            "movdqu"
                        }
                    }
                    0xd6 => "movq",
                    0x58 => match self.mand {
                        0xf3 => "addss",
                        0xf2 => "addsd",
                        _ => "addps",
                    },
                    0x59 => match self.mand {
                        0xf3 => "mulss",
                        0xf2 => "mulsd",
                        _ => "mulps",
                    },
                    0x5c => match self.mand {
                        0xf3 => "subss",
                        0xf2 => "subsd",
                        _ => "subps",
                    },
                    0x5e => match self.mand {
                        0xf3 => "divss",
                        0xf2 => "divsd",
                        _ => "divps",
                    },
                    0x51 => match self.mand {
                        0xf3 => "sqrtss",
                        0xf2 => "sqrtsd",
                        _ => "sqrtps",
                    },
                    0x5a => {
                        if self.mand == 0xf2 {
                            "cvtsd2ss"
                        } else {
                            "cvtss2sd"
                        }
                    }
                    0x2e => {
                        if self.mand == 0x66 {
                            "ucomisd"
                        } else {
                            "ucomiss"
                        }
                    }
                    0x54 => "andps",
                    0x55 => "andnps",
                    0x56 => "orps",
                    0x57 => "xorps",
                    0xc2 => "cmpps",
                    0xc6 => "shufps",
                    0x70 => "pshufd",
                    0x73 => match o.reg & 7 {
                        2 => "psrlq",
                        3 => "psrldq",
                        6 => "psllq",
                        _ => "pslldq",
                    },
                    0xef => "pxor",
                    0xdb => "pand",
                    0xeb => "por",
                    0xdf => "pandn",
                    0xd4 => "paddq",
                    0xfb => "psubq",
                    0xfe => "paddd",
                    0xfa => "psubd",
                    0xf4 => "pmuludq",
                    0xd5 => "pmullw",
                    0xfc => "paddb",
                    0xfd => "paddw",
                    0xec => "paddsb",
                    0xed => "paddsw",
                    0xe2 => "psrad",
                    0x76 => "pcmpeqd",
                    0x66 => "pcmpgtd",
                    _ => return None,
                },
                matches!(op, 0xc2 | 0xc6 | 0x70 | 0x73),
            )
        };
        let store = (self.map == 1 && matches!(op, 0x11 | 0x29 | 0x7f | 0xd6))
            || (self.map == 3 && matches!(op, 0x19 | 0x1d));
        let mut s = format!("{}{name} ", if self.vex { "v" } else { "" });
        if store {
            s.push_str(&self.xmm_rm(&o));
            let _ = write!(s, ", {}", self.xr(o.reg));
        } else {
            let _ = write!(s, "{}, ", self.xr(o.reg));
            if self.vex
                && self.map == 1
                && matches!(
                    op,
                    0x58 | 0x59 | 0x5c | 0x5e | 0x54 | 0x55 | 0x56 | 0x57 | 0xc2
                )
            {
                let _ = write!(s, "{}, ", self.xr(self.v));
            }
            if self.map == 3 && op == 0x22 {
                s.push_str(&self.operand(&o, "dword ptr ", GPR32[o.rm as usize]));
            } else {
                s.push_str(&self.xmm_rm(&o));
            }
        }
        if immediate {
            let _ = write!(s, ", {}", self.byte()?);
        }
        Some(s)
    }
}
pub fn disasm_inst(out: &mut String, code: &[u8], pos: usize) -> usize {
    if pos >= code.len() {
        return 0;
    }
    let mut d = D {
        code,
        i: pos,
        mand: 0,
        w: false,
        r: 0,
        x: 0,
        b: 0,
        vex: false,
        wide: false,
        v: 0,
        map: 0,
    };
    if let Some(text) = d.decode() {
        out.push_str(&text);
        d.i - pos
    } else {
        let _ = write!(out, ".byte 0x{:02x}", code[pos]);
        1
    }
}
pub fn format(code: &[u8]) -> String {
    format_module_with_lines(code, &[], &[])
}
pub fn format_module(code: &[u8], syms: &[Sym]) -> String {
    format_module_with_lines(code, syms, &[])
}
pub fn format_module_with_lines(code: &[u8], syms: &[Sym], lines: &[AddrLine]) -> String {
    let mut out = String::new();
    let mut pos = 0;
    while pos < code.len() {
        for s in syms.iter().filter(|s| s.offset == pos) {
            if pos != 0 {
                out.push('\n');
            }
            let _ = writeln!(out, "{}:", s.name);
        }
        for l in lines.iter().filter(|l| l.addr == pos as u64) {
            let _ = writeln!(out, "; line {}", l.line);
        }
        let _ = write!(out, "{pos:04x}: ");
        let len = disasm_inst(&mut out, code, pos);
        if code[pos] == 0xe8 && len == 5 {
            let rel = i32::from_le_bytes(code[pos + 1..pos + 5].try_into().expect("decoded rel32"));
            let target = pos as i64 + 5 + rel as i64;
            if let Some(s) = syms.iter().find(|s| s.offset as i64 == target) {
                let _ = write!(out, "  <{}>", s.name);
            }
        }
        out.push('\n');
        pos += len;
    }
    out
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::x86_64::encode::*;
    #[test]
    fn integer_roundtrip() {
        let mut e = Encoder::default();
        e.imm(RAX, 42);
        e.mov(RAX, RBX);
        e.shift(4, RAX, Some(2), true);
        assert_eq!(
            format(&e.code),
            "0000: mov rax, 42\n000a: mov rax, rbx\n000d: shl rax, 2\n"
        );
    }
    #[test]
    fn truncated_instruction_is_byte() {
        assert_eq!(
            format(&[0x48, 0xb8, 1]),
            "0000: .byte 0x48\n0001: .byte 0xb8\n0002: .byte 0x01\n"
        );
    }
    #[test]
    fn xmm_and_memory_roundtrip() {
        let mut e = Encoder::default();
        e.xmm_mem(13, RBP, -32, 8, false);
        assert_eq!(format(&e.code), "0000: movsd xmm13, qword ptr [rbp - 32]\n");
    }
    #[test]
    fn sse_avx_and_lane_roundtrip() {
        let mut e = Encoder::default();
        e.rr(Some(0xf3), &[0x0f, 0x58], 0, 1, false);
        e.vex_rr(1, 0, 0x58, 0, 1, 2, true);
        e.vex_rr(3, 1, 0x19, 1, 0, 0, true);
        e.code.push(1);
        assert_eq!(
            format(&e.code),
            "0000: addss xmm0, xmm1\n0004: vaddps ymm0, ymm1, ymm2\n0009: vextractf128 xmm0, ymm1, 1\n"
        );
    }
    #[test]
    fn source_line_markers_and_symbolized_call() {
        let code = [0xe8, 0, 0, 0, 0, 0xc3];
        let syms = [
            Sym {
                name: "caller".into(),
                offset: 0,
            },
            Sym {
                name: "callee".into(),
                offset: 5,
            },
        ];
        let lines = [AddrLine { addr: 0, line: 5 }, AddrLine { addr: 5, line: 6 }];
        let out = format_module_with_lines(&code, &syms, &lines);
        assert_eq!(
            out,
            "caller:\n; line 5\n0000: call .+0  <callee>\n\ncallee:\n; line 6\n0005: ret\n"
        );
        assert!(!format_module(&code, &syms).contains("; line"));
    }
}
