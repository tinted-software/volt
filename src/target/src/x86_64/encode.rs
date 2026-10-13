// SPDX-License-Identifier: Apache-2.0
// Copyright LilithSemi (Vulcan). Modified for the Volt Rust port.
//! x86-64 instruction encoding. Register numbers follow the architectural encoding.
use alloc::vec::Vec;

pub type Reg = u8;
pub const RAX: Reg = 0;
pub const RCX: Reg = 1;
pub const RDX: Reg = 2;
pub const RBX: Reg = 3;
pub const RSP: Reg = 4;
pub const RBP: Reg = 5;
pub const RSI: Reg = 6;
pub const RDI: Reg = 7;
pub const R8: Reg = 8;
pub const R9: Reg = 9;
pub const R10: Reg = 10;
pub const R11: Reg = 11;

#[derive(Default)]
pub struct Encoder {
    pub code: Vec<u8>,
}
impl Encoder {
    pub fn bytes(&mut self, bytes: &[u8]) {
        self.code.extend_from_slice(bytes);
    }
    pub fn rex(&mut self, wide: bool, reg: Reg, base: Reg, byte: bool) {
        let r = 0x40 | ((wide as u8) << 3) | ((reg >> 3) << 2) | (base >> 3);
        if r != 0x40 || byte {
            self.code.push(r);
        }
    }
    pub fn rr(&mut self, prefix: Option<u8>, op: &[u8], reg: Reg, rm: Reg, wide: bool) {
        if let Some(p) = prefix {
            self.code.push(p);
        }
        self.rex(wide, reg, rm, false);
        self.bytes(op);
        self.code.push(0xc0 | ((reg & 7) << 3) | (rm & 7));
    }
    pub fn mem(
        &mut self,
        prefix: Option<u8>,
        op: &[u8],
        reg: Reg,
        base: Reg,
        off: i32,
        wide: bool,
        byte: bool,
    ) {
        if let Some(p) = prefix {
            self.code.push(p);
        }
        self.rex(wide, reg, base, byte);
        self.bytes(op);
        self.code.push(0x80 | ((reg & 7) << 3) | (base & 7));
        if base & 7 == 4 {
            self.code.push(0x24);
        }
        self.bytes(&off.to_le_bytes());
    }
    pub fn mov(&mut self, dst: Reg, src: Reg) {
        if dst != src {
            self.rr(None, &[0x89], src, dst, true);
        }
    }
    pub fn imm(&mut self, dst: Reg, imm: u64) {
        self.rex(true, 0, dst, false);
        self.code.push(0xb8 | (dst & 7));
        self.bytes(&imm.to_le_bytes());
    }
    /// Loads `bits` of memory into `dst`, zero-extending to 64 bits. Sign extension is an
    /// explicit operation of the instruction selector, never part of a load.
    pub fn load(&mut self, dst: Reg, base: Reg, off: i32, bits: u16) {
        match bits {
            1..=8 => self.mem(None, &[0x0f, 0xb6], dst, base, off, true, false),
            9..=16 => self.mem(None, &[0x0f, 0xb7], dst, base, off, true, false),
            17..=32 => self.mem(None, &[0x8b], dst, base, off, false, false),
            _ => self.mem(None, &[0x8b], dst, base, off, true, false),
        }
    }
    pub fn store(&mut self, base: Reg, off: i32, src: Reg, bits: u16) {
        self.mem(
            if bits > 8 && bits <= 16 {
                Some(0x66)
            } else {
                None
            },
            &[if bits <= 8 { 0x88 } else { 0x89 }],
            src,
            base,
            off,
            bits > 32,
            bits <= 8,
        );
    }
    /// `xchg [base + off], src` of `bits` (8, 16, 32 or 64): swaps `src` with memory. With a
    /// memory operand the exchange carries an implicit `lock`, so it is a full barrier.
    pub fn xchg(&mut self, base: Reg, off: i32, src: Reg, bits: u16) {
        self.mem(
            if bits > 8 && bits <= 16 {
                Some(0x66)
            } else {
                None
            },
            &[if bits <= 8 { 0x86 } else { 0x87 }],
            src,
            base,
            off,
            bits > 32,
            bits <= 8,
        );
    }
    pub fn push(&mut self, reg: Reg) {
        self.rex(false, 0, reg, false);
        self.code.push(0x50 | (reg & 7));
    }
    pub fn pop(&mut self, reg: Reg) {
        self.rex(false, 0, reg, false);
        self.code.push(0x58 | (reg & 7));
    }
    pub fn alu_imm(&mut self, digit: u8, reg: Reg, imm: i32, wide: bool) {
        self.rr(None, &[0x81], digit, reg, wide);
        self.bytes(&imm.to_le_bytes());
    }
    pub fn shift(&mut self, digit: u8, reg: Reg, imm: Option<u8>, wide: bool) {
        self.rr(
            None,
            &[if imm.is_some() { 0xc1 } else { 0xd3 }],
            digit,
            reg,
            wide,
        );
        if let Some(i) = imm {
            self.code.push(i);
        }
    }
    pub fn setcc(&mut self, cc: u8, reg: Reg) {
        self.rex(false, 0, reg, true);
        self.bytes(&[0x0f, 0x90 | cc, 0xc0 | (reg & 7)]);
        self.rr(None, &[0x0f, 0xb6], reg, reg, false);
    }
    pub fn branch(&mut self, cc: Option<u8>) -> usize {
        if let Some(c) = cc {
            self.bytes(&[0x0f, 0x80 | c]);
        } else {
            self.code.push(0xe9);
        }
        let at = self.code.len();
        self.bytes(&[0; 4]);
        at
    }
    pub fn patch(&mut self, at: usize, target: usize) -> Result<(), super::Error> {
        let rel = i32::try_from(target as i64 - (at + 4) as i64)
            .map_err(|_| super::Error::RelocationOutOfRange)?;
        self.code[at..at + 4].copy_from_slice(&rel.to_le_bytes());
        Ok(())
    }
    pub fn xmm_mov(&mut self, dst: u8, src: u8, wide: bool) {
        if dst != src {
            if wide {
                self.vex_rr(1, 0, 0x10, dst, 0, src, true);
            } else {
                self.rr(None, &[0x0f, 0x10], dst, src, false);
            }
        }
    }
    pub fn xmm_mem(&mut self, reg: u8, base: Reg, off: i32, bytes: usize, store: bool) {
        if bytes == 32 {
            self.vex(1, 0, reg, 0, base, true);
            self.code.push(if store { 0x11 } else { 0x10 });
            self.code.push(0x80 | ((reg & 7) << 3) | (base & 7));
            if base & 7 == 4 {
                self.code.push(0x24);
            }
            self.bytes(&off.to_le_bytes());
        } else {
            self.mem(
                match bytes {
                    4 => Some(0xf3),
                    8 => Some(0xf2),
                    _ => None,
                },
                &[0x0f, if store { 0x11 } else { 0x10 }],
                reg,
                base,
                off,
                false,
                false,
            );
        }
    }
    pub fn vex(&mut self, map: u8, pp: u8, dst: u8, lhs: u8, rhs: u8, wide: bool) {
        self.bytes(&[
            0xc4,
            ((!dst & 8) << 4) | 0x40 | ((!rhs & 8) << 2) | map,
            ((!lhs & 15) << 3) | ((wide as u8) << 2) | pp,
        ]);
    }
    pub fn vex_rr(&mut self, map: u8, pp: u8, op: u8, dst: u8, lhs: u8, rhs: u8, wide: bool) {
        self.vex(map, pp, dst, lhs, rhs, wide);
        self.bytes(&[op, 0xc0 | ((dst & 7) << 3) | (rhs & 7)]);
    }
    pub fn half_round(&mut self, reg: u8) {
        self.vex_rr(3, 1, 0x1d, reg, 0, reg, false);
        self.code.push(0);
        self.vex_rr(2, 1, 0x13, reg, 0, reg, false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn high_register_and_exact_store_width() {
        let mut e = Encoder::default();
        e.imm(R11, 0x123456789abcdef0);
        e.store(R10, 7, R11, 8);
        assert_eq!(
            e.code,
            &[
                0x49, 0xbb, 0xf0, 0xde, 0xbc, 0x9a, 0x78, 0x56, 0x34, 0x12, 0x45, 0x88, 0x9a, 7, 0,
                0, 0
            ]
        );
    }
    #[test]
    fn backward_branch() {
        let mut e = Encoder::default();
        let at = e.branch(None);
        e.patch(at, 0).unwrap();
        assert_eq!(e.code, &[0xe9, 0xfb, 0xff, 0xff, 0xff]);
    }
}
