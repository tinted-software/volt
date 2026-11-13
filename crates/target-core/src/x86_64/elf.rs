// SPDX-License-Identifier: Apache-2.0
// Copyright LilithSemi (Vulcan). Modified for the Volt Rust port.
//! Static Linux ELF64 executable writer with a single read/execute PT_LOAD.
use alloc::{vec, vec::Vec};
pub const LOAD_ADDR: u64 = 0x1000_0000;
pub const CODE_OFFSET: usize = 120;
pub fn write_exec(code: &[u8], entry_offset: usize) -> Vec<u8> {
    let total = CODE_OFFSET + code.len();
    let mut buf = vec![0; total];
    buf[..4].copy_from_slice(b"\x7fELF");
    buf[4] = 2;
    buf[5] = 1;
    buf[6] = 1;
    fn put(buf: &mut [u8], off: usize, value: u64, len: usize) {
        buf[off..off + len].copy_from_slice(&value.to_le_bytes()[..len]);
    }
    put(&mut buf, 16, 2, 2);
    put(&mut buf, 18, 62, 2);
    put(&mut buf, 20, 1, 4);
    put(
        &mut buf,
        24,
        LOAD_ADDR + CODE_OFFSET as u64 + entry_offset as u64,
        8,
    );
    put(&mut buf, 32, 64, 8);
    put(&mut buf, 52, 64, 2);
    put(&mut buf, 54, 56, 2);
    put(&mut buf, 56, 1, 2);
    put(&mut buf, 64, 1, 4);
    put(&mut buf, 68, 5, 4);
    put(&mut buf, 80, LOAD_ADDR, 8);
    put(&mut buf, 88, LOAD_ADDR, 8);
    put(&mut buf, 96, total as u64, 8);
    put(&mut buf, 104, total as u64, 8);
    put(&mut buf, 112, 0x1000, 8);
    buf[CODE_OFFSET..].copy_from_slice(code);
    buf
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn executable_header() {
        let elf = write_exec(&[0xc3, 0x90], 0);
        assert_eq!(&elf[..4], b"\x7fELF");
        assert_eq!(u16::from_le_bytes(elf[18..20].try_into().unwrap()), 62);
        assert_eq!(
            u64::from_le_bytes(elf[24..32].try_into().unwrap()),
            LOAD_ADDR + CODE_OFFSET as u64
        );
        assert_eq!(&elf[CODE_OFFSET..], &[0xc3, 0x90]);
    }
}
