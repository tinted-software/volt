//! AArch64 `ldr`/`str` → `ldp`/`stp` peephole: decode + fusion stage.
//!
//! [`decode_mem`] recognizes exactly the offset-form `ldr`/`str` encodings
//! the isel emits (the inverse of the `super::encode` offset loaders); anything
//! else yields `None`. [`try_fuse`] decides whether two adjacent decoded memory
//! ops may be combined into one `ldp`/`stp`. [`pair_memory`] is the shrinking
//! pass over an emitted word buffer that remaps the isel side tables.

use alloc::vec::Vec;

use super::encode;
use super::isel::{Fixup, LineEntry, Reloc};

/// The offset-form `ldr`/`str` encodings the aarch64 isel emits. Anything else
/// is opaque to this peephole and decodes to `None`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemKind {
    LdrX,
    StrX,
    LdrW,
    StrW,
    LdrS,
    StrS,
    LdrD,
    StrD,
    LdrQ,
    StrQ,
}

impl MemKind {
    /// Element size in bytes (also the scale of the imm12 field).
    pub fn size(self) -> u32 {
        match self {
            MemKind::LdrX | MemKind::StrX | MemKind::LdrD | MemKind::StrD => 8,
            MemKind::LdrW | MemKind::StrW | MemKind::LdrS | MemKind::StrS => 4,
            MemKind::LdrQ | MemKind::StrQ => 16,
        }
    }

    /// True for the `ldr_*` variants (a load writes `rt`; a store never does).
    pub fn is_load(self) -> bool {
        matches!(
            self,
            MemKind::LdrX | MemKind::LdrW | MemKind::LdrS | MemKind::LdrD | MemKind::LdrQ
        )
    }
}

/// A decoded memory instruction. `off` is the BYTE offset: the raw u12
/// scaled-imm field already multiplied out by the element size.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemInsn {
    pub kind: MemKind,
    pub rt: u8,
    pub rn: u8,
    pub off: u32,
}

/// Top-10-bit class selector (bits [31:22]) every recognized offset form uses.
const CLASS_MASK: u32 = 0xFFC0_0000;

/// Decode `word` if it is one of the recognized offset-form `ldr`/`str`
/// encodings, else `None`. Classes mirror `encode.zig` one-for-one.
pub fn decode_mem(word: u32) -> Option<MemInsn> {
    let rt = (word & 0x1F) as u8;
    let rn = ((word >> 5) & 0x1F) as u8;
    let imm12 = (word >> 10) & 0xFFF;
    let kind = match word & CLASS_MASK {
        0xF940_0000 => MemKind::LdrX,
        0xF900_0000 => MemKind::StrX,
        0xB940_0000 => MemKind::LdrW,
        0xB900_0000 => MemKind::StrW,
        0xBD40_0000 => MemKind::LdrS,
        0xBD00_0000 => MemKind::StrS,
        0xFD40_0000 => MemKind::LdrD,
        0xFD00_0000 => MemKind::StrD,
        0x3DC0_0000 => MemKind::LdrQ,
        0x3D80_0000 => MemKind::StrQ,
        _ => return None,
    };
    Some(MemInsn {
        kind,
        rt,
        rn,
        off: imm12 * kind.size(),
    })
}

/// Given two adjacent decoded mem insns (`a` then `b` in program order),
/// return the fused `ldp`/`stp` word if they may be paired, else `None`.
///
/// Scalar-float (`s`/`d`) and 128-bit vector (`q`) pairs decode correctly but
/// never fuse: the SIMD&FP register-pair encoders are not yet validated, so
/// they are deliberately deferred rather than emitted unconfirmed.
pub fn try_fuse(a: MemInsn, b: MemInsn) -> Option<u32> {
    if a.kind != b.kind {
        return None;
    }
    if a.rn != b.rn {
        return None;
    }
    let size = a.kind.size();
    if b.off != a.off + size {
        return None;
    }
    if a.kind.is_load() {
        if a.rt == b.rt {
            return None;
        }
        if a.rt == a.rn {
            return None;
        }
    }
    if a.off / size > 63 {
        return None;
    }
    let rt1 = encode::Reg::from_u5(a.rt as u32);
    let rt2 = encode::Reg::from_u5(b.rt as u32);
    let rn = encode::Reg::from_u5(a.rn as u32);
    let imm = a.off as i32;
    match a.kind {
        MemKind::LdrX => Some(encode::ldp_off_x(rt1, rt2, rn, imm)),
        MemKind::StrX => Some(encode::stp_off_x(rt1, rt2, rn, imm)),
        MemKind::LdrW => Some(encode::ldp_off_w(rt1, rt2, rn, imm)),
        MemKind::StrW => Some(encode::stp_off_w(rt1, rt2, rn, imm)),
        MemKind::LdrS
        | MemKind::StrS
        | MemKind::LdrD
        | MemKind::StrD
        | MemKind::LdrQ
        | MemKind::StrQ => None,
    }
}

/// Fuse adjacent `ldr`/`str` pairs into `ldp`/`stp` in `words`, updating the
/// side tables to the new layout. Only fuses immediately-adjacent words where
/// both decode, [`try_fuse`] succeeds, AND the second word is not a block
/// start. Fusion deletes the second word and replaces the first with the fused
/// pair. Mutates `words` (shrinks it) and remaps `block_starts`, `fixups`,
/// `relocs`, and `lines` in place.
pub fn pair_memory(
    words: &mut Vec<u32>,
    block_starts: &mut [usize],
    fixups: &mut [Fixup],
    relocs: &mut [Reloc],
    lines: &mut [LineEntry],
) {
    let len = words.len();
    let mut is_block_start = vec![false; len];
    for &bs in block_starts.iter() {
        debug_assert!(bs <= len);
        if bs < len {
            is_block_start[bs] = true;
        }
    }
    // `removed_before[w]` = words deleted strictly before `w`, except a deleted
    // slot records the count INCLUDING its own deletion so it collapses onto
    // the fused instruction. Length `len + 1` so a block start at `== len`
    // still has a defined slot.
    let mut removed_before = vec![0u32; len + 1];
    let mut compacted: Vec<u32> = Vec::with_capacity(len);
    let mut removed: u32 = 0;
    let mut i = 0;
    while i < len {
        let mut did_fuse = false;
        if i + 1 < len
            && !is_block_start[i + 1]
            && let (Some(a), Some(b)) = (decode_mem(words[i]), decode_mem(words[i + 1]))
            && let Some(fused) = try_fuse(a, b)
        {
            removed_before[i] = removed;
            removed += 1;
            removed_before[i + 1] = removed;
            compacted.push(fused);
            i += 2;
            did_fuse = true;
        }
        if !did_fuse {
            removed_before[i] = removed;
            compacted.push(words[i]);
            i += 1;
        }
    }
    removed_before[len] = removed;
    debug_assert_eq!(compacted.len(), len - removed as usize);
    *words = compacted;
    let write = words.len();
    for bs in block_starts.iter_mut() {
        *bs -= removed_before[*bs] as usize;
        debug_assert!(*bs <= write);
    }
    for f in fixups.iter_mut() {
        f.at -= removed_before[f.at] as usize;
        debug_assert!(f.at < write);
    }
    for r in relocs.iter_mut() {
        r.offset -= removed_before[r.offset] as usize;
        debug_assert!(r.offset < write);
    }
    for l in lines.iter_mut() {
        // Line offsets are BYTES; a line on a deleted word collapses onto the
        // fused instruction, which is acceptable for debug info.
        let w = (l.offset / 4) as usize;
        l.offset = ((w - removed_before[w] as usize) * 4) as u32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_round_trip_against_encode() {
        let r = encode::Reg::from_u5;
        assert_eq!(
            decode_mem(encode::ldr_off(r(0), r(1), 16)),
            Some(MemInsn {
                kind: MemKind::LdrX,
                rt: 0,
                rn: 1,
                off: 16
            })
        );
        assert_eq!(
            decode_mem(encode::str_off(r(2), r(3), 8)),
            Some(MemInsn {
                kind: MemKind::StrX,
                rt: 2,
                rn: 3,
                off: 8
            })
        );
        assert_eq!(
            decode_mem(encode::ldr_w(r(4), r(5), 12)),
            Some(MemInsn {
                kind: MemKind::LdrW,
                rt: 4,
                rn: 5,
                off: 12
            })
        );
        assert_eq!(
            decode_mem(encode::str_w(r(6), r(7), 4)),
            Some(MemInsn {
                kind: MemKind::StrW,
                rt: 6,
                rn: 7,
                off: 4
            })
        );
        assert_eq!(
            decode_mem(encode::ldr_fp(r(0), r(1), 8, false)),
            Some(MemInsn {
                kind: MemKind::LdrS,
                rt: 0,
                rn: 1,
                off: 8
            })
        );
        assert_eq!(
            decode_mem(encode::str_fp(r(2), r(3), 4, false)),
            Some(MemInsn {
                kind: MemKind::StrS,
                rt: 2,
                rn: 3,
                off: 4
            })
        );
        assert_eq!(
            decode_mem(encode::ldr_fp(r(4), r(5), 16, true)),
            Some(MemInsn {
                kind: MemKind::LdrD,
                rt: 4,
                rn: 5,
                off: 16
            })
        );
        assert_eq!(
            decode_mem(encode::str_fp(r(6), r(7), 8, true)),
            Some(MemInsn {
                kind: MemKind::StrD,
                rt: 6,
                rn: 7,
                off: 8
            })
        );
        assert_eq!(
            decode_mem(encode::ldr_q(r(0), r(1), 32)),
            Some(MemInsn {
                kind: MemKind::LdrQ,
                rt: 0,
                rn: 1,
                off: 32
            })
        );
        assert_eq!(
            decode_mem(encode::str_q(r(2), r(3), 16)),
            Some(MemInsn {
                kind: MemKind::StrQ,
                rt: 2,
                rn: 3,
                off: 16
            })
        );
        assert_eq!(decode_mem(encode::nop()), None);
        assert_eq!(decode_mem(0xF840_0400), None);
    }

    #[test]
    fn fuse_round_trip_x_loads() {
        let a = MemInsn {
            kind: MemKind::LdrX,
            rt: 0,
            rn: 2,
            off: 0,
        };
        let b = MemInsn {
            kind: MemKind::LdrX,
            rt: 1,
            rn: 2,
            off: 8,
        };
        let r = encode::Reg::from_u5;
        assert_eq!(try_fuse(a, b), Some(encode::ldp_off_x(r(0), r(1), r(2), 0)));
    }

    #[test]
    fn fuse_rejects_fp_kinds() {
        let s1 = MemInsn {
            kind: MemKind::LdrS,
            rt: 0,
            rn: 1,
            off: 0,
        };
        let s2 = MemInsn {
            kind: MemKind::LdrS,
            rt: 2,
            rn: 1,
            off: 4,
        };
        assert_eq!(try_fuse(s1, s2), None);
    }

    #[test]
    fn pair_memory_fuses_and_shrinks() {
        let r = encode::Reg::from_u5;
        let mut words = vec![
            encode::ldr_off(r(0), r(2), 0),
            encode::ldr_off(r(1), r(2), 8),
            encode::ret(),
        ];
        let mut block_starts = vec![0];
        pair_memory(&mut words, &mut block_starts, &mut [], &mut [], &mut []);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0], encode::ldp_off_x(r(0), r(1), r(2), 0));
        assert_eq!(words[1], encode::ret());
        assert_eq!(decode_mem(words[0]), None);
    }

    #[test]
    fn pair_memory_respects_block_boundary() {
        let r = encode::Reg::from_u5;
        let original = vec![
            encode::ldr_off(r(0), r(2), 0),
            encode::ldr_off(r(1), r(2), 8),
            encode::ret(),
        ];
        let mut words = original.clone();
        let mut block_starts = vec![0, 1];
        pair_memory(&mut words, &mut block_starts, &mut [], &mut [], &mut []);
        assert_eq!(words, original);
        assert_eq!(block_starts, vec![0, 1]);
    }

    #[test]
    fn pair_memory_remaps_fixup_and_block_start() {
        let r = encode::Reg::from_u5;
        let mut words = vec![
            encode::ldr_off(r(0), r(2), 0),
            encode::ldr_off(r(1), r(2), 8),
            encode::b(0),
            encode::ret(),
        ];
        let mut block_starts = vec![0, 3];
        let mut fixups = vec![Fixup { at: 2, target: 1 }];
        pair_memory(&mut words, &mut block_starts, &mut fixups, &mut [], &mut []);
        assert_eq!(words.len(), 3);
        assert_eq!(fixups[0].at, 1);
        assert_eq!(fixups[0].target, 1);
        assert_eq!(block_starts, vec![0, 2]);
    }

    #[test]
    fn pair_memory_noop_leaves_tables_untouched() {
        let r = encode::Reg::from_u5;
        let original = vec![
            encode::ldr_off(r(0), r(2), 0),
            encode::ldr_off(r(1), r(2), 16),
            encode::ret(),
        ];
        let mut words = original.clone();
        let mut block_starts = vec![0];
        let mut fixups = vec![Fixup { at: 2, target: 0 }];
        let mut relocs = vec![Reloc {
            offset: 2,
            symbol: "x".to_string(),
            kind: crate::aarch64::isel::RelocKind::Call,
        }];
        let mut lines = vec![LineEntry { offset: 8, line: 7 }];
        pair_memory(
            &mut words,
            &mut block_starts,
            &mut fixups,
            &mut relocs,
            &mut lines,
        );
        assert_eq!(words, original);
        assert_eq!(fixups[0].at, 2);
        assert_eq!(relocs[0].offset, 2);
        assert_eq!(lines[0].offset, 8);
    }

    #[test]
    fn fuse_rejects_bad_pairs() {
        let base = MemInsn {
            kind: MemKind::LdrX,
            rt: 0,
            rn: 2,
            off: 0,
        };
        assert_eq!(
            try_fuse(
                base,
                MemInsn {
                    kind: MemKind::LdrX,
                    rt: 1,
                    rn: 2,
                    off: 16
                }
            ),
            None
        );
        assert_eq!(
            try_fuse(
                base,
                MemInsn {
                    kind: MemKind::LdrX,
                    rt: 1,
                    rn: 3,
                    off: 8
                }
            ),
            None
        );
        let clobbers = MemInsn {
            kind: MemKind::LdrX,
            rt: 2,
            rn: 2,
            off: 0,
        };
        assert_eq!(
            try_fuse(
                clobbers,
                MemInsn {
                    kind: MemKind::LdrX,
                    rt: 1,
                    rn: 2,
                    off: 8
                }
            ),
            None
        );
        assert_eq!(
            try_fuse(
                base,
                MemInsn {
                    kind: MemKind::LdrX,
                    rt: 0,
                    rn: 2,
                    off: 8
                }
            ),
            None
        );
        assert_eq!(
            try_fuse(
                base,
                MemInsn {
                    kind: MemKind::LdrW,
                    rt: 1,
                    rn: 2,
                    off: 8
                }
            ),
            None
        );
        let far = MemInsn {
            kind: MemKind::LdrX,
            rt: 0,
            rn: 2,
            off: 512,
        };
        assert_eq!(
            try_fuse(
                far,
                MemInsn {
                    kind: MemKind::LdrX,
                    rt: 1,
                    rn: 2,
                    off: 520
                }
            ),
            None
        );
        let store_base = MemInsn {
            kind: MemKind::StrX,
            rt: 2,
            rn: 2,
            off: 0,
        };
        assert!(
            try_fuse(
                store_base,
                MemInsn {
                    kind: MemKind::StrX,
                    rt: 3,
                    rn: 2,
                    off: 8
                }
            )
            .is_some()
        );
    }
}
