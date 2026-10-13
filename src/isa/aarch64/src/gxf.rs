//! Guarded execution (GXF): `genter` and `gexit`, the Apple instructions that move
//! between EL and the guarded levels GL1 and GL2 (Asahi Linux documentation, "Apple
//! Proprietary Instructions" and "SPRR and GXF").

/// `gexit`: leave guarded mode.
const GEXIT: u32 = 0x0020_1400;
/// `genter` with its `imm5` in bits 4:0 (stored in `ESR_GLx[5:0]`).
const GENTER: u32 = 0x0020_1420;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Gexit,
    Genter { imm5: u8 },
}

/// Decode `word` as `genter` or `gexit`, or `None` when it is neither.
pub fn decode(word: u32) -> Option<Op> {
    if word == GEXIT {
        Some(Op::Gexit)
    } else if word & !0x1f == GENTER {
        Some(Op::Genter {
            imm5: (word & 0x1f) as u8,
        })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn genter_and_gexit_are_recognised_with_any_genter_immediate() {
        assert_eq!(decode(GEXIT), Some(Op::Gexit));
        assert_eq!(decode(GENTER), Some(Op::Genter { imm5: 0 }));
        assert_eq!(decode(GENTER | 0x1f), Some(Op::Genter { imm5: 31 }));
        assert_eq!(decode(0xd503_201f), None);
        assert_eq!(decode(GEXIT + 4), None);
    }
}
