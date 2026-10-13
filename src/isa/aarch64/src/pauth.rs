//! Pointer authentication (FEAT_PAuth) instructions: the decoded shape of every
//! word [`decode`] recognises, including the hint-space forms (`PACIASP`, ...).
//!
//! The cipher and what authentication does to a pointer are the emulator's business.

/// The key the instruction uses. `Ga` is the generic key, used only by `PACGA`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyId {
    Ia,
    Ib,
    Da,
    Db,
    Ga,
}

impl KeyId {
    /// The `SCTLR_EL1` bit that enables the key, if any. The generic key has none.
    pub fn enable_bit(self) -> Option<u32> {
        match self {
            Self::Ia => Some(31),
            Self::Ib => Some(30),
            Self::Da => Some(27),
            Self::Db => Some(13),
            Self::Ga => None,
        }
    }

    /// The `keynumber` the architecture reports in an authentication failure.
    pub fn number(self) -> u64 {
        match self {
            Self::Ia => 0,
            Self::Ib => 1,
            Self::Da => 2,
            Self::Db => 3,
            Self::Ga => 4,
        }
    }
}

/// The modifier operand: zero, or a register where register 31 is `SP`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Modifier {
    Zero,
    Reg(u8),
}

/// One PAuth instruction, decoded.
///
/// `hint` marks the hint-space spellings (`PACIASP`, `PACIA1716`, `PACIAZ`, `XPACLRI`, ...),
/// which execute as no-ops without PAuth; their register and modifier are fixed by the
/// opcode (`LR` signed with `SP`, `X17` with `X16`, `LR` with zero, `LR` stripped).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// `PACIA`, `PACIZA`, `PACDA`, ... and `PACIASP`, `PACIA1716`, `PACIAZ`, ...
    Sign {
        key: KeyId,
        reg: u8,
        modifier: Modifier,
        hint: bool,
    },
    /// `AUTIA`, `AUTIZA`, `AUTDA`, ... and `AUTIASP`, `AUTIA1716`, `AUTIAZ`, ...
    Auth {
        key: KeyId,
        reg: u8,
        modifier: Modifier,
        hint: bool,
    },
    /// `XPACI` (`data` clear), `XPACD` (`data` set) and the hint `XPACLRI`.
    Strip {
        data: bool,
        reg: u8,
        hint: bool,
    },
    Pacga {
        rd: u8,
        rn: u8,
        modifier: Modifier,
    },
    /// `BRAA`, `BRAB`, `BLRAA`, `BLRAB` and their zero-modifier forms.
    Branch {
        key: KeyId,
        link: bool,
        target: u8,
        modifier: Modifier,
    },
    /// `RETAA` and `RETAB`: return to `LR`, authenticated with `SP`.
    Return {
        key: KeyId,
    },
    /// `ERETAA` and `ERETAB`: return from the exception, authenticating `ELR_EL1`.
    ExceptionReturn {
        key: KeyId,
    },
}

/// The hint-space forms (`PACIASP` and friends), which are no-ops without PAuth.
fn hint(word: u32) -> Option<Op> {
    if word & 0xffff_f01f != 0xd503_201f {
        return None;
    }
    let hint = (word >> 5) & 0x7f;
    // Bit 2 selects authenticate over sign, bit 1 the B key over the A key.
    let key = if hint & 2 != 0 { KeyId::Ib } else { KeyId::Ia };
    let (reg, modifier) = match hint {
        7 => {
            return Some(Op::Strip {
                data: false,
                reg: 30,
                hint: true,
            });
        }
        // PACIA1716, PACIB1716, AUTIA1716, AUTIB1716.
        8 | 10 | 12 | 14 => (17, Modifier::Reg(16)),
        // PAC/AUT{IA,IB}Z (even) and PAC/AUT{IA,IB}SP (odd).
        24..=31 if hint & 1 == 0 => (30, Modifier::Zero),
        24..=31 => (30, Modifier::Reg(31)),
        _ => return None,
    };
    Some(if hint & 4 != 0 {
        Op::Auth {
            key,
            reg,
            modifier,
            hint: true,
        }
    } else {
        Op::Sign {
            key,
            reg,
            modifier,
            hint: true,
        }
    })
}

/// Decode `word` as a pointer-authentication instruction, or `None` when it is not one.
pub fn decode(word: u32) -> Option<Op> {
    let rd = (word & 31) as u8;
    let rn = ((word >> 5) & 31) as u8;
    let rm = rd;
    // Data-processing, one source: PAC and AUT (opcodes 0-15), and XPAC (16-17).
    if word & 0xffff_0000 == 0xdac1_0000 {
        let opcode = (word >> 10) & 0x3f;
        let key = [KeyId::Ia, KeyId::Ib, KeyId::Da, KeyId::Db][(opcode & 3) as usize];
        // The zero-modifier forms have no modifier register: Rn must be 31.
        let modifier = if opcode & 8 != 0 {
            if rn != 31 {
                return None;
            }
            Modifier::Zero
        } else {
            Modifier::Reg(rn)
        };
        return match opcode {
            0..=15 if opcode & 4 == 0 => Some(Op::Sign {
                key,
                reg: rd,
                modifier,
                hint: false,
            }),
            0..=15 => Some(Op::Auth {
                key,
                reg: rd,
                modifier,
                hint: false,
            }),
            16 | 17 if rn == 31 => Some(Op::Strip {
                data: opcode == 17,
                reg: rd,
                hint: false,
            }),
            _ => None,
        };
    }
    // Data-processing, two sources: PACGA Xd, Xn, Xm|SP.
    if word & 0xffe0_fc00 == 0x9ac0_3000 {
        let rm = ((word >> 16) & 31) as u8;
        return Some(Op::Pacga {
            rd,
            rn,
            modifier: Modifier::Reg(rm),
        });
    }
    // Branches with authentication. Bit 10 selects the B key; bit 11 is always set.
    let b_key = if word & 0x400 != 0 {
        KeyId::Ib
    } else {
        KeyId::Ia
    };
    if word & 0xffff_f800 == 0xd71f_0800 {
        return Some(Op::Branch {
            key: b_key,
            link: false,
            target: rn,
            modifier: Modifier::Reg(rm),
        });
    }
    if word & 0xffff_f81f == 0xd61f_081f {
        return Some(Op::Branch {
            key: b_key,
            link: false,
            target: rn,
            modifier: Modifier::Zero,
        });
    }
    if word & 0xffff_f800 == 0xd73f_0800 {
        return Some(Op::Branch {
            key: b_key,
            link: true,
            target: rn,
            modifier: Modifier::Reg(rm),
        });
    }
    if word & 0xffff_f81f == 0xd63f_081f {
        return Some(Op::Branch {
            key: b_key,
            link: true,
            target: rn,
            modifier: Modifier::Zero,
        });
    }
    if word & 0xffff_fbff == 0xd65f_0bff {
        return Some(Op::Return { key: b_key });
    }
    if word & 0xffff_fbff == 0xd69f_0bff {
        return Some(Op::ExceptionReturn { key: b_key });
    }
    hint(word)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sign(key: KeyId, reg: u8, modifier: Modifier, hint: bool) -> Option<Op> {
        Some(Op::Sign {
            key,
            reg,
            modifier,
            hint,
        })
    }

    fn auth(key: KeyId, reg: u8, modifier: Modifier, hint: bool) -> Option<Op> {
        Some(Op::Auth {
            key,
            reg,
            modifier,
            hint,
        })
    }

    #[test]
    fn the_instruction_encodings_decode_as_their_names() {
        // Assembled with GNU as for armv8.3-a.
        assert_eq!(
            decode(0xdac1_0020),
            sign(KeyId::Ia, 0, Modifier::Reg(1), false)
        );
        assert_eq!(
            decode(0xdac1_11ac),
            auth(KeyId::Ia, 12, Modifier::Reg(13), false)
        );
        assert_eq!(
            decode(0xdac1_23f0),
            sign(KeyId::Ia, 16, Modifier::Zero, false)
        );
        assert_eq!(
            decode(0xdac1_43f8),
            Some(Op::Strip {
                data: false,
                reg: 24,
                hint: false
            })
        );
        assert_eq!(
            decode(0xd71f_0c43),
            Some(Op::Branch {
                key: KeyId::Ib,
                link: false,
                target: 2,
                modifier: Modifier::Reg(3)
            })
        );
        assert_eq!(
            decode(0xd63f_095f),
            Some(Op::Branch {
                key: KeyId::Ia,
                link: true,
                target: 10,
                modifier: Modifier::Zero
            })
        );
        assert_eq!(decode(0xd65f_0fff), Some(Op::Return { key: KeyId::Ib }));
        assert_eq!(
            decode(0xd69f_0bff),
            Some(Op::ExceptionReturn { key: KeyId::Ia })
        );
        // Plain BR and ordinary one-source data processing are not PAuth.
        assert_eq!(decode(0xd61f_0000), None);
        assert_eq!(decode(0xdac0_0000), None);
    }

    #[test]
    fn xpaci_and_xpacd_are_distinct() {
        // `xpaci x0`, `xpacd x0`, `xpaci x30`, `xpacd x30`.
        let strip = |data, reg, hint| Some(Op::Strip { data, reg, hint });
        assert_eq!(decode(0xdac1_43e0), strip(false, 0, false));
        assert_eq!(decode(0xdac1_47e0), strip(true, 0, false));
        assert_eq!(decode(0xdac1_43fe), strip(false, 30, false));
        assert_eq!(decode(0xdac1_47fe), strip(true, 30, false));
        // The hint `xpaclri` is its own spelling of stripping LR.
        assert_eq!(decode(0xd503_20ff), strip(false, 30, true));
    }

    #[test]
    fn hint_space_forms_differ_from_the_register_forms() {
        use KeyId::{Ia, Ib};
        use Modifier::{Reg, Zero};
        // paciaz, paciasp, pacibz, pacibsp, autiaz, autiasp, autibz, autibsp.
        assert_eq!(decode(0xd503_231f), sign(Ia, 30, Zero, true));
        assert_eq!(decode(0xd503_233f), sign(Ia, 30, Reg(31), true));
        assert_eq!(decode(0xd503_235f), sign(Ib, 30, Zero, true));
        assert_eq!(decode(0xd503_237f), sign(Ib, 30, Reg(31), true));
        assert_eq!(decode(0xd503_239f), auth(Ia, 30, Zero, true));
        assert_eq!(decode(0xd503_23bf), auth(Ia, 30, Reg(31), true));
        assert_eq!(decode(0xd503_23df), auth(Ib, 30, Zero, true));
        assert_eq!(decode(0xd503_23ff), auth(Ib, 30, Reg(31), true));
        // pacia1716, pacib1716, autia1716, autib1716.
        assert_eq!(decode(0xd503_211f), sign(Ia, 17, Reg(16), true));
        assert_eq!(decode(0xd503_215f), sign(Ib, 17, Reg(16), true));
        assert_eq!(decode(0xd503_219f), auth(Ia, 17, Reg(16), true));
        assert_eq!(decode(0xd503_21df), auth(Ib, 17, Reg(16), true));
        // The register forms with the same operands: `pacia x30, sp`, `pacia x17, x16`,
        // `paciza x30`, `autia x30, sp`.
        assert_eq!(decode(0xdac1_03fe), sign(Ia, 30, Reg(31), false));
        assert_eq!(decode(0xdac1_0211), sign(Ia, 17, Reg(16), false));
        assert_eq!(decode(0xdac1_23fe), sign(Ia, 30, Zero, false));
        assert_eq!(decode(0xdac1_13fe), auth(Ia, 30, Reg(31), false));
        // Other hint numbers are not PAuth.
        assert_eq!(decode(0xd503_203f), None);
        assert_eq!(decode(0xd503_213f), None);
        assert_eq!(decode(0xd503_243f), None);
    }

    #[test]
    fn zero_modifier_forms_need_register_31() {
        // `paciza x0` with Rn = 1 is unallocated (objdump: `.inst ... ; undefined`).
        assert_eq!(decode(0xdac1_2020), None);
        assert_eq!(decode(0xdac1_3020), None);
        // XPAC likewise needs Rn = 31.
        assert_eq!(decode(0xdac1_4020), None);
    }
}
