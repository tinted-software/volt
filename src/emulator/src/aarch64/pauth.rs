//! Pointer authentication (FEAT_PAuth): the QARMA5 `ComputePAC` the architecture
//! defines, the sign, authenticate and strip operations on 48-bit pointers, and the
//! PAuth instructions the JIT hands to the host ([`super::host`]).
//!
//! The emulated CPU advertises the architected QARMA5 algorithm without TBI, EPAC or
//! FPAC (`ID_AA64ISAR1_EL1.APA` = 1). A pointer's PAC then occupies bits 63:56 and 54:48,
//! and authentication failure corrupts bits 62:61 instead of faulting.
//!
//! The cipher follows the pseudocode in the architecture as ported from QEMU's
//! `target/arm/tcg/pauth_helper.c`; the tests check it against that implementation.

use super::{Cpu, exception};

/// A 128-bit PAuth key: `hi` is bits 127:64 (`APxAKeyHi_EL1`), `lo` bits 63:0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Key {
    pub hi: u64,
    pub lo: u64,
}

/// The key the instruction uses, as an index into [`super::cpu::System::pac_keys`]
/// (two words per key, low word first).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyId {
    Ia,
    Ib,
    Da,
    Db,
    Ga,
}

impl KeyId {
    fn index(self) -> usize {
        match self {
            Self::Ia => 0,
            Self::Ib => 1,
            Self::Da => 2,
            Self::Db => 3,
            Self::Ga => 4,
        }
    }

    /// The `SCTLR_EL1` bit that enables the key, if any. The generic key has none.
    fn enable_bit(self) -> Option<u32> {
        match self {
            Self::Ia => Some(31),
            Self::Ib => Some(30),
            Self::Da => Some(27),
            Self::Db => Some(13),
            Self::Ga => None,
        }
    }

    /// The `keynumber` the architecture reports in an authentication failure.
    fn number(self) -> u64 {
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    Sign {
        key: KeyId,
        reg: u8,
        modifier: Modifier,
    },
    Auth {
        key: KeyId,
        reg: u8,
        modifier: Modifier,
    },
    Strip {
        reg: u8,
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
const HINTS: [(u32, Op); 13] = [
    (
        0xd503_233f,
        Op::Sign {
            key: KeyId::Ia,
            reg: 30,
            modifier: Modifier::Reg(31),
        },
    ),
    (
        0xd503_23bf,
        Op::Auth {
            key: KeyId::Ia,
            reg: 30,
            modifier: Modifier::Reg(31),
        },
    ),
    (
        0xd503_237f,
        Op::Sign {
            key: KeyId::Ib,
            reg: 30,
            modifier: Modifier::Reg(31),
        },
    ),
    (
        0xd503_23ff,
        Op::Auth {
            key: KeyId::Ib,
            reg: 30,
            modifier: Modifier::Reg(31),
        },
    ),
    (
        0xd503_211f,
        Op::Sign {
            key: KeyId::Ia,
            reg: 17,
            modifier: Modifier::Reg(16),
        },
    ),
    (
        0xd503_219f,
        Op::Auth {
            key: KeyId::Ia,
            reg: 17,
            modifier: Modifier::Reg(16),
        },
    ),
    (
        0xd503_215f,
        Op::Sign {
            key: KeyId::Ib,
            reg: 17,
            modifier: Modifier::Reg(16),
        },
    ),
    (
        0xd503_21df,
        Op::Auth {
            key: KeyId::Ib,
            reg: 17,
            modifier: Modifier::Reg(16),
        },
    ),
    (
        0xd503_231f,
        Op::Sign {
            key: KeyId::Ia,
            reg: 30,
            modifier: Modifier::Zero,
        },
    ),
    (
        0xd503_239f,
        Op::Auth {
            key: KeyId::Ia,
            reg: 30,
            modifier: Modifier::Zero,
        },
    ),
    (
        0xd503_235f,
        Op::Sign {
            key: KeyId::Ib,
            reg: 30,
            modifier: Modifier::Zero,
        },
    ),
    (
        0xd503_23df,
        Op::Auth {
            key: KeyId::Ib,
            reg: 30,
            modifier: Modifier::Zero,
        },
    ),
    (0xd503_20ff, Op::Strip { reg: 30 }),
];

fn decode_word(word: u32) -> Option<Op> {
    let rd = (word & 31) as u8;
    let rn = ((word >> 5) & 31) as u8;
    let rm = rd;
    // Data-processing, one source: PAC and AUT (opcodes 0-15), and XPAC (16-17).
    if word & 0xffff_0000 == 0xdac1_0000 {
        let opcode = (word >> 10) & 0x3f;
        let key = [KeyId::Ia, KeyId::Ib, KeyId::Da, KeyId::Db][(opcode & 3) as usize];
        let modifier = if opcode & 8 != 0 {
            Modifier::Zero
        } else {
            Modifier::Reg(rn)
        };
        return match opcode {
            0..=15 if opcode & 4 == 0 => Some(Op::Sign {
                key,
                reg: rd,
                modifier,
            }),
            0..=15 => Some(Op::Auth {
                key,
                reg: rd,
                modifier,
            }),
            16 | 17 if rn == 31 => Some(Op::Strip { reg: rd }),
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
    HINTS
        .iter()
        .find(|(hint, _)| *hint == word)
        .map(|&(_, op)| op)
}

/// Whether [`run`] implements `word`.
pub fn is_supported(word: u32) -> bool {
    decode_word(word).is_some()
}

/// The 48-bit pointer's bits 63:48 all equal bit 47, so the pointer is canonical.
fn is_canonical(ptr: u64) -> bool {
    let top = ptr >> 48;
    top == 0 || top == 0xffff
}

/// Bit 55 selects the address range. A canonical pointer's bits 63:48 equal it.
fn original_ptr(ptr: u64) -> u64 {
    if ptr & (1 << 55) != 0 {
        ptr | 0xffff_0000_0000_0000
    } else {
        ptr & 0x0000_ffff_ffff_ffff
    }
}

/// The PAC of a pointer: the `ComputePAC` of its extension bits filled in.
pub fn add_pac(ptr: u64, modifier: u64, key: Key) -> u64 {
    let ext = (ptr >> 63) & 1;
    let ext_ptr = (ptr & 0x0000_ffff_ffff_ffff)
        | if (ptr >> 63) & 1 != 0 {
            0xffff_0000_0000_0000
        } else {
            0
        };
    let mut pac = compute_pac(ext_ptr, modifier, key);
    if !is_canonical(ptr) {
        pac ^= 1 << 62;
    }
    let pac = pac & !((1 << 55) | 0x0000_ffff_ffff_ffff);
    pac | (ext << 55) | (ptr & 0x0000_ffff_ffff_ffff)
}

/// Authenticate `ptr` against `modifier`. On success the pointer is returned with its
/// PAC removed. On failure its bits 62:61 carry an error code and it is non-canonical,
/// so its next use faults.
pub fn auth(ptr: u64, modifier: u64, key: Key, key_number: u64) -> u64 {
    let orig = original_ptr(ptr);
    let pac = compute_pac(orig, modifier, key);
    let cmp_mask = 0xff7f_0000_0000_0000u64;
    if (pac ^ ptr) & cmp_mask != 0 {
        let error = (key_number << 1) | (key_number ^ 1);
        return (orig & !(3 << 61)) | ((error & 3) << 61);
    }
    orig
}

/// Remove the PAC from a pointer.
pub fn strip(ptr: u64) -> u64 {
    original_ptr(ptr)
}

/// `PACGA`: a 32-bit generic MAC in the top half of the result.
pub fn pacga(data: u64, modifier: u64, key: Key) -> u64 {
    compute_pac(data, modifier, key) & 0xffff_ffff_0000_0000
}

const RC: [u64; 5] = [
    0x0000_0000_0000_0000,
    0x1319_8A2E_0370_7344,
    0xA409_3822_299F_31D0,
    0x082E_FA98_EC4E_6C89,
    0x4528_21E6_38D0_1377,
];
const ALPHA: u64 = 0xC0AC_29B7_C97C_50DD;
const SBOX: [u8; 16] = [
    0xb, 0x6, 0x8, 0xf, 0xc, 0x0, 0x9, 0xe, 0x3, 0x7, 0x4, 0x5, 0xd, 0x2, 0x1, 0xa,
];
const INV_SBOX: [u8; 16] = [
    0x5, 0xe, 0xd, 0x8, 0xa, 0xb, 0x1, 0x9, 0x2, 0x6, 0xf, 0x0, 0x4, 0xc, 0x7, 0x3,
];

/// Cell permutations as `(source nibble, destination nibble)`.
const CELL_SHUFFLE: [(u8, u8); 16] = [
    (13, 0),
    (6, 1),
    (11, 2),
    (0, 3),
    (7, 4),
    (12, 5),
    (1, 6),
    (10, 7),
    (8, 8),
    (3, 9),
    (14, 10),
    (5, 11),
    (2, 12),
    (9, 13),
    (4, 14),
    (15, 15),
];
const CELL_INV_SHUFFLE: [(u8, u8); 16] = [
    (3, 0),
    (6, 1),
    (12, 2),
    (9, 3),
    (14, 4),
    (11, 5),
    (1, 6),
    (4, 7),
    (8, 8),
    (13, 9),
    (7, 10),
    (2, 11),
    (5, 12),
    (0, 13),
    (10, 14),
    (15, 15),
];
/// The tweak permutations, with the cells the tweak LFSR steps marked.
const TWEAK_SHUFFLE: [(u8, u8, bool); 16] = [
    (4, 0, false),
    (5, 1, false),
    (6, 2, true),
    (7, 3, false),
    (11, 4, true),
    (2, 5, false),
    (3, 6, false),
    (8, 7, true),
    (12, 8, false),
    (13, 9, false),
    (14, 10, false),
    (15, 11, true),
    (0, 12, true),
    (1, 13, false),
    (10, 14, true),
    (9, 15, true),
];
const TWEAK_INV_SHUFFLE: [(u8, u8, bool); 16] = [
    (12, 0, true),
    (13, 1, false),
    (5, 2, false),
    (6, 3, false),
    (0, 4, false),
    (1, 5, false),
    (2, 6, true),
    (3, 7, false),
    (7, 8, true),
    (15, 9, true),
    (14, 10, true),
    (4, 11, true),
    (8, 12, false),
    (9, 13, false),
    (10, 14, false),
    (11, 15, true),
];

fn nibble(value: u64, n: u8) -> u64 {
    (value >> (4 * n)) & 0xf
}

fn permute(value: u64, table: &[(u8, u8)]) -> u64 {
    table
        .iter()
        .fold(0, |out, &(src, dst)| out | nibble(value, src) << (4 * dst))
}

fn permute_tweak(value: u64, table: &[(u8, u8, bool)], step: fn(u64) -> u64) -> u64 {
    table.iter().fold(0, |out, &(src, dst, stepped)| {
        let cell = nibble(value, src);
        let cell = if stepped { step(cell) } else { cell };
        out | cell << (4 * dst)
    })
}

fn cell_shuffle(value: u64) -> u64 {
    permute(value, &CELL_SHUFFLE)
}

fn cell_inv_shuffle(value: u64) -> u64 {
    permute(value, &CELL_INV_SHUFFLE)
}

fn tweak_cell_rot(cell: u64) -> u64 {
    (cell >> 1) | (((cell ^ (cell >> 1)) & 1) << 3)
}

fn tweak_cell_inv_rot(cell: u64) -> u64 {
    ((cell << 1) & 0xf) | ((cell & 1) ^ (cell >> 3))
}

fn tweak_shuffle(value: u64) -> u64 {
    permute_tweak(value, &TWEAK_SHUFFLE, tweak_cell_rot)
}

fn tweak_inv_shuffle(value: u64) -> u64 {
    permute_tweak(value, &TWEAK_INV_SHUFFLE, tweak_cell_inv_rot)
}

fn rot_cell(cell: u64, n: u32) -> u64 {
    let doubled = cell | cell << 4;
    (doubled >> (4 - n)) & 0xf
}

/// The MixColumns-like diffusion layer, applied to each column of the 4x4 cell matrix.
fn mult(value: u64) -> u64 {
    let mut out = 0;
    for column in 0..4u8 {
        let i0 = nibble(value, column);
        let i4 = nibble(value, column + 4);
        let i8 = nibble(value, column + 8);
        let ic = nibble(value, column + 12);
        let t0 = rot_cell(i8, 1) ^ rot_cell(i4, 2) ^ rot_cell(i0, 1);
        let t1 = rot_cell(ic, 1) ^ rot_cell(i4, 1) ^ rot_cell(i0, 2);
        let t2 = rot_cell(ic, 2) ^ rot_cell(i8, 1) ^ rot_cell(i0, 1);
        let t3 = rot_cell(ic, 1) ^ rot_cell(i8, 2) ^ rot_cell(i4, 1);
        out |= t3 << (4 * column);
        out |= t2 << (4 * (column + 4));
        out |= t1 << (4 * (column + 8));
        out |= t0 << (4 * (column + 12));
    }
    out
}

fn sub(value: u64, table: &[u8; 16]) -> u64 {
    (0..16).fold(0, |out, n| {
        out | u64::from(table[nibble(value, n) as usize]) << (4 * n)
    })
}

/// `ComputePAC` of the architecture: the QARMA5 cipher over `data`, keyed by `key`,
/// tweaked by `modifier`.
pub fn compute_pac(data: u64, modifier: u64, key: Key) -> u64 {
    let key0 = key.hi;
    let key1 = key.lo;
    let modk0 = (key0 << 63) | ((key0 >> 1) ^ (key0 >> 63));
    let mut runningmod = modifier;
    let mut working = data ^ key0;
    for (i, rc) in RC.iter().enumerate() {
        working ^= key1 ^ runningmod;
        working ^= rc;
        if i > 0 {
            working = mult(cell_shuffle(working));
        }
        working = sub(working, &SBOX);
        runningmod = tweak_shuffle(runningmod);
    }
    working ^= modk0 ^ runningmod;
    working = mult(cell_shuffle(working));
    working = sub(working, &SBOX);
    working = mult(cell_shuffle(working));
    working ^= key1;
    working = sub(cell_inv_shuffle(working), &INV_SBOX);
    working = cell_inv_shuffle(mult(working));
    working ^= key0 ^ runningmod;
    for i in 0..=4usize {
        working = sub(working, &INV_SBOX);
        if i < 4 {
            working = cell_inv_shuffle(mult(working));
        }
        runningmod = tweak_inv_shuffle(runningmod);
        working ^= RC[4 - i] ^ (key1 ^ runningmod) ^ ALPHA;
    }
    working ^ modk0
}

fn key(cpu: &Cpu, id: KeyId) -> Key {
    let keys = &cpu.system.pac_keys;
    Key {
        lo: keys[2 * id.index()],
        hi: keys[2 * id.index() + 1],
    }
}

/// Whether the key is enabled for the current translation regime (`SCTLR_EL1`).
fn enabled(cpu: &Cpu, id: KeyId) -> bool {
    id.enable_bit()
        .is_none_or(|bit| cpu.system.sctlr_el1 & (1 << bit) != 0)
}

/// Register `n` read as `Xn|SP` (31 is the stack pointer).
fn read_sp(cpu: &Cpu, n: u8) -> u64 {
    if n == 31 { cpu.sp } else { cpu.x[n as usize] }
}

/// Register `n` read as `Xn` (31 is zero).
fn read_zr(cpu: &Cpu, n: u8) -> u64 {
    if n == 31 { 0 } else { cpu.x[n as usize] }
}

fn write_x(cpu: &mut Cpu, n: u8, value: u64) {
    if n != 31 {
        cpu.x[n as usize] = value;
    }
}

fn modifier_value(cpu: &Cpu, modifier: Modifier) -> u64 {
    match modifier {
        Modifier::Zero => 0,
        Modifier::Reg(n) => read_sp(cpu, n),
    }
}

/// Execute the PAuth instruction `word` on `cpu`. `word` must satisfy [`is_supported`].
/// `cpu.pc` is the address of the next instruction, which a link writes to `LR`.
pub fn run(cpu: &mut Cpu, word: u32) {
    let op = decode_word(word).expect("pauth::run is only called for supported words");
    match op {
        Op::Sign {
            key: id,
            reg,
            modifier,
        } => {
            if enabled(cpu, id) {
                let value = add_pac(
                    cpu.x[reg as usize],
                    modifier_value(cpu, modifier),
                    key(cpu, id),
                );
                write_x(cpu, reg, value);
            }
        }
        Op::Auth {
            key: id,
            reg,
            modifier,
        } => {
            if enabled(cpu, id) {
                let value = auth(
                    cpu.x[reg as usize],
                    modifier_value(cpu, modifier),
                    key(cpu, id),
                    id.number(),
                );
                write_x(cpu, reg, value);
            }
        }
        Op::Strip { reg } => {
            let value = strip(cpu.x[reg as usize]);
            write_x(cpu, reg, value);
        }
        Op::Pacga { rd, rn, modifier } => {
            let value = pacga(
                read_zr(cpu, rn),
                modifier_value(cpu, modifier),
                key(cpu, KeyId::Ga),
            );
            write_x(cpu, rd, value);
        }
        Op::Branch {
            key: id,
            link,
            target,
            modifier,
        } => {
            let next = cpu.pc;
            let mut destination = read_zr(cpu, target);
            if enabled(cpu, id) {
                destination = auth(
                    destination,
                    modifier_value(cpu, modifier),
                    key(cpu, id),
                    id.number(),
                );
            }
            if link {
                write_x(cpu, 30, next);
            }
            cpu.pc = destination;
        }
        Op::Return { key: id } => {
            let mut destination = cpu.x[30];
            if enabled(cpu, id) {
                destination = auth(destination, cpu.sp, key(cpu, id), id.number());
            }
            cpu.pc = destination;
        }
        Op::ExceptionReturn { key: id } => {
            let mut elr = cpu.system.elr_el1;
            if enabled(cpu, id) {
                elr = auth(elr, cpu.sp, key(cpu, id), id.number());
            }
            cpu.system.elr_el1 = elr;
            cpu.pc = exception::eret(cpu);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The key the QEMU reference was run with.
    const KEY: Key = Key {
        hi: 0xec28_02d4_e0a4_88e9,
        lo: 0x84be_85ce_9804_e94b,
    };

    /// `ComputePAC` values from QEMU's `pauth_computepac_architected`, compiled from its
    /// source with the same key: `(data, modifier, pac)`.
    const REFERENCE: [(u64, u64, u64); 6] = [
        (
            0x0000_0000_0000_0000,
            0x0000_0000_0000_0000,
            0xbc2f_6e1c_6e1a_f447,
        ),
        (
            0xffff_fe00_0700_4000,
            0x477d_469d_ec0b_8762,
            0xc1b6_9701_e911_0f0c,
        ),
        (
            0x0000_0008_1234_5678,
            0x0000_0000_0000_0000,
            0xc3ad_cffe_19ff_629c,
        ),
        (
            0x0000_0008_1234_5678,
            0x2222_2222_2222_2222,
            0xfe68_3d52_004b_603d,
        ),
        (
            0x1234_5678_1234_5678,
            0x0000_0000_0000_0000,
            0x29e3_4d27_b244_88a6,
        ),
        (
            0x1234_5678_1234_5678,
            0x477d_469d_ec0b_8762,
            0x2b7a_744b_38ed_59f6,
        ),
    ];

    #[test]
    fn compute_pac_matches_the_architected_reference() {
        for (data, modifier, pac) in REFERENCE {
            assert_eq!(compute_pac(data, modifier, KEY), pac, "data {data:#x}");
        }
    }

    #[test]
    fn sign_then_authenticate_returns_the_pointer_and_strip_removes_the_pac() {
        let ptr = 0x0000_0008_1234_5678;
        let signed = add_pac(ptr, 0x2222, KEY);
        assert_ne!(signed, ptr, "a PAC is inserted");
        assert_eq!(auth(signed, 0x2222, KEY, 0), ptr);
        assert_eq!(strip(signed), ptr);
    }

    #[test]
    fn a_kernel_pointer_round_trips_through_sign_and_auth() {
        let ptr = 0xffff_8000_8004_0000;
        let signed = add_pac(ptr, 0x1234, KEY);
        assert_eq!(auth(signed, 0x1234, KEY, 2), ptr);
    }

    #[test]
    fn a_wrong_modifier_or_key_leaves_a_non_canonical_pointer() {
        let ptr = 0x0000_0008_1234_5678;
        let signed = add_pac(ptr, 0x2222, KEY);
        let other_key = Key {
            lo: KEY.lo ^ 1,
            ..KEY
        };
        for failed in [
            auth(signed, 0x3333, KEY, 0),
            auth(signed, 0x2222, other_key, 0),
        ] {
            assert!(!is_canonical(failed), "{failed:#x} must not be usable");
            assert_eq!((failed >> 61) & 3, 1, "error code for key A");
        }
    }

    #[test]
    fn a_non_canonical_input_is_not_signed_into_a_valid_pointer() {
        // A pointer whose extension bits are wrong gets a corrupted PAC, so it cannot
        // authenticate later.
        let bad = 0x1234_0008_1234_5678;
        let signed = add_pac(bad, 0, KEY);
        assert!(!is_canonical(auth(signed, 0, KEY, 0)));
    }

    #[test]
    fn pacga_keeps_only_the_top_word() {
        let mac = pacga(0x1234, 0x5678, KEY);
        assert_eq!(mac & 0xffff_ffff, 0);
        assert_eq!(
            mac,
            compute_pac(0x1234, 0x5678, KEY) & 0xffff_ffff_0000_0000
        );
    }

    #[test]
    fn the_instruction_encodings_decode_as_their_names() {
        // Assembled with GNU as for armv8.3-a.
        assert_eq!(
            decode_word(0xdac1_0020),
            Some(Op::Sign {
                key: KeyId::Ia,
                reg: 0,
                modifier: Modifier::Reg(1)
            })
        );
        assert_eq!(
            decode_word(0xdac1_11ac),
            Some(Op::Auth {
                key: KeyId::Ia,
                reg: 12,
                modifier: Modifier::Reg(13)
            })
        );
        assert_eq!(
            decode_word(0xdac1_23f0),
            Some(Op::Sign {
                key: KeyId::Ia,
                reg: 16,
                modifier: Modifier::Zero
            })
        );
        assert_eq!(decode_word(0xdac1_43f8), Some(Op::Strip { reg: 24 }));
        assert_eq!(
            decode_word(0xd71f_0c43),
            Some(Op::Branch {
                key: KeyId::Ib,
                link: false,
                target: 2,
                modifier: Modifier::Reg(3)
            })
        );
        assert_eq!(
            decode_word(0xd63f_095f),
            Some(Op::Branch {
                key: KeyId::Ia,
                link: true,
                target: 10,
                modifier: Modifier::Zero
            })
        );
        assert_eq!(
            decode_word(0xd65f_0fff),
            Some(Op::Return { key: KeyId::Ib })
        );
        assert_eq!(
            decode_word(0xd503_233f),
            Some(Op::Sign {
                key: KeyId::Ia,
                reg: 30,
                modifier: Modifier::Reg(31)
            })
        );
        // Plain BR and ordinary one-source data processing are not PAuth.
        assert!(!is_supported(0xd61f_0000));
        assert!(!is_supported(0xdac0_0000));
    }

    #[test]
    fn a_pac_instruction_signs_its_register_and_a_branch_jumps_to_the_authenticated_target() {
        let mut cpu = Cpu::default();
        cpu.system.sctlr_el1 = 1 << 31;
        cpu.system.pac_keys[0] = KEY.lo;
        cpu.system.pac_keys[1] = KEY.hi;
        let ptr = 0x0000_0008_1234_5678;
        cpu.x[0] = ptr;
        cpu.x[1] = 0x2222;
        run(&mut cpu, 0xdac1_0020);
        assert_eq!(cpu.x[0], add_pac(ptr, 0x2222, KEY));
        cpu.x[2] = add_pac(0x4000, 0x2222, KEY);
        cpu.x[3] = 0x2222;
        cpu.pc = 0x1000;
        run(&mut cpu, 0xd71f_0800 | 2 << 5 | 3);
        assert_eq!(cpu.pc, 0x4000);
    }
}
