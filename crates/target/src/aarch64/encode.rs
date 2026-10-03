#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Reg {
    X0,
    X1,
    X2,
    X3,
    X4,
    X5,
    X6,
    X7,
    X8,
    X9,
    X10,
    X11,
    X12,
    X13,
    X14,
    X15,
    X16,
    X17,
    X18,
    X19,
    X20,
    X21,
    X22,
    X23,
    X24,
    X25,
    X26,
    X27,
    X28,
    X29,
    X30,
    Zr,
}

impl Reg {
    pub fn idx(self) -> u32 {
        self as u32
    }

    pub fn from_u5(v: u32) -> Reg {
        match v & 31 {
            0 => Reg::X0,
            1 => Reg::X1,
            2 => Reg::X2,
            3 => Reg::X3,
            4 => Reg::X4,
            5 => Reg::X5,
            6 => Reg::X6,
            7 => Reg::X7,
            8 => Reg::X8,
            9 => Reg::X9,
            10 => Reg::X10,
            11 => Reg::X11,
            12 => Reg::X12,
            13 => Reg::X13,
            14 => Reg::X14,
            15 => Reg::X15,
            16 => Reg::X16,
            17 => Reg::X17,
            18 => Reg::X18,
            19 => Reg::X19,
            20 => Reg::X20,
            21 => Reg::X21,
            22 => Reg::X22,
            23 => Reg::X23,
            24 => Reg::X24,
            25 => Reg::X25,
            26 => Reg::X26,
            27 => Reg::X27,
            28 => Reg::X28,
            29 => Reg::X29,
            30 => Reg::X30,
            _ => Reg::Zr,
        }
    }
}

fn n(reg: Reg) -> u32 {
    reg.idx()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Cond {
    Eq = 0,
    Ne = 1,
    Hs = 2,
    Lo = 3,
    Mi = 4,
    Pl = 5,
    Hi = 8,
    Ls = 9,
    Ge = 10,
    Lt = 11,
    Gt = 12,
    Le = 13,
}

impl Cond {
    pub fn from_u4(v: u32) -> Option<Cond> {
        match v & 15 {
            0 => Some(Cond::Eq),
            1 => Some(Cond::Ne),
            2 => Some(Cond::Hs),
            3 => Some(Cond::Lo),
            4 => Some(Cond::Mi),
            5 => Some(Cond::Pl),
            8 => Some(Cond::Hi),
            9 => Some(Cond::Ls),
            10 => Some(Cond::Ge),
            11 => Some(Cond::Lt),
            12 => Some(Cond::Gt),
            13 => Some(Cond::Le),
            _ => None,
        }
    }
}

pub fn invert(c: Cond) -> Cond {
    match c {
        Cond::Eq => Cond::Ne,
        Cond::Ne => Cond::Eq,
        Cond::Hs => Cond::Lo,
        Cond::Lo => Cond::Hs,
        Cond::Mi => Cond::Pl,
        Cond::Pl => Cond::Mi,
        Cond::Hi => Cond::Ls,
        Cond::Ls => Cond::Hi,
        Cond::Ge => Cond::Lt,
        Cond::Lt => Cond::Ge,
        Cond::Gt => Cond::Le,
        Cond::Le => Cond::Gt,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FKind {
    Single,
    Double,
    Half,
}

fn ftype(kind: FKind) -> u32 {
    match kind {
        FKind::Single => 0,
        FKind::Double => 1 << 22,
        FKind::Half => 3 << 22,
    }
}

const SF64: u32 = 1 << 31;

pub fn ret() -> u32 {
    0xD65F0000 | (30 << 5)
}

pub fn nop() -> u32 {
    0xD503201F
}

pub fn svc(imm: u16) -> u32 {
    0xD4000001 | ((imm as u32) << 5)
}

pub fn add(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    0x0B000000 | (n(rm) << 16) | (n(rn) << 5) | n(rd)
}

pub fn sub(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    0x4B000000 | (n(rm) << 16) | (n(rn) << 5) | n(rd)
}

pub fn mul(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    0x1B000000 | (n(rm) << 16) | (31 << 10) | (n(rn) << 5) | n(rd)
}

pub fn smulh(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    0x9B407C00 | (n(rm) << 16) | (n(rn) << 5) | n(rd)
}

pub fn umulh(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    0x9BC07C00 | (n(rm) << 16) | (n(rn) << 5) | n(rd)
}

pub fn sbfm(rd: Reg, rn: Reg, immr: u8, imms: u8) -> u32 {
    0x93400000 | ((immr as u32) << 16) | ((imms as u32) << 10) | (n(rn) << 5) | n(rd)
}

pub fn ubfm(rd: Reg, rn: Reg, immr: u8, imms: u8) -> u32 {
    0xD3400000 | ((immr as u32) << 16) | ((imms as u32) << 10) | (n(rn) << 5) | n(rd)
}

pub fn andr(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    0x0A000000 | (n(rm) << 16) | (n(rn) << 5) | n(rd)
}

pub fn orr(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    0x2A000000 | (n(rm) << 16) | (n(rn) << 5) | n(rd)
}

pub fn eor(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    0x4A000000 | (n(rm) << 16) | (n(rn) << 5) | n(rd)
}

pub fn mov(rd: Reg, rm: Reg) -> u32 {
    SF64 | orr(rd, Reg::Zr, rm)
}

pub fn movz(rd: Reg, imm: u16, shift: u8) -> u32 {
    0x52800000 | ((shift as u32) << 21) | ((imm as u32) << 5) | n(rd)
}

pub fn movk(rd: Reg, imm: u16, shift: u8) -> u32 {
    0x72800000 | ((shift as u32) << 21) | ((imm as u32) << 5) | n(rd)
}

pub fn movz64(rd: Reg, imm: u16, shift: u8) -> u32 {
    0xD2800000 | ((shift as u32) << 21) | ((imm as u32) << 5) | n(rd)
}

pub fn movk64(rd: Reg, imm: u16, shift: u8) -> u32 {
    0xF2800000 | ((shift as u32) << 21) | ((imm as u32) << 5) | n(rd)
}

pub fn add_imm(rd: Reg, rn: Reg, imm: u16) -> u32 {
    debug_assert!(imm <= 0xFFF);
    0x11000000 | ((imm as u32) << 10) | (n(rn) << 5) | n(rd)
}

pub fn sub_imm(rd: Reg, rn: Reg, imm: u16) -> u32 {
    debug_assert!(imm <= 0xFFF);
    0x51000000 | ((imm as u32) << 10) | (n(rn) << 5) | n(rd)
}

pub fn cmp(rn: Reg, rm: Reg) -> u32 {
    0x6B00001F | (n(rm) << 16) | (n(rn) << 5)
}

pub fn cmp64(rn: Reg, rm: Reg) -> u32 {
    SF64 | cmp(rn, rm)
}

pub fn subs(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    0x6B000000 | (n(rm) << 16) | (n(rn) << 5) | n(rd)
}

pub fn subs64(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    SF64 | subs(rd, rn, rm)
}

pub fn adds(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    0x2B000000 | (n(rm) << 16) | (n(rn) << 5) | n(rd)
}

pub fn adds64(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    SF64 | adds(rd, rn, rm)
}

pub fn ands(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    0x6A000000 | (n(rm) << 16) | (n(rn) << 5) | n(rd)
}

pub fn ands64(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    SF64 | ands(rd, rn, rm)
}

pub fn subs_imm(rd: Reg, rn: Reg, imm: u16) -> u32 {
    debug_assert!(imm <= 0xFFF);
    0x71000000 | ((imm as u32) << 10) | (n(rn) << 5) | n(rd)
}

pub fn subs_imm64(rd: Reg, rn: Reg, imm: u16) -> u32 {
    SF64 | subs_imm(rd, rn, imm)
}

pub fn adds_imm(rd: Reg, rn: Reg, imm: u16) -> u32 {
    debug_assert!(imm <= 0xFFF);
    0x31000000 | ((imm as u32) << 10) | (n(rn) << 5) | n(rd)
}

pub fn adds_imm64(rd: Reg, rn: Reg, imm: u16) -> u32 {
    SF64 | adds_imm(rd, rn, imm)
}

pub fn cset(rd: Reg, cond: Cond) -> u32 {
    let inv: u32 = (cond as u32) ^ 1;
    0x1A800400 | (31 << 16) | (inv << 12) | (31 << 5) | n(rd)
}

pub fn cbnz(rt: Reg, off: i32) -> u32 {
    let imm19: u32 = ((off >> 2) as u32) & 0x7FFFF;
    0x35000000 | (imm19 << 5) | n(rt)
}

pub fn cbz(rt: Reg, off: i32) -> u32 {
    let imm19: u32 = ((off >> 2) as u32) & 0x7FFFF;
    0x34000000 | (imm19 << 5) | n(rt)
}

pub fn bcc(cond: Cond, off: i32) -> u32 {
    let imm19: u32 = ((off >> 2) as u32) & 0x7FFFF;
    0x54000000 | (imm19 << 5) | (cond as u32)
}

pub fn b(off: i32) -> u32 {
    let imm26: u32 = ((off >> 2) as u32) & 0x3FFFFFF;
    0x14000000 | imm26
}

pub fn bl(off: i32) -> u32 {
    let imm26: u32 = ((off >> 2) as u32) & 0x3FFFFFF;
    0x94000000 | imm26
}

pub fn blr(rn: Reg) -> u32 {
    0xD63F0000 | (n(rn) << 5)
}

pub fn stp_pre(rt1: Reg, rt2: Reg, rn: Reg, imm: i32) -> u32 {
    let imm7: u32 = ((imm >> 3) as u32) & 0x7F;
    0xA9800000 | (imm7 << 15) | (n(rt2) << 10) | (n(rn) << 5) | n(rt1)
}

pub fn ldp_post(rt1: Reg, rt2: Reg, rn: Reg, imm: i32) -> u32 {
    let imm7: u32 = ((imm >> 3) as u32) & 0x7F;
    0xA8C00000 | (imm7 << 15) | (n(rt2) << 10) | (n(rn) << 5) | n(rt1)
}

pub fn ldp_off_x(rt1: Reg, rt2: Reg, rn: Reg, imm: i32) -> u32 {
    let imm7: u32 = ((imm >> 3) as u32) & 0x7F;
    0xA9400000 | (imm7 << 15) | (n(rt2) << 10) | (n(rn) << 5) | n(rt1)
}

pub fn stp_off_x(rt1: Reg, rt2: Reg, rn: Reg, imm: i32) -> u32 {
    let imm7: u32 = ((imm >> 3) as u32) & 0x7F;
    0xA9000000 | (imm7 << 15) | (n(rt2) << 10) | (n(rn) << 5) | n(rt1)
}

pub fn ldp_off_w(rt1: Reg, rt2: Reg, rn: Reg, imm: i32) -> u32 {
    let imm7: u32 = ((imm >> 2) as u32) & 0x7F;
    0x29400000 | (imm7 << 15) | (n(rt2) << 10) | (n(rn) << 5) | n(rt1)
}

pub fn stp_off_w(rt1: Reg, rt2: Reg, rn: Reg, imm: i32) -> u32 {
    let imm7: u32 = ((imm >> 2) as u32) & 0x7F;
    0x29000000 | (imm7 << 15) | (n(rt2) << 10) | (n(rn) << 5) | n(rt1)
}

pub fn str_off(rt: Reg, rn: Reg, off: u16) -> u32 {
    0xF9000000 | (((off as u32) >> 3) << 10) | (n(rn) << 5) | n(rt)
}

pub fn ldr_off(rt: Reg, rn: Reg, off: u16) -> u32 {
    0xF9400000 | (((off as u32) >> 3) << 10) | (n(rn) << 5) | n(rt)
}

pub fn prfm(rn: Reg) -> u32 {
    0xF9800000 | (n(rn) << 5)
}

pub fn add_imm64(rd: Reg, rn: Reg, imm: u16) -> u32 {
    debug_assert!(imm <= 0xFFF);
    0x91000000 | ((imm as u32) << 10) | (n(rn) << 5) | n(rd)
}

pub fn sub_imm64(rd: Reg, rn: Reg, imm: u16) -> u32 {
    debug_assert!(imm <= 0xFFF);
    0xD1000000 | ((imm as u32) << 10) | (n(rn) << 5) | n(rd)
}

pub fn add_imm64_shift(rd: Reg, rn: Reg, imm: u16) -> u32 {
    debug_assert!(imm <= 0xFFF);
    0x91400000 | ((imm as u32) << 10) | (n(rn) << 5) | n(rd)
}

pub fn sub_imm64_shift(rd: Reg, rn: Reg, imm: u16) -> u32 {
    debug_assert!(imm <= 0xFFF);
    0xD1400000 | ((imm as u32) << 10) | (n(rn) << 5) | n(rd)
}

pub fn adrp(rd: Reg, imm: i32) -> u32 {
    let u: u32 = (imm as u32) & 0x1FFFFF;
    let immlo: u32 = u & 0x3;
    let immhi: u32 = u >> 2;
    0x90000000 | (immlo << 29) | (immhi << 5) | n(rd)
}

pub fn str_w(rt: Reg, rn: Reg, off: u16) -> u32 {
    0xB9000000 | (((off as u32) >> 2) << 10) | (n(rn) << 5) | n(rt)
}

pub fn ldr_w(rt: Reg, rn: Reg, off: u16) -> u32 {
    0xB9400000 | (((off as u32) >> 2) << 10) | (n(rn) << 5) | n(rt)
}

pub fn ldrsw(rt: Reg, rn: Reg, off: u16) -> u32 {
    0xB9800000 | (((off as u32) >> 2) << 10) | (n(rn) << 5) | n(rt)
}

pub fn strb(rt: Reg, rn: Reg) -> u32 {
    0x39000000 | (n(rn) << 5) | n(rt)
}

pub fn ldrsb(rt: Reg, rn: Reg) -> u32 {
    0x39C00000 | (n(rn) << 5) | n(rt)
}

pub fn ldrb(rt: Reg, rn: Reg) -> u32 {
    0x39400000 | (n(rn) << 5) | n(rt)
}

pub fn strh(rt: Reg, rn: Reg) -> u32 {
    0x79000000 | (n(rn) << 5) | n(rt)
}

pub fn ldrh(rt: Reg, rn: Reg) -> u32 {
    0x79400000 | (n(rn) << 5) | n(rt)
}

pub fn ldrsh(rt: Reg, rn: Reg) -> u32 {
    0x79C00000 | (n(rn) << 5) | n(rt)
}

pub fn strb_off(rt: Reg, rn: Reg, off: u16) -> u32 {
    0x39000000 | ((off as u32) << 10) | (n(rn) << 5) | n(rt)
}

pub fn ldrb_off(rt: Reg, rn: Reg, off: u16) -> u32 {
    0x39400000 | ((off as u32) << 10) | (n(rn) << 5) | n(rt)
}

pub fn ldrsb_off(rt: Reg, rn: Reg, off: u16) -> u32 {
    0x39C00000 | ((off as u32) << 10) | (n(rn) << 5) | n(rt)
}

pub fn strh_off(rt: Reg, rn: Reg, off: u16) -> u32 {
    0x79000000 | (((off as u32) >> 1) << 10) | (n(rn) << 5) | n(rt)
}

pub fn ldrh_off(rt: Reg, rn: Reg, off: u16) -> u32 {
    0x79400000 | (((off as u32) >> 1) << 10) | (n(rn) << 5) | n(rt)
}

pub fn ldrsh_off(rt: Reg, rn: Reg, off: u16) -> u32 {
    0x79C00000 | (((off as u32) >> 1) << 10) | (n(rn) << 5) | n(rt)
}

pub fn sdiv(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    0x1AC00C00 | (n(rm) << 16) | (n(rn) << 5) | n(rd)
}

pub fn udiv(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    0x1AC00800 | (n(rm) << 16) | (n(rn) << 5) | n(rd)
}

pub fn msub(rd: Reg, rn: Reg, rm: Reg, ra: Reg) -> u32 {
    0x1B008000 | (n(rm) << 16) | (n(ra) << 10) | (n(rn) << 5) | n(rd)
}

pub fn lslv(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    0x1AC02000 | (n(rm) << 16) | (n(rn) << 5) | n(rd)
}

pub fn lsrv(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    0x1AC02400 | (n(rm) << 16) | (n(rn) << 5) | n(rd)
}

pub fn asrv(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    0x1AC02800 | (n(rm) << 16) | (n(rn) << 5) | n(rd)
}

pub fn add64(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    SF64 | add(rd, rn, rm)
}

pub fn sub64(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    SF64 | sub(rd, rn, rm)
}

pub fn add_shifted(rd: Reg, rn: Reg, rm: Reg, shift: u8) -> u32 {
    add(rd, rn, rm) | ((shift as u32) << 10)
}

pub fn add_shifted64(rd: Reg, rn: Reg, rm: Reg, shift: u8) -> u32 {
    SF64 | add_shifted(rd, rn, rm, shift)
}

pub fn sub_shifted(rd: Reg, rn: Reg, rm: Reg, shift: u8) -> u32 {
    sub(rd, rn, rm) | ((shift as u32) << 10)
}

pub fn sub_shifted64(rd: Reg, rn: Reg, rm: Reg, shift: u8) -> u32 {
    SF64 | sub_shifted(rd, rn, rm, shift)
}

pub fn mul64(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    SF64 | mul(rd, rn, rm)
}

pub fn lslv64(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    SF64 | lslv(rd, rn, rm)
}

pub fn lsrv64(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    SF64 | lsrv(rd, rn, rm)
}

pub fn asrv64(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    SF64 | asrv(rd, rn, rm)
}

pub fn csel(rd: Reg, rn: Reg, rm: Reg, cond: Cond) -> u32 {
    0x1A800000 | (n(rm) << 16) | ((cond as u32) << 12) | (n(rn) << 5) | n(rd)
}

pub fn fadd(rd: Reg, rn: Reg, rm: Reg, kind: FKind) -> u32 {
    0x1E202800 | ftype(kind) | (n(rm) << 16) | (n(rn) << 5) | n(rd)
}

pub fn fsub(rd: Reg, rn: Reg, rm: Reg, kind: FKind) -> u32 {
    0x1E203800 | ftype(kind) | (n(rm) << 16) | (n(rn) << 5) | n(rd)
}

pub fn fmul(rd: Reg, rn: Reg, rm: Reg, kind: FKind) -> u32 {
    0x1E200800 | ftype(kind) | (n(rm) << 16) | (n(rn) << 5) | n(rd)
}

pub fn fdiv(rd: Reg, rn: Reg, rm: Reg, kind: FKind) -> u32 {
    0x1E201800 | ftype(kind) | (n(rm) << 16) | (n(rn) << 5) | n(rd)
}

pub fn fmadd(rd: Reg, rn: Reg, rm: Reg, ra: Reg, dbl: bool) -> u32 {
    0x1F000000
        | ftype(if dbl { FKind::Double } else { FKind::Single })
        | (n(rm) << 16)
        | (n(ra) << 10)
        | (n(rn) << 5)
        | n(rd)
}

pub fn fmsub(rd: Reg, rn: Reg, rm: Reg, ra: Reg, dbl: bool) -> u32 {
    0x1F008000
        | ftype(if dbl { FKind::Double } else { FKind::Single })
        | (n(rm) << 16)
        | (n(ra) << 10)
        | (n(rn) << 5)
        | n(rd)
}

pub fn fnmsub(rd: Reg, rn: Reg, rm: Reg, ra: Reg, dbl: bool) -> u32 {
    0x1F208000
        | ftype(if dbl { FKind::Double } else { FKind::Single })
        | (n(rm) << 16)
        | (n(ra) << 10)
        | (n(rn) << 5)
        | n(rd)
}

pub fn ldr_q(rt: Reg, rn: Reg, off: u16) -> u32 {
    0x3DC00000 | (((off as u32) >> 4) << 10) | (n(rn) << 5) | n(rt)
}

pub fn str_q(rt: Reg, rn: Reg, off: u16) -> u32 {
    0x3D800000 | (((off as u32) >> 4) << 10) | (n(rn) << 5) | n(rt)
}

pub fn fadd_vec(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    0x4E20D400 | (n(rm) << 16) | (n(rn) << 5) | n(rd)
}

pub fn fsub_vec(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    0x4EA0D400 | (n(rm) << 16) | (n(rn) << 5) | n(rd)
}

pub fn fmul_vec(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    0x6E20DC00 | (n(rm) << 16) | (n(rn) << 5) | n(rd)
}

pub fn fdiv_vec(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    0x6E20FC00 | (n(rm) << 16) | (n(rn) << 5) | n(rd)
}

pub fn fmla_vec(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    0x4E20CC00 | (n(rm) << 16) | (n(rn) << 5) | n(rd)
}

pub fn fmls_vec(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    0x4EA0CC00 | (n(rm) << 16) | (n(rn) << 5) | n(rd)
}

pub fn fneg_vec(rd: Reg, rn: Reg) -> u32 {
    0x6EA0F800 | (n(rn) << 5) | n(rd)
}

pub fn fsqrt_vec(rd: Reg, rn: Reg) -> u32 {
    0x6EA1F800 | (n(rn) << 5) | n(rd)
}

pub fn fmin_vec(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    0x4EA0F400 | (n(rm) << 16) | (n(rn) << 5) | n(rd)
}

pub fn fmax_vec(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    0x4E20F400 | (n(rm) << 16) | (n(rn) << 5) | n(rd)
}

pub fn fcmeq_vec(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    0x4E20E400 | (n(rm) << 16) | (n(rn) << 5) | n(rd)
}

pub fn fcmgt_vec(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    0x6EA0E400 | (n(rm) << 16) | (n(rn) << 5) | n(rd)
}

pub fn fcmge_vec(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    0x6E20E400 | (n(rm) << 16) | (n(rn) << 5) | n(rd)
}

pub fn mvn_vec(rd: Reg, rn: Reg) -> u32 {
    0x6E205800 | (n(rn) << 5) | n(rd)
}

pub fn bsl_vec(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    0x6E601C00 | (n(rm) << 16) | (n(rn) << 5) | n(rd)
}

pub fn dup_from_gpr(rd: Reg, rn: Reg) -> u32 {
    0x4E040C00 | (n(rn) << 5) | n(rd)
}

pub fn dup_vec_lane(rd: Reg, rn: Reg, index: u8) -> u32 {
    let imm5: u32 = ((index as u32) << 3) | 0b100;
    0x4E000400 | (imm5 << 16) | (n(rn) << 5) | n(rd)
}

pub fn dup_lane(rd: Reg, rn: Reg, index: u8) -> u32 {
    let imm5: u32 = ((index as u32) << 3) | 0b100;
    0x5E000400 | (imm5 << 16) | (n(rn) << 5) | n(rd)
}

pub fn ins_lane(rd: Reg, lane: u8, rn: Reg) -> u32 {
    let imm5: u32 = ((lane as u32) << 3) | 0b100;
    0x6E000400 | (imm5 << 16) | (n(rn) << 5) | n(rd)
}

pub fn umov_lane(rd: Reg, rn: Reg, index: u8) -> u32 {
    let imm5: u32 = ((index as u32) << 3) | 0b100;
    0x0E003C00 | (imm5 << 16) | (n(rn) << 5) | n(rd)
}

pub fn ins_lane_from_gpr(rd: Reg, lane: u8, rn: Reg) -> u32 {
    let imm5: u32 = ((lane as u32) << 3) | 0b100;
    0x4E001C00 | (imm5 << 16) | (n(rn) << 5) | n(rd)
}

pub fn addv(rd: Reg, rn: Reg) -> u32 {
    0x4EB1B800 | (n(rn) << 5) | n(rd)
}

pub fn faddp_vec(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    0x6E20D400 | (n(rm) << 16) | (n(rn) << 5) | n(rd)
}

pub fn faddp_scalar(rd: Reg, rn: Reg) -> u32 {
    0x7E30D800 | (n(rn) << 5) | n(rd)
}

pub fn mov_vec(rd: Reg, rn: Reg) -> u32 {
    0x4EA01C00 | (n(rn) << 16) | (n(rn) << 5) | n(rd)
}

pub fn ins_d1_from_gpr(rd: Reg, rn: Reg) -> u32 {
    0x4E181C00 | (n(rn) << 5) | n(rd)
}

pub fn sdot(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    0x4E809400 | (n(rm) << 16) | (n(rn) << 5) | n(rd)
}

pub fn udot(rd: Reg, rn: Reg, rm: Reg) -> u32 {
    sdot(rd, rn, rm) | (1 << 29)
}

pub fn fcmp(rn: Reg, rm: Reg, kind: FKind) -> u32 {
    0x1E202000 | ftype(kind) | (n(rm) << 16) | (n(rn) << 5)
}

pub fn fcsel(rd: Reg, rn: Reg, rm: Reg, cond: Cond, kind: FKind) -> u32 {
    0x1E200C00 | ftype(kind) | (n(rm) << 16) | ((cond as u32) << 12) | (n(rn) << 5) | n(rd)
}

pub fn fmov_reg(rd: Reg, rn: Reg) -> u32 {
    0x1E604000 | (n(rn) << 5) | n(rd)
}

pub fn fmov_from_gpr(rd: Reg, rn: Reg, dbl: bool) -> u32 {
    if dbl {
        0x9E670000 | (n(rn) << 5) | n(rd)
    } else {
        0x1E270000 | (n(rn) << 5) | n(rd)
    }
}

pub fn fmov_to_gpr(rd: Reg, rn: Reg, dbl: bool) -> u32 {
    if dbl {
        0x9E660000 | (n(rn) << 5) | n(rd)
    } else {
        0x1E260000 | (n(rn) << 5) | n(rd)
    }
}

pub fn fmov_hfrom_gpr(rd: Reg, rn: Reg) -> u32 {
    0x1EE70000 | (n(rn) << 5) | n(rd)
}

pub fn fsqrt(rd: Reg, rn: Reg, dbl: bool) -> u32 {
    (if dbl { 0x1E61C000 } else { 0x1E21C000 }) | (n(rn) << 5) | n(rd)
}

fn frint(base: u32, rd: Reg, rn: Reg, dbl: bool) -> u32 {
    (if dbl { base | 0x00400000 } else { base }) | (n(rn) << 5) | n(rd)
}

pub fn frintn(rd: Reg, rn: Reg, dbl: bool) -> u32 {
    frint(0x1E244000, rd, rn, dbl)
}

pub fn frintp(rd: Reg, rn: Reg, dbl: bool) -> u32 {
    frint(0x1E24C000, rd, rn, dbl)
}

pub fn frintm(rd: Reg, rn: Reg, dbl: bool) -> u32 {
    frint(0x1E254000, rd, rn, dbl)
}

pub fn frintz(rd: Reg, rn: Reg, dbl: bool) -> u32 {
    frint(0x1E25C000, rd, rn, dbl)
}

pub fn cvt_int_to_float(rd: Reg, rn: Reg, kind: FKind, signed: bool) -> u32 {
    let base: u32 = if signed { 0x1E220000 } else { 0x1E230000 };
    base | ftype(kind) | (n(rn) << 5) | n(rd)
}

pub fn cvt_float_to_int(rd: Reg, rn: Reg, kind: FKind, signed: bool) -> u32 {
    let base: u32 = if signed { 0x1E380000 } else { 0x1E390000 };
    base | ftype(kind) | (n(rn) << 5) | n(rd)
}

pub fn fcvt(rd: Reg, rn: Reg, to_double: bool) -> u32 {
    if to_double {
        0x1E22C000 | (n(rn) << 5) | n(rd)
    } else {
        0x1E624000 | (n(rn) << 5) | n(rd)
    }
}

pub fn fcvt_sfrom_h(rd: Reg, rn: Reg) -> u32 {
    0x1EE24000 | (n(rn) << 5) | n(rd)
}

pub fn fcvt_hfrom_s(rd: Reg, rn: Reg) -> u32 {
    0x1E23C000 | (n(rn) << 5) | n(rd)
}

pub fn fcvt_hfrom_d(rd: Reg, rn: Reg) -> u32 {
    0x1E63C000 | (n(rn) << 5) | n(rd)
}

pub fn fcvt_dfrom_h(rd: Reg, rn: Reg) -> u32 {
    0x1EE2C000 | (n(rn) << 5) | n(rd)
}

pub fn str_fp(rt: Reg, rn: Reg, off: u16, dbl: bool) -> u32 {
    if dbl {
        0xFD000000 | (((off as u32) >> 3) << 10) | (n(rn) << 5) | n(rt)
    } else {
        0xBD000000 | (((off as u32) >> 2) << 10) | (n(rn) << 5) | n(rt)
    }
}

pub fn ldr_fp(rt: Reg, rn: Reg, off: u16, dbl: bool) -> u32 {
    if dbl {
        0xFD400000 | (((off as u32) >> 3) << 10) | (n(rn) << 5) | n(rt)
    } else {
        0xBD400000 | (((off as u32) >> 2) << 10) | (n(rn) << 5) | n(rt)
    }
}

pub fn ldr_hfp(rt: Reg, rn: Reg, off: u16) -> u32 {
    debug_assert!(off.is_multiple_of(2));
    0x7D400000 | (((off as u32) >> 1) << 10) | (n(rn) << 5) | n(rt)
}

pub fn str_hfp(rt: Reg, rn: Reg, off: u16) -> u32 {
    debug_assert!(off.is_multiple_of(2));
    0x7D000000 | (((off as u32) >> 1) << 10) | (n(rn) << 5) | n(rt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cmp64_golden() {
        assert_eq!(0xEB01001F, cmp64(Reg::X0, Reg::X1));
        assert_eq!(cmp64(Reg::X3, Reg::X9), subs64(Reg::Zr, Reg::X3, Reg::X9));
    }

    #[test]
    fn known_encodings() {
        assert_eq!(0xD65F03C0, ret());
        assert_eq!(0xD503201F, nop());
        assert_eq!(0xD4000001, svc(0));
        assert_eq!(0x0B020020, add(Reg::X0, Reg::X1, Reg::X2));
        assert_eq!(0x4B020020, sub(Reg::X0, Reg::X1, Reg::X2));
        assert_eq!(0x52800540, movz(Reg::X0, 42, 0));
        assert_eq!(0x72800540, movk(Reg::X0, 42, 0));
        assert_eq!(0x1100A800, add_imm(Reg::X0, Reg::X0, 42));
        assert_eq!(0x94000000, bl(0));
        assert_eq!(0x14000000, b(0));
        assert_eq!(0x94000001, bl(4));
        assert_eq!(0x5400000B, bcc(Cond::Lt, 0));
        assert_eq!(0xEB01001F, cmp64(Reg::X0, Reg::X1));
        assert_eq!(0x8B020020, add64(Reg::X0, Reg::X1, Reg::X2));
        assert_eq!(0xCB020020, sub64(Reg::X0, Reg::X1, Reg::X2));
        assert_eq!(0xD2800000 | (42 << 5), movz64(Reg::X0, 42, 0));
        assert_eq!(invert(Cond::Eq), Cond::Ne);
        assert_eq!(invert(invert(Cond::Gt)), Cond::Gt);
    }

    #[test]
    fn smulh_umulh_golden() {
        assert_eq!(0x9b4a7d28, smulh(Reg::X8, Reg::X9, Reg::X10));
        assert_eq!(0x9bca7d28, umulh(Reg::X8, Reg::X9, Reg::X10));
    }

    #[test]
    fn sbfm_ubfm_golden() {
        assert_eq!(0x93407d28, sbfm(Reg::X8, Reg::X9, 0, 31));
        assert_eq!(0xd3407d28, ubfm(Reg::X8, Reg::X9, 0, 31));
        assert_eq!(0x93401d28, sbfm(Reg::X8, Reg::X9, 0, 7));
    }

    #[test]
    fn bcc_golden() {
        assert_eq!(0x5400000B, bcc(Cond::Lt, 0));
        assert_eq!(0x54000040, bcc(Cond::Eq, 8));
        assert_eq!(0x5400006C, bcc(Cond::Gt, 12));
        assert_eq!(0x54FFFFEA, bcc(Cond::Ge, -4));
        assert_eq!(0, bcc(Cond::Hi, 20) & 0x10);
    }

    #[test]
    fn neon_vec_golden() {
        assert_eq!(0x6EA0F800, fneg_vec(Reg::X0, Reg::X0));
        assert_eq!(0x6EA1F800, fsqrt_vec(Reg::X0, Reg::X0));
        assert_eq!(0x4EA2F400, fmin_vec(Reg::X0, Reg::X0, Reg::X2));
        assert_eq!(0x4E22F420, fmax_vec(Reg::X0, Reg::X1, Reg::X2));
        assert_eq!(0x4E22E420, fcmeq_vec(Reg::X0, Reg::X1, Reg::X2));
        assert_eq!(0x6EA2E420, fcmgt_vec(Reg::X0, Reg::X1, Reg::X2));
        assert_eq!(0x6E22E420, fcmge_vec(Reg::X0, Reg::X1, Reg::X2));
        assert_eq!(0x6E621C20, bsl_vec(Reg::X0, Reg::X1, Reg::X2));
        assert_eq!(0x4E040C20, dup_from_gpr(Reg::X0, Reg::X1));
        assert_eq!(0x4E140420, dup_vec_lane(Reg::X0, Reg::X1, 2));
        assert_eq!(0x6E205820, mvn_vec(Reg::X0, Reg::X1));
        assert_eq!(0x4E181C20, ins_d1_from_gpr(Reg::X0, Reg::X1));
        assert_eq!(0x4E181D34, ins_d1_from_gpr(Reg::X20, Reg::X9));
        assert_eq!(0x0E043C00, umov_lane(Reg::X0, Reg::X0, 0));
        assert_eq!(0x0E1C3C61, umov_lane(Reg::X1, Reg::X3, 3));
        assert_eq!(0x4E041C00, ins_lane_from_gpr(Reg::X0, 0, Reg::X0));
        assert_eq!(0x4E1C1C61, ins_lane_from_gpr(Reg::X1, 3, Reg::X3));
    }

    #[test]
    fn horiz_reduce_golden() {
        assert_eq!(0x4EB1B820, addv(Reg::X0, Reg::X1));
        assert_eq!(0x4EB1B862, addv(Reg::X2, Reg::X3));
        assert_eq!(0x6E22D420, faddp_vec(Reg::X0, Reg::X1, Reg::X2));
        assert_eq!(0x6E25D4A4, faddp_vec(Reg::X4, Reg::X5, Reg::X5));
        assert_eq!(0x7E30D820, faddp_scalar(Reg::X0, Reg::X1));
        assert_eq!(0x7E30D883, faddp_scalar(Reg::X3, Reg::X4));
    }

    #[test]
    fn fmla_fmls_golden() {
        assert_eq!(0x4E22CC20, fmla_vec(Reg::X0, Reg::X1, Reg::X2));
        assert_eq!(0x4E23CC41, fmla_vec(Reg::X1, Reg::X2, Reg::X3));
        assert_eq!(0x4EA2CC20, fmls_vec(Reg::X0, Reg::X1, Reg::X2));
        assert_eq!(0x4EA3CC41, fmls_vec(Reg::X1, Reg::X2, Reg::X3));
    }

    #[test]
    fn sdot_udot_golden() {
        assert_eq!(0x4E809400, sdot(Reg::X0, Reg::X0, Reg::X0));
        assert_eq!(0x6E809400, udot(Reg::X0, Reg::X0, Reg::X0));
        assert_eq!(0x4E839441, sdot(Reg::X1, Reg::X2, Reg::X3));
        assert_eq!(0x6E839441, udot(Reg::X1, Reg::X2, Reg::X3));
    }

    #[test]
    fn fused_mul_add_golden() {
        assert_eq!(0x1f020c20, fmadd(Reg::X0, Reg::X1, Reg::X2, Reg::X3, false));
        assert_eq!(0x1f420c20, fmadd(Reg::X0, Reg::X1, Reg::X2, Reg::X3, true));
        assert_eq!(0x1f028c20, fmsub(Reg::X0, Reg::X1, Reg::X2, Reg::X3, false));
        assert_eq!(0x1f428c20, fmsub(Reg::X0, Reg::X1, Reg::X2, Reg::X3, true));
        assert_eq!(
            0x1f228c20,
            fnmsub(Reg::X0, Reg::X1, Reg::X2, Reg::X3, false)
        );
        assert_eq!(0x1f628c20, fnmsub(Reg::X0, Reg::X1, Reg::X2, Reg::X3, true));
    }

    #[test]
    fn half_fcvt_golden() {
        assert_eq!(0x1EE24000, fcvt_sfrom_h(Reg::X0, Reg::X0));
        assert_eq!(0x1EE240E5, fcvt_sfrom_h(Reg::X5, Reg::X7));
        assert_eq!(0x1E23C000, fcvt_hfrom_s(Reg::X0, Reg::X0));
        assert_eq!(0x1E23C0E5, fcvt_hfrom_s(Reg::X5, Reg::X7));
        assert_eq!(0x1E63C000, fcvt_hfrom_d(Reg::X0, Reg::X0));
        assert_eq!(0x1E63C0E5, fcvt_hfrom_d(Reg::X5, Reg::X7));
    }

    #[test]
    fn native_fp16_golden() {
        assert_eq!(0x1EE22820, fadd(Reg::X0, Reg::X1, Reg::X2, FKind::Half));
        assert_eq!(0x1EE23820, fsub(Reg::X0, Reg::X1, Reg::X2, FKind::Half));
        assert_eq!(0x1EE20820, fmul(Reg::X0, Reg::X1, Reg::X2, FKind::Half));
        assert_eq!(0x1EE21820, fdiv(Reg::X0, Reg::X1, Reg::X2, FKind::Half));
        assert_eq!(0x1EE22020, fcmp(Reg::X1, Reg::X2, FKind::Half));
        assert_eq!(
            0x1EE21C20,
            fcsel(Reg::X0, Reg::X1, Reg::X2, Cond::Ne, FKind::Half)
        );
        assert_eq!(0x1E222820, fadd(Reg::X0, Reg::X1, Reg::X2, FKind::Single));
        assert_eq!(0x1E622820, fadd(Reg::X0, Reg::X1, Reg::X2, FKind::Double));
    }

    #[test]
    fn half_mem_golden() {
        assert_eq!(0x7D400000, ldr_hfp(Reg::X0, Reg::X0, 0));
        assert_eq!(0x7D400065, ldr_hfp(Reg::X5, Reg::X3, 0));
        assert_eq!(0x7D000000, str_hfp(Reg::X0, Reg::X0, 0));
        assert_eq!(0x7D401041, ldr_hfp(Reg::X1, Reg::X2, 8));
        assert_eq!(0x7D002082, str_hfp(Reg::X2, Reg::X4, 16));
        assert_ne!(ldr_hfp(Reg::X0, Reg::X0, 0), ldrh(Reg::X0, Reg::X0));
        assert_ne!(str_hfp(Reg::X0, Reg::X0, 0), strh(Reg::X0, Reg::X0));
    }

    #[test]
    fn sf_bit_widens_alu() {
        assert_eq!(
            add(Reg::X0, Reg::X1, Reg::X2) | (1 << 31),
            add64(Reg::X0, Reg::X1, Reg::X2)
        );
        assert_eq!(0x8B020020, add64(Reg::X0, Reg::X1, Reg::X2));
        assert_eq!(0xCB020020, sub64(Reg::X0, Reg::X1, Reg::X2));
        assert_eq!(0x9AC22020, lslv64(Reg::X0, Reg::X1, Reg::X2));
    }

    #[test]
    fn shifted_add_sub() {
        assert_eq!(0x8B020C20, add_shifted64(Reg::X0, Reg::X1, Reg::X2, 3));
        assert_eq!(
            add64(Reg::X0, Reg::X1, Reg::X2),
            add_shifted64(Reg::X0, Reg::X1, Reg::X2, 0)
        );
        assert_eq!(
            sub64(Reg::X0, Reg::X1, Reg::X2),
            sub_shifted64(Reg::X0, Reg::X1, Reg::X2, 0)
        );
        assert_eq!(
            add(Reg::X0, Reg::X1, Reg::X2),
            add_shifted(Reg::X0, Reg::X1, Reg::X2, 0)
        );
        assert_eq!(
            sub(Reg::X0, Reg::X1, Reg::X2),
            sub_shifted(Reg::X0, Reg::X1, Reg::X2, 0)
        );
    }

    #[test]
    fn shifted_imm12_frame() {
        assert_eq!(0xD14007FF, sub_imm64_shift(Reg::Zr, Reg::Zr, 1));
        assert_eq!(0x914007FF, add_imm64_shift(Reg::Zr, Reg::Zr, 1));
        assert_eq!(
            sub_imm64(Reg::X0, Reg::X1, 5) | (1 << 22),
            sub_imm64_shift(Reg::X0, Reg::X1, 5)
        );
        assert_eq!(
            add_imm64(Reg::X0, Reg::X1, 5) | (1 << 22),
            add_imm64_shift(Reg::X0, Reg::X1, 5)
        );
    }

    #[test]
    fn sform_flag_setting() {
        assert_eq!(0x6B090108, subs(Reg::X8, Reg::X8, Reg::X9));
        assert_eq!(0xEB090108, subs64(Reg::X8, Reg::X8, Reg::X9));
        assert_eq!(0xAB090108, adds64(Reg::X8, Reg::X8, Reg::X9));
        assert_eq!(0x71000508, subs_imm(Reg::X8, Reg::X8, 1));
        assert_eq!(0xF1000508, subs_imm64(Reg::X8, Reg::X8, 1));
        assert_eq!(0xB1000508, adds_imm64(Reg::X8, Reg::X8, 1));
        assert_eq!(
            add(Reg::X0, Reg::X1, Reg::X2) | (1 << 29),
            adds(Reg::X0, Reg::X1, Reg::X2)
        );
        assert_eq!(
            sub(Reg::X0, Reg::X1, Reg::X2) | (1 << 29),
            subs(Reg::X0, Reg::X1, Reg::X2)
        );
    }

    #[test]
    fn ldp_stp_offset_golden() {
        assert_eq!(0xA9000440, stp_off_x(Reg::X0, Reg::X1, Reg::X2, 0));
        assert_eq!(0xA9410440, ldp_off_x(Reg::X0, Reg::X1, Reg::X2, 16));
        assert_eq!(0x29010440, stp_off_w(Reg::X0, Reg::X1, Reg::X2, 8));
        assert_eq!(0x297F10A3, ldp_off_w(Reg::X3, Reg::X4, Reg::X5, -8));
    }

    #[test]
    fn byte_halfword_offset_golden() {
        assert_eq!(0x39000C20, strb_off(Reg::X0, Reg::X1, 3));
        assert_eq!(0x39401C20, ldrb_off(Reg::X0, Reg::X1, 7));
        assert_eq!(0x39C00420, ldrsb_off(Reg::X0, Reg::X1, 1));
        assert_eq!(0x79000C20, strh_off(Reg::X0, Reg::X1, 6));
        assert_eq!(0x79401420, ldrh_off(Reg::X0, Reg::X1, 10));
        assert_eq!(0x79C00862, ldrsh_off(Reg::X2, Reg::X3, 4));
        assert_eq!(strb(Reg::X0, Reg::X1), strb_off(Reg::X0, Reg::X1, 0));
        assert_eq!(ldrb(Reg::X0, Reg::X1), ldrb_off(Reg::X0, Reg::X1, 0));
        assert_eq!(strh(Reg::X0, Reg::X1), strh_off(Reg::X0, Reg::X1, 0));
        assert_eq!(ldrh(Reg::X0, Reg::X1), ldrh_off(Reg::X0, Reg::X1, 0));
        assert_eq!(ldrsh(Reg::X0, Reg::X1), ldrsh_off(Reg::X0, Reg::X1, 0));
    }
}
