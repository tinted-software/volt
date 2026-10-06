use super::decode::SystemRegister;
#[allow(non_upper_case_globals)]
pub const counter_hz: u64 = 62_500_000;
pub fn value(register: SystemRegister) -> Option<u64> {
    use SystemRegister::*;
    Some(match register {
        MidrEl1 => 0x410f_d0f0,
        MpidrEl1 => 0x8000_0000,
        CtrEl0 => {
            (1 << 31) | (1 << 29) | (1 << 28) | (4 << 24) | (4 << 20) | (4 << 16) | (3 << 14) | 4
        }
        DczidEl0 => 4,
        CntfrqEl0 => counter_hz,
        IdAa64pfr0El1 => (0xf << 20) | (0xf << 16) | (1 << 4) | 1,
        IdAa64dfr0El1 => 6,
        IdAa64mmfr0El1 => (0xf << 24) | 2,
        RevidrEl1 | ClidrEl1 | IdAa64pfr1El1 | IdAa64pfr2El1 | IdAa64zfr0El1 | IdAa64smfr0El1
        | IdAa64fpfr0El1 | IdAa64isar3El1 | AidrEl1 | IdAa64dfr1El1 | IdAa64isar0El1
        | IdAa64isar1El1 | IdAa64isar2El1 | IdAa64mmfr1El1 | IdAa64mmfr2El1 | IdAa64mmfr3El1
        | IdAa64mmfr4El1 => 0,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn advertised_features_match_integer_only_single_core_machine() {
        use SystemRegister::*;
        assert_eq!(value(CntfrqEl0), Some(62_500_000));
        assert_eq!(value(DczidEl0), Some(4));
        let pfr0 = value(IdAa64pfr0El1).unwrap();
        assert_eq!(pfr0 & 0xffff, 0x11);
        assert_eq!((pfr0 >> 16) & 0xff, 0xff);
        assert_eq!((pfr0 >> 24) & 15, 0);
        let mmfr0 = value(IdAa64mmfr0El1).unwrap();
        assert_eq!(mmfr0 & 15, 2);
        assert_eq!((mmfr0 >> 24) & 15, 15);
        assert_eq!((mmfr0 >> 20) & 15, 0);
        assert_eq!(value(IdAa64isar0El1), Some(0));
        assert_eq!(value(MpidrEl1), Some(1 << 31));
        assert_eq!(value(Ttbr0El1), None);
    }
}
