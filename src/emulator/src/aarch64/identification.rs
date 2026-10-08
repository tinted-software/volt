use super::decode::SystemRegister;
#[allow(non_upper_case_globals)]
pub const counter_hz: u64 = 62_500_000;
/// Cache hierarchy `CLIDR_EL1` reports: split L1, unified L2, 64-byte lines.
/// LoC = 2, LoU = 1.
pub const CLIDR: u64 = 3 | 4 << 3 | 2 << 24 | 1 << 27;
/// `CCSIDR_EL1` (no CCIDX) for the cache `CSSELR_EL1` selects: cache type in
/// bits 31:28 (4 = write-back, 2 = read-allocate), sets - 1, ways - 1, and a
/// `LineSize` of 2 (64 bytes).
const fn ccsidr(cache_type: u64, sets: u64, ways: u64) -> u64 {
    cache_type << 28 | (sets - 1) << 13 | (ways - 1) << 3 | 2
}
/// L1 data: 32 KiB, 4-way.
pub const CCSIDR_L1D: u64 = ccsidr(4, 128, 4);
/// L1 instruction: 32 KiB, 4-way.
pub const CCSIDR_L1I: u64 = ccsidr(2, 128, 4);
/// L2 unified: 1 MiB, 8-way.
pub const CCSIDR_L2: u64 = ccsidr(4, 2048, 8);
/// The virt board's `MIDR_EL1`, `ID_AA64PFR0_EL1` and `ID_AA64MMFR0_EL1`. A board
/// may advertise different identity, so `value` leaves these three to the CPU state.
pub const DEFAULT_MIDR_EL1: u64 = 0x410f_d0f0;
pub const DEFAULT_ID_AA64PFR0_EL1: u64 = (0xf << 20) | (0xf << 16) | (1 << 4) | 1;
pub const DEFAULT_ID_AA64MMFR0_EL1: u64 = (0xf << 24) | 2;
pub fn value(register: SystemRegister) -> Option<u64> {
    use SystemRegister::*;
    Some(match register {
        MpidrEl1 => 0x8000_0000,
        CtrEl0 => {
            (1 << 31) | (1 << 29) | (1 << 28) | (4 << 24) | (4 << 20) | (4 << 16) | (3 << 14) | 4
        }
        DczidEl0 => 4,
        CntfrqEl0 => counter_hz,
        IdAa64dfr0El1 => 6,
        IdAa64isar0El1 => 2 << 20,
        IdAa64isar1El1 => 1 << 20,
        ClidrEl1 => CLIDR,
        RevidrEl1 | IdAa64pfr1El1 | IdAa64pfr2El1 | IdAa64zfr0El1 | IdAa64smfr0El1
        | IdAa64fpfr0El1 | IdAa64isar3El1 | AidrEl1 | IdAa64dfr1El1 | IdAa64isar2El1
        | IdAa64mmfr1El1 | IdAa64mmfr2El1 | IdAa64mmfr3El1 | IdAa64mmfr4El1 => 0,
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
        let pfr0 = DEFAULT_ID_AA64PFR0_EL1;
        assert_eq!(pfr0 & 0xffff, 0x11);
        assert_eq!((pfr0 >> 16) & 0xff, 0xff);
        assert_eq!((pfr0 >> 24) & 15, 0);
        let mmfr0 = DEFAULT_ID_AA64MMFR0_EL1;
        assert_eq!(mmfr0 & 15, 2);
        assert_eq!((mmfr0 >> 24) & 15, 15);
        assert_eq!((mmfr0 >> 20) & 15, 0);
        // FEAT_LSE (Atomics = 2) and FEAT_LRCPC (LDAPR) are implemented.
        assert_eq!(value(IdAa64isar0El1), Some(2 << 20));
        assert_eq!(value(IdAa64isar1El1), Some(1 << 20));
        assert_eq!(value(MpidrEl1), Some(1 << 31));
        assert_eq!(value(Ttbr0El1), None);
    }
    #[test]
    fn cache_geometry_decodes_like_xnu_cpuid() {
        // The size arithmetic of `osfmk/arm/cpuid.c`.
        let size = |ccsidr: u64| {
            let line = 4 * (1u64 << ((ccsidr & 7) + 2));
            (((ccsidr >> 13) & 0x7fff) + 1) * line * (((ccsidr >> 3) & 0x3ff) + 1)
        };
        assert_eq!(size(CCSIDR_L1D), 32 << 10);
        assert_eq!(size(CCSIDR_L1I), 32 << 10);
        assert_eq!(size(CCSIDR_L2), 1 << 20);
        // Separate L1 I/D (Ctype1 = 3), unified L2 (Ctype2 = 4), nothing beyond.
        assert_eq!(CLIDR & 0x1ff, 3 | 4 << 3);
        // XNU treats c_type 4 as write-back; the L1 data and L2 caches are.
        assert_eq!(CCSIDR_L1D >> 28, 4);
        assert_eq!(CCSIDR_L2 >> 28, 4);
    }
}
