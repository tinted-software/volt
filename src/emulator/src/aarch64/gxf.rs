//! Guarded execution (GXF): `genter` and `gexit`, the instructions that move between
//! EL and the guarded levels GL1 and GL2 (Asahi Linux documentation, "Apple Proprietary
//! Instructions" and "SPRR and GXF").
//!
//! The feature is off by default and only a monitor firmware (SPTM or TXM) enables it,
//! which this model does not run. Its enable bit reads as zero, so both instructions
//! take an undefined-instruction exception to EL1. The documentation says they "fault by
//! default" without naming the exception class; undefined-instruction is an assumption.

use super::{Cpu, exception};

/// `gexit`: leave guarded mode.
const GEXIT: u32 = 0x0020_1400;
/// `genter` with its `imm5` in bits 4:0 (stored in `ESR_GLx[5:0]`).
const GENTER: u32 = 0x0020_1420;

/// Whether `word` is `genter` or `gexit`.
pub fn is_supported(word: u32) -> bool {
    word == GEXIT || word & !0x1f == GENTER
}

/// Take the undefined-instruction exception `genter` or `gexit` raises while GXF is off.
/// `cpu.pc` is the address of the next instruction; the exception returns to the
/// instruction itself.
pub fn run(cpu: &mut Cpu, word: u32) {
    debug_assert!(is_supported(word));
    cpu.pc = cpu.pc.wrapping_sub(4);
    let pc = cpu.pc;
    exception::take(
        cpu,
        exception::Reason {
            ec: 0,
            instruction: true,
            status: 0,
            ..Default::default()
        },
        pc,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn genter_and_gexit_are_recognised_with_any_genter_immediate() {
        assert!(is_supported(GEXIT));
        assert!(is_supported(GENTER));
        assert!(is_supported(GENTER | 0x1f));
        assert!(!is_supported(0xd503_201f));
        assert!(!is_supported(GEXIT + 4));
    }

    #[test]
    fn with_guarded_execution_off_genter_is_an_undefined_instruction_at_the_instruction() {
        let mut cpu = Cpu::default();
        cpu.system.el = 1;
        cpu.system.spsel = true;
        cpu.system.vbar_el1 = 0x4000_1000;
        // The executor runs after the instruction at 0x4000_0000 has advanced `pc`.
        cpu.pc = 0x4000_0004;
        run(&mut cpu, GENTER);
        assert_eq!(
            cpu.system.elr_el1, 0x4000_0000,
            "returns to the genter itself"
        );
        assert_eq!(cpu.system.esr_el1 >> 26, 0, "EC is unknown reason");
        assert_eq!(
            cpu.pc,
            0x4000_1000 + 0x200,
            "current EL with SP_ELx, synchronous"
        );
    }
}
