//! Instructions the JIT does not compile. It stores the raw word in `Cpu::value`
//! and traps; the machine (or user mode) then runs the word on the host here.
//! Floating point and the wider integer SIMD families live in [`fp`] and [`simd`].
//!
//! Which words these are is decided by the one decoder, `volt_isa_aarch64::decode`:
//! the instruction variants [`runs_on_host`] names.

use super::{Cpu, fp, gxf, pauth, simd};
use volt_isa_aarch64::decode::{Instruction, decode};

/// Whether the emulator runs `instruction` on the host instead of lifting it.
pub fn runs_on_host(instruction: &Instruction) -> bool {
    matches!(
        instruction,
        Instruction::Fp(_) | Instruction::Simd(_) | Instruction::Pauth(_) | Instruction::Gxf(_)
    )
}

/// Whether the emulator ends a translated block after `instruction`.
///
/// This is policy of this emulator's [`CpuEnvironment`](super::environment), not a
/// property of the instruction set: everything the environment forwards to the
/// machine (memory, exclusives, atomics, system events, words run on the host) must
/// leave the block so the machine can run it, and so must every architectural
/// branch. With `inline_memory`, plain loads and stores are compiled with an inline
/// data-TLB fast path whose slow path exits mid-block, so they no longer end it.
pub fn ends_block(instruction: &Instruction, inline_memory: bool) -> bool {
    use Instruction::*;
    if runs_on_host(instruction) {
        return true;
    }
    if inline_memory
        && matches!(
            instruction,
            Memory(_) | Pair(_) | Literal(_) | SimdMemory(_) | SimdPair(_)
        )
    {
        return false;
    }
    matches!(
        instruction,
        Memory(_)
            | Exclusive(_)
            | Atomic(_)
            | Literal(_)
            | SimdLiteral(_)
            | Pair(_)
            | SimdPair(_)
            | SimdMemory(_)
            | SimdStruct(_)
            | B(_)
            | BCond(_)
            | TestBranch(_)
            | Call(_)
            | Indirect(_)
            | Svc(_)
            | Psci
            | Wfi
            | Wfe
            | Brk(_)
            | Isb(_)
            | DcZva(_)
            | AddressTranslate(_)
            | Eret
            | Tlbi(_)
    )
}

/// Run `word` on `cpu`. `word` must decode to an instruction [`runs_on_host`] names.
pub fn run(cpu: &mut Cpu, word: u32) {
    match decode(word) {
        Ok(Instruction::Fp(op)) => fp::run(cpu, op),
        Ok(Instruction::Simd(op)) => simd::run(cpu, op),
        Ok(Instruction::Pauth(op)) => pauth::run(cpu, op),
        Ok(Instruction::Gxf(op)) => gxf::run(cpu, op),
        other => unreachable!("word {word:08x} is not a host instruction: {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_policy_cuts_at_machine_effects_and_branches() {
        let word = |w| decode(w).unwrap();
        assert!(!ends_block(&word(0xd503309f), false)); // dsb
        assert!(ends_block(&word(0xd5033fdf), false)); // isb
        assert!(ends_block(&word(0xd69f03e0), false)); // eret
        assert!(ends_block(&word(0x1400_0000), false)); // b .
        assert!(ends_block(&word(0x1e22_2820), true)); // fadd s0, s1, s2: runs on the host
    }

    #[test]
    fn inline_memory_only_lets_plain_accesses_continue() {
        let load = decode(0xf940_0020).unwrap(); // ldr x0, [x1]
        let exclusive = decode(0xc85f_7c20).unwrap(); // ldxr x0, [x1]
        assert!(ends_block(&load, false));
        assert!(!ends_block(&load, true));
        assert!(ends_block(&exclusive, true));
    }

    #[test]
    fn floating_point_pointer_authentication_and_integer_simd_run_on_the_host() {
        assert!(runs_on_host(&decode(0x1e22_2820).unwrap())); // fadd s0, s1, s2
        assert!(runs_on_host(&decode(0xdac1_0020).unwrap())); // pacia x0, x1
        assert!(runs_on_host(&decode(0x4e82_1820).unwrap())); // uzp1 v0.4s, v1.4s, v2.4s
        assert!(!runs_on_host(&decode(0x8b02_0020).unwrap())); // add x0, x1, x2: lifted
    }
}
