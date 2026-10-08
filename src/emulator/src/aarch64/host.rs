//! Instructions the JIT does not compile. It stores the raw word in `Cpu::value`
//! and traps; the machine (or user mode) then runs the word on the host here.
//! Floating point and the wider integer SIMD families live in [`fp`] and [`simd`].

use super::{Cpu, fp, pauth, simd};

/// Whether [`run`] implements `word`.
pub fn is_supported(word: u32) -> bool {
    fp::is_supported(word) || simd::is_supported(word) || pauth::is_supported(word)
}

/// Run `word` on `cpu`. `word` must satisfy [`is_supported`].
pub fn run(cpu: &mut Cpu, word: u32) {
    if fp::is_supported(word) {
        fp::run(cpu, word)
    } else if simd::is_supported(word) {
        simd::run(cpu, word)
    } else {
        pauth::run(cpu, word)
    }
}
