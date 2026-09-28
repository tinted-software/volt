# volt-jit

Independent Rust workspace for modular dynamic binary translation.

## Crates

- `volt-jit-decoder`: ISA-neutral decoder trait and rejected-word error.
- `volt-jit-aarch64`: A64 instruction decoder and per-vCPU integer state. This crate does not depend on LLVM.
- `volt-jit-aarch64-llvm`: A64 block lowering and executable block API. Translates a contiguous little-endian instruction stream, terminates at the first branch/return, and advances `Cpu.pc` after execution.
- `volt-jit-llvm`: Pliron LLVM-dialect verification/conversion and native LLVM JIT ownership. It has no AArch64 dependency.
- `volt-jit`: facade and caller-owned guest-memory contract.

The current native A64 lowering executes NOP, MOVZ, MOVK, ADD/SUB immediate without flags or SP operands, B, and RET (X0–X30). Decoding additionally recognizes B.cond, but lowering rejects it explicitly. No MMU, guest load/store, exceptions, system instructions, or code cache is implemented. This is **not** yet a replacement for QEMU TCG or a complete AArch64 emulator.

`Block::compile(pc, bytes)` decodes A64 bytes, constructs a function module, converts its instructions through the Pliron LLVM dialect, and compiles it with LLVM's native JIT. `Block::run(&mut Cpu)` updates general-purpose registers and the next PC. A future ARMv6, x86_64, or RISC-V frontend can supply its own decoder/state/lowering crate without changing the LLVM backend.

## Build and smoke test

Pliron 0.18 requires LLVM 23.1. Point `LLVM_SYS_231_PREFIX` to an LLVM **development** output containing `bin/llvm-config`, headers, and libraries; pointing it to the binaries-only output is insufficient. For example:

```sh
LLVM_SYS_231_PREFIX=/path/to/llvm-23.1-dev cargo test --workspace
```

The workspace tests execute translated A64 MOVZ/MOVK/ADD/RET and backward-branch blocks on the host, in addition to testing decoding and the Pliron-to-LLVM JIT boundary.
