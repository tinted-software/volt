# Migration Baseline

Recorded on macOS (arm64, darwin 27.3.0) before Phase 1 restructuring.

## Workspace status
- `volt-ir`: `cargo check -p volt-ir` passes.
- `volt-target`: **FAILS** on macOS (`E0433: cannot find module or crate arena in this scope` at `src/target/src/native.rs:67` because `chunk: Option<std::sync::Arc<arena::Chunk>>` is unconditional while `mod arena` is `#[cfg(target_os = "linux")]`).
- `volt-emulator` / `volt-emulator-cli`: blocked on `volt-target`.
- `no_std` crates: none yet (`volt-ir` has `pliron` std dependency).
- `aarch64-unknown-none` crates: none yet; `aarch64-unknown-none` toolchain target installed.

## Upstream Vantage source
- Source: `/tmp/vantage` at commit `24fa4c5` (`feat: pure-Rust AMDGPU (GFX10.3) shader codegen and hardware rasterization backend`).
- Crates: `vantage-{codegen,drm,egl,gbm,gles,hal,raster,shader,vulkan}`.
