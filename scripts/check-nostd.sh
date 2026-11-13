#!/usr/bin/env bash
set -euo pipefail

# Crates that must compile with `--no-default-features` (no_std + alloc).
NO_STD_CRATES=(
  "volt-platform"
  "volt-target-core"
  "volt-emulator-core"
  "volt-shader-abi"
  "volt-raster"
  "volt-shader"
  "volt-hal"
  "volt-gles"
)

# Crates that intentionally require std (build-time compilers, host adapters, binaries, C ABI).
STD_REQUIRED_SKIP_LIST=(
  "volt-ir"
  "volt-codegen"
  "volt-codegen-gfx10"
  "volt-shader-spirv"
  "volt-host"
  "volt-target"
  "volt-emulator"
  "volt-emulator-cli"
  "volt-drm"
  "volt-wsi"
  "volt-gl"
  "volt-vulkan"
)

echo "no_std skip list (std-required): ${STD_REQUIRED_SKIP_LIST[*]}"

for crate in "${NO_STD_CRATES[@]}"; do
  echo "==> Checking $crate (--no-default-features)"
  cargo check -p "$crate" --no-default-features
done

echo "==> Checking volt-shader (--no-default-features --features backend-amdgpu)"
cargo check -p volt-shader --no-default-features --features backend-amdgpu

echo "==> Checking volt-hal (--no-default-features --features amdgpu)"
cargo check -p volt-hal --no-default-features --features amdgpu
