#!/usr/bin/env bash
set -euo pipefail

TARGET="aarch64-unknown-none"

BAREMETAL_CRATES=(
  "volt-platform"
  "volt-target-core"
  "volt-emulator-core"
  "volt-shader-abi"
  "volt-raster"
  "volt-shader"
  "volt-hal"
  "volt-gles"
)

if ! rustup target list --installed | grep -qx "$TARGET"; then
  echo "Target $TARGET not installed. Run: rustup target add aarch64-unknown-none" >&2
  exit 1
fi

for crate in "${BAREMETAL_CRATES[@]}"; do
  echo "==> Checking $crate ($TARGET, --no-default-features)"
  cargo check -p "$crate" --target "$TARGET" --no-default-features
done

echo "==> Building volt-baremetal ($TARGET, release)"
(cd crates/baremetal && cargo build --release)
