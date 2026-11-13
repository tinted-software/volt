#!/usr/bin/env bash
set -euo pipefail

cargo build -p volt-gl -p volt-vulkan

UNAME="$(uname -s)"
if [ "$UNAME" = "Darwin" ]; then
  GL_LIB="target/debug/libvolt_gl.dylib"
  VK_LIB="target/debug/libvolt_vk.dylib"
  GL_SYMS="$(nm -gU "$GL_LIB" | awk '{print $3}' | sed 's/^_//' | grep -E '^(egl|gl|gbm|angle_wgpu|__egl_Main)' | sort -u)"
  VK_SYMS="$(nm -gU "$VK_LIB" | awk '{print $3}' | sed 's/^_//' | grep -E '^vk' | sort -u)"
else
  GL_LIB="target/debug/libvolt_gl.so"
  VK_LIB="target/debug/libvolt_vk.so"
  GL_SYMS="$(nm -D --defined-only "$GL_LIB" | awk '{print $3}' | grep -E '^(egl|gl|gbm|angle_wgpu|__egl_Main)' | sort -u)"
  VK_SYMS="$(nm -D --defined-only "$VK_LIB" | awk '{print $3}' | grep -E '^vk' | sort -u)"
fi

check_list() {
  local label="$1"
  local actual="$2"
  local expected_file="$3"
  local missing
  missing="$(comm -23 <(sort -u "$expected_file") <(printf '%s\n' "$actual"))"
  if [ -n "$missing" ]; then
    echo "Missing exported symbols in $label ($expected_file):" >&2
    printf '%s\n' "$missing" >&2
    exit 1
  fi
  echo "OK: $label matches $(wc -l < "$expected_file" | tr -d ' ') expected symbols"
}

check_list "volt-gl (EGL+GL)" "$GL_SYMS" "docs/c_abi_symbols_gl.txt"
if [ "$UNAME" = "Linux" ]; then
  check_list "volt-gl (GBM)" "$GL_SYMS" "docs/c_abi_symbols_gbm.txt"
fi
check_list "volt-vulkan" "$VK_SYMS" "docs/c_abi_symbols_vulkan.txt"

# Verify loader boundary: volt-gl must not export vk*, volt-vulkan must not export egl*/gl*.
if printf '%s\n' "$GL_SYMS" | grep -qE '^vk'; then
  echo "ERROR: volt-gl exported vk* symbols" >&2
  exit 1
fi
if printf '%s\n' "$VK_SYMS" | grep -qE '^(egl|gl|gbm)'; then
  echo "ERROR: volt-vulkan exported GL/EGL/GBM symbols" >&2
  exit 1
fi
