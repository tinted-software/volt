#!/bin/sh
# Builds and runs the DRI3 smoke against libvolt_gl.so via glvnd-style paths:
# libEGL dispatches to our vendor JSON (see glvnd/egl_vendor.d).
cc -O1 -o dri3_smoke_main main.c \
  -I../../include \
  -lX11 \
  -L../../target/debug -lvolt_gl \
  -Wl,-rpath,"$PWD/../../target/debug"
