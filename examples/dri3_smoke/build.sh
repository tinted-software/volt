#!/bin/sh
# Builds and runs the DRI3 smoke against libvolt_gl.so.
cc -O1 -o dri3_smoke dri3_smoke.c \
  -I../../include \
  -lX11 \
  -L../../target/debug -lvolt_gl \
  -Wl,-rpath,"$PWD/../../target/debug"
