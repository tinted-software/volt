//! Vantage Shader — fixed-function fragment evaluation and AMDGPU shader programs.
//!
//! Two responsibilities:
//!
//! * **Fragment pipeline** (always compiled): a portable scalar evaluator for
//!   the fixed-function fragment stage — per-unit TexEnv, alpha-test discard
//!   and fog blend — bound into the software rasterizer through the
//!   [`vantage_raster::FragFn`] seam. `vantage_raster::reference_frag` is an
//!   independent implementation of the same contract and the differential
//!   oracle for the tests below.
//! * **AMDGPU programs** (`backend-amdgpu`): the legacy VS and the
//!   fixed-function PS are built in pliron's LLVM dialect and compiled to
//!   GFX10.3 machine code by `vantage-codegen`. Nothing here links LLVM and
//!   nothing emits an ELF container: the driver loads the raw machine code plus
//!   the compiler's `(register, value)` loader configuration directly.
#![no_std]
extern crate alloc;

pub use volt_raster::{FragFn, FragState, SampledTexture, Varyings, gl};

#[cfg(feature = "backend-amdgpu")]
pub mod amdgcn;
#[cfg(all(test, feature = "backend-amdgpu"))]
mod shader_ir;
// ============================================================================
// Helper evaluation functions
// ============================================================================

#[inline]
fn clamp_unit(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

#[inline]
fn eval_alpha_pass(func: u32, a: f32, reference: f32) -> bool {
    match func {
        gl::NEVER => false,
        gl::LESS => a < reference,
        gl::EQUAL => a == reference,
        gl::LEQUAL => a <= reference,
        gl::GREATER => a > reference,
        gl::NOTEQUAL => a != reference,
        gl::GEQUAL => a >= reference,
        gl::ALWAYS => true,
        _ => true,
    }
}

#[inline]
fn apply_texenv(mode: u32, tex: [f32; 4], color: [f32; 4], env: [f32; 4]) -> [f32; 4] {
    match mode {
        gl::TEXENV_REPLACE => tex,
        gl::TEXENV_MODULATE => [
            tex[0] * color[0],
            tex[1] * color[1],
            tex[2] * color[2],
            tex[3] * color[3],
        ],
        gl::TEXENV_DECAL => {
            let a = tex[3];
            [
                color[0] * (1.0 - a) + tex[0] * a,
                color[1] * (1.0 - a) + tex[1] * a,
                color[2] * (1.0 - a) + tex[2] * a,
                color[3],
            ]
        }
        gl::TEXENV_BLEND => [
            color[0] * (1.0 - tex[0]) + env[0] * tex[0],
            color[1] * (1.0 - tex[1]) + env[1] * tex[1],
            color[2] * (1.0 - tex[2]) + env[2] * tex[2],
            color[3] * tex[3],
        ],
        gl::TEXENV_ADD => [
            (color[0] + tex[0]).min(1.0),
            (color[1] + tex[1]).min(1.0),
            (color[2] + tex[2]).min(1.0),
            (color[3] * tex[3]).min(1.0),
        ],
        _ => [
            tex[0] * color[0],
            tex[1] * color[1],
            tex[2] * color[2],
            tex[3] * color[3],
        ],
    }
}

#[inline]
fn apply_fog_eval(state: &FragState, color: &mut [f32; 4], fog: f32) {
    let f = match state.fog_mode {
        1 => (state.fog_end - fog) / (state.fog_end - state.fog_start),
        2 => libm::expf(-state.fog_density * fog),
        3 => {
            let d = state.fog_density * fog;
            libm::expf(-(d * d))
        }
        _ => return,
    };
    let f = clamp_unit(f);
    for i in 0..3 {
        color[i] = color[i] * f + state.fog_color[i] * (1.0 - f);
    }
}

#[inline(always)]
fn wrap_c(x: f32, mode: u32) -> f32 {
    if mode == gl::REPEAT {
        if (0.0..1.0).contains(&x) {
            x
        } else {
            let i = x as i32;
            let f = x - (i as f32);
            if f < 0.0 { f + 1.0 } else { f }
        }
    } else if mode == gl::CLAMP_TO_EDGE {
        clamp_unit(x)
    } else {
        let f = if (0.0..1.0).contains(&x) {
            x
        } else {
            x - libm::floorf(x)
        };
        if (libm::floorf(x) as i64) & 1 == 0 {
            f
        } else {
            1.0 - f
        }
    }
}

#[inline(always)]
fn sample_tex(tex: &SampledTexture, u: f32, v: f32, linear: bool) -> [f32; 4] {
    if !tex.enabled || tex.width == 0 || tex.height == 0 || tex.data.is_null() {
        return [0.0, 0.0, 0.0, 1.0];
    }
    let fu = wrap_c(u, tex.wrap_s) * tex.width as f32;
    let fv = wrap_c(v, tex.wrap_t) * tex.height as f32;

    let fetch_texel = |x: i64, y: i64| -> [f32; 4] {
        let x = x.clamp(0, tex.width as i64 - 1);
        let y = y.clamp(0, tex.height as i64 - 1);
        let o = ((y * tex.width as i64 + x) * 4) as usize;
        if o + 4 > tex.data_len {
            return [0.0, 0.0, 0.0, 1.0];
        }
        unsafe {
            let p = tex.data.add(o);
            [
                *p as f32 / 255.0,
                *p.add(1) as f32 / 255.0,
                *p.add(2) as f32 / 255.0,
                *p.add(3) as f32 / 255.0,
            ]
        }
    };

    let [r, g, b, a] = if linear {
        let x0 = libm::floorf(fu - 0.5) as i64;
        let y0 = libm::floorf(fv - 0.5) as i64;
        let tx = fu - 0.5 - x0 as f32;
        let ty = fv - 0.5 - y0 as f32;
        let c00 = fetch_texel(x0, y0);
        let c10 = fetch_texel(x0 + 1, y0);
        let c01 = fetch_texel(x0, y0 + 1);
        let c11 = fetch_texel(x0 + 1, y0 + 1);
        let mut out = [0.0f32; 4];
        for i in 0..4 {
            let l0 = c00[i] + (c10[i] - c00[i]) * tx;
            let l1 = c01[i] + (c11[i] - c01[i]) * tx;
            out[i] = l0 + (l1 - l0) * ty;
        }
        out
    } else {
        let x = libm::floorf(fu) as i64;
        let y = libm::floorf(fv) as i64;
        fetch_texel(x, y)
    };

    match tex.format {
        gl::GL_LUMINANCE => [r, r, r, 1.0],
        gl::GL_LUMINANCE_ALPHA => [r, r, r, a],
        _ => [r, g, b, a],
    }
}

// ============================================================================
// Fragment entry point
// ============================================================================

/// Portable scalar evaluation of the fixed-function fragment pipeline.
///
/// Writes the pre-blend BGRA8 color and returns whether the fragment passes the
/// alpha test (`false` = discard). Bound directly as the rasterizer's
/// [`FragFn`]: the pipeline is a pure function of [`FragState`] and the
/// interpolated [`Varyings`], so no per-draw lookup or code generation is
/// involved. Blending, depth and stencil handling stay in the raster core.
///
/// # Safety
///
/// `ctx` must point to a live [`FragState`], `varying` to a live [`Varyings`],
/// and `out` to a 4-byte writable array. Any [`SampledTexture`] with
/// `enabled == true` in `ctx` must have `data` pointing to at least `data_len`
/// readable bytes.
pub unsafe fn fragment_entry_point(
    ctx: *const FragState,
    varying: *const Varyings,
    out: *mut [u8; 4],
) -> bool {
    unsafe {
        let st = &*ctx;
        let v = &*varying;
        let mut color = v.color;

        // 1. TexEnv evaluation for units 0 and 1.
        for unit in 0..2 {
            let tex = &st.textures[unit];
            if !tex.enabled {
                continue;
            }
            let tc = if unit == 0 { v.tex0 } else { v.tex1 };
            let linear = tex.mag_filter != gl::NEAREST;
            let s = sample_tex(tex, tc[0], tc[1], linear);
            color = apply_texenv(st.texenv_mode[unit], s, color, st.texenv_color[unit]);
        }

        // 2. Fog evaluation.
        if st.fog_mode != 0 {
            apply_fog_eval(st, &mut color, v.fog);
        }

        // 3. Alpha test.
        if !eval_alpha_pass(st.alpha_func, color[3], st.alpha_ref) {
            return false;
        }

        let o = &mut *out;
        // Write BGRA8 output (pre-blend).
        o[0] = (clamp_unit(color[2]) * 255.0 + 0.5) as u8; // B
        o[1] = (clamp_unit(color[1]) * 255.0 + 0.5) as u8; // G
        o[2] = (clamp_unit(color[0]) * 255.0 + 0.5) as u8; // R
        o[3] = (clamp_unit(color[3]) * 255.0 + 0.5) as u8; // A
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evaluate(state: &FragState, varying: &Varyings) -> (bool, [u8; 4]) {
        let mut out = [0u8; 4];
        let pass =
            unsafe { fragment_entry_point(state, varying, out.as_mut_ptr() as *mut [u8; 4]) };
        (pass, out)
    }

    fn reference(state: &FragState, varying: &Varyings) -> (bool, [u8; 4]) {
        let mut out = [0u8; 4];
        let pass = unsafe {
            volt_raster::reference_frag(state, varying, out.as_mut_ptr() as *mut [u8; 4])
        };
        (pass, out)
    }

    /// 1x1 texture: with a single texel every wrap mode and filter agree, so
    /// this isolates TexEnv/format handling from filter-footprint wrapping.
    fn one_texel(data: &mut [u8], format: u32, mag_filter: u32) -> SampledTexture {
        SampledTexture {
            data: data.as_mut_ptr(),
            data_len: data.len(),
            width: 1,
            height: 1,
            format,
            min_filter: gl::NEAREST,
            mag_filter,
            wrap_s: gl::REPEAT,
            wrap_t: gl::REPEAT,
            enabled: true,
        }
    }

    fn base_state() -> FragState {
        FragState {
            textures: [SampledTexture::disabled(), SampledTexture::disabled()],
            texenv_mode: [gl::TEXENV_MODULATE, gl::TEXENV_MODULATE],
            texenv_color: [[0.25, 0.5, 0.75, 1.0], [0.5, 0.25, 0.75, 1.0]],
            alpha_func: gl::ALWAYS,
            alpha_ref: 0.0,
            fog_mode: 0,
            fog_start: 0.0,
            fog_end: 10.0,
            fog_density: 0.3,
            fog_color: [0.5, 0.5, 0.5, 1.0],
        }
    }

    #[test]
    fn fragment_pipeline_matches_reference_across_alpha_tests() {
        let mut state = base_state();
        let varying = Varyings {
            color: [0.3, 0.4, 0.5, 0.75],
            tex0: [0.0, 0.0],
            tex1: [0.0, 0.0],
            fog: 0.0,
            view_z: 1.0,
        };
        for alpha_func in [
            gl::NEVER,
            gl::LESS,
            gl::EQUAL,
            gl::LEQUAL,
            gl::GREATER,
            gl::NOTEQUAL,
            gl::GEQUAL,
            gl::ALWAYS,
        ] {
            for alpha_ref in [0.0, 0.75, 1.0] {
                state.alpha_func = alpha_func;
                state.alpha_ref = alpha_ref;
                assert_eq!(
                    evaluate(&state, &varying),
                    reference(&state, &varying),
                    "alpha_func={alpha_func:#x} alpha_ref={alpha_ref}"
                );
            }
        }
    }

    #[test]
    fn fragment_pipeline_matches_reference_across_fog_modes() {
        let mut state = base_state();
        let varying = Varyings {
            color: [0.3, 0.4, 0.5, 1.0],
            tex0: [0.0, 0.0],
            tex1: [0.0, 0.0],
            fog: 0.7,
            view_z: 1.0,
        };
        // Mode 2 (EXP) is the case that used to need an `expf` relocation in
        // the retired JIT backend; every mode is checked here.
        for fog_mode in 0..4 {
            for fog in [0.0, 0.7, 10.0, 25.0] {
                state.fog_mode = fog_mode;
                state.fog_density = 0.3;
                let varying = Varyings { fog, ..varying };
                assert_eq!(
                    evaluate(&state, &varying),
                    reference(&state, &varying),
                    "fog_mode={fog_mode} fog={fog}"
                );
            }
        }
    }

    #[test]
    fn fragment_pipeline_matches_reference_across_texenv_and_formats() {
        let mut texel = [255u8, 128, 64, 200];
        let varying = Varyings {
            color: [0.8, 0.6, 0.4, 0.9],
            tex0: [0.5, 0.5],
            tex1: [0.0, 0.0],
            fog: 0.0,
            view_z: 1.0,
        };
        for format in [gl::GL_RGBA, gl::GL_LUMINANCE, gl::GL_LUMINANCE_ALPHA] {
            for mag_filter in [gl::NEAREST, gl::LINEAR] {
                for mode in [
                    gl::TEXENV_REPLACE,
                    gl::TEXENV_MODULATE,
                    gl::TEXENV_DECAL,
                    gl::TEXENV_BLEND,
                    gl::TEXENV_ADD,
                ] {
                    let mut state = base_state();
                    state.textures[0] = one_texel(&mut texel, format, mag_filter);
                    state.texenv_mode = [mode, gl::TEXENV_REPLACE];
                    assert_eq!(
                        evaluate(&state, &varying),
                        reference(&state, &varying),
                        "format={format:#x} mag_filter={mag_filter:#x} texenv={mode:#x}"
                    );
                }
            }
        }
    }

    /// CLAMP_TO_EDGE is the one wrap mode where the filter footprint is
    /// clamped by every implementation, so it exercises the bilinear taps and
    /// their weighting without depending on seam wrapping policy.
    #[test]
    fn fragment_pipeline_matches_reference_across_bilinear_weights() {
        let mut texels = [
            10u8, 20, 30, 255, //
            40, 50, 60, 255, //
            70, 80, 90, 255, //
            100, 110, 120, 255,
        ];
        let mut state = base_state();
        for mag_filter in [gl::NEAREST, gl::LINEAR] {
            state.textures[0] = SampledTexture {
                data: texels.as_mut_ptr(),
                data_len: texels.len(),
                width: 2,
                height: 2,
                format: gl::GL_RGBA,
                min_filter: gl::NEAREST,
                mag_filter,
                wrap_s: gl::CLAMP_TO_EDGE,
                wrap_t: gl::CLAMP_TO_EDGE,
                enabled: true,
            };
            for u in [0.0, 0.25, 0.5, 0.75, 1.0] {
                for v in [0.0, 0.25, 0.5, 0.75, 1.0] {
                    let varying = Varyings {
                        color: [1.0, 1.0, 1.0, 1.0],
                        tex0: [u, v],
                        tex1: [0.0, 0.0],
                        fog: 0.0,
                        view_z: 1.0,
                    };
                    assert_eq!(
                        evaluate(&state, &varying),
                        reference(&state, &varying),
                        "mag_filter={mag_filter:#x} uv=({u}, {v})"
                    );
                }
            }
        }
    }

    #[test]
    fn disabled_alpha_test_streams_color_through() {
        let state = base_state();
        let varying = Varyings {
            color: [1.0, 0.0, 0.0, 1.0],
            tex0: [0.0, 0.0],
            tex1: [0.0, 0.0],
            fog: 0.0,
            view_z: 1.0,
        };
        assert_eq!(evaluate(&state, &varying), (true, [0, 0, 255, 255]));
    }
}
