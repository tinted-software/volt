//! Precompiled AMDGPU (GFX10.3 / RDNA2) hardware shader blobs.
//!
//! Shaders are compiled at build time into [`volt_shader_abi`] blobs and
//! embedded via `include_bytes!`. The runtime has zero `pliron` or
//! `volt-codegen` dependencies and stays `no_std`.

use volt_raster::gl;
use volt_shader_abi::ShaderBlob;

mod generated {
    include!(concat!(env!("OUT_DIR"), "/shaders_gen.rs"));
}

/// Byte stride of one vertex in the VS input buffer (`Vertex` is `repr(C)`).
pub const VERTEX_STRIDE: u32 = core::mem::size_of::<volt_raster::Vertex>() as u32;
/// Number of 32-bit user SGPRs the VS consumes (vertex buffer address).
pub const VS_USER_SGPRS: u32 = 2;
/// Number of 32-bit user SGPRs the PS consumes (see [`PsConstants`]).
pub const PS_USER_SGPRS: u32 = 7;
/// Number of VS parameter exports (PS `NUM_INTERP` counts consumed inputs instead).
pub const VS_PARAM_EXPORTS: u32 = 3;

/// Runtime PS constants, passed in user SGPRs in this order.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PsConstants {
    pub fog_color: [f32; 3],
    pub fog_start: f32,
    pub fog_end: f32,
    pub fog_density: f32,
    pub alpha_ref: f32,
}

impl PsConstants {
    pub fn to_sgprs(&self) -> [u32; PS_USER_SGPRS as usize] {
        [
            self.fog_color[0].to_bits(),
            self.fog_color[1].to_bits(),
            self.fog_color[2].to_bits(),
            self.fog_start.to_bits(),
            self.fog_end.to_bits(),
            self.fog_density.to_bits(),
            self.alpha_ref.to_bits(),
        ]
    }
}

/// Precompiled PS variant key. Texturing stays on the CPU path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PsKey {
    /// GL alpha func (`gl::ALWAYS` disables the test).
    pub alpha_func: u32,
    /// 0 = none, 1 = linear, 2 = exp, 3 = exp2.
    pub fog_mode: u32,
}

impl PsKey {
    /// `None` when the fragment state needs features the GPU path lacks.
    pub fn from_state(st: &volt_raster::FragState) -> Option<Self> {
        if st.textures.iter().any(|t| t.enabled) || st.fog_mode > 3 {
            return None;
        }
        Some(Self {
            alpha_func: st.alpha_func,
            fog_mode: st.fog_mode,
        })
    }

    /// Whether this variant can discard pixels (`DB_SHADER_CONTROL.KILL_ENABLE`).
    pub fn uses_kill(&self) -> bool {
        self.alpha_func != gl::ALWAYS
    }
}

/// Map IP-discovery GC 10.3.x (e.g. `0x0a0306`) to a processor name.
pub const fn gfx103_processor(ip_discovery_version: u32) -> &'static str {
    match ip_discovery_version & 0xff {
        0 => "gfx1030",
        1 => "gfx1033",
        2 => "gfx1031",
        3 => "gfx1035",
        4 => "gfx1032",
        5 => "gfx1034",
        6 => "gfx1036",
        7 => "gfx1037",
        _ => "gfx1030",
    }
}

/// Return the precompiled fixed-function vertex shader blob.
pub fn vertex_shader() -> ShaderBlob<'static> {
    ShaderBlob::parse(generated::VS_BLOB).expect("embedded VS shader blob is valid")
}

/// Return the precompiled fixed-function pixel shader blob for `key`.
pub fn pixel_shader(key: &PsKey) -> Option<ShaderBlob<'static>> {
    let raw = generated::lookup_ps_blob(key.fog_mode, key.alpha_func)?;
    Some(ShaderBlob::parse(raw).expect("embedded PS shader blob is valid"))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::shader_ir;

    const SPI_SHADER_PGM_RSRC1_PS: u32 = 0xB028;
    const SPI_SHADER_PGM_RSRC1_VS: u32 = 0xB128;
    const SPI_PS_INPUT_ENA: u32 = 0x286CC;
    const SPI_TMPRING_SIZE: u32 = 0x286E8;

    #[test]
    fn ip_discovery_selects_correct_processor() {
        assert_eq!(gfx103_processor(0x0a0304), "gfx1032");
        assert_eq!(gfx103_processor(0x0a0306), "gfx1036");
    }

    #[test]
    fn embedded_vs_matches_compiler_byte_for_byte() {
        let embedded = vertex_shader();
        assert!(!embedded.code().is_empty() && embedded.code().len().is_multiple_of(4));
        assert!(embedded.reg(SPI_SHADER_PGM_RSRC1_VS).is_some());
        assert_eq!(embedded.reg(SPI_TMPRING_SIZE), Some(0));

        let compiled =
            shader_ir::compile_to_blob(&shader_ir::build_vertex_shader(), "gfx1036", false)
                .unwrap();
        assert_eq!(embedded.to_owned(), compiled);
        assert_eq!(generated::VS_BLOB, compiled.encode().as_slice());
    }

    #[test]
    fn embedded_ps_variants_match_compiler_byte_for_byte() {
        for fog_mode in 0..4 {
            for &alpha_func in &shader_ir::ALPHA_FUNCS {
                let key = PsKey {
                    alpha_func,
                    fog_mode,
                };
                let embedded = pixel_shader(&key).expect("embedded PS variant present");
                assert!(embedded.reg(SPI_SHADER_PGM_RSRC1_PS).is_some());
                assert_eq!(embedded.reg(SPI_PS_INPUT_ENA), Some(0x2), "{key:?}");
                assert_eq!(embedded.reg(SPI_TMPRING_SIZE), Some(0));

                let ir_key = shader_ir::PsKey {
                    alpha_func,
                    fog_mode,
                };
                let compiled = shader_ir::compile_to_blob(
                    &shader_ir::build_pixel_shader(&ir_key),
                    "gfx1032",
                    ir_key.uses_kill(),
                )
                .unwrap();
                assert_eq!(embedded.to_owned(), compiled, "mismatch for {key:?}");
            }
        }
    }
}
