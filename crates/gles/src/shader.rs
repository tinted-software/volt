//! Fixed-function uniform layout and GLES2 shader/program bookkeeping.
//!
//! The WGSL fixed-function shader that used to live here was the behavioral
//! spec for all per-fragment math (texenv, alpha test, fog, the
//! `(z+w)*0.5` clip-Z remap); that math now lives in `vantage-shader`'s
//! portable fragment pipeline.

use alloc::string::String;
use alloc::vec::Vec;
use hashbrown::HashMap;

/// Memory layout for fixed-function uniform buffer (must match WGSL alignment rules).
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct FixedFunctionUniforms {
    pub model_view: [f32; 16],
    pub projection: [f32; 16],
    pub texture_matrix: [f32; 16],
    pub normal_matrix: [f32; 16],
    pub color: [f32; 4],
    pub fog_color: [f32; 4],
    // x = start, y = end, z = density, w = mode (0=none, 1=linear, 2=exp, 3=exp2)
    pub fog_params: [f32; 4],
    pub light0_dir: [f32; 4],
    pub light0_diffuse: [f32; 4],
    pub light0_ambient: [f32; 4],
    pub light1_dir: [f32; 4],
    pub light1_diffuse: [f32; 4],
    pub light1_ambient: [f32; 4],
    pub light_model_ambient: [f32; 4],
    // x = texture_enabled, y = lighting_enabled, z = fog_enabled, w = alpha_test_func
    pub flags: [u32; 4],
    // x = alpha_ref, y = tex_gen_enabled, z = rescale_normal, w = pad
    pub alpha_ref: [f32; 4],
    pub tex_gen_s: [f32; 4],
    pub tex_gen_t: [f32; 4],
}

impl Default for FixedFunctionUniforms {
    fn default() -> Self {
        let identity = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        Self {
            model_view: identity,
            projection: identity,
            texture_matrix: identity,
            normal_matrix: identity,
            color: [1.0, 1.0, 1.0, 1.0],
            fog_color: [0.0, 0.0, 0.0, 1.0],
            fog_params: [0.0, 1.0, 1.0, 0.0],
            light0_dir: [0.0, 0.0, -1.0, 0.0],
            light0_diffuse: [1.0, 1.0, 1.0, 1.0],
            light0_ambient: [0.0, 0.0, 0.0, 1.0],
            light1_dir: [0.0, 0.0, -1.0, 0.0],
            light1_diffuse: [1.0, 1.0, 1.0, 1.0],
            light1_ambient: [0.0, 0.0, 0.0, 1.0],
            light_model_ambient: [0.2, 0.2, 0.2, 1.0],
            flags: [0, 0, 0, 7], // alpha_test_func = GL_ALWAYS (7)
            alpha_ref: [0.0, 0.0, 0.0, 0.0],
            tex_gen_s: [0.0; 4],
            tex_gen_t: [0.0; 4],
        }
    }
}

// ============================================================================
// GLES2 Shader and Program emulation
// ============================================================================

#[derive(Debug, Clone)]
pub struct ShaderObject {
    pub id: u32,
    pub shader_type: u32, // GL_VERTEX_SHADER, GL_FRAGMENT_SHADER
    pub source: String,
    pub compiled: bool,
    pub info_log: String,
}

#[derive(Debug, Clone)]
pub struct ProgramObject {
    pub id: u32,
    pub attached_shaders: Vec<u32>,
    pub linked: bool,
    pub info_log: String,
    pub uniforms_f32: HashMap<i32, Vec<f32>>,
    pub uniforms_i32: HashMap<i32, Vec<i32>>,
    pub uniform_names: HashMap<String, i32>,
    pub attrib_names: HashMap<String, i32>,
}

impl ProgramObject {
    pub fn new(id: u32) -> Self {
        Self {
            id,
            attached_shaders: Vec::new(),
            linked: false,
            info_log: String::new(),
            uniforms_f32: HashMap::new(),
            uniforms_i32: HashMap::new(),
            uniform_names: HashMap::new(),
            attrib_names: HashMap::new(),
        }
    }
}
