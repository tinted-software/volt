//! Core OpenGL Context and State Machine.
#![allow(unused_imports, dead_code, unused_mut, unsafe_op_in_unsafe_fn)]
use crate::display_list::{DisplayList, DisplayListOp, DisplayListRegistry, VertexData};
use crate::matrix::{Mat4, MatrixMode, MatrixStack};
use crate::pipeline::PipelineKey;
use crate::shader::FixedFunctionUniforms;
use crate::sync::Mutex;
use crate::texture::{TextureManager, TextureObject};
use crate::types::*;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ffi::c_void;

#[derive(Debug, Clone)]
pub struct LightState {
    pub position: [f32; 4],
    pub ambient: [f32; 4],
    pub diffuse: [f32; 4],
    pub specular: [f32; 4],
    pub spot_direction: [f32; 3],
    pub spot_exponent: f32,
    pub spot_cutoff: f32,
    pub constant_attenuation: f32,
    pub linear_attenuation: f32,
    pub quadratic_attenuation: f32,
}

impl Default for LightState {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 1.0, 0.0],
            ambient: [0.0, 0.0, 0.0, 1.0],
            diffuse: [1.0, 1.0, 1.0, 1.0],
            specular: [1.0, 1.0, 1.0, 1.0],
            spot_direction: [0.0, 0.0, -1.0],
            spot_exponent: 0.0,
            spot_cutoff: 180.0,
            constant_attenuation: 1.0,
            linear_attenuation: 0.0,
            quadratic_attenuation: 0.0,
        }
    }
}

// Global current-context registry.
//
// no_std note: `std::thread_local!` is unavailable, and the target usage
// (a single implicit GL context shared across threads, as on the consoles
// this was ported from) makes globally-current state the correct semantics
// anyway. Both slots are plain lock-protected globals.
pub static CURRENT_CONTEXT: Mutex<Option<Arc<Mutex<GlContext>>>> = Mutex::new(None);
static LAST_ACTIVE_CONTEXT: Mutex<Option<Arc<Mutex<GlContext>>>> = Mutex::new(None);

/// The context current on this logical "thread group": the explicitly
/// current context, falling back to whichever context was last made current
/// on any thread (see egl.rs notes on implicit-context semantics).
pub fn get_current_gl_context() -> Option<Arc<Mutex<GlContext>>> {
    if let Some(ctx) = CURRENT_CONTEXT.lock().clone() {
        return Some(ctx);
    }
    LAST_ACTIVE_CONTEXT.lock().clone()
}

/// Set by `eglMakeCurrent` / `egl_release_context` in vantage-egl.
pub fn set_current_gl_context(ctx: Option<Arc<Mutex<GlContext>>>) {
    *CURRENT_CONTEXT.lock() = ctx.clone();
    if ctx.is_some() {
        *LAST_ACTIVE_CONTEXT.lock() = ctx;
    }
}

pub struct GlContext {
    pub id: u32,
    /// Owned by the context: every access already happens under the
    /// context lock (`with_context` / `&mut self` methods), so no inner
    /// mutex is needed.
    pub texture_manager: TextureManager,
    pub display_lists: Arc<DisplayListRegistry>,

    // Matrix state
    pub matrix_mode: MatrixMode,
    pub modelview_stack: MatrixStack,
    pub projection_stack: MatrixStack,
    pub texture_stack: MatrixStack,

    // Current vertex attributes
    pub current_color: [f32; 4],
    pub material_ambient_diffuse: [f32; 4],
    pub current_normal: [f32; 3],
    pub current_texcoord: [f32; 2],

    // Client vertex arrays
    pub vertex_array_enabled: bool,
    pub vertex_pointer_size: GLint,
    pub vertex_pointer_type: GLenum,
    pub vertex_pointer_stride: GLsizei,
    pub vertex_pointer: *const c_void,

    pub texcoord_array_enabled: bool,
    pub texcoord_pointer_size: GLint,
    pub texcoord_pointer_type: GLenum,
    pub texcoord_pointer_stride: GLsizei,
    pub texcoord_pointer: *const c_void,

    pub color_array_enabled: bool,
    pub color_pointer_size: GLint,
    pub color_pointer_type: GLenum,
    pub color_pointer_stride: GLsizei,
    pub color_pointer: *const c_void,

    pub normal_array_enabled: bool,
    pub normal_pointer_type: GLenum,
    pub normal_pointer_stride: GLsizei,
    pub normal_pointer: *const c_void,

    pub array_buffer_binding: GLuint,
    pub element_array_buffer_binding: GLuint,
    pub buffers: hashbrown::HashMap<GLuint, Vec<u8>>,
    pub next_buffer_id: GLuint,

    // Server enables
    pub texture_2d_enabled: bool,
    pub texenv_mode: [GLenum; 2],
    pub texenv_color: [[f32; 4]; 2],
    pub blend_enabled: bool,
    pub depth_test_enabled: bool,
    pub alpha_test_enabled: bool,
    pub cull_face_enabled: bool,
    pub cull_face_mode: GLenum,
    pub front_face: GLenum,
    pub lighting_enabled: bool,
    pub lights_enabled: [bool; 8],
    pub color_material_enabled: bool,
    pub rescale_normal_enabled: bool,
    pub normalize_enabled: bool,
    pub fog_enabled: bool,
    pub scissor_test_enabled: bool,
    pub stencil_test_enabled: bool,
    pub polygon_offset_fill_enabled: bool,

    // Blend state
    pub src_factor: GLenum,
    pub dst_factor: GLenum,
    pub src_factor_alpha: GLenum,
    pub dst_factor_alpha: GLenum,
    pub blend_color: [f32; 4],

    // Depth state
    pub depth_func: GLenum,
    pub depth_mask: bool,
    pub depth_range: (f32, f32),
    pub hints: hashbrown::HashMap<GLenum, GLenum>,

    // Alpha test state
    pub alpha_func: GLenum,
    pub alpha_ref: f32,

    // Fog state
    pub fog_mode: GLenum,
    pub fog_start: f32,
    pub fog_end: f32,
    pub fog_density: f32,
    pub fog_color: [f32; 4],

    // Lighting state
    pub lights: [LightState; 8],
    pub light_model_ambient: [f32; 4],

    // Viewport & Scissor
    pub viewport: (i32, i32, i32, i32),
    pub scissor: (i32, i32, i32, i32),

    // Clear values
    pub clear_color: [f32; 4],
    pub clear_depth: f32,
    pub clear_stencil: i32,

    // Color Mask
    pub color_mask: (bool, bool, bool, bool),

    // Polygon Offset
    pub polygon_offset_factor: f32,
    pub polygon_offset_units: f32,

    pub line_width: f32,
    pub point_size: f32,
    pub shade_model: GLenum,

    // Stencil
    pub stencil_func: GLenum,
    pub stencil_ref: i32,
    pub stencil_value_mask: u32,
    pub stencil_fail: GLenum,
    pub stencil_zfail: GLenum,
    pub stencil_zpass: GLenum,
    pub stencil_writemask: u32,

    // Immediate Mode (glBegin/glEnd)
    pub immediate_mode: Option<GLenum>,
    pub immediate_vertices: Vec<VertexData>,

    // Display List recording
    pub active_display_list: Option<DisplayList>,
    pub display_list_mode: GLenum,

    pub error: GLenum,

    // HAL device & recorded commands
    pub hal_device: volt_hal::Device,
    pub command_buffer: volt_hal::CommandBuffer,
    pub color_image: Option<volt_hal::ImageId>,
    pub depth_image: Option<volt_hal::ImageId>,
}

unsafe impl Send for GlContext {}
unsafe impl Sync for GlContext {}

impl GlContext {
    pub fn new(
        id: u32,
        texture_manager: TextureManager,
        display_lists: Arc<DisplayListRegistry>,
    ) -> Self {
        Self {
            id,
            texture_manager,
            display_lists,
            matrix_mode: MatrixMode::ModelView,
            modelview_stack: MatrixStack::new(32),
            projection_stack: MatrixStack::new(32),
            texture_stack: MatrixStack::new(32),
            current_color: [1.0, 1.0, 1.0, 1.0],
            material_ambient_diffuse: [1.0, 1.0, 1.0, 1.0],
            current_normal: [0.0, 0.0, 1.0],
            current_texcoord: [0.0, 0.0],
            vertex_array_enabled: false,
            vertex_pointer_size: 3,
            vertex_pointer_type: GL_FLOAT,
            vertex_pointer_stride: 0,
            vertex_pointer: core::ptr::null(),
            texcoord_array_enabled: false,
            texcoord_pointer_size: 2,
            texcoord_pointer_type: GL_FLOAT,
            texcoord_pointer_stride: 0,
            texcoord_pointer: core::ptr::null(),
            color_array_enabled: false,
            color_pointer_size: 4,
            color_pointer_type: GL_FLOAT,
            color_pointer_stride: 0,
            color_pointer: core::ptr::null(),
            normal_array_enabled: false,
            normal_pointer_type: GL_FLOAT,
            normal_pointer_stride: 0,
            normal_pointer: core::ptr::null(),
            array_buffer_binding: 0,
            element_array_buffer_binding: 0,
            buffers: hashbrown::HashMap::new(),
            next_buffer_id: 1,
            texture_2d_enabled: false,
            texenv_mode: [GL_MODULATE; 2],
            texenv_color: [[0.0; 4]; 2],
            blend_enabled: false,
            depth_test_enabled: false,
            alpha_test_enabled: false,
            cull_face_enabled: false,
            cull_face_mode: GL_BACK,
            front_face: GL_CCW,
            lighting_enabled: false,
            lights_enabled: [false; 8],
            color_material_enabled: false,
            rescale_normal_enabled: false,
            normalize_enabled: false,
            fog_enabled: false,
            scissor_test_enabled: false,
            stencil_test_enabled: false,
            polygon_offset_fill_enabled: false,
            src_factor: GL_SRC_ALPHA,
            dst_factor: GL_ONE_MINUS_SRC_ALPHA,
            src_factor_alpha: GL_SRC_ALPHA,
            dst_factor_alpha: GL_ONE_MINUS_SRC_ALPHA,
            blend_color: [0.0, 0.0, 0.0, 0.0],
            depth_func: GL_LEQUAL,
            depth_mask: true,
            depth_range: (0.0, 1.0),
            hints: hashbrown::HashMap::new(),
            alpha_func: GL_ALWAYS,
            alpha_ref: 0.0,
            fog_mode: GL_LINEAR,
            fog_start: 0.0,
            fog_end: 1.0,
            fog_density: 1.0,
            fog_color: [0.0, 0.0, 0.0, 0.0],
            lights: [
                LightState::default(),
                LightState::default(),
                LightState::default(),
                LightState::default(),
                LightState::default(),
                LightState::default(),
                LightState::default(),
                LightState::default(),
            ],
            light_model_ambient: [0.2, 0.2, 0.2, 1.0],
            viewport: (0, 0, 1280, 720),
            scissor: (0, 0, 1280, 720),
            clear_color: [0.4, 0.6, 0.9, 1.0],
            clear_depth: 1.0,
            clear_stencil: 0,
            color_mask: (true, true, true, true),
            polygon_offset_factor: 0.0,
            polygon_offset_units: 0.0,
            line_width: 1.0,
            point_size: 1.0,
            shade_model: GL_SMOOTH,
            stencil_func: GL_ALWAYS,
            stencil_ref: 0,
            stencil_value_mask: 0xFF,
            stencil_fail: GL_KEEP,
            stencil_zfail: GL_KEEP,
            stencil_zpass: GL_KEEP,
            stencil_writemask: 0xFF,
            immediate_mode: None,
            immediate_vertices: Vec::new(),
            active_display_list: None,
            display_list_mode: GL_COMPILE,
            error: GL_NO_ERROR,
            hal_device: volt_hal::Device::new(),
            command_buffer: volt_hal::CommandBuffer::default(),
            color_image: None,
            depth_image: None,
        }
    }

    #[inline]
    pub fn ensure_buffer(&mut self, id: GLuint) {
        if id != 0 {
            self.buffers.entry(id).or_default();
            self.next_buffer_id = self.next_buffer_id.max(id + 1);
        }
    }

    /// Resolve a client-array pointer. If an ARRAY_BUFFER is bound, `pointer`
    /// is a byte offset into that buffer (including offset 0).
    pub fn array_base(&self, pointer: *const c_void) -> Option<*const u8> {
        if self.array_buffer_binding != 0 {
            let buf = self.buffers.get(&self.array_buffer_binding)?;
            let off = pointer as usize;
            if off > buf.len() {
                return None;
            }
            Some(unsafe { buf.as_ptr().add(off) })
        } else if !pointer.is_null() {
            Some(pointer as *const u8)
        } else {
            None
        }
    }

    pub fn element_base(&self, pointer: *const c_void) -> Option<*const u8> {
        if self.element_array_buffer_binding != 0 {
            let buf = self.buffers.get(&self.element_array_buffer_binding)?;
            let off = pointer as usize;
            if off > buf.len() {
                return None;
            }
            Some(unsafe { buf.as_ptr().add(off) })
        } else if !pointer.is_null() {
            Some(pointer as *const u8)
        } else {
            None
        }
    }

    pub unsafe fn read_vertex(&self, index: usize) -> VertexData {
        let v_stride = if self.vertex_pointer_stride > 0 {
            self.vertex_pointer_stride as usize
        } else {
            (self.vertex_pointer_size * 4) as usize
        };
        let t_stride = if self.texcoord_pointer_stride > 0 {
            self.texcoord_pointer_stride as usize
        } else {
            (self.texcoord_pointer_size * 4) as usize
        };
        let c_stride = if self.color_pointer_stride > 0 {
            self.color_pointer_stride as usize
        } else if self.color_pointer_type == GL_UNSIGNED_BYTE {
            self.color_pointer_size.max(1) as usize
        } else {
            (self.color_pointer_size * 4) as usize
        };
        let n_stride = if self.normal_pointer_stride > 0 {
            self.normal_pointer_stride as usize
        } else {
            12
        };

        let mut pos = [0.0f32; 3];
        if let Some(base) = self.array_base(self.vertex_pointer) {
            let p = base.add(index * v_stride) as *const f32;
            if self.vertex_pointer_size >= 2 {
                pos[0] = *p.add(0);
                pos[1] = *p.add(1);
            }
            if self.vertex_pointer_size >= 3 {
                pos[2] = *p.add(2);
            }
        }

        let mut tex = self.current_texcoord;
        if self.texcoord_array_enabled
            && let Some(base) = self.array_base(self.texcoord_pointer)
        {
            let p = base.add(index * t_stride) as *const f32;
            if self.texcoord_pointer_size >= 2 {
                tex[0] = *p.add(0);
                tex[1] = *p.add(1);
            }
        }

        let mut col = self.current_color;
        if self.color_array_enabled
            && let Some(base) = self.array_base(self.color_pointer)
        {
            let p = base.add(index * c_stride);
            if self.color_pointer_type == GL_UNSIGNED_BYTE {
                col[0] = (*p.add(0) as f32) / 255.0;
                col[1] = (*p.add(1) as f32) / 255.0;
                col[2] = (*p.add(2) as f32) / 255.0;
                col[3] = if self.color_pointer_size >= 4 {
                    (*p.add(3) as f32) / 255.0
                } else {
                    1.0
                };
            } else {
                let pf = p as *const f32;
                col[0] = *pf.add(0);
                col[1] = *pf.add(1);
                col[2] = *pf.add(2);
                col[3] = if self.color_pointer_size >= 4 {
                    *pf.add(3)
                } else {
                    1.0
                };
            }
        }

        let mut norm = self.current_normal;
        if self.normal_array_enabled
            && let Some(base) = self.array_base(self.normal_pointer)
        {
            if self.normal_pointer_type == GL_BYTE {
                let p = base.add(index * n_stride);
                norm[0] = (*p.add(0) as i8 as f32) / 127.0;
                norm[1] = (*p.add(1) as i8 as f32) / 127.0;
                norm[2] = (*p.add(2) as i8 as f32) / 127.0;
            } else {
                let p = base.add(index * n_stride) as *const f32;
                norm[0] = *p.add(0);
                norm[1] = *p.add(1);
                norm[2] = *p.add(2);
            }
        }

        VertexData {
            position: pos,
            tex_coord: tex,
            color: col,
            normal: norm,
        }
    }

    #[inline]
    pub fn current_matrix_stack(&mut self) -> &mut MatrixStack {
        match self.matrix_mode {
            MatrixMode::ModelView => &mut self.modelview_stack,
            MatrixMode::Projection => &mut self.projection_stack,
            MatrixMode::Texture => &mut self.texture_stack,
        }
    }

    #[inline]
    pub fn current_matrix_stack_ref(&self) -> &MatrixStack {
        match self.matrix_mode {
            MatrixMode::ModelView => &self.modelview_stack,
            MatrixMode::Projection => &self.projection_stack,
            MatrixMode::Texture => &self.texture_stack,
        }
    }

    pub fn build_uniforms(&self) -> FixedFunctionUniforms {
        let mv = self.modelview_stack.current;
        let proj = self.projection_stack.current;
        let tex_m = self.texture_stack.current;

        let norm_m = mv.normal_matrix_3x3();
        let normal_mat4 = [
            norm_m[0], norm_m[1], norm_m[2], 0.0, norm_m[3], norm_m[4], norm_m[5], 0.0, norm_m[6],
            norm_m[7], norm_m[8], 0.0, 0.0, 0.0, 0.0, 1.0,
        ];

        let alpha_test_code = if self.alpha_test_enabled {
            match self.alpha_func {
                GL_NEVER => 0,
                GL_LESS => 1,
                GL_EQUAL => 2,
                GL_LEQUAL => 3,
                GL_GREATER => 4,
                GL_NOTEQUAL => 5,
                GL_GEQUAL => 6,
                GL_ALWAYS => 7,
                _ => 7,
            }
        } else {
            7 // GL_ALWAYS
        };

        let fog_mode_code = if self.fog_enabled {
            match self.fog_mode {
                GL_LINEAR => 1.0,
                GL_EXP => 2.0,
                GL_EXP2 => 3.0,
                _ => 1.0,
            }
        } else {
            0.0
        };

        FixedFunctionUniforms {
            model_view: mv.to_array(),
            projection: proj.to_array(),
            texture_matrix: tex_m.to_array(),
            normal_matrix: normal_mat4,
            color: if self.lighting_enabled && !self.color_material_enabled {
                self.material_ambient_diffuse
            } else {
                self.current_color
            },
            fog_color: self.fog_color,
            fog_params: [
                self.fog_start,
                self.fog_end,
                self.fog_density,
                fog_mode_code,
            ],
            light0_dir: self.lights[0].position,
            light0_diffuse: self.lights[0].diffuse,
            light0_ambient: self.lights[0].ambient,
            light1_dir: self.lights[1].position,
            light1_diffuse: self.lights[1].diffuse,
            light1_ambient: self.lights[1].ambient,
            light_model_ambient: self.light_model_ambient,
            flags: [
                if self.texture_2d_enabled { 1 } else { 0 },
                if self.lighting_enabled { 1 } else { 0 },
                if self.fog_enabled { 1 } else { 0 },
                alpha_test_code,
            ],
            alpha_ref: [
                self.alpha_ref,
                0.0, // tex_gen
                if self.rescale_normal_enabled {
                    1.0
                } else {
                    0.0
                },
                0.0,
            ],
            tex_gen_s: [0.0; 4],
            tex_gen_t: [0.0; 4],
        }
    }

    pub fn build_pipeline_key(&self, mode: GLenum) -> PipelineKey {
        let topology = match mode {
            GL_TRIANGLES | GL_QUADS | GL_POLYGON => 0,
            GL_TRIANGLE_STRIP => 1,
            GL_TRIANGLE_FAN => 2,
            GL_LINES => 3,
            GL_LINE_STRIP | GL_LINE_LOOP => 4,
            GL_POINTS => 5,
            _ => 0,
        };

        let cull_mode = if self.cull_face_enabled {
            // RasterState.cull_mode expects actual GL enums (0 = none).
            match self.cull_face_mode {
                GL_FRONT | GL_BACK | GL_FRONT_AND_BACK => self.cull_face_mode,
                _ => 0,
            }
        } else {
            0
        };

        let mut color_mask_byte = 0u8;
        if self.color_mask.0 {
            color_mask_byte |= 1;
        }
        if self.color_mask.1 {
            color_mask_byte |= 2;
        }
        if self.color_mask.2 {
            color_mask_byte |= 4;
        }
        if self.color_mask.3 {
            color_mask_byte |= 8;
        }

        PipelineKey {
            topology,
            cull_mode,
            front_face_ccw: self.front_face == GL_CCW,
            blend_enabled: self.blend_enabled,
            src_factor: self.src_factor,
            dst_factor: self.dst_factor,
            depth_test: self.depth_test_enabled,
            depth_write: self.depth_mask,
            depth_func: self.depth_func,
            color_mask: color_mask_byte,
        }
    }

    pub fn draw_vertex_data(
        &mut self,
        mode: GLenum,
        vertices: &[VertexData],
        indices: Option<&[u32]>,
    ) {
        if let Some(list) = &mut self.active_display_list {
            list.push_op(DisplayListOp::Draw {
                mode,
                vertices: vertices.to_vec(),
                indices: indices.map(|i| i.to_vec()).unwrap_or_default(),
            });
            if self.display_list_mode == GL_COMPILE {
                return;
            }
        }

        let (final_vertices, final_indices): (&[VertexData], Option<alloc::borrow::Cow<[u32]>>) =
            if mode == GL_QUADS && indices.is_none() {
                // Sequential quad expansion from packed vertex data.
                let quad_count = vertices.len() / 4;
                let mut inds = Vec::with_capacity(quad_count * 6);
                for q in 0..quad_count as u32 {
                    let base = q * 4;
                    inds.push(base);
                    inds.push(base + 1);
                    inds.push(base + 2);
                    inds.push(base);
                    inds.push(base + 2);
                    inds.push(base + 3);
                }
                (vertices, Some(alloc::borrow::Cow::Owned(inds)))
            } else if mode == GL_QUADS {
                // Indexed quads: expand from the actual index values so shared
                // vertices between quads keep working with compacted gathers.
                let inds = indices.unwrap_or(&[]);
                let quad_count = inds.len() / 4;
                let mut out = Vec::with_capacity(quad_count * 6);
                for q in 0..quad_count {
                    let b0 = inds[q * 4];
                    let b1 = inds[q * 4 + 1];
                    let b2 = inds[q * 4 + 2];
                    let b3 = inds[q * 4 + 3];
                    out.push(b0);
                    out.push(b1);
                    out.push(b2);
                    out.push(b0);
                    out.push(b2);
                    out.push(b3);
                }
                (vertices, Some(alloc::borrow::Cow::Owned(out)))
            } else if let Some(inds) = indices {
                (vertices, Some(alloc::borrow::Cow::Borrowed(inds)))
            } else {
                (vertices, None)
            };

        let mv = self.modelview_stack.current.0;
        let proj = self.projection_stack.current.0;
        let mvp = proj.mul_mat4(&mv);

        let mut transformed = Vec::with_capacity(final_vertices.len());
        for v in final_vertices {
            // Transform position by MVP matrix
            let p = glam::Vec4::new(v.position[0], v.position[1], v.position[2], 1.0);
            let clip = mvp.mul_vec4(p);

            // In GLES 1.1, lighting & color material are applied per vertex on CPU
            let mut color = v.color;
            if self.lighting_enabled {
                let mut lit_rgb = [
                    self.light_model_ambient[0] * color[0],
                    self.light_model_ambient[1] * color[1],
                    self.light_model_ambient[2] * color[2],
                ];
                let norm = glam::Vec3::new(v.normal[0], v.normal[1], v.normal[2]);
                let norm = if self.normalize_enabled {
                    norm.normalize_or_zero()
                } else {
                    norm
                };

                for i in 0..8 {
                    if self.lights_enabled[i] {
                        let l = &self.lights[i];
                        let ldir = glam::Vec3::new(l.position[0], l.position[1], l.position[2])
                            .normalize_or_zero();
                        let n_dot_l = norm.dot(ldir).max(0.0);

                        lit_rgb[0] += l.ambient[0] * color[0] + l.diffuse[0] * color[0] * n_dot_l;
                        lit_rgb[1] += l.ambient[1] * color[1] + l.diffuse[1] * color[1] * n_dot_l;
                        lit_rgb[2] += l.ambient[2] * color[2] + l.diffuse[2] * color[2] * n_dot_l;
                    }
                }
                color = [
                    lit_rgb[0].min(1.0),
                    lit_rgb[1].min(1.0),
                    lit_rgb[2].min(1.0),
                    color[3],
                ];
            }

            // Eye distance for fog: in OpenGL §3.10, fog coordinate is eye-space distance |z_eye|
            let eye_p = mv.mul_vec4(p);
            let eye_dist = eye_p.z.abs();

            transformed.push(volt_raster::Vertex {
                pos: [clip.x, clip.y, clip.z, clip.w],
                color,
                tex0: [v.tex_coord[0], v.tex_coord[1]],
                tex1: [0.0, 0.0],
                fog: eye_dist,
                _pad: 0.0,
            });
        }

        // Create pipeline
        let key = self.build_pipeline_key(mode);
        let pipeline = volt_hal::Pipeline {
            topology: key.topology,
            cull_mode: key.cull_mode,
            front_face_ccw: key.front_face_ccw,
            blend_enabled: key.blend_enabled,
            src_factor: key.src_factor,
            dst_factor: key.dst_factor,
            depth_test: key.depth_test,
            depth_write: key.depth_write,
            depth_func: key.depth_func,
            color_mask: key.color_mask,
        };

        // Construct FragState with active texture and fragment settings
        let mut frag_state = volt_raster::FragState {
            texenv_mode: self.texenv_mode,
            texenv_color: self.texenv_color,
            ..Default::default()
        };

        if self.texture_2d_enabled
            && let Some(tex) = self.texture_manager.get_current_texture()
            && let Some(level0) = tex.level_data.get(&0)
        {
            frag_state.textures[0] = volt_raster::SampledTexture {
                data: level0.as_ptr(),
                data_len: level0.len(),
                width: tex.width,
                height: tex.height,
                format: tex.format,
                min_filter: tex.min_filter,
                mag_filter: tex.mag_filter,
                wrap_s: tex.wrap_s,
                wrap_t: tex.wrap_t,
                enabled: true,
            };
        }

        frag_state.alpha_func = if self.alpha_test_enabled {
            self.alpha_func
        } else {
            volt_raster::gl::ALWAYS
        };
        frag_state.alpha_ref = self.alpha_ref;

        if self.fog_enabled {
            frag_state.fog_mode = match self.fog_mode {
                GL_LINEAR => 1,
                GL_EXP => 2,
                GL_EXP2 => 3,
                _ => 0,
            };
            frag_state.fog_start = self.fog_start;
            frag_state.fog_end = self.fog_end;
            frag_state.fog_density = self.fog_density;
            frag_state.fog_color = self.fog_color;
        }

        self.command_buffer
            .push(volt_hal::Cmd::SetFragState(alloc::boxed::Box::new(
                frag_state,
            )));

        self.command_buffer.push(volt_hal::Cmd::SetViewport {
            x: self.viewport.0,
            y: self.viewport.1,
            w: self.viewport.2.max(1) as u32,
            h: self.viewport.3.max(1) as u32,
        });
        // Scissor state is emitted on every draw so a disable is represented
        // in the command stream instead of leaking from a previous draw.
        self.command_buffer
            .push(volt_hal::Cmd::SetScissor(if self.scissor_test_enabled {
                Some((
                    self.scissor.0,
                    self.scissor.1,
                    self.scissor.2.max(1) as u32,
                    self.scissor.3.max(1) as u32,
                ))
            } else {
                None
            }));

        self.command_buffer.push(volt_hal::Cmd::DrawMesh {
            vertices: transformed,
            indices: final_indices.map(|c| c.into_owned()),
            pipeline,
        });
    }

    pub fn call_display_list(&mut self, list_id: GLuint) {
        let Some(list) = self.display_lists.get_list(list_id) else {
            return;
        };
        for op in list.ops {
            match op {
                DisplayListOp::Draw {
                    mode,
                    vertices,
                    indices,
                } => {
                    let idx_slice = if indices.is_empty() {
                        None
                    } else {
                        Some(indices.as_slice())
                    };
                    self.draw_vertex_data(mode, &vertices, idx_slice);
                }
                DisplayListOp::MatrixPush(mode) => {
                    let prev = self.matrix_mode;
                    self.matrix_mode = mode;
                    let _ = self.current_matrix_stack().push();
                    self.matrix_mode = prev;
                }
                DisplayListOp::MatrixPop(mode) => {
                    let prev = self.matrix_mode;
                    self.matrix_mode = mode;
                    let _ = self.current_matrix_stack().pop();
                    self.matrix_mode = prev;
                }
                DisplayListOp::MatrixLoad(mode, m) => {
                    let prev = self.matrix_mode;
                    self.matrix_mode = mode;
                    self.current_matrix_stack().load_matrix(&m);
                    self.matrix_mode = prev;
                }
                DisplayListOp::MatrixMult(mode, m) => {
                    let prev = self.matrix_mode;
                    self.matrix_mode = mode;
                    self.current_matrix_stack().mult_matrix(&m);
                    self.matrix_mode = prev;
                }
                DisplayListOp::MatrixTranslate(x, y, z) => {
                    self.current_matrix_stack().translate(x, y, z);
                }
                DisplayListOp::MatrixRotate(a, x, y, z) => {
                    self.current_matrix_stack().rotate(a, x, y, z);
                }
                DisplayListOp::MatrixScale(x, y, z) => {
                    self.current_matrix_stack().scale(x, y, z);
                }
                DisplayListOp::BindTexture(id) => {
                    self.texture_manager.bind_texture(GL_TEXTURE_2D, id);
                }
                DisplayListOp::Enable(cap) => {
                    self.set_enable(cap, true);
                }
                DisplayListOp::Disable(cap) => {
                    self.set_enable(cap, false);
                }
                DisplayListOp::Color4f(r, g, b, a) => {
                    self.current_color = [r, g, b, a];
                }
                DisplayListOp::Normal3f(x, y, z) => {
                    self.current_normal = [x, y, z];
                }
                DisplayListOp::TexCoord2f(u, v) => {
                    self.current_texcoord = [u, v];
                }
                DisplayListOp::BlendFunc(src, dst) => {
                    self.src_factor = src;
                    self.dst_factor = dst;
                }
                DisplayListOp::DepthFunc(func) => {
                    self.depth_func = func;
                }
                DisplayListOp::DepthMask(mask) => {
                    self.depth_mask = mask;
                }
                DisplayListOp::AlphaFunc(func, r) => {
                    self.alpha_func = func;
                    self.alpha_ref = r;
                }
                DisplayListOp::CullFace(mode) => {
                    self.cull_face_mode = mode;
                }
                DisplayListOp::PolygonOffset(factor, units) => {
                    self.polygon_offset_factor = factor;
                    self.polygon_offset_units = units;
                }
                DisplayListOp::ShadeModel(model) => {
                    self.shade_model = model;
                }
                DisplayListOp::CallList(child_id) => {
                    self.call_display_list(child_id);
                }
            }
        }
    }

    pub fn set_enable(&mut self, cap: GLenum, enable: bool) {
        match cap {
            GL_TEXTURE_2D => self.texture_2d_enabled = enable,
            GL_BLEND => self.blend_enabled = enable,
            GL_DEPTH_TEST => self.depth_test_enabled = enable,
            GL_ALPHA_TEST => self.alpha_test_enabled = enable,
            GL_CULL_FACE => self.cull_face_enabled = enable,
            GL_LIGHTING => self.lighting_enabled = enable,
            GL_LIGHT0 => self.lights_enabled[0] = enable,
            GL_LIGHT1 => self.lights_enabled[1] = enable,
            GL_LIGHT2 => self.lights_enabled[2] = enable,
            GL_LIGHT3 => self.lights_enabled[3] = enable,
            GL_LIGHT4 => self.lights_enabled[4] = enable,
            GL_LIGHT5 => self.lights_enabled[5] = enable,
            GL_LIGHT6 => self.lights_enabled[6] = enable,
            GL_LIGHT7 => self.lights_enabled[7] = enable,
            GL_COLOR_MATERIAL => self.color_material_enabled = enable,
            GL_RESCALE_NORMAL => self.rescale_normal_enabled = enable,
            GL_NORMALIZE => self.normalize_enabled = enable,
            GL_FOG => self.fog_enabled = enable,
            GL_SCISSOR_TEST => self.scissor_test_enabled = enable,
            GL_STENCIL_TEST => self.stencil_test_enabled = enable,
            GL_POLYGON_OFFSET_FILL => self.polygon_offset_fill_enabled = enable,
            _ => {}
        }
    }
}
