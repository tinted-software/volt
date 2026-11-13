//! Vantage GLES 1.1 (CM) state machine and C ABI entry points.
//!
//! Entry points operate on the globally-current `GlContext` (see
//! context-registry semantics in `gl_context`). GLES 2.0 entry points are
//! documented stubs that raise `GL_INVALID_OPERATION` (see the `MISSING:`
//! doc comments on each stub).
#![allow(
    non_snake_case,
    non_camel_case_types,
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc
)]
#![no_std]
extern crate alloc;

pub mod display_list;
pub mod error;
pub mod gl_context;
pub mod matrix;
pub mod pipeline;
pub mod shader;
pub mod sync;
pub mod texture;
pub mod types;

pub use crate::error::VantageError;
pub use crate::types::*;

use crate::display_list::VertexData;
use crate::gl_context::GlContext;
use crate::gl_context::get_current_gl_context;
use crate::matrix::MatrixMode;
use alloc::vec;
use alloc::vec::Vec;
use core::ffi::c_void;

/// integer as-is (see `glFogx`).
#[inline]
fn fixed_to_float(x: GLfixed) -> GLfloat {
    x as GLfloat / 65536.0
}

#[inline]
fn clampx_to_float(x: GLclampx) -> GLclampf {
    (x as GLfloat / 65536.0).clamp(0.0, 1.0)
}

// Diagnostics: this crate has no logger and prints nothing. Failures are
// reported through typed `VantageError`s internally and through the
// EGL/GL error state (`eglGetError` / `glGetError`) at the C ABI boundary.
// ============================================================================
// Internal Helper: Helper to execute with current GL context
// ============================================================================

#[inline]
fn with_context<F, R>(f: F) -> R
where
    F: FnOnce(&mut GlContext) -> R,
    R: Default,
{
    if let Some(ctx_arc) = get_current_gl_context() {
        let mut ctx = ctx_arc.lock();
        f(&mut ctx)
    } else {
        R::default()
    }
}

// ============================================================================
// OpenGL Matrix and Transform Entry Points
// ============================================================================

pub unsafe extern "C" fn glMatrixMode(mode: GLenum) {
    with_context(|ctx| {
        ctx.matrix_mode = match mode {
            GL_MODELVIEW => MatrixMode::ModelView,
            GL_PROJECTION => MatrixMode::Projection,
            GL_TEXTURE => MatrixMode::Texture,
            _ => MatrixMode::ModelView,
        };
    });
}

pub unsafe extern "C" fn glLoadIdentity() {
    with_context(|ctx| {
        ctx.current_matrix_stack().load_identity();
    });
}

pub unsafe extern "C" fn glPushMatrix() {
    with_context(|ctx| {
        let mode = ctx.matrix_mode;
        if let Some(list) = &mut ctx.active_display_list {
            list.push_op(crate::display_list::DisplayListOp::MatrixPush(mode));
        }
        let _ = ctx.current_matrix_stack().push();
    });
}

pub unsafe extern "C" fn glPopMatrix() {
    with_context(|ctx| {
        let mode = ctx.matrix_mode;
        if let Some(list) = &mut ctx.active_display_list {
            list.push_op(crate::display_list::DisplayListOp::MatrixPop(mode));
        }
        let _ = ctx.current_matrix_stack().pop();
    });
}

pub unsafe extern "C" fn glTranslatef(x: GLfloat, y: GLfloat, z: GLfloat) {
    with_context(|ctx| {
        if let Some(list) = &mut ctx.active_display_list {
            list.push_op(crate::display_list::DisplayListOp::MatrixTranslate(x, y, z));
        }
        ctx.current_matrix_stack().translate(x, y, z);
    });
}

pub unsafe extern "C" fn glTranslated(x: GLdouble, y: GLdouble, z: GLdouble) {
    glTranslatef(x as GLfloat, y as GLfloat, z as GLfloat);
}

pub unsafe extern "C" fn glRotatef(angle: GLfloat, x: GLfloat, y: GLfloat, z: GLfloat) {
    with_context(|ctx| {
        if let Some(list) = &mut ctx.active_display_list {
            list.push_op(crate::display_list::DisplayListOp::MatrixRotate(
                angle, x, y, z,
            ));
        }
        ctx.current_matrix_stack().rotate(angle, x, y, z);
    });
}

pub unsafe extern "C" fn glScalef(x: GLfloat, y: GLfloat, z: GLfloat) {
    with_context(|ctx| {
        if let Some(list) = &mut ctx.active_display_list {
            list.push_op(crate::display_list::DisplayListOp::MatrixScale(x, y, z));
        }
        ctx.current_matrix_stack().scale(x, y, z);
    });
}

pub unsafe extern "C" fn glScaled(x: GLdouble, y: GLdouble, z: GLdouble) {
    glScalef(x as GLfloat, y as GLfloat, z as GLfloat);
}

pub unsafe extern "C" fn glOrtho(
    left: GLdouble,
    right: GLdouble,
    bottom: GLdouble,
    top: GLdouble,
    near_val: GLdouble,
    far_val: GLdouble,
) {
    glOrthof(
        left as GLfloat,
        right as GLfloat,
        bottom as GLfloat,
        top as GLfloat,
        near_val as GLfloat,
        far_val as GLfloat,
    );
}

pub unsafe extern "C" fn glOrthof(
    left: GLfloat,
    right: GLfloat,
    bottom: GLfloat,
    top: GLfloat,
    near_val: GLfloat,
    far_val: GLfloat,
) {
    with_context(|ctx| {
        let ortho_m = crate::matrix::Mat4::ortho(left, right, bottom, top, near_val, far_val);
        ctx.current_matrix_stack().mult_matrix(&ortho_m);
    });
}

pub unsafe extern "C" fn glFrustum(
    left: GLdouble,
    right: GLdouble,
    bottom: GLdouble,
    top: GLdouble,
    near_val: GLdouble,
    far_val: GLdouble,
) {
    glFrustumf(
        left as GLfloat,
        right as GLfloat,
        bottom as GLfloat,
        top as GLfloat,
        near_val as GLfloat,
        far_val as GLfloat,
    );
}

pub unsafe extern "C" fn glFrustumf(
    left: GLfloat,
    right: GLfloat,
    bottom: GLfloat,
    top: GLfloat,
    near_val: GLfloat,
    far_val: GLfloat,
) {
    with_context(|ctx| {
        let frustum_m = crate::matrix::Mat4::frustum(left, right, bottom, top, near_val, far_val);
        ctx.current_matrix_stack().mult_matrix(&frustum_m);
    });
}

pub unsafe extern "C" fn glMultMatrixf(m: *const GLfloat) {
    if m.is_null() {
        return;
    }
    with_context(|ctx| {
        let mut data = [0.0f32; 16];
        data.copy_from_slice(core::slice::from_raw_parts(m, 16));
        let mat = crate::matrix::Mat4::from_array(data);
        ctx.current_matrix_stack().mult_matrix(&mat);
    });
}

pub unsafe extern "C" fn glLoadMatrixf(m: *const GLfloat) {
    if m.is_null() {
        return;
    }
    with_context(|ctx| {
        let mut data = [0.0f32; 16];
        data.copy_from_slice(core::slice::from_raw_parts(m, 16));
        let mat = crate::matrix::Mat4::from_array(data);
        ctx.current_matrix_stack().load_matrix(&mat);
    });
}

// ============================================================================

#[inline]
fn set_gl_error(err: GLenum) {
    with_context(|ctx| {
        if ctx.error == GL_NO_ERROR {
            ctx.error = err;
        }
    });
}

// Client State and Vertex Arrays
// ============================================================================

pub unsafe extern "C" fn glEnableClientState(array: GLenum) {
    with_context(|ctx| match array {
        GL_VERTEX_ARRAY => ctx.vertex_array_enabled = true,
        GL_TEXTURE_COORD_ARRAY => ctx.texcoord_array_enabled = true,
        GL_COLOR_ARRAY => ctx.color_array_enabled = true,
        GL_NORMAL_ARRAY => ctx.normal_array_enabled = true,
        _ => {}
    });
}

pub unsafe extern "C" fn glDisableClientState(array: GLenum) {
    with_context(|ctx| match array {
        GL_VERTEX_ARRAY => ctx.vertex_array_enabled = false,
        GL_TEXTURE_COORD_ARRAY => ctx.texcoord_array_enabled = false,
        GL_COLOR_ARRAY => ctx.color_array_enabled = false,
        GL_NORMAL_ARRAY => ctx.normal_array_enabled = false,
        _ => {}
    });
}

pub unsafe extern "C" fn glVertexPointer(
    size: GLint,
    type_: GLenum,
    stride: GLsizei,
    pointer: *const c_void,
) {
    with_context(|ctx| {
        ctx.vertex_pointer_size = size;
        ctx.vertex_pointer_type = type_;
        ctx.vertex_pointer_stride = stride;
        ctx.vertex_pointer = pointer;
    });
}

pub unsafe extern "C" fn glTexCoordPointer(
    size: GLint,
    type_: GLenum,
    stride: GLsizei,
    pointer: *const c_void,
) {
    with_context(|ctx| {
        ctx.texcoord_pointer_size = size;
        ctx.texcoord_pointer_type = type_;
        ctx.texcoord_pointer_stride = stride;
        ctx.texcoord_pointer = pointer;
    });
}

pub unsafe extern "C" fn glColorPointer(
    size: GLint,
    type_: GLenum,
    stride: GLsizei,
    pointer: *const c_void,
) {
    with_context(|ctx| {
        ctx.color_pointer_size = size;
        ctx.color_pointer_type = type_;
        ctx.color_pointer_stride = stride;
        ctx.color_pointer = pointer;
    });
}

pub unsafe extern "C" fn glNormalPointer(type_: GLenum, stride: GLsizei, pointer: *const c_void) {
    with_context(|ctx| {
        ctx.normal_pointer_type = type_;
        ctx.normal_pointer_stride = stride;
        ctx.normal_pointer = pointer;
    });
}

pub unsafe extern "C" fn glClientActiveTexture(texture: GLenum) {
    with_context(|ctx| {
        let idx = (texture.saturating_sub(GL_TEXTURE0)) as usize;
        if idx < 8 {
            ctx.texture_manager.client_active_unit = idx;
        }
    });
}

// ============================================================================
// Draw Calls
// ============================================================================

pub unsafe extern "C" fn glDrawArrays(mode: GLenum, first: GLint, count: GLsizei) {
    if count <= 0 {
        return;
    }

    with_context(|ctx| {
        if !ctx.vertex_array_enabled || ctx.array_base(ctx.vertex_pointer).is_none() {
            return;
        }

        let total = count as usize;
        let mut vertices = Vec::with_capacity(total);
        for i in 0..total {
            vertices.push(ctx.read_vertex((first as usize) + i));
        }
        ctx.draw_vertex_data(mode, &vertices, None);
    });
}

pub unsafe extern "C" fn glDrawElements(
    mode: GLenum,
    count: GLsizei,
    type_: GLenum,
    indices: *const c_void,
) {
    if count <= 0 {
        return;
    }

    with_context(|ctx| {
        if !ctx.vertex_array_enabled || ctx.array_base(ctx.vertex_pointer).is_none() {
            return;
        }

        let Some(idx_base) = ctx.element_base(indices) else {
            return;
        };

        let num_indices = count as usize;
        let mut idx_vec = Vec::with_capacity(num_indices);
        // Compact gather: transform each referenced vertex exactly once and
        // remap indices, instead of reading the whole 0..=max_index span
        // (MCPE index buffers routinely reference a sparse subset).
        let mut remap: Vec<u32> = Vec::new();
        let mut vertices: Vec<VertexData> = Vec::new();
        for i in 0..num_indices {
            let idx = match type_ {
                GL_UNSIGNED_BYTE => *idx_base.add(i) as u32,
                GL_UNSIGNED_SHORT => *(idx_base.add(i * 2) as *const u16) as u32,
                GL_UNSIGNED_INT => *(idx_base.add(i * 4) as *const u32),
                _ => *(idx_base.add(i * 2) as *const u16) as u32,
            };
            let slot = {
                let needed = idx as usize + 1;
                if remap.len() < needed {
                    remap.resize(needed, u32::MAX);
                }
                let r = &mut remap[idx as usize];
                if *r == u32::MAX {
                    *r = vertices.len() as u32;
                    vertices.push(ctx.read_vertex(idx as usize));
                }
                *r
            };
            idx_vec.push(slot);
        }

        ctx.draw_vertex_data(mode, &vertices, Some(&idx_vec));
    });
}

// ============================================================================
// Immediate Mode (glBegin/glEnd)
// ============================================================================

pub unsafe extern "C" fn glBegin(mode: GLenum) {
    with_context(|ctx| {
        ctx.immediate_mode = Some(mode);
        ctx.immediate_vertices.clear();
    });
}

pub unsafe extern "C" fn glEnd() {
    with_context(|ctx| {
        if let Some(mode) = ctx.immediate_mode.take() {
            let verts = core::mem::take(&mut ctx.immediate_vertices);
            ctx.draw_vertex_data(mode, &verts, None);
        }
    });
}

pub unsafe extern "C" fn glVertex3f(x: GLfloat, y: GLfloat, z: GLfloat) {
    with_context(|ctx| {
        if ctx.immediate_mode.is_some() {
            ctx.immediate_vertices.push(VertexData {
                position: [x, y, z],
                tex_coord: ctx.current_texcoord,
                color: ctx.current_color,
                normal: ctx.current_normal,
            });
        }
    });
}

pub unsafe extern "C" fn glVertex2f(x: GLfloat, y: GLfloat) {
    glVertex3f(x, y, 0.0);
}

pub unsafe extern "C" fn glTexCoord2f(u: GLfloat, v: GLfloat) {
    with_context(|ctx| {
        ctx.current_texcoord = [u, v];
        if let Some(list) = &mut ctx.active_display_list {
            list.push_op(crate::display_list::DisplayListOp::TexCoord2f(u, v));
        }
    });
}

pub unsafe extern "C" fn glMultiTexCoord2f(_target: GLenum, s: GLfloat, t: GLfloat) {
    glTexCoord2f(s, t);
}

pub unsafe extern "C" fn glMultiTexCoord4f(
    _target: GLenum,
    s: GLfloat,
    t: GLfloat,
    _r: GLfloat,
    _q: GLfloat,
) {
    glTexCoord2f(s, t);
}

pub unsafe extern "C" fn glColor4f(r: GLfloat, g: GLfloat, b: GLfloat, a: GLfloat) {
    with_context(|ctx| {
        ctx.current_color = [r, g, b, a];
        if let Some(list) = &mut ctx.active_display_list {
            list.push_op(crate::display_list::DisplayListOp::Color4f(r, g, b, a));
        }
    });
}

pub unsafe extern "C" fn glColor3f(r: GLfloat, g: GLfloat, b: GLfloat) {
    glColor4f(r, g, b, 1.0);
}

pub unsafe extern "C" fn glColor4ub(r: GLubyte, g: GLubyte, b: GLubyte, a: GLubyte) {
    glColor4f(
        r as f32 / 255.0,
        g as f32 / 255.0,
        b as f32 / 255.0,
        a as f32 / 255.0,
    );
}

pub unsafe extern "C" fn glColor3ub(r: GLubyte, g: GLubyte, b: GLubyte) {
    glColor4ub(r, g, b, 255);
}

pub unsafe extern "C" fn glColor4fv(v: *const GLfloat) {
    if !v.is_null() {
        glColor4f(*v.add(0), *v.add(1), *v.add(2), *v.add(3));
    }
}

pub unsafe extern "C" fn glNormal3f(x: GLfloat, y: GLfloat, z: GLfloat) {
    with_context(|ctx| {
        ctx.current_normal = [x, y, z];
        if let Some(list) = &mut ctx.active_display_list {
            list.push_op(crate::display_list::DisplayListOp::Normal3f(x, y, z));
        }
    });
}

// ============================================================================
// Display Lists
// ============================================================================

pub unsafe extern "C" fn glGenLists(range: GLsizei) -> GLuint {
    if range <= 0 {
        return 0;
    }
    with_context(|ctx| ctx.display_lists.gen_lists(range as usize))
}

pub unsafe extern "C" fn glDeleteLists(list: GLuint, range: GLsizei) {
    if range <= 0 {
        return;
    }
    with_context(|ctx| {
        ctx.display_lists.delete_lists(list, range as usize);
    });
}

pub unsafe extern "C" fn glNewList(list: GLuint, mode: GLenum) {
    with_context(|ctx| {
        ctx.active_display_list = Some(crate::display_list::DisplayList::new(list));
        ctx.display_list_mode = mode;
    });
}

pub unsafe extern "C" fn glEndList() {
    with_context(|ctx| {
        if let Some(list) = ctx.active_display_list.take() {
            ctx.display_lists.store_list(list);
        }
    });
}

pub unsafe extern "C" fn glCallList(list: GLuint) {
    with_context(|ctx| {
        if let Some(active) = &mut ctx.active_display_list {
            active.push_op(crate::display_list::DisplayListOp::CallList(list));
            if ctx.display_list_mode == GL_COMPILE {
                return;
            }
        }
        ctx.call_display_list(list);
    });
}

pub unsafe extern "C" fn glCallLists(n: GLsizei, type_: GLenum, lists: *const c_void) {
    if n <= 0 || lists.is_null() {
        return;
    }
    for i in 0..n as usize {
        let list_id = match type_ {
            GL_UNSIGNED_BYTE => *(lists as *const u8).add(i) as u32,
            GL_UNSIGNED_SHORT => *(lists as *const u16).add(i) as u32,
            GL_UNSIGNED_INT => *(lists as *const u32).add(i),
            GL_INT => *(lists as *const i32).add(i) as u32,
            _ => *(lists as *const u32).add(i),
        };
        glCallList(list_id);
    }
}

pub unsafe extern "C" fn glIsList(list: GLuint) -> GLboolean {
    with_context(|ctx| {
        if ctx.display_lists.is_list(list) {
            GL_TRUE
        } else {
            GL_FALSE
        }
    })
}

// ============================================================================
// Textures
// ============================================================================

pub unsafe extern "C" fn glGenTextures(n: GLsizei, textures: *mut GLuint) {
    if n <= 0 || textures.is_null() {
        return;
    }
    with_context(|ctx| {
        let tm = &mut ctx.texture_manager;
        let ids = tm.gen_textures(n as usize);
        for (i, id) in ids.iter().enumerate() {
            *textures.add(i) = *id;
        }
    });
}

pub unsafe extern "C" fn glDeleteTextures(n: GLsizei, textures: *const GLuint) {
    if n <= 0 || textures.is_null() {
        return;
    }
    with_context(|ctx| {
        crate::flush_pending(ctx);
        let slice = core::slice::from_raw_parts(textures, n as usize);
        ctx.texture_manager.delete_textures(slice);
    });
}

pub unsafe extern "C" fn glBindTexture(target: GLenum, texture: GLuint) {
    with_context(|ctx| {
        if let Some(list) = &mut ctx.active_display_list {
            list.push_op(crate::display_list::DisplayListOp::BindTexture(texture));
        }
        ctx.texture_manager.bind_texture(target, texture);
    });
}

pub unsafe extern "C" fn glTexImage2D(
    _target: GLenum,
    level: GLint,
    internalformat: GLint,
    width: GLsizei,
    height: GLsizei,
    _border: GLint,
    format: GLenum,
    type_: GLenum,
    pixels: *const c_void,
) {
    if width <= 0 || height <= 0 {
        return;
    }
    with_context(|ctx| {
        crate::flush_pending(ctx);
        let tm = &mut ctx.texture_manager;
        if let Some(tex) = tm.get_current_texture_mut() {
            let data_slice = if !pixels.is_null() {
                let num_pixels = (width * height) as usize;
                let bpp = crate::texture::bytes_per_pixel(format, type_);
                Some(core::slice::from_raw_parts(
                    pixels as *const u8,
                    num_pixels * bpp,
                ))
            } else {
                None
            };

            tex.set_image_data(
                level as u32,
                internalformat as GLenum,
                width as u32,
                height as u32,
                format,
                type_,
                data_slice,
            );
        }
    });
}

pub unsafe extern "C" fn glTexSubImage2D(
    _target: GLenum,
    level: GLint,
    xoffset: GLint,
    yoffset: GLint,
    width: GLsizei,
    height: GLsizei,
    format: GLenum,
    type_: GLenum,
    pixels: *const c_void,
) {
    if width <= 0 || height <= 0 || pixels.is_null() {
        return;
    }
    with_context(|ctx| {
        crate::flush_pending(ctx);
        let tm = &mut ctx.texture_manager;
        if let Some(tex) = tm.get_current_texture_mut() {
            let num_pixels = (width * height) as usize;
            let bpp = crate::texture::bytes_per_pixel(format, type_);
            let data_slice = core::slice::from_raw_parts(pixels as *const u8, num_pixels * bpp);
            tex.set_sub_image_data(
                level as u32,
                xoffset as u32,
                yoffset as u32,
                width as u32,
                height as u32,
                format,
                type_,
                data_slice,
            );
        }
    });
}

pub unsafe extern "C" fn glTexParameteri(_target: GLenum, pname: GLenum, param: GLint) {
    with_context(|ctx| {
        let tm = &mut ctx.texture_manager;
        if let Some(tex) = tm.get_current_texture_mut() {
            match pname {
                GL_TEXTURE_MIN_FILTER => tex.min_filter = param as GLenum,
                GL_TEXTURE_MAG_FILTER => tex.mag_filter = param as GLenum,
                GL_TEXTURE_WRAP_S => tex.wrap_s = param as GLenum,
                GL_TEXTURE_WRAP_T => tex.wrap_t = param as GLenum,
                _ => {}
            }
            tex.dirty = true;
        }
    });
}

pub unsafe extern "C" fn glTexParameterf(target: GLenum, pname: GLenum, param: GLfloat) {
    glTexParameteri(target, pname, param as GLint);
}

pub unsafe extern "C" fn glTexParameteriv(target: GLenum, pname: GLenum, params: *const GLint) {
    if !params.is_null() {
        glTexParameteri(target, pname, *params);
    }
}

pub unsafe extern "C" fn glTexParameterfv(target: GLenum, pname: GLenum, params: *const GLfloat) {
    if !params.is_null() {
        glTexParameterf(target, pname, *params);
    }
}

pub unsafe extern "C" fn glActiveTexture(texture: GLenum) {
    with_context(|ctx| {
        let idx = (texture.saturating_sub(GL_TEXTURE0)) as usize;
        if idx < 8 {
            ctx.texture_manager.active_unit = idx;
        }
    });
}

pub unsafe extern "C" fn glTexImage1D(
    target: GLenum,
    level: GLint,
    internalformat: GLint,
    width: GLsizei,
    border: GLint,
    format: GLenum,
    type_: GLenum,
    pixels: *const c_void,
) {
    glTexImage2D(
        target,
        level,
        internalformat,
        width,
        1,
        border,
        format,
        type_,
        pixels,
    );
}

pub unsafe extern "C" fn glTexImage3D(
    _target: GLenum,
    _level: GLint,
    _internalformat: GLint,
    _width: GLsizei,
    _height: GLsizei,
    _depth: GLsizei,
    _border: GLint,
    _format: GLenum,
    _type_: GLenum,
    _pixels: *const c_void,
) {
}

pub unsafe extern "C" fn glGetTexLevelParameteri(
    _target: GLenum,
    _level: GLint,
    pname: GLenum,
    params: *mut GLint,
) {
    if params.is_null() {
        return;
    }
    with_context(|ctx| {
        let tm = &ctx.texture_manager;
        if let Some(tex) = tm.get_current_texture() {
            match pname {
                GL_TEXTURE_WIDTH => *params = tex.width as GLint,
                GL_TEXTURE_HEIGHT => *params = tex.height as GLint,
                GL_TEXTURE_INTERNAL_FORMAT => *params = tex.internal_format as GLint,
                _ => *params = 0,
            }
        }
    });
}

pub unsafe extern "C" fn glTexGen(_coord: GLenum, _pname: GLenum, _param: GLfloat) {}

pub unsafe extern "C" fn glTexGeni(_coord: GLenum, _pname: GLenum, _param: GLint) {}

pub unsafe extern "C" fn glTexEnvf(target: GLenum, pname: GLenum, param: GLfloat) {
    glTexEnvi(target, pname, param as GLint);
}

pub unsafe extern "C" fn glTexEnvi(target: GLenum, pname: GLenum, param: GLint) {
    with_context(|ctx| {
        let unit = ctx.texture_manager.active_unit.min(1);
        if target == GL_TEXTURE_ENV && pname == GL_TEXTURE_ENV_MODE {
            ctx.texenv_mode[unit] = param as GLenum;
        }
    });
}

pub unsafe extern "C" fn glTexEnvfv(target: GLenum, pname: GLenum, params: *const GLfloat) {
    if params.is_null() {
        return;
    }
    with_context(|ctx| {
        let unit = ctx.texture_manager.active_unit.min(1);
        if target == GL_TEXTURE_ENV {
            if pname == GL_TEXTURE_ENV_COLOR {
                unsafe {
                    ctx.texenv_color[unit] =
                        [*params, *params.add(1), *params.add(2), *params.add(3)];
                }
            } else if pname == GL_TEXTURE_ENV_MODE {
                unsafe {
                    ctx.texenv_mode[unit] = *params as GLenum;
                }
            }
        }
    });
}

// ============================================================================
// States and Enables
// ============================================================================

pub unsafe extern "C" fn glEnable(cap: GLenum) {
    with_context(|ctx| {
        if let Some(list) = &mut ctx.active_display_list {
            list.push_op(crate::display_list::DisplayListOp::Enable(cap));
        }
        ctx.set_enable(cap, true);
    });
}

pub unsafe extern "C" fn glDisable(cap: GLenum) {
    with_context(|ctx| {
        if let Some(list) = &mut ctx.active_display_list {
            list.push_op(crate::display_list::DisplayListOp::Disable(cap));
        }
        ctx.set_enable(cap, false);
    });
}

pub unsafe extern "C" fn glIsEnabled(cap: GLenum) -> GLboolean {
    with_context(|ctx| {
        let is_on = match cap {
            GL_TEXTURE_2D => ctx.texture_2d_enabled,
            GL_BLEND => ctx.blend_enabled,
            GL_DEPTH_TEST => ctx.depth_test_enabled,
            GL_ALPHA_TEST => ctx.alpha_test_enabled,
            GL_CULL_FACE => ctx.cull_face_enabled,
            GL_LIGHTING => ctx.lighting_enabled,
            GL_FOG => ctx.fog_enabled,
            GL_SCISSOR_TEST => ctx.scissor_test_enabled,
            GL_STENCIL_TEST => ctx.stencil_test_enabled,
            GL_COLOR_MATERIAL => ctx.color_material_enabled,
            GL_RESCALE_NORMAL => ctx.rescale_normal_enabled,
            GL_NORMALIZE => ctx.normalize_enabled,
            GL_POLYGON_OFFSET_FILL => ctx.polygon_offset_fill_enabled,
            _ => false,
        };
        if is_on { GL_TRUE } else { GL_FALSE }
    })
}

pub unsafe extern "C" fn glAlphaFunc(func: GLenum, ref_val: GLclampf) {
    with_context(|ctx| {
        if let Some(list) = &mut ctx.active_display_list {
            list.push_op(crate::display_list::DisplayListOp::AlphaFunc(func, ref_val));
        }
        ctx.alpha_func = func;
        ctx.alpha_ref = ref_val;
    });
}

pub unsafe extern "C" fn glBlendFunc(sfactor: GLenum, dfactor: GLenum) {
    with_context(|ctx| {
        if let Some(list) = &mut ctx.active_display_list {
            list.push_op(crate::display_list::DisplayListOp::BlendFunc(
                sfactor, dfactor,
            ));
        }
        ctx.src_factor = sfactor;
        ctx.dst_factor = dfactor;
        ctx.src_factor_alpha = sfactor;
        ctx.dst_factor_alpha = dfactor;
    });
}

pub unsafe extern "C" fn glBlendColor(
    red: GLclampf,
    green: GLclampf,
    blue: GLclampf,
    alpha: GLclampf,
) {
    with_context(|ctx| {
        ctx.blend_color = [red, green, blue, alpha];
    });
}

pub unsafe extern "C" fn glBlendFuncSeparate(
    srcRGB: GLenum,
    dstRGB: GLenum,
    srcAlpha: GLenum,
    dstAlpha: GLenum,
) {
    with_context(|ctx| {
        ctx.src_factor = srcRGB;
        ctx.dst_factor = dstRGB;
        ctx.src_factor_alpha = srcAlpha;
        ctx.dst_factor_alpha = dstAlpha;
    });
}

pub unsafe extern "C" fn glDepthFunc(func: GLenum) {
    with_context(|ctx| {
        if let Some(list) = &mut ctx.active_display_list {
            list.push_op(crate::display_list::DisplayListOp::DepthFunc(func));
        }
        ctx.depth_func = func;
    });
}

pub unsafe extern "C" fn glDepthMask(flag: GLboolean) {
    with_context(|ctx| {
        let enable = flag != GL_FALSE;
        if let Some(list) = &mut ctx.active_display_list {
            list.push_op(crate::display_list::DisplayListOp::DepthMask(enable));
        }
        ctx.depth_mask = enable;
    });
}

pub unsafe extern "C" fn glColorMask(
    red: GLboolean,
    green: GLboolean,
    blue: GLboolean,
    alpha: GLboolean,
) {
    with_context(|ctx| {
        ctx.color_mask = (
            red != GL_FALSE,
            green != GL_FALSE,
            blue != GL_FALSE,
            alpha != GL_FALSE,
        );
    });
}

pub unsafe extern "C" fn glCullFace(mode: GLenum) {
    with_context(|ctx| {
        if let Some(list) = &mut ctx.active_display_list {
            list.push_op(crate::display_list::DisplayListOp::CullFace(mode));
        }
        ctx.cull_face_mode = mode;
    });
}

pub unsafe extern "C" fn glFrontFace(mode: GLenum) {
    with_context(|ctx| {
        ctx.front_face = mode;
    });
}

pub unsafe extern "C" fn glPolygonOffset(factor: GLfloat, units: GLfloat) {
    with_context(|ctx| {
        if let Some(list) = &mut ctx.active_display_list {
            list.push_op(crate::display_list::DisplayListOp::PolygonOffset(
                factor, units,
            ));
        }
        ctx.polygon_offset_factor = factor;
        ctx.polygon_offset_units = units;
    });
}

pub unsafe extern "C" fn glLineWidth(width: GLfloat) {
    with_context(|ctx| {
        ctx.line_width = width;
    });
}

pub unsafe extern "C" fn glPointSize(size: GLfloat) {
    with_context(|ctx| {
        ctx.point_size = size;
    });
}

pub unsafe extern "C" fn glShadeModel(mode: GLenum) {
    with_context(|ctx| {
        if let Some(list) = &mut ctx.active_display_list {
            list.push_op(crate::display_list::DisplayListOp::ShadeModel(mode));
        }
        ctx.shade_model = mode;
    });
}

pub unsafe extern "C" fn glColorMaterial(_face: GLenum, _mode: GLenum) {
    with_context(|ctx| {
        ctx.color_material_enabled = true;
    });
}

pub unsafe extern "C" fn glFogf(pname: GLenum, param: GLfloat) {
    with_context(|ctx| match pname {
        GL_FOG_START => ctx.fog_start = param,
        GL_FOG_END => ctx.fog_end = param,
        GL_FOG_DENSITY => ctx.fog_density = param,
        GL_FOG_MODE => ctx.fog_mode = param as GLenum,
        _ => {}
    });
}

pub unsafe extern "C" fn glFogfv(pname: GLenum, params: *const GLfloat) {
    if params.is_null() {
        return;
    }
    with_context(|ctx| {
        if pname == GL_FOG_COLOR {
            ctx.fog_color = [
                *params.add(0),
                *params.add(1),
                *params.add(2),
                *params.add(3),
            ];
        } else {
            glFogf(pname, *params);
        }
    });
}

pub unsafe extern "C" fn glFogi(pname: GLenum, param: GLint) {
    glFogf(pname, param as GLfloat);
}

pub unsafe extern "C" fn glFogx(pname: GLenum, param: GLfixed) {
    if pname == GL_FOG_MODE {
        glFogf(pname, param as GLfloat);
    } else {
        glFogf(pname, fixed_to_float(param));
    }
}

pub unsafe extern "C" fn glFogxv(pname: GLenum, params: *const GLfixed) {
    if params.is_null() {
        return;
    }
    with_context(|ctx| {
        if pname == GL_FOG_COLOR {
            ctx.fog_color = [
                fixed_to_float(*params.add(0)),
                fixed_to_float(*params.add(1)),
                fixed_to_float(*params.add(2)),
                fixed_to_float(*params.add(3)),
            ];
        } else {
            glFogx(pname, *params);
        }
    });
}

pub unsafe extern "C" fn glHint(target: GLenum, mode: GLenum) {
    match mode {
        GL_DONT_CARE | GL_FASTEST | GL_NICEST => {}
        _ => {
            with_context(|ctx| ctx.error = GL_INVALID_ENUM);
            return;
        }
    }
    with_context(|ctx| {
        ctx.hints.insert(target, mode);
    });
}

pub unsafe extern "C" fn glDepthRangef(n: GLclampf, f: GLclampf) {
    with_context(|ctx| {
        ctx.depth_range = (n.clamp(0.0, 1.0), f.clamp(0.0, 1.0));
    });
}

pub unsafe extern "C" fn glDepthRange(n: GLclampd, f: GLclampd) {
    glDepthRangef(n as GLclampf, f as GLclampf);
}

pub unsafe extern "C" fn glDepthRangex(n: GLclampx, f: GLclampx) {
    glDepthRangef(clampx_to_float(n), clampx_to_float(f));
}

pub unsafe extern "C" fn glLightf(_light: GLenum, _pname: GLenum, _param: GLfloat) {}

pub unsafe extern "C" fn glLightfv(light: GLenum, pname: GLenum, params: *const GLfloat) {
    if params.is_null() {
        return;
    }
    with_context(|ctx| {
        let idx = (light.saturating_sub(GL_LIGHT0)) as usize;
        if idx < 8 {
            match pname {
                GL_POSITION => {
                    ctx.lights[idx].position = [
                        *params.add(0),
                        *params.add(1),
                        *params.add(2),
                        *params.add(3),
                    ];
                }
                GL_DIFFUSE => {
                    ctx.lights[idx].diffuse = [
                        *params.add(0),
                        *params.add(1),
                        *params.add(2),
                        *params.add(3),
                    ];
                }
                GL_AMBIENT => {
                    ctx.lights[idx].ambient = [
                        *params.add(0),
                        *params.add(1),
                        *params.add(2),
                        *params.add(3),
                    ];
                }
                GL_SPECULAR => {
                    ctx.lights[idx].specular = [
                        *params.add(0),
                        *params.add(1),
                        *params.add(2),
                        *params.add(3),
                    ];
                }
                _ => {}
            }
        }
    });
}

pub unsafe extern "C" fn glLightModelf(_pname: GLenum, _param: GLfloat) {}

pub unsafe extern "C" fn glLightModelfv(pname: GLenum, params: *const GLfloat) {
    if params.is_null() {
        return;
    }
    with_context(|ctx| {
        if pname == GL_LIGHT_MODEL_AMBIENT {
            ctx.light_model_ambient = [
                *params.add(0),
                *params.add(1),
                *params.add(2),
                *params.add(3),
            ];
        }
    });
}

pub unsafe extern "C" fn glMaterialf(_face: GLenum, _pname: GLenum, _param: GLfloat) {}

pub unsafe extern "C" fn glMaterialfv(_face: GLenum, pname: GLenum, params: *const GLfloat) {
    if params.is_null() {
        return;
    }
    with_context(|ctx| match pname {
        GL_AMBIENT_AND_DIFFUSE | GL_DIFFUSE | GL_AMBIENT => {
            ctx.material_ambient_diffuse = [
                *params.add(0),
                *params.add(1),
                *params.add(2),
                if pname == GL_AMBIENT {
                    1.0
                } else {
                    *params.add(3)
                },
            ];
        }
        _ => {}
    });
}

pub unsafe extern "C" fn glStencilFunc(func: GLenum, ref_val: GLint, mask: GLuint) {
    with_context(|ctx| {
        ctx.stencil_func = func;
        ctx.stencil_ref = ref_val;
        ctx.stencil_value_mask = mask;
    });
}

pub unsafe extern "C" fn glStencilMask(mask: GLuint) {
    with_context(|ctx| {
        ctx.stencil_writemask = mask;
    });
}

pub unsafe extern "C" fn glStencilOp(fail: GLenum, zfail: GLenum, zpass: GLenum) {
    with_context(|ctx| {
        ctx.stencil_fail = fail;
        ctx.stencil_zfail = zfail;
        ctx.stencil_zpass = zpass;
    });
}

pub unsafe extern "C" fn glViewport(x: GLint, y: GLint, width: GLsizei, height: GLsizei) {
    with_context(|ctx| {
        ctx.viewport = (x, y, width, height);
    });
}

pub unsafe extern "C" fn glScissor(x: GLint, y: GLint, width: GLsizei, height: GLsizei) {
    with_context(|ctx| {
        ctx.scissor = (x, y, width, height);
    });
}

pub unsafe extern "C" fn glClearColor(
    red: GLclampf,
    green: GLclampf,
    blue: GLclampf,
    alpha: GLclampf,
) {
    with_context(|ctx| {
        ctx.clear_color = [red, green, blue, alpha];
    });
}

pub unsafe extern "C" fn glClearDepthf(depth: GLclampf) {
    with_context(|ctx| {
        ctx.clear_depth = depth;
    });
}

pub unsafe extern "C" fn glClearDepth(depth: GLclampd) {
    glClearDepthf(depth as GLclampf);
}

pub unsafe extern "C" fn glClearStencil(s: GLint) {
    with_context(|ctx| {
        ctx.clear_stencil = s;
    });
}

pub unsafe extern "C" fn glClear(mask: GLbitfield) {
    with_context(|ctx| {
        let mut hal_mask = 0u32;
        if (mask & GL_COLOR_BUFFER_BIT) != 0 {
            hal_mask |= 1;
        }
        if (mask & GL_DEPTH_BUFFER_BIT) != 0 {
            hal_mask |= 2;
        }
        if (mask & GL_STENCIL_BUFFER_BIT) != 0 {
            hal_mask |= 4;
        }

        ctx.command_buffer.push(volt_hal::Cmd::ClearAttachments {
            color: ctx.clear_color,
            depth: ctx.clear_depth,
            stencil: ctx.clear_stencil as u8,
            mask: hal_mask,
        });
    });
}

pub unsafe extern "C" fn glPixelStorei(_pname: GLenum, _param: GLint) {}

pub unsafe extern "C" fn glReadPixels(
    _x: GLint,
    _y: GLint,
    width: GLsizei,
    height: GLsizei,
    _format: GLenum,
    _type_: GLenum,
    pixels: *mut c_void,
) {
    if pixels.is_null() || width <= 0 || height <= 0 {
        return;
    }
    // Fill with opaque black by default if not read back
    let size = (width * height * 4) as usize;
    core::ptr::write_bytes(pixels as *mut u8, 0, size);
    // MISSING: readback via hal CopyImageToBuffer (Phase 2/4).
}

pub unsafe extern "C" fn glFlush() {
    with_context(|ctx| {
        let cmd = core::mem::take(&mut ctx.command_buffer);
        ctx.hal_device.submit(&cmd);
    });
}

/// Submit pending GL commands. Called before operations that would invalidate
/// state already referenced by queued commands (texture redefine/delete) so
/// rasterization observes a consistent snapshot.
pub fn flush_pending(ctx: &mut GlContext) {
    if !ctx.command_buffer.ops.is_empty() {
        let cmd = core::mem::take(&mut ctx.command_buffer);
        ctx.hal_device.submit(&cmd);
    }
}

pub unsafe extern "C" fn glFinish() {
    glFlush();
}
pub unsafe extern "C" fn glGetIntegerv(pname: GLenum, params: *mut GLint) {
    if params.is_null() {
        return;
    }
    with_context(|ctx| match pname {
        GL_VIEWPORT => {
            *params.add(0) = ctx.viewport.0;
            *params.add(1) = ctx.viewport.1;
            *params.add(2) = ctx.viewport.2;
            *params.add(3) = ctx.viewport.3;
        }
        GL_SCISSOR_BOX => {
            *params.add(0) = ctx.scissor.0;
            *params.add(1) = ctx.scissor.1;
            *params.add(2) = ctx.scissor.2;
            *params.add(3) = ctx.scissor.3;
        }
        GL_MATRIX_MODE => {
            *params = match ctx.matrix_mode {
                MatrixMode::ModelView => GL_MODELVIEW as GLint,
                MatrixMode::Projection => GL_PROJECTION as GLint,
                MatrixMode::Texture => GL_TEXTURE as GLint,
            };
        }
        GL_MAX_TEXTURE_SIZE => *params = 8192,
        GL_MAX_VIEWPORT_DIMS => {
            *params.add(0) = 8192;
            *params.add(1) = 8192;
        }
        GL_STENCIL_BITS => *params = 8,
        GL_DEPTH_BITS => *params = 24,
        GL_RED_BITS => *params = 8,
        GL_GREEN_BITS => *params = 8,
        GL_BLUE_BITS => *params = 8,
        GL_ALPHA_BITS => *params = 8,
        _ => *params = 0,
    });
}

pub unsafe extern "C" fn glGetFloatv(pname: GLenum, params: *mut GLfloat) {
    if params.is_null() {
        return;
    }
    with_context(|ctx| match pname {
        GL_MODELVIEW_MATRIX => {
            let data = ctx.modelview_stack.current.to_array();
            core::ptr::copy_nonoverlapping(data.as_ptr(), params, 16);
        }
        GL_PROJECTION_MATRIX => {
            let data = ctx.projection_stack.current.to_array();
            core::ptr::copy_nonoverlapping(data.as_ptr(), params, 16);
        }
        GL_TEXTURE_MATRIX => {
            let data = ctx.texture_stack.current.to_array();
            core::ptr::copy_nonoverlapping(data.as_ptr(), params, 16);
        }
        GL_DEPTH_RANGE => {
            *params.add(0) = ctx.depth_range.0;
            *params.add(1) = ctx.depth_range.1;
        }
        _ => {}
    });
}

pub unsafe extern "C" fn glGetBooleanv(pname: GLenum, params: *mut GLboolean) {
    if params.is_null() {
        return;
    }
    *params = glIsEnabled(pname);
}

pub unsafe extern "C" fn glGetString(name: GLenum) -> *const GLubyte {
    match name {
        GL_VENDOR => c"Vantage".as_ptr().cast(),
        GL_RENDERER => c"Vantage Software (pliron)".as_ptr().cast(),
        GL_VERSION => c"OpenGL ES-CM 1.1".as_ptr().cast(),
        GL_EXTENSIONS => {
            c"GL_OES_texture_npot GL_OES_packed_depth_stencil GL_EXT_texture_format_BGRA8888"
                .as_ptr()
                .cast()
        }
        _ => c"".as_ptr().cast(),
    }
}

pub unsafe extern "C" fn glGetError() -> GLenum {
    with_context(|ctx| {
        let err = ctx.error;
        ctx.error = GL_NO_ERROR;
        err
    })
}

pub unsafe extern "C" fn glPushAttrib(_mask: GLbitfield) {}

pub unsafe extern "C" fn glPopAttrib() {}

pub unsafe extern "C" fn glPushClientAttrib(_mask: GLbitfield) {}

pub unsafe extern "C" fn glPopClientAttrib() {}

// ============================================================================
// GLES2 Shaders & Buffers & Programs
// ============================================================================

/// MISSING: GLES2 program pipeline — needs GLSL ES 1.00 -> pliron frontend (planned: pliron-based codegen, see prism vulcan-glsl).
pub unsafe extern "C" fn glCreateShader(_shader_type: GLenum) -> GLuint {
    set_gl_error(GL_INVALID_OPERATION);
    0
}

/// MISSING: GLES2 program pipeline — needs GLSL ES 1.00 -> pliron frontend.
pub unsafe extern "C" fn glShaderSource(
    _shader: GLuint,
    _count: GLsizei,
    _string: *const *const GLchar,
    _length: *const GLint,
) {
    set_gl_error(GL_INVALID_OPERATION);
}

/// MISSING: GLES2 program pipeline — needs GLSL ES 1.00 -> pliron frontend.
pub unsafe extern "C" fn glCompileShader(_shader: GLuint) {
    set_gl_error(GL_INVALID_OPERATION);
}

pub unsafe extern "C" fn glGetShaderiv(_shader: GLuint, pname: GLenum, params: *mut GLint) {
    if !params.is_null() {
        if pname == GL_COMPILE_STATUS {
            *params = GL_TRUE as GLint;
        } else {
            *params = 0;
        }
    }
}

pub unsafe extern "C" fn glGetShaderInfoLog(
    _shader: GLuint,
    _buf_size: GLsizei,
    length: *mut GLsizei,
    info_log: *mut GLchar,
) {
    if !length.is_null() {
        *length = 0;
    }
    if !info_log.is_null() {
        *info_log = 0;
    }
}

pub unsafe extern "C" fn glDeleteShader(_shader: GLuint) {}

/// MISSING: GLES2 program pipeline — needs GLSL ES 1.00 -> pliron frontend.
pub unsafe extern "C" fn glCreateProgram() -> GLuint {
    set_gl_error(GL_INVALID_OPERATION);
    0
}

/// MISSING: GLES2 program pipeline — needs GLSL ES 1.00 -> pliron frontend.
pub unsafe extern "C" fn glAttachShader(_program: GLuint, _shader: GLuint) {
    set_gl_error(GL_INVALID_OPERATION);
}

pub unsafe extern "C" fn glDetachShader(_program: GLuint, _shader: GLuint) {}

/// MISSING: GLES2 program pipeline — needs GLSL ES 1.00 -> pliron frontend.
pub unsafe extern "C" fn glLinkProgram(_program: GLuint) {
    set_gl_error(GL_INVALID_OPERATION);
}

pub unsafe extern "C" fn glGetProgramiv(_program: GLuint, pname: GLenum, params: *mut GLint) {
    if !params.is_null() {
        if pname == GL_LINK_STATUS {
            *params = GL_TRUE as GLint;
        } else {
            *params = 0;
        }
    }
}

pub unsafe extern "C" fn glGetProgramInfoLog(
    _program: GLuint,
    _buf_size: GLsizei,
    length: *mut GLsizei,
    info_log: *mut GLchar,
) {
    if !length.is_null() {
        *length = 0;
    }
    if !info_log.is_null() {
        *info_log = 0;
    }
}

/// MISSING: GLES2 program pipeline — needs GLSL ES 1.00 -> pliron frontend.
pub unsafe extern "C" fn glUseProgram(_program: GLuint) {
    set_gl_error(GL_INVALID_OPERATION);
}

pub unsafe extern "C" fn glDeleteProgram(_program: GLuint) {}

pub unsafe extern "C" fn glGetUniformLocation(_program: GLuint, _name: *const GLchar) -> GLint {
    -1
}

pub unsafe extern "C" fn glGetAttribLocation(_program: GLuint, _name: *const GLchar) -> GLint {
    -1
}

pub unsafe extern "C" fn glUniform1f(_location: GLint, _v0: GLfloat) {}
pub unsafe extern "C" fn glUniform2f(_location: GLint, _v0: GLfloat, _v1: GLfloat) {}
pub unsafe extern "C" fn glUniform3f(_location: GLint, _v0: GLfloat, _v1: GLfloat, _v2: GLfloat) {}
pub unsafe extern "C" fn glUniform4f(
    _location: GLint,
    _v0: GLfloat,
    _v1: GLfloat,
    _v2: GLfloat,
    _v3: GLfloat,
) {
}
pub unsafe extern "C" fn glUniform1i(_location: GLint, _v0: GLint) {}
pub unsafe extern "C" fn glUniform2i(_location: GLint, _v0: GLint, _v1: GLint) {}
pub unsafe extern "C" fn glUniform3i(_location: GLint, _v0: GLint, _v1: GLint, _v2: GLint) {}
pub unsafe extern "C" fn glUniform4i(
    _location: GLint,
    _v0: GLint,
    _v1: GLint,
    _v2: GLint,
    _v3: GLint,
) {
}
pub unsafe extern "C" fn glUniformMatrix4fv(
    _location: GLint,
    _count: GLsizei,
    _transpose: GLboolean,
    _value: *const GLfloat,
) {
}

pub unsafe extern "C" fn glGenBuffers(n: GLsizei, buffers: *mut GLuint) {
    if n <= 0 || buffers.is_null() {
        return;
    }
    with_context(|ctx| {
        for i in 0..n as usize {
            let id = ctx.next_buffer_id;
            ctx.next_buffer_id = ctx.next_buffer_id.saturating_add(1);
            ctx.buffers.insert(id, Vec::new());
            *buffers.add(i) = id;
        }
    });
}

pub unsafe extern "C" fn glGenBuffersARB(n: GLsizei, buffers: *mut GLuint) {
    glGenBuffers(n, buffers);
}

pub unsafe extern "C" fn glBindBuffer(target: GLenum, buffer: GLuint) {
    with_context(|ctx| {
        ctx.ensure_buffer(buffer);
        match target {
            GL_ELEMENT_ARRAY_BUFFER => ctx.element_array_buffer_binding = buffer,
            _ => ctx.array_buffer_binding = buffer,
        }
    });
}

pub unsafe extern "C" fn glBindBufferARB(target: GLenum, buffer: GLuint) {
    glBindBuffer(target, buffer);
}

pub unsafe extern "C" fn glBufferData(
    target: GLenum,
    size: GLsizeiptr,
    data: *const c_void,
    _usage: GLenum,
) {
    if size < 0 {
        return;
    }
    with_context(|ctx| {
        let binding = if target == GL_ELEMENT_ARRAY_BUFFER {
            ctx.element_array_buffer_binding
        } else {
            ctx.array_buffer_binding
        };
        if binding == 0 {
            return;
        }
        ctx.ensure_buffer(binding);
        let nbytes = size as usize;
        let mut bytes = vec![0u8; nbytes];
        if !data.is_null() && nbytes > 0 {
            core::ptr::copy_nonoverlapping(data as *const u8, bytes.as_mut_ptr(), nbytes);
        }
        ctx.buffers.insert(binding, bytes);
    });
}

pub unsafe extern "C" fn glBufferDataARB(
    target: GLenum,
    size: GLsizeiptr,
    data: *const c_void,
    usage: GLenum,
) {
    glBufferData(target, size, data, usage);
}

pub unsafe extern "C" fn glBufferSubData(
    target: GLenum,
    offset: GLintptr,
    size: GLsizeiptr,
    data: *const c_void,
) {
    if size <= 0 || offset < 0 || data.is_null() {
        return;
    }
    with_context(|ctx| {
        let binding = if target == GL_ELEMENT_ARRAY_BUFFER {
            ctx.element_array_buffer_binding
        } else {
            ctx.array_buffer_binding
        };
        let Some(buf) = ctx.buffers.get_mut(&binding) else {
            return;
        };
        let start = offset as usize;
        let end = start.saturating_add(size as usize);
        if end > buf.len() {
            buf.resize(end, 0);
        }
        core::ptr::copy_nonoverlapping(
            data as *const u8,
            buf.as_mut_ptr().add(start),
            size as usize,
        );
    });
}

pub unsafe extern "C" fn glDeleteBuffers(n: GLsizei, buffers: *const GLuint) {
    if n <= 0 || buffers.is_null() {
        return;
    }
    with_context(|ctx| {
        for i in 0..n as usize {
            let id = *buffers.add(i);
            ctx.buffers.remove(&id);
            if ctx.array_buffer_binding == id {
                ctx.array_buffer_binding = 0;
            }
            if ctx.element_array_buffer_binding == id {
                ctx.element_array_buffer_binding = 0;
            }
        }
    });
}

pub unsafe extern "C" fn glDeleteBuffersARB(n: GLsizei, buffers: *const GLuint) {
    glDeleteBuffers(n, buffers);
}

pub unsafe extern "C" fn glVertexAttribPointer(
    _index: GLuint,
    _size: GLint,
    _type_: GLenum,
    _normalized: GLboolean,
    _stride: GLsizei,
    _pointer: *const c_void,
) {
}

pub unsafe extern "C" fn glEnableVertexAttribArray(_index: GLuint) {}

pub unsafe extern "C" fn glDisableVertexAttribArray(_index: GLuint) {}

pub unsafe extern "C" fn glGenFramebuffers(n: GLsizei, framebuffers: *mut GLuint) {
    if n <= 0 || framebuffers.is_null() {
        return;
    }
    for i in 0..n as usize {
        *framebuffers.add(i) = (i + 1) as GLuint;
    }
}

pub unsafe extern "C" fn glBindFramebuffer(_target: GLenum, _framebuffer: GLuint) {}

pub unsafe extern "C" fn glFramebufferTexture2D(
    _target: GLenum,
    _attachment: GLenum,
    _textarget: GLenum,
    _texture: GLuint,
    _level: GLint,
) {
}

pub unsafe extern "C" fn glDeleteFramebuffers(_n: GLsizei, _framebuffers: *const GLuint) {}

/// MISSING: Framebuffer object status — unsupported in fixed-function core.
pub unsafe extern "C" fn glCheckFramebufferStatus(_target: GLenum) -> GLenum {
    GL_FRAMEBUFFER_UNSUPPORTED
}

pub unsafe extern "C" fn glGenRenderbuffers(n: GLsizei, renderbuffers: *mut GLuint) {
    if n <= 0 || renderbuffers.is_null() {
        return;
    }
    for i in 0..n as usize {
        *renderbuffers.add(i) = (i + 1) as GLuint;
    }
}

pub unsafe extern "C" fn glBindRenderbuffer(_target: GLenum, _renderbuffer: GLuint) {}

pub unsafe extern "C" fn glRenderbufferStorage(
    _target: GLenum,
    _internalformat: GLenum,
    _width: GLsizei,
    _height: GLsizei,
) {
}

pub unsafe extern "C" fn glDeleteRenderbuffers(_n: GLsizei, _renderbuffers: *const GLuint) {}

pub unsafe extern "C" fn glGenerateMipmap(_target: GLenum) {}

pub unsafe extern "C" fn glGenQueries(n: GLsizei, ids: *mut GLuint) {
    if n <= 0 || ids.is_null() {
        return;
    }
    for i in 0..n as usize {
        *ids.add(i) = (i + 1) as GLuint;
    }
}

pub unsafe extern "C" fn glGenQueriesARB(n: GLsizei, ids: *mut GLuint) {
    glGenQueries(n, ids);
}

pub unsafe extern "C" fn glBeginQuery(_target: GLenum, _id: GLuint) {}
pub unsafe extern "C" fn glBeginQueryARB(target: GLenum, id: GLuint) {
    glBeginQuery(target, id);
}

pub unsafe extern "C" fn glEndQuery(_target: GLenum) {}
pub unsafe extern "C" fn glEndQueryARB(target: GLenum) {
    glEndQuery(target);
}

pub unsafe extern "C" fn glGetQueryObjectuiv(_id: GLuint, _pname: GLenum, params: *mut GLuint) {
    if !params.is_null() {
        *params = 1;
    }
}

pub unsafe extern "C" fn glGetQueryObjectuivARB(id: GLuint, pname: GLenum, params: *mut GLuint) {
    glGetQueryObjectuiv(id, pname, params);
}

pub unsafe extern "C" fn glDeleteQueries(_n: GLsizei, _ids: *const GLuint) {}
pub unsafe extern "C" fn glDeleteQueriesARB(n: GLsizei, ids: *const GLuint) {
    glDeleteQueries(n, ids);
}

// ============================================================================
