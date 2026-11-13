//! Vantage EGL 1.4 implementation and C ABI exports.
//!
//! This crate exposes the EGL/GLES/GBM C ABI (`cdylib`/`staticlib` and
//! cargo-c metadata); it links GLES entry points from `volt-gles` and EGL
//! here into `libvolt_gl.so`.
#![allow(
    non_snake_case,
    non_camel_case_types,
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc
)]
#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod egl;
#[cfg(all(feature = "std", target_os = "linux"))]
pub mod gbm;
pub mod gl_reexports;
pub mod glvnd;

pub use crate::egl::*;
#[cfg(all(feature = "std", target_os = "linux"))]
pub use gbm::*;
pub use volt_gles::types::*;

use core::ffi::c_char;
pub use volt_gles::*;

/// Freestanding support (no `std`): the cdylib still needs a panic handler
/// and a global allocator to link. A fixed-heap bump allocator is provided
/// as the baseline; real freestanding deployments link their own.
// MISSING: freestanding allocator policy — this is a compile-enable baseline.
#[cfg(not(feature = "std"))]
pub mod freestanding {
    extern crate alloc;
    use alloc::alloc::Layout;
    use core::alloc::GlobalAlloc;
    use core::sync::atomic::{AtomicUsize, Ordering};

    const HEAP_SIZE: usize = 16 * 1024 * 1024;
    static mut HEAP: [u8; HEAP_SIZE] = [0; HEAP_SIZE];
    static OFFSET: AtomicUsize = AtomicUsize::new(0);

    pub struct Bump;
    unsafe impl GlobalAlloc for Bump {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let align = layout.align().max(1);
            let mut off = OFFSET.load(Ordering::Relaxed);
            let base = (&raw const HEAP) as usize;
            loop {
                let aligned = (base + off + align - 1) & !(align - 1);
                let next = aligned - base + layout.size();
                if next > HEAP_SIZE {
                    return core::ptr::null_mut();
                }
                if OFFSET
                    .compare_exchange_weak(off, next, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok()
                {
                    return aligned as *mut u8;
                }
                off = OFFSET.load(Ordering::Relaxed);
            }
        }
        unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {}
    }

    #[global_allocator]
    static A: Bump = Bump;

    #[panic_handler]
    fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
        loop {
            core::hint::spin_loop();
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn eglWaitNative(_engine: EGLint) -> EGLBoolean {
    EGL_TRUE
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn eglWaitGL() -> EGLBoolean {
    EGL_TRUE
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn eglWaitClient() -> EGLBoolean {
    EGL_TRUE
}

// ============================================================================
// EGL Exports
// ============================================================================

// ============================================================================

#[unsafe(no_mangle)]
pub unsafe extern "C" fn eglGetDisplay(display_id: NativeDisplayType) -> EGLDisplay {
    egl_get_display(display_id)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn eglInitialize(
    dpy: EGLDisplay,
    major: *mut EGLint,
    minor: *mut EGLint,
) -> EGLBoolean {
    egl_initialize(dpy, major, minor)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn eglTerminate(dpy: EGLDisplay) -> EGLBoolean {
    egl_terminate(dpy)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn eglGetConfigs(
    dpy: EGLDisplay,
    configs: *mut EGLConfig,
    config_size: EGLint,
    num_config: *mut EGLint,
) -> EGLBoolean {
    egl_get_configs(dpy, configs, config_size, num_config)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn eglChooseConfig(
    dpy: EGLDisplay,
    attrib_list: *const EGLint,
    configs: *mut EGLConfig,
    config_size: EGLint,
    num_config: *mut EGLint,
) -> EGLBoolean {
    egl_choose_config(dpy, attrib_list, configs, config_size, num_config)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn eglGetConfigAttrib(
    dpy: EGLDisplay,
    config: EGLConfig,
    attribute: EGLint,
    value: *mut EGLint,
) -> EGLBoolean {
    egl_get_config_attrib(dpy, config, attribute, value)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn eglCreateWindowSurface(
    dpy: EGLDisplay,
    config: EGLConfig,
    win: NativeWindowType,
    attrib_list: *const EGLint,
) -> EGLSurface {
    egl_create_window_surface(dpy, config, win, attrib_list)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn angle_wgpu_create_native_window_surface(
    dpy: EGLDisplay,
    _config: EGLConfig,
    native: *const AngleWgpuNativeWindow,
) -> EGLSurface {
    egl_create_native_window_surface(dpy, native)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn eglCreatePixmapSurface(
    _dpy: EGLDisplay,
    _config: EGLConfig,
    _pixmap: NativePixmapType,
    _attrib_list: *const EGLint,
) -> EGLSurface {
    egl::set_egl_error(EGL_BAD_PARAMETER);
    EGL_NO_SURFACE
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn eglBindAPI(_api: EGLenum) -> EGLBoolean {
    EGL_TRUE
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn eglCreatePbufferSurface(
    dpy: EGLDisplay,
    config: EGLConfig,
    attrib_list: *const EGLint,
) -> EGLSurface {
    egl_create_pbuffer_surface(dpy, config, attrib_list)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn eglDestroySurface(dpy: EGLDisplay, surface: EGLSurface) -> EGLBoolean {
    egl_destroy_surface(dpy, surface)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn eglCreateContext(
    dpy: EGLDisplay,
    config: EGLConfig,
    share_context: EGLContext,
    attrib_list: *const EGLint,
) -> EGLContext {
    egl_create_context(dpy, config, share_context, attrib_list)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn eglDestroyContext(dpy: EGLDisplay, ctx: EGLContext) -> EGLBoolean {
    egl_destroy_context(dpy, ctx)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn eglMakeCurrent(
    dpy: EGLDisplay,
    draw: EGLSurface,
    read: EGLSurface,
    ctx: EGLContext,
) -> EGLBoolean {
    egl_make_current(dpy, draw, read, ctx)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn eglGetCurrentContext() -> EGLContext {
    egl_get_current_context()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn eglGetCurrentSurface(readdraw: EGLint) -> EGLSurface {
    egl_get_current_surface(readdraw)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn eglGetCurrentDisplay() -> EGLDisplay {
    egl_get_current_display()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn eglQuerySurface(
    dpy: EGLDisplay,
    surface: EGLSurface,
    attribute: EGLint,
    value: *mut EGLint,
) -> EGLBoolean {
    egl_query_surface(dpy, surface, attribute, value)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn eglSwapBuffers(dpy: EGLDisplay, surface: EGLSurface) -> EGLBoolean {
    egl_swap_buffers(dpy, surface)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn angle_wgpu_resize_surface(
    surface: EGLSurface,
    width: u32,
    height: u32,
) -> EGLBoolean {
    egl_resize_surface(surface, width, height)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn eglSwapInterval(dpy: EGLDisplay, interval: EGLint) -> EGLBoolean {
    egl_swap_interval(dpy, interval)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn eglGetError() -> EGLint {
    egl_get_error()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn eglQueryString(dpy: EGLDisplay, name: EGLint) -> *const c_char {
    egl_query_string(dpy, name)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn eglGetProcAddress(
    procname: *const c_char,
) -> __eglMustCastToProperFunctionPointerType {
    egl_get_proc_address(procname)
}

pub fn get_gl_proc_address(name: &str) -> __eglMustCastToProperFunctionPointerType {
    let ptr = match name {
        "glMatrixMode" => glMatrixMode as *const (),
        "glLoadIdentity" => glLoadIdentity as *const (),
        "glPushMatrix" => glPushMatrix as *const (),
        "glPopMatrix" => glPopMatrix as *const (),
        "glTranslatef" => glTranslatef as *const (),
        "glRotatef" => glRotatef as *const (),
        "glScalef" => glScalef as *const (),
        "glOrtho" => glOrtho as *const (),
        "glOrthof" => glOrthof as *const (),
        "glFrustum" => glFrustum as *const (),
        "glFrustumf" => glFrustumf as *const (),
        "glMultMatrixf" => glMultMatrixf as *const (),
        "glLoadMatrixf" => glLoadMatrixf as *const (),
        "glEnableClientState" => glEnableClientState as *const (),
        "glDisableClientState" => glDisableClientState as *const (),
        "glVertexPointer" => glVertexPointer as *const (),
        "glTexCoordPointer" => glTexCoordPointer as *const (),
        "glColorPointer" => glColorPointer as *const (),
        "glNormalPointer" => glNormalPointer as *const (),
        "glClientActiveTexture" => glClientActiveTexture as *const (),
        "glDrawArrays" => glDrawArrays as *const (),
        "glDrawElements" => glDrawElements as *const (),
        "glBegin" => glBegin as *const (),
        "glEnd" => glEnd as *const (),
        "glVertex3f" => glVertex3f as *const (),
        "glVertex2f" => glVertex2f as *const (),
        "glTexCoord2f" => glTexCoord2f as *const (),
        "glColor4f" => glColor4f as *const (),
        "glColor3f" => glColor3f as *const (),
        "glColor4ub" => glColor4ub as *const (),
        "glNormal3f" => glNormal3f as *const (),
        "glGenLists" => glGenLists as *const (),
        "glDeleteLists" => glDeleteLists as *const (),
        "glNewList" => glNewList as *const (),
        "glEndList" => glEndList as *const (),
        "glCallList" => glCallList as *const (),
        "glCallLists" => glCallLists as *const (),
        "glGenTextures" => glGenTextures as *const (),
        "glDeleteTextures" => glDeleteTextures as *const (),
        "glBindTexture" => glBindTexture as *const (),
        "glTexImage2D" => glTexImage2D as *const (),
        "glTexSubImage2D" => glTexSubImage2D as *const (),
        "glTexParameteri" => glTexParameteri as *const (),
        "glTexParameterf" => glTexParameterf as *const (),
        "glActiveTexture" => glActiveTexture as *const (),
        "glEnable" => glEnable as *const (),
        "glDisable" => glDisable as *const (),
        "glIsEnabled" => glIsEnabled as *const (),
        "glAlphaFunc" => glAlphaFunc as *const (),
        "glBlendFunc" => glBlendFunc as *const (),
        "glBlendColor" => glBlendColor as *const (),
        "glDepthFunc" => glDepthFunc as *const (),
        "glDepthMask" => glDepthMask as *const (),
        "glColorMask" => glColorMask as *const (),
        "glCullFace" => glCullFace as *const (),
        "glFrontFace" => glFrontFace as *const (),
        "glPolygonOffset" => glPolygonOffset as *const (),
        "glLineWidth" => glLineWidth as *const (),
        "glPointSize" => glPointSize as *const (),
        "glShadeModel" => glShadeModel as *const (),
        "glFogf" => glFogf as *const (),
        "glFogfv" => glFogfv as *const (),
        "glFogi" => glFogi as *const (),
        "glFogx" => glFogx as *const (),
        "glFogxv" => glFogxv as *const (),
        "glHint" => glHint as *const (),
        "glDepthRangef" => glDepthRangef as *const (),
        "glDepthRange" => glDepthRange as *const (),
        "glDepthRangex" => glDepthRangex as *const (),
        "glLightf" => glLightf as *const (),
        "glLightfv" => glLightfv as *const (),
        "glLightModelfv" => glLightModelfv as *const (),
        "glStencilFunc" => glStencilFunc as *const (),
        "glStencilMask" => glStencilMask as *const (),
        "glStencilOp" => glStencilOp as *const (),
        "glViewport" => glViewport as *const (),
        "glScissor" => glScissor as *const (),
        "glClearColor" => glClearColor as *const (),
        "glClearDepthf" => glClearDepthf as *const (),
        "glClear" => glClear as *const (),
        "glReadPixels" => glReadPixels as *const (),
        "glFlush" => glFlush as *const (),
        "glFinish" => glFinish as *const (),
        "glGetIntegerv" => glGetIntegerv as *const (),
        "glGetFloatv" => glGetFloatv as *const (),
        "glGetBooleanv" => glGetBooleanv as *const (),
        "glGetString" => glGetString as *const (),
        "glGetError" => glGetError as *const (),
        "glCreateShader" => glCreateShader as *const (),
        "glShaderSource" => glShaderSource as *const (),
        "glCompileShader" => glCompileShader as *const (),
        "glGetShaderiv" => glGetShaderiv as *const (),
        "glDeleteShader" => glDeleteShader as *const (),
        "glCreateProgram" => glCreateProgram as *const (),
        "glAttachShader" => glAttachShader as *const (),
        "glLinkProgram" => glLinkProgram as *const (),
        "glGetProgramiv" => glGetProgramiv as *const (),
        "glUseProgram" => glUseProgram as *const (),
        "glDeleteProgram" => glDeleteProgram as *const (),
        "glGenBuffers" => glGenBuffers as *const (),
        "glBindBuffer" => glBindBuffer as *const (),
        "glBufferData" => glBufferData as *const (),
        "glDeleteBuffers" => glDeleteBuffers as *const (),
        "glGenFramebuffers" => glGenFramebuffers as *const (),
        "glBindFramebuffer" => glBindFramebuffer as *const (),
        "glFramebufferTexture2D" => glFramebufferTexture2D as *const (),
        "glDeleteFramebuffers" => glDeleteFramebuffers as *const (),
        "eglGetDisplay" => eglGetDisplay as *const (),
        "eglInitialize" => eglInitialize as *const (),
        "eglTerminate" => eglTerminate as *const (),
        "eglChooseConfig" => eglChooseConfig as *const (),
        "eglCreateWindowSurface" => eglCreateWindowSurface as *const (),
        "eglCreatePixmapSurface" => eglCreatePixmapSurface as *const (),
        "eglCreateContext" => eglCreateContext as *const (),
        "eglMakeCurrent" => eglMakeCurrent as *const (),
        "eglSwapBuffers" => eglSwapBuffers as *const (),
        "eglSwapInterval" => eglSwapInterval as *const (),
        "eglQueryString" => eglQueryString as *const (),
        "eglGetProcAddress" => eglGetProcAddress as *const (),
        "eglWaitNative" => eglWaitNative as *const (),
        "eglWaitGL" => eglWaitGL as *const (),
        "eglWaitClient" => eglWaitClient as *const (),
        _ => core::ptr::null(),
    };

    if ptr.is_null() {
        None
    } else {
        Some(unsafe { core::mem::transmute::<*const (), unsafe extern "C" fn()>(ptr) })
    }
}

#[used]
static GLES_EXPORT_KEEP: [unsafe extern "C" fn(); 1] = [unsafe {
    core::mem::transmute::<*const (), unsafe extern "C" fn()>(volt_gles::glMatrixMode as *const ())
}];

#[cfg(test)]
mod tests {
    use super::*;
    use core::ptr;

    #[test]
    fn egl_pbuffer_clear_and_swap() {
        unsafe {
            let dpy = eglGetDisplay(ptr::null_mut());
            assert!(!dpy.is_null());
            let mut major = 0;
            let mut minor = 0;
            assert_eq!(eglInitialize(dpy, &mut major, &mut minor), EGL_TRUE);
            assert_eq!((major, minor), (1, 4));

            let attribs = [EGL_WIDTH, 16, EGL_HEIGHT, 16, EGL_NONE];
            let surf = eglCreatePbufferSurface(dpy, ptr::null_mut(), attribs.as_ptr());
            assert!(!surf.is_null());
            let ctx = eglCreateContext(dpy, ptr::null_mut(), ptr::null_mut(), ptr::null());
            assert!(!ctx.is_null());
            assert_eq!(eglMakeCurrent(dpy, surf, surf, ctx), EGL_TRUE);

            glClearColor(0.25, 0.5, 0.75, 1.0);
            glClear(GL_COLOR_BUFFER_BIT);
            assert_eq!(eglSwapBuffers(dpy, surf), EGL_TRUE);

            let gl_ctx = volt_gles::gl_context::get_current_gl_context().unwrap();
            let gl = gl_ctx.lock();
            let img = gl.hal_device.image(gl.color_image.unwrap()).unwrap();
            let px = img.slice();
            // Format::B8G8R8A8Unorm -> [B, G, R, A]
            assert!(px[0] > 150 && px[1] > 100 && px[2] > 50 && px[3] == 255);
            drop(gl);
            assert!(eglGetProcAddress(c"glClear".as_ptr()).is_some());
            assert!(eglGetProcAddress(c"eglSwapBuffers".as_ptr()).is_some());

            eglMakeCurrent(dpy, ptr::null_mut(), ptr::null_mut(), ptr::null_mut());
            eglDestroyContext(dpy, ctx);
            eglDestroySurface(dpy, surf);
            eglTerminate(dpy);
        }
    }
}
