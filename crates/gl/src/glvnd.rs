//! GLVND EGL vendor-library support.
//!
//! Implements the vendor side of libglvnd's EGL ABI so this library can be
//! loaded as an EGL vendor (see `/usr/share/glvnd/egl_vendor.d/*.json`):
//! glvnd `dlopen`s us, calls `__egl_Main`, and builds its dispatch tables
//! from the `__EGLapiImports` callbacks filled in there.
//!
//! Reference: NVIDIA/libglvnd `include/glvnd/libeglabi.h`,
//! `src/EGL/libeglvendor.c`; Mesa `src/egl/main/eglglvnd.c`.
//!
use crate::egl;
use crate::eglBindAPI;
use crate::eglCreatePixmapSurface;
use alloc::sync::Arc;
use core::ffi::{CStr, c_char, c_void};
use volt_gles::types::*;

/// Pointer-sized attribute type used by eglGetPlatformDisplay (EGLAttrib).
type EGLAttrib = isize;
/// Version of the vendor ABI implemented here (major 0).
const ABI_MAJOR: u32 = 0;

/// How native windows handed to `eglCreateWindowSurface` should be
/// interpreted. Recorded by `get_platform_display`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum NativePlatform {
    None,
    X11,
    Wayland,
}

static PLATFORM: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);
static NATIVE_DISPLAY: core::sync::atomic::AtomicPtr<c_void> =
    core::sync::atomic::AtomicPtr::new(core::ptr::null_mut());

pub fn current_native_platform() -> NativePlatform {
    match PLATFORM.load(core::sync::atomic::Ordering::Relaxed) {
        1 => NativePlatform::X11,
        2 => NativePlatform::Wayland,
        _ => NativePlatform::None,
    }
}

pub fn native_display_ptr() -> *mut c_void {
    NATIVE_DISPLAY.load(core::sync::atomic::Ordering::Relaxed)
}

pub(crate) fn set_platform(platform: EGLenum, native_display: *mut c_void) -> bool {
    let tag = match platform {
        // EGL_DEFAULT_DISPLAY / EGL_NONE: pick from the environment.
        0 | 0x3038 => match env_cstr(b"XDG_SESSION_TYPE") {
            Some(s) if s == b"wayland" => 2,
            Some(s) if s == b"x11" => 1,
            // X11 is the common fallback; SDL/Xlib apps expect it.
            _ => 1,
        },
        0x31D5 => 1, // EGL_PLATFORM_X11_KHR / EGL_PLATFORM_X11_EXT
        0x31D8 => 2, // EGL_PLATFORM_WAYLAND_KHR / EGL_PLATFORM_WAYLAND_EXT
        // Supported but without native-window import: rendering stays
        // headless/pbuffer.
        0x31DD | 0x31D6 | 0x313F => 0,
        _ => return false,
    };
    PLATFORM.store(tag, core::sync::atomic::Ordering::Relaxed);
    NATIVE_DISPLAY.store(native_display, core::sync::atomic::Ordering::Relaxed);
    true
}

fn env_cstr(name: &[u8]) -> Option<&'static [u8]> {
    unsafe {
        unsafe extern "C" {
            fn getenv(name: *const c_char) -> *const c_char;
        }
        let mut buf = name.to_vec();
        buf.push(0);
        let val = getenv(buf.as_ptr() as *const c_char);
        if val.is_null() {
            return None;
        }
        // Leaks a small allocation per lookup; called a handful of times at
        // most during initialization.
        let len = CStr::from_ptr(val).to_bytes().len();
        Some(core::slice::from_raw_parts(val as *const u8, len))
    }
}

// ---------------------------------------------------------------------------
// __EGLapiImports layout — must match libeglabi.h exactly.
// ---------------------------------------------------------------------------

#[repr(C)]
pub struct EglApiImports {
    pub get_platform_display: unsafe extern "C" fn(
        platform: EGLenum,
        native_display: *mut c_void,
        attrib_list: *const EGLAttrib,
    ) -> EGLDisplay,
    pub get_supports_api: unsafe extern "C" fn(api: EGLenum) -> EGLBoolean,
    pub get_vendor_string: Option<unsafe extern "C" fn(name: i32) -> *const c_char>,
    pub get_proc_address: unsafe extern "C" fn(proc_name: *const c_char) -> *mut c_void,
    pub get_dispatch_address: unsafe extern "C" fn(proc_name: *const c_char) -> *mut c_void,
    pub set_dispatch_index: unsafe extern "C" fn(proc_name: *const c_char, index: i32),
    pub is_patch_supported: Option<unsafe extern "C" fn(stub_type: i32, stub_size: i32) -> bool>,
    pub initiate_patch: Option<unsafe extern "C" fn()>,
    pub release_patch: Option<unsafe extern "C" fn()>,
    pub patch_thread_attach: Option<unsafe extern "C" fn()>,
}

/// `__egl_Main`: the single required export of a glvnd EGL vendor library.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __egl_Main(
    version: u32,
    exports: *const c_void,
    vendor: *mut c_void,
    imports: *mut EglApiImports,
) -> EGLBoolean {
    if (version >> 16) != ABI_MAJOR || imports.is_null() {
        return EGL_FALSE;
    }
    core::ptr::write(
        imports,
        EglApiImports {
            get_platform_display: vendor_get_platform_display,
            get_supports_api: vendor_get_supports_api,
            get_vendor_string: Some(vendor_get_vendor_string),
            get_proc_address: vendor_get_proc_address,
            get_dispatch_address: vendor_get_proc_address,
            set_dispatch_index: vendor_set_dispatch_index,
            is_patch_supported: None,
            initiate_patch: None,
            release_patch: None,
            patch_thread_attach: None,
        },
    );
    let _ = (exports, vendor);
    EGL_TRUE
}

unsafe extern "C" fn vendor_set_dispatch_index(_proc_name: *const c_char, _index: i32) {}

unsafe extern "C" fn vendor_get_platform_display(
    platform: EGLenum,
    native_display: *mut c_void,
    _attrib_list: *const EGLAttrib,
) -> EGLDisplay {
    if !set_platform(platform, native_display) {
        return core::ptr::null_mut();
    }
    let dpy = egl::get_or_create_display();
    Arc::into_raw(dpy) as EGLDisplay
}

unsafe extern "C" fn vendor_get_supports_api(api: EGLenum) -> EGLBoolean {
    match api {
        EGL_OPENGL_ES_API | EGL_OPENGL_API => EGL_TRUE,
        _ => EGL_FALSE,
    }
}

static VENDOR_STRING_PLATFORM_EXTENSIONS: &[u8] =
    b"EGL_EXT_platform_x11 EGL_EXT_platform_wayland EGL_MESA_platform_surfaceless\0";

unsafe extern "C" fn vendor_get_vendor_string(name: i32) -> *const c_char {
    match name {
        0 => VENDOR_STRING_PLATFORM_EXTENSIONS.as_ptr() as *const c_char,
        _ => core::ptr::null(),
    }
}

/// Name table served through both `getProcAddress` and `getDispatchAddress`.
/// glvnd requires every EGL 1.5 core entry point to be present here; stubs
/// cover operations this backend does not model.
unsafe extern "C" fn vendor_get_proc_address(proc_name: *const c_char) -> *mut c_void {
    if proc_name.is_null() {
        return core::ptr::null_mut();
    }
    let Ok(name) = CStr::from_ptr(proc_name).to_str() else {
        return core::ptr::null_mut();
    };

    match name {
        "eglInitialize" => egl::egl_initialize as *mut c_void,
        "eglTerminate" => egl::egl_terminate as *mut c_void,
        "eglChooseConfig" => egl::egl_choose_config as *mut c_void,
        "eglCopyBuffers" => egl_copy_buffers_stub as *mut c_void,
        "eglCreateContext" => egl::egl_create_context as *mut c_void,
        "eglCreatePbufferSurface" => egl::egl_create_pbuffer_surface as *mut c_void,
        "eglCreatePixmapSurface" => eglCreatePixmapSurface as *mut c_void,
        "eglCreateWindowSurface" => egl::egl_create_window_surface as *mut c_void,
        "eglDestroyContext" => egl::egl_destroy_context as *mut c_void,
        "eglDestroySurface" => egl::egl_destroy_surface as *mut c_void,
        "eglGetConfigAttrib" => egl::egl_get_config_attrib as *mut c_void,
        "eglGetConfigs" => egl::egl_get_configs as *mut c_void,
        "eglMakeCurrent" => egl::egl_make_current as *mut c_void,
        "eglQueryContext" => egl_query_context_stub as *mut c_void,
        "eglQueryString" => egl::egl_query_string as *mut c_void,
        "eglQuerySurface" => egl::egl_query_surface as *mut c_void,
        "eglSwapBuffers" => egl::egl_swap_buffers as *mut c_void,
        "eglWaitGL" => egl_wait_stub as *mut c_void,
        "eglWaitNative" => egl_wait_stub as *mut c_void,
        "eglWaitClient" => egl_wait_stub as *mut c_void,
        "eglReleaseThread" => egl_wait_stub as *mut c_void,
        "eglBindTexImage" => egl_boolean_stub as *mut c_void,
        "eglReleaseTexImage" => egl_boolean_stub as *mut c_void,
        "eglSurfaceAttrib" => egl_boolean_stub as *mut c_void,
        "eglCreatePbufferFromClientBuffer" => egl_no_surface_stub as *mut c_void,
        "eglSwapInterval" => egl::egl_swap_interval as *mut c_void,
        "eglGetError" => egl::egl_get_error as *mut c_void,
        "eglBindAPI" => eglBindAPI as *mut c_void,
        "eglGetCurrentContext" => egl::egl_get_current_context as *mut c_void,
        "eglGetCurrentDisplay" => egl::egl_get_current_display as *mut c_void,
        "eglGetCurrentSurface" => egl::egl_get_current_surface as *mut c_void,
        "eglGetDisplay" => egl::egl_get_display as *mut c_void,
        "eglGetProcAddress" => egl::egl_get_proc_address as *mut c_void,
        _ => crate::get_gl_proc_address(name)
            .map(|f| f as *mut c_void)
            .unwrap_or(core::ptr::null_mut()),
    }
}

// ---------------------------------------------------------------------------
// Stubs for entry points glvnd requires but this backend does not model.
// ---------------------------------------------------------------------------

unsafe extern "C" fn egl_boolean_stub(_dpy: EGLDisplay, _s: EGLSurface, _i: EGLint) -> EGLBoolean {
    EGL_TRUE
}
unsafe extern "C" fn egl_wait_stub() -> EGLBoolean {
    EGL_TRUE
}
unsafe extern "C" fn egl_copy_buffers_stub(
    _dpy: EGLDisplay,
    _surface: EGLSurface,
    _target: *mut c_void,
) -> EGLBoolean {
    egl::set_egl_error(EGL_BAD_MATCH);
    EGL_FALSE
}

unsafe extern "C" fn egl_query_context_stub(
    _dpy: EGLDisplay,
    _ctx: EGLContext,
    attribute: EGLint,
    value: *mut EGLint,
) -> EGLBoolean {
    if value.is_null() {
        return EGL_FALSE;
    }
    *value = match attribute {
        EGL_CONTEXT_CLIENT_VERSION => 1,
        _ => 0,
    };
    EGL_TRUE
}
unsafe extern "C" fn egl_no_surface_stub(
    _dpy: EGLDisplay,
    _buftype: EGLenum,
    _buffer: *mut c_void,
    _config: EGLConfig,
    _attrib_list: *const EGLint,
) -> EGLSurface {
    core::ptr::null_mut()
}
