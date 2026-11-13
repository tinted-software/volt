//! EGL 1.4 API implementation and context/surface management.
#![allow(unused_imports, dead_code)]
use crate::glvnd::{NativePlatform, current_native_platform, native_display_ptr, set_platform};
use alloc::collections::BTreeMap as HashMap;
use alloc::sync::Arc;
use core::ffi::{CStr, c_void};
use core::sync::atomic::{AtomicU32, Ordering};
use volt_gles::display_list::DisplayListRegistry;
use volt_gles::gl_context::{CURRENT_CONTEXT, GlContext, set_current_gl_context};
use volt_gles::sync::Mutex;
use volt_gles::texture::TextureManager;
use volt_gles::types::*;
#[cfg(all(feature = "std", target_os = "linux"))]
use volt_wsi::x11::X11ShmSurface;
// every surface-creation path returns EGL_BAD_ALLOC.

pub struct EglSurfaceState {
    pub id: u32,
    pub width: u32,
    pub height: u32,
    pub native_window: NativeWindowType,
    pub swap_interval: EGLint,
    pub hal_color_image: Option<volt_hal::ImageId>,
    /// Next-frame zero-copy target: HAL image aliasing the acquired DRI3
    /// GEM slot mapping (dri3 path only).
    #[cfg(all(feature = "std", target_os = "linux"))]
    pub hal_ext_color_image: Option<volt_hal::ImageId>,
    /// Device-owned fallback color image, presented via copy when the
    /// zero-copy ring has no slot available.
    #[cfg(all(feature = "std", target_os = "linux"))]
    pub hal_fallback_color_image: Option<volt_hal::ImageId>,
    pub hal_depth_image: Option<volt_hal::ImageId>,
    #[cfg(all(feature = "std", target_os = "linux"))]
    pub x11_surface: Option<X11ShmSurface>,
    /// DRI3/Present zero-copy path; preferred over SHM when available.
    #[cfg(all(feature = "std", target_os = "linux"))]
    pub dri3_surface: Option<volt_wsi::dri3::Dri3Surface>,
    #[cfg(not(target_os = "linux"))]
    pub sw_surface: volt_wsi::sw::SoftwareSurface,
}

pub struct EglContextState {
    pub id: u32,
    pub gl_context: Arc<Mutex<GlContext>>,
}

pub struct EglDisplayState {
    pub initialized: bool,
    pub surfaces: HashMap<u32, Arc<Mutex<EglSurfaceState>>>,
    pub contexts: HashMap<u32, Arc<Mutex<EglContextState>>>,
    pub shared_display_lists: Arc<DisplayListRegistry>,
    next_surface_id: AtomicU32,
    next_context_id: AtomicU32,
}

impl Default for EglDisplayState {
    fn default() -> Self {
        Self {
            initialized: false,
            surfaces: HashMap::new(),
            contexts: HashMap::new(),
            shared_display_lists: Arc::new(DisplayListRegistry::new()),
            next_surface_id: AtomicU32::new(1),
            next_context_id: AtomicU32::new(1),
        }
    }
}
impl EglDisplayState {
    pub fn allocate_surface_id(&self) -> u32 {
        self.next_surface_id.fetch_add(1, Ordering::SeqCst)
    }

    pub fn allocate_context_id(&self) -> u32 {
        self.next_context_id.fetch_add(1, Ordering::SeqCst)
    }
}

unsafe impl Send for EglSurfaceState {}
unsafe impl Sync for EglSurfaceState {}
unsafe impl Send for EglContextState {}
unsafe impl Sync for EglContextState {}
unsafe impl Send for EglDisplayState {}
unsafe impl Sync for EglDisplayState {}

// Global display singleton
static GLOBAL_DISPLAY: Mutex<Option<Arc<Mutex<EglDisplayState>>>> = Mutex::new(None);
static LAST_EGL_ERROR: Mutex<EGLint> = Mutex::new(EGL_SUCCESS);

// Current-context registry lives in vantage-gles (`gl_context`): the GL
// entry points there resolve "the" context through `get_current_gl_context`,
// with an implicit fall-back to the last context made current on any thread
// (see that module's notes on console-style shared-context semantics).

pub static CURRENT_SURFACE: Mutex<Option<Arc<Mutex<EglSurfaceState>>>> = Mutex::new(None);

pub fn set_egl_error(err: EGLint) {
    *LAST_EGL_ERROR.lock() = err;
}

pub fn get_or_create_display() -> Arc<Mutex<EglDisplayState>> {
    let mut gd = GLOBAL_DISPLAY.lock();
    if gd.is_none() {
        *gd = Some(Arc::new(Mutex::new(EglDisplayState::default())));
    }
    gd.as_ref().unwrap().clone()
}

// ============================================================================
// EGL C API
// ============================================================================

pub unsafe fn egl_get_display(display_id: NativeDisplayType) -> EGLDisplay {
    // Direct-link clients (no glvnd) pass the native display handle here:
    // an X11 `Display*`/Wayland `wl_display*`, or EGL_DEFAULT_DISPLAY (null).
    // Record it so window-surface creation can present to the right window.
    let native: *mut c_void = display_id;
    if !native.is_null() || current_native_platform() == NativePlatform::None {
        // Platform 0 lets set_platform pick from the environment.
        set_platform(0, native);
    }
    let dpy = get_or_create_display();
    Arc::into_raw(dpy) as EGLDisplay
}

pub unsafe fn egl_initialize(
    dpy: EGLDisplay,
    major: *mut EGLint,
    minor: *mut EGLint,
) -> EGLBoolean {
    if dpy.is_null() {
        set_egl_error(EGL_BAD_DISPLAY);
        return EGL_FALSE;
    }

    let dpy_arc = Arc::from_raw(dpy as *const Mutex<EglDisplayState>);
    let res = {
        let mut d = dpy_arc.lock();
        d.initialized = true;
        if !major.is_null() {
            *major = 1;
        }
        if !minor.is_null() {
            *minor = 4;
        }
        EGL_TRUE
    };
    core::mem::forget(dpy_arc);
    res
}

pub unsafe fn egl_terminate(dpy: EGLDisplay) -> EGLBoolean {
    if dpy.is_null() {
        set_egl_error(EGL_BAD_DISPLAY);
        return EGL_FALSE;
    }

    let dpy_arc = Arc::from_raw(dpy as *const Mutex<EglDisplayState>);
    {
        let mut d = dpy_arc.lock();
        d.initialized = false;
        d.contexts.clear();
        d.surfaces.clear();
    }
    core::mem::forget(dpy_arc);
    EGL_TRUE
}

pub unsafe fn egl_get_configs(
    _dpy: EGLDisplay,
    configs: *mut EGLConfig,
    config_size: EGLint,
    num_config: *mut EGLint,
) -> EGLBoolean {
    if !num_config.is_null() {
        *num_config = 1;
    }
    if !configs.is_null() && config_size > 0 {
        *configs = 1 as EGLConfig;
    }
    EGL_TRUE
}

pub unsafe fn egl_choose_config(
    _dpy: EGLDisplay,
    attrib_list: *const EGLint,
    configs: *mut EGLConfig,
    config_size: EGLint,
    num_config: *mut EGLint,
) -> EGLBoolean {
    if num_config.is_null() {
        set_egl_error(EGL_BAD_PARAMETER);
        return EGL_FALSE;
    }

    // The single exported config. Values must stay consistent with
    // `egl_get_config_attrib`, which is the query side of the same contract.
    const CONFIG_ID: EGLint = 1;
    const SURFACE_TYPE_VAL: EGLint = EGL_WINDOW_BIT | EGL_PBUFFER_BIT;
    const RENDERABLE_TYPE_VAL: EGLint = EGL_OPENGL_ES_BIT | EGL_OPENGL_ES2_BIT;
    // Match kind: 0 = exact, 1 = at-least (config value >= requested),
    // 2 = bitmask (every requested bit set in the config's value).
    const CONFIG_ATTRIBS: &[(EGLint, EGLint, u8)] = &[
        (EGL_BUFFER_SIZE, 32, 1),
        (EGL_RED_SIZE, 8, 1),
        (EGL_GREEN_SIZE, 8, 1),
        (EGL_BLUE_SIZE, 8, 1),
        (EGL_ALPHA_SIZE, 8, 1),
        (EGL_DEPTH_SIZE, 24, 1),
        (EGL_STENCIL_SIZE, 8, 1),
        (EGL_SAMPLES, 0, 1),
        (EGL_SAMPLE_BUFFERS, 0, 1),
        (EGL_LUMINANCE_SIZE, 0, 1),
        (EGL_ALPHA_MASK_SIZE, 0, 1),
        (EGL_MAX_PBUFFER_WIDTH, 16384, 1),
        (EGL_MAX_PBUFFER_HEIGHT, 16384, 1),
        (EGL_MAX_PBUFFER_PIXELS, 16384 * 16384, 1),
        (EGL_LEVEL, 0, 0),
        (EGL_CONFIG_CAVEAT, EGL_NONE, 0),
        (EGL_CONFIG_ID, CONFIG_ID, 0),
        (EGL_COLOR_BUFFER_TYPE, EGL_RGB_BUFFER, 0),
        (EGL_NATIVE_RENDERABLE, EGL_FALSE as EGLint, 0),
        (EGL_NATIVE_VISUAL_ID, 0, 0),
        (EGL_NATIVE_VISUAL_TYPE, 0, 0),
        (EGL_BIND_TO_TEXTURE_RGB, EGL_TRUE as EGLint, 0),
        (EGL_BIND_TO_TEXTURE_RGBA, EGL_TRUE as EGLint, 0),
        (EGL_SURFACE_TYPE, SURFACE_TYPE_VAL, 2),
        (EGL_RENDERABLE_TYPE, RENDERABLE_TYPE_VAL, 2),
        (EGL_CONFORMANT, RENDERABLE_TYPE_VAL, 2),
    ];

    if !attrib_list.is_null() {
        let mut ptr = attrib_list;
        while *ptr != EGL_NONE {
            let attr = *ptr;
            let want = *ptr.add(1);
            ptr = ptr.add(2);
            if attr == EGL_DONT_CARE || want == EGL_DONT_CARE {
                continue;
            }
            let Some(&(_, have, kind)) = CONFIG_ATTRIBS.iter().find(|&&(a, _, _)| a == attr) else {
                set_egl_error(EGL_BAD_ATTRIBUTE);
                *num_config = 0;
                return EGL_FALSE;
            };
            let ok = match kind {
                1 => have >= want,
                2 => (have & want) == want,
                _ => have == want,
            };
            if !ok {
                *num_config = 0;
                return EGL_TRUE;
            }
        }
    }

    *num_config = 1;
    if !configs.is_null() && config_size > 0 {
        *configs = CONFIG_ID as EGLConfig;
    }
    EGL_TRUE
}
static STRING_VENDOR: &[u8] = b"Vantage\0";
static STRING_VERSION: &[u8] = b"1.4 Vantage EGL 1.4\0";
static STRING_CLIENT_APIS: &[u8] = b"OpenGL_ES \0";
static STRING_EXTENSIONS: &[u8] = b"EGL_KHR_create_context EGL_KHR_surfaceless_context \
EGL_KHR_image_base EGL_KHR_fence_sync EGL_KHR_wait_sync EGL_KHR_get_all_proc_addresses\0";
static CLIENT_EXTENSIONS: &[u8] = b"EGL_EXT_platform_base EGL_EXT_platform_x11 \
EGL_EXT_platform_wayland EGL_MESA_platform_surfaceless \
EGL_KHR_client_get_all_proc_addresses\0";

/// Returns a static NUL-terminated string, or null for unknown queries.
pub unsafe fn egl_query_string(dpy: EGLDisplay, name: EGLint) -> *const core::ffi::c_char {
    let bytes: &[u8] = match name {
        EGL_VENDOR => STRING_VENDOR,
        EGL_VERSION => STRING_VERSION,
        EGL_CLIENT_APIS => STRING_CLIENT_APIS,
        // Client extensions are display-independent.
        EGL_EXTENSIONS if dpy.is_null() => CLIENT_EXTENSIONS,
        EGL_EXTENSIONS => STRING_EXTENSIONS,
        _ => return core::ptr::null(),
    };
    bytes.as_ptr() as *const core::ffi::c_char
}

/// MISSING: std-gated only — freestanding builds report no visual.
#[cfg(not(all(feature = "std", target_os = "linux")))]
fn x11_default_visual_id() -> EGLint {
    0
}

pub unsafe fn egl_get_config_attrib(
    _dpy: EGLDisplay,
    _config: EGLConfig,
    attribute: EGLint,
    value: *mut EGLint,
) -> EGLBoolean {
    if value.is_null() {
        return EGL_FALSE;
    }

    match attribute {
        EGL_BUFFER_SIZE => *value = 32,
        EGL_RED_SIZE => *value = 8,
        EGL_GREEN_SIZE => *value = 8,
        EGL_BLUE_SIZE => *value = 8,
        EGL_ALPHA_SIZE => *value = 8,
        EGL_DEPTH_SIZE => *value = 24,
        EGL_STENCIL_SIZE => *value = 8,
        EGL_SURFACE_TYPE => *value = EGL_WINDOW_BIT | EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE => *value = EGL_OPENGL_ES_BIT | EGL_OPENGL_ES2_BIT,
        EGL_NATIVE_VISUAL_ID => *value = x11_default_visual_id(),
        _ => *value = 0,
    }
    EGL_TRUE
}

/// Best-effort XVisualID of the default visual for the current native X11
/// display, resolved through libX11 at runtime (no hard dependency).
/// Returns 0 when unavailable or not running on X11.
#[cfg(all(feature = "std", target_os = "linux"))]
fn x11_default_visual_id() -> EGLint {
    if current_native_platform() != NativePlatform::X11 {
        return 0;
    }
    let dpy = native_display_ptr();
    if dpy.is_null() {
        return 0;
    }

    const RTLD_LAZY: i32 = 1;
    let lib = unsafe { libloading::Library::new("libX11.so.6") };
    let lib = match lib {
        Ok(l) => l,
        Err(_) => return 0,
    };

    type XDefaultScreenFn = unsafe extern "C" fn(*mut c_void) -> i32;
    type XDefaultVisualFn = unsafe extern "C" fn(*mut c_void, i32) -> *mut c_void;

    unsafe {
        let screen: libloading::Symbol<XDefaultScreenFn> = match lib.get(b"XDefaultScreen\0") {
            Ok(s) => s,
            Err(_) => return 0,
        };
        let visual_fn: libloading::Symbol<XDefaultVisualFn> = match lib.get(b"XDefaultVisual\0") {
            Ok(s) => s,
            Err(_) => return 0,
        };
        let vis = visual_fn(dpy, screen(dpy));
        if vis.is_null() {
            return 0;
        }
        // struct Visual { XExtData *ext_data; VisualID visualid; ... }
        // visualid is the member right after the ext_data pointer.
        *(vis as *const usize).add(1) as EGLint
    }
}
pub unsafe fn egl_create_window_surface(
    dpy: EGLDisplay,
    _config: EGLConfig,
    win: NativeWindowType,
    attrib_list: *const EGLint,
) -> EGLSurface {
    let mut width = 1280;
    let mut height = 720;

    if !attrib_list.is_null() {
        let mut ptr = attrib_list;
        while *ptr != EGL_NONE {
            let attr = *ptr;
            let val = *ptr.add(1);
            if attr == EGL_WIDTH {
                width = val.max(1) as u32;
            } else if attr == EGL_HEIGHT {
                height = val.max(1) as u32;
            }
            ptr = ptr.add(2);
        }
    }

    let dpy_arc = if !dpy.is_null() {
        Arc::from_raw(dpy as *const Mutex<EglDisplayState>)
    } else {
        get_or_create_display()
    };

    let id = {
        let d = dpy_arc.lock();
        d.next_surface_id.fetch_add(1, Ordering::SeqCst)
    };

    #[cfg(all(feature = "std", target_os = "linux"))]
    let dri3_surface = if win != 0 {
        // DRI3 first: zero-copy GEM buffer sharing + Present flips.
        volt_wsi::dri3::Dri3Surface::new(win as u32, width, height)
    } else {
        None
    };

    #[cfg(all(feature = "std", target_os = "linux"))]
    let x11_surface = if win != 0 && dri3_surface.is_none() {
        let dpy_ptr = native_display_ptr() as *mut volt_wsi::x11::Display;
        match unsafe { X11ShmSurface::new(dpy_ptr, win as u64, width, height) } {
            Ok(s) => Some(s),
            Err(_) => {
                set_egl_error(EGL_BAD_ALLOC);
                return EGL_NO_SURFACE;
            }
        }
    } else {
        None
    };

    let surface_state = Arc::new(Mutex::new(EglSurfaceState {
        id,
        width,
        height,
        native_window: win,
        swap_interval: 0,
        hal_color_image: None,
        #[cfg(all(feature = "std", target_os = "linux"))]
        hal_ext_color_image: None,
        #[cfg(all(feature = "std", target_os = "linux"))]
        hal_fallback_color_image: None,
        hal_depth_image: None,
        #[cfg(all(feature = "std", target_os = "linux"))]
        x11_surface,
        #[cfg(all(feature = "std", target_os = "linux"))]
        dri3_surface,
        #[cfg(not(target_os = "linux"))]
        sw_surface: volt_wsi::sw::SoftwareSurface::new(width, height),
    }));

    {
        let mut d = dpy_arc.lock();
        d.surfaces.insert(id, surface_state.clone());
    }

    if !dpy.is_null() {
        core::mem::forget(dpy_arc);
    }

    Arc::into_raw(surface_state) as EGLSurface
}

pub unsafe fn egl_create_native_window_surface(
    dpy: EGLDisplay,
    native: *const AngleWgpuNativeWindow,
) -> EGLSurface {
    if native.is_null() {
        set_egl_error(EGL_BAD_NATIVE_WINDOW);
        return EGL_NO_SURFACE;
    }
    let n = *native;
    if n.kind != ANGLE_WGPU_NATIVE_X11
        && n.kind != ANGLE_WGPU_NATIVE_WAYLAND
        && n.kind != ANGLE_WGPU_NATIVE_WIN32
    {
        set_egl_error(EGL_BAD_PARAMETER);
        return EGL_NO_SURFACE;
    }

    let width = if n.width == 0 { 800 } else { n.width }.max(1);
    let height = if n.height == 0 { 600 } else { n.height }.max(1);

    #[cfg(all(feature = "std", any(target_os = "linux", target_os = "freebsd")))]
    let dri3_surface = if n.kind == ANGLE_WGPU_NATIVE_X11 {
        volt_wsi::dri3::Dri3Surface::new(n.window as u32, width, height)
    } else {
        None
    };

    #[cfg(all(feature = "std", any(target_os = "linux", target_os = "freebsd")))]
    let x11_surface = if n.kind == ANGLE_WGPU_NATIVE_X11 && dri3_surface.is_none() {
        let dpy_ptr = n.display as *mut volt_wsi::x11::Display;
        match unsafe { X11ShmSurface::new(dpy_ptr, n.window, width, height) } {
            Ok(s) => Some(s),
            Err(_) => {
                set_egl_error(EGL_BAD_ALLOC);
                return EGL_NO_SURFACE;
            }
        }
    } else {
        None
    };

    let dpy_arc = if !dpy.is_null() {
        Arc::from_raw(dpy as *const Mutex<EglDisplayState>)
    } else {
        get_or_create_display()
    };

    let id = {
        let d = dpy_arc.lock();
        d.next_surface_id.fetch_add(1, Ordering::SeqCst)
    };

    let surface_state = Arc::new(Mutex::new(EglSurfaceState {
        id,
        width,
        height,
        native_window: n.window as NativeWindowType,
        swap_interval: 0,
        hal_color_image: None,
        #[cfg(all(feature = "std", target_os = "linux"))]
        hal_ext_color_image: None,
        #[cfg(all(feature = "std", target_os = "linux"))]
        hal_fallback_color_image: None,
        hal_depth_image: None,
        #[cfg(all(feature = "std", target_os = "linux"))]
        x11_surface,
        #[cfg(all(feature = "std", target_os = "linux"))]
        dri3_surface,
        #[cfg(not(target_os = "linux"))]
        sw_surface: volt_wsi::sw::SoftwareSurface::new(width, height),
    }));

    {
        let mut d = dpy_arc.lock();
        d.surfaces.insert(id, surface_state.clone());
    }

    if !dpy.is_null() {
        core::mem::forget(dpy_arc);
    }

    Arc::into_raw(surface_state) as EGLSurface
}

pub unsafe fn egl_create_pbuffer_surface(
    dpy: EGLDisplay,
    _config: EGLConfig,
    attrib_list: *const EGLint,
) -> EGLSurface {
    egl_create_window_surface(dpy, _config, core::mem::zeroed(), attrib_list)
}

pub unsafe fn egl_destroy_surface(dpy: EGLDisplay, surface: EGLSurface) -> EGLBoolean {
    if surface.is_null() {
        return EGL_TRUE;
    }
    let surf_arc = Arc::from_raw(surface as *const Mutex<EglSurfaceState>);
    let id = surf_arc.lock().id;

    if !dpy.is_null() {
        let dpy_arc = Arc::from_raw(dpy as *const Mutex<EglDisplayState>);
        dpy_arc.lock().surfaces.remove(&id);
        core::mem::forget(dpy_arc);
    }

    EGL_TRUE
}

pub unsafe fn egl_create_context(
    dpy: EGLDisplay,
    _config: EGLConfig,
    _share_context: EGLContext,
    _attrib_list: *const EGLint,
) -> EGLContext {
    let dpy_arc = if !dpy.is_null() {
        Arc::from_raw(dpy as *const Mutex<EglDisplayState>)
    } else {
        get_or_create_display()
    };

    let (id, shared_lists) = {
        let d = dpy_arc.lock();
        (
            d.next_context_id.fetch_add(1, Ordering::SeqCst),
            d.shared_display_lists.clone(),
        )
    };

    let gl_ctx = Arc::new(Mutex::new(GlContext::new(
        id,
        TextureManager::new(),
        shared_lists,
    )));
    let ctx_state = Arc::new(Mutex::new(EglContextState {
        id,
        gl_context: gl_ctx,
    }));

    {
        let mut d = dpy_arc.lock();
        d.contexts.insert(id, ctx_state.clone());
    }

    if !dpy.is_null() {
        core::mem::forget(dpy_arc);
    }

    Arc::into_raw(ctx_state) as EGLContext
}

pub unsafe fn egl_destroy_context(dpy: EGLDisplay, ctx: EGLContext) -> EGLBoolean {
    if ctx.is_null() {
        return EGL_TRUE;
    }
    let ctx_arc = Arc::from_raw(ctx as *const Mutex<EglContextState>);
    let id = ctx_arc.lock().id;

    if !dpy.is_null() {
        let dpy_arc = Arc::from_raw(dpy as *const Mutex<EglDisplayState>);
        dpy_arc.lock().contexts.remove(&id);
        core::mem::forget(dpy_arc);
    }

    EGL_TRUE
}

pub unsafe fn egl_make_current(
    _dpy: EGLDisplay,
    draw: EGLSurface,
    _read: EGLSurface,
    ctx: EGLContext,
) -> EGLBoolean {
    if ctx.is_null() {
        set_current_gl_context(None);
        *CURRENT_SURFACE.lock() = None;
        return EGL_TRUE;
    }

    let ctx_arc = Arc::from_raw(ctx as *const Mutex<EglContextState>);
    let gl_ctx = ctx_arc.lock().gl_context.clone();
    core::mem::forget(ctx_arc);

    if !draw.is_null() {
        let surf_arc = Arc::from_raw(draw as *const Mutex<EglSurfaceState>);
        let (w, h) = {
            let s = surf_arc.lock();
            (s.width, s.height)
        };
        core::mem::forget(surf_arc.clone());

        {
            let mut gl = gl_ctx.lock();
            gl.viewport = (0, 0, w as i32, h as i32);
            gl.scissor = (0, 0, w as i32, h as i32);

            let mut surf = surf_arc.lock();
            if surf.hal_color_image.is_none() {
                let c_img = gl
                    .hal_device
                    .create_image(volt_hal::Format::B8G8R8A8Unorm, w, h);
                let d_img = gl
                    .hal_device
                    .create_image(volt_hal::Format::D32Sfloat, w, h);
                surf.hal_color_image = Some(c_img);
                #[cfg(all(feature = "std", target_os = "linux"))]
                {
                    surf.hal_fallback_color_image = Some(c_img);
                }
                surf.hal_depth_image = Some(d_img);
            }
            let c_img = surf.hal_color_image;
            let d_img = surf.hal_depth_image;
            gl.color_image = c_img;
            gl.depth_image = d_img;

            gl.command_buffer.push(volt_hal::Cmd::BindAttachments {
                color: c_img,
                depth: d_img,
                stencil: None,
            });
        }

        *CURRENT_SURFACE.lock() = Some(surf_arc);
    }

    set_current_gl_context(Some(gl_ctx));
    EGL_TRUE
}

pub unsafe fn egl_get_current_context() -> EGLContext {
    // EGLContext handles are Arc<Mutex<EglContextState>> pointers (see
    // egl_create_context); the GL registry stores the inner GlContext, whose
    // address is a different object. Look the owning EGL context up in the
    // display's context table.
    let cur = CURRENT_CONTEXT.lock();
    let Some(gl_arc) = cur.as_ref().cloned() else {
        return EGL_NO_CONTEXT;
    };
    drop(cur);
    let dpy_arc = get_or_create_display();
    let d = dpy_arc.lock();
    for state in d.contexts.values() {
        if Arc::ptr_eq(&state.lock().gl_context, &gl_arc) {
            return Arc::into_raw(state.clone()) as EGLContext;
        }
    }
    EGL_NO_CONTEXT
}

pub unsafe fn egl_get_current_surface(_readdraw: EGLint) -> EGLSurface {
    let cur = CURRENT_SURFACE.lock();
    match cur.as_ref() {
        Some(surf) => Arc::as_ptr(surf) as EGLSurface,
        None => EGL_NO_SURFACE,
    }
}

pub unsafe fn egl_get_current_display() -> EGLDisplay {
    let dpy = get_or_create_display();
    Arc::as_ptr(&dpy) as EGLDisplay
}

pub unsafe fn egl_query_surface(
    _dpy: EGLDisplay,
    surface: EGLSurface,
    attribute: EGLint,
    value: *mut EGLint,
) -> EGLBoolean {
    if surface.is_null() || value.is_null() {
        return EGL_FALSE;
    }

    let surf_arc = Arc::from_raw(surface as *const Mutex<EglSurfaceState>);
    {
        let s = surf_arc.lock();
        match attribute {
            EGL_WIDTH => *value = s.width as EGLint,
            EGL_HEIGHT => *value = s.height as EGLint,
            _ => *value = 0,
        }
    }
    core::mem::forget(surf_arc);
    EGL_TRUE
}

pub unsafe fn egl_swap_buffers(_dpy: EGLDisplay, surface: EGLSurface) -> EGLBoolean {
    if surface.is_null() {
        return EGL_FALSE;
    }

    let surf_arc = Arc::from_raw(surface as *const Mutex<EglSurfaceState>);
    if let Some(ctx_arc) = volt_gles::gl_context::get_current_gl_context() {
        let mut ctx = ctx_arc.lock();
        let cmd = core::mem::take(&mut ctx.command_buffer);
        ctx.hal_device.submit(&cmd);

        // Re-bind attachments for subsequent frame's draw calls
        let c_img = ctx.color_image;
        let d_img = ctx.depth_image;
        ctx.command_buffer.push(volt_hal::Cmd::BindAttachments {
            color: c_img,
            depth: d_img,
            stencil: None,
        });

        #[cfg(all(feature = "std", target_os = "linux"))]
        {
            let mut surf_guard = surf_arc.lock();
            let s = &mut *surf_guard;
            let interval = s.swap_interval;
            let width = s.width;
            let height = s.height;
            let fallback_cid = s.hal_fallback_color_image.or(s.hal_color_image);
            let ext_cid = s.hal_ext_color_image;
            let depth_cid = s.hal_depth_image;
            if let Some(ref mut dri3) = s.dri3_surface {
                dri3.swap_interval = interval;
                // 1. Present the frame just rendered. Zero-copy if a GEM
                // slot was bound as the render target; fallback copy otherwise.
                let ok = if dri3.render_slot.is_some() {
                    dri3.present_slot()
                } else if let Some(c_id) = fallback_cid {
                    match ctx.hal_device.image(c_id) {
                        Some(img) => {
                            let stride = img.row_pitch as usize;
                            let data = img.slice();
                            dri3.present(data, stride)
                        }
                        None => true,
                    }
                } else {
                    true
                };
                if !ok {
                    set_egl_error(EGL_BAD_ACCESS);
                    return EGL_FALSE;
                }

                // 2. Pre-acquire next frame's GEM slot: render directly
                // into it so next swap requires zero framebuffer copy.
                if let Some((_slot, ptr, pitch)) = dri3.acquire() {
                    let ext_id = ext_cid.unwrap_or_else(|| unsafe {
                        ctx.hal_device.create_image_external(
                            volt_hal::Format::B8G8R8A8Unorm,
                            width,
                            height,
                            ptr,
                            pitch * height as usize,
                            pitch as u32,
                        )
                    });
                    unsafe {
                        ctx.hal_device.set_image_external(
                            ext_id,
                            ptr,
                            pitch * height as usize,
                            pitch as u32,
                        );
                    }
                    s.hal_color_image = Some(ext_id);
                    s.hal_ext_color_image = Some(ext_id);
                    ctx.command_buffer.push(volt_hal::Cmd::BindAttachments {
                        color: Some(ext_id),
                        depth: depth_cid,
                        stencil: None,
                    });
                } else {
                    s.hal_color_image = s.hal_fallback_color_image;
                    ctx.command_buffer.push(volt_hal::Cmd::BindAttachments {
                        color: s.hal_fallback_color_image,
                        depth: depth_cid,
                        stencil: None,
                    });
                }
            } else if let Some(ref mut x11) = s.x11_surface
                && let Some(c_id) = fallback_cid
                && let Some(img) = ctx.hal_device.image(c_id)
            {
                unsafe {
                    x11.present(img.slice(), img.row_pitch as usize);
                }
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            use volt_platform::{PixelFormat, Surface};
            let mut surf_guard = surf_arc.lock();
            if let Some(c_id) = surf_guard.hal_color_image
                && let Some(img) = ctx.hal_device.image(c_id)
            {
                let w = surf_guard.width;
                let h = surf_guard.height;
                let pitch = img.row_pitch;
                let fmt = match img.format {
                    volt_hal::Format::B8G8R8A8Unorm => PixelFormat::B8G8R8A8Unorm,
                    _ => PixelFormat::R8G8B8A8Unorm,
                };
                surf_guard.sw_surface.present(w, h, pitch, fmt, img.slice());
            }
        }
    }
    core::mem::forget(surf_arc);
    EGL_TRUE
}

pub unsafe fn egl_resize_surface(surface: EGLSurface, width: u32, height: u32) -> EGLBoolean {
    if surface.is_null() {
        return EGL_FALSE;
    }
    let w = width.max(1);
    let h = height.max(1);
    let surf_arc = Arc::from_raw(surface as *const Mutex<EglSurfaceState>);
    {
        let mut s = surf_arc.lock();
        if s.width != w || s.height != h {
            s.width = w;
            s.height = h;
            s.hal_color_image = None;
            #[cfg(all(feature = "std", target_os = "linux"))]
            {
                s.hal_ext_color_image = None;
                s.hal_fallback_color_image = None;
            }
            s.hal_depth_image = None;

            #[cfg(all(feature = "std", target_os = "linux"))]
            if let Some(ref mut dri3) = s.dri3_surface
                && !dri3.resize(w, h)
            {
                set_egl_error(EGL_BAD_ALLOC);
                return EGL_FALSE;
            }

            #[cfg(all(feature = "std", target_os = "linux"))]
            if let Some(ref mut x11) = s.x11_surface {
                let dpy = x11.dpy;
                let win = x11.win;
                if let Ok(new_x11) = X11ShmSurface::new(dpy, win, w, h) {
                    *x11 = new_x11;
                }
            }
        }
    }

    let is_current = CURRENT_SURFACE
        .lock()
        .as_ref()
        .map(|s| Arc::ptr_eq(s, &surf_arc))
        .unwrap_or(false);
    if is_current && let Some(ctx) = CURRENT_CONTEXT.lock().as_ref() {
        let mut gl = ctx.lock();
        gl.viewport = (0, 0, w as i32, h as i32);
        gl.scissor = (0, 0, w as i32, h as i32);

        let mut surf = surf_arc.lock();
        if surf.hal_color_image.is_none() {
            let c_img = gl
                .hal_device
                .create_image(volt_hal::Format::B8G8R8A8Unorm, w, h);
            let d_img = gl
                .hal_device
                .create_image(volt_hal::Format::D32Sfloat, w, h);
            surf.hal_color_image = Some(c_img);
            #[cfg(all(feature = "std", target_os = "linux"))]
            {
                surf.hal_fallback_color_image = Some(c_img);
            }
            surf.hal_depth_image = Some(d_img);
        }
        let c_img = surf.hal_color_image;
        let d_img = surf.hal_depth_image;
        gl.color_image = c_img;
        gl.depth_image = d_img;

        gl.command_buffer.push(volt_hal::Cmd::BindAttachments {
            color: c_img,
            depth: d_img,
            stencil: None,
        });
    }

    core::mem::forget(surf_arc);
    EGL_TRUE
}

pub unsafe fn egl_swap_interval(_dpy: EGLDisplay, interval: EGLint) -> EGLBoolean {
    if interval != 0 && interval != 1 {
        set_egl_error(EGL_BAD_PARAMETER);
        return EGL_FALSE;
    }

    let Some(surface) = CURRENT_SURFACE.lock().clone() else {
        set_egl_error(EGL_BAD_SURFACE);
        return EGL_FALSE;
    };
    // MISSING: presentation (Phase 3) — honor the interval with XSync at
    // present time.
    surface.lock().swap_interval = interval;
    EGL_TRUE
}

pub unsafe fn egl_get_error() -> EGLint {
    let mut err = LAST_EGL_ERROR.lock();
    let res = *err;
    *err = EGL_SUCCESS;
    res
}

pub unsafe fn egl_get_proc_address(
    procname: *const core::ffi::c_char,
) -> __eglMustCastToProperFunctionPointerType {
    if procname.is_null() {
        return None;
    }
    let Ok(name) = CStr::from_ptr(procname).to_str() else {
        return None;
    };

    crate::get_gl_proc_address(name)
}
