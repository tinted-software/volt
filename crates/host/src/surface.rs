use core::ffi::{c_char, c_int, c_void};
use std::path::PathBuf;
use volt_platform::{PixelFormat, Surface};

/// Writes the most recently presented frame to disk as a binary PPM image.
pub struct DumpSurface {
    path: PathBuf,
    frames: u64,
}

impl DumpSurface {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            frames: 0,
        }
    }

    pub fn frames(&self) -> u64 {
        self.frames
    }
}

impl Surface for DumpSurface {
    fn present(
        &mut self,
        width: u32,
        height: u32,
        stride_bytes: u32,
        format: PixelFormat,
        pixels: &[u8],
    ) {
        use std::io::Write;
        if width == 0 || height == 0 {
            return;
        }
        let mut ppm = format!("P6\n{width} {height}\n255\n").into_bytes();
        ppm.reserve(width as usize * height as usize * 3);
        let stride = stride_bytes as usize;
        for y in 0..height as usize {
            let row_start = y * stride;
            for x in 0..width as usize {
                let px = row_start + x * 4;
                if px + 4 > pixels.len() {
                    ppm.extend_from_slice(&[0, 0, 0]);
                    continue;
                }
                let (r, g, b) = match format {
                    PixelFormat::B8G8R8A8Unorm | PixelFormat::B8G8R8X8Unorm => {
                        (pixels[px + 2], pixels[px + 1], pixels[px])
                    }
                    PixelFormat::R8G8B8A8Unorm | PixelFormat::R8G8B8X8Unorm => {
                        (pixels[px], pixels[px + 1], pixels[px + 2])
                    }
                };
                ppm.extend_from_slice(&[r, g, b]);
            }
        }
        if let Ok(mut file) = std::fs::File::create(&self.path) {
            let _ = file.write_all(&ppm);
            self.frames += 1;
        }
    }
}

/// Host desktop window presentation surface backed by dynamically loaded `SDL3`.
///
/// Loads `libSDL3` at runtime via `libloading` so binaries have no link-time
/// window-system dependency.
pub struct WindowSurface {
    _lib: libloading::Library,
    window: *mut c_void,
    renderer: *mut c_void,
    texture: *mut c_void,
    tex_w: u32,
    tex_h: u32,
    fn_create_texture: unsafe extern "C" fn(*mut c_void, u32, c_int, c_int, c_int) -> *mut c_void,
    fn_destroy_texture: unsafe extern "C" fn(*mut c_void),
    fn_update_texture:
        unsafe extern "C" fn(*mut c_void, *const c_void, *const c_void, c_int) -> bool,
    fn_render_clear: unsafe extern "C" fn(*mut c_void) -> bool,
    fn_render_texture:
        unsafe extern "C" fn(*mut c_void, *mut c_void, *const c_void, *const c_void) -> bool,
    fn_render_present: unsafe extern "C" fn(*mut c_void) -> bool,
    fn_pump_events: unsafe extern "C" fn(),
    fn_destroy_renderer: unsafe extern "C" fn(*mut c_void),
    fn_destroy_window: unsafe extern "C" fn(*mut c_void),
}

// SAFETY: `WindowSurface` is owned exclusively by the VirtIO-GPU device lock.
unsafe impl Send for WindowSurface {}

impl WindowSurface {
    pub fn open(title: &str, width: u32, height: u32) -> Result<Self, String> {
        #[cfg(target_os = "macos")]
        let lib_names = ["libSDL3.dylib", "libSDL3.0.dylib"];
        #[cfg(target_os = "linux")]
        let lib_names = ["libSDL3.so.0", "libSDL3.so"];
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        let lib_names = ["SDL3.dll"];

        let lib = lib_names
            .iter()
            .find_map(|&name| unsafe { libloading::Library::new(name) }.ok())
            .ok_or_else(|| String::from("libSDL3 not found in dynamic linker search path"))?;

        unsafe {
            let init: libloading::Symbol<unsafe extern "C" fn(u32) -> bool> =
                lib.get(b"SDL_Init\0").map_err(|e| e.to_string())?;
            if !init(0x0000_0020 /* SDL_INIT_VIDEO */) {
                return Err(String::from("SDL_Init(SDL_INIT_VIDEO) failed"));
            }
            let create_window_and_renderer: libloading::Symbol<
                unsafe extern "C" fn(
                    *const c_char,
                    c_int,
                    c_int,
                    u64,
                    *mut *mut c_void,
                    *mut *mut c_void,
                ) -> bool,
            > = lib
                .get(b"SDL_CreateWindowAndRenderer\0")
                .map_err(|e| e.to_string())?;

            let c_title = alloc::ffi::CString::new(title).unwrap_or_default();
            let mut window = core::ptr::null_mut();
            let mut renderer = core::ptr::null_mut();
            if !create_window_and_renderer(
                c_title.as_ptr(),
                width as c_int,
                height as c_int,
                0x20, /* SDL_WINDOW_RESIZABLE */
                &mut window,
                &mut renderer,
            ) {
                return Err(String::from("SDL_CreateWindowAndRenderer failed"));
            }

            let fn_create_texture = *lib
                .get::<unsafe extern "C" fn(*mut c_void, u32, c_int, c_int, c_int) -> *mut c_void>(
                    b"SDL_CreateTexture\0",
                )
                .map_err(|e| e.to_string())?;
            let fn_destroy_texture = *lib
                .get::<unsafe extern "C" fn(*mut c_void)>(b"SDL_DestroyTexture\0")
                .map_err(|e| e.to_string())?;
            let fn_update_texture = *lib
                .get::<unsafe extern "C" fn(*mut c_void, *const c_void, *const c_void, c_int) -> bool>(
                    b"SDL_UpdateTexture\0",
                )
                .map_err(|e| e.to_string())?;
            let fn_render_clear = *lib
                .get::<unsafe extern "C" fn(*mut c_void) -> bool>(b"SDL_RenderClear\0")
                .map_err(|e| e.to_string())?;
            let fn_render_texture =
                *lib.get::<unsafe extern "C" fn(
                    *mut c_void,
                    *mut c_void,
                    *const c_void,
                    *const c_void,
                ) -> bool>(b"SDL_RenderTexture\0")
                    .map_err(|e| e.to_string())?;
            let fn_render_present = *lib
                .get::<unsafe extern "C" fn(*mut c_void) -> bool>(b"SDL_RenderPresent\0")
                .map_err(|e| e.to_string())?;
            let fn_pump_events = *lib
                .get::<unsafe extern "C" fn()>(b"SDL_PumpEvents\0")
                .map_err(|e| e.to_string())?;
            let fn_destroy_renderer = *lib
                .get::<unsafe extern "C" fn(*mut c_void)>(b"SDL_DestroyRenderer\0")
                .map_err(|e| e.to_string())?;
            let fn_destroy_window = *lib
                .get::<unsafe extern "C" fn(*mut c_void)>(b"SDL_DestroyWindow\0")
                .map_err(|e| e.to_string())?;

            Ok(Self {
                _lib: lib,
                window,
                renderer,
                texture: core::ptr::null_mut(),
                tex_w: 0,
                tex_h: 0,
                fn_create_texture,
                fn_destroy_texture,
                fn_update_texture,
                fn_render_clear,
                fn_render_texture,
                fn_render_present,
                fn_pump_events,
                fn_destroy_renderer,
                fn_destroy_window,
            })
        }
    }
}

impl Drop for WindowSurface {
    fn drop(&mut self) {
        unsafe {
            if !self.texture.is_null() {
                (self.fn_destroy_texture)(self.texture);
            }
            if !self.renderer.is_null() {
                (self.fn_destroy_renderer)(self.renderer);
            }
            if !self.window.is_null() {
                (self.fn_destroy_window)(self.window);
            }
        }
    }
}

impl Surface for WindowSurface {
    fn present(
        &mut self,
        width: u32,
        height: u32,
        stride_bytes: u32,
        format: PixelFormat,
        pixels: &[u8],
    ) {
        if width == 0 || height == 0 {
            return;
        }
        // SDL_PIXELFORMAT_ARGB8888 (little-endian BGRA8888) vs SDL_PIXELFORMAT_ABGR8888 (RGBA8888)
        const SDL_PIXELFORMAT_ARGB8888: u32 = 0x1636_2004;
        const SDL_PIXELFORMAT_ABGR8888: u32 = 0x1676_2004;
        const SDL_TEXTUREACCESS_STREAMING: c_int = 1;
        let sdl_fmt = match format {
            PixelFormat::B8G8R8A8Unorm | PixelFormat::B8G8R8X8Unorm => SDL_PIXELFORMAT_ARGB8888,
            PixelFormat::R8G8B8A8Unorm | PixelFormat::R8G8B8X8Unorm => SDL_PIXELFORMAT_ABGR8888,
        };
        unsafe {
            if self.texture.is_null() || self.tex_w != width || self.tex_h != height {
                if !self.texture.is_null() {
                    (self.fn_destroy_texture)(self.texture);
                }
                self.texture = (self.fn_create_texture)(
                    self.renderer,
                    sdl_fmt,
                    SDL_TEXTUREACCESS_STREAMING,
                    width as c_int,
                    height as c_int,
                );
                self.tex_w = width;
                self.tex_h = height;
            }
            if !self.texture.is_null() {
                (self.fn_update_texture)(
                    self.texture,
                    core::ptr::null(),
                    pixels.as_ptr().cast(),
                    stride_bytes as c_int,
                );
                (self.fn_render_clear)(self.renderer);
                (self.fn_render_texture)(
                    self.renderer,
                    self.texture,
                    core::ptr::null(),
                    core::ptr::null(),
                );
                (self.fn_render_present)(self.renderer);
            }
            (self.fn_pump_events)();
        }
    }
}
