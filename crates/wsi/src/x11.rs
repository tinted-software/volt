//! X11 MIT-SHM presentation backend for Vantage EGL.
//!
//! libX11/libXext are resolved at runtime through `libloading` (workspace
//! preference: no link-time X dependencies — the module compiles and links
//! on build hosts without X installed). Fallback path for window surfaces
//! when the DRI3 backend is unavailable.
#![allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code,
    unsafe_op_in_unsafe_fn
)]

use alloc::sync::Arc;
use core::ffi::{c_char, c_int, c_uint, c_ulong, c_void};

use libloading::Library;
use spin::Mutex;

pub type Display = c_void;
pub type Window = c_ulong;
pub type VisualID = c_ulong;
pub type GC = *mut c_void;
pub type Bool = c_int;
pub type Status = c_int;

pub const ZPixmap: c_int = 2;
pub const IPC_CREAT: c_int = 0o1000;
pub const IPC_RMID: c_int = 0;

#[repr(C)]
pub struct XShmSegmentInfo {
    pub shmseg: c_ulong,
    pub shmid: c_int,
    pub shmaddr: *mut c_char,
    pub readOnly: Bool,
}

#[repr(C)]
pub struct XImage {
    pub width: c_int,
    pub height: c_int,
    pub xoffset: c_int,
    pub format: c_int,
    pub data: *mut c_char,
    pub byte_order: c_int,
    pub bitmap_unit: c_int,
    pub bitmap_bit_order: c_int,
    pub bitmap_pad: c_int,
    pub depth: c_int,
    pub bytes_per_line: c_int,
    pub bits_per_pixel: c_int,
    pub red_mask: c_ulong,
    pub green_mask: c_ulong,
    pub blue_mask: c_ulong,
    pub obdata: *mut c_char,
    // function pointers omitted
}

// libX11 entry points
type FnXOpenDisplay = unsafe extern "C" fn(*const c_char) -> *mut Display;
type FnXCloseDisplay = unsafe extern "C" fn(*mut Display) -> c_int;
type FnXDefaultScreen = unsafe extern "C" fn(*mut Display) -> c_int;
type FnXDefaultVisual = unsafe extern "C" fn(*mut Display, c_int) -> *mut c_void;
type FnXDefaultDepth = unsafe extern "C" fn(*mut Display, c_int) -> c_int;
type FnXCreateGC = unsafe extern "C" fn(*mut Display, Window, c_ulong, *mut c_void) -> GC;
type FnXFreeGC = unsafe extern "C" fn(*mut Display, GC) -> c_int;
type FnXFlush = unsafe extern "C" fn(*mut Display) -> c_int;
type FnXSync = unsafe extern "C" fn(*mut Display, Bool) -> c_int;
type FnXDestroyImage = unsafe extern "C" fn(*mut XImage) -> c_int;
// libXext (MIT-SHM) entry points
type FnXShmQueryExtension = unsafe extern "C" fn(*mut Display) -> Bool;
type FnXShmCreateImage = unsafe extern "C" fn(
    *mut Display,
    *mut c_void,
    c_uint,
    c_int,
    *mut c_char,
    *mut XShmSegmentInfo,
    c_uint,
    c_uint,
) -> *mut XImage;
type FnXShmAttach = unsafe extern "C" fn(*mut Display, *mut XShmSegmentInfo) -> Status;
type FnXShmDetach = unsafe extern "C" fn(*mut Display, *mut XShmSegmentInfo) -> Status;
type FnXShmPutImage = unsafe extern "C" fn(
    *mut Display,
    Window,
    GC,
    *mut XImage,
    c_int,
    c_int,
    c_int,
    c_int,
    c_uint,
    c_uint,
    Bool,
) -> Status;

/// Runtime-resolved X11/SHM dispatch table.
struct X11Libs {
    _x11: Library,
    _xext: Library,
    XOpenDisplay: FnXOpenDisplay,
    XCloseDisplay: FnXCloseDisplay,
    XDefaultScreen: FnXDefaultScreen,
    XDefaultVisual: FnXDefaultVisual,
    XDefaultDepth: FnXDefaultDepth,
    XCreateGC: FnXCreateGC,
    XFreeGC: FnXFreeGC,
    XFlush: FnXFlush,
    XSync: FnXSync,
    XDestroyImage: FnXDestroyImage,
    XShmQueryExtension: FnXShmQueryExtension,
    XShmCreateImage: FnXShmCreateImage,
    XShmAttach: FnXShmAttach,
    XShmDetach: FnXShmDetach,
    XShmPutImage: FnXShmPutImage,
}

// Process-wide lazy table; `None` is sticky (fallback = no SHM surface).
static LIBS: Mutex<Option<Option<Arc<X11Libs>>>> = Mutex::new(None);

unsafe fn load_libs() -> Option<Arc<X11Libs>> {
    let x11 = Library::new("libX11.so.6").ok()?;
    let xext = Library::new("libXext.so.6").ok()?;
    macro_rules! sym {
        ($lib:expr, $name:literal, $ty:ty) => {{
            let s: libloading::Symbol<$ty> = $lib.get($name).ok()?;
            *s
        }};
    }
    // Resolve every symbol before the Library handles are moved into the table.
    let (
        XOpenDisplay,
        XCloseDisplay,
        XDefaultScreen,
        XDefaultVisual,
        XDefaultDepth,
        XCreateGC,
        XFreeGC,
        XFlush,
        XSync,
        XDestroyImage,
    ) = (
        sym!(x11, b"XOpenDisplay\0", FnXOpenDisplay),
        sym!(x11, b"XCloseDisplay\0", FnXCloseDisplay),
        sym!(x11, b"XDefaultScreen\0", FnXDefaultScreen),
        sym!(x11, b"XDefaultVisual\0", FnXDefaultVisual),
        sym!(x11, b"XDefaultDepth\0", FnXDefaultDepth),
        sym!(x11, b"XCreateGC\0", FnXCreateGC),
        sym!(x11, b"XFreeGC\0", FnXFreeGC),
        sym!(x11, b"XFlush\0", FnXFlush),
        sym!(x11, b"XSync\0", FnXSync),
        sym!(x11, b"XDestroyImage\0", FnXDestroyImage),
    );
    let (XShmQueryExtension, XShmCreateImage, XShmAttach, XShmDetach, XShmPutImage) = (
        sym!(xext, b"XShmQueryExtension\0", FnXShmQueryExtension),
        sym!(xext, b"XShmCreateImage\0", FnXShmCreateImage),
        sym!(xext, b"XShmAttach\0", FnXShmAttach),
        sym!(xext, b"XShmDetach\0", FnXShmDetach),
        sym!(xext, b"XShmPutImage\0", FnXShmPutImage),
    );
    Some(Arc::new(X11Libs {
        _x11: x11,
        _xext: xext,
        XOpenDisplay,
        XCloseDisplay,
        XDefaultScreen,
        XDefaultVisual,
        XDefaultDepth,
        XCreateGC,
        XFreeGC,
        XFlush,
        XSync,
        XDestroyImage,
        XShmQueryExtension,
        XShmCreateImage,
        XShmAttach,
        XShmDetach,
        XShmPutImage,
    }))
}

fn libs() -> Option<Arc<X11Libs>> {
    let mut slot = LIBS.lock();
    if slot.is_none() {
        *slot = Some(unsafe { load_libs() });
    }
    slot.as_ref().and_then(|o| o.clone())
}

pub struct X11ShmSurface {
    libs: Arc<X11Libs>,
    pub dpy: *mut Display,
    pub win: Window,
    pub gc: GC,
    pub ximage: *mut XImage,
    pub shminfo: alloc::boxed::Box<XShmSegmentInfo>,
    pub width: u32,
    pub height: u32,
    pub owns_display: bool,
}

unsafe impl Send for X11ShmSurface {}
unsafe impl Sync for X11ShmSurface {}

impl X11ShmSurface {
    /// # Safety
    /// `dpy` must be null or a valid open X11 `Display*`, and `win` must be a valid X11 window.
    pub unsafe fn new(
        mut dpy: *mut Display,
        win: Window,
        width: u32,
        height: u32,
    ) -> Result<Self, &'static str> {
        let libs = libs().ok_or("libX11/libXext unavailable")?;

        let owns_display = if dpy.is_null() {
            dpy = (libs.XOpenDisplay)(core::ptr::null());
            if dpy.is_null() {
                return Err("Failed to open X11 display");
            }
            true
        } else {
            false
        };

        if (libs.XShmQueryExtension)(dpy) == 0 {
            if owns_display {
                (libs.XCloseDisplay)(dpy);
            }
            return Err("X11 MIT-SHM extension unavailable");
        }

        let screen = (libs.XDefaultScreen)(dpy);
        let visual = (libs.XDefaultVisual)(dpy, screen);
        let depth = (libs.XDefaultDepth)(dpy, screen) as c_uint;

        let gc = (libs.XCreateGC)(dpy, win, 0, core::ptr::null_mut());
        if gc.is_null() {
            if owns_display {
                (libs.XCloseDisplay)(dpy);
            }
            return Err("XCreateGC failed");
        }

        let mut shminfo = alloc::boxed::Box::new(core::mem::zeroed::<XShmSegmentInfo>());
        let ximage = (libs.XShmCreateImage)(
            dpy,
            visual,
            depth,
            ZPixmap,
            core::ptr::null_mut(),
            shminfo.as_mut() as *mut XShmSegmentInfo,
            width.max(1),
            height.max(1),
        );

        if ximage.is_null() {
            (libs.XFreeGC)(dpy, gc);
            if owns_display {
                (libs.XCloseDisplay)(dpy);
            }
            return Err("XShmCreateImage failed");
        }

        let size = ((*ximage).bytes_per_line * (*ximage).height) as usize;
        shminfo.shmid = libc::shmget(libc::IPC_PRIVATE, size, IPC_CREAT | 0o777);
        if shminfo.shmid < 0 {
            (libs.XDestroyImage)(ximage);
            (libs.XFreeGC)(dpy, gc);
            if owns_display {
                (libs.XCloseDisplay)(dpy);
            }
            return Err("shmget failed");
        }

        shminfo.shmaddr = libc::shmat(shminfo.shmid, core::ptr::null(), 0) as *mut c_char;
        if shminfo.shmaddr == (-1isize) as *mut c_char {
            libc::shmctl(shminfo.shmid, IPC_RMID, core::ptr::null_mut());
            (libs.XDestroyImage)(ximage);
            (libs.XFreeGC)(dpy, gc);
            if owns_display {
                (libs.XCloseDisplay)(dpy);
            }
            return Err("shmat failed");
        }

        (*ximage).data = shminfo.shmaddr;
        shminfo.readOnly = 0;

        if (libs.XShmAttach)(dpy, shminfo.as_mut() as *mut XShmSegmentInfo) == 0 {
            libc::shmdt(shminfo.shmaddr as *mut c_void);
            libc::shmctl(shminfo.shmid, IPC_RMID, core::ptr::null_mut());
            (libs.XDestroyImage)(ximage);
            (libs.XFreeGC)(dpy, gc);
            if owns_display {
                (libs.XCloseDisplay)(dpy);
            }
            return Err("XShmAttach failed");
        }

        // Must sync with X server so it attaches the shared memory segment before we proceed
        (libs.XSync)(dpy, 0);

        Ok(Self {
            libs,
            dpy,
            win,
            gc,
            ximage,
            shminfo,
            width,
            height,
            owns_display,
        })
    }

    /// # Safety
    /// `src_rgba` must hold at least `self.height * src_stride` valid bytes for the configured surface.
    pub unsafe fn present(&mut self, src_rgba: &[u8], src_stride: usize) {
        let dst = (*self.ximage).data as *mut u8;
        let dst_stride = (*self.ximage).bytes_per_line as usize;
        let h = (self.height as usize).min((*self.ximage).height as usize);
        let w = (self.width as usize).min((*self.ximage).width as usize);

        // Framebuffer is already in native X11 TrueColor BGRA format
        let row_bytes = w * 4;
        for y in 0..h {
            let src_ptr = src_rgba.as_ptr().add(y * src_stride);
            let dst_ptr = dst.add(y * dst_stride);
            core::ptr::copy_nonoverlapping(src_ptr, dst_ptr, row_bytes);
        }

        (self.libs.XShmPutImage)(
            self.dpy,
            self.win,
            self.gc,
            self.ximage,
            0,
            0,
            0,
            0,
            self.width,
            self.height,
            0,
        );
        (self.libs.XFlush)(self.dpy);
    }
}

impl Drop for X11ShmSurface {
    fn drop(&mut self) {
        unsafe {
            (self.libs.XShmDetach)(self.dpy, self.shminfo.as_mut() as *mut XShmSegmentInfo);
            (self.libs.XSync)(self.dpy, 0);
            (self.libs.XDestroyImage)(self.ximage);
            libc::shmdt(self.shminfo.shmaddr as *mut c_void);
            libc::shmctl(self.shminfo.shmid, IPC_RMID, core::ptr::null_mut());
            (self.libs.XFreeGC)(self.dpy, self.gc);
            if self.owns_display {
                (self.libs.XCloseDisplay)(self.dpy);
            }
        }
    }
}
