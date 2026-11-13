//! DRI3 + Present X11 presentation backend for Vantage EGL.
//!
//! Replaces the MIT-SHM copy path with true buffer sharing: frames are
//! rendered by the (pliron-driven) rasterizer into DRM dumb GEM buffers,
//! exported as dma-bufs via PRIME, turned into X pixmaps with
//! `DRI3:PixmapFromBuffer`, and flipped with `Present:Pixmap`. Buffer reuse
//! is gated on `Present:IdleNotify`/`CompleteNotify` — the same
//! triple-buffered ring Mesa's dri3 front-end maintains, minus GPU command
//! submission (the CPU raster path supplies frames; a future amdgpu CS path
//! attaches kernel sync fds via `DRI3:FenceFromFd`).
//!
//! libxcb is loaded at runtime through `libloading` (workspace preference —
//! no link-time xcb dependency): `libxcb.so.1` for connection/request/event
//! entry points, `libxcb-dri3.so.0` / `libxcb-present.so.0` solely for their
//! `xcb_dri3_id` / `xcb_present_id` extension-registry data symbols, which
//! `xcb_send_request*` requires in `xcb_protocol_request_t.ext`.
//!
//! Wire opcodes/structs pinned to this machine's `xcb/dri3.h` + `xcb/present.h`
//! (DRI3 1.4: QueryVersion=0, Open=1, PixmapFromBuffer=2; Present:
//! QueryVersion=0, Pixmap=1, SelectInput=3; Present events:
//! ConfigureNotify=0, CompleteNotify=1, IdleNotify=2). A 1.0-era server also
//! matches opcodes 0-2/0/1/3.
//!
//! A dedicated xcb connection is used ($DISPLAY) instead of sharing the
//! application's Xlib display: Present notify events must be consumed by the
//! driver, not injected into the app's event queue.
#![allow(
    non_snake_case,
    non_camel_case_types,
    dead_code,
    unsafe_op_in_unsafe_fn
)]

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ffi::{c_char, c_int, c_uint, c_void};
use spin::Mutex;

use libloading::Library;

use volt_drm::{DrmDevice, GemBuffer};

// ------------------------------------------------------------------------
// libxcb surface (dlsym'd)
// ------------------------------------------------------------------------

type FnConnect = unsafe extern "C" fn(*const c_char, *mut c_int) -> *mut c_void;
type FnHasError = unsafe extern "C" fn(*mut c_void) -> c_int;
type FnFlush = unsafe extern "C" fn(*mut c_void) -> c_int;
type FnGenerateId = unsafe extern "C" fn(*mut c_void) -> c_uint;
type FnGetExtensionData = unsafe extern "C" fn(*mut c_void, *const c_void) -> *const u8;
type FnSendRequest64 =
    unsafe extern "C" fn(*mut c_void, c_int, *mut Iovec, *const ProtocolRequest) -> u64;
type FnSendRequestWithFds64 = unsafe extern "C" fn(
    *mut c_void,
    c_int,
    *mut Iovec,
    *const ProtocolRequest,
    c_uint,
    *const c_int,
) -> u64;
type FnWaitForReply64 = unsafe extern "C" fn(*mut c_void, u64, *mut *mut c_void) -> *mut c_void;
/// `xcb_void_cookie_t` (struct { unsigned sequence }) arrives by value.
type FnRequestCheck = unsafe extern "C" fn(*mut c_void, VoidCookie) -> *mut c_void;
type FnPollForEvent = unsafe extern "C" fn(*mut c_void) -> *mut u8;
type FnDisconnect = unsafe extern "C" fn(*mut c_void);

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Iovec {
    base: *mut c_void,
    len: usize,
}

#[repr(C)]
struct ProtocolRequest {
    count: usize,
    ext: *const c_void, // xcb_extension_t*
    opcode: u8,
    isvoid: u8,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VoidCookie {
    sequence: c_uint,
}

const XCB_REQUEST_CHECKED: c_int = 1 << 0;

unsafe extern "C" {
    fn free(ptr: *mut c_void);
    fn dup(fd: c_int) -> c_int;
    fn close(fd: c_int) -> c_int;
}

// ------------------------------------------------------------------------
// X extension wire constants
// ------------------------------------------------------------------------

// DRI3 minor opcodes (xcb/dri3.h @ this system)
const DRI3_QUERY_VERSION: u8 = 0;
const DRI3_PIXMAP_FROM_BUFFER: u8 = 2;
// Present (xcb/present.h)
const PRESENT_QUERY_VERSION: u8 = 0;
const PRESENT_PIXMAP: u8 = 1;
const PRESENT_SELECT_INPUT: u8 = 3;
const PRESENT_EVENT_COMPLETE_NOTIFY: u16 = 1;
const PRESENT_EVENT_IDLE_NOTIFY: u16 = 2;
const PRESENT_EVENT_COMPLETE_MASK: u32 = 2;
const PRESENT_EVENT_IDLE_MASK: u32 = 4;
const PRESENT_OPTION_ASYNC: u32 = 1;
const PRESENT_COMPLETE_KIND_PIXMAP: u8 = 0;
// X protocol core
const XCB_CORE_GET_GEOMETRY: u8 = 10;
const XCB_CORE_FREE_PIXMAP: u8 = 13;
const XCB_GENERIC_EVENT: u8 = 35;

// ------------------------------------------------------------------------
// Request structs (layouts pinned against installed headers)
// ------------------------------------------------------------------------

macro_rules! assert_size {
    ($ty:ty, $n:expr) => {
        const _: () = assert!(core::mem::size_of::<$ty>() == $n);
    };
}

#[repr(C)]
struct ReqHdr {
    major_opcode: u8,
    minor_opcode: u8,
    length: u16, // 4-byte units, header included
}

#[repr(C)]
struct QueryVersionReq {
    hdr: ReqHdr,
    major_version: u32,
    minor_version: u32,
}
assert_size!(QueryVersionReq, 12);

#[repr(C)]
struct PixmapFromBufferReq {
    hdr: ReqHdr,
    pixmap: u32,
    drawable: u32,
    size: u32,
    width: u16,
    height: u16,
    stride: u16,
    depth: u8,
    bpp: u8,
}
assert_size!(PixmapFromBufferReq, 24);

#[repr(C)]
struct PresentPixmapReq {
    hdr: ReqHdr,
    window: u32,
    pixmap: u32,
    serial: u32,
    valid: u32,
    update: u32,
    x_off: i16,
    y_off: i16,
    target_crtc: u32,
    wait_fence: u32,
    idle_fence: u32,
    options: u32,
    pad0: u32,
    target_msc: u64,
    divisor: u64,
    remainder: u64,
}
assert_size!(PresentPixmapReq, 72);

#[repr(C)]
struct PresentSelectInputReq {
    hdr: ReqHdr,
    eid: u32,
    window: u32,
    event_mask: u32,
}
assert_size!(PresentSelectInputReq, 16);

#[repr(C)]
struct GetGeometryReq {
    hdr: ReqHdr,
    drawable: u32,
}
assert_size!(GetGeometryReq, 8);

#[repr(C)]
struct FreePixmapReq {
    hdr: ReqHdr,
    pixmap: u32,
}
assert_size!(FreePixmapReq, 8);

// ------------------------------------------------------------------------
// Connection singleton
// ------------------------------------------------------------------------

struct Conn {
    lib: Library,          // core libxcb (must outlive all fn ptrs)
    _dri3_lib: Library,    // for xcb_dri3_id
    _present_lib: Library, // for xcb_present_id
    conn: *mut c_void,
    dri3_ext: *const c_void,
    present_ext: *const c_void,
    dri3_opcode: u8,
    present_opcode: u8,

    connect: FnConnect,
    has_error: FnHasError,
    flush: FnFlush,
    generate_id: FnGenerateId,
    get_extension_data: FnGetExtensionData,
    send: FnSendRequest64,
    send_fds: FnSendRequestWithFds64,
    wait_reply: FnWaitForReply64,
    request_check: FnRequestCheck,
    poll_event: FnPollForEvent,
    #[allow(dead_code)]
    disconnect: FnDisconnect,

    /// Present events parked by one surface's pump for another surface.
    parked: Mutex<Vec<Vec<u8>>>,
}

unsafe impl Send for Conn {}
unsafe impl Sync for Conn {}

impl Conn {
    /// Send a fully-formed request buffer. `vecs` is caller-owned and
    /// lives across the call; indices -1/-2 are reserved by libxcb.
    unsafe fn send_buf(
        &self,
        vecs: &mut [Iovec; 3],
        buf: &mut [u8],
        ext: *const c_void,
        opcode: u8,
        checked: bool,
        fds: &[c_int],
    ) -> u64 {
        vecs[2] = Iovec {
            base: buf.as_mut_ptr() as *mut c_void,
            len: buf.len(),
        };
        let req = ProtocolRequest {
            count: 1,
            ext,
            opcode,
            isvoid: 1,
        };
        let flags = if checked { XCB_REQUEST_CHECKED } else { 0 };
        let iov = vecs.as_mut_ptr().add(2);
        if fds.is_empty() {
            (self.send)(self.conn, flags, iov, &req)
        } else {
            (self.send_fds)(
                self.conn,
                flags,
                iov,
                &req,
                fds.len() as c_uint,
                fds.as_ptr(),
            )
        }
    }

    unsafe fn send_req<T>(&self, req: &mut T, ext: *const c_void, opcode: u8) -> u64 {
        let mut vecs = [Iovec::default(); 3];
        let buf = bytes_of(req);
        self.send_buf(&mut vecs, buf, ext, opcode, true, &[])
    }

    /// Checked reply request: errors return via `wait_reply`, never the
    /// event queue. Returns the owned reply bytes.
    unsafe fn reply_req<T>(&self, req: &mut T, ext: *const c_void, opcode: u8) -> Option<Vec<u8>> {
        let seq = self.send_req(req, ext, opcode);
        let mut err: *mut c_void = core::ptr::null_mut();
        let reply = (self.wait_reply)(self.conn, seq, &mut err);
        if reply.is_null() {
            if !err.is_null() {
                free(err);
            }
            return None;
        }
        // Reply header: type(1) pad(1) sequence(2) extra-length(4, units of
        // 4 bytes beyond the fixed 32-byte reply).
        let extra = *(reply as *const u32).add(1) as usize;
        let total = 32 + extra * 4;
        let bytes = Vec::from(core::slice::from_raw_parts(reply as *const u8, total));
        free(reply);
        Some(bytes)
    }

    fn check_void(&self, seq: u64) -> Option<()> {
        unsafe {
            let cookie = VoidCookie {
                sequence: seq as c_uint,
            };
            let e = (self.request_check)(self.conn, cookie);
            if e.is_null() {
                Some(())
            } else {
                free(e);
                None
            }
        }
    }

    unsafe fn void_req<T>(&self, req: &mut T, ext: *const c_void, opcode: u8) -> Option<()> {
        let seq = self.send_req(req, ext, opcode);
        self.check_void(seq)
    }
}

fn bytes_of<T>(v: &mut T) -> &mut [u8] {
    unsafe { core::slice::from_raw_parts_mut(v as *mut T as *mut u8, core::mem::size_of::<T>()) }
}

fn read_u32_at(buf: &[u8], off: usize) -> Option<u32> {
    buf.get(off..off + 4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn read_u16_at(buf: &[u8], off: usize) -> Option<u16> {
    buf.get(off..off + 2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
}

static CONN: Mutex<Option<Option<Arc<Conn>>>> = Mutex::new(None);

/// Process-wide connection, created on first use. `None` is sticky so
/// every later surface falls back to MIT-SHM without retry cost.
fn conn() -> Option<Arc<Conn>> {
    // Debug knob: force the MIT-SHM presentation fallback (mirrors Mesa's
    // driver env overrides) for isolating DRI3 vs rasterizer issues.
    let set = unsafe { !libc::getenv(c"VANTAGE_DISABLE_DRI3".as_ptr()).is_null() };
    if set {
        return None;
    }
    let mut slot = CONN.lock();
    if slot.is_none() {
        *slot = Some(unsafe { establish() });
    }
    slot.as_ref().and_then(|o| o.clone())
}

unsafe fn establish() -> Option<Arc<Conn>> {
    let core = Library::new("libxcb.so.1").ok()?;
    let dri3_lib = Library::new("libxcb-dri3.so.0").ok()?;
    let present_lib = Library::new("libxcb-present.so.0").ok()?;

    macro_rules! sym {
        ($lib:expr, $name:literal, $ty:ty) => {{
            let s: libloading::Symbol<$ty> = $lib.get($name).ok()?;
            *s
        }};
        ($lib:expr, $name:literal => data) => {{
            let s: libloading::Symbol<*const c_void> = $lib.get($name).ok()?;
            *s as *const c_void
        }};
    }

    let connect: FnConnect = sym!(core, b"xcb_connect\0", FnConnect);
    let has_error: FnHasError = sym!(core, b"xcb_connection_has_error\0", FnHasError);
    let flush: FnFlush = sym!(core, b"xcb_flush\0", FnFlush);
    let generate_id: FnGenerateId = sym!(core, b"xcb_generate_id\0", FnGenerateId);
    let get_extension_data: FnGetExtensionData =
        sym!(core, b"xcb_get_extension_data\0", FnGetExtensionData);
    let send: FnSendRequest64 = sym!(core, b"xcb_send_request64\0", FnSendRequest64);
    let send_fds: FnSendRequestWithFds64 = sym!(
        core,
        b"xcb_send_request_with_fds64\0",
        FnSendRequestWithFds64
    );
    let wait_reply: FnWaitForReply64 = sym!(core, b"xcb_wait_for_reply64\0", FnWaitForReply64);
    let request_check: FnRequestCheck = sym!(core, b"xcb_request_check\0", FnRequestCheck);
    let poll_event: FnPollForEvent = sym!(core, b"xcb_poll_for_event\0", FnPollForEvent);
    let disconnect: FnDisconnect = sym!(core, b"xcb_disconnect\0", FnDisconnect);
    let dri3_ext: *const c_void = sym!(dri3_lib, b"xcb_dri3_id\0" => data);
    let present_ext: *const c_void = sym!(present_lib, b"xcb_present_id\0" => data);

    // NULL display name: libxcb resolves $DISPLAY + Xauthority itself.
    let conn = (connect)(core::ptr::null(), core::ptr::null_mut());
    if conn.is_null() || (has_error)(conn) != 0 {
        return None;
    }

    let major_of = |ext: *const c_void| -> Option<u8> {
        let reply = (get_extension_data)(conn, ext);
        if reply.is_null() {
            return None;
        }
        // xcb_query_extension_reply_t: present_flag @8, major_opcode @12.
        let present = *reply.add(8);
        let major = *(reply.add(12) as *const u16);
        if present == 0 || major == 0 {
            None
        } else {
            Some(major as u8)
        }
    };
    let dri3_opcode = major_of(dri3_ext)?;
    let present_opcode = major_of(present_ext)?;

    let c = Conn {
        lib: core,
        _dri3_lib: dri3_lib,
        _present_lib: present_lib,
        conn,
        dri3_ext,
        present_ext,
        dri3_opcode,
        present_opcode,
        connect,
        has_error,
        flush,
        generate_id,
        get_extension_data,
        send,
        send_fds,
        wait_reply,
        request_check,
        poll_event,
        disconnect,
        parked: Mutex::new(Vec::new()),
    };

    // Negotiate versions (fail closed: both extensions must answer; the
    // server's major must be >= 1 to speak this protocol at all).
    let mut rq = QueryVersionReq {
        hdr: ReqHdr {
            major_opcode: dri3_opcode,
            minor_opcode: DRI3_QUERY_VERSION,
            length: 3,
        },
        major_version: 1,
        minor_version: 2,
    };
    let reply = c.reply_req(&mut rq, dri3_ext, dri3_opcode)?;
    if read_u32_at(&reply, 8)? < 1 {
        return None;
    }
    let mut rq = QueryVersionReq {
        hdr: ReqHdr {
            major_opcode: present_opcode,
            minor_opcode: PRESENT_QUERY_VERSION,
            length: 3,
        },
        major_version: 1,
        minor_version: 0,
    };
    let reply = c.reply_req(&mut rq, present_ext, present_opcode)?;
    if read_u32_at(&reply, 8)? < 1 {
        return None;
    }
    (c.flush)(c.conn);
    Some(Arc::new(c))
}

// ------------------------------------------------------------------------
// Surface
// ------------------------------------------------------------------------

struct Slot {
    bo: GemBuffer,
    /// X pixmap id created from the bo's dmabuf, 0 until first present.
    pixmap: u32,
    /// Slot handed to the server (present in flight) with its serial.
    busy_serial: Option<u32>,
}

/// A DRI3/Present-backed window surface: ring of GEM buffers, each with a
/// lazily created shared pixmap, flipped via `Present:Pixmap`.
pub struct Dri3Surface {
    c: Arc<Conn>,
    dev: Arc<DrmDevice>,
    win: u32,
    pub width: u32,
    pub height: u32,
    pub swap_interval: i32,
    depth: u8,
    serial: u32,
    slots: Vec<Slot>,
    /// Present event id (XID) selected on the window.
    eid: u32,
    /// Slot the application is currently rendering into via an external
    /// (zero-copy) image; presented by [`Dri3Surface::present_slot`].
    pub render_slot: Option<usize>,
}

unsafe impl Send for Dri3Surface {}
unsafe impl Sync for Dri3Surface {}

/// One DRM device per process for the DRI3 path (single-GPU target).
static DEV: Mutex<Option<Arc<DrmDevice>>> = Mutex::new(None);

fn drm_device() -> Option<Arc<DrmDevice>> {
    let mut slot = DEV.lock();
    if slot.is_none() {
        *slot = DrmDevice::open_best().ok().map(Arc::new);
    }
    slot.clone()
}

impl Dri3Surface {
    /// # Safety
    /// `win` must be a live X11 Window id on `$DISPLAY`.
    pub unsafe fn new(win: u32, width: u32, height: u32) -> Option<Self> {
        let c = conn()?;
        if width == 0 || height == 0 {
            return None;
        }
        let dev = drm_device()?;

        // Validate the window and learn its depth: the shared pixmap must
        // carry the window's own depth (24 on classic servers, 32 under
        // compositing), and a dead window fails here.
        let mut g = GetGeometryReq {
            hdr: ReqHdr {
                major_opcode: XCB_CORE_GET_GEOMETRY,
                minor_opcode: 0,
                length: 2,
            },
            drawable: win,
        };
        let reply = c.reply_req(&mut g, core::ptr::null(), XCB_CORE_GET_GEOMETRY)?;
        let depth = *reply.get(8)?;

        let mut slots = Vec::new();
        for _ in 0..3 {
            let bo = dev.create_dumb(width, height, 32).ok()?;
            slots.push(Slot {
                bo,
                pixmap: 0,
                busy_serial: None,
            });
        }

        let eid = (c.generate_id)(c.conn);
        let mut sel = PresentSelectInputReq {
            hdr: ReqHdr {
                major_opcode: c.present_opcode,
                minor_opcode: PRESENT_SELECT_INPUT,
                length: 4,
            },
            eid,
            window: win,
            event_mask: PRESENT_EVENT_COMPLETE_MASK | PRESENT_EVENT_IDLE_MASK,
        };
        c.void_req(&mut sel, c.present_ext, c.present_opcode)?;
        (c.flush)(c.conn);

        Some(Self {
            c,
            dev,
            win,
            width,
            height,
            swap_interval: 1,
            depth,
            serial: 0,
            slots,
            eid,
            render_slot: None,
        })
    }

    /// Copy the rendered frame into a free bo and flip. `src` is the hal
    /// color image (B8G8R8A8 linear, `src_stride` bytes/row). Returns
    /// false on unrecoverable failure — caller should degrade to SHM.
    pub fn present(&mut self, src: &[u8], src_stride: usize) -> bool {
        // Find a free slot, releasing finished ones from the event stream.
        let idx = match self.free_slot() {
            Some(i) => i,
            None => {
                if !self.pump_wait_for_slot() {
                    return false;
                }
                match self.free_slot() {
                    Some(i) => i,
                    None => return false,
                }
            }
        };

        // Copy rendered frame into the shared buffer (row-aware: the bo
        // pitch may exceed the image stride).
        {
            let slot = &mut self.slots[idx];
            let pitch = slot.bo.pitch as usize;
            let dst = slot.bo.as_mut_slice();
            let w4 = (self.width as usize) * 4;
            let h = core::cmp::min(self.height as usize, dst.len() / pitch.max(1));
            for y in 0..h {
                let s = y * src_stride;
                let d = y * pitch;
                if s + w4 > src.len() || d + w4 > dst.len() {
                    break;
                }
                dst[d..d + w4].copy_from_slice(&src[s..s + w4]);
            }
        }

        // Flip. update=None: the server treats it as the full pixmap
        // extents (hw/xfree86 present_pixmap); options gates vblank wait.
        self.present_pixmap(idx)
    }

    /// Send `Present:Pixmap` for slot `idx` and mark it busy until the
    /// server's IdleNotify. Shared by the copy path and the zero-copy path.
    fn present_pixmap(&mut self, idx: usize) -> bool {
        // Lazily import the bo into X as a pixmap (dma-buf fd transfer).
        if self.slots[idx].pixmap == 0 && !unsafe { self.import_pixmap(idx) } {
            return false;
        }
        self.serial = self.serial.wrapping_add(1).max(1);
        let serial = self.serial;
        let options = if self.swap_interval == 0 {
            PRESENT_OPTION_ASYNC
        } else {
            0
        };
        let mut req = PresentPixmapReq {
            hdr: ReqHdr {
                major_opcode: self.c.present_opcode,
                minor_opcode: PRESENT_PIXMAP,
                length: 18,
            },
            window: self.win,
            pixmap: self.slots[idx].pixmap,
            serial,
            valid: 0,
            update: 0,
            x_off: 0,
            y_off: 0,
            target_crtc: 0,
            wait_fence: 0,
            idle_fence: 0,
            options,
            pad0: 0,
            target_msc: 0,
            divisor: 0,
            remainder: 0,
        };
        if unsafe {
            self.c
                .void_req(&mut req, self.c.present_ext, self.c.present_opcode)
        }
        .is_none()
        {
            return false;
        }
        unsafe { (self.c.flush)(self.c.conn) };
        self.slots[idx].busy_serial = Some(serial);
        true
    }

    /// Zero-copy path, step 1 (called at swap for the *next* frame): pick a
    /// free GEM slot and hand its mapped memory to the caller so the
    /// rasterizer can render directly into it. Blocks (draining Present
    /// events) only when all three slots are still in flight — the same
    /// throttling the copy path applies.
    ///
    /// Returns `(slot, map_ptr, pitch)`. The pointer stays valid until the
    /// slot is presented or the surface is resized/dropped.
    pub fn acquire(&mut self) -> Option<(usize, *mut u8, usize)> {
        if !self.pump_events() {
            return None;
        }
        let idx = match self.free_slot() {
            Some(i) => i,
            None => {
                if !self.pump_wait_for_slot() {
                    return None;
                }
                self.free_slot()?
            }
        };
        self.render_slot = Some(idx);
        let slot = &mut self.slots[idx];
        let mem = slot.bo.as_mut_slice();
        Some((idx, mem.as_mut_ptr(), slot.bo.pitch as usize))
    }

    /// Zero-copy path, step 2 (called at the following swap): flip the slot
    /// that was rendered into. No framebuffer copy occurs.
    pub fn present_slot(&mut self) -> bool {
        let Some(idx) = self.render_slot.take() else {
            return false;
        };
        self.present_pixmap(idx)
    }

    unsafe fn import_pixmap(&mut self, idx: usize) -> bool {
        let slot = &self.slots[idx];
        let pixmap = (self.c.generate_id)(self.c.conn);
        let fd = match self.dev.prime_handle_to_fd(slot.bo.handle) {
            Ok(fd) => fd,
            Err(_) => return false,
        };
        let sent_fd = dup(fd);
        close(fd);
        if sent_fd < 0 {
            return false;
        }
        let mut req = PixmapFromBufferReq {
            hdr: ReqHdr {
                major_opcode: self.c.dri3_opcode,
                minor_opcode: DRI3_PIXMAP_FROM_BUFFER,
                length: 6,
            },
            pixmap,
            drawable: self.win,
            size: slot.bo.size as u32,
            width: self.width as u16,
            height: self.height as u16,
            stride: slot.bo.pitch as u16,
            depth: self.depth,
            bpp: 32,
        };
        let mut vecs = [Iovec::default(); 3];
        let buf = bytes_of(&mut req);
        let seq = self.c.send_buf(
            &mut vecs,
            buf,
            self.c.dri3_ext,
            self.c.dri3_opcode,
            true,
            &[sent_fd],
        );
        if self.c.check_void(seq).is_none() {
            // On error the fd was not consumed by the server.
            close(sent_fd);
            return false;
        }
        // Ownership of sent_fd transfers to the server per DRI3/libxcb.
        self.slots[idx].pixmap = pixmap;
        true
    }

    fn free_slot(&mut self) -> Option<usize> {
        // Prefer already-imported (pixmap cached) slots, then fresh ones. The
        // slot currently bound as the render target is never handed out.
        self.slots
            .iter()
            .position(|s| s.busy_serial.is_none() && s.pixmap != 0)
            .filter(|&i| Some(i) != self.render_slot)
            .or_else(|| {
                self.slots
                    .iter()
                    .position(|s| s.busy_serial.is_none())
                    .filter(|&i| Some(i) != self.render_slot)
            })
    }

    /// Drain X events, releasing slots the server finished with. Returns
    /// false on connection error.
    fn pump_events(&mut self) -> bool {
        // First re-examine events parked by other surfaces.
        let mine: Vec<Vec<u8>> = {
            let mut p = self.c.parked.lock();
            let src = core::mem::take(&mut *p);
            let mut took = Vec::new();
            for ev in src {
                if self.matches_window(&ev) {
                    took.push(ev);
                } else {
                    p.push(ev);
                }
            }
            took
        };
        for ev in &mine {
            self.handle_event(ev);
        }
        for _ in 0..256 {
            let ev = unsafe { (self.c.poll_event)(self.c.conn) };
            if ev.is_null() {
                return true;
            }
            let response_type = unsafe { *ev } & 0x7f;
            if response_type == XCB_GENERIC_EVENT {
                // GenericEvent: extension @1, length @4 (extra 4-byte
                // units beyond the fixed 32).
                let extra = unsafe { *(ev.add(4) as *const u32) } as usize;
                let n = 32 + extra * 4;
                let bytes = unsafe { Vec::from(core::slice::from_raw_parts(ev, n)) };
                let ext = bytes.get(1).copied().unwrap_or(0);
                if ext == self.c.present_opcode {
                    if self.matches_window(&bytes) {
                        self.handle_event(&bytes);
                    } else {
                        self.c.parked.lock().push(bytes);
                    }
                }
            }
            unsafe { free(ev as *mut c_void) };
        }
        true
    }

    fn matches_window(&self, ev: &[u8]) -> bool {
        ev.len() >= 20
            && (ev[0] & 0x7f) == XCB_GENERIC_EVENT
            && ev.get(1) == Some(&self.c.present_opcode)
            && read_u32_at(ev, 16) == Some(self.win)
    }

    fn handle_event(&mut self, ev: &[u8]) {
        let evtype = match read_u16_at(ev, 8) {
            Some(t) => t,
            None => return,
        };
        if evtype != PRESENT_EVENT_COMPLETE_NOTIFY && evtype != PRESENT_EVENT_IDLE_NOTIFY {
            return;
        }
        // Both notify layouts: eid @12, window @16, serial @20.
        let serial = match read_u32_at(ev, 20) {
            Some(s) => s,
            None => return,
        };
        if evtype == PRESENT_EVENT_COMPLETE_NOTIFY
            && ev.get(10) != Some(&PRESENT_COMPLETE_KIND_PIXMAP)
        {
            return; // NotifyMsc completions carry no buffer state
        }
        for s in self.slots.iter_mut() {
            if s.busy_serial == Some(serial) {
                s.busy_serial = None;
            }
        }
    }

    /// Block (spin + X event drain) until some slot frees.
    fn pump_wait_for_slot(&mut self) -> bool {
        loop {
            if !self.pump_events() {
                return false;
            }
            if self.free_slot().is_some() {
                return true;
            }
            if unsafe { (self.c.has_error)(self.c.conn) } != 0 {
                return false;
            }
            core::hint::spin_loop();
        }
    }

    /// Recreate buffers for a new geometry (EGL surface resize).
    pub fn resize(&mut self, width: u32, height: u32) -> bool {
        if width == self.width && height == self.height {
            return true;
        }
        if width == 0 || height == 0 {
            return false;
        }
        unsafe {
            for s in self.slots.iter() {
                if s.pixmap != 0 {
                    self.free_pixmap(s.pixmap);
                }
            }
        }
        self.slots.clear();
        self.render_slot = None;
        let mut slots = Vec::new();
        for _ in 0..3 {
            match self.dev.create_dumb(width, height, 32) {
                Ok(bo) => slots.push(Slot {
                    bo,
                    pixmap: 0,
                    busy_serial: None,
                }),
                Err(_) => return false, // old ring already gone; surface is dead
            }
        }
        self.slots = slots;
        self.width = width;
        self.height = height;
        true
    }

    unsafe fn free_pixmap(&self, pixmap: u32) {
        let mut req = FreePixmapReq {
            hdr: ReqHdr {
                major_opcode: XCB_CORE_FREE_PIXMAP,
                minor_opcode: 0,
                length: 2,
            },
            pixmap,
        };
        let _ = self
            .c
            .void_req(&mut req, core::ptr::null(), XCB_CORE_FREE_PIXMAP);
        unsafe { (self.c.flush)(self.c.conn) };
    }
}

impl Drop for Dri3Surface {
    fn drop(&mut self) {
        unsafe {
            for s in self.slots.iter() {
                if s.pixmap != 0 {
                    self.free_pixmap(s.pixmap);
                }
            }
        }
    }
}
