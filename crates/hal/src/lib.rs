//! Vantage HAL — Vulkan-shaped internal device layer.
//!
//! Mirrors the Device/Context/CommandBuffer shape of prism's `hal.zig`.
//! Future public Vulkan ICD (see `../prism/lib/prism-vk/icd.zig`) will be a
//! thin extern-C shim over these types.
//!
//! The software device keeps images as plain linear CPU allocations:
//! `R8G8B8A8Unorm` / `B8G8R8A8Unorm` color, `D32Sfloat` depth and separate
//! `S8Uint` stencil — exactly what the rasterizer consumes and what X11
//! MIT-SHM can take (with a swizzle stage at present time, Phase 5).
//! `PhysicalDevice::limits` advertise only what is implemented (mesa rule:
//! never lie in queries).
#![no_std]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

pub use volt_raster::{FragFn, FragState, RasterState, SampledTexture, Targets, Varyings, Vertex};

#[cfg(feature = "amdgpu")]
pub mod amdgpu;
#[cfg(feature = "amdgpu")]
pub use amdgpu::*;

use alloc::vec::Vec;
use hashbrown::HashMap;

// ============================================================================
// Resource ids
// ============================================================================

macro_rules! define_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(pub u32);
    };
}

define_id!(
    /// Handle to a linear CPU buffer.
    BufferId
);
define_id!(
    /// Handle to a 2D image (color, depth, or stencil).
    ImageId
);
define_id!(
    /// Handle to a fixed-function pipeline state object.
    PipelineId
);
define_id!(
    /// Handle to a descriptor set (texture bindings for now).
    DescriptorSetId
);

// ============================================================================
// Formats and limits
// ============================================================================

/// Pixel formats implemented by the software device. Nothing else exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    R8G8B8A8Unorm,
    B8G8R8A8Unorm,
    D32Sfloat,
    S8Uint,
}

impl Format {
    pub fn bytes_per_pixel(self) -> u32 {
        match self {
            Format::R8G8B8A8Unorm | Format::B8G8R8A8Unorm | Format::D32Sfloat => 4,
            Format::S8Uint => 1,
        }
    }

    pub fn is_color(self) -> bool {
        matches!(self, Format::R8G8B8A8Unorm | Format::B8G8R8A8Unorm)
    }
}

/// Device limits: exactly the implemented feature envelope, no more.
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    pub max_image_dimension_2d: u32,
    pub max_push_constants_size: u32,
}

// ============================================================================
// Indices and vertex input
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexType {
    U16,
    U32,
}

// ============================================================================
// Resource registries
// ============================================================================

/// A linear CPU buffer.
pub struct Buffer {
    pub data: Vec<u8>,
}

/// A 2D image. Interleaved planes: color images are one plane; depth/stencil
/// are separate `D32Sfloat` / `S8Uint` images (Vulkan-style separate depth
/// and stencil aspects).
///
/// `ext` aliases memory owned elsewhere (a mapped DRM GEM buffer backing a
/// DRI3 presentation slot). When set, `data` is empty and all access goes
/// through [`Image::slice`] / [`Image::slice_mut`]; the pointed memory must
/// outlive the image.
pub struct Image {
    pub format: Format,
    pub width: u32,
    pub height: u32,
    /// Row pitch in bytes. Equals `width * bpp` for owned images; may exceed
    /// it for externally-backed images (GEM buffer pitch).
    pub row_pitch: u32,
    pub data: Vec<u8>,
    pub ext: Option<ExtMem>,
}

/// Memory borrowed from outside the device (e.g. a mapped GEM bo).
pub struct ExtMem {
    pub ptr: *mut u8,
    pub len: usize,
}
// The borrow is managed by the presentation surface (DRI3 slot ring); the
// device never frees or moves it. Single-GPU driver target.
unsafe impl Send for ExtMem {}
unsafe impl Sync for ExtMem {}

impl Image {
    fn new(format: Format, width: u32, height: u32) -> Self {
        Image {
            format,
            width,
            height,
            row_pitch: width * format.bytes_per_pixel(),
            data: alloc::vec![0u8; (width * height * format.bytes_per_pixel()) as usize],
            ext: None,
        }
    }

    /// Full backing memory as bytes. For external images the buffer includes
    /// pitch padding; use `row_pitch` for row addressing.
    pub fn slice(&self) -> &[u8] {
        match &self.ext {
            Some(e) => unsafe { core::slice::from_raw_parts(e.ptr, e.len) },
            None => &self.data,
        }
    }

    pub fn slice_mut(&mut self) -> &mut [u8] {
        match &self.ext {
            Some(e) => unsafe { core::slice::from_raw_parts_mut(e.ptr, e.len) },
            None => &mut self.data,
        }
    }
}

/// Per-draw texture bindings (unit -> image id). Sampling state itself lives
/// with the pipeline/descriptor set consumer; the hal only carries handles.
#[derive(Debug, Clone, Default)]
pub struct DescriptorSet {
    pub textures: hashbrown::HashMap<u32, ImageId>,
}

/// Fixed-function pipeline state. Populated by `vantage-gles` from its
/// `PipelineKey`; consumed wholesale by the rasterizer at draw time.
#[derive(Debug, Clone)]
pub struct Pipeline {
    pub topology: u32,
    pub cull_mode: u32,
    pub front_face_ccw: bool,
    pub blend_enabled: bool,
    pub src_factor: u32,
    pub dst_factor: u32,
    pub depth_test: bool,
    pub depth_write: bool,
    pub depth_func: u32,
    pub color_mask: u8,
}

// ============================================================================
// Commands
// ============================================================================

/// Recorded command. This enum is the future Vulkan ICD shim surface.
#[derive(Debug, Clone)]
pub enum Cmd {
    FillBuffer {
        dst: BufferId,
        offset: u64,
        size: u64,
        value: u8,
    },
    CopyBufferToImage {
        src: BufferId,
        dst: ImageId,
        extent: (u32, u32),
    },
    BindPipeline {
        pipeline: PipelineId,
    },
    BindVertexBuffers {
        first: u32,
        buffers: smallvec::SmallVec<[(BufferId, u64); 8]>,
    },
    BindIndexBuffer {
        buffer: BufferId,
        offset: u64,
        index_ty: IndexType,
    },
    BindDescriptorSet {
        set: DescriptorSetId,
    },
    PushConstants {
        data: [u8; 256], // sized to hold FixedFunctionUniforms
    },
    SetViewport {
        x: i32,
        y: i32,
        w: u32,
        h: u32,
    },
    SetScissor(Option<(i32, i32, u32, u32)>),
    SetFragState(alloc::boxed::Box<volt_raster::FragState>),
    BindAttachments {
        color: Option<ImageId>,
        depth: Option<ImageId>,
        stencil: Option<ImageId>,
    },
    ClearAttachments {
        color: [f32; 4],
        depth: f32,
        stencil: u8,
        mask: u32,
    },
    Draw {
        count: u32,
        first: u32,
    },
    DrawIndexed {
        count: u32,
        first: u32,
    },
    DrawMesh {
        vertices: alloc::vec::Vec<Vertex>,
        indices: Option<alloc::vec::Vec<u32>>,
        pipeline: Pipeline,
    },
    /// Readback for `glReadPixels`.
    CopyImageToBuffer {
        src: ImageId,
        dst: BufferId,
    },
}

/// One-time recording bundle submitted to the queue.
#[derive(Debug, Default)]
pub struct CommandBuffer {
    pub ops: Vec<Cmd>,
}

impl CommandBuffer {
    pub fn push(&mut self, cmd: Cmd) {
        self.ops.push(cmd);
    }
}

// ============================================================================
// Instance / physical device
// ============================================================================

/// Enumerates software devices. One device: the CPU rasterizer.
pub struct Instance;

impl Instance {
    pub fn enumerate() -> Vec<PhysicalDevice> {
        alloc::vec![PhysicalDevice {
            limits: Limits {
                max_image_dimension_2d: 16384,
                max_push_constants_size: 256,
            },
        }]
    }
}

/// Properties of the single software device.
#[derive(Debug, Clone, Copy)]
pub struct PhysicalDevice {
    pub limits: Limits,
}

// ============================================================================
// Device and queue
// ============================================================================

/// The logical device: owns all resources and the single queue.
pub struct Device {
    buffers: HashMap<BufferId, Buffer>,
    images: HashMap<ImageId, Image>,
    pipelines: HashMap<PipelineId, Pipeline>,
    descriptor_sets: HashMap<DescriptorSetId, DescriptorSet>,
    pub color_attachment: Option<ImageId>,
    pub depth_attachment: Option<ImageId>,
    pub stencil_attachment: Option<ImageId>,
    next_id: u32,
    pub queue: Queue,
    /// AMD GPU hardware execution driver (opt-in via VANTAGE_AMDGPU=1).
    #[cfg(feature = "amdgpu")]
    pub hw: Option<crate::amdgpu::AmdgpuHardwareDriver>,
}

impl Default for Device {
    fn default() -> Self {
        Self::new()
    }
}

impl Device {
    pub fn new() -> Self {
        Device {
            buffers: HashMap::new(),
            images: HashMap::new(),
            pipelines: HashMap::new(),
            descriptor_sets: HashMap::new(),
            color_attachment: None,
            depth_attachment: None,
            stencil_attachment: None,
            next_id: 1,
            queue: Queue,
            #[cfg(feature = "amdgpu")]
            hw: None,
        }
    }

    #[cfg(feature = "amdgpu")]
    pub fn with_amdgpu(transport: alloc::boxed::Box<dyn crate::amdgpu::AmdgpuTransport>) -> Self {
        let mut dev = Self::new();
        dev.hw = crate::amdgpu::AmdgpuHardwareDriver::from_transport(transport).ok();
        dev
    }

    fn alloc_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    pub fn create_buffer(&mut self, size: u64) -> BufferId {
        let id = BufferId(self.alloc_id());
        self.buffers.insert(
            id,
            Buffer {
                data: alloc::vec![0u8; size as usize],
            },
        );
        id
    }

    pub fn create_image(&mut self, format: Format, width: u32, height: u32) -> ImageId {
        let id = ImageId(self.alloc_id());
        #[cfg(feature = "amdgpu")]
        if let Some(hw) = self.hw.as_mut()
            && format.is_color()
        {
            let unaligned_pitch = width * format.bytes_per_pixel();
            let pitch = (unaligned_pitch + 255) & !255;
            let size = (pitch as u64) * (height as u64);
            if let Ok(bo) = hw.alloc_gtt(size) {
                let ptr = bo.cpu_ptr();
                let len = bo.size() as usize;
                hw.image_bos.insert(id, bo);
                self.images.insert(
                    id,
                    Image {
                        format,
                        width,
                        height,
                        row_pitch: pitch,
                        data: alloc::vec![],
                        ext: Some(ExtMem { ptr, len }),
                    },
                );
                return id;
            }
        }
        self.images.insert(id, Image::new(format, width, height));
        id
    }

    /// Create an image aliasing externally-owned memory (a mapped GEM bo
    /// backing a DRI3 presentation slot). `pitch` is the row stride in bytes
    /// and may exceed `width * bpp`.
    ///
    /// # Safety
    /// `ptr` must remain valid and immutable-alias-free for the image's
    /// lifetime; the owner (presentation surface) guarantees that.
    pub unsafe fn create_image_external(
        &mut self,
        format: Format,
        width: u32,
        height: u32,
        ptr: *mut u8,
        len: usize,
        pitch: u32,
    ) -> ImageId {
        let id = ImageId(self.alloc_id());
        self.images.insert(
            id,
            Image {
                format,
                width,
                height,
                row_pitch: pitch,
                data: alloc::vec![],
                ext: Some(ExtMem { ptr, len }),
            },
        );
        id
    }

    /// Re-point an external image at new memory (next presentation slot of
    /// the same geometry). No-op for owned images.
    ///
    /// # Safety
    /// Same contract as [`Device::create_image_external`].
    pub unsafe fn set_image_external(&mut self, id: ImageId, ptr: *mut u8, len: usize, pitch: u32) {
        if let Some(im) = self.images.get_mut(&id)
            && im.ext.is_some()
        {
            im.ext = Some(ExtMem { ptr, len });
            im.row_pitch = pitch;
        }
    }

    pub fn create_pipeline(&mut self, pipeline: Pipeline) -> PipelineId {
        let id = PipelineId(self.alloc_id());
        self.pipelines.insert(id, pipeline);
        id
    }

    pub fn create_descriptor_set(&mut self, set: DescriptorSet) -> DescriptorSetId {
        let id = DescriptorSetId(self.alloc_id());
        self.descriptor_sets.insert(id, set);
        id
    }

    pub fn destroy_buffer(&mut self, id: BufferId) {
        self.buffers.remove(&id);
    }

    pub fn destroy_image(&mut self, id: ImageId) {
        #[cfg(feature = "amdgpu")]
        if let Some(hw) = self.hw.as_mut() {
            let _ = hw.flush();
            hw.image_bos.remove(&id);
        }
        self.images.remove(&id);
    }

    /// Borrow a buffer's bytes (readback / upload staging).
    pub fn buffer(&self, id: BufferId) -> Option<&Buffer> {
        self.buffers.get(&id)
    }

    pub fn buffer_mut(&mut self, id: BufferId) -> Option<&mut Buffer> {
        self.buffers.get_mut(&id)
    }

    pub fn image(&self, id: ImageId) -> Option<&Image> {
        self.images.get(&id)
    }

    pub fn image_mut(&mut self, id: ImageId) -> Option<&mut Image> {
        self.images.get_mut(&id)
    }

    /// Execute a recorded command buffer on the single queue.
    pub fn submit(&mut self, cmd: &CommandBuffer) {
        Queue::execute(cmd, self);
    }
}

/// The single in-order queue. Stateful draw execution lives in `execute`.
#[derive(Default)]
pub struct Queue;

impl Queue {
    fn execute(cmd: &CommandBuffer, dev: &mut Device) {
        use Cmd::*;
        // Draw-time state carried across ops.
        let mut bound_pipeline: Option<PipelineId> = None;
        let mut _bound_descriptors: Option<DescriptorSetId> = None;
        let mut vertex_buffers: smallvec::SmallVec<[(BufferId, u64); 8]> =
            smallvec::SmallVec::new();
        let mut index_buffer: Option<(BufferId, u64, IndexType)> = None;
        let mut viewport = (0i32, 0i32, 1u32, 1u32);
        let mut scissor: Option<(i32, i32, u32, u32)> = None;
        let mut _push: [u8; 256] = [0; 256];
        // Target attachments for the current pass (set by ClearAttachments).
        let mut color_target = dev.color_attachment;
        let mut depth_target = dev.depth_attachment;
        let mut stencil_target = dev.stencil_attachment;
        let mut current_frag_state = volt_raster::FragState::default();

        for op in &cmd.ops {
            match op {
                FillBuffer {
                    dst,
                    offset,
                    size,
                    value,
                } => {
                    #[cfg(feature = "amdgpu")]
                    if let Some(hw) = dev.hw.as_mut() {
                        let _ = hw.flush();
                    }
                    if let Some(b) = dev.buffer_mut(*dst) {
                        let start = (*offset as usize).min(b.data.len());
                        let end = (start + *size as usize).min(b.data.len());
                        b.data[start..end].fill(*value);
                    }
                }
                CopyBufferToImage { src, dst, extent } => {
                    #[cfg(feature = "amdgpu")]
                    if let Some(hw) = dev.hw.as_mut() {
                        let _ = hw.flush();
                    }
                    let Some(sb) = dev.buffers.remove(src) else {
                        continue;
                    };
                    let Some(im) = dev.images.get_mut(dst) else {
                        dev.buffers.insert(*src, sb);
                        continue;
                    };
                    let bpp = im.format.bytes_per_pixel() as usize;
                    let row = im.width as usize * bpp;
                    let pitch = im.row_pitch as usize;
                    let rows = (extent.1 as usize).min(im.height as usize);
                    let mem = im.slice_mut();
                    for y in 0..rows {
                        let src_off = y * row;
                        let dst_off = y * pitch;
                        let n = row.min(sb.data.len().saturating_sub(src_off));
                        if dst_off + n <= mem.len() {
                            mem[dst_off..dst_off + n]
                                .copy_from_slice(&sb.data[src_off..src_off + n]);
                        }
                    }
                    dev.buffers.insert(*src, sb);
                }
                CopyImageToBuffer { src, dst } => {
                    #[cfg(feature = "amdgpu")]
                    if let Some(hw) = dev.hw.as_mut() {
                        let _ = hw.flush();
                    }
                    let Some(im) = dev.images.get(src) else {
                        continue;
                    };
                    let Some(db) = dev.buffers.get_mut(dst) else {
                        continue;
                    };
                    let src_mem = im.slice();
                    let row = im.width as usize * im.format.bytes_per_pixel() as usize;
                    let rows = im.height as usize;
                    for y in 0..rows {
                        let s = y * im.row_pitch as usize;
                        let d = y * row;
                        if d + row <= db.data.len() && s + row <= src_mem.len() {
                            db.data[d..d + row].copy_from_slice(&src_mem[s..s + row]);
                        }
                    }
                }
                BindPipeline { pipeline } => bound_pipeline = Some(*pipeline),
                BindDescriptorSet { set } => _bound_descriptors = Some(*set),
                BindVertexBuffers { first, buffers } => {
                    let first = *first as usize;
                    while vertex_buffers.len() < first {
                        vertex_buffers.push((BufferId(0), 0));
                    }
                    for (i, b) in buffers.iter().enumerate() {
                        let idx = first + i;
                        if idx < vertex_buffers.len() {
                            vertex_buffers[idx] = *b;
                        } else {
                            vertex_buffers.push(*b);
                        }
                    }
                }
                BindIndexBuffer {
                    buffer,
                    offset,
                    index_ty,
                } => {
                    index_buffer = Some((*buffer, *offset, *index_ty));
                }
                PushConstants { data } => _push = *data,
                SetViewport { x, y, w, h } => viewport = (*x, *y, *w, *h),
                SetScissor(rect) => scissor = *rect,
                BindAttachments {
                    color,
                    depth,
                    stencil,
                } => {
                    color_target = *color;
                    depth_target = *depth;
                    stencil_target = *stencil;
                    dev.color_attachment = *color;
                    dev.depth_attachment = *depth;
                    dev.stencil_attachment = *stencil;
                }
                ClearAttachments {
                    color,
                    depth,
                    stencil,
                    mask,
                } => {
                    #[cfg(feature = "amdgpu")]
                    if let Some(hw) = dev.hw.as_mut() {
                        let _ = hw.flush();
                    }
                    if mask & 1 != 0
                        && let Some(im) = color_target.and_then(|id| dev.image_mut(id))
                    {
                        clear_color_image(im, *color);
                    }
                    if mask & 2 != 0
                        && let Some(im) = depth_target.and_then(|id| dev.image_mut(id))
                    {
                        clear_depth_image(im, *depth);
                    }
                    if mask & 4 != 0
                        && let Some(im) = stencil_target.and_then(|id| dev.image_mut(id))
                    {
                        im.slice_mut().fill(*stencil);
                    }
                }
                SetFragState(fs) => {
                    current_frag_state = (**fs).clone();
                }
                Draw { count, first } => {
                    execute_draw(
                        dev,
                        bound_pipeline,
                        &vertex_buffers,
                        None,
                        *first,
                        *count,
                        viewport,
                        scissor,
                        color_target,
                        depth_target,
                        stencil_target,
                        &current_frag_state,
                    );
                }
                DrawIndexed { count, first } => {
                    execute_draw(
                        dev,
                        bound_pipeline,
                        &vertex_buffers,
                        index_buffer,
                        *first,
                        *count,
                        viewport,
                        scissor,
                        color_target,
                        depth_target,
                        stencil_target,
                        &current_frag_state,
                    );
                }
                DrawMesh {
                    vertices,
                    indices,
                    pipeline,
                } => {
                    execute_draw_mesh(
                        dev,
                        pipeline,
                        vertices,
                        indices.as_deref(),
                        viewport,
                        scissor,
                        color_target,
                        depth_target,
                        stencil_target,
                        &current_frag_state,
                    );
                }
            }
        }

        // Flush any remaining batched GPU commands before returning.
        #[cfg(feature = "amdgpu")]
        if let Some(hw) = dev.hw.as_mut() {
            let _ = hw.flush();
        }
    }
}

pub(crate) fn clear_color_image(im: &mut Image, color: [f32; 4]) {
    let (b0, b1, b2, b3) = match im.format {
        // Stored byte order: R,G,B,A.
        Format::R8G8B8A8Unorm => (color[0], color[1], color[2], color[3]),
        // Stored byte order: B,G,R,A.
        Format::B8G8R8A8Unorm => (color[2], color[1], color[0], color[3]),
        _ => return,
    };
    let mut row_bytes = [0u8; 4];
    row_bytes[0] = (b0 * 255.0 + 0.5) as u8;
    row_bytes[1] = (b1 * 255.0 + 0.5) as u8;
    row_bytes[2] = (b2 * 255.0 + 0.5) as u8;
    row_bytes[3] = (b3 * 255.0 + 0.5) as u8;
    let (pitch, height, width) = (im.row_pitch as usize, im.height as usize, im.width as usize);
    let mem = im.slice_mut();
    for y in 0..height {
        let off = y * pitch;
        if off + 4 > mem.len() {
            break;
        }
        let end = (off + width * 4).min(mem.len());
        let row = &mut mem[off..end];
        for px in row.as_chunks_mut::<4>().0 {
            *px = row_bytes;
        }
    }
}

pub(crate) fn clear_depth_image(im: &mut Image, depth: f32) {
    if im.format != Format::D32Sfloat {
        return;
    }
    let mut row_bytes = [0u8; 4];
    row_bytes.copy_from_slice(&depth.to_ne_bytes());
    let (pitch, height, width) = (im.row_pitch as usize, im.height as usize, im.width as usize);
    let mem = im.slice_mut();
    for y in 0..height {
        let off = y * pitch;
        if off + 4 > mem.len() {
            break;
        }
        let end = (off + width * 4).min(mem.len());
        let row = &mut mem[off..end];
        for px in row.as_chunks_mut::<4>().0 {
            *px = row_bytes;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_draw(
    dev: &mut Device,
    pipeline_id: Option<PipelineId>,
    vertex_buffers: &[(BufferId, u64)],
    index_info: Option<(BufferId, u64, IndexType)>,
    first: u32,
    count: u32,
    viewport: (i32, i32, u32, u32),
    scissor: Option<(i32, i32, u32, u32)>,
    color_id: Option<ImageId>,
    depth_id: Option<ImageId>,
    stencil_id: Option<ImageId>,
    frag_state: &volt_raster::FragState,
) {
    let Some(pipe_id) = pipeline_id else {
        return;
    };
    let Some(pipe) = dev.pipelines.get(&pipe_id).cloned() else {
        return;
    };
    let Some(c_id) = color_id else {
        return;
    };

    // Get vertex buffer data
    let Some((v_buf_id, v_offset)) = vertex_buffers.first().cloned() else {
        return;
    };
    let Some(v_buf) = dev.buffer(v_buf_id) else {
        return;
    };

    let v_slice = &v_buf.data[(v_offset as usize)..];
    let v_size = core::mem::size_of::<Vertex>();
    let num_verts = v_slice.len() / v_size;
    let vertices: &[Vertex] =
        unsafe { core::slice::from_raw_parts(v_slice.as_ptr() as *const Vertex, num_verts) };

    // Subslice according to first..first+count
    #[allow(unused_variables)]
    let v_sub = if (first as usize) < vertices.len() {
        let end = (first as usize + count as usize).min(vertices.len());
        &vertices[first as usize..end]
    } else {
        &[]
    };

    let indices_vec: Option<alloc::vec::Vec<u32>> = match index_info {
        Some((i_buf_id, i_offset, i_type)) => dev.buffer(i_buf_id).map(|i_buf| {
            let i_slice = &i_buf.data[(i_offset as usize)..];
            match i_type {
                IndexType::U16 => {
                    let num = i_slice.len() / 2;
                    let s: &[u16] =
                        unsafe { core::slice::from_raw_parts(i_slice.as_ptr() as *const u16, num) };
                    s.iter().map(|&idx| idx as u32).collect()
                }
                IndexType::U32 => {
                    let num = i_slice.len() / 4;
                    let s: &[u32] =
                        unsafe { core::slice::from_raw_parts(i_slice.as_ptr() as *const u32, num) };
                    s.to_vec()
                }
            }
        }),
        None => None,
    };

    // Attempt GPU hardware execution first.
    #[cfg(feature = "amdgpu")]
    if let Some(ref mut hw) = dev.hw {
        if let Some(ps_key) = hw.can_draw(color_id, &pipe, frag_state) {
            let (w, h, pitch, is_bgra) = {
                let im = dev.images.get(&c_id).unwrap();
                (
                    im.width,
                    im.height,
                    im.row_pitch,
                    im.format == Format::B8G8R8A8Unorm,
                )
            };
            if hw
                .record_draw(
                    c_id,
                    &pipe,
                    v_sub,
                    indices_vec.as_deref(),
                    viewport,
                    scissor,
                    frag_state,
                    &ps_key,
                    w,
                    h,
                    pitch,
                    is_bgra,
                )
                .is_ok()
            {
                return;
            }
        }
        let _ = hw.flush();
    }

    // Construct raster state for CPU fallback
    let raster_state = RasterState {
        viewport,
        depth_range: (0.0, 1.0),
        scissor,
        cull_mode: pipe.cull_mode,
        front_face_ccw: pipe.front_face_ccw,
        shade_flat: false,
        depth_test: pipe.depth_test,
        depth_write: pipe.depth_write,
        depth_func: pipe.depth_func,
        stencil_test: false,
        stencil_func: volt_raster::gl::ALWAYS,
        stencil_ref: 0,
        stencil_func_mask: !0,
        stencil_write_mask: !0,
        stencil_fail: volt_raster::gl::STENCIL_KEEP,
        stencil_zfail: volt_raster::gl::STENCIL_KEEP,
        stencil_zpass: volt_raster::gl::STENCIL_KEEP,
        blend_enabled: pipe.blend_enabled,
        src_rgb: pipe.src_factor,
        dst_rgb: pipe.dst_factor,
        src_alpha: pipe.src_factor,
        dst_alpha: pipe.dst_factor,
        blend_color: [0.0; 4],
        color_mask: pipe.color_mask,
        polygon_offset_fill: false,
        polygon_offset_factor: 0.0,
        polygon_offset_units: 0.0,
        point_size: 1.0,
        line_width: 1.0,
    };

    let mut color_img = dev.images.remove(&c_id);
    let mut depth_img = depth_id.and_then(|id| dev.images.remove(&id));
    let mut stencil_img = stencil_id.and_then(|id| dev.images.remove(&id));

    if let Some(ref mut c_im) = color_img {
        let is_bgra = c_im.format == Format::B8G8R8A8Unorm;
        let c_stride = c_im.row_pitch as usize;
        let w = c_im.width;
        let h = c_im.height;

        let d_slice = depth_img.as_mut().map(|im| im.slice_mut());
        let s_slice = stencil_img.as_mut().map(|im| im.slice_mut());

        let targets = Targets {
            color: c_im.slice_mut(),
            color_stride: c_stride as u32,
            width: w,
            height: h,
            bgra_order: is_bgra,
            depth: d_slice,
            stencil: s_slice,
        };

        let frag_fn: FragFn = volt_shader::fragment_entry_point;

        let mut raster = volt_raster::PrimitiveRasterizer::new(
            &raster_state,
            targets,
            frag_fn,
            frag_state as *const FragState,
        );

        if let Some(ref indices) = indices_vec {
            let end = (first + count) as usize;
            let mut i = first as usize;
            while i + 2 < end.min(indices.len()) {
                let idx0 = indices[i] as usize;
                let idx1 = indices[i + 1] as usize;
                let idx2 = indices[i + 2] as usize;
                if idx0 < vertices.len() && idx1 < vertices.len() && idx2 < vertices.len() {
                    raster.draw_triangle(&vertices[idx0], &vertices[idx1], &vertices[idx2]);
                }
                i += 3;
            }
        } else {
            let end = (first + count) as usize;
            let mut i = first as usize;
            while i + 2 < end.min(vertices.len()) {
                raster.draw_triangle(&vertices[i], &vertices[i + 1], &vertices[i + 2]);
                i += 3;
            }
        }
    }

    if let Some(c_im) = color_img {
        dev.images.insert(c_id, c_im);
    }
    if let (Some(d_id), Some(d_im)) = (depth_id, depth_img) {
        dev.images.insert(d_id, d_im);
    }
    if let (Some(s_id), Some(s_im)) = (stencil_id, stencil_img) {
        dev.images.insert(s_id, s_im);
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_draw_mesh(
    dev: &mut Device,
    pipe: &Pipeline,
    vertices: &[Vertex],
    indices_opt: Option<&[u32]>,
    viewport: (i32, i32, u32, u32),
    scissor: Option<(i32, i32, u32, u32)>,
    color_id: Option<ImageId>,
    depth_id: Option<ImageId>,
    stencil_id: Option<ImageId>,
    frag_state: &volt_raster::FragState,
) {
    let Some(c_id) = color_id else {
        return;
    };

    // Attempt GPU hardware execution first.
    #[cfg(feature = "amdgpu")]
    if let Some(ref mut hw) = dev.hw {
        if let Some(ps_key) = hw.can_draw(color_id, pipe, frag_state) {
            let (w, h, pitch, is_bgra) = {
                let im = dev.images.get(&c_id).unwrap();
                (
                    im.width,
                    im.height,
                    im.row_pitch,
                    im.format == Format::B8G8R8A8Unorm,
                )
            };
            if hw
                .record_draw(
                    c_id,
                    pipe,
                    vertices,
                    indices_opt,
                    viewport,
                    scissor,
                    frag_state,
                    &ps_key,
                    w,
                    h,
                    pitch,
                    is_bgra,
                )
                .is_ok()
            {
                return;
            }
        }
        let _ = hw.flush();
    }

    let raster_state = RasterState {
        viewport,
        depth_range: (0.0, 1.0),
        scissor,
        cull_mode: pipe.cull_mode,
        front_face_ccw: pipe.front_face_ccw,
        shade_flat: false,
        depth_test: pipe.depth_test,
        depth_write: pipe.depth_write,
        depth_func: pipe.depth_func,
        stencil_test: false,
        stencil_func: volt_raster::gl::ALWAYS,
        stencil_ref: 0,
        stencil_func_mask: !0,
        stencil_write_mask: !0,
        stencil_fail: volt_raster::gl::STENCIL_KEEP,
        stencil_zfail: volt_raster::gl::STENCIL_KEEP,
        stencil_zpass: volt_raster::gl::STENCIL_KEEP,
        blend_enabled: pipe.blend_enabled,
        src_rgb: pipe.src_factor,
        dst_rgb: pipe.dst_factor,
        src_alpha: pipe.src_factor,
        dst_alpha: pipe.dst_factor,
        blend_color: [0.0; 4],
        color_mask: pipe.color_mask,
        polygon_offset_fill: false,
        polygon_offset_factor: 0.0,
        polygon_offset_units: 0.0,
        point_size: 1.0,
        line_width: 1.0,
    };

    let mut color_img = dev.images.remove(&c_id);
    let mut depth_img = depth_id.and_then(|id| dev.images.remove(&id));
    let mut stencil_img = stencil_id.and_then(|id| dev.images.remove(&id));

    if let Some(ref mut c_im) = color_img {
        let is_bgra = c_im.format == Format::B8G8R8A8Unorm;
        let c_stride = c_im.row_pitch as usize;
        let w = c_im.width;
        let h = c_im.height;

        let d_slice = depth_img.as_mut().map(|im| im.slice_mut());
        let s_slice = stencil_img.as_mut().map(|im| im.slice_mut());

        let targets = Targets {
            color: c_im.slice_mut(),
            color_stride: c_stride as u32,
            width: w,
            height: h,
            bgra_order: is_bgra,
            depth: d_slice,
            stencil: s_slice,
        };

        let frag_fn: FragFn = volt_shader::fragment_entry_point;

        let mut raster = volt_raster::PrimitiveRasterizer::new(
            &raster_state,
            targets,
            frag_fn,
            frag_state as *const FragState,
        );

        if let Some(indices) = indices_opt {
            let mut i = 0;
            while i + 2 < indices.len() {
                let idx0 = indices[i] as usize;
                let idx1 = indices[i + 1] as usize;
                let idx2 = indices[i + 2] as usize;
                if idx0 < vertices.len() && idx1 < vertices.len() && idx2 < vertices.len() {
                    raster.draw_triangle(&vertices[idx0], &vertices[idx1], &vertices[idx2]);
                }
                i += 3;
            }
        } else {
            let mut i = 0;
            while i + 2 < vertices.len() {
                raster.draw_triangle(&vertices[i], &vertices[i + 1], &vertices[i + 2]);
                i += 3;
            }
        }
    }

    if let Some(c_im) = color_img {
        dev.images.insert(c_id, c_im);
    }
    if let (Some(d_id), Some(d_im)) = (depth_id, depth_img) {
        dev.images.insert(d_id, d_im);
    }
    if let (Some(s_id), Some(s_im)) = (stencil_id, stencil_img) {
        dev.images.insert(s_id, s_im);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draw_indexed_triangle_into_image_and_readback() {
        let mut dev = Device::new();

        let color_img = dev.create_image(Format::R8G8B8A8Unorm, 64, 64);
        let depth_img = dev.create_image(Format::D32Sfloat, 64, 64);
        let readback_buf = dev.create_buffer(64 * 64 * 4);

        let v0 = Vertex {
            pos: [-1.0, -1.0, 0.5, 1.0],
            color: [0.0, 1.0, 0.0, 1.0], // Green
            tex0: [0.0, 0.0],
            tex1: [0.0, 0.0],
            fog: 0.0,
            _pad: 0.0,
        };
        let v1 = Vertex {
            pos: [1.0, -1.0, 0.5, 1.0],
            color: [0.0, 1.0, 0.0, 1.0],
            tex0: [1.0, 0.0],
            tex1: [0.0, 0.0],
            fog: 0.0,
            _pad: 0.0,
        };
        let v2 = Vertex {
            pos: [0.0, 1.0, 0.5, 1.0],
            color: [0.0, 1.0, 0.0, 1.0],
            tex0: [0.5, 1.0],
            tex1: [0.0, 0.0],
            fog: 0.0,
            _pad: 0.0,
        };

        let v_array = [v0, v1, v2];
        let v_bytes: &[u8] = bytemuck::cast_slice(&v_array);
        let v_buf = dev.create_buffer(v_bytes.len() as u64);
        dev.buffer_mut(v_buf).unwrap().data.copy_from_slice(v_bytes);

        let indices: [u16; 3] = [0, 1, 2];
        let i_bytes: &[u8] = bytemuck::cast_slice(&indices);
        let i_buf = dev.create_buffer(i_bytes.len() as u64);
        dev.buffer_mut(i_buf).unwrap().data.copy_from_slice(i_bytes);

        let pipe = dev.create_pipeline(Pipeline {
            topology: 0,
            cull_mode: 0,
            front_face_ccw: true,
            blend_enabled: false,
            src_factor: 1,
            dst_factor: 0,
            depth_test: true,
            depth_write: true,
            depth_func: volt_raster::gl::LEQUAL,
            color_mask: 0x0F,
        });

        let mut cmd = CommandBuffer::default();
        cmd.push(Cmd::BindAttachments {
            color: Some(color_img),
            depth: Some(depth_img),
            stencil: None,
        });
        cmd.push(Cmd::ClearAttachments {
            color: [0.0, 0.0, 0.0, 1.0],
            depth: 1.0,
            stencil: 0,
            mask: 1 | 2,
        });
        cmd.push(Cmd::BindPipeline { pipeline: pipe });
        cmd.push(Cmd::SetViewport {
            x: 0,
            y: 0,
            w: 64,
            h: 64,
        });
        let mut v_vec = smallvec::SmallVec::new();
        v_vec.push((v_buf, 0));
        cmd.push(Cmd::BindVertexBuffers {
            first: 0,
            buffers: v_vec,
        });
        cmd.push(Cmd::BindIndexBuffer {
            buffer: i_buf,
            offset: 0,
            index_ty: IndexType::U16,
        });
        cmd.push(Cmd::DrawIndexed { count: 3, first: 0 });
        cmd.push(Cmd::CopyImageToBuffer {
            src: color_img,
            dst: readback_buf,
        });

        dev.submit(&cmd);

        let out_bytes = &dev.buffer(readback_buf).unwrap().data;
        let mut green_count = 0;
        for px in out_bytes.as_chunks::<4>().0 {
            if px[0] == 0 && px[1] == 255 && px[2] == 0 && px[3] == 255 {
                green_count += 1;
            }
        }
        assert!(green_count > 100, "Green triangle should be rasterized");
    }

    #[test]
    fn clear_without_attachment_does_not_panic() {
        let mut dev = Device::new();
        let color = dev.create_image(Format::R8G8B8A8Unorm, 4, 4);
        let mut cmd = CommandBuffer::default();
        cmd.push(Cmd::ClearAttachments {
            color: [1.0, 0.0, 0.0, 1.0],
            depth: 0.5,
            stencil: 0,
            mask: 1 | 2,
        });
        dev.submit(&cmd);
        let im = dev.image(color).unwrap();
        assert!(im.data.iter().all(|&b| b == 0));
    }
}
