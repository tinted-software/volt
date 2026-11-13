//! Fixed-function pipeline state key.
//!
//! Identifies a unique combination of raster/fixed-function state for the
//! pipeline cache. Carried over from the wgpu renderer; it also covers the
//! fragment-state portion of the fragment pipeline.

/// Raster/fixed-function pipeline identity.
pub struct PipelineKey {
    pub topology: u32, // 0=triangles, 1=triangle_strip, 2=triangle_fan, 3=lines, 4=line_strip, 5=points
    pub cull_mode: u32, // 0=none, 1=front, 2=back
    pub front_face_ccw: bool,
    pub blend_enabled: bool,
    pub src_factor: u32,
    pub dst_factor: u32,
    pub depth_test: bool,
    pub depth_write: bool,
    pub depth_func: u32,
    pub color_mask: u8,
}
