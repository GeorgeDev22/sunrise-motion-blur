//! The data the shaders read, laid out as `src/shaders/blur.wgsl` declares it
//! (`tests/shader.rs` holds them to it).

use bytemuck::{Pod, Zeroable};

/// One vertex of something that moves on its own: where it is now and where it was last frame,
/// each relative to that frame's camera.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct MotionVertex {
    pub position: [f32; 3],
    pub previous: [f32; 3],
    /// The atlas UV, for the hand's cut-out texels.
    pub uv: [f32; 2],
}

impl MotionVertex {
    pub fn bytes(vertices: &[MotionVertex]) -> &[u8] {
        bytemuck::cast_slice(vertices)
    }
}

/// `Params` in the shader: one frame's cameras and blur settings.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct BlurParams {
    /// Camera-relative world to clip space, without TAA's jitter.
    pub view_proj: [f32; 16],
    pub inverse_view_proj: [f32; 16],
    /// Last frame's `view_proj`.
    pub previous_view_proj: [f32; 16],
    /// What the geometry was rasterized with, jitter included.
    pub jittered: [f32; 16],
    /// xyz: the camera's motion since last frame, in blocks.
    pub camera_motion: [f32; 4],
    /// The picture's size and the world depth's, in pixels.
    pub size: [f32; 2],
    pub depth_size: [f32; 2],
    /// How much of a frame's motion becomes the streak.
    pub exposure: f32,
    /// The longest blur radius, in pixels.
    pub max_radius: f32,
    pub tile_size: u32,
    pub samples: u32,
}

impl BlurParams {
    pub fn bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}
