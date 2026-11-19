//! The textures the blur draws into, made again when the frame's size or format changes.

use super::pipelines::{HAND_DEPTH_FORMAT, MOTION_FORMAT, TILE_FORMAT};
use internals::sunrise_mod_render::render::wgpu;

pub struct Targets {
    size: [u32; 2],
    format: wgpu::TextureFormat,
    tile_size: u32,
    /// A copy of the frame: the blur reads it while writing the frame.
    pub scene: wgpu::Texture,
    pub scene_view: wgpu::TextureView,
    pub motion: wgpu::TextureView,
    pub tile_rows: wgpu::TextureView,
    pub tiles: wgpu::TextureView,
    pub hand_depth: wgpu::TextureView,
}

impl Targets {
    pub fn fits(&self, size: [u32; 2], format: wgpu::TextureFormat, tile_size: u32) -> bool {
        self.size == size && self.format == format && self.tile_size == tile_size
    }

    pub fn new(
        device: &wgpu::Device,
        size: [u32; 2],
        format: wgpu::TextureFormat,
        tile_size: u32,
    ) -> Self {
        let [width, height] = size;
        let tiles_wide = width.div_ceil(tile_size);
        let tiles_high = height.div_ceil(tile_size);
        let drawn = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let scene = texture(
            device,
            "Motion blur scene",
            size,
            format,
            wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        Self {
            size,
            format,
            tile_size,
            scene_view: scene.create_view(&Default::default()),
            scene,
            motion: view(device, "Motion blur motion", size, MOTION_FORMAT, drawn),
            tile_rows: view(
                device,
                "Motion blur tile rows",
                [tiles_wide, height],
                TILE_FORMAT,
                drawn,
            ),
            tiles: view(
                device,
                "Motion blur tiles",
                [tiles_wide, tiles_high],
                TILE_FORMAT,
                drawn,
            ),
            hand_depth: view(
                device,
                "Motion blur hand depth",
                size,
                HAND_DEPTH_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT,
            ),
        }
    }
}

fn texture(
    device: &wgpu::Device,
    label: &str,
    [width, height]: [u32; 2],
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

fn view(
    device: &wgpu::Device,
    label: &str,
    size: [u32; 2],
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::TextureView {
    texture(device, label, size, format, usage).create_view(&Default::default())
}
