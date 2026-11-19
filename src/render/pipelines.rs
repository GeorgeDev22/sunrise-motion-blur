//! The blur's render pipelines, one per entry point of `shaders/blur.wgsl`.

use internals::sunrise_mod_render::render::wgpu;
use motion_blur_core::MotionVertex;

/// Blur radius in pixels (xy) and distance from the camera in blocks (z), per pixel.
pub const MOTION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// The longest blur radius in each tile.
pub const TILE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;
/// The hand's own depth, so its nearer faces win.
pub const HAND_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

pub struct Pipelines {
    module: wgpu::ShaderModule,
    pub camera: wgpu::RenderPipeline,
    pub objects: wgpu::RenderPipeline,
    pub hand: wgpu::RenderPipeline,
    pub tile_rows: wgpu::RenderPipeline,
    pub tile_columns: wgpu::RenderPipeline,
    /// Writes into the frame, so it is made for the frame's format when first needed.
    reconstruct: Option<(wgpu::TextureFormat, wgpu::RenderPipeline)>,
    pub atlas_sampler: wgpu::Sampler,
    pub smooth_sampler: wgpu::Sampler,
}

impl Pipelines {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Motion blur"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/blur.wgsl").into()),
        });
        let hand_depth = wgpu::DepthStencilState {
            format: HAND_DEPTH_FORMAT,
            depth_write_enabled: true,
            // Reversed depth, as the game draws it: nearer is larger.
            depth_compare: wgpu::CompareFunction::GreaterEqual,
            stencil: Default::default(),
            bias: Default::default(),
        };
        Self {
            camera: fullscreen(device, &module, "camera_motion", MOTION_FORMAT),
            objects: moving(device, &module, "object_motion", None),
            hand: moving(device, &module, "hand_motion", Some(hand_depth)),
            tile_rows: fullscreen(device, &module, "tile_max_rows", TILE_FORMAT),
            tile_columns: fullscreen(device, &module, "tile_max_columns", TILE_FORMAT),
            reconstruct: None,
            atlas_sampler: sampler(device, wgpu::FilterMode::Nearest),
            smooth_sampler: sampler(device, wgpu::FilterMode::Linear),
            module,
        }
    }

    /// Makes the blur pipeline for frames in `format`, unless it was made already.
    pub fn prepare_reconstruct(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat) {
        if self
            .reconstruct
            .as_ref()
            .is_none_or(|(made_for, _)| *made_for != format)
        {
            let pipeline = fullscreen(device, &self.module, "reconstruct", format);
            self.reconstruct = Some((format, pipeline));
        }
    }

    pub fn reconstruct(&self) -> &wgpu::RenderPipeline {
        &self
            .reconstruct
            .as_ref()
            .expect("prepare_reconstruct ran")
            .1
    }
}

fn fullscreen(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    entry: &str,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(entry),
        layout: None,
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("fullscreen"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            targets: &[Some(format.into())],
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn moving(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    entry: &str,
    depth_stencil: Option<wgpu::DepthStencilState>,
) -> wgpu::RenderPipeline {
    let attributes = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2];
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(entry),
        layout: None,
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("moving"),
            compilation_options: Default::default(),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: size_of::<MotionVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &attributes,
            })],
        },
        primitive: Default::default(),
        depth_stencil,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            targets: &[Some(MOTION_FORMAT.into())],
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn sampler(device: &wgpu::Device, filter: wgpu::FilterMode) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Motion blur"),
        mag_filter: filter,
        min_filter: filter,
        ..Default::default()
    })
}
