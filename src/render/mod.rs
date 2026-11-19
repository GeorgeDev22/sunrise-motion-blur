//! The blur on the GPU, run from the game's render functions.
//!
//! A frame goes through it in two steps. Once the world is drawn (`world_drawn`), each pixel's
//! motion is worked out: the camera's from the depth, and each moving mob's from its own
//! vertices. Then the picture is blurred, there, or, when the hand is blurred too, once the
//! hand is drawn over it (`hand_drawn`). The HUD comes after, so it always stays sharp.
//!
//! Nothing runs while the blur is off or nothing on screen moves.

mod pipelines;
mod targets;

use crate::settings;
use internals::sunrise_mod_render::render::{HandDrawn, RenderNeeds, WorldDrawn, wgpu};
use motion_blur_core::{
    BlurParams, CameraHistory, FrameClock, HandHistory, Mat4, MeshHistory, MotionVertex, Settings,
};
use pipelines::Pipelines;
use std::sync::{Mutex, PoisonError};
use std::time::Instant;
use targets::Targets;

/// The blur's GPU state, made on the first frame it draws and dropped when it's turned off.
static BLUR: Mutex<Option<Blur>> = Mutex::new(None);

/// The tile size bounds: the longest radius the blur reaches is one tile.
const MIN_TILE: u32 = 8;
const MAX_TILE: u32 = 64;

pub fn render_needs(needs: &mut RenderNeeds) {
    if settings::current().active() {
        needs.world_depth = true;
    }
}

pub fn world_drawn(world: &mut WorldDrawn<'_>) {
    let settings = settings::current();
    let mut blur = BLUR.lock().unwrap_or_else(PoisonError::into_inner);
    if !settings.active() {
        *blur = None;
        return;
    }
    let blur = match blur.take() {
        Some(existing) if existing.device == *world.device => blur.insert(existing),
        _ => blur.insert(Blur::new(world.device)),
    };
    blur.world_drawn(world, &settings);
}

pub fn hand_drawn(hand: &mut HandDrawn<'_>) {
    let settings = settings::current();
    let mut blur = BLUR.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(blur) = blur.as_mut().filter(|blur| blur.device == *hand.device) {
        blur.hand_drawn(hand, &settings);
    }
}

/// What the world step leaves for the hand step of the same frame.
struct Pending {
    params: BlurParams,
    /// Whether the world step wrote the motion texture; if not, nothing in the world moved.
    world_moved: bool,
}

struct Blur {
    device: wgpu::Device,
    pipelines: Pipelines,
    targets: Option<Targets>,
    params: wgpu::Buffer,
    hand_params: wgpu::Buffer,
    objects: VertexBuffer,
    hand_vertices: VertexBuffer,
    clock: FrameClock,
    camera: CameraHistory,
    meshes: MeshHistory,
    hand: HandHistory,
    /// Reused each frame, so a settled scene allocates nothing.
    moving: Vec<MotionVertex>,
    pending: Option<Pending>,
}

impl Blur {
    fn new(device: &wgpu::Device) -> Self {
        Self {
            pipelines: Pipelines::new(device),
            targets: None,
            params: uniform_buffer(device),
            hand_params: uniform_buffer(device),
            objects: VertexBuffer::default(),
            hand_vertices: VertexBuffer::default(),
            clock: FrameClock::default(),
            camera: CameraHistory::default(),
            meshes: MeshHistory::default(),
            hand: HandHistory::default(),
            moving: Vec::new(),
            pending: None,
            device: device.clone(),
        }
    }

    fn world_drawn(&mut self, world: &mut WorldDrawn<'_>, settings: &Settings) {
        self.pending = None;
        let frame_seconds = self.clock.tick(Instant::now());
        let view_proj = Mat4(world.view_proj.to_cols_array());
        let camera = [world.camera.x, world.camera.y, world.camera.z];
        let reprojection = self.camera.advance(view_proj, camera);

        let objects_moved = if settings.moving_objects {
            let entities = world.entities;
            self.meshes.update(
                world.motion,
                entities.len(),
                |i| entities[i].pos,
                reprojection.map(|r| r.camera_motion),
                &mut self.moving,
            )
        } else {
            self.meshes.reset();
            false
        };

        let (Some(frame_seconds), Some(reprojection)) = (frame_seconds, reprojection) else {
            return;
        };
        let world_moved = objects_moved || !reprojection.is_still(&view_proj);
        if !world_moved && !settings.hand {
            return;
        }
        let Some(inverse) = view_proj.inverse() else {
            return;
        };

        let height = world.size[1];
        let tile_size = (settings.max_radius(height).ceil() as u32).clamp(MIN_TILE, MAX_TILE);
        let params = BlurParams {
            view_proj: view_proj.0,
            inverse_view_proj: inverse.0,
            previous_view_proj: reprojection.previous_view_proj.0,
            jittered: world.jittered.to_cols_array(),
            camera_motion: [
                reprojection.camera_motion[0],
                reprojection.camera_motion[1],
                reprojection.camera_motion[2],
                0.0,
            ],
            size: world.size.map(|n| n as f32),
            depth_size: world.depth_size.map(|n| n as f32),
            exposure: settings.exposure(frame_seconds),
            max_radius: settings.max_radius(height).min(tile_size as f32),
            tile_size,
            samples: settings.quality.samples(),
        };
        if !self
            .targets
            .as_ref()
            .is_some_and(|t| t.fits(world.size, world.color_format, tile_size))
        {
            self.targets = Some(Targets::new(
                &self.device,
                world.size,
                world.color_format,
                tile_size,
            ));
        }
        world.queue.write_buffer(&self.params, 0, params.bytes());

        if world_moved {
            self.draw_world_motion(world, objects_moved);
        }
        if settings.hand {
            self.pending = Some(Pending {
                params,
                world_moved,
            });
        } else {
            self.blur(
                world.encoder,
                world.color,
                world.color_view,
                world.color_format,
            );
        }
    }

    fn hand_drawn(&mut self, hand: &mut HandDrawn<'_>, settings: &Settings) {
        if !settings.hand {
            self.hand.reset();
            return;
        }
        let view_proj = Mat4(hand.view_proj.to_cols_array());
        let vertices = hand.hand;
        let previous_view_proj = self.hand.update(
            vertices.len(),
            |i| (vertices[i].pos, vertices[i].uv),
            view_proj,
            &mut self.moving,
        );
        let Some(Pending {
            mut params,
            world_moved,
        }) = self.pending.take()
        else {
            return;
        };
        let hand_moved =
            previous_view_proj != view_proj || self.moving.iter().any(|v| v.position != v.previous);
        if !world_moved && !hand_moved {
            return;
        }

        if let Some(atlas) = hand.atlas.filter(|_| !self.moving.is_empty()) {
            params.view_proj = view_proj.0;
            params.jittered = view_proj.0;
            params.previous_view_proj = previous_view_proj.0;
            hand.queue
                .write_buffer(&self.hand_params, 0, params.bytes());
            self.hand_vertices
                .upload(&self.device, hand.queue, &self.moving);
            self.draw_hand_motion(hand.encoder, atlas, hand.quad_indices, world_moved);
        } else if !world_moved {
            return;
        }
        self.blur(hand.encoder, hand.color, hand.color_view, hand.color_format);
    }

    /// Each pixel's motion: the camera's everywhere, then each moving mob's over it.
    fn draw_world_motion(&mut self, world: &mut WorldDrawn<'_>, objects_moved: bool) {
        let targets = self.targets.as_ref().expect("made for this frame");
        let depth = wgpu::BindingResource::TextureView(world.depth);
        let group = bind(
            &self.device,
            &self.pipelines.camera,
            &[(0, self.params.as_entire_binding()), (1, depth.clone())],
        );
        let mut pass = begin(
            world.encoder,
            "Motion blur camera",
            &targets.motion,
            None,
            None,
        );
        pass.set_pipeline(&self.pipelines.camera);
        pass.set_bind_group(0, &group, &[]);
        pass.draw(0..3, 0..1);
        drop(pass);

        if !objects_moved {
            return;
        }
        self.objects.upload(&self.device, world.queue, &self.moving);
        let group = bind(
            &self.device,
            &self.pipelines.objects,
            &[(0, self.params.as_entire_binding()), (1, depth)],
        );
        let mut pass = begin(
            world.encoder,
            "Motion blur objects",
            &targets.motion,
            None,
            None,
        );
        pass.set_pipeline(&self.pipelines.objects);
        pass.set_bind_group(0, &group, &[]);
        self.objects.draw(&mut pass, world.quad_indices);
    }

    /// The hand's motion over the world's. With nothing moving in the world, the world step
    /// left the motion texture as it was: it is cleared to "still and far away" first.
    fn draw_hand_motion(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        atlas: &wgpu::Texture,
        quad_indices: &wgpu::Buffer,
        world_moved: bool,
    ) {
        let targets = self.targets.as_ref().expect("made for this frame");
        let atlas = atlas.create_view(&Default::default());
        let group = bind(
            &self.device,
            &self.pipelines.hand,
            &[
                (0, self.hand_params.as_entire_binding()),
                (2, wgpu::BindingResource::TextureView(&atlas)),
                (
                    3,
                    wgpu::BindingResource::Sampler(&self.pipelines.atlas_sampler),
                ),
            ],
        );
        let still = wgpu::Color {
            r: 0.0,
            g: 0.0,
            b: 1e6,
            a: 0.0,
        };
        let clear = (!world_moved).then_some(still);
        let mut pass = begin(
            encoder,
            "Motion blur hand",
            &targets.motion,
            clear,
            Some(&targets.hand_depth),
        );
        pass.set_pipeline(&self.pipelines.hand);
        pass.set_bind_group(0, &group, &[]);
        self.hand_vertices.draw(&mut pass, quad_indices);
    }

    /// Tiles, then the blur itself into the frame.
    fn blur(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        color: &wgpu::Texture,
        color_view: &wgpu::TextureView,
        color_format: wgpu::TextureFormat,
    ) {
        self.pipelines
            .prepare_reconstruct(&self.device, color_format);
        let targets = self.targets.as_ref().expect("made for this frame");
        encoder.copy_texture_to_texture(
            color.as_image_copy(),
            targets.scene.as_image_copy(),
            color.size(),
        );

        let params = || (0, self.params.as_entire_binding());
        let rows = bind(
            &self.device,
            &self.pipelines.tile_rows,
            &[
                params(),
                (4, wgpu::BindingResource::TextureView(&targets.motion)),
            ],
        );
        fullscreen(
            encoder,
            "Motion blur tile rows",
            &targets.tile_rows,
            &self.pipelines.tile_rows,
            &rows,
        );
        let columns = bind(
            &self.device,
            &self.pipelines.tile_columns,
            &[
                params(),
                (5, wgpu::BindingResource::TextureView(&targets.tile_rows)),
            ],
        );
        fullscreen(
            encoder,
            "Motion blur tiles",
            &targets.tiles,
            &self.pipelines.tile_columns,
            &columns,
        );

        let pipeline = self.pipelines.reconstruct();
        let group = bind(
            &self.device,
            pipeline,
            &[
                params(),
                (4, wgpu::BindingResource::TextureView(&targets.motion)),
                (6, wgpu::BindingResource::TextureView(&targets.tiles)),
                (7, wgpu::BindingResource::TextureView(&targets.scene_view)),
                (
                    8,
                    wgpu::BindingResource::Sampler(&self.pipelines.smooth_sampler),
                ),
            ],
        );
        fullscreen(encoder, "Motion blur", color_view, pipeline, &group);
    }
}

/// A vertex buffer that grows to fit and is reused.
#[derive(Default)]
struct VertexBuffer {
    buffer: Option<wgpu::Buffer>,
    count: u32,
}

impl VertexBuffer {
    fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, vertices: &[MotionVertex]) {
        let bytes = MotionVertex::bytes(vertices);
        self.count = vertices.len() as u32;
        if bytes.is_empty() {
            return;
        }
        if self
            .buffer
            .as_ref()
            .is_none_or(|b| b.size() < bytes.len() as u64)
        {
            self.buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Motion blur vertices"),
                size: (bytes.len() as u64).next_power_of_two(),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        queue.write_buffer(self.buffer.as_ref().expect("made above"), 0, bytes);
    }

    /// Draws the uploaded quads with the game's shared quad indices.
    fn draw(&self, pass: &mut wgpu::RenderPass<'_>, quad_indices: &wgpu::Buffer) {
        let Some(buffer) = self.buffer.as_ref().filter(|_| self.count >= 4) else {
            return;
        };
        pass.set_vertex_buffer(0, buffer.slice(..));
        pass.set_index_buffer(quad_indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.count / 4 * 6, 0, 0..1);
    }
}

fn uniform_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Motion blur params"),
        size: size_of::<BlurParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn bind(
    device: &wgpu::Device,
    pipeline: &wgpu::RenderPipeline,
    resources: &[(u32, wgpu::BindingResource<'_>)],
) -> wgpu::BindGroup {
    let entries: Vec<_> = resources
        .iter()
        .map(|(binding, resource)| wgpu::BindGroupEntry {
            binding: *binding,
            resource: resource.clone(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Motion blur"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    })
}

/// A pass into `target`: cleared to `clear`, or drawn over what it holds.
fn begin<'e>(
    encoder: &'e mut wgpu::CommandEncoder,
    label: &str,
    target: &wgpu::TextureView,
    clear: Option<wgpu::Color>,
    depth: Option<&wgpu::TextureView>,
) -> wgpu::RenderPass<'e> {
    let color = [Some(wgpu::RenderPassColorAttachment {
        view: target,
        resolve_target: None,
        depth_slice: None,
        ops: wgpu::Operations {
            load: clear.map_or(wgpu::LoadOp::Load, wgpu::LoadOp::Clear),
            store: wgpu::StoreOp::Store,
        },
    })];
    let depth = depth.map(|view| wgpu::RenderPassDepthStencilAttachment {
        view,
        depth_ops: Some(wgpu::Operations {
            load: wgpu::LoadOp::Clear(0.0),
            store: wgpu::StoreOp::Discard,
        }),
        stencil_ops: None,
    });
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &color,
        depth_stencil_attachment: depth,
        ..Default::default()
    })
}

fn fullscreen(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    target: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    group: &wgpu::BindGroup,
) {
    let mut pass = begin(encoder, label, target, None, None);
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, group, &[]);
    pass.draw(0..3, 0..1);
}
