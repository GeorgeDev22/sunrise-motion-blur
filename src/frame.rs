//! What each frame's blur needs, decided on the CPU.
//!
//! Once the world is drawn (`world_drawn`) the mod works out how the camera and each mob moved.
//! Once the hand is drawn over it (`work`), the hand's motion joins in and the GPU half
//! (`src/mixins/hand_drawn.rs`) is handed a [`Work`] to draw. The HUD comes after, so it stays
//! sharp. When nothing on screen moves there is no `Work`, and the GPU does nothing.

use crate::settings;
use internals::sunrise_mod_render::render::{HandDrawn, RenderNeeds, WorldDrawn};
use motion_blur_core::{
    BlurParams, CameraHistory, FrameClock, HandHistory, Mat4, MeshHistory, MotionVertex,
};
use std::any::Any;
use std::sync::{LazyLock, Mutex, MutexGuard, PoisonError};
use std::time::Instant;

/// The tile size bounds: the longest radius the blur reaches is one tile.
const MIN_TILE: u32 = 8;
const MAX_TILE: u32 = 64;

static FRAMES: LazyLock<Mutex<Frames>> = LazyLock::new(Default::default);

/// One frame's blur, for the GPU half to draw.
pub struct Work {
    pub params: BlurParams,
    /// The same frame seen with the hand's camera, for drawing the hand's motion.
    pub hand_params: BlurParams,
    /// The vertices of mobs that moved by themselves (whole quads); empty when none did.
    pub objects: Vec<MotionVertex>,
    /// The hand's vertices: with its motion when the hand is blurred, still when it isn't, so
    /// that the blur keeps it sharp. Empty in third person.
    pub hand: Vec<MotionVertex>,
    /// Whether anything in the world moved: if not, only the hand did.
    pub world_moved: bool,
    /// The world's depth view and its quad index buffer, kept from `world_drawn`.
    pub depth: Box<dyn Any + Send + Sync>,
    pub quad_indices: Box<dyn Any + Send + Sync>,
}

/// What the world step leaves for the hand step of the same frame.
struct Pending {
    params: BlurParams,
    world_moved: bool,
    depth: Box<dyn Any + Send + Sync>,
    quad_indices: Box<dyn Any + Send + Sync>,
}

#[derive(Default)]
struct Frames {
    clock: FrameClock,
    camera: CameraHistory,
    meshes: MeshHistory,
    hand: HandHistory,
    objects: Vec<MotionVertex>,
    hand_vertices: Vec<MotionVertex>,
    pending: Option<Pending>,
}

impl Frames {
    fn reset(&mut self) {
        self.clock.reset();
        self.camera.reset();
        self.meshes.reset();
        self.hand.reset();
        self.pending = None;
    }
}

fn frames() -> MutexGuard<'static, Frames> {
    FRAMES.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Whether the blur is on; while it isn't, the GPU half lets go of what it made.
pub fn active() -> bool {
    settings::current().active()
}

pub fn render_needs(needs: &mut RenderNeeds) {
    if active() {
        needs.world_depth = true;
    }
}

pub fn world_drawn(world: &mut WorldDrawn<'_>) {
    let settings = settings::current();
    let mut frames = frames();
    let frames = &mut *frames;
    if !settings.active() {
        frames.reset();
        return;
    }
    frames.pending = None;

    let frame_seconds = frames.clock.tick(Instant::now());
    let view_proj = Mat4(world.view_proj.to_cols_array());
    let camera = [world.camera.x, world.camera.y, world.camera.z];
    let reprojection = frames.camera.advance(view_proj, camera);

    let objects_moved = if settings.moving_objects {
        let entities = world.entities;
        frames.meshes.update(
            world.motion,
            entities.len(),
            |i| entities[i].pos,
            reprojection.map(|r| r.camera_motion),
            &mut frames.objects,
        )
    } else {
        frames.meshes.reset();
        frames.objects.clear();
        false
    };

    let (Some(frame_seconds), Some(reprojection)) = (frame_seconds, reprojection) else {
        return;
    };
    let Some(inverse) = view_proj.inverse() else {
        return;
    };
    let world_moved = objects_moved || !reprojection.is_still(&view_proj);
    if !world_moved && !settings.hand {
        return;
    }

    let height = world.size[1];
    let tile_size = (settings.max_radius(height).ceil() as u32).clamp(MIN_TILE, MAX_TILE);
    let [x, y, z] = reprojection.camera_motion;
    frames.pending = Some(Pending {
        params: BlurParams {
            view_proj: view_proj.0,
            inverse_view_proj: inverse.0,
            previous_view_proj: reprojection.previous_view_proj.0,
            jittered: world.jittered.to_cols_array(),
            camera_motion: [x, y, z, 0.0],
            size: world.size.map(|n| n as f32),
            depth_size: world.depth_size.map(|n| n as f32),
            exposure: settings.exposure(frame_seconds),
            max_radius: settings.max_radius(height).min(tile_size as f32),
            tile_size,
            samples: settings.quality.samples(),
        },
        world_moved,
        depth: Box::new(world.depth.clone()),
        quad_indices: Box::new(world.quad_indices.clone()),
    });
}

/// The frame's blur, once the hand is drawn: `None` when nothing on screen moves.
pub fn work(hand: &HandDrawn<'_>) -> Option<Work> {
    let settings = settings::current();
    let mut frames = frames();
    let frames = &mut *frames;
    let pending = frames.pending.take()?;

    // A hand that isn't blurred is still drawn into the motion, standing still, so the blur
    // knows it is in front and leaves it sharp.
    if !settings.hand {
        frames.hand.reset();
    }
    let view_proj = Mat4(hand.view_proj.to_cols_array());
    let vertices = if hand.atlas.is_some() { hand.hand } else { &[] };
    let previous_view_proj = frames.hand.update(
        vertices.len(),
        |i| (vertices[i].pos, vertices[i].uv),
        view_proj,
        &mut frames.hand_vertices,
    );
    let hand_moved = previous_view_proj != view_proj
        || frames
            .hand_vertices
            .iter()
            .any(|v| v.position != v.previous);
    if !pending.world_moved && !hand_moved {
        return None;
    }

    let hand_params = BlurParams {
        view_proj: view_proj.0,
        jittered: view_proj.0,
        previous_view_proj: previous_view_proj.0,
        ..pending.params
    };
    Some(Work {
        params: pending.params,
        hand_params,
        objects: std::mem::take(&mut frames.objects),
        hand: std::mem::take(&mut frames.hand_vertices),
        world_moved: pending.world_moved,
        depth: pending.depth,
        quad_indices: pending.quad_indices,
    })
}

/// Hands a drawn frame's vertex lists back, so the next frame reuses them.
pub fn recycle(work: Work) {
    let mut frames = frames();
    frames.objects = work.objects;
    frames.hand_vertices = work.hand;
}
