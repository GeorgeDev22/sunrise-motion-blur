//! What each frame's blur needs, decided on the CPU.
//!
//! Once the world is drawn (`world_drawn`) the mod works out how the camera and each mob moved.
//! Once the hand is drawn over it (`work`), the hand's motion joins in and the GPU half
//! (`src/mixins/hand_drawn.rs`) is handed a [`Work`] to draw. The HUD comes after, so it stays
//! sharp. When nothing on screen moves there is no `Work`, and the GPU does nothing.
//!
//! The game's frame types can't be named here, only in the mixin helpers, so the helpers read
//! the frame into a [`World`] and a [`Hand`] and keep its GPU objects as [`Kept`] values.

use crate::settings;
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

/// A GPU object of the game's, which only the helpers can name: the world's depth view and its
/// quad index buffer, kept from one helper to the other.
pub type Kept = Box<dyn Any + Send + Sync>;

/// The drawn world, as the `world_drawn` helper reads it.
pub struct World<'a> {
    /// Camera-relative world to clip space, without TAA's jitter, and with it.
    pub view_proj: [f32; 16],
    pub jittered: [f32; 16],
    /// The camera's position in the world.
    pub camera: [f64; 3],
    /// The picture's size and the world depth's, in pixels.
    pub size: [u32; 2],
    pub depth_size: [u32; 2],
    /// Which entity each run of the entity vertices is (the game's `motion` field).
    pub motion: &'a [(u32, u64)],
    /// How many entity vertices there are, and where vertex `i` is, camera-relative.
    pub entities: usize,
    pub entity_position: &'a dyn Fn(usize) -> [f32; 3],
    /// Keeps the depth view and the quad indices, when the frame is blurred.
    pub keep: &'a dyn Fn() -> (Kept, Kept),
}

/// The drawn hand, as the `hand_drawn` helper reads it.
pub struct Hand<'a> {
    /// The hand's own camera-relative world to clip space.
    pub view_proj: [f32; 16],
    /// How many hand vertices there are (0 in third person, or without an atlas to read its
    /// cut-outs from), and vertex `i`'s position and atlas UV.
    pub vertices: usize,
    pub vertex: &'a dyn Fn(usize) -> ([f32; 3], [f32; 2]),
}

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
    pub depth: Kept,
    pub quad_indices: Kept,
}

/// What the world step leaves for the hand step of the same frame.
struct Pending {
    params: BlurParams,
    world_moved: bool,
    depth: Kept,
    quad_indices: Kept,
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

pub fn world_drawn(world: World<'_>) {
    let settings = settings::current();
    let mut frames = frames();
    let frames = &mut *frames;
    if !settings.active() {
        frames.reset();
        return;
    }
    frames.pending = None;

    let frame_seconds = frames.clock.tick(Instant::now());
    let view_proj = Mat4(world.view_proj);
    let reprojection = frames.camera.advance(view_proj, world.camera);

    let objects_moved = if settings.moving_objects {
        frames.meshes.update(
            world.motion,
            world.entities,
            world.entity_position,
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
    let (depth, quad_indices) = (world.keep)();
    frames.pending = Some(Pending {
        params: BlurParams {
            view_proj: view_proj.0,
            inverse_view_proj: inverse.0,
            previous_view_proj: reprojection.previous_view_proj.0,
            jittered: world.jittered,
            camera_motion: [x, y, z, 0.0],
            size: world.size.map(|n| n as f32),
            depth_size: world.depth_size.map(|n| n as f32),
            exposure: settings.exposure(frame_seconds),
            max_radius: settings.max_radius(height).min(tile_size as f32),
            tile_size,
            samples: settings.quality.samples(),
        },
        world_moved,
        depth,
        quad_indices,
    });
}

/// The frame's blur, once the hand is drawn: `None` when nothing on screen moves.
pub fn work(hand: Hand<'_>) -> Option<Work> {
    let settings = settings::current();
    let mut frames = frames();
    let frames = &mut *frames;
    let pending = frames.pending.take()?;

    // A hand that isn't blurred is still drawn into the motion, standing still, so the blur
    // knows it is in front and leaves it sharp.
    if !settings.hand {
        frames.hand.reset();
    }
    let view_proj = Mat4(hand.view_proj);
    let previous_view_proj = frames.hand.update(
        hand.vertices,
        hand.vertex,
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
