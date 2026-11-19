//! Last frame's geometry of what moves on its own, matched with this frame's vertex by vertex.
//!
//! The game draws a mob as the same quads in the same order every frame while its model stays
//! the same, so a mob whose vertex count didn't change gets each vertex's own motion: legs,
//! head and spin included. One whose model changed (armour put on, a baby growing up) is
//! skipped for a frame and the camera's motion covers it.

use crate::{Mat4, MotionVertex};
use std::collections::HashMap;

/// Motion below this (in blocks) is no motion: float noise in the game's own vertex maths.
const STILL: f32 = 1e-4;

/// The game's mobs, players and dropped items, by entity.
#[derive(Debug, Default)]
pub struct MeshHistory {
    previous: HashMap<u64, Vec<[f32; 3]>>,
    current: HashMap<u64, Vec<[f32; 3]>>,
    /// Emptied vectors, kept so a frame allocates nothing once the scene has settled.
    spare: Vec<Vec<[f32; 3]>>,
}

impl MeshHistory {
    /// This frame's entity geometry: `runs` are `(first vertex, entity key)` as the game lists
    /// them (a run lasts until the next; key 0 is something that isn't tracked), `count` the
    /// number of vertices and `position(i)` vertex `i`, camera-relative.
    ///
    /// With `camera_motion` (`None` when there is no last frame), the vertices of each entity
    /// that moved by itself are written to `out` with where they were last frame. Returns
    /// whether any did.
    pub fn update(
        &mut self,
        runs: &[(u32, u64)],
        count: usize,
        position: impl Fn(usize) -> [f32; 3],
        camera_motion: Option<[f32; 3]>,
        out: &mut Vec<MotionVertex>,
    ) -> bool {
        out.clear();
        for (index, &(start, key)) in runs.iter().enumerate() {
            if key == 0 {
                continue;
            }
            let end = runs.get(index + 1).map_or(count, |next| next.0 as usize);
            let spare = &mut self.spare;
            let vertices = self
                .current
                .entry(key)
                .or_insert_with(|| spare.pop().unwrap_or_default());
            vertices.extend((start as usize..end.min(count)).map(&position));
        }

        if let Some(motion) = camera_motion {
            for (key, now) in &self.current {
                let Some(before) = self.previous.get(key) else {
                    continue;
                };
                if before.len() != now.len() || now.len() % 4 != 0 {
                    continue;
                }
                let moved = now
                    .iter()
                    .zip(before)
                    .any(|(p, q)| (0..3).any(|i| (p[i] - q[i] + motion[i]).abs() > STILL));
                if moved {
                    out.extend(
                        now.iter()
                            .zip(before)
                            .map(|(&position, &previous)| MotionVertex {
                                position,
                                previous,
                                uv: [0.0; 2],
                            }),
                    );
                }
            }
        }

        // This frame becomes the last one; the old last frame's vectors go back to the pool.
        std::mem::swap(&mut self.previous, &mut self.current);
        for (_, mut vertices) in self.current.drain() {
            vertices.clear();
            self.spare.push(vertices);
        }
        !out.is_empty()
    }

    pub fn reset(&mut self) {
        for (_, mut vertices) in self.previous.drain() {
            vertices.clear();
            self.spare.push(vertices);
        }
    }
}

/// The first-person hand and the item it holds, which the game draws with a camera of its own.
#[derive(Debug, Default)]
pub struct HandHistory {
    previous: Vec<[f32; 3]>,
    previous_view_proj: Option<Mat4>,
}

impl HandHistory {
    /// This frame's hand: `count` vertices, `vertex(i)` its camera-relative position and atlas
    /// UV, drawn with `view_proj`. Writes every vertex to `out` (the hand covers what lies
    /// behind it, moving or not) and returns last frame's matrix to reproject them with.
    /// A hand whose shape changed (another item) gets no motion this frame.
    pub fn update(
        &mut self,
        count: usize,
        vertex: impl Fn(usize) -> ([f32; 3], [f32; 2]),
        view_proj: Mat4,
        out: &mut Vec<MotionVertex>,
    ) -> Mat4 {
        out.clear();
        let matched = self.previous.len() == count && self.previous_view_proj.is_some();
        out.extend((0..count).map(|i| {
            let (position, uv) = vertex(i);
            let previous = if matched { self.previous[i] } else { position };
            MotionVertex {
                position,
                previous,
                uv,
            }
        }));
        let previous_view_proj = match self.previous_view_proj {
            Some(before) if matched => before,
            _ => view_proj,
        };
        self.previous.clear();
        self.previous.extend(out.iter().map(|v| v.position));
        self.previous_view_proj = Some(view_proj);
        previous_view_proj
    }

    pub fn reset(&mut self) {
        self.previous.clear();
        self.previous_view_proj = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ZOMBIE: u64 = (1 << 32) | 7;
    const ITEM: u64 = (2 << 32) | 9;

    fn quad(x: f32) -> Vec<[f32; 3]> {
        vec![
            [x, 0.0, 0.0],
            [x + 1.0, 0.0, 0.0],
            [x + 1.0, 1.0, 0.0],
            [x, 1.0, 0.0],
        ]
    }

    fn frame(
        history: &mut MeshHistory,
        geometry: &[(u64, Vec<[f32; 3]>)],
        camera_motion: Option<[f32; 3]>,
    ) -> Vec<MotionVertex> {
        let mut runs = Vec::new();
        let mut vertices = Vec::new();
        for (key, quad) in geometry {
            runs.push((vertices.len() as u32, *key));
            vertices.extend_from_slice(quad);
        }
        let mut out = Vec::new();
        history.update(
            &runs,
            vertices.len(),
            |i| vertices[i],
            camera_motion,
            &mut out,
        );
        out
    }

    #[test]
    fn a_walking_mob_brings_its_last_position() {
        let mut history = MeshHistory::default();
        frame(&mut history, &[(ZOMBIE, quad(0.0))], None);
        let out = frame(&mut history, &[(ZOMBIE, quad(0.25))], Some([0.0; 3]));
        assert_eq!(out.len(), 4);
        assert_eq!(out[0].position, [0.25, 0.0, 0.0]);
        assert_eq!(out[0].previous, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn a_mob_that_only_seems_to_move_with_the_camera_is_left_to_the_camera() {
        let mut history = MeshHistory::default();
        frame(&mut history, &[(ZOMBIE, quad(0.0))], None);
        // The camera stepped 0.5 east, so the standing mob is 0.5 further west of it.
        let out = frame(&mut history, &[(ZOMBIE, quad(-0.5))], Some([0.5, 0.0, 0.0]));
        assert!(out.is_empty());
    }

    #[test]
    fn untracked_geometry_and_new_entities_get_no_motion() {
        let mut history = MeshHistory::default();
        frame(&mut history, &[(0, quad(0.0))], None);
        let out = frame(
            &mut history,
            &[(0, quad(1.0)), (ITEM, quad(3.0))],
            Some([0.0; 3]),
        );
        assert!(out.is_empty());
    }

    #[test]
    fn a_mob_whose_model_changed_waits_a_frame() {
        let mut history = MeshHistory::default();
        frame(&mut history, &[(ZOMBIE, quad(0.0))], None);
        let mut armoured = quad(0.1);
        armoured.extend(quad(0.1));
        assert!(frame(&mut history, &[(ZOMBIE, armoured.clone())], Some([0.0; 3])).is_empty());
        let moved: Vec<_> = armoured.iter().map(|p| [p[0] + 0.1, p[1], p[2]]).collect();
        assert_eq!(
            frame(&mut history, &[(ZOMBIE, moved)], Some([0.0; 3])).len(),
            8
        );
    }

    #[test]
    fn the_hand_reprojects_with_last_frames_camera() {
        let mut hand = HandHistory::default();
        let mut out = Vec::new();
        let first = Mat4::IDENTITY;
        let mut second = Mat4::IDENTITY;
        second.0[12] = 0.1;
        let vertex = |x: f32| move |i: usize| ([x + i as f32, 0.0, -1.0], [0.0, 0.0]);

        assert_eq!(hand.update(4, vertex(0.0), first, &mut out), first);
        assert_eq!(out[0].previous, out[0].position);

        assert_eq!(hand.update(4, vertex(0.2), second, &mut out), first);
        assert_eq!(out[0].position, [0.2, 0.0, -1.0]);
        assert_eq!(out[0].previous, [0.0, 0.0, -1.0]);
    }
}
