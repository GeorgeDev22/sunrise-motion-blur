use crate::Mat4;

/// A camera that jumped further than this in one frame was teleported (a portal, an ender
/// pearl, `/tp`): smearing the whole screen across the jump would look broken.
const TELEPORT_BLOCKS: f64 = 16.0;

/// How to find where a point on screen was last frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reprojection {
    /// Last frame's camera-relative world to clip space.
    pub previous_view_proj: Mat4,
    /// How far the camera moved since last frame, in blocks: a point at camera-relative `p` now
    /// was at `p + camera_motion` relative to last frame's camera.
    pub camera_motion: [f32; 3],
}

impl Reprojection {
    /// Whether the camera neither moved nor turned.
    pub fn is_still(&self, view_proj: &Mat4) -> bool {
        self.camera_motion == [0.0; 3] && self.previous_view_proj == *view_proj
    }
}

/// Last frame's camera.
#[derive(Debug, Default)]
pub struct CameraHistory {
    previous: Option<(Mat4, [f64; 3])>,
}

impl CameraHistory {
    /// This frame's camera: `view_proj` (camera-relative, without jitter) at `position` in the
    /// world. Returns how to reach last frame, or `None` when there is no usable last frame.
    pub fn advance(&mut self, view_proj: Mat4, position: [f64; 3]) -> Option<Reprojection> {
        let (previous_view_proj, previous_position) =
            self.previous.replace((view_proj, position))?;
        let motion = [0, 1, 2].map(|i| position[i] - previous_position[i]);
        let distance = motion.iter().map(|d| d * d).sum::<f64>().sqrt();
        (distance <= TELEPORT_BLOCKS).then(|| Reprojection {
            previous_view_proj,
            camera_motion: motion.map(|d| d as f32),
        })
    }

    pub fn reset(&mut self) {
        self.previous = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_motion_is_this_camera_minus_the_last() {
        let mut history = CameraHistory::default();
        assert_eq!(history.advance(Mat4::IDENTITY, [100.0, 64.0, -20.0]), None);
        let step = history
            .advance(Mat4::IDENTITY, [100.5, 64.0, -21.0])
            .unwrap();
        assert_eq!(step.camera_motion, [0.5, 0.0, -1.0]);
    }

    #[test]
    fn far_from_the_origin_the_motion_keeps_its_precision() {
        let mut history = CameraHistory::default();
        history.advance(Mat4::IDENTITY, [29_999_000.0, 64.0, 0.0]);
        let step = history
            .advance(Mat4::IDENTITY, [29_999_000.001, 64.0, 0.0])
            .unwrap();
        assert!((step.camera_motion[0] - 0.001).abs() < 1e-6);
    }

    #[test]
    fn a_teleport_is_not_blurred() {
        let mut history = CameraHistory::default();
        history.advance(Mat4::IDENTITY, [0.0, 64.0, 0.0]);
        assert_eq!(history.advance(Mat4::IDENTITY, [500.0, 64.0, 0.0]), None);
        assert!(
            history
                .advance(Mat4::IDENTITY, [500.2, 64.0, 0.0])
                .is_some()
        );
    }

    #[test]
    fn a_camera_that_stays_put_is_still() {
        let mut history = CameraHistory::default();
        history.advance(Mat4::IDENTITY, [1.0, 2.0, 3.0]);
        let step = history.advance(Mat4::IDENTITY, [1.0, 2.0, 3.0]).unwrap();
        assert!(step.is_still(&Mat4::IDENTITY));
    }
}
