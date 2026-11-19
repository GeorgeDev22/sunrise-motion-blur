/// Samples each blurred pixel takes: smooth on long streaks, light enough for integrated GPUs.
pub const SAMPLES: u32 = 10;

/// The longest streak, as a share of the screen's height.
const MAX_LENGTH: f32 = 0.04;

/// How long the shutter stays open at full strength, in seconds: a film camera's 180° shutter at
/// 60 fps. A fixed time, so the same motion leaves the same streak at any frame rate.
const SHUTTER_SECONDS: f32 = 1.0 / 120.0;

/// The player's choices, as the mod's settings page keeps them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settings {
    pub enabled: bool,
    /// 1.0 is the default shutter; 0 turns the blur off, 2.0 doubles it.
    pub strength: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl Settings {
    pub const DEFAULT: Settings = Settings {
        enabled: true,
        strength: 1.0,
    };

    /// Whether there is any blur to draw.
    pub fn active(&self) -> bool {
        self.enabled && self.strength > 0.0
    }

    /// How far a pixel's motion this frame is stretched into a streak, given the time since the
    /// last frame: the share of the frame the shutter is open for.
    pub fn exposure(&self, frame_seconds: f32) -> f32 {
        SHUTTER_SECONDS / frame_seconds * self.strength
    }

    /// The longest blur radius (half a streak) in pixels, on a screen `height` pixels tall.
    pub fn max_radius(&self, height: u32) -> f32 {
        0.5 * MAX_LENGTH * height as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_motion_leaves_the_same_streak_at_any_frame_rate() {
        let settings = Settings::default();
        // A pan of 1200 px/s: per frame that is 40 px at 30 fps and 5 px at 240 fps.
        let at_30 = 40.0 * settings.exposure(1.0 / 30.0);
        let at_240 = 5.0 * settings.exposure(1.0 / 240.0);
        assert!((at_30 - 10.0).abs() < 1e-4, "{at_30}");
        assert!((at_240 - 10.0).abs() < 1e-4, "{at_240}");
    }

    #[test]
    fn no_strength_means_no_blur() {
        let settings = Settings {
            strength: 0.0,
            ..Settings::default()
        };
        assert!(!settings.active());
    }
}
