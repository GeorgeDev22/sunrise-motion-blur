/// How the blur's length follows the frame rate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// The same streak for the same motion at any frame rate: the shutter stays open for a fixed
    /// time (a 180° shutter at 60 fps, 1/120 s, at full strength).
    Consistent,
    /// A film camera at the game's own frame rate: the shutter is open for half of each frame,
    /// so lower frame rates get longer streaks that hide the gaps between frames.
    Cinematic,
}

/// How many samples each blurred pixel takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quality {
    Low,
    Medium,
    High,
    Ultra,
}

impl Quality {
    pub const ALL: [Quality; 4] = [Quality::Low, Quality::Medium, Quality::High, Quality::Ultra];

    pub fn samples(self) -> u32 {
        match self {
            Quality::Low => 6,
            Quality::Medium => 10,
            Quality::High => 16,
            Quality::Ultra => 24,
        }
    }
}

/// The player's choices, as the mod's settings page keeps them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settings {
    pub enabled: bool,
    /// 1.0 is the default shutter; 0 turns the blur off, 2.0 doubles it.
    pub strength: f32,
    pub mode: Mode,
    pub quality: Quality,
    /// The longest streak, as a share of the screen's height (0.04 is 4%).
    pub max_length: f32,
    /// Blur mobs, players and dropped items by their own motion, not only the camera's.
    pub moving_objects: bool,
    /// Blur the first-person hand and held item by their swing and bob.
    pub hand: bool,
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
        mode: Mode::Consistent,
        quality: Quality::Medium,
        max_length: 0.04,
        moving_objects: true,
        hand: false,
    };

    /// Whether there is any blur to draw.
    pub fn active(&self) -> bool {
        self.enabled && self.strength > 0.0
    }

    /// How far a pixel's motion this frame is stretched into a streak, given the time since the
    /// last frame: the share of the frame the shutter is open for.
    pub fn exposure(&self, frame_seconds: f32) -> f32 {
        let open = match self.mode {
            Mode::Consistent => (1.0 / 120.0) / frame_seconds,
            Mode::Cinematic => 0.5,
        };
        open * self.strength
    }

    /// The longest blur radius (half a streak) in pixels, on a screen `height` pixels tall.
    pub fn max_radius(&self, height: u32) -> f32 {
        0.5 * self.max_length * height as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consistent_mode_gives_the_same_streak_at_any_frame_rate() {
        let settings = Settings::default();
        // A pan of 1200 px/s: per frame that is 40 px at 30 fps and 5 px at 240 fps.
        let at_30 = 40.0 * settings.exposure(1.0 / 30.0);
        let at_240 = 5.0 * settings.exposure(1.0 / 240.0);
        assert!((at_30 - 10.0).abs() < 1e-4, "{at_30}");
        assert!((at_240 - 10.0).abs() < 1e-4, "{at_240}");
    }

    #[test]
    fn cinematic_mode_keeps_half_of_each_frame() {
        let settings = Settings {
            mode: Mode::Cinematic,
            strength: 2.0,
            ..Settings::default()
        };
        assert_eq!(settings.exposure(1.0 / 30.0), 1.0);
        assert_eq!(settings.exposure(1.0 / 144.0), 1.0);
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
