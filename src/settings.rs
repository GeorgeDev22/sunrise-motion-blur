//! The mod's page in Options > Mods, and the copy of it the render thread reads each frame.

use mod_api::{ClientHost, Registrar, SettingSpec};
use motion_blur_core::{Mode, Quality, Settings};
use std::sync::{PoisonError, RwLock};

static CURRENT: RwLock<Settings> = RwLock::new(Settings::DEFAULT);

const ENABLED: &str = "enabled";
const STRENGTH: &str = "strength";
const MODE: &str = "mode";
const QUALITY: &str = "quality";
const MAX_LENGTH: &str = "max_length";
const MOVING_OBJECTS: &str = "moving_objects";
const HAND: &str = "hand";

const MODES: [Mode; 2] = [Mode::Consistent, Mode::Cinematic];

fn label(key: &str) -> String {
    format!("motion_blur.setting.{key}")
}

fn tooltip(key: &str) -> String {
    format!("motion_blur.setting.{key}.tooltip")
}

fn option(key: &str, value: &str) -> String {
    format!("motion_blur.setting.{key}.{value}")
}

pub fn register(registrar: &mut dyn Registrar) {
    let defaults = Settings::DEFAULT;
    let specs = [
        SettingSpec::toggle(ENABLED, label(ENABLED), defaults.enabled),
        SettingSpec::range(STRENGTH, label(STRENGTH), 0.0, 200.0, 5.0, 100.0),
        SettingSpec::choice(
            MODE,
            label(MODE),
            [option(MODE, "consistent"), option(MODE, "cinematic")],
            0,
        ),
        SettingSpec::choice(
            QUALITY,
            label(QUALITY),
            ["low", "medium", "high", "ultra"].map(|q| option(QUALITY, q)),
            1,
        ),
        SettingSpec::range(MAX_LENGTH, label(MAX_LENGTH), 1.0, 10.0, 0.5, 4.0),
        SettingSpec::toggle(
            MOVING_OBJECTS,
            label(MOVING_OBJECTS),
            defaults.moving_objects,
        ),
        SettingSpec::toggle(HAND, label(HAND), defaults.hand),
    ];
    for spec in specs {
        let key = spec.key.clone();
        if let Err(error) = registrar.register_setting(spec.tooltip(tooltip(&key))) {
            eprintln!("motion_blur: the {key} setting was refused: {error:?}");
        }
    }
}

/// Reads the player's choices into the copy the renderer uses.
pub fn load(host: &dyn ClientHost) {
    let defaults = Settings::DEFAULT;
    let settings = Settings {
        enabled: host.setting_bool(ENABLED).unwrap_or(defaults.enabled),
        strength: host
            .setting_f64(STRENGTH)
            .map_or(defaults.strength, |percent| (percent / 100.0) as f32),
        mode: host
            .setting_choice(MODE)
            .and_then(|i| MODES.get(i as usize).copied())
            .unwrap_or(defaults.mode),
        quality: host
            .setting_choice(QUALITY)
            .and_then(|i| Quality::ALL.get(i as usize).copied())
            .unwrap_or(defaults.quality),
        max_length: host
            .setting_f64(MAX_LENGTH)
            .map_or(defaults.max_length, |percent| (percent / 100.0) as f32),
        moving_objects: host
            .setting_bool(MOVING_OBJECTS)
            .unwrap_or(defaults.moving_objects),
        hand: host.setting_bool(HAND).unwrap_or(defaults.hand),
    };
    *CURRENT.write().unwrap_or_else(PoisonError::into_inner) = settings;
}

pub fn current() -> Settings {
    *CURRENT.read().unwrap_or_else(PoisonError::into_inner)
}
