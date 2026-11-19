//! The mod's page in Options > Mods, and the copy of it the render thread reads each frame.

use mod_api::{ClientHost, Registrar, SettingSpec};
use motion_blur_core::Settings;
use std::sync::{PoisonError, RwLock};

static CURRENT: RwLock<Settings> = RwLock::new(Settings::DEFAULT);

const ENABLED: &str = "enabled";
const STRENGTH: &str = "strength";

pub fn register(registrar: &mut dyn Registrar) {
    let specs = [
        SettingSpec::toggle(
            ENABLED,
            "motion_blur.setting.enabled",
            Settings::DEFAULT.enabled,
        ),
        SettingSpec::range(
            STRENGTH,
            "motion_blur.setting.strength",
            0.0,
            200.0,
            5.0,
            100.0,
        )
        .tooltip("motion_blur.setting.strength.tooltip"),
    ];
    for spec in specs {
        let key = spec.key.clone();
        if let Err(error) = registrar.register_setting(spec) {
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
    };
    *CURRENT.write().unwrap_or_else(PoisonError::into_inner) = settings;
}

pub fn current() -> Settings {
    *CURRENT.read().unwrap_or_else(PoisonError::into_inner)
}
