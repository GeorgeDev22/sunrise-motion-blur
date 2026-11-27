//! The mod's page in Options > Mods, and the copy of it the render thread reads each frame.

use mod_api::{Host, Registrar, SettingKey, SettingSpec};
use motion_blur_core::Settings;
use std::sync::{OnceLock, PoisonError, RwLock};

static CURRENT: RwLock<Settings> = RwLock::new(Settings::DEFAULT);

const ENABLED: &str = "enabled";
const STRENGTH: &str = "strength";

/// The handles the game gave each setting when it was registered: reads go through them.
static ENABLED_KEY: OnceLock<SettingKey> = OnceLock::new();
static STRENGTH_KEY: OnceLock<SettingKey> = OnceLock::new();

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
    for (spec, slot) in specs.into_iter().zip([&ENABLED_KEY, &STRENGTH_KEY]) {
        let key = spec.key.clone();
        match registrar.register_setting(spec) {
            Ok(handle) => {
                let _ = slot.set(handle);
            }
            Err(error) => eprintln!("motion_blur: the {key} setting was refused: {error:?}"),
        }
    }
}

/// Reads the player's choices into the copy the renderer uses. A setting that was refused at
/// registration, or can't be read, keeps its default.
pub fn load(host: &dyn Host) {
    let defaults = Settings::DEFAULT;
    let enabled = ENABLED_KEY
        .get()
        .and_then(|&key| host.setting_bool(key).ok());
    let percent = STRENGTH_KEY
        .get()
        .and_then(|&key| host.setting_f64(key).ok());
    let settings = Settings {
        enabled: enabled.unwrap_or(defaults.enabled),
        strength: percent.map_or(defaults.strength, |percent| (percent / 100.0) as f32),
    };
    *CURRENT.write().unwrap_or_else(PoisonError::into_inner) = settings;
}

pub fn current() -> Settings {
    *CURRENT.read().unwrap_or_else(PoisonError::into_inner)
}
