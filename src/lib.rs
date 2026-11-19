//! Motion blur for Sunrise: every pixel is smeared along the way it moved since the last frame,
//! by the camera's motion and, for mobs, players, dropped items and the hand, by their own.
//!
//! The game calls into `render` through three render functions it lets native mods take over
//! (the `[[native.mixin]]` rows in `sunrise-mod.toml`, whose helpers are in `src/mixins/`).
//! What doesn't need the game, the shader included, lives in `motion-blur-core` (`src/core/`),
//! where it is tested.

mod render;
mod settings;

pub use render::{hand_drawn, render_needs, world_drawn};

use mod_api::{
    ClientHost, ClientLevelLoadEvent, Hooks, Registrar, RegistrationContext, SettingChangedEvent,
};

#[derive(Default)]
struct MotionBlur;

#[mod_api::sunrise_mod]
impl Hooks for MotionBlur {
    fn register(&self, registrar: &mut dyn Registrar, _: &RegistrationContext) {
        settings::register(registrar);
    }

    /// Settings can be read once a world is open: read them each time one opens.
    fn client_level_load(&self, host: &mut dyn ClientHost, _: &mut ClientLevelLoadEvent<'_>) {
        settings::load(&*host);
    }

    fn setting_changed(&self, host: &mut dyn ClientHost, _: &mut SettingChangedEvent<'_>) {
        settings::load(&*host);
    }
}
