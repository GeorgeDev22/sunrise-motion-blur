//! Motion blur for Sunrise: every pixel is smeared along the way it moved since the last frame,
//! by the camera's motion and, for mobs, players, dropped items and the hand, by their own.
//!
//! The game hands the mod each frame through three render functions a native mod takes over
//! (the `[[native.mixin]]` rows in `sunrise-mod.toml`, whose helpers are in `src/mixins/`):
//!
//! - `frame` decides what to blur, on the CPU;
//! - `src/mixins/hand_drawn.rs` draws it. It lives in the helper because only code compiled
//!   beside the game's render functions can use the game's own `wgpu`;
//! - what doesn't need the game, the shader included, is in `motion-blur-core` (`src/core/`),
//!   where it is tested.

mod frame;
mod settings;

// For the mixin helpers.
pub use frame::{Hand, Kept, Work, World, active, follows_objects, recycle, work, world_drawn};
pub use motion_blur_core::{BlurParams, MotionVertex, SHADER};

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
