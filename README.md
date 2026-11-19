# Motion Blur for Sunrise

A native [Sunrise](https://github.com/sunrise-org/sunrise) mod that adds smooth, per-pixel motion blur.

- **The camera and what moves on its own.** Turning and moving blur the world by how far each pixel
  travelled. Mobs, players and dropped items blur by their own motion too, limbs included. The
  first-person hand can blur as it swings and bobs (off by default).
- **The same at any frame rate.** By default the streak a motion leaves doesn't depend on your FPS.
  The *Cinematic* mode behaves like a film camera instead: half of each frame, so it gets longer as the
  frame rate drops.
- **Lightweight.** Off, or with nothing on screen moving, it does no GPU work at all. Where nothing moves
  near a pixel, the blur skips it after a handful of reads. One small dependency (`bytemuck`). The HUD
  and menus are never blurred.

## Settings

**Options > Mods > Motion Blur > Config**

| Setting | Default | |
|---|---|---|
| Motion Blur | On | Off turns the mod off completely. |
| Strength | 100% | Length of the streaks. 100% is a 180° shutter at 60 FPS. |
| Frame Rate | Consistent | *Consistent* or *Cinematic* (see above). |
| Quality | Medium | Samples per blurred pixel: 6, 10, 16 or 24. |
| Longest Streak | 4% | The longest a streak can get, as a share of the screen's height. |
| Moving Mobs | On | Blur mobs by their own motion. |
| Hand | Off | Blur the first-person hand. |

## How it works

The blur follows McGuire et al., *A Reconstruction Filter for Plausible Motion Blur* (2012), with
Jimenez's refinements (2014):

1. **Motion.** Each pixel's position last frame comes from the depth and last frame's camera. Moving
   mobs are drawn again with each vertex's previous position, wherever the world's depth shows them.
2. **Tiles.** The longest motion in each tile, so every pixel knows how far a streak can reach it.
3. **Reconstruction.** Each pixel gathers a few samples along that motion, weighted by depth, so a
   moving thing smears over what is behind it and never over what is in front.

```
src/lib.rs       the mod's hooks: settings
src/render/      the passes, their pipelines and textures
src/mixins/      the three render functions the game hands to the mod
src/core/        timing, camera and mob history, the shader and its data layout: plain Rust, tested
```

The game calls the mod through Sunrise's render functions (`world_drawn`, `hand_drawn`, `render_needs`),
which a native mod takes over with a `[[native.mixin]]` row each.

## Building

You need [`cargo sunrise-mod`](https://sunrise-org.github.io/sunrise/start/install-tool.html).

```sh
cargo test --manifest-path src/core/Cargo.toml   # the logic and the shader
cargo sunrise-mod build                          # target/sunrise-mod/motion_blur.smod
cargo sunrise-mod run --resources <prepared resources> -- --world Test --creative
```

Every push builds the mod on GitHub Actions too, and uploads the `.smod` and a game with the mod in it.

The mod is pinned to one Sunrise build (`[native]` in `sunrise-mod.toml`). `cargo sunrise-mod update`
moves it to the current Nightly.
