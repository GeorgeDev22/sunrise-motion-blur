# Motion Blur for Sunrise

A native [Sunrise](https://github.com/sunrise-org/sunrise) mod that adds smooth, per-pixel motion blur.

- **The camera and what moves on its own.** Turning and moving blur the world by how far each pixel
  travelled. Things the game draws on the CPU blur by their own motion too. Without a shader pack or
  frame generation, Sunrise poses mobs on the GPU, where a mod can't see them: they blur with the
  camera's motion only, as in other motion blur mods. The hand, the HUD and menus stay sharp.
- **The same at any frame rate.** The streak a motion leaves doesn't depend on your FPS.
- **Lightweight.** Off, or with nothing on screen moving, it does no GPU work at all. Where nothing moves
  near a pixel, the blur skips it after a handful of reads. One small dependency (`bytemuck`).

## Settings

**Options > Mods > Motion Blur > Config**

| Setting | Default | |
|---|---|---|
| Motion Blur | On | Off turns the mod off completely. |
| Strength | 100 | How strong the blur is, from 0 to 200. |

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

Build with the tool from the SDK pinned in `sunrise-mod.toml`. The package tracks the API and render hooks
it uses, so Sunrise can rebuild it for compatible game builds. Check a build with
`cargo sunrise-mod check target/sunrise-mod/motion_blur.smod --build <build>`.

## License

MIT, see [LICENSE](LICENSE).
