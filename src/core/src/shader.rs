//! The blur shader, in WGSL. A mod package carries only Rust and TOML under `src/`, so it lives
//! in a string; `tests/shader.rs` compiles it with the game's shader compiler.

pub const SHADER: &str = r#"
// Per-pixel motion blur, after McGuire et al., "A Reconstruction Filter for Plausible Motion
// Blur" (2012), with Jimenez's alternating sample directions & temporal IGN (2014) and
// Guertin's tile-boundary jitter (2014).
//
// 1. camera_motion / object_motion / hand_motion: packs each pixel's blur radius (rg, 10-bit
//    each) and log-encoded linear view depth (b, 10-bit) into Rgb10a2Unorm (4 bytes/pixel).
// 2. tile_max_rows, tile_max_columns, neighbor_max: the longest radius in each tile_size² tile
//    and its 3×3 neighbourhood (xy), plus the velocity spread across that neighbourhood (z).
// 3. reconstruct: on uniform-velocity tiles (sky, terrain turns with no silhouette/hand/mob),
//    gathers symmetric pairs directly from `scene` with zero motion fetches; on mixed tiles,
//    runs full depth-aware McGuire/Jimenez reconstruction.

struct Params {
    view_proj: mat4x4<f32>,
    reproject: mat4x4<f32>,
    previous_view_proj: mat4x4<f32>,
    jittered: mat4x4<f32>,
    inv_view_proj_row3: vec4<f32>,
    size: vec2<f32>,
    depth_size: vec2<f32>,
    exposure: f32,
    max_radius: f32,
    tile_size: u32,
    samples: u32,
    frame: u32,
    linearize: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
// The world's depth, reversed (0 is the sky). Bound as unfilterable float: it is only loaded.
@group(0) @binding(1) var world_depth: texture_2d<f32>;
@group(0) @binding(2) var atlas: texture_2d<f32>;
@group(0) @binding(3) var atlas_sampler: sampler;
@group(0) @binding(4) var motion: texture_2d<f32>;
@group(0) @binding(5) var tile_rows: texture_2d<f32>;
@group(0) @binding(6) var tiles: texture_2d<f32>;
@group(0) @binding(7) var scene: texture_2d<f32>;
@group(0) @binding(8) var smooth_sampler: sampler;

// Further than anything in the world: the sky.
const SKY: f32 = 1e6;

// Rgb10a2Unorm packing: 0.0 velocity maps exactly to 512/1023, ±max_radius to 1/1023 and 1.0.
// Depth 0..1023 blocks is log2-encoded across 0..1022/1023; sky maps to 1.0.
fn pack_motion(radius: vec2<f32>, range: f32) -> vec4<f32> {
    let inv_max = 1.0 / max(params.max_radius, 1e-3);
    let vel = clamp(radius * inv_max, vec2(-1.0), vec2(1.0)) * (511.0 / 1023.0) + (512.0 / 1023.0);
    let depth = select(clamp(log2(1.0 + max(range, 0.0)) * 0.1, 0.0, 1022.0 / 1023.0), 1.0, range >= 1023.0);
    return vec4(vel, depth, 1.0);
}

fn unpack_vel(packed_xy: vec2<f32>) -> vec2<f32> {
    return ((packed_xy * 1023.0 - 512.0) / 511.0) * params.max_radius;
}

fn unpack_depth(packed_z: f32) -> f32 {
    return select(exp2(packed_z * 10.0) - 1.0, SKY, packed_z >= 1022.5 / 1023.0);
}

@vertex fn fullscreen(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let corner = vec2(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}

fn to_pixel(clip: vec4<f32>) -> vec2<f32> {
    let ndc = clip.xy / clip.w;
    return vec2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * params.size;
}

// A frame's motion in pixels, as the blur radius: half the streak the shutter would see,
// no longer than the longest the player allows.
fn blur_radius(motion_px: vec2<f32>) -> vec2<f32> {
    let radius = motion_px * (0.5 * params.exposure);
    let size = length(radius);
    if size > params.max_radius {
        return radius * (params.max_radius / size);
    }
    return radius;
}

fn depth_at(pixel: vec2<f32>) -> f32 {
    let texel = vec2<i32>(pixel * params.depth_size / params.size);
    let last = vec2<i32>(params.depth_size) - 1;
    return textureLoad(world_depth, clamp(texel, vec2(0), last), 0).r;
}

// 1. Motion

@fragment fn camera_motion(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = frag.xy;
    let ndc = vec2(pixel.x / params.size.x * 2.0 - 1.0, 1.0 - pixel.y / params.size.y * 2.0);
    let depth = depth_at(pixel);

    // Single pre-multiplied matrix handles both world geometry (depth > 0) and sky (depth == 0).
    let clip_in = vec4(ndc, depth, 1.0);
    let previous = params.reproject * clip_in;

    var range = SKY;
    if depth > 0.0 {
        let inv_w = dot(params.inv_view_proj_row3, clip_in);
        range = 1.0 / max(inv_w, 1e-6);
    }
    if previous.w <= 0.0 {
        // Behind last frame's camera: no meaningful streak.
        return pack_motion(vec2(0.0), range);
    }
    return pack_motion(blur_radius(pixel - to_pixel(previous)), range);
}

struct Moving {
    @builtin(position) clip: vec4<f32>,
    @location(0) current: vec4<f32>,
    @location(1) previous: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) range: f32,
};

@vertex fn moving(
    @location(0) position: vec3<f32>,
    @location(1) previous: vec3<f32>,
    @location(2) uv: vec2<f32>,
) -> Moving {
    var out: Moving;
    out.clip = params.jittered * vec4(position, 1.0);
    out.current = params.view_proj * vec4(position, 1.0);
    out.previous = params.previous_view_proj * vec4(previous, 1.0);
    out.uv = uv;
    // Matches `1.0 / inv_w` in `camera_motion` (linear view-space depth in blocks).
    out.range = max(out.current.w, 0.0);
    return out;
}

fn moving_radius(in: Moving) -> vec2<f32> {
    if in.previous.w <= 0.0 {
        return vec2(0.0);
    }
    return blur_radius(to_pixel(in.current) - to_pixel(in.previous));
}

// A mob's own motion, kept only where the world shows it: where the world's depth is this
// surface's. Behind a wall, or at a cut-out texel, the depth belongs to something else.
@fragment fn object_motion(in: Moving) -> @location(0) vec4<f32> {
    let stored = depth_at(in.clip.xy);
    if abs(in.clip.z - stored) > stored * 2e-3 + 1e-7 {
        discard;
    }
    return pack_motion(moving_radius(in), in.range);
}

// The hand is drawn over everything: it covers the world wherever its texture shows.
@fragment fn hand_motion(in: Moving) -> @location(0) vec4<f32> {
    if textureSampleLevel(atlas, atlas_sampler, in.uv, 0.0).a < 0.1 {
        discard;
    }
    return pack_motion(moving_radius(in), 0.0);
}

// 2. Tiles

fn longer(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return select(a, b, dot(b, b) > dot(a, a));
}

@fragment fn tile_max_rows(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let width = i32(params.size.x);
    let first = i32(frag.x) * i32(params.tile_size);
    let end = min(first + i32(params.tile_size), width);
    let y = i32(frag.y);
    var best = vec2(0.0);
    var min_v = vec2(1e6);
    var max_v = vec2(-1e6);
    for (var x = first; x < end; x++) {
        let v = unpack_vel(textureLoad(motion, vec2(x, y), 0).xy);
        best = longer(best, v);
        min_v = min(min_v, v);
        max_v = max(max_v, v);
    }
    let spread = select(0.0, length(max_v - min_v), first < end);
    return vec4(best, spread, 0.0);
}

@fragment fn tile_max_columns(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let height = i32(params.size.y);
    let first = i32(frag.y) * i32(params.tile_size);
    let end = min(first + i32(params.tile_size), height);
    let x = i32(frag.x);
    var best = vec2(0.0);
    var max_spread = 0.0;
    var min_b = vec2(1e6);
    var max_b = vec2(-1e6);
    for (var y = first; y < end; y++) {
        let row = textureLoad(tile_rows, vec2(x, y), 0);
        best = longer(best, row.xy);
        max_spread = max(max_spread, row.z);
        min_b = min(min_b, row.xy);
        max_b = max(max_b, row.xy);
    }
    let spread = select(0.0, max(max_spread, length(max_b - min_b)), first < end);
    return vec4(best, spread, 0.0);
}

// 3×3 neighbourhood max + velocity spread, computed once per tile instead of per full-res pixel.
@fragment fn neighbor_max(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let tile = vec2<i32>(frag.xy);
    let last = vec2<i32>(textureDimensions(tiles)) - 1;
    var best = vec2(0.0);
    var max_spread = 0.0;
    var min_b = vec2(1e6);
    var max_b = vec2(-1e6);
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let at = clamp(tile + vec2(dx, dy), vec2(0), last);
            let t = textureLoad(tiles, at, 0);
            best = longer(best, t.xy);
            max_spread = max(max_spread, t.z);
            min_b = min(min_b, t.xy);
            max_b = max(max_b, t.xy);
        }
    }
    return vec4(best, max(max_spread, length(max_b - min_b)), 0.0);
}

// 3. Reconstruction

// Jorge Jimenez's interleaved gradient noise, temporally cycled so the dither disappears across
// frames instead of freezing as a static grain.
fn noise(pixel: vec2<f32>, frame: u32) -> f32 {
    let p = pixel + vec2(5.588238 * f32(frame & 63u));
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

// 1 where a lies in front of b, fading out over a short distance behind it.
fn in_front(a: f32, b: f32) -> f32 {
    return clamp(1.0 - (a - b) / (0.1 + 0.02 * min(a, b)), 0.0, 1.0);
}

// Whether a pixel blurred by `radius` reaches `gap` pixels away: fully at its centre, not at all
// at its end.
fn cone(gap: f32, inv_radius: f32) -> f32 {
    return clamp(1.0 - gap * inv_radius, 0.0, 1.0);
}

// Whether two pixels blurred by the same motion overlap: their streaks share the span.
fn cylinder(gap: f32, radius: f32) -> f32 {
    return 1.0 - smoothstep(0.95 * radius, 1.05 * radius, gap);
}

fn sample_weight(
    other_z: f32,
    other_len: f32,
    own_z: f32,
    inv_own_len: f32,
    gap: f32,
    own_cyl: f32,
) -> f32 {
    let inv_other_len = 1.0 / max(other_len, 1e-3);
    return in_front(other_z, own_z) * cone(gap, inv_other_len)
        + in_front(own_z, other_z) * cone(gap, inv_own_len)
        + cylinder(gap, other_len) * own_cyl * 2.0;
}

// Approximate linear-light conversion when the render target is non-sRGB Unorm, so bright
// highlights (clouds, torches, sky edges) don't darken when blurred.
fn to_linear(c: vec3<f32>) -> vec3<f32> {
    return select(c, c * c, params.linearize != 0u);
}

fn from_linear(c: vec3<f32>) -> vec3<f32> {
    return select(c, sqrt(max(c, vec3(0.0))), params.linearize != 0u);
}

@fragment fn reconstruct(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = frag.xy;
    let inv_size = 1.0 / params.size;
    let centre = textureSampleLevel(scene, smooth_sampler, pixel * inv_size, 0.0);

    // Jitter the tile lookup slightly (Guertin 2014) to hide tile-boundary seams, with 1 load.
    let n = noise(pixel, params.frame);
    let n2 = fract(n * 1.6180339 + 0.5);
    let tile_f = pixel / f32(params.tile_size) + (vec2(n, n2) - 0.5) * 0.5;
    let last_tile = vec2<i32>(textureDimensions(tiles)) - 1;
    let tile_info = textureLoad(tiles, clamp(vec2<i32>(tile_f), vec2(0), last_tile), 0);
    let reach = tile_info.xy;
    let reach_length = length(reach);
    // Nothing nearby moves more than half a pixel: most of a still screen stops here.
    if reach_length < 0.5 {
        return centre;
    }

    // Smooth fade-in across 0.5..1.5 px so starting/stopping a slow pan never pops.
    let fade = smoothstep(0.5, 1.5, reach_length);

    // Scale tap pairs with blur radius (uniform within a tile -> no warp divergence).
    let max_pairs = max(params.samples / 2u, 1u);
    let pairs = clamp(u32(ceil(reach_length * 0.25)), 2u, max_pairs);
    let inv_pairs = 1.0 / f32(pairs);

    let own_packed = textureLoad(motion, vec2<i32>(pixel), 0);
    let own_vel = unpack_vel(own_packed.xy);
    let centre_lin = to_linear(centre.rgb);

    // Fast path: when every pixel in the 3×3 tile neighbourhood moves with nearly the same
    // velocity (no hand, moving mob, or strong parallax edge), gather directly from `scene`
    // with a soft shutter taper and zero motion fetches inside the loop.
    let spread = tile_info.z;
    if spread < max(1.0, reach_length * 0.25) {
        var sum = centre_lin;
        var total = 1.0;
        for (var i = 0u; i < pairs; i++) {
            let t = (f32(i) + n) * inv_pairs;
            // Gentle parabolic shutter falloff: softer streak tips, less box-filter ringing.
            let w = 1.0 - 0.35 * t * t;
            let offset = own_vel * t;
            let at_pos = pixel + offset;
            if all(at_pos >= vec2(0.0)) && all(at_pos < params.size) {
                sum += to_linear(textureSampleLevel(scene, smooth_sampler, at_pos * inv_size, 0.0).rgb) * w;
                total += w;
            }
            let at_neg = pixel - offset;
            if all(at_neg >= vec2(0.0)) && all(at_neg < params.size) {
                sum += to_linear(textureSampleLevel(scene, smooth_sampler, at_neg * inv_size, 0.0).rgb) * w;
                total += w;
            }
        }
        let blurred = from_linear(sum / total);
        return vec4(mix(centre.rgb, blurred, fade), centre.a);
    }

    // Full depth-aware McGuire/Jimenez reconstruction for silhouettes, mobs, parallax, and hand.
    let last = vec2<i32>(params.size) - 1;
    let own_z = unpack_depth(own_packed.z);
    let own_length = length(own_vel);
    let inv_own_len = 1.0 / max(own_length, 1e-3);

    var own_direction = reach;
    if own_length > 0.5 {
        own_direction = own_vel * (reach_length / own_length);
    }

    let center_weight = 1.0 / max(own_length, 0.5);
    var sum = centre_lin * center_weight;
    var total = center_weight;

    for (var i = 0u; i < pairs; i++) {
        let t = (f32(i) + n) * inv_pairs;
        let direction = select(reach, own_direction, (i & 1u) == 1u);
        let offset = direction * t;
        let gap = t * reach_length;
        let own_cyl = cylinder(gap, own_length);

        let at_pos = pixel + offset;
        if all(at_pos >= vec2(0.0)) && all(at_pos < params.size) {
            let other = textureLoad(motion, clamp(vec2<i32>(at_pos), vec2(0), last), 0);
            let other_len = length(unpack_vel(other.xy));
            let w = sample_weight(unpack_depth(other.z), other_len, own_z, inv_own_len, gap, own_cyl);
            sum += to_linear(textureSampleLevel(scene, smooth_sampler, at_pos * inv_size, 0.0).rgb) * w;
            total += w;
        }

        let at_neg = pixel - offset;
        if all(at_neg >= vec2(0.0)) && all(at_neg < params.size) {
            let other = textureLoad(motion, clamp(vec2<i32>(at_neg), vec2(0), last), 0);
            let other_len = length(unpack_vel(other.xy));
            let w = sample_weight(unpack_depth(other.z), other_len, own_z, inv_own_len, gap, own_cyl);
            sum += to_linear(textureSampleLevel(scene, smooth_sampler, at_neg * inv_size, 0.0).rgb) * w;
            total += w;
        }
    }
    let blurred = from_linear(sum / total);
    return vec4(mix(centre.rgb, blurred, fade), centre.a);
}
"#;
