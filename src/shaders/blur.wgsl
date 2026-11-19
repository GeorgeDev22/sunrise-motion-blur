// Per-pixel motion blur, after McGuire et al., "A Reconstruction Filter for Plausible Motion
// Blur" (2012), with Jimenez's alternating sample directions (2014).
//
// 1. camera_motion / object_motion / hand_motion: each pixel's blur radius in pixels (xy) and
//    its distance from the camera (z).
// 2. tile_max_rows, tile_max_columns: the longest radius in each tile_size² tile.
// 3. reconstruct: each pixel gathers samples along the longest radius around it, weighted so
//    that a moving thing smears over what's behind it but never under what's in front.

struct Params {
    view_proj: mat4x4<f32>,
    inverse_view_proj: mat4x4<f32>,
    previous_view_proj: mat4x4<f32>,
    jittered: mat4x4<f32>,
    camera_motion: vec4<f32>,
    size: vec2<f32>,
    depth_size: vec2<f32>,
    exposure: f32,
    max_radius: f32,
    tile_size: u32,
    samples: u32,
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

    var previous: vec4<f32>;
    var range = SKY;
    if depth <= 0.0 {
        // The sky is infinitely far: only the camera's turn moves it.
        let far = params.inverse_view_proj * vec4(ndc, 1e-6, 1.0);
        previous = params.previous_view_proj * vec4(far.xyz / far.w, 0.0);
    } else {
        let point = params.inverse_view_proj * vec4(ndc, depth, 1.0);
        let position = point.xyz / point.w;
        range = length(position);
        previous = params.previous_view_proj * vec4(position + params.camera_motion.xyz, 1.0);
    }
    if previous.w <= 0.0 {
        // Behind last frame's camera: no meaningful streak.
        return vec4(0.0, 0.0, range, 0.0);
    }
    return vec4(blur_radius(pixel - to_pixel(previous)), range, 0.0);
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
    out.range = length(position);
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
    return vec4(moving_radius(in), in.range, 0.0);
}

// The hand is drawn over everything: it covers the world wherever its texture shows.
@fragment fn hand_motion(in: Moving) -> @location(0) vec4<f32> {
    if textureSampleLevel(atlas, atlas_sampler, in.uv, 0.0).a < 0.1 {
        discard;
    }
    return vec4(moving_radius(in), 0.0, 0.0);
}

// 2. Tiles

fn longer(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return select(a, b, dot(b, b) > dot(a, a));
}

@fragment fn tile_max_rows(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let width = i32(params.size.x);
    let first = i32(frag.x) * i32(params.tile_size);
    let y = i32(frag.y);
    var best = vec2(0.0);
    for (var x = first; x < min(first + i32(params.tile_size), width); x++) {
        best = longer(best, textureLoad(motion, vec2(x, y), 0).xy);
    }
    return vec4(best, 0.0, 0.0);
}

@fragment fn tile_max_columns(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let height = i32(params.size.y);
    let first = i32(frag.y) * i32(params.tile_size);
    let x = i32(frag.x);
    var best = vec2(0.0);
    for (var y = first; y < min(first + i32(params.tile_size), height); y++) {
        best = longer(best, textureLoad(tile_rows, vec2(x, y), 0).xy);
    }
    return vec4(best, 0.0, 0.0);
}

// 3. Reconstruction

// The longest radius in this pixel's tile and the eight around it: anything that can reach it.
fn neighbourhood_max(pixel: vec2<f32>) -> vec2<f32> {
    let tile = vec2<i32>(pixel) / i32(params.tile_size);
    let last = vec2<i32>(textureDimensions(tiles)) - 1;
    var best = vec2(0.0);
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let at = clamp(tile + vec2(dx, dy), vec2(0), last);
            best = longer(best, textureLoad(tiles, at, 0).xy);
        }
    }
    return best;
}

// Jorge Jimenez's interleaved gradient noise: offsets each pixel's samples so that a few of
// them read as a smooth streak instead of a ladder of copies.
fn noise(pixel: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(pixel, vec2(0.06711056, 0.00583715))));
}

// 1 where a lies in front of b, fading out over a short distance behind it.
fn in_front(a: f32, b: f32) -> f32 {
    return clamp(1.0 - (a - b) / (0.1 + 0.02 * min(a, b)), 0.0, 1.0);
}

// Whether a pixel blurred by `radius` reaches `gap` pixels away: fully at its centre, not at all
// at its end.
fn cone(gap: f32, radius: f32) -> f32 {
    return clamp(1.0 - gap / max(radius, 1e-3), 0.0, 1.0);
}

// Whether two pixels blurred by the same motion overlap: their streaks share the span.
fn cylinder(gap: f32, radius: f32) -> f32 {
    return 1.0 - smoothstep(0.95 * radius, 1.05 * radius, gap);
}

@fragment fn reconstruct(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = frag.xy;
    let centre = textureSampleLevel(scene, smooth_sampler, pixel / params.size, 0.0);
    let reach = neighbourhood_max(pixel);
    let reach_length = length(reach);
    // Nothing nearby moves more than half a pixel: most of a still screen stops here.
    if reach_length < 0.5 {
        return centre;
    }

    let last = vec2<i32>(params.size) - 1;
    let own = textureLoad(motion, vec2<i32>(pixel), 0);
    let own_length = length(own.xy);
    // Half the samples follow the pixel's own motion, so a fast background behind a slow
    // object, or the other way round, keeps both streaks.
    var own_direction = reach;
    if own_length > 0.5 {
        own_direction = own.xy * (reach_length / own_length);
    }

    let jitter = noise(pixel) - 0.5;
    var sum = centre.rgb / max(own_length, 0.5);
    var total = 1.0 / max(own_length, 0.5);
    for (var i = 0u; i < params.samples; i++) {
        let t = mix(-1.0, 1.0, (f32(i) + 0.5 + jitter) / f32(params.samples));
        let direction = select(reach, own_direction, (i & 1u) == 1u);
        let offset = direction * t;
        let at = pixel + offset;
        let other = textureLoad(motion, clamp(vec2<i32>(at), vec2(0), last), 0);
        let other_length = length(other.xy);
        let gap = length(offset);

        let weight = in_front(other.z, own.z) * cone(gap, other_length)
            + in_front(own.z, other.z) * cone(gap, own_length)
            + cylinder(gap, other_length) * cylinder(gap, own_length) * 2.0;
        sum += textureSampleLevel(scene, smooth_sampler, at / params.size, 0.0).rgb * weight;
        total += weight;
    }
    return vec4(sum / total, centre.a);
}
