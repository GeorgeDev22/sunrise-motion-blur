fn world_drawn(world: &mut WorldDrawn<'_>) {
    let entities = world.entities;
    let depth = world.depth;
    let quad_indices = world.quad_indices;
    mod_motion_blur::world_drawn(mod_motion_blur::World {
        view_proj: world.view_proj.to_cols_array(),
        jittered: world.jittered.to_cols_array(),
        camera: [world.camera.x, world.camera.y, world.camera.z],
        size: world.size,
        depth_size: world.depth_size,
        motion: world.motion,
        entities: entities.len(),
        entity_position: &|i| entities[i].pos,
        keep: &|| -> (mod_motion_blur::Kept, mod_motion_blur::Kept) {
            (Box::new(depth.clone()), Box::new(quad_indices.clone()))
        },
    });
}
