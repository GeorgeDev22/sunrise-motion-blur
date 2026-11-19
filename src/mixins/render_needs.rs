fn render_needs(needs: &mut RenderNeeds) {
    // The blur reads the world's depth, and the entities' own geometry to follow their motion.
    if mod_motion_blur::active() {
        needs.world_depth = true;
    }
    if mod_motion_blur::follows_objects() {
        needs.entity_spans = true;
    }
}
