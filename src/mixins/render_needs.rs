fn render_needs(needs: &mut RenderNeeds) {
    // The blur reads the world's depth.
    if mod_motion_blur::active() {
        needs.world_depth = true;
    }
}
