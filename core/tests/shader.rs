//! The blur shader compiles, and its `Params` is laid out as `BlurParams` is.

const SOURCE: &str = include_str!("../../src/shaders/blur.wgsl");

fn module() -> naga::Module {
    naga::front::wgsl::parse_str(SOURCE).unwrap_or_else(|e| panic!("{}", e.emit_to_string(SOURCE)))
}

#[test]
fn the_shader_validates() {
    let module = module();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{}", e.emit_to_string(SOURCE)));
}

#[test]
fn params_match_blur_params() {
    let module = module();
    let params = module
        .types
        .iter()
        .find(|(_, ty)| ty.name.as_deref() == Some("Params"))
        .expect("a Params struct")
        .1;
    let naga::TypeInner::Struct { span, .. } = params.inner else {
        panic!("Params is a struct");
    };
    assert_eq!(
        span as usize,
        std::mem::size_of::<motion_blur_core::BlurParams>()
    );
}
