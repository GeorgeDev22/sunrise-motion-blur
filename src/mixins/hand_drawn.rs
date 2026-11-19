fn hand_drawn(hand: &mut HandDrawn<'_>) {
    // The blur's GPU half, run once the world and the hand are drawn and before the HUD.
    //
    // It lives in this helper because only code compiled beside the game's render functions
    // can use the game's own wgpu. What to draw is decided in the mod's crate (`src/frame.rs`)
    // and handed over as a `Work`; this draws it:
    //
    // 1. motion: the camera's from the depth, mobs that moved over it, then the hand standing
    //    still, which keeps it sharp;
    // 2. the longest motion in each tile;
    // 3. the blur itself, from a copy of the frame back into the frame.

    static GPU: std::sync::Mutex<Option<Gpu>> = std::sync::Mutex::new(None);

    let mut gpu = GPU
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if !mod_motion_blur::active() {
        // Turned off: let go of the pipelines and textures.
        *gpu = None;
        return;
    }
    // Without the atlas the hand's cut-out texels can't be told apart: leave it out.
    let vertices = if hand.atlas.is_some() { hand.hand } else { &[] };
    let Some(work) = mod_motion_blur::work(mod_motion_blur::Hand {
        view_proj: hand.view_proj.to_cols_array(),
        vertices: vertices.len(),
        vertex: &|i| (vertices[i].pos, vertices[i].uv),
    }) else {
        return;
    };
    if gpu.as_ref().is_none_or(|gpu| gpu.device != *hand.device) {
        *gpu = Some(Gpu::new(hand.device));
    }
    gpu.as_mut().expect("made above").draw(hand, &work);
    drop(gpu);
    mod_motion_blur::recycle(work);

    /// Blur radius in pixels (xy) and distance from the camera in blocks (z), per pixel.
    const MOTION: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
    /// The longest blur radius in each tile.
    const TILE: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;
    /// The hand's own depth, so its nearer faces win.
    const HAND_DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    struct Gpu {
        device: wgpu::Device,
        module: wgpu::ShaderModule,
        camera: wgpu::RenderPipeline,
        objects: wgpu::RenderPipeline,
        hand: wgpu::RenderPipeline,
        tile_rows: wgpu::RenderPipeline,
        tile_columns: wgpu::RenderPipeline,
        /// Writes into the frame, so it is made for the frame's format.
        reconstruct: Option<(wgpu::TextureFormat, wgpu::RenderPipeline)>,
        atlas_sampler: wgpu::Sampler,
        smooth_sampler: wgpu::Sampler,
        params: wgpu::Buffer,
        hand_params: wgpu::Buffer,
        object_vertices: Option<wgpu::Buffer>,
        hand_vertices: Option<wgpu::Buffer>,
        targets: Option<Targets>,
    }

    impl Gpu {
        fn new(device: &wgpu::Device) -> Self {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Motion blur"),
                source: wgpu::ShaderSource::Wgsl(mod_motion_blur::SHADER.into()),
            });
            let hand_depth = wgpu::DepthStencilState {
                format: HAND_DEPTH,
                depth_write_enabled: Some(true),
                // Reversed depth, as the game draws it: nearer is larger.
                depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                stencil: Default::default(),
                bias: Default::default(),
            };
            Self {
                camera: fullscreen_pipeline(device, &module, "camera_motion", MOTION),
                objects: moving_pipeline(device, &module, "object_motion", None),
                hand: moving_pipeline(device, &module, "hand_motion", Some(hand_depth)),
                tile_rows: fullscreen_pipeline(device, &module, "tile_max_rows", TILE),
                tile_columns: fullscreen_pipeline(device, &module, "tile_max_columns", TILE),
                reconstruct: None,
                atlas_sampler: sampler(device, wgpu::FilterMode::Nearest),
                smooth_sampler: sampler(device, wgpu::FilterMode::Linear),
                params: uniform_buffer(device),
                hand_params: uniform_buffer(device),
                object_vertices: None,
                hand_vertices: None,
                targets: None,
                module,
                device: device.clone(),
            }
        }

        fn draw(&mut self, hand: &mut HandDrawn<'_>, work: &mod_motion_blur::Work) {
            let size = hand.size;
            let tile_size = work.params.tile_size;
            if !self.targets.as_ref().is_some_and(|t| {
                t.size == size && t.format == hand.color_format && t.tile_size == tile_size
            }) {
                self.targets = Some(Targets::new(
                    &self.device,
                    size,
                    hand.color_format,
                    tile_size,
                ));
            }
            if self
                .reconstruct
                .as_ref()
                .is_none_or(|(format, _)| *format != hand.color_format)
            {
                let pipeline = fullscreen_pipeline(
                    &self.device,
                    &self.module,
                    "reconstruct",
                    hand.color_format,
                );
                self.reconstruct = Some((hand.color_format, pipeline));
            }
            let targets = self.targets.as_ref().expect("made above");
            let encoder = &mut *hand.encoder;
            hand.queue
                .write_buffer(&self.params, 0, work.params.bytes());

            // 1. Motion.
            let Some(depth) = work.depth.downcast_ref::<wgpu::TextureView>() else {
                return;
            };
            let depth = wgpu::BindingResource::TextureView(depth);
            let group = bind(
                &self.device,
                &self.camera,
                &[(0, self.params.as_entire_binding()), (1, depth.clone())],
            );
            let mut pass = begin(encoder, "Motion blur camera", &targets.motion, None, None);
            pass.set_pipeline(&self.camera);
            pass.set_bind_group(0, &group, &[]);
            pass.draw(0..3, 0..1);
            drop(pass);

            let indices = work.quad_indices.downcast_ref::<wgpu::Buffer>();
            if let Some(indices) = indices.filter(|_| work.objects.len() >= 4) {
                let vertices = upload(
                    &self.device,
                    hand.queue,
                    &mut self.object_vertices,
                    &work.objects,
                );
                let group = bind(
                    &self.device,
                    &self.objects,
                    &[(0, self.params.as_entire_binding()), (1, depth)],
                );
                let mut pass = begin(encoder, "Motion blur objects", &targets.motion, None, None);
                pass.set_pipeline(&self.objects);
                pass.set_bind_group(0, &group, &[]);
                draw_quads(&mut pass, &vertices, indices, work.objects.len());
            }
            if let Some(atlas) = hand.atlas.filter(|_| work.hand.len() >= 4) {
                hand.queue
                    .write_buffer(&self.hand_params, 0, work.hand_params.bytes());
                let vertices = upload(
                    &self.device,
                    hand.queue,
                    &mut self.hand_vertices,
                    &work.hand,
                );
                let atlas = atlas.create_view(&Default::default());
                let group = bind(
                    &self.device,
                    &self.hand,
                    &[
                        (0, self.hand_params.as_entire_binding()),
                        (2, wgpu::BindingResource::TextureView(&atlas)),
                        (3, wgpu::BindingResource::Sampler(&self.atlas_sampler)),
                    ],
                );
                let mut pass = begin(
                    encoder,
                    "Motion blur hand",
                    &targets.motion,
                    None,
                    Some(&targets.hand_depth),
                );
                pass.set_pipeline(&self.hand);
                pass.set_bind_group(0, &group, &[]);
                draw_quads(&mut pass, &vertices, hand.quad_indices, work.hand.len());
            }

            // 2. Tiles.
            let rows = bind(
                &self.device,
                &self.tile_rows,
                &[
                    (0, self.params.as_entire_binding()),
                    (4, wgpu::BindingResource::TextureView(&targets.motion)),
                ],
            );
            fullscreen(
                encoder,
                "Motion blur tile rows",
                &targets.tile_rows,
                &self.tile_rows,
                &rows,
            );
            let columns = bind(
                &self.device,
                &self.tile_columns,
                &[
                    (0, self.params.as_entire_binding()),
                    (5, wgpu::BindingResource::TextureView(&targets.tile_rows)),
                ],
            );
            fullscreen(
                encoder,
                "Motion blur tiles",
                &targets.tiles,
                &self.tile_columns,
                &columns,
            );

            // 3. The blur, from a copy of the frame into the frame.
            encoder.copy_texture_to_texture(
                hand.color.as_image_copy(),
                targets.scene.as_image_copy(),
                hand.color.size(),
            );
            let reconstruct = &self.reconstruct.as_ref().expect("made above").1;
            let group = bind(
                &self.device,
                reconstruct,
                &[
                    (0, self.params.as_entire_binding()),
                    (4, wgpu::BindingResource::TextureView(&targets.motion)),
                    (6, wgpu::BindingResource::TextureView(&targets.tiles)),
                    (7, wgpu::BindingResource::TextureView(&targets.scene_view)),
                    (8, wgpu::BindingResource::Sampler(&self.smooth_sampler)),
                ],
            );
            fullscreen(encoder, "Motion blur", hand.color_view, reconstruct, &group);
        }
    }

    /// The textures the blur draws into, made again when the frame's size or format changes.
    struct Targets {
        size: [u32; 2],
        format: wgpu::TextureFormat,
        tile_size: u32,
        /// A copy of the frame: the blur reads it while writing the frame.
        scene: wgpu::Texture,
        scene_view: wgpu::TextureView,
        motion: wgpu::TextureView,
        tile_rows: wgpu::TextureView,
        tiles: wgpu::TextureView,
        hand_depth: wgpu::TextureView,
    }

    impl Targets {
        fn new(
            device: &wgpu::Device,
            size: [u32; 2],
            format: wgpu::TextureFormat,
            tile_size: u32,
        ) -> Self {
            let [width, height] = size;
            let tiles_wide = width.div_ceil(tile_size);
            let tiles_high = height.div_ceil(tile_size);
            let drawn =
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
            let copied = wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING;
            let scene = texture(device, "Motion blur scene", size, format, copied);
            let view = |label, size, format, usage| {
                texture(device, label, size, format, usage).create_view(&Default::default())
            };
            Self {
                size,
                format,
                tile_size,
                scene_view: scene.create_view(&Default::default()),
                scene,
                motion: view("Motion blur motion", size, MOTION, drawn),
                tile_rows: view("Motion blur tile rows", [tiles_wide, height], TILE, drawn),
                tiles: view("Motion blur tiles", [tiles_wide, tiles_high], TILE, drawn),
                hand_depth: view(
                    "Motion blur hand depth",
                    size,
                    HAND_DEPTH,
                    wgpu::TextureUsages::RENDER_ATTACHMENT,
                ),
            }
        }
    }

    fn texture(
        device: &wgpu::Device,
        label: &str,
        [width, height]: [u32; 2],
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    }

    fn fullscreen_pipeline(
        device: &wgpu::Device,
        module: &wgpu::ShaderModule,
        entry: &str,
        format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(entry),
            layout: None,
            vertex: wgpu::VertexState {
                module,
                entry_point: Some("fullscreen"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                targets: &[Some(format.into())],
            }),
            multiview_mask: None,
            cache: None,
        })
    }

    /// A pipeline for what moves on its own: `MotionVertex`es, drawn as the game's quads.
    fn moving_pipeline(
        device: &wgpu::Device,
        module: &wgpu::ShaderModule,
        entry: &str,
        depth_stencil: Option<wgpu::DepthStencilState>,
    ) -> wgpu::RenderPipeline {
        let attributes = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2];
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(entry),
            layout: None,
            vertex: wgpu::VertexState {
                module,
                entry_point: Some("moving"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: size_of::<mod_motion_blur::MotionVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &attributes,
                })],
            },
            primitive: Default::default(),
            depth_stencil,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                targets: &[Some(MOTION.into())],
            }),
            multiview_mask: None,
            cache: None,
        })
    }

    fn sampler(device: &wgpu::Device, filter: wgpu::FilterMode) -> wgpu::Sampler {
        device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Motion blur"),
            mag_filter: filter,
            min_filter: filter,
            ..Default::default()
        })
    }

    fn uniform_buffer(device: &wgpu::Device) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Motion blur params"),
            size: size_of::<mod_motion_blur::BlurParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Writes `vertices` into `buffer`, growing it when they don't fit, and returns it.
    fn upload(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &mut Option<wgpu::Buffer>,
        vertices: &[mod_motion_blur::MotionVertex],
    ) -> wgpu::Buffer {
        let bytes = mod_motion_blur::MotionVertex::bytes(vertices);
        if buffer
            .as_ref()
            .is_none_or(|b| b.size() < bytes.len() as u64)
        {
            *buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Motion blur vertices"),
                size: (bytes.len() as u64).next_power_of_two(),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let buffer = buffer.clone().expect("made above");
        queue.write_buffer(&buffer, 0, bytes);
        buffer
    }

    /// Draws `count` vertices as quads with the game's shared quad indices.
    fn draw_quads(
        pass: &mut wgpu::RenderPass<'_>,
        vertices: &wgpu::Buffer,
        indices: &wgpu::Buffer,
        count: usize,
    ) {
        pass.set_vertex_buffer(0, vertices.slice(..));
        pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..(count / 4 * 6) as u32, 0, 0..1);
    }

    fn bind(
        device: &wgpu::Device,
        pipeline: &wgpu::RenderPipeline,
        resources: &[(u32, wgpu::BindingResource<'_>)],
    ) -> wgpu::BindGroup {
        let entries: Vec<_> = resources
            .iter()
            .map(|(binding, resource)| wgpu::BindGroupEntry {
                binding: *binding,
                resource: resource.clone(),
            })
            .collect();
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Motion blur"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        })
    }

    /// A pass into `target`: cleared to `clear`, or drawn over what it holds.
    fn begin<'e>(
        encoder: &'e mut wgpu::CommandEncoder,
        label: &str,
        target: &wgpu::TextureView,
        clear: Option<wgpu::Color>,
        depth: Option<&wgpu::TextureView>,
    ) -> wgpu::RenderPass<'e> {
        let color = [Some(wgpu::RenderPassColorAttachment {
            view: target,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: clear.map_or(wgpu::LoadOp::Load, wgpu::LoadOp::Clear),
                store: wgpu::StoreOp::Store,
            },
        })];
        let depth = depth.map(|view| wgpu::RenderPassDepthStencilAttachment {
            view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(0.0),
                store: wgpu::StoreOp::Discard,
            }),
            stencil_ops: None,
        });
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &color,
            depth_stencil_attachment: depth,
            ..Default::default()
        })
    }

    fn fullscreen(
        encoder: &mut wgpu::CommandEncoder,
        label: &str,
        target: &wgpu::TextureView,
        pipeline: &wgpu::RenderPipeline,
        group: &wgpu::BindGroup,
    ) {
        let mut pass = begin(encoder, label, target, None, None);
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.draw(0..3, 0..1);
    }
}
