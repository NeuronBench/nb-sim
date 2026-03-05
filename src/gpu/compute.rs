//! GPU compute systems: extraction, buffer management, dispatch, and readback.

use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResource;
use bevy::render::render_graph::{self, RenderLabel};
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::storage::{GpuShaderStorageBuffer, ShaderStorageBuffer};
use bevy::render::render_asset::RenderAssets;
use bevy::render::Extract;

use crate::gpu::buffers::{
    GpuJunctionData, GpuJunctionNeighbor, GpuSegmentData, GpuSynapseData, SimParams,
};
use crate::gpu::pipeline::{BiophysicsLayoutDescriptor, BiophysicsPipelines};

const WORKGROUP_SIZE: u32 = 64;

fn div_ceil(n: u32, d: u32) -> u32 {
    (n + d - 1) / d
}

// ---------------------------------------------------------------------------
// Main-world resources
// ---------------------------------------------------------------------------

/// Handle to the voltages_out ShaderStorageBuffer asset (lives in main world).
/// Extracted to render world via ExtractResource.
#[derive(Resource, Clone, ExtractResource)]
pub struct VoltagesOutHandle(pub Handle<ShaderStorageBuffer>);

/// Entity that holds the Readback component for voltages_out.
#[derive(Resource)]
pub struct VoltagesReadbackEntity(pub Entity);

/// Main-world resource holding simulation data to send to the render world.
/// Populated each frame in Update, read by ExtractSchedule system.
#[derive(Resource, Clone, Default)]
pub struct MainWorldSimInput {
    pub input_currents: Vec<f32>,
    pub params: Option<SimParams>,
    /// Full topology data, sent only when dirty.
    pub segments: Option<Vec<GpuSegmentData>>,
    pub junctions: Option<Vec<GpuJunctionData>>,
    pub synapses: Option<Vec<GpuSynapseData>>,
    pub junction_adj: Option<Vec<GpuJunctionNeighbor>>,
    pub junction_adj_offsets: Option<Vec<u32>>,
    pub synapse_adj: Option<Vec<u32>>,
    pub synapse_adj_offsets: Option<Vec<u32>>,
    pub num_segments: u32,
    pub num_junctions: u32,
    pub num_synapses: u32,
    /// Whether this frame has valid data to extract.
    pub active: bool,
    /// Whether topology has been sent to the render world at least once.
    pub topology_sent: bool,
}

// ---------------------------------------------------------------------------
// Extracted data (render world)
// ---------------------------------------------------------------------------

/// Render-world copy of the simulation input data.
#[derive(Resource, Default)]
pub struct ExtractedSimInput {
    pub input_currents: Vec<f32>,
    pub params: Option<SimParams>,
    pub segments: Option<Vec<GpuSegmentData>>,
    pub junctions: Option<Vec<GpuJunctionData>>,
    pub synapses: Option<Vec<GpuSynapseData>>,
    pub junction_adj: Option<Vec<GpuJunctionNeighbor>>,
    pub junction_adj_offsets: Option<Vec<u32>>,
    pub synapse_adj: Option<Vec<u32>>,
    pub synapse_adj_offsets: Option<Vec<u32>>,
    pub num_segments: u32,
    pub num_junctions: u32,
    pub num_synapses: u32,
}

/// ExtractSchedule system: copies MainWorldSimInput → ExtractedSimInput.
pub fn extract_sim_input(
    main_input: Extract<Res<MainWorldSimInput>>,
    mut extracted: ResMut<ExtractedSimInput>,
) {
    if !main_input.active {
        extracted.params = None;
        return;
    }
    extracted.input_currents.clone_from(&main_input.input_currents);
    extracted.params = main_input.params;
    extracted.segments = main_input.segments.clone();
    extracted.junctions = main_input.junctions.clone();
    extracted.synapses = main_input.synapses.clone();
    extracted.junction_adj = main_input.junction_adj.clone();
    extracted.junction_adj_offsets = main_input.junction_adj_offsets.clone();
    extracted.synapse_adj = main_input.synapse_adj.clone();
    extracted.synapse_adj_offsets = main_input.synapse_adj_offsets.clone();
    extracted.num_segments = main_input.num_segments;
    extracted.num_junctions = main_input.num_junctions;
    extracted.num_synapses = main_input.num_synapses;
}

// ---------------------------------------------------------------------------
// Render-world GPU state
// ---------------------------------------------------------------------------

/// Holds the raw wgpu Buffers for the simulation, persisted across frames.
#[derive(Resource)]
pub struct GpuComputeState {
    pub segments_buf: Buffer,
    pub junctions_buf: Buffer,
    pub synapses_buf: Buffer,
    pub params_buf: Buffer,
    pub synapse_deltas_buf: Buffer,
    pub input_currents_buf: Buffer,
    pub junction_adj_buf: Buffer,
    pub junction_adj_offsets_buf: Buffer,
    pub synapse_adj_buf: Buffer,
    pub synapse_adj_offsets_buf: Buffer,
    pub num_segments: u32,
    pub num_junctions: u32,
    pub num_synapses: u32,
    pub bind_group: Option<BindGroup>,
}

// ---------------------------------------------------------------------------
// Render graph node
// ---------------------------------------------------------------------------

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct BiophysicsComputeLabel;

#[derive(Default)]
pub struct BiophysicsComputeNode;

impl render_graph::Node for BiophysicsComputeNode {
    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let Some(state) = world.get_resource::<GpuComputeState>() else {
            return Ok(());
        };
        let bind_group = match &state.bind_group {
            Some(bg) => bg,
            None => return Ok(()),
        };
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipelines = world.resource::<BiophysicsPipelines>();
        let input = world.resource::<ExtractedSimInput>();
        let params = match &input.params {
            Some(p) => p,
            None => return Ok(()),
        };

        // Check all pipelines are ready
        if !pipelines.all_ready(pipeline_cache) {
            return Ok(());
        }

        let step_channels_pl = pipeline_cache
            .get_compute_pipeline(pipelines.step_channels)
            .unwrap();
        let apply_junctions_pl = pipeline_cache
            .get_compute_pipeline(pipelines.apply_junctions)
            .unwrap();
        let step_synapses_pl = pipeline_cache
            .get_compute_pipeline(pipelines.step_synapses)
            .unwrap();
        let apply_synapse_deltas_pl = pipeline_cache
            .get_compute_pipeline(pipelines.apply_synapse_deltas)
            .unwrap();
        let write_voltages_pl = pipeline_cache
            .get_compute_pipeline(pipelines.write_voltages)
            .unwrap();

        let n_seg = state.num_segments;
        let n_junc = state.num_junctions;
        let n_syn = state.num_synapses;
        let steps = params.steps_per_dispatch;

        if n_seg == 0 {
            return Ok(());
        }

        let seg_wg = div_ceil(n_seg, WORKGROUP_SIZE);
        let syn_wg = div_ceil(n_syn.max(1), WORKGROUP_SIZE);

        let encoder = render_context.command_encoder();

        // Single compute pass for all steps + write_voltages.
        // On Metal, dispatches within one pass execute in order with full
        // memory coherence, so no barriers are needed between them.
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("biophysics_compute"),
            ..default()
        });
        pass.set_bind_group(0, bind_group, &[]);

        for _step in 0..steps {
            pass.set_pipeline(step_channels_pl);
            pass.dispatch_workgroups(seg_wg, 1, 1);

            if n_junc > 0 {
                pass.set_pipeline(apply_junctions_pl);
                pass.dispatch_workgroups(seg_wg, 1, 1);
            }

            if n_syn > 0 {
                pass.set_pipeline(step_synapses_pl);
                pass.dispatch_workgroups(syn_wg, 1, 1);

                pass.set_pipeline(apply_synapse_deltas_pl);
                pass.dispatch_workgroups(seg_wg, 1, 1);
            }
        }

        pass.set_pipeline(write_voltages_pl);
        pass.dispatch_workgroups(seg_wg, 1, 1);

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Render-world systems
// ---------------------------------------------------------------------------

/// Create or recreate GPU buffers when topology changes, upload per-frame data.
pub fn prepare_gpu_buffers(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    input: Res<ExtractedSimInput>,
    existing_state: Option<ResMut<GpuComputeState>>,
) {
    let params = match &input.params {
        Some(p) => p,
        None => return,
    };

    let need_recreate = input.segments.is_some()
        || existing_state.is_none();

    if need_recreate {
        let segments_data = match input.segments.as_ref() {
            Some(s) if !s.is_empty() => s,
            _ => return, // No topology yet; wait for next frame
        };
        let junctions_data = input.junctions.as_ref().unwrap();
        let synapses_data = input.synapses.as_ref().unwrap();

        let n_seg = segments_data.len();
        let n_junc = junctions_data.len();
        let n_syn = synapses_data.len();

        // Create GPU buffers with initial data
        let segments_buf = render_device.create_buffer_with_data(
            &BufferInitDescriptor {
                label: Some("biophysics_segments"),
                contents: bytemuck::cast_slice(segments_data),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            },
        );

        // Junctions: need at least 4 bytes for runtime-sized array
        let junctions_buf = if n_junc > 0 {
            render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("biophysics_junctions"),
                contents: bytemuck::cast_slice(junctions_data),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            })
        } else {
            render_device.create_buffer(&BufferDescriptor {
                label: Some("biophysics_junctions_empty"),
                size: 16,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        // Synapses: need at least 4 bytes
        let synapses_buf = if n_syn > 0 {
            render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("biophysics_synapses"),
                contents: bytemuck::cast_slice(synapses_data),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            })
        } else {
            render_device.create_buffer(&BufferDescriptor {
                label: Some("biophysics_synapses_empty"),
                size: 16,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        let params_buf = render_device.create_buffer_with_data(
            &BufferInitDescriptor {
                label: Some("biophysics_params"),
                contents: bytemuck::bytes_of(params),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            },
        );

        // Synapse deltas: 1 float per synapse
        let sd_size = (n_syn.max(1) * std::mem::size_of::<f32>()) as u64;
        let synapse_deltas_buf = render_device.create_buffer(&BufferDescriptor {
            label: Some("biophysics_synapse_deltas"),
            size: sd_size,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Input currents
        let ic_data = &input.input_currents;
        let input_currents_buf = if !ic_data.is_empty() {
            render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("biophysics_input_currents"),
                contents: bytemuck::cast_slice(ic_data),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            })
        } else {
            render_device.create_buffer(&BufferDescriptor {
                label: Some("biophysics_input_currents_empty"),
                size: 4,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        // Junction adjacency (CSR format)
        let junction_adj_buf = match input.junction_adj.as_ref().filter(|d| !d.is_empty()) {
            Some(data) => render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("biophysics_junction_adj"),
                contents: bytemuck::cast_slice(data),
                usage: BufferUsages::STORAGE,
            }),
            None => render_device.create_buffer(&BufferDescriptor {
                label: Some("biophysics_junction_adj_empty"),
                size: 8,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
        };
        let junction_adj_offsets_buf =
            match input.junction_adj_offsets.as_ref().filter(|d| !d.is_empty()) {
                Some(data) => render_device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("biophysics_junction_adj_offsets"),
                    contents: bytemuck::cast_slice(data),
                    usage: BufferUsages::STORAGE,
                }),
                None => render_device.create_buffer(&BufferDescriptor {
                    label: Some("biophysics_junction_adj_offsets_empty"),
                    size: 4,
                    usage: BufferUsages::STORAGE,
                    mapped_at_creation: false,
                }),
            };

        // Synapse adjacency (CSR format)
        let synapse_adj_buf = match input.synapse_adj.as_ref().filter(|d| !d.is_empty()) {
            Some(data) => render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("biophysics_synapse_adj"),
                contents: bytemuck::cast_slice(data),
                usage: BufferUsages::STORAGE,
            }),
            None => render_device.create_buffer(&BufferDescriptor {
                label: Some("biophysics_synapse_adj_empty"),
                size: 4,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
        };
        let synapse_adj_offsets_buf =
            match input.synapse_adj_offsets.as_ref().filter(|d| !d.is_empty()) {
                Some(data) => render_device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("biophysics_synapse_adj_offsets"),
                    contents: bytemuck::cast_slice(data),
                    usage: BufferUsages::STORAGE,
                }),
                None => render_device.create_buffer(&BufferDescriptor {
                    label: Some("biophysics_synapse_adj_offsets_empty"),
                    size: 4,
                    usage: BufferUsages::STORAGE,
                    mapped_at_creation: false,
                }),
            };

        commands.insert_resource(GpuComputeState {
            segments_buf,
            junctions_buf,
            synapses_buf,
            params_buf,
            synapse_deltas_buf,
            input_currents_buf,
            junction_adj_buf,
            junction_adj_offsets_buf,
            synapse_adj_buf,
            synapse_adj_offsets_buf,
            num_segments: n_seg as u32,
            num_junctions: n_junc as u32,
            num_synapses: n_syn as u32,
            bind_group: None,
        });
    } else if let Some(mut state) = existing_state {
        // Just update per-frame data: params + input_currents
        render_queue.write_buffer(&state.params_buf, 0, bytemuck::bytes_of(params));

        // Recreate input_currents buffer if size changed, otherwise write
        let ic_data = &input.input_currents;
        if !ic_data.is_empty() {
            let needed_size = (ic_data.len() * std::mem::size_of::<f32>()) as u64;
            if state.input_currents_buf.size() >= needed_size {
                render_queue.write_buffer(
                    &state.input_currents_buf,
                    0,
                    bytemuck::cast_slice(ic_data),
                );
            } else {
                state.input_currents_buf =
                    render_device.create_buffer_with_data(&BufferInitDescriptor {
                        label: Some("biophysics_input_currents"),
                        contents: bytemuck::cast_slice(ic_data),
                        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                    });
                // Invalidate bind group so it gets recreated
                state.bind_group = None;
            }
        }
    }
}

/// Create the bind group from GPU buffers + voltages_out asset.
pub fn prepare_bind_group(
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    layout_desc: Option<Res<BiophysicsLayoutDescriptor>>,
    voltages_handle: Option<Res<VoltagesOutHandle>>,
    ssbos: Res<RenderAssets<GpuShaderStorageBuffer>>,
    mut state: Option<ResMut<GpuComputeState>>,
) {
    let Some(layout_desc) = layout_desc else { return };
    let Some(ref mut state) = state else { return };
    let Some(voltages_handle) = voltages_handle else {
        return;
    };

    // Only recreate bind group when it's None (invalidated on topology change)
    if state.bind_group.is_some() {
        return;
    }

    let Some(voltages_gpu) = ssbos.get(&voltages_handle.0) else {
        return;
    };

    let layout = pipeline_cache.get_bind_group_layout(&layout_desc.0);

    state.bind_group = Some(render_device.create_bind_group(
        "biophysics_bind_group",
        &layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: state.segments_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: state.junctions_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: state.synapses_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: state.params_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: state.synapse_deltas_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 5,
                resource: state.input_currents_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 6,
                resource: voltages_gpu.buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 7,
                resource: state.junction_adj_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 8,
                resource: state.junction_adj_offsets_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 9,
                resource: state.synapse_adj_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 10,
                resource: state.synapse_adj_offsets_buf.as_entire_binding(),
            },
        ],
    ));
}
