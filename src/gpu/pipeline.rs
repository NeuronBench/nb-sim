//! Compute pipeline initialization for the biophysics shader.
//!
//! Runs once during `RenderStartup` to queue all 6 compute pipelines
//! (one per shader entry point) and store the shared bind group layout descriptor.

use bevy::prelude::*;
use bevy::render::render_resource::binding_types::{
    storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer_sized,
};
use bevy::render::render_resource::*;
use std::borrow::Cow;
use std::num::NonZero;

use crate::gpu::buffers::SimParams;

/// Size of SimParams in bytes (16 f32 = 64 bytes).
const SIM_PARAMS_SIZE: u64 = std::mem::size_of::<SimParams>() as u64;

/// Storage buffers the biophysics bind group uses in the compute stage.
/// WebGPU guarantees 8 (`max_storage_buffers_per_shader_stage`); Firefox
/// allows 9. Keep `make_layout_desc` within this, and update it if it changes.
pub const REQUIRED_STORAGE_BUFFERS: u32 = 8;

/// Holds the compile-time-embedded shader handle (avoids HTTP fetch in wasm).
#[derive(Resource, Clone)]
pub struct BiophysicsShaderHandle(pub Handle<Shader>);

/// Holds the bind group layout descriptor (used to get the actual BindGroupLayout
/// from PipelineCache when creating bind groups).
#[derive(Resource)]
pub struct BiophysicsLayoutDescriptor(pub BindGroupLayoutDescriptor);

/// Holds the 5 cached compute pipeline IDs (one per shader entry point).
#[derive(Resource)]
pub struct BiophysicsPipelines {
    pub step_channels: CachedComputePipelineId,
    pub apply_junctions: CachedComputePipelineId,
    pub step_synapses: CachedComputePipelineId,
    pub apply_synapse_deltas: CachedComputePipelineId,
    pub write_voltages: CachedComputePipelineId,
}

impl BiophysicsPipelines {
    /// Returns true if all 5 pipelines are ready.
    pub fn all_ready(&self, cache: &PipelineCache) -> bool {
        [
            self.step_channels,
            self.apply_junctions,
            self.step_synapses,
            self.apply_synapse_deltas,
            self.write_voltages,
        ]
        .iter()
        .all(|id| cache.get_compute_pipeline(*id).is_some())
    }
}

fn make_layout_desc() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "biophysics_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                // binding 0: segments (read-write storage, runtime-sized)
                storage_buffer_sized(false, None),
                // binding 1: junctions (read-only storage, runtime-sized)
                storage_buffer_read_only_sized(false, None),
                // binding 2: synapses (read-write storage, runtime-sized)
                storage_buffer_sized(false, None),
                // binding 3: params (uniform)
                uniform_buffer_sized(
                    false,
                    Some(NonZero::new(SIM_PARAMS_SIZE).unwrap()),
                ),
                // binding 4: synapse_deltas (read-write storage, runtime-sized)
                storage_buffer_sized(false, None),
                // binding 5: input_currents (read-only storage, runtime-sized)
                storage_buffer_read_only_sized(false, None),
                // binding 6: voltages_out (read-write storage, runtime-sized)
                storage_buffer_sized(false, None),
                // binding 7: junction_adj (read-only, CSR data)
                storage_buffer_read_only_sized(false, None),
                // binding 8: adj_index (read-only, packed CSR offsets + synapse indices)
                storage_buffer_read_only_sized(false, None),
            ),
        ),
    )
}

pub fn init_biophysics_pipelines(
    mut commands: Commands,
    shader_handle: Res<BiophysicsShaderHandle>,
    pipeline_cache: Res<PipelineCache>,
) {
    let layout_desc = make_layout_desc();
    let shader: Handle<Shader> = shader_handle.0.clone();

    let make_pipeline = |entry_point: &'static str| -> CachedComputePipelineId {
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(Cow::from(format!("biophysics_{entry_point}"))),
            layout: vec![layout_desc.clone()],
            shader: shader.clone(),
            entry_point: Some(Cow::from(entry_point)),
            ..default()
        })
    };

    let pipelines = BiophysicsPipelines {
        step_channels: make_pipeline("step_channels"),
        apply_junctions: make_pipeline("apply_junctions"),
        step_synapses: make_pipeline("step_synapses"),
        apply_synapse_deltas: make_pipeline("apply_synapse_deltas"),
        write_voltages: make_pipeline("write_voltages"),
    };

    commands.insert_resource(BiophysicsLayoutDescriptor(layout_desc));
    commands.insert_resource(pipelines);
    info!("Biophysics compute pipelines queued");
}

/// Render-world system: if any biophysics pipeline failed to build, flag GPU
/// compute as unusable so the main world falls back to the CPU backend.
pub fn detect_pipeline_errors(
    pipelines: Option<Res<BiophysicsPipelines>>,
    cache: Res<PipelineCache>,
    health: Res<crate::gpu::GpuComputeHealth>,
) {
    let Some(pipelines) = pipelines else { return };
    if health.failed() {
        return;
    }
    for id in [
        pipelines.step_channels,
        pipelines.apply_junctions,
        pipelines.step_synapses,
        pipelines.apply_synapse_deltas,
        pipelines.write_voltages,
    ] {
        if let CachedPipelineState::Err(err) = cache.get_compute_pipeline_state(id) {
            error!("biophysics compute pipeline failed, falling back to CPU: {err}");
            health.mark_failed();
            return;
        }
    }
}
