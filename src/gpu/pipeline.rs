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

/// Holds the bind group layout descriptor (used to get the actual BindGroupLayout
/// from PipelineCache when creating bind groups).
#[derive(Resource)]
pub struct BiophysicsLayoutDescriptor(pub BindGroupLayoutDescriptor);

/// Holds the 6 cached compute pipeline IDs (one per shader entry point).
#[derive(Resource)]
pub struct BiophysicsPipelines {
    pub step_channels: CachedComputePipelineId,
    pub compute_junction_deltas: CachedComputePipelineId,
    pub apply_junction_deltas: CachedComputePipelineId,
    pub step_synapses: CachedComputePipelineId,
    pub apply_synapse_deltas: CachedComputePipelineId,
    pub write_voltages: CachedComputePipelineId,
}

impl BiophysicsPipelines {
    /// Returns true if all 6 pipelines are ready.
    pub fn all_ready(&self, cache: &PipelineCache) -> bool {
        [
            self.step_channels,
            self.compute_junction_deltas,
            self.apply_junction_deltas,
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
                // binding 4: junction_deltas (read-write storage, runtime-sized)
                storage_buffer_sized(false, None),
                // binding 5: synapse_deltas (read-write storage, runtime-sized)
                storage_buffer_sized(false, None),
                // binding 6: input_currents (read-only storage, runtime-sized)
                storage_buffer_read_only_sized(false, None),
                // binding 7: voltages_out (read-write storage, runtime-sized)
                storage_buffer_sized(false, None),
            ),
        ),
    )
}

pub fn init_biophysics_pipelines(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let layout_desc = make_layout_desc();
    let shader: Handle<Shader> = asset_server.load("shaders/biophysics.wgsl");

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
        compute_junction_deltas: make_pipeline("compute_junction_deltas"),
        apply_junction_deltas: make_pipeline("apply_junction_deltas"),
        step_synapses: make_pipeline("step_synapses"),
        apply_synapse_deltas: make_pipeline("apply_synapse_deltas"),
        write_voltages: make_pipeline("write_voltages"),
    };

    commands.insert_resource(BiophysicsLayoutDescriptor(layout_desc));
    commands.insert_resource(pipelines);
    info!("Biophysics compute pipelines queued");
}
