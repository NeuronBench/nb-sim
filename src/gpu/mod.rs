pub mod buffers;
pub mod compute;
pub mod extract;
pub mod pipeline;

use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResourcePlugin;
use bevy::render::render_graph::RenderGraph;
use bevy::render::{ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems};
use std::collections::HashMap;

use crate::gpu::buffers::{GpuJunctionData, GpuSegmentData, GpuSynapseData};
use crate::gpu::compute::{
    extract_sim_input, prepare_bind_group, prepare_gpu_buffers, BiophysicsComputeLabel,
    BiophysicsComputeNode, ExtractedSimInput, MainWorldSimInput, VoltagesOutHandle,
};
use crate::gpu::pipeline::init_biophysics_pipelines;

/// Which simulation backend is active.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimBackend {
    Gpu,
    Cpu,
}

impl Default for SimBackend {
    fn default() -> Self {
        SimBackend::Cpu
    }
}

/// Holds the flat-array simulation state for GPU upload / CPU fallback.
#[derive(Resource)]
pub struct GpuSimState {
    pub segments: Vec<GpuSegmentData>,
    pub junctions: Vec<GpuJunctionData>,
    pub synapses: Vec<GpuSynapseData>,
    pub segment_entities: Vec<Entity>,
    pub entity_to_index: HashMap<Entity, u32>,
    pub topology_dirty: bool,
    pub initialized: bool,
}

impl Default for GpuSimState {
    fn default() -> Self {
        GpuSimState {
            segments: Vec::new(),
            junctions: Vec::new(),
            synapses: Vec::new(),
            segment_entities: Vec::new(),
            entity_to_index: HashMap::new(),
            topology_dirty: true,
            initialized: false,
        }
    }
}

/// Plugin that wires up GPU compute for biophysics simulation.
pub struct GpuComputePlugin;

impl Plugin for GpuComputePlugin {
    fn build(&self, app: &mut App) {
        // Main world resources
        app.init_resource::<MainWorldSimInput>();

        // Extract VoltagesOutHandle to render world
        app.add_plugins(ExtractResourcePlugin::<VoltagesOutHandle>::default());

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            warn!("No RenderApp found, GPU compute disabled");
            return;
        };

        // Render-world resources
        render_app.init_resource::<ExtractedSimInput>();

        // ExtractSchedule: copy main world data to render world
        render_app.add_systems(ExtractSchedule, extract_sim_input);

        // Render startup: init pipelines
        render_app.add_systems(RenderStartup, init_biophysics_pipelines);

        // Render frame: prepare buffers and bind group
        render_app.add_systems(
            Render,
            (
                prepare_gpu_buffers.in_set(RenderSystems::PrepareResources),
                prepare_bind_group.in_set(RenderSystems::PrepareBindGroups),
            ),
        );

        // Render graph: compute node runs before camera rendering
        let mut render_graph = render_app
            .world_mut()
            .resource_mut::<RenderGraph>();
        render_graph.add_node(BiophysicsComputeLabel, BiophysicsComputeNode::default());
        render_graph.add_node_edge(
            BiophysicsComputeLabel,
            bevy::render::graph::CameraDriverLabel,
        );
    }
}
