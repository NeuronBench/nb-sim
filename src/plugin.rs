use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::render_resource::BufferUsages;
use bevy::render::storage::ShaderStorageBuffer;
use std::fmt::{self, Display};
use std::time::Duration;

use crate::constants::{
    BODY_TEMPERATURE, CONDUCTANCE_PER_SQUARE_CM, GAS_CONSTANT, INVERSE_FARADAY,
};
use crate::cpu_sim;
use crate::dimension::{Interval, Kelvin, SimulationStepSeconds, StepsPerFrame, Timestamp};
use crate::gpu::buffers::SimParams;
use crate::gpu::compute::{MainWorldSimInput, VoltagesOutHandle, VoltagesReadbackEntity};
use crate::gpu::extract;
use crate::gpu::{GpuComputePlugin, GpuSimState, SimBackend};
use crate::stimulator::{Stimulation, Stimulator, StimulatorMaterials};

use crate::gui;
use crate::gui::oscilloscope::{step_oscilloscope_system, Oscilloscope};
use crate::integrations::grace::Synapse;
use crate::neuron::channel::{ca_reversal, cl_reversal, k_reversal, na_reversal};
use crate::neuron::membrane::{Membrane, MembraneMaterials, MembraneVoltage};
use crate::neuron::segment::{ecs::InputCurrent, ecs::Segment, Geometry};
use crate::neuron::solution::{Solution, INTERSTICIAL_FLUID};
use crate::neuron::Junction;

pub struct NbSimPlugin;

impl Plugin for NbSimPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(default_env())
            .insert_resource(Timestamp(0.0))
            .insert_resource(StepsPerFrame(100))
            .init_resource::<gui::NextClickAction>()
            .init_resource::<Oscilloscope>()
            .insert_resource(Stimulator::default())
            .insert_resource(SimulationStepSeconds(5e-7))
            .init_resource::<MembraneMaterials>()
            .init_resource::<StimulatorMaterials>()
            .init_resource::<GpuSimState>()
            .insert_resource(SimBackend::Cpu)
            .insert_resource(StdoutRenderTimer {
                timer: Timer::new(Duration::from_millis(2000), TimerMode::Repeating),
            });

        // GPU compute plugin (sets up render-world pipeline, extraction, dispatch)
        app.add_plugins(GpuComputePlugin);

        // Readback observer: GPU voltages → ECS MembraneVoltage
        app.add_observer(apply_voltage_readback);

        app.add_systems(Update, (
            step_biophysics_flat,
            setup_gpu_readback,
            extract_sim_input_to_render_world
                .after(step_biophysics_flat)
                .after(setup_gpu_readback),
        ));

        app.add_systems(Update, apply_voltage_to_materials)
            .add_systems(Update, apply_current_to_stimulator_material)
            .add_systems(Update, step_oscilloscope_system)
            // .add_systems(Update, print_oscilloscope_system)
            .add_systems(Update, print_voltages);
        gui::load::setup(app);
    }
}

#[derive(Resource)]
pub struct StdoutRenderTimer {
    timer: Timer,
}

fn step_biophysics(
    env: Res<Env>,
    simulation_step: Res<SimulationStepSeconds>,
    mut timestamp: ResMut<Timestamp>,
    steps_per_frame: Res<StepsPerFrame>,
    mut segments_query: Query<(
        &Segment,
        &Solution,
        &Geometry,
        &mut Membrane,
        &mut MembraneVoltage,
        Option<&InputCurrent>,
        Option<&Stimulator>,
    )>,
    junctions_query: Query<&Junction>,
    mut synapses_query: Query<&mut Synapse>,
) {
    for _ in 0..steps_per_frame.0 {
        for (
            _,
            solution,
            geometry,
            mut membrane,
            mut membrane_voltage,
            maybe_input_current,
            maybe_stimulator,
        ) in &mut segments_query
        {
            // ***********************************
            // ***** Apply channel currents. *****
            // ***********************************
            let surface_area = geometry.surface_area();

            let current =
                -1.0 * membrane.current_per_square_cm(
                    &k_reversal(&solution, &env.extracellular_solution, &env.temperature),
                    &na_reversal(&solution, &env.extracellular_solution, &env.temperature),
                    &cl_reversal(&solution, &env.extracellular_solution, &env.temperature),
                    &ca_reversal(&solution, &env.extracellular_solution, &env.temperature),
                    &membrane_voltage.0,
                ) * surface_area;
            let capacitance = membrane.capacitance.0 * surface_area;
            let dv_dt: f32 = current / capacitance;

            membrane_voltage.0 .0 += 1000.0 * dv_dt * simulation_step.0;

            // ***********************************
            // ***** Update membrane conductances.
            // ***********************************
            membrane
                .membrane_channels
                .iter_mut()
                .for_each(|membrane_channel| {
                    membrane_channel
                        .channel
                        .step(&membrane_voltage.0, &Interval(simulation_step.0))
                });

            // ***************************************************
            // ***** Apply input currents and stimulators. *******
            // ***************************************************
            let input_current = maybe_input_current.map_or(0.0, |i| i.0 .0);
            let stimulator_current =
                maybe_stimulator.map_or(0.0, |stimulator| stimulator.current(timestamp.clone()).0);
            let current_microamps = input_current + stimulator_current;
            let capacitance = membrane.capacitance.0 * surface_area;
            let current = current_microamps * 1e-6 * surface_area;
            let dv_dt = current / capacitance;
            membrane_voltage.0 .0 += 1000.0 * dv_dt * simulation_step.0;
        }

        for Junction {
            first_segment,
            second_segment,
            pore_diameter,
        } in &junctions_query
        {
            let interval_seconds = simulation_step.0;

            let results =
                segments_query.get_many_mut([first_segment.clone(), second_segment.clone()]);
            match results {
                Ok(
                    [(_, _, geom1, membrane1, mut vm1, _, _), (_, _, geom2, membrane2, mut vm2, _, _)],
                ) => {
                    let capacitance1 = membrane1.capacitance.0 * geom1.surface_area();
                    let capacitance2 = membrane2.capacitance.0 * geom2.surface_area();

                    let mutual_conductance =
                        pore_diameter.0 * std::f32::consts::PI * CONDUCTANCE_PER_SQUARE_CM;
                    let first_to_second_current = mutual_conductance * (vm1.0 .0 - vm2.0 .0) * 1e-3;

                    vm1.0 .0 -= first_to_second_current / capacitance1 * interval_seconds;
                    vm2.0 .0 += first_to_second_current / capacitance2 * interval_seconds;
                }
                Err(e) => panic!("Other error {e}"),
            }
        }

        for mut synapse in &mut synapses_query {
            // TODO: This fails if the source and target of the synapse are the same Entity.
            let interval_seconds = simulation_step.0;
            let results = segments_query
                .get_many_mut([synapse.pre_segment.clone(), synapse.post_segment.clone()]);
            match results {
                Ok([(_, _, _, _, vm1, _, _), (_, solution, _, _, mut vm2, _, _)]) => {
                    synapse.synapse_membranes.step(
                        &BODY_TEMPERATURE,
                        &vm1.0,
                        &vm2.0,
                        &Interval(interval_seconds),
                    );
                    synapse.synapse_membranes.apply_current(
                        &Interval(interval_seconds),
                        &BODY_TEMPERATURE,
                        &mut vm2.0,
                        &solution,
                    );
                }
                Err(e) => {
                    eprintln!("Synapse query error: {e}");
                }
            }
        }

        // ***************************************
        // ***** Advance simulation time. *******
        // ***************************************
        timestamp.0 += simulation_step.0;
    }
}

/// Flat-array simulation backend (CPU path). Extracts ECS → flat arrays,
/// runs CPU simulation, writes back. Skipped when GPU backend is active.
fn step_biophysics_flat(
    env: Res<Env>,
    simulation_step: Res<SimulationStepSeconds>,
    mut timestamp: ResMut<Timestamp>,
    steps_per_frame: Res<StepsPerFrame>,
    backend: Res<SimBackend>,
    mut gpu_state: ResMut<GpuSimState>,
    mut segments_query: Query<(
        Entity,
        &Segment,
        &Solution,
        &Geometry,
        &mut Membrane,
        &mut MembraneVoltage,
        Option<&InputCurrent>,
        Option<&Stimulator>,
    )>,
    junctions_query: Query<&Junction>,
    mut synapses_query: Query<&mut Synapse>,
) {
    if segments_query.is_empty() {
        return;
    }

    // When GPU is active, we still need to extract ECS→flat arrays for the
    // render world (topology + input currents), but skip CPU simulation.
    if *backend == SimBackend::Gpu {
        // Keep GpuSimState in sync so extract_sim_input_to_render_world has data
        if gpu_state.topology_dirty || !gpu_state.initialized {
            extract::extract_to_gpu_state(
                &segments_query,
                &junctions_query,
                &synapses_query,
                &timestamp,
                &mut gpu_state,
            );
        } else {
            extract::update_input_currents(&segments_query, &timestamp, &mut gpu_state);
        }
        timestamp.0 += simulation_step.0 * steps_per_frame.0 as f32;
        return;
    }

    extract::extract_to_gpu_state(
        &segments_query,
        &junctions_query,
        &synapses_query,
        &timestamp,
        &mut gpu_state,
    );

    let params = SimParams {
        extra_k: env.extracellular_solution.k_concentration.0,
        extra_na: env.extracellular_solution.na_concentration.0,
        extra_ca: env.extracellular_solution.ca_concentration.0,
        extra_cl: env.extracellular_solution.cl_concentration.0,
        temperature: env.temperature.0,
        dt: simulation_step.0,
        steps_per_dispatch: steps_per_frame.0 as u32,
        num_segments: gpu_state.segments.len() as u32,
        num_junctions: gpu_state.junctions.len() as u32,
        num_synapses: gpu_state.synapses.len() as u32,
        gas_constant: GAS_CONSTANT,
        inverse_faraday: INVERSE_FARADAY,
        conductance_per_square_cm: CONDUCTANCE_PER_SQUARE_CM,
        synapse_resistance_ohms: 1_000_000_000.0,
        _pad0: 0.0,
        _pad1: 0.0,
    };

    let state = &mut *gpu_state;
    cpu_sim::step_biophysics_cpu(
        &mut state.segments,
        &state.junctions,
        &mut state.synapses,
        &params,
    );

    extract::writeback_to_ecs(state, &mut segments_query, &mut synapses_query);
    timestamp.0 += simulation_step.0 * steps_per_frame.0 as f32;
}

/// One-shot system: when segments first appear, create the ShaderStorageBuffer
/// for voltages_out and spawn a Readback entity for async GPU→CPU transfer.
fn setup_gpu_readback(
    mut commands: Commands,
    mut ssbos: ResMut<Assets<ShaderStorageBuffer>>,
    mut backend: ResMut<SimBackend>,
    segments_query: Query<Entity, With<Segment>>,
    existing_handle: Option<Res<VoltagesOutHandle>>,
) {
    // Only set up once, and only when we have segments
    if existing_handle.is_some() || segments_query.is_empty() {
        return;
    }

    let n_segments = segments_query.iter().count();
    let buf_size = n_segments * std::mem::size_of::<f32>();

    let mut buffer = ShaderStorageBuffer::with_size(buf_size, RenderAssetUsages::RENDER_WORLD);
    buffer.buffer_description.usage =
        BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
    let handle = ssbos.add(buffer);

    // Spawn readback entity
    let readback_entity = commands.spawn(Readback::buffer(handle.clone())).id();

    commands.insert_resource(VoltagesOutHandle(handle));
    commands.insert_resource(VoltagesReadbackEntity(readback_entity));

    // Switch to GPU backend. The render graph node gracefully handles
    // the case where pipelines aren't compiled yet (just returns Ok).
    *backend = SimBackend::Gpu;
    info!(
        "GPU readback set up for {} segments, switching to GPU backend",
        n_segments
    );
}

/// Populate MainWorldSimInput each frame so the render world can read it.
fn extract_sim_input_to_render_world(
    env: Res<Env>,
    simulation_step: Res<SimulationStepSeconds>,
    steps_per_frame: Res<StepsPerFrame>,
    backend: Res<SimBackend>,
    gpu_state: Res<GpuSimState>,
    timestamp: Res<Timestamp>,
    mut main_input: ResMut<MainWorldSimInput>,
    segments_query: Query<(
        Entity,
        &Segment,
        &Solution,
        &Geometry,
        &Membrane,
        &MembraneVoltage,
        Option<&InputCurrent>,
        Option<&Stimulator>,
    )>,
) {
    // Only populate when GPU is active
    if *backend != SimBackend::Gpu {
        main_input.active = false;
        return;
    }

    if segments_query.is_empty() {
        main_input.active = false;
        return;
    }

    // Build input_currents from ECS
    let n_seg = gpu_state.segment_entities.len();
    main_input.input_currents.resize(n_seg, 0.0);
    for (entity, _seg, _sol, _geom, _membrane, _voltage, maybe_input, maybe_stim) in
        segments_query.iter()
    {
        if let Some(&idx) = gpu_state.entity_to_index.get(&entity) {
            let ic = maybe_input.map_or(0.0, |i| i.0 .0);
            let sc = maybe_stim.map_or(0.0, |s| s.current(timestamp.clone()).0);
            main_input.input_currents[idx as usize] = ic + sc;
        }
    }

    // Build params
    main_input.params = Some(SimParams {
        extra_k: env.extracellular_solution.k_concentration.0,
        extra_na: env.extracellular_solution.na_concentration.0,
        extra_ca: env.extracellular_solution.ca_concentration.0,
        extra_cl: env.extracellular_solution.cl_concentration.0,
        temperature: env.temperature.0,
        dt: simulation_step.0,
        steps_per_dispatch: steps_per_frame.0 as u32,
        num_segments: gpu_state.segments.len() as u32,
        num_junctions: gpu_state.junctions.len() as u32,
        num_synapses: gpu_state.synapses.len() as u32,
        gas_constant: GAS_CONSTANT,
        inverse_faraday: INVERSE_FARADAY,
        conductance_per_square_cm: CONDUCTANCE_PER_SQUARE_CM,
        synapse_resistance_ohms: 1_000_000_000.0,
        _pad0: 0.0,
        _pad1: 0.0,
    });

    // Send topology if not yet sent to render world (and data is available)
    if !main_input.topology_sent && !gpu_state.segments.is_empty() {
        main_input.segments = Some(gpu_state.segments.clone());
        main_input.junctions = Some(gpu_state.junctions.clone());
        main_input.synapses = Some(gpu_state.synapses.clone());
        main_input.topology_sent = true;
    } else {
        main_input.segments = None;
        main_input.junctions = None;
        main_input.synapses = None;
    }

    main_input.num_segments = gpu_state.segments.len() as u32;
    main_input.num_junctions = gpu_state.junctions.len() as u32;
    main_input.num_synapses = gpu_state.synapses.len() as u32;
    main_input.active = true;
}

/// Handle voltage readback from GPU: update MembraneVoltage components.
fn apply_voltage_readback(
    event: On<ReadbackComplete>,
    gpu_state: Res<GpuSimState>,
    mut segments_query: Query<&mut MembraneVoltage>,
) {
    let data: &[u8] = &event.data;
    let expected_size = gpu_state.segment_entities.len() * std::mem::size_of::<f32>();
    if data.len() < expected_size {
        return;
    }
    let voltages: &[f32] = bytemuck::cast_slice(&data[..expected_size]);

    for (i, entity) in gpu_state.segment_entities.iter().enumerate() {
        if let Ok(mut mv) = segments_query.get_mut(*entity) {
            mv.0 .0 = voltages[i];
        }
    }
}

// SegmentBundle removed in Bevy 0.18 migration - spawn components directly.

impl Display for MembraneVoltage {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} mV", self.0 .0)
    }
}

#[derive(Component)]
pub struct Neuron;

#[derive(Resource)]
pub struct Env {
    pub temperature: Kelvin,
    pub extracellular_solution: Solution,
}

fn default_env() -> Env {
    Env {
        temperature: BODY_TEMPERATURE,
        extracellular_solution: INTERSTICIAL_FLUID,
    }
}

fn apply_voltage_to_materials(
    membrane_materials: Res<MembraneMaterials>,
    mut query: Query<(&MembraneVoltage, &mut MeshMaterial3d<StandardMaterial>)>,
) {
    for (v, mut material) in &mut query {
        material.0 = membrane_materials.from_voltage(&v.0);
    }
}

fn apply_current_to_stimulator_material(
    stimulator_materials: Res<StimulatorMaterials>,
    segments_query: Query<(&Segment, &Stimulator)>,
    timestamp: Res<Timestamp>,
    mut stimulations_query: Query<(&Stimulation, &mut MeshMaterial3d<StandardMaterial>)>,
) {
    for (
        Stimulation {
            stimulation_segment,
        },
        mut material,
    ) in &mut stimulations_query
    {
        if let Ok((_, stimulator)) = segments_query.get(*stimulation_segment) {
            let current = stimulator.current(Timestamp(timestamp.0));
            material.0 = stimulator_materials.from_selected_and_current(false, &current);
        } else {
            println!("Error, stimulation's segment not found.");
        }
    }
}

fn print_voltages(
    timestamp: Res<Timestamp>,
    mut stdout_render_timer: ResMut<StdoutRenderTimer>,
    query: Query<&MembraneVoltage>,
    time: Res<Time>,
) {
    stdout_render_timer.timer.tick(time.delta());

    if stdout_render_timer.timer.just_finished() {
        if let Some(membrane_voltage) = &query.iter().next() {
            println!(
                "SimulationTime: {} ms. First Voltage: {membrane_voltage}",
                timestamp.0 * 1000.0
            );
        }
        if let Some(membrane_voltage) = &query.iter().next() {
            println!(
                "SimulationTime: {} ms. Second Voltage: {membrane_voltage}",
                timestamp.0
            );
        }
        if let Some(membrane_voltage) = &query.iter().next() {
            println!(
                "SimulationTime: {} ms. Third Voltage: {membrane_voltage}",
                timestamp.0
            );
        }
        println!("");
    }
}

// pub fn serialize_simulation (
//     extracellular_solution: &Solution,
//     segments: &[(Membrane, MembraneVoltage, Stimulator)]
// ) -> serialize::Scene {
//     serialize::Scene {
//         extracellular_solution: extracellular_solution.serialize(),
//         membranes: unimplemented!(),
//         neurons: unimplemented!(),
//         synapses: vec![],
//     }
// }
