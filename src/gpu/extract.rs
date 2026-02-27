//! ECS-to-flat-array extraction for the GPU / CPU simulation backends.

use bevy::prelude::*;
use bytemuck::Zeroable;

use crate::constants::CONDUCTANCE_PER_SQUARE_CM;
use crate::dimension::Timestamp;
use crate::gpu::buffers::*;
use crate::gpu::GpuSimState;
use crate::integrations::grace::Synapse;
use crate::neuron::channel::{GateState, TimeConstant};
use crate::neuron::membrane::{Membrane, MembraneChannel, MembraneVoltage};
use crate::neuron::segment::ecs::{InputCurrent, Segment};
use crate::neuron::segment::Geometry;
use crate::neuron::solution::Solution;
use crate::neuron::synapse::{Receptor, Transmitter, TransmitterPump};
use crate::neuron::Junction;
use crate::stimulator::Stimulator;

// --- Conversion helpers ---

fn gate_state_to_gpu(gate: &Option<GateState>) -> GpuGateState {
    match gate {
        None => GpuGateState::zeroed(),
        Some(gs) => {
            let (tc_type, tc_param0, tc_param1, tc_param2, tc_param3) =
                match &gs.parameters.time_constant {
                    TimeConstant::Instantaneous => (0.0, 0.0, 0.0, 0.0, 0.0),
                    TimeConstant::Gaussian {
                        v_at_max_tau,
                        c_base,
                        c_amp,
                        sigma,
                    } => (1.0, v_at_max_tau.0, *c_base, *c_amp, *sigma),
                    TimeConstant::LinearExp {
                        coef,
                        v_offset,
                        inner_coef,
                    } => (2.0, *coef, v_offset.0, *inner_coef, 0.0),
                };
            GpuGateState {
                present: 1.0,
                gates: gs.parameters.gates as f32,
                magnitude: gs.magnitude,
                ss_v_at_half_max: gs.parameters.steady_state_magnitude.v_at_half_max.0,
                ss_slope: gs.parameters.steady_state_magnitude.slope,
                tc_type,
                tc_param0,
                tc_param1,
                tc_param2,
                tc_param3,
                _pad0: 0.0,
                _pad1: 0.0,
            }
        }
    }
}

fn membrane_channel_to_gpu(mc: &MembraneChannel) -> GpuMembraneChannel {
    GpuMembraneChannel {
        siemens_per_square_cm: mc.siemens_per_square_cm,
        ion_k: mc.channel.ion_selectivity.k,
        ion_na: mc.channel.ion_selectivity.na,
        ion_ca: mc.channel.ion_selectivity.ca,
        ion_cl: mc.channel.ion_selectivity.cl,
        _pad: 0.0,
        activation: gate_state_to_gpu(&mc.channel.activation),
        inactivation: gate_state_to_gpu(&mc.channel.inactivation),
    }
}

fn transmitter_to_f32(t: &Transmitter) -> f32 {
    match t {
        Transmitter::Glutamate => 0.0,
        Transmitter::Gaba => 1.0,
    }
}

fn transmitter_pump_to_gpu(pump: &TransmitterPump) -> GpuTransmitterPump {
    GpuTransmitterPump {
        present: 1.0,
        transmitter: transmitter_to_f32(&pump.transmitter),
        target_conc_max: pump.transmitter_pump_params.target_concentration_max.0,
        target_conc_min: pump.transmitter_pump_params.target_concentration_min.0,
        target_conc_v_half: pump.transmitter_pump_params.target_concentration_v_at_half_max.0,
        target_conc_v_slope: pump.transmitter_pump_params.target_concentration_v_slope,
        tc_v_at_max_tau: pump.transmitter_pump_params.time_constant_v_at_max_tau.0,
        tc_c_base: pump.transmitter_pump_params.time_constant_c_base,
        tc_c_amp: pump.transmitter_pump_params.time_constant_c_amp,
        tc_sigma: pump.transmitter_pump_params.time_constant_sigma,
        _pad0: 0.0,
        _pad1: 0.0,
    }
}

fn receptor_to_gpu(receptor: &Receptor) -> GpuReceptor {
    GpuReceptor {
        present: 1.0,
        sens_transmitter: transmitter_to_f32(&receptor.neurotransmitter_sensitivity.transmitter),
        sens_conc_half_max: receptor.neurotransmitter_sensitivity.concentration_at_half_max.0,
        sens_slope: receptor.neurotransmitter_sensitivity.slope,
        channel: membrane_channel_to_gpu(&receptor.membrane_channel),
    }
}

// --- Writeback helpers: GPU flat arrays → ECS components ---

fn gpu_gate_to_ecs(gpu: &GpuGateState, gate: &mut Option<GateState>) {
    if let Some(gs) = gate.as_mut() {
        gs.magnitude = gpu.magnitude;
    }
}

fn gpu_channel_to_ecs(gpu: &GpuMembraneChannel, mc: &mut MembraneChannel) {
    gpu_gate_to_ecs(&gpu.activation, &mut mc.channel.activation);
    gpu_gate_to_ecs(&gpu.inactivation, &mut mc.channel.inactivation);
}

// --- Full extraction: build GpuSimState from ECS ---

pub fn extract_to_gpu_state(
    segments_query: &Query<
        (
            Entity,
            &Segment,
            &Solution,
            &Geometry,
            &mut Membrane,
            &mut MembraneVoltage,
            Option<&InputCurrent>,
            Option<&Stimulator>,
        ),
    >,
    junctions_query: &Query<&Junction>,
    synapses_query: &Query<&mut Synapse>,
    timestamp: &Timestamp,
    state: &mut GpuSimState,
) {
    state.segment_entities.clear();
    state.entity_to_index.clear();
    state.segments.clear();
    state.junctions.clear();
    state.synapses.clear();

    // Extract segments
    for (entity, _seg, solution, geometry, membrane, voltage, maybe_input, maybe_stim) in
        segments_query.iter()
    {
        let idx = state.segments.len() as u32;
        state.entity_to_index.insert(entity, idx);
        state.segment_entities.push(entity);

        let surface_area = geometry.surface_area();
        let (geom_type, geom_diameter, geom_length) = match geometry {
            Geometry::Cylinder { diameter, length } => (0.0, diameter.0, *length),
            Geometry::Sphere { diameter } => (1.0, diameter.0, 0.0),
        };

        let input_current = maybe_input.map_or(0.0, |i| i.0 .0);
        let stimulator_current = maybe_stim.map_or(0.0, |s| s.current(timestamp.clone()).0);
        let total_input = input_current + stimulator_current;

        let num_channels = membrane.membrane_channels.len().min(MAX_CHANNELS);
        let mut channels = [GpuMembraneChannel::zeroed(); MAX_CHANNELS];
        for (i, mc) in membrane.membrane_channels.iter().enumerate().take(MAX_CHANNELS) {
            channels[i] = membrane_channel_to_gpu(mc);
        }

        state.segments.push(GpuSegmentData {
            voltage: voltage.0 .0,
            capacitance: membrane.capacitance.0,
            intra_k: solution.k_concentration.0,
            intra_na: solution.na_concentration.0,
            intra_ca: solution.ca_concentration.0,
            intra_cl: solution.cl_concentration.0,
            geom_type,
            geom_diameter,
            geom_length,
            surface_area,
            input_current: total_input,
            num_channels: num_channels as f32,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
            _pad3: 0.0,
            channels,
        });
    }

    // Extract junctions
    for junction in junctions_query.iter() {
        let first_idx = state
            .entity_to_index
            .get(&junction.first_segment)
            .copied();
        let second_idx = state
            .entity_to_index
            .get(&junction.second_segment)
            .copied();
        if let (Some(first), Some(second)) = (first_idx, second_idx) {
            let conductance =
                junction.pore_diameter.0 * std::f32::consts::PI * CONDUCTANCE_PER_SQUARE_CM;
            state.junctions.push(GpuJunctionData {
                first_segment_idx: first,
                second_segment_idx: second,
                conductance,
                _pad: 0.0,
            });
        }
    }

    // Extract synapses
    for synapse in synapses_query.iter() {
        let pre_idx = state.entity_to_index.get(&synapse.pre_segment).copied();
        let post_idx = state.entity_to_index.get(&synapse.post_segment).copied();
        if let (Some(pre), Some(post)) = (pre_idx, post_idx) {
            let sm = &synapse.synapse_membranes;

            let num_pumps = sm.presynaptic_pumps.len().min(MAX_PUMPS);
            let mut pumps = [GpuTransmitterPump::zeroed(); MAX_PUMPS];
            for (i, pump) in sm.presynaptic_pumps.iter().enumerate().take(MAX_PUMPS) {
                pumps[i] = transmitter_pump_to_gpu(pump);
            }

            let num_receptors = sm.postsynaptic_receptors.len().min(MAX_RECEPTORS);
            let mut receptors = [GpuReceptor::zeroed(); MAX_RECEPTORS];
            for (i, receptor) in sm
                .postsynaptic_receptors
                .iter()
                .enumerate()
                .take(MAX_RECEPTORS)
            {
                receptors[i] = receptor_to_gpu(receptor);
            }

            state.synapses.push(GpuSynapseData {
                pre_segment_idx: pre,
                post_segment_idx: post,
                cleft_k: sm.cleft_solution.k_concentration.0,
                cleft_na: sm.cleft_solution.na_concentration.0,
                cleft_ca: sm.cleft_solution.ca_concentration.0,
                cleft_cl: sm.cleft_solution.cl_concentration.0,
                glutamate_conc: sm.transmitter_concentrations.glutamate.0,
                gaba_conc: sm.transmitter_concentrations.gaba.0,
                surface_area: sm.surface_area.0,
                num_pumps: num_pumps as f32,
                num_receptors: num_receptors as f32,
                _pad: 0.0,
                pumps,
                receptors,
            });
        }
    }

    // Build junction adjacency lists (CSR format)
    let n_segments = state.segments.len();
    let mut junction_neighbors: Vec<Vec<GpuJunctionNeighbor>> = vec![vec![]; n_segments];
    for junc in &state.junctions {
        junction_neighbors[junc.first_segment_idx as usize].push(GpuJunctionNeighbor {
            neighbor_segment_idx: junc.second_segment_idx,
            conductance: junc.conductance,
        });
        junction_neighbors[junc.second_segment_idx as usize].push(GpuJunctionNeighbor {
            neighbor_segment_idx: junc.first_segment_idx,
            conductance: junc.conductance,
        });
    }
    state.junction_adj_offsets = vec![0u32; n_segments + 1];
    for i in 0..n_segments {
        state.junction_adj_offsets[i + 1] =
            state.junction_adj_offsets[i] + junction_neighbors[i].len() as u32;
    }
    state.junction_adj = junction_neighbors.into_iter().flatten().collect();

    // Build synapse adjacency lists (post-segment → synapse indices, CSR format)
    let mut synapse_targets: Vec<Vec<u32>> = vec![vec![]; n_segments];
    for (syn_idx, syn) in state.synapses.iter().enumerate() {
        synapse_targets[syn.post_segment_idx as usize].push(syn_idx as u32);
    }
    state.synapse_adj_offsets = vec![0u32; n_segments + 1];
    for i in 0..n_segments {
        state.synapse_adj_offsets[i + 1] =
            state.synapse_adj_offsets[i] + synapse_targets[i].len() as u32;
    }
    state.synapse_adj = synapse_targets.into_iter().flatten().collect();

    state.initialized = true;
    state.topology_dirty = false;
}

/// Update only the per-frame mutable data (stimulator currents) without
/// rebuilding topology.
pub fn update_input_currents(
    segments_query: &Query<
        (
            Entity,
            &Segment,
            &Solution,
            &Geometry,
            &mut Membrane,
            &mut MembraneVoltage,
            Option<&InputCurrent>,
            Option<&Stimulator>,
        ),
    >,
    timestamp: &Timestamp,
    state: &mut GpuSimState,
) {
    for (entity, _seg, _solution, _geometry, _membrane, _voltage, maybe_input, maybe_stim) in
        segments_query.iter()
    {
        if let Some(&idx) = state.entity_to_index.get(&entity) {
            let input_current = maybe_input.map_or(0.0, |i| i.0 .0);
            let stimulator_current = maybe_stim.map_or(0.0, |s| s.current(timestamp.clone()).0);
            state.segments[idx as usize].input_current = input_current + stimulator_current;
        }
    }
}

/// Write back simulation results from flat arrays to ECS components.
pub fn writeback_to_ecs(
    state: &GpuSimState,
    segments_query: &mut Query<
        (
            Entity,
            &Segment,
            &Solution,
            &Geometry,
            &mut Membrane,
            &mut MembraneVoltage,
            Option<&InputCurrent>,
            Option<&Stimulator>,
        ),
    >,
    synapses_query: &mut Query<&mut Synapse>,
) {
    // Write back voltages and gate states to segments
    for (i, entity) in state.segment_entities.iter().enumerate() {
        if let Ok((_entity, _seg, _sol, _geom, mut membrane, mut voltage, _input, _stim)) =
            segments_query.get_mut(*entity)
        {
            let gpu_seg = &state.segments[i];
            voltage.0 .0 = gpu_seg.voltage;

            // Write back gate magnitudes
            let num_channels = membrane.membrane_channels.len().min(MAX_CHANNELS);
            for c in 0..num_channels {
                gpu_channel_to_ecs(&gpu_seg.channels[c], &mut membrane.membrane_channels[c]);
            }
        }
    }

    // Write back synapse state
    for (i, mut synapse) in synapses_query.iter_mut().enumerate() {
        if i < state.synapses.len() {
            let gpu_syn = &state.synapses[i];
            synapse.synapse_membranes.transmitter_concentrations.glutamate.0 =
                gpu_syn.glutamate_conc;
            synapse.synapse_membranes.transmitter_concentrations.gaba.0 = gpu_syn.gaba_conc;

            // Write back receptor channel gate states
            let num_receptors = synapse
                .synapse_membranes
                .postsynaptic_receptors
                .len()
                .min(MAX_RECEPTORS);
            for r in 0..num_receptors {
                gpu_channel_to_ecs(
                    &gpu_syn.receptors[r].channel,
                    &mut synapse.synapse_membranes.postsynaptic_receptors[r].membrane_channel,
                );
            }
        }
    }
}
