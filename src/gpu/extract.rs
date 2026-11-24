//! ECS-to-flat-array extraction for the GPU / CPU simulation backends.

use bevy::prelude::*;
use bytemuck::Zeroable;

use crate::constants::{CONDUCTANCE_PER_SQUARE_CM, INVERSE_FARADAY};
use crate::dimension::Timestamp;
use crate::gpu::buffers::*;
use crate::gpu::GpuSimState;
use crate::integrations::grace::{ModelSynapseGain, Synapse};
use crate::neuron::channel::{GateState, TimeConstant};
use crate::neuron::membrane::{Membrane, MembraneChannel, MembraneVoltage};
use crate::neuron::segment::ecs::{InputCurrent, ModelConductanceScale, ModelCurrent, Segment};
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
        bound: receptor.bound,
        binding_rise: receptor.binding.as_ref().map_or(0.0, |b| b.rise_sec),
        binding_decay: receptor.binding.as_ref().map_or(0.0, |b| b.decay_sec),
        magnesium_molar: receptor.magnesium_block.as_ref().map_or(0.0, |m| m.magnesium_molar),
        plastic: if receptor.plastic { 1.0 } else { 0.0 },
        _pad0: 0.0,
        _pad1: 0.0,
        _pad2: 0.0,
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
    synapses_query: &Query<(Entity, &mut Synapse)>,
    model_currents: &Query<&ModelCurrent>,
    timestamp: &Timestamp,
    state: &mut GpuSimState,
) {
    state.segment_entities.clear();
    state.entity_to_index.clear();
    state.segments.clear();
    state.junctions.clear();
    state.synapses.clear();
    state.synapse_entities.clear();

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
        let model_current = model_currents.get(entity).map_or(0.0, |m| m.0 .0);
        let total_input = input_current + stimulator_current + model_current;

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
    for (synapse_entity, synapse) in synapses_query.iter() {
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

            state.synapse_entities.push(synapse_entity);
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
                calcium: sm.calcium_molar,
                calcium_rest: sm.calcium.as_ref().map_or(0.0, |c| c.resting_molar),
                calcium_decay: sm.calcium.as_ref().map_or(0.0, |c| c.decay_sec),
                calcium_per_microamp_second: sm
                    .calcium
                    .as_ref()
                    .map_or(0.0, |c| 1e-6 * INVERSE_FARADAY / (2.0 * c.spine_volume_cubic_um * 1e-15)),
                efficacy: sm.efficacy,
                plasticity_tau: sm.plasticity.as_ref().map_or(0.0, |p| p.time_constant_sec),
                depression_threshold: sm.plasticity.as_ref().map_or(0.0, |p| p.depression_threshold_molar),
                potentiation_threshold: sm.plasticity.as_ref().map_or(0.0, |p| p.potentiation_threshold_molar),
                depression_rate: sm.plasticity.as_ref().map_or(0.0, |p| p.depression_rate),
                potentiation_rate: sm.plasticity.as_ref().map_or(0.0, |p| p.potentiation_rate),
                bistability: sm.plasticity.as_ref().map_or(0.0, |p| p.bistability),
                weight_at_0: sm.weight_range().0,
                weight_at_1: sm.weight_range().1,
                stdp_threshold: sm.spike_timing.as_ref().map_or(0.0, |p| p.spike_threshold_mv),
                stdp_pre_tau: sm.spike_timing.as_ref().map_or(0.0, |p| p.pre_trace_sec),
                stdp_post_tau: sm.spike_timing.as_ref().map_or(0.0, |p| p.post_trace_sec),
                stdp_potentiation: sm.spike_timing.as_ref().map_or(0.0, |p| p.potentiation),
                stdp_depression: sm.spike_timing.as_ref().map_or(0.0, |p| p.depression),
                pre_trace: sm.spike_traces[0],
                post_trace: sm.spike_traces[1],
                pre_above: sm.spike_traces[2],
                post_above: sm.spike_traces[3],
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
    state.topology_generation += 1;
}

/// What the per-frame controls read from each segment entity.
pub type SegmentControls<'a> = (
    Option<&'a InputCurrent>,
    Option<&'a Stimulator>,
    Option<&'a ModelCurrent>,
    Option<&'a ModelConductanceScale>,
);

/// Rebuild `state.controls` (see [`GpuSimState::controls`]) from the ECS:
/// input currents (input + stimulator + model), conductance multipliers and
/// synapse gains. Runs every frame on both backends.
pub fn fill_controls(
    segments: &Query<SegmentControls>,
    synapse_gains: &Query<&ModelSynapseGain>,
    timestamp: &Timestamp,
    state: &mut GpuSimState,
) {
    let n = state.segment_entities.len();
    state.controls.clear();
    state.controls.resize(n + n * MAX_CHANNELS, 0.0);
    let (currents, scales) = state.controls.split_at_mut(n);
    for (i, entity) in state.segment_entities.iter().enumerate() {
        let Ok((input, stimulator, model_current, model_scale)) = segments.get(*entity) else {
            scales[i * MAX_CHANNELS..(i + 1) * MAX_CHANNELS].fill(1.0);
            continue;
        };
        currents[i] = input.map_or(0.0, |c| c.0 .0)
            + stimulator.map_or(0.0, |s| s.current(timestamp.clone()).0)
            + model_current.map_or(0.0, |m| m.0 .0);
        let factors = model_scale.map_or([1.0; MAX_CHANNELS], |s| s.0);
        scales[i * MAX_CHANNELS..(i + 1) * MAX_CHANNELS].copy_from_slice(&factors);
    }
    for entity in &state.synapse_entities {
        state.controls.push(synapse_gains.get(*entity).map_or(1.0, |g| g.0));
    }
}

/// Offsets into [`GpuSimState::controls`] for `n` segments.
pub fn control_offsets(n: usize) -> (usize, usize) {
    (n, n + n * MAX_CHANNELS)
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
    synapses_query: &mut Query<(Entity, &mut Synapse)>,
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
    for (i, entity) in state.synapse_entities.iter().enumerate() {
        if let Ok((_, mut synapse)) = synapses_query.get_mut(*entity) {
            let gpu_syn = &state.synapses[i];
            synapse.synapse_membranes.transmitter_concentrations.glutamate.0 =
                gpu_syn.glutamate_conc;
            synapse.synapse_membranes.transmitter_concentrations.gaba.0 = gpu_syn.gaba_conc;
            synapse.synapse_membranes.calcium_molar = gpu_syn.calcium;
            synapse.synapse_membranes.efficacy = gpu_syn.efficacy;
            synapse.synapse_membranes.spike_traces =
                [gpu_syn.pre_trace, gpu_syn.post_trace, gpu_syn.pre_above, gpu_syn.post_above];

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
                synapse.synapse_membranes.postsynaptic_receptors[r].bound = gpu_syn.receptors[r].bound;
            }
        }
    }
}
