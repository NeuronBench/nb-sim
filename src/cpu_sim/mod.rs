//! CPU fallback simulation over flat GPU buffer arrays.
//!
//! This mirrors the WGSL shader logic exactly, operating on the same
//! GpuSegmentData / GpuJunctionData / GpuSynapseData structs.

use crate::gpu::buffers::*;

// --- Gate / channel helpers ---

fn steady_state(v: f32, v_at_half_max: f32, slope: f32) -> f32 {
    1.0 / (1.0 + ((v_at_half_max - v) / slope).exp())
}

fn tau(gate: &GpuGateState, v: f32) -> Option<f32> {
    if gate.tc_type < 0.5 {
        // Instantaneous
        None
    } else if gate.tc_type < 1.5 {
        // Gaussian
        let num = -1.0 * (gate.tc_param0 - v).powi(2);
        let den = gate.tc_param3.powi(2);
        Some(gate.tc_param1 + gate.tc_param2 * (num / den).exp())
    } else {
        // LinearExp
        Some(gate.tc_param0 * ((gate.tc_param1 - v) * gate.tc_param2).exp() * 0.001)
    }
}

fn step_gate(gate: &mut GpuGateState, v: f32, dt: f32) {
    if gate.present < 0.5 {
        return;
    }
    let v_inf = steady_state(v, gate.ss_v_at_half_max, gate.ss_slope);
    match tau(gate, v) {
        None => {
            gate.magnitude = v_inf;
        }
        Some(t) => {
            let df_dt = (v_inf - gate.magnitude) / t;
            gate.magnitude = (gate.magnitude + df_dt * dt).clamp(-1.0, 1.0);
        }
    }
}

fn conductance_coefficient(gate: &GpuGateState) -> f32 {
    if gate.present < 0.5 {
        1.0
    } else {
        gate.magnitude.powf(gate.gates)
    }
}

fn reversal_potential(intra: f32, extra: f32, temperature: f32, valence: f32) -> f32 {
    let gas_constant: f32 = 8.314;
    let inverse_faraday: f32 = 1.0 / 96485.3;
    gas_constant * inverse_faraday * temperature / valence * (extra / intra).ln() * 1000.0
}

// --- Per-segment channel step ---

fn step_channels_cpu(seg: &mut GpuSegmentData, params: &SimParams) {
    let v = seg.voltage;
    let temp = params.temperature;

    // Reversal potentials
    let e_k = reversal_potential(seg.intra_k, params.extra_k, temp, 1.0);
    let e_na = reversal_potential(seg.intra_na, params.extra_na, temp, 1.0);
    let e_cl = reversal_potential(seg.intra_cl, params.extra_cl, temp, -1.0);
    let e_ca = reversal_potential(seg.intra_ca, params.extra_ca, temp, 2.0);

    // Sum channel currents
    let n_chan = seg.num_channels as usize;
    let mut total_current_per_cm2: f32 = 0.0;
    for c in 0..n_chan {
        let ch = &seg.channels[c];
        let act_coef = conductance_coefficient(&ch.activation);
        let inact_coef = conductance_coefficient(&ch.inactivation);
        let g_coef = act_coef * inact_coef;

        let i_k = ch.ion_k * g_coef * (v - e_k) * 0.001;
        let i_na = ch.ion_na * g_coef * (v - e_na) * 0.001;
        let i_ca = ch.ion_ca * g_coef * (v - e_ca) * 0.001;
        let i_cl = ch.ion_cl * g_coef * (v - e_cl) * 0.001;

        total_current_per_cm2 += (i_k + i_na + i_ca + i_cl) * ch.siemens_per_square_cm;
    }

    // Update voltage from channel currents
    let sa = seg.surface_area;
    let current = -1.0 * total_current_per_cm2 * sa;
    let cap = seg.capacitance * sa;
    let dv_dt = current / cap;
    seg.voltage += 1000.0 * dv_dt * params.dt;

    // Apply input/stimulator current
    let input_current = seg.input_current * 1e-6 * sa;
    let input_dv_dt = input_current / cap;
    seg.voltage += 1000.0 * input_dv_dt * params.dt;

    // Step all gates
    let v_after = seg.voltage;
    for c in 0..n_chan {
        step_gate(&mut seg.channels[c].activation, v_after, params.dt);
        step_gate(&mut seg.channels[c].inactivation, v_after, params.dt);
    }
}

// --- Per-synapse step ---

fn step_synapse_cpu(
    syn: &mut GpuSynapseData,
    segments: &mut [GpuSegmentData],
    params: &SimParams,
) {
    let pre_v = segments[syn.pre_segment_idx as usize].voltage;
    let post_v = segments[syn.post_segment_idx as usize].voltage;
    let post_seg = &segments[syn.post_segment_idx as usize];
    let post_intra_k = post_seg.intra_k;
    let post_intra_na = post_seg.intra_na;
    let post_intra_ca = post_seg.intra_ca;
    let post_intra_cl = post_seg.intra_cl;

    // Step transmitter pumps
    let num_pumps = syn.num_pumps as usize;
    for p in 0..num_pumps {
        let pump = &syn.pumps[p];
        if pump.present < 0.5 {
            continue;
        }

        // Target concentration (sigmoid of presynaptic voltage)
        let target = pump.target_conc_min
            + (pump.target_conc_max - pump.target_conc_min)
                / (1.0 + ((pump.target_conc_v_half - pre_v) / pump.target_conc_v_slope).exp());

        // Time constant (Gaussian of presynaptic voltage)
        let num = -1.0 * (pump.tc_v_at_max_tau - pre_v).powi(2);
        let den = pump.tc_sigma.powi(2);
        let tc = pump.tc_c_base + pump.tc_c_amp * (num / den).exp();

        // Update the appropriate transmitter concentration
        if pump.transmitter < 0.5 {
            // Glutamate
            let slope = (target - syn.glutamate_conc) / tc;
            syn.glutamate_conc += slope * params.dt;
        } else {
            // GABA
            let slope = (target - syn.gaba_conc) / tc;
            syn.gaba_conc += slope * params.dt;
        }
    }

    // Step receptor channels and compute synaptic current
    let num_receptors = syn.num_receptors as usize;
    let temp = params.temperature;

    // Reversal potentials for postsynaptic current (using cleft as extracellular)
    let e_k = reversal_potential(post_intra_k, syn.cleft_k, temp, 1.0);
    let e_na = reversal_potential(post_intra_na, syn.cleft_na, temp, 1.0);
    let e_cl = reversal_potential(post_intra_cl, syn.cleft_cl, temp, -1.0);
    let e_ca = reversal_potential(post_intra_ca, syn.cleft_ca, temp, 2.0);

    let mut total_current_per_cm2: f32 = 0.0;
    for r in 0..num_receptors {
        let receptor = &mut syn.receptors[r];
        if receptor.present < 0.5 {
            continue;
        }

        // Step receptor channel gates with postsynaptic voltage
        step_gate(&mut receptor.channel.activation, post_v, params.dt);
        step_gate(&mut receptor.channel.inactivation, post_v, params.dt);

        // Compute channel current
        let ch = &receptor.channel;
        let act_coef = conductance_coefficient(&ch.activation);
        let inact_coef = conductance_coefficient(&ch.inactivation);
        let g_coef = act_coef * inact_coef;

        let i_k = ch.ion_k * g_coef * (post_v - e_k) * 0.001;
        let i_na = ch.ion_na * g_coef * (post_v - e_na) * 0.001;
        let i_ca = ch.ion_ca * g_coef * (post_v - e_ca) * 0.001;
        let i_cl = ch.ion_cl * g_coef * (post_v - e_cl) * 0.001;

        let channel_current = (i_k + i_na + i_ca + i_cl) * ch.siemens_per_square_cm;

        // Neurotransmitter gating
        let conc = if receptor.sens_transmitter < 0.5 {
            syn.glutamate_conc
        } else {
            syn.gaba_conc
        };
        let gating =
            1.0 / (1.0 + ((-1.0 * (conc - receptor.sens_conc_half_max) * receptor.sens_slope).exp()));

        total_current_per_cm2 += channel_current * gating;
    }

    // Apply synaptic current to postsynaptic segment
    let current_microamps = total_current_per_cm2 * syn.surface_area;
    let synapse_resistance_ohms = params.synapse_resistance_ohms;
    let dv_dt_volts_per_second = current_microamps * 1e-6 * synapse_resistance_ohms * -1.0;
    let delta_mv = dv_dt_volts_per_second * params.dt * 1000.0;
    segments[syn.post_segment_idx as usize].voltage += delta_mv;
}

// --- Main CPU simulation entry point ---

/// Run the full simulation for `steps_per_dispatch` steps over flat arrays.
pub fn step_biophysics_cpu(
    segments: &mut [GpuSegmentData],
    junctions: &[GpuJunctionData],
    synapses: &mut [GpuSynapseData],
    params: &SimParams,
) {
    for _step in 0..params.steps_per_dispatch {
        // Stage 1: Channel currents and voltage update (per-segment, independent)
        for seg in segments.iter_mut() {
            step_channels_cpu(seg, params);
        }

        // Stage 2: Gap junction coupling
        // Accumulate voltage deltas first, then apply (avoids order-dependent results)
        if !junctions.is_empty() {
            let mut voltage_deltas = vec![0.0f32; segments.len()];
            for junc in junctions.iter() {
                let idx1 = junc.first_segment_idx as usize;
                let idx2 = junc.second_segment_idx as usize;
                let v1 = segments[idx1].voltage;
                let v2 = segments[idx2].voltage;
                let cap1 = segments[idx1].capacitance * segments[idx1].surface_area;
                let cap2 = segments[idx2].capacitance * segments[idx2].surface_area;
                let first_to_second_current = junc.conductance * (v1 - v2) * 1e-3;
                voltage_deltas[idx1] -= first_to_second_current / cap1 * params.dt;
                voltage_deltas[idx2] += first_to_second_current / cap2 * params.dt;
            }
            for (i, seg) in segments.iter_mut().enumerate() {
                seg.voltage += voltage_deltas[i];
            }
        }

        // Stage 3: Synaptic transmission
        for syn_idx in 0..synapses.len() {
            // Split borrow: take synapse out, operate, put back
            let mut syn = synapses[syn_idx];
            step_synapse_cpu(&mut syn, segments, params);
            synapses[syn_idx] = syn;
        }
    }
}
