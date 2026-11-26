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

/// `input_current` is in µA/cm²; `scales` holds one conductance multiplier
/// per channel slot.
fn step_channels_cpu(seg: &mut GpuSegmentData, input_current: f32, scales: &[f32], params: &SimParams) {
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

        total_current_per_cm2 += (i_k + i_na + i_ca + i_cl) * ch.siemens_per_square_cm * scales[c];
    }

    // Update voltage from channel currents
    let sa = seg.surface_area;
    let current = -1.0 * total_current_per_cm2 * sa;
    let cap = seg.capacitance * sa;
    let dv_dt = current / cap;
    seg.voltage += 1000.0 * dv_dt * params.dt;

    // Apply input/stimulator current
    let input_current = input_current * 1e-6 * sa;
    let input_dv_dt = input_current / cap;
    seg.voltage += 1000.0 * input_dv_dt * params.dt;

    // Step all gates
    let v_after = seg.voltage;
    for c in 0..n_chan {
        step_gate(&mut seg.channels[c].activation, v_after, params.dt);
        step_gate(&mut seg.channels[c].inactivation, v_after, params.dt);
    }
    count_spike(seg);
}

/// Count an upward crossing of [`SPIKE_THRESHOLD_MV`].
pub fn count_spike(seg: &mut GpuSegmentData) {
    let up = seg.voltage > SPIKE_THRESHOLD_MV;
    if up && seg.above_threshold < 0.5 {
        seg.spike_count += 1.0;
    }
    seg.above_threshold = if up { 1.0 } else { 0.0 };
}

// --- Per-synapse step ---

/// `gain` multiplies every receptor current.
fn step_synapse_cpu(
    syn: &mut GpuSynapseData,
    segments: &mut [GpuSegmentData],
    gain: f32,
    target_hz: f32,
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

    let weight = syn.weight_at_0 + syn.efficacy * (syn.weight_at_1 - syn.weight_at_0);
    let mut total_current_per_cm2: f32 = 0.0;
    let mut calcium_current_per_cm2: f32 = 0.0;
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
        let channel_calcium = i_ca * ch.siemens_per_square_cm;

        // Neurotransmitter gating: the bound fraction relaxes toward the
        // sensitivity curve (or follows it at once without kinetics).
        let conc = if receptor.sens_transmitter < 0.5 {
            syn.glutamate_conc
        } else {
            syn.gaba_conc
        };
        let target =
            1.0 / (1.0 + ((-1.0 * (conc - receptor.sens_conc_half_max) * receptor.sens_slope).exp()));
        receptor.bound = relax_binding(receptor.bound, target, receptor.binding_rise, receptor.binding_decay, params.dt);

        let unblocked = magnesium_unblocked(receptor.magnesium_molar, post_v);
        let scale = if receptor.plastic > 0.5 { weight } else { 1.0 };
        let open = receptor.bound * unblocked * scale;
        total_current_per_cm2 += channel_current * open;
        calcium_current_per_cm2 += channel_calcium * open;
    }

    // Apply synaptic current to postsynaptic segment
    let current_microamps = total_current_per_cm2 * gain * syn.surface_area;
    let synapse_resistance_ohms = params.synapse_resistance_ohms;
    let dv_dt_volts_per_second = current_microamps * 1e-6 * synapse_resistance_ohms * -1.0;
    let delta_mv = dv_dt_volts_per_second * params.dt * 1000.0;
    segments[syn.post_segment_idx as usize].voltage += delta_mv;

    step_calcium_and_efficacy(syn, calcium_current_per_cm2 * gain * syn.surface_area, params.dt);
    step_spike_timing(syn, pre_v, post_v, target_hz, params.dt);
}

/// Pair STDP (see `serialize::SpikeTimingPlasticity`) or, for a
/// `supervised` synapse, ReSuMe toward `target_hz` (see
/// `serialize::SupervisedPlasticity`; negative: no teacher, no learning):
/// decay the traces, detect upward threshold crossings, and move the
/// efficacy. With both spikes in one step, each sees the other's trace from
/// before it.
pub fn step_spike_timing(syn: &mut GpuSynapseData, pre_v: f32, post_v: f32, target_hz: f32, dt: f32) {
    if syn.stdp_pre_tau <= 0.0 {
        return;
    }
    syn.pre_trace *= (-dt / syn.stdp_pre_tau).exp();
    syn.post_trace *= (-dt / syn.stdp_post_tau).exp();
    let pre_up = pre_v > syn.stdp_threshold;
    let post_up = post_v > syn.stdp_threshold;
    let pre_spike = pre_up && syn.pre_above < 0.5;
    let post_spike = post_up && syn.post_above < 0.5;
    syn.pre_above = if pre_up { 1.0 } else { 0.0 };
    syn.post_above = if post_up { 1.0 } else { 0.0 };
    let mut rho = syn.efficacy;
    if syn.supervised > 0.5 {
        if target_hz >= 0.0 {
            rho += syn.stdp_potentiation * syn.pre_trace * target_hz * dt;
            if post_spike {
                rho -= syn.stdp_potentiation * syn.pre_trace;
            }
        }
        if pre_spike {
            syn.pre_trace += 1.0;
        }
    } else {
        if pre_spike {
            rho -= syn.stdp_depression * rho * syn.post_trace;
        }
        if post_spike {
            rho += syn.stdp_potentiation * (1.0 - rho) * syn.pre_trace;
        }
        if pre_spike {
            syn.pre_trace = 1.0;
        }
        if post_spike {
            syn.post_trace = 1.0;
        }
    }
    syn.efficacy = rho.clamp(0.0, 1.0);
}

/// The bound fraction after `dt`, relaxing toward `target` with time
/// constant `rise` (growing) or `decay` (falling); both 0 means no kinetics.
pub fn relax_binding(bound: f32, target: f32, rise: f32, decay: f32, dt: f32) -> f32 {
    let tau = if target > bound { rise } else { decay };
    if tau <= 0.0 {
        return target;
    }
    bound + (target - bound) * (dt / tau).min(1.0)
}

/// Fraction of a Mg2+-blocked pore open at `v` mV (Jahr & Stevens 1990);
/// 1 without magnesium.
pub fn magnesium_unblocked(magnesium_molar: f32, v: f32) -> f32 {
    if magnesium_molar <= 0.0 {
        return 1.0;
    }
    1.0 / (1.0 + magnesium_molar * 1000.0 / 3.57 * (-0.062 * v).exp())
}

/// Advance the spine calcium pool by the receptors' calcium current
/// (`calcium_microamps`, inward negative), then the Graupner–Brunel
/// efficacy by the new calcium level.
pub fn step_calcium_and_efficacy(syn: &mut GpuSynapseData, calcium_microamps: f32, dt: f32) {
    if syn.calcium_decay > 0.0 {
        let influx = -calcium_microamps * syn.calcium_per_microamp_second;
        let decay = (syn.calcium_rest - syn.calcium) / syn.calcium_decay;
        syn.calcium = (syn.calcium + dt * (decay + influx)).max(0.0);
    }
    if syn.plasticity_tau > 0.0 {
        let rho = syn.efficacy;
        let mut drho = -syn.bistability * rho * (1.0 - rho) * (0.5 - rho);
        if syn.calcium > syn.potentiation_threshold {
            drho += syn.potentiation_rate * (1.0 - rho);
        }
        if syn.calcium > syn.depression_threshold {
            drho -= syn.depression_rate * rho;
        }
        syn.efficacy = (rho + dt * drho / syn.plasticity_tau).clamp(0.0, 1.0);
    }
}

// --- Main CPU simulation entry point ---

/// Run the full simulation for `steps_per_dispatch` steps over flat arrays.
/// `controls` is laid out as [`crate::gpu::GpuSimState::controls`].
pub fn step_biophysics_cpu(
    segments: &mut [GpuSegmentData],
    junctions: &[GpuJunctionData],
    synapses: &mut [GpuSynapseData],
    controls: &[f32],
    params: &SimParams,
) {
    let (scales_at, gains_at) = crate::gpu::extract::control_offsets(segments.len());
    let targets_at = gains_at + synapses.len();
    assert_eq!(controls.len(), targets_at + synapses.len(), "controls don't match the flat arrays");
    for _step in 0..params.steps_per_dispatch {
        // Stage 1: Channel currents and voltage update (per-segment, independent)
        for (i, seg) in segments.iter_mut().enumerate() {
            let scales = &controls[scales_at + i * MAX_CHANNELS..scales_at + (i + 1) * MAX_CHANNELS];
            step_channels_cpu(seg, controls[i], scales, params);
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
            step_synapse_cpu(&mut syn, segments, controls[gains_at + syn_idx], controls[targets_at + syn_idx], params);
            synapses[syn_idx] = syn;
        }
    }
}
