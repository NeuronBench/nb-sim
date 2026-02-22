// biophysics.wgsl — Neuron biophysics compute shader.
//
// Mirrors the CPU implementation in src/cpu_sim/mod.rs.
// Dispatched as 5 passes per simulation step:
//   1. step_channels        (per-segment): channel currents, voltage update, gate stepping
//   2. compute_junction_deltas (per-junction): gap junction voltage deltas
//   3. apply_junction_deltas   (per-segment): accumulate junction deltas into voltage
//   4. step_synapses           (per-synapse): transmitter pumps, receptor currents
//   5. apply_synapse_deltas    (per-segment): accumulate synapse deltas into voltage

// --- Struct definitions (must match Rust #[repr(C)] layout in gpu/buffers.rs) ---

struct GpuGateState {
    present: f32,
    gates: f32,
    magnitude: f32,
    ss_v_at_half_max: f32,
    ss_slope: f32,
    tc_type: f32,
    // Gaussian: v_at_max_tau. LinearExp: coef.
    tc_param0: f32,
    // Gaussian: c_base. LinearExp: v_offset.
    tc_param1: f32,
    // Gaussian: c_amp. LinearExp: inner_coef.
    tc_param2: f32,
    // Gaussian: sigma. LinearExp: unused.
    tc_param3: f32,
    _pad0: f32,
    _pad1: f32,
}

struct GpuMembraneChannel {
    siemens_per_square_cm: f32,
    ion_k: f32,
    ion_na: f32,
    ion_ca: f32,
    ion_cl: f32,
    _pad: f32,
    activation: GpuGateState,
    inactivation: GpuGateState,
}

struct GpuSegmentData {
    voltage: f32,
    capacitance: f32,
    intra_k: f32,
    intra_na: f32,
    intra_ca: f32,
    intra_cl: f32,
    geom_type: f32,
    geom_diameter: f32,
    geom_length: f32,
    surface_area: f32,
    input_current: f32,
    num_channels: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
    channels: array<GpuMembraneChannel, 8>,
}

struct GpuJunctionData {
    first_segment_idx: u32,
    second_segment_idx: u32,
    conductance: f32,
    _pad: f32,
}

struct GpuTransmitterPump {
    present: f32,
    transmitter: f32,
    target_conc_max: f32,
    target_conc_min: f32,
    target_conc_v_half: f32,
    target_conc_v_slope: f32,
    tc_v_at_max_tau: f32,
    tc_c_base: f32,
    tc_c_amp: f32,
    tc_sigma: f32,
    _pad0: f32,
    _pad1: f32,
}

struct GpuReceptor {
    present: f32,
    sens_transmitter: f32,
    sens_conc_half_max: f32,
    sens_slope: f32,
    channel: GpuMembraneChannel,
}

struct GpuSynapseData {
    pre_segment_idx: u32,
    post_segment_idx: u32,
    cleft_k: f32,
    cleft_na: f32,
    cleft_ca: f32,
    cleft_cl: f32,
    glutamate_conc: f32,
    gaba_conc: f32,
    surface_area: f32,
    num_pumps: f32,
    num_receptors: f32,
    _pad: f32,
    pumps: array<GpuTransmitterPump, 4>,
    receptors: array<GpuReceptor, 4>,
}

struct SimParams {
    extra_k: f32,
    extra_na: f32,
    extra_ca: f32,
    extra_cl: f32,
    temperature: f32,
    dt: f32,
    steps_per_dispatch: u32,
    num_segments: u32,
    num_junctions: u32,
    num_synapses: u32,
    gas_constant: f32,
    inverse_faraday: f32,
    conductance_per_square_cm: f32,
    synapse_resistance_ohms: f32,
    _pad0: f32,
    _pad1: f32,
}

// --- Buffer bindings ---

@group(0) @binding(0) var<storage, read_write> segments: array<GpuSegmentData>;
@group(0) @binding(1) var<storage, read>       junctions: array<GpuJunctionData>;
@group(0) @binding(2) var<storage, read_write> synapses: array<GpuSynapseData>;
@group(0) @binding(3) var<uniform>             params: SimParams;
// Per-junction scratch: [j*2] = delta for first_segment, [j*2+1] = delta for second_segment
@group(0) @binding(4) var<storage, read_write> junction_deltas: array<f32>;
// Per-synapse scratch: voltage delta for post_segment
@group(0) @binding(5) var<storage, read_write> synapse_deltas: array<f32>;
// Per-segment input currents (stimulators + InputCurrent), uploaded from CPU each frame
@group(0) @binding(6) var<storage, read> input_currents: array<f32>;
// Per-segment voltage output for readback to CPU (written once per frame after all steps)
@group(0) @binding(7) var<storage, read_write> voltages_out: array<f32>;

// --- Helper functions ---

fn steady_state(v: f32, v_at_half_max: f32, slope: f32) -> f32 {
    return 1.0 / (1.0 + exp((v_at_half_max - v) / slope));
}

// Returns vec2(has_tau, tau_value). has_tau < 0.5 means instantaneous.
fn gate_tau(gate: GpuGateState, v: f32) -> vec2<f32> {
    if gate.tc_type < 0.5 {
        // Instantaneous
        return vec2<f32>(0.0, 0.0);
    } else if gate.tc_type < 1.5 {
        // Gaussian: c_base + c_amp * exp(-(v_at_max_tau - v)^2 / sigma^2)
        let diff = gate.tc_param0 - v;
        let sigma_sq = gate.tc_param3 * gate.tc_param3;
        let t = gate.tc_param1 + gate.tc_param2 * exp(-1.0 * diff * diff / sigma_sq);
        return vec2<f32>(1.0, t);
    } else {
        // LinearExp: coef * exp((v_offset - v) * inner_coef) * 0.001
        let t = gate.tc_param0 * exp((gate.tc_param1 - v) * gate.tc_param2) * 0.001;
        return vec2<f32>(1.0, t);
    }
}

fn step_gate(gate: ptr<function, GpuGateState>, v: f32, dt: f32) {
    if (*gate).present < 0.5 {
        return;
    }
    let v_inf = steady_state(v, (*gate).ss_v_at_half_max, (*gate).ss_slope);
    let tau_result = gate_tau(*gate, v);
    if tau_result.x < 0.5 {
        // Instantaneous: snap to steady state
        (*gate).magnitude = v_inf;
    } else {
        let df_dt = (v_inf - (*gate).magnitude) / tau_result.y;
        (*gate).magnitude = clamp((*gate).magnitude + df_dt * dt, -1.0, 1.0);
    }
}

fn conductance_coefficient(gate: GpuGateState) -> f32 {
    if gate.present < 0.5 {
        return 1.0;
    }
    return pow(gate.magnitude, gate.gates);
}

fn reversal_potential(intra: f32, extra: f32, temperature: f32, valence: f32) -> f32 {
    return params.gas_constant * params.inverse_faraday * temperature / valence
           * log(extra / intra) * 1000.0;
}

// ===========================================================================
// Entry point 1: Channel currents and voltage update (one thread per segment)
// ===========================================================================

@compute @workgroup_size(64)
fn step_channels(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= params.num_segments {
        return;
    }

    var seg = segments[idx];
    let v = seg.voltage;
    let dt = params.dt;

    // Reversal potentials
    let e_k  = reversal_potential(seg.intra_k,  params.extra_k,  params.temperature, 1.0);
    let e_na = reversal_potential(seg.intra_na, params.extra_na, params.temperature, 1.0);
    let e_cl = reversal_potential(seg.intra_cl, params.extra_cl, params.temperature, -1.0);
    let e_ca = reversal_potential(seg.intra_ca, params.extra_ca, params.temperature, 2.0);

    // Sum channel currents
    let n_chan = u32(seg.num_channels);
    var total_current_per_cm2: f32 = 0.0;
    for (var c: u32 = 0u; c < n_chan; c++) {
        let ch = seg.channels[c];
        let act_coef = conductance_coefficient(ch.activation);
        let inact_coef = conductance_coefficient(ch.inactivation);
        let g_coef = act_coef * inact_coef;

        let i_k  = ch.ion_k  * g_coef * (v - e_k)  * 0.001;
        let i_na = ch.ion_na * g_coef * (v - e_na) * 0.001;
        let i_ca = ch.ion_ca * g_coef * (v - e_ca) * 0.001;
        let i_cl = ch.ion_cl * g_coef * (v - e_cl) * 0.001;

        total_current_per_cm2 += (i_k + i_na + i_ca + i_cl) * ch.siemens_per_square_cm;
    }

    // Voltage update from channel currents
    let sa = seg.surface_area;
    let cap = seg.capacitance * sa;
    let current = -1.0 * total_current_per_cm2 * sa;
    seg.voltage += 1000.0 * (current / cap) * dt;

    // Stimulator / input current (read from separate buffer, uploaded from CPU each frame)
    let input_current = input_currents[idx] * 1e-6 * sa;
    seg.voltage += 1000.0 * (input_current / cap) * dt;

    // Step all gates at the new voltage
    let v_after = seg.voltage;
    for (var c: u32 = 0u; c < n_chan; c++) {
        step_gate(&seg.channels[c].activation, v_after, dt);
        step_gate(&seg.channels[c].inactivation, v_after, dt);
    }

    // Write back
    segments[idx] = seg;
}

// ===========================================================================
// Entry point 2: Compute junction deltas (one thread per junction)
// ===========================================================================

@compute @workgroup_size(64)
fn compute_junction_deltas(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= params.num_junctions {
        return;
    }

    let junc = junctions[idx];
    let v1 = segments[junc.first_segment_idx].voltage;
    let v2 = segments[junc.second_segment_idx].voltage;
    let cap1 = segments[junc.first_segment_idx].capacitance
             * segments[junc.first_segment_idx].surface_area;
    let cap2 = segments[junc.second_segment_idx].capacitance
             * segments[junc.second_segment_idx].surface_area;

    let first_to_second = junc.conductance * (v1 - v2) * 1e-3;

    // Store per-junction deltas: [j*2] for first segment, [j*2+1] for second
    junction_deltas[idx * 2u]      = -first_to_second / cap1 * params.dt;
    junction_deltas[idx * 2u + 1u] =  first_to_second / cap2 * params.dt;
}

// ===========================================================================
// Entry point 3: Apply junction deltas (one thread per segment, gathers)
// ===========================================================================

@compute @workgroup_size(64)
fn apply_junction_deltas(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let seg_idx = global_id.x;
    if seg_idx >= params.num_segments {
        return;
    }

    var delta: f32 = 0.0;
    for (var j: u32 = 0u; j < params.num_junctions; j++) {
        if junctions[j].first_segment_idx == seg_idx {
            delta += junction_deltas[j * 2u];
        }
        if junctions[j].second_segment_idx == seg_idx {
            delta += junction_deltas[j * 2u + 1u];
        }
    }

    segments[seg_idx].voltage += delta;
}

// ===========================================================================
// Entry point 4: Synapse step (one thread per synapse)
// ===========================================================================

@compute @workgroup_size(64)
fn step_synapses(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= params.num_synapses {
        return;
    }

    var syn = synapses[idx];
    let pre_v = segments[syn.pre_segment_idx].voltage;
    let post_v = segments[syn.post_segment_idx].voltage;
    let post_seg = segments[syn.post_segment_idx];
    let dt = params.dt;

    // --- Step transmitter pumps ---
    let num_pumps = u32(syn.num_pumps);
    for (var p: u32 = 0u; p < num_pumps; p++) {
        let pump = syn.pumps[p];
        if pump.present < 0.5 {
            continue;
        }

        // Target concentration (sigmoid of presynaptic voltage)
        let target_conc = pump.target_conc_min
            + (pump.target_conc_max - pump.target_conc_min)
              / (1.0 + exp((pump.target_conc_v_half - pre_v) / pump.target_conc_v_slope));

        // Time constant (Gaussian of presynaptic voltage)
        let diff = pump.tc_v_at_max_tau - pre_v;
        let sigma_sq = pump.tc_sigma * pump.tc_sigma;
        let tc = pump.tc_c_base + pump.tc_c_amp * exp(-1.0 * diff * diff / sigma_sq);

        if pump.transmitter < 0.5 {
            // Glutamate
            let slope = (target_conc - syn.glutamate_conc) / tc;
            syn.glutamate_conc += slope * dt;
        } else {
            // GABA
            let slope = (target_conc - syn.gaba_conc) / tc;
            syn.gaba_conc += slope * dt;
        }
    }

    // --- Step receptor channels and compute synaptic current ---
    // Reversal potentials using cleft as extracellular
    let e_k  = reversal_potential(post_seg.intra_k,  syn.cleft_k,  params.temperature, 1.0);
    let e_na = reversal_potential(post_seg.intra_na, syn.cleft_na, params.temperature, 1.0);
    let e_cl = reversal_potential(post_seg.intra_cl, syn.cleft_cl, params.temperature, -1.0);
    let e_ca = reversal_potential(post_seg.intra_ca, syn.cleft_ca, params.temperature, 2.0);

    let num_receptors = u32(syn.num_receptors);
    var total_current_per_cm2: f32 = 0.0;
    for (var r: u32 = 0u; r < num_receptors; r++) {
        if syn.receptors[r].present < 0.5 {
            continue;
        }

        // Step receptor channel gates with postsynaptic voltage
        step_gate(&syn.receptors[r].channel.activation, post_v, dt);
        step_gate(&syn.receptors[r].channel.inactivation, post_v, dt);

        // Channel current
        let ch = syn.receptors[r].channel;
        let act_coef = conductance_coefficient(ch.activation);
        let inact_coef = conductance_coefficient(ch.inactivation);
        let g_coef = act_coef * inact_coef;

        let i_k  = ch.ion_k  * g_coef * (post_v - e_k)  * 0.001;
        let i_na = ch.ion_na * g_coef * (post_v - e_na) * 0.001;
        let i_ca = ch.ion_ca * g_coef * (post_v - e_ca) * 0.001;
        let i_cl = ch.ion_cl * g_coef * (post_v - e_cl) * 0.001;

        let channel_current = (i_k + i_na + i_ca + i_cl) * ch.siemens_per_square_cm;

        // Neurotransmitter gating
        var conc: f32;
        if syn.receptors[r].sens_transmitter < 0.5 {
            conc = syn.glutamate_conc;
        } else {
            conc = syn.gaba_conc;
        }
        let gating = 1.0 / (1.0 + exp(-1.0 * (conc - syn.receptors[r].sens_conc_half_max)
                                             * syn.receptors[r].sens_slope));

        total_current_per_cm2 += channel_current * gating;
    }

    // Voltage delta for postsynaptic segment (V = I * R model)
    let current_microamps = total_current_per_cm2 * syn.surface_area;
    let dv = current_microamps * 1e-6 * params.synapse_resistance_ohms * -1.0 * dt * 1000.0;
    synapse_deltas[idx] = dv;

    // Write back updated synapse (transmitter concentrations + gate states)
    synapses[idx] = syn;
}

// ===========================================================================
// Entry point 5: Apply synapse deltas (one thread per segment, gathers)
// ===========================================================================

@compute @workgroup_size(64)
fn apply_synapse_deltas(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let seg_idx = global_id.x;
    if seg_idx >= params.num_segments {
        return;
    }

    var delta: f32 = 0.0;
    for (var s: u32 = 0u; s < params.num_synapses; s++) {
        if synapses[s].post_segment_idx == seg_idx {
            delta += synapse_deltas[s];
        }
    }

    segments[seg_idx].voltage += delta;
}

// ===========================================================================
// Entry point 6: Copy voltages to output buffer for CPU readback (per-segment)
// ===========================================================================

@compute @workgroup_size(64)
fn write_voltages(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= params.num_segments {
        return;
    }
    voltages_out[idx] = segments[idx].voltage;
}
