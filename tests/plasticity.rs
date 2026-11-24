//! Calcium-based synaptic plasticity in the CPU kernel: NMDA receptors with
//! Mg2+ block and slow unbinding feed a spine calcium pool, and the
//! Graupner–Brunel rule turns calcium into lasting changes of the AMPA
//! conductance. Pre- and postsynaptic voltages are clamped to a spike-timing
//! protocol so every step in between is the synapse's own biophysics.

use bytemuck::Zeroable;
use nb_sim::cpu_sim::{magnesium_unblocked, relax_binding, step_biophysics_cpu};
use nb_sim::gpu::buffers::*;
use nb_sim::gpu::extract::control_offsets;

const DT: f32 = 1e-5;
const REST_MV: f32 = -70.0;

fn params() -> SimParams {
    SimParams {
        extra_k: 5.0,
        extra_na: 145.0,
        extra_ca: 2.0,
        extra_cl: 110.0,
        temperature: 310.0,
        dt: DT,
        steps_per_dispatch: 1,
        num_segments: 2,
        num_junctions: 0,
        num_synapses: 1,
        gas_constant: 8.314,
        inverse_faraday: 1.0 / 96485.3,
        conductance_per_square_cm: 0.0,
        synapse_resistance_ohms: 1e9,
        _pad0: 0.0,
        _pad1: 0.0,
    }
}

/// A passive compartment whose voltage the protocol sets every step.
fn clamped() -> GpuSegmentData {
    let mut seg = GpuSegmentData::zeroed();
    seg.voltage = REST_MV;
    seg.capacitance = 1e-6;
    seg.surface_area = 1e-5;
    seg.intra_k = 140e-3;
    seg.intra_na = 10e-3;
    seg.intra_ca = 1e-7;
    seg.intra_cl = 10e-3;
    seg
}

/// Glutamate released while the presynaptic terminal is depolarized,
/// cleared with a 1 ms time constant.
fn glutamate_pump() -> GpuTransmitterPump {
    GpuTransmitterPump {
        present: 1.0,
        transmitter: 0.0,
        target_conc_max: 1.1e-2,
        target_conc_min: 1e-4,
        target_conc_v_half: 0.0,
        target_conc_v_slope: 2.0,
        tc_v_at_max_tau: 0.0,
        tc_c_base: 1e-3,
        tc_c_amp: 0.0,
        tc_sigma: 1.0,
        _pad0: 0.0,
        _pad1: 0.0,
    }
}

fn receptor(ion_na: f32, ion_k: f32, ion_ca: f32, siemens: f32) -> GpuReceptor {
    let mut r = GpuReceptor::zeroed();
    r.present = 1.0;
    r.sens_transmitter = 0.0;
    r.sens_conc_half_max = 3e-3;
    r.sens_slope = 2000.0;
    r.channel.siemens_per_square_cm = siemens;
    r.channel.ion_na = ion_na;
    r.channel.ion_k = ion_k;
    r.channel.ion_ca = ion_ca;
    r
}

/// AMPA (plastic, fast) and NMDA (Mg2+ blocked, slow unbinding, some
/// calcium) receptors, a spine calcium pool, and the plasticity rule.
fn synapse(plasticity: Plasticity) -> GpuSynapseData {
    let mut s = GpuSynapseData::zeroed();
    s.pre_segment_idx = 0;
    s.post_segment_idx = 1;
    s.cleft_k = 5e-3;
    s.cleft_na = 145e-3;
    s.cleft_ca = 2.5e-3;
    s.cleft_cl = 110e-3;
    s.glutamate_conc = 1e-4;
    s.surface_area = 1e-6;
    s.num_pumps = 1.0;
    s.pumps[0] = glutamate_pump();
    s.num_receptors = 2.0;
    let mut ampa = receptor(0.5, 0.5, 0.0, 1e3);
    ampa.plastic = 1.0;
    let mut nmda = receptor(0.45, 0.45, 0.1, 10.0);
    nmda.binding_rise = 5e-3;
    nmda.binding_decay = 50e-3;
    nmda.magnesium_molar = 1e-3;
    s.receptors[0] = ampa;
    s.receptors[1] = nmda;
    s.calcium = 50e-9;
    s.calcium_rest = 50e-9;
    s.calcium_decay = 12e-3;
    // 1 / (2 F V) for a 0.1 µm³ spine.
    s.calcium_per_microamp_second = 1e-6 / (2.0 * 96485.3 * 0.1 * 1e-15);
    s.efficacy = plasticity.initial;
    s.plasticity_tau = plasticity.tau;
    s.depression_threshold = plasticity.theta_d;
    s.potentiation_threshold = plasticity.theta_p;
    s.depression_rate = plasticity.gamma_d;
    s.potentiation_rate = plasticity.gamma_p;
    s.bistability = 0.0;
    s.weight_at_0 = 0.0;
    s.weight_at_1 = 2.0;
    s
}

#[derive(Clone, Copy)]
struct Plasticity {
    initial: f32,
    tau: f32,
    theta_d: f32,
    theta_p: f32,
    gamma_d: f32,
    gamma_p: f32,
}

const OFF: Plasticity = Plasticity { initial: 0.5, tau: 0.0, theta_d: 0.0, theta_p: 0.0, gamma_d: 0.0, gamma_p: 0.0 };

/// Graupner & Brunel's rates, with thresholds placed around this synapse's
/// calcium: a lone presynaptic spike peaks near 1 µM, just over the
/// depression threshold; only a postsynaptic spike while NMDA receptors are
/// still bound pushes calcium well over the potentiation threshold. A 15 s
/// time constant (theirs is 150 s) keeps the test short.
const GB: Plasticity = Plasticity { initial: 0.5, tau: 15.0, theta_d: 0.93e-6, theta_p: 1.1e-6, gamma_d: 200.0, gamma_p: 321.8 };

/// Five pairings at 10 Hz, the postsynaptic spike `lag` seconds after the
/// presynaptic one (before it when negative).
fn pairings(lag: f32) -> Run {
    let pre: Vec<f32> = (0..5).map(|k| 0.05 + 0.1 * k as f32).collect();
    let post: Vec<f32> = pre.iter().map(|t| t + lag).collect();
    run(GB, &pre, &post, 0.6)
}

/// Clamped membrane voltages: a 1 ms presynaptic spike, and a 2 ms
/// back-propagating postsynaptic spike.
fn pre_mv(t: f32, spikes: &[f32]) -> f32 {
    if spikes.iter().any(|s| t >= *s && t < s + 1e-3) { 20.0 } else { REST_MV }
}

fn post_mv(t: f32, spikes: &[f32]) -> f32 {
    if spikes.iter().any(|s| t >= *s && t < s + 2e-3) { 10.0 } else { REST_MV }
}

struct Run {
    peak_calcium: f32,
    efficacy: f32,
}

/// Simulate `duration` seconds of the clamped protocol.
fn run(plasticity: Plasticity, pre: &[f32], post: &[f32], duration: f32) -> Run {
    let mut segs = [clamped(), clamped()];
    let mut syns = [synapse(plasticity)];
    let (scales_at, gains_at) = control_offsets(2);
    let mut controls = vec![0.0; gains_at + 1];
    controls[scales_at..].fill(1.0);
    let params = params();
    let mut peak: f32 = 0.0;
    let steps = (duration / DT) as usize;
    for k in 0..steps {
        let t = k as f32 * DT;
        segs[0].voltage = pre_mv(t, pre);
        segs[1].voltage = post_mv(t, post);
        step_biophysics_cpu(&mut segs, &[], &mut syns, &controls, &params);
        peak = peak.max(syns[0].calcium);
    }
    Run { peak_calcium: peak, efficacy: syns[0].efficacy }
}

#[test]
fn magnesium_blocks_at_rest_and_unblocks_when_depolarized() {
    // Jahr & Stevens with 1 mM Mg2+: about 4% open at -70 mV, 78% at 0 mV.
    assert!((magnesium_unblocked(1e-3, -70.0) - 0.044).abs() < 0.002);
    assert!((magnesium_unblocked(1e-3, 0.0) - 0.781).abs() < 0.002);
    assert_eq!(magnesium_unblocked(0.0, -70.0), 1.0);
}

#[test]
fn binding_rises_and_decays_with_its_own_time_constants() {
    assert_eq!(relax_binding(0.2, 0.9, 0.0, 0.0, 1e-5), 0.9, "no kinetics: follow at once");
    // One step of 1 ms toward 1 with a 5 ms rise; toward 0 with an 80 ms decay.
    assert!((relax_binding(0.0, 1.0, 5e-3, 80e-3, 1e-3) - 0.2).abs() < 1e-6);
    assert!((relax_binding(1.0, 0.0, 5e-3, 80e-3, 1e-3) - (1.0 - 1.0 / 80.0)).abs() < 1e-6);
}

#[test]
fn nmda_calcium_detects_pre_then_post_coincidence() {
    let pre_alone = run(OFF, &[0.01], &[], 0.15).peak_calcium;
    let post_alone = run(OFF, &[], &[0.01], 0.15).peak_calcium;
    let pre_post = run(OFF, &[0.01], &[0.02], 0.15).peak_calcium;
    let post_pre = run(OFF, &[0.02], &[0.01], 0.15).peak_calcium;
    // Without glutamate there is almost no NMDA current; with it alone the
    // Mg2+ block passes a little; a postsynaptic spike while glutamate is
    // still bound expels the Mg2+ and roughly doubles the calcium.
    assert!(post_alone < 0.2e-6, "post alone {post_alone:e}");
    assert!((0.9e-6..1.1e-6).contains(&pre_alone), "pre alone {pre_alone:e}");
    assert!(pre_post > 1.5 * pre_alone, "pre then post {pre_post:e}");
    // The postsynaptic spike before release finds nothing bound to unblock.
    assert!((post_pre - pre_alone).abs() < 0.05 * pre_alone, "post then pre {post_pre:e}");
}

#[test]
fn spike_timing_sets_the_sign_of_plasticity() {
    let quiet = run(GB, &[], &[], 0.6).efficacy;
    assert_eq!(quiet, 0.5, "resting calcium is below both thresholds");
    // The Bi & Poo window: potentiation when the presynaptic spike leads by
    // a few ms, depression when it follows, and depression again when the
    // postsynaptic spike comes too late to meet bound NMDA receptors.
    for lag in [0.005, 0.01, 0.02] {
        let e = pairings(lag).efficacy;
        assert!(e > 0.52, "pre {} ms before post: efficacy {e}", lag * 1e3);
    }
    for lag in [-0.005, -0.01, -0.02, -0.04] {
        let e = pairings(lag).efficacy;
        assert!(e < 0.47, "post {} ms before pre: efficacy {e}", -lag * 1e3);
    }
    assert!(pairings(0.08).efficacy < 0.45, "unpaired release depresses");
    assert!(pairings(0.005).efficacy > pairings(0.04).efficacy);
}

#[test]
fn the_weight_scales_only_plastic_receptors() {
    // At efficacy 1 the AMPA conductance doubles (weight_at_1 = 2); NMDA,
    // not plastic, is unchanged. Compare the postsynaptic voltage change of
    // one step with glutamate fully bound, AMPA only vs NMDA only.
    let delta = |efficacy: f32, keep: usize| {
        let mut segs = [clamped(), clamped()];
        segs[1].voltage = -20.0;
        let mut syn = synapse(OFF);
        syn.efficacy = efficacy;
        syn.glutamate_conc = 1.1e-2;
        for r in 0..2 {
            syn.receptors[r].bound = 1.0;
            syn.receptors[r].binding_rise = 0.0;
            syn.receptors[r].binding_decay = 0.0;
            // Large enough for a voltage change well above f32 noise.
            syn.receptors[r].channel.siemens_per_square_cm *= 1e4;
            if r != keep {
                syn.receptors[r].present = 0.0;
            }
        }
        segs[0].voltage = 20.0;
        let mut syns = [syn];
        let (scales_at, gains_at) = control_offsets(2);
        let mut controls = vec![0.0; gains_at + 1];
        controls[scales_at..].fill(1.0);
        step_biophysics_cpu(&mut segs, &[], &mut syns, &controls, &params());
        segs[1].voltage + 20.0
    };
    let (ampa_low, ampa_high) = (delta(0.25, 0), delta(1.0, 0));
    assert!((ampa_high / ampa_low - 2.0 / 0.5).abs() < 1e-3, "{ampa_low} {ampa_high}");
    assert_eq!(delta(0.25, 1), delta(1.0, 1));
}


