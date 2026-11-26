//! Pair-based STDP in the CPU kernel (the WGSL kernel mirrors it): spikes are
//! upward threshold crossings of the clamped pre- and postsynaptic voltages,
//! and each pairing moves the efficacy by an amount that falls off
//! exponentially with the delay, up when pre leads and down when it lags.

use bytemuck::Zeroable;
use nb_sim::cpu_sim::{step_biophysics_cpu, step_spike_timing};
use nb_sim::gpu::buffers::*;
use nb_sim::gpu::extract::neutral_controls;

const DT: f32 = 1e-5;
const TAU_PRE: f32 = 0.017;
const TAU_POST: f32 = 0.034;
const A_PLUS: f32 = 0.1;
const A_MINUS: f32 = 0.05;

fn synapse(efficacy: f32) -> GpuSynapseData {
    let mut s = GpuSynapseData::zeroed();
    s.pre_segment_idx = 0;
    s.post_segment_idx = 1;
    s.efficacy = efficacy;
    s.weight_at_0 = 0.0;
    s.weight_at_1 = 2.0;
    s.stdp_threshold = -20.0;
    s.stdp_pre_tau = TAU_PRE;
    s.stdp_post_tau = TAU_POST;
    s.stdp_potentiation = A_PLUS;
    s.stdp_depression = A_MINUS;
    s
}

/// 1 ms spikes to +20 mV from -70 mV.
fn mv(t: f32, spikes: &[f32]) -> f32 {
    if spikes.iter().any(|s| t >= *s && t < s + 1e-3) { 20.0 } else { -70.0 }
}

/// Drive the rule directly with clamped voltages; the final efficacy.
fn run(efficacy: f32, pre: &[f32], post: &[f32], duration: f32) -> f32 {
    let mut s = synapse(efficacy);
    for k in 0..(duration / DT).round() as usize {
        let t = k as f32 * DT;
        step_spike_timing(&mut s, mv(t, pre), mv(t, post), -1.0, DT);
    }
    s.efficacy
}

#[test]
fn one_pairing_follows_the_exponential_window() {
    for lag_ms in [2.0f32, 5.0, 10.0, 20.0, 40.0] {
        let lag = lag_ms * 1e-3;
        // Pre leads: potentiation A+ (1 - rho) exp(-lag / tau+).
        let up = run(0.5, &[0.01], &[0.01 + lag], 0.1) - 0.5;
        let want = A_PLUS * 0.5 * (-lag / TAU_PRE).exp();
        assert!((up - want).abs() < 0.02 * want, "pre leads by {lag_ms} ms: {up} vs {want}");
        // Pre lags: depression A- rho exp(-lag / tau-).
        let down = run(0.5, &[0.01 + lag], &[0.01], 0.1) - 0.5;
        let want = -A_MINUS * 0.5 * (-lag / TAU_POST).exp();
        assert!((down - want).abs() < 0.02 * want.abs(), "pre lags by {lag_ms} ms: {down} vs {want}");
    }
}

#[test]
fn lone_spikes_and_quiet_leave_the_synapse_alone() {
    assert_eq!(run(0.5, &[], &[], 0.1), 0.5);
    assert_eq!(run(0.5, &[0.01, 0.03, 0.05], &[], 0.1), 0.5);
    assert_eq!(run(0.5, &[], &[0.01, 0.03, 0.05], 0.1), 0.5);
}

#[test]
fn a_long_spike_counts_once() {
    // A 1 ms spike spans 100 steps above threshold: one pairing, not 100.
    let once = run(0.5, &[0.01], &[0.015], 0.05);
    assert!(once < 0.5 + A_PLUS, "{once}");
}

#[test]
fn soft_bounds_saturate_inside_0_and_1() {
    // 300 ms apart, so each spike pairs only with its partner.
    let times: Vec<f32> = (0..200).map(|k| 0.01 + 0.3 * k as f32).collect();
    let later: Vec<f32> = times.iter().map(|t| t + 0.005).collect();
    let up = run(0.5, &times, &later, 60.1);
    let down = run(0.5, &later, &times, 60.1);
    assert!(up > 0.95 && up < 1.0, "{up}");
    assert!(down < 0.05 && down > 0.0, "{down}");
}

#[test]
fn the_kernel_runs_the_rule() {
    let mut segs = [GpuSegmentData::zeroed(), GpuSegmentData::zeroed()];
    let mut syns = [synapse(0.5)];
    let controls = neutral_controls(2, 1);
    let params = SimParams {
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
    };
    for seg in &mut segs {
        seg.capacitance = 1e-6;
        seg.surface_area = 1e-5;
        seg.intra_k = 140e-3;
        seg.intra_na = 10e-3;
        seg.intra_ca = 1e-7;
        seg.intra_cl = 10e-3;
    }
    for k in 0..(0.05 / DT) as usize {
        let t = k as f32 * DT;
        segs[0].voltage = mv(t, &[0.01]);
        segs[1].voltage = mv(t, &[0.02]);
        step_biophysics_cpu(&mut segs, &[], &mut syns, &controls, &params);
    }
    assert!(syns[0].efficacy > 0.5 + 0.5 * A_PLUS * 0.5, "{}", syns[0].efficacy);
}

// --- The supervised rule (ReSuMe toward a target rate) ---

const ETA: f32 = 0.01;

fn supervised(efficacy: f32) -> GpuSynapseData {
    let mut s = synapse(efficacy);
    s.supervised = 1.0;
    s.stdp_potentiation = ETA;
    s.stdp_depression = 0.0;
    s
}

/// Spikes every `1 / hz` seconds from `start` to `end`.
fn train(hz: f32, start: f32, end: f32) -> Vec<f32> {
    let mut t = start;
    let mut out = vec![];
    while t < end {
        out.push(t);
        t += 1.0 / hz;
    }
    out
}

fn supervise(target_hz: f32, pre: &[f32], post: &[f32], duration: f32) -> f32 {
    let mut s = supervised(0.5);
    for k in 0..(duration / DT).round() as usize {
        let t = k as f32 * DT;
        step_spike_timing(&mut s, mv(t, pre), mv(t, post), target_hz, DT);
    }
    s.efficacy
}

#[test]
fn without_a_teacher_nothing_is_learned() {
    let pre = train(20.0, 0.003, 2.0);
    let post = train(30.0, 0.011, 2.0);
    assert_eq!(supervise(-1.0, &pre, &post, 2.0), 0.5);
}

#[test]
fn a_silent_cell_with_a_target_strengthens_its_active_inputs() {
    // Expected: eta * mean trace (rate * tau) * target * time.
    let pre = train(20.0, 0.003, 2.0);
    let got = supervise(30.0, &pre, &[], 2.0) - 0.5;
    let want = ETA * 20.0 * TAU_PRE * 30.0 * 2.0;
    assert!((got - want).abs() < 0.1 * want, "{got} vs {want}");
    // A silent input learns nothing.
    assert_eq!(supervise(30.0, &[], &[], 2.0), 0.5);
}

#[test]
fn learning_stops_when_the_cell_fires_at_its_target() {
    let pre = train(20.0, 0.003, 4.0);
    let post = train(30.0, 0.0117, 4.0);
    let at_target = (supervise(30.0, &pre, &post, 4.0) - 0.5).abs();
    let too_fast = supervise(10.0, &pre, &post, 4.0) - 0.5;
    let too_slow = supervise(50.0, &pre, &post, 4.0) - 0.5;
    assert!(too_fast < -0.05 && too_slow > 0.05, "{too_fast} {too_slow}");
    assert!(at_target < 0.15 * too_slow.min(-too_fast), "{at_target}");
}

#[test]
fn a_negative_rate_teaches_inhibition() {
    // An inhibitory synapse onto a cell firing above its target strengthens.
    let pre = train(20.0, 0.003, 2.0);
    let post = train(50.0, 0.0117, 2.0);
    let mut s = supervised(0.5);
    s.stdp_potentiation = -ETA;
    for k in 0..(2.0 / DT) as usize {
        let t = k as f32 * DT;
        step_spike_timing(&mut s, mv(t, &pre), mv(t, &post), 10.0, DT);
    }
    assert!(s.efficacy > 0.55, "{}", s.efficacy);
}
