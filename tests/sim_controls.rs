//! The per-frame controls (input currents, conductance multipliers, synapse
//! gains) as the simulation kernels read them: the CPU kernels directly, and
//! the WGSL shader by validating it and checking its layout constants.

use bytemuck::Zeroable;
use nb_sim::cpu_sim::step_biophysics_cpu;
use nb_sim::gpu::buffers::*;
use nb_sim::gpu::extract::control_offsets;

const DT: f32 = 1e-5;

/// `a ≈ b` relative to their size (one step's change, measured as the
/// difference of two f32 voltages near -65 mV, carries ~1e-5 mV of noise).
fn close(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-3 * a.abs().max(b.abs()) + 2e-5
}

fn params(segments: usize, synapses: usize) -> SimParams {
    SimParams {
        extra_k: 5.0,
        extra_na: 145.0,
        extra_ca: 2.0,
        extra_cl: 110.0,
        temperature: 310.0,
        dt: DT,
        steps_per_dispatch: 1,
        num_segments: segments as u32,
        num_junctions: 0,
        num_synapses: synapses as u32,
        gas_constant: 8.314,
        inverse_faraday: 1.0 / 96485.3,
        conductance_per_square_cm: 0.0,
        synapse_resistance_ohms: 1e9,
        _pad0: 0.0,
        _pad1: 0.0,
    }
}

/// A segment at `v` mV with ungated K and Na channels.
fn segment(v: f32) -> GpuSegmentData {
    let mut seg = GpuSegmentData::zeroed();
    seg.voltage = v;
    seg.capacitance = 1e-6;
    seg.surface_area = 1e-5;
    seg.intra_k = 140.0;
    seg.intra_na = 10.0;
    seg.intra_ca = 1e-4;
    seg.intra_cl = 10.0;
    seg.num_channels = 2.0;
    seg.channels[0].siemens_per_square_cm = 0.01;
    seg.channels[0].ion_k = 1.0;
    seg.channels[1].siemens_per_square_cm = 0.002;
    seg.channels[1].ion_na = 1.0;
    seg
}

/// Controls for `segments` segments and `synapses` synapses: no input,
/// unit multipliers.
fn controls(segments: usize, synapses: usize) -> Vec<f32> {
    let (scales_at, gains_at) = control_offsets(segments);
    let mut c = vec![0.0; gains_at + synapses];
    c[scales_at..].fill(1.0);
    c
}

/// Voltage change of segment 0 after one step.
fn dv(segments: &[GpuSegmentData], synapses: &[GpuSynapseData], controls: &[f32]) -> f32 {
    let mut segs = segments.to_vec();
    let mut syns = synapses.to_vec();
    let params = params(segs.len(), syns.len());
    step_biophysics_cpu(&mut segs, &[], &mut syns, controls, &params);
    segs.last().unwrap().voltage - segments.last().unwrap().voltage
}

#[test]
fn input_current_comes_from_controls() {
    let mut seg = segment(-65.0);
    seg.num_channels = 0.0;
    seg.input_current = 999.0; // ignored: the controls carry the current
    let mut c = controls(1, 0);
    c[0] = 10.0;
    // 10 µA/cm² into 1 µF/cm² for 10 µs: 10 V/s * 1e-5 s = 0.1 mV.
    assert!(close(dv(&[seg], &[], &c), 0.1), "{}", dv(&[seg], &[], &c));
}

#[test]
fn conductance_scale_multiplies_each_channel() {
    let seg = segment(-30.0);
    let (scales_at, _) = control_offsets(1);
    let both = dv(&[seg], &[], &controls(1, 0));
    let mut no_k = controls(1, 0);
    no_k[scales_at] = 0.0;
    let mut no_na = controls(1, 0);
    no_na[scales_at + 1] = 0.0;
    let (na_only, k_only) = (dv(&[seg], &[], &no_k), dv(&[seg], &[], &no_na));
    assert!(k_only < 0.0 && na_only > 0.0, "K pulls down, Na pulls up: {k_only} {na_only}");
    assert!(close(both, k_only + na_only), "{both} vs {k_only} + {na_only}");

    let mut doubled = controls(1, 0);
    doubled[scales_at..scales_at + MAX_CHANNELS].fill(2.0);
    let twice = dv(&[seg], &[], &doubled);
    assert!(close(twice, 2.0 * both), "{twice} vs 2 × {both}");
}

#[test]
fn synapse_gain_multiplies_receptor_current() {
    // Presynaptic segment 0, postsynaptic segment 1 (no channels of its own),
    // with a fully gated Na receptor.
    let pre = segment(-65.0);
    let mut post = segment(-65.0);
    post.num_channels = 0.0;
    let mut syn = GpuSynapseData::zeroed();
    syn.pre_segment_idx = 0;
    syn.post_segment_idx = 1;
    syn.cleft_na = 145.0;
    syn.cleft_k = 5.0;
    syn.cleft_ca = 2.0;
    syn.cleft_cl = 110.0;
    syn.glutamate_conc = 1.0;
    syn.surface_area = 1e4;
    syn.num_receptors = 1.0;
    syn.receptors[0].present = 1.0;
    syn.receptors[0].sens_conc_half_max = 0.0;
    syn.receptors[0].sens_slope = 100.0;
    syn.receptors[0].channel.siemens_per_square_cm = 1.0;
    syn.receptors[0].channel.ion_na = 1.0;

    let (_, gains_at) = control_offsets(2);
    let with_gain = |g: f32| {
        let mut c = controls(2, 1);
        c[gains_at] = g;
        dv(&[pre, post], &[syn], &c)
    };
    let unit = with_gain(1.0);
    assert!(unit > 0.0, "an Na receptor depolarizes: {unit}");
    assert_eq!(with_gain(0.0), 0.0);
    let triple = with_gain(3.0);
    assert!(close(triple, 3.0 * unit), "{triple} vs 3 × {unit}");
}

#[test]
fn shader_is_valid_and_agrees_on_the_layout() {
    let source = include_str!("../assets/shaders/biophysics.wgsl");
    let module = naga::front::wgsl::parse_str(source).unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty())
        .validate(&module)
        .unwrap_or_else(|e| panic!("{e:?}"));

    assert!(source.contains(&format!("const MAX_CHANNELS: u32 = {MAX_CHANNELS}u;")));
    // The shader's accessors, mirrored: scales start after n currents, gains
    // after n * (1 + MAX_CHANNELS) values.
    let n = 5;
    let (scales_at, gains_at) = control_offsets(n);
    assert_eq!(scales_at, n);
    assert_eq!(gains_at, n * (1 + MAX_CHANNELS));
}
