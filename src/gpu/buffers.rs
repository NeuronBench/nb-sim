//! Flat GPU-compatible buffer structs for the neuron simulation.
//!
//! These structs use `#[repr(C)]` and `bytemuck` for direct GPU buffer upload.
//! They mirror the WGSL shader struct layout exactly.

use bytemuck::{Pod, Zeroable};

pub const MAX_CHANNELS: usize = 8;
pub const MAX_PUMPS: usize = 4;
pub const MAX_RECEPTORS: usize = 4;

/// Flattened gate state (activation or inactivation).
/// `present == 0.0` means the gate is absent (treated as magnitude=1.0).
/// TimeConstant type: 0=Instantaneous, 1=Gaussian, 2=LinearExp.
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct GpuGateState {
    pub present: f32,
    pub gates: f32,
    pub magnitude: f32,
    pub ss_v_at_half_max: f32,
    pub ss_slope: f32,
    pub tc_type: f32,
    /// Gaussian: v_at_max_tau. LinearExp: coef.
    pub tc_param0: f32,
    /// Gaussian: c_base. LinearExp: v_offset.
    pub tc_param1: f32,
    /// Gaussian: c_amp. LinearExp: inner_coef.
    pub tc_param2: f32,
    /// Gaussian: sigma. LinearExp: unused.
    pub tc_param3: f32,
    pub _pad0: f32,
    pub _pad1: f32,
}

/// A membrane channel: peak conductance, ion selectivity, and two gate states.
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct GpuMembraneChannel {
    pub siemens_per_square_cm: f32,
    pub ion_k: f32,
    pub ion_na: f32,
    pub ion_ca: f32,
    pub ion_cl: f32,
    pub _pad: f32,
    pub activation: GpuGateState,
    pub inactivation: GpuGateState,
}

/// Per-segment simulation state.
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct GpuSegmentData {
    // Header (16 f32)
    pub voltage: f32,
    pub capacitance: f32,
    pub intra_k: f32,
    pub intra_na: f32,
    pub intra_ca: f32,
    pub intra_cl: f32,
    /// 0.0 = Cylinder, 1.0 = Sphere
    pub geom_type: f32,
    pub geom_diameter: f32,
    pub geom_length: f32,
    pub surface_area: f32,
    pub input_current: f32,
    pub num_channels: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
    pub _pad3: f32,
    // Fixed array of channels
    pub channels: [GpuMembraneChannel; MAX_CHANNELS],
}

/// Per-junction (gap junction / electrical synapse) data.
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct GpuJunctionData {
    pub first_segment_idx: u32,
    pub second_segment_idx: u32,
    /// Precomputed: pore_diameter * PI * CONDUCTANCE_PER_SQUARE_CM
    pub conductance: f32,
    pub _pad: f32,
}

/// Adjacency list entry: a junction neighbor of a segment.
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct GpuJunctionNeighbor {
    pub neighbor_segment_idx: u32,
    pub conductance: f32,
}

/// Per-transmitter-pump data.
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct GpuTransmitterPump {
    pub present: f32,
    /// 0.0 = Glutamate, 1.0 = GABA
    pub transmitter: f32,
    pub target_conc_max: f32,
    pub target_conc_min: f32,
    pub target_conc_v_half: f32,
    pub target_conc_v_slope: f32,
    pub tc_v_at_max_tau: f32,
    pub tc_c_base: f32,
    pub tc_c_amp: f32,
    pub tc_sigma: f32,
    pub _pad0: f32,
    pub _pad1: f32,
}

/// Per-receptor data (includes an embedded membrane channel).
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct GpuReceptor {
    pub present: f32,
    /// 0.0 = Glutamate, 1.0 = GABA
    pub sens_transmitter: f32,
    pub sens_conc_half_max: f32,
    pub sens_slope: f32,
    pub channel: GpuMembraneChannel,
}

/// Per-synapse (chemical synapse) data.
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct GpuSynapseData {
    pub pre_segment_idx: u32,
    pub post_segment_idx: u32,
    pub cleft_k: f32,
    pub cleft_na: f32,
    pub cleft_ca: f32,
    pub cleft_cl: f32,
    pub glutamate_conc: f32,
    pub gaba_conc: f32,
    pub surface_area: f32,
    pub num_pumps: f32,
    pub num_receptors: f32,
    pub _pad: f32,
    pub pumps: [GpuTransmitterPump; MAX_PUMPS],
    pub receptors: [GpuReceptor; MAX_RECEPTORS],
}

/// Simulation-wide parameters (uploaded as a uniform buffer).
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct SimParams {
    pub extra_k: f32,
    pub extra_na: f32,
    pub extra_ca: f32,
    pub extra_cl: f32,
    pub temperature: f32,
    pub dt: f32,
    pub steps_per_dispatch: u32,
    pub num_segments: u32,
    pub num_junctions: u32,
    pub num_synapses: u32,
    pub gas_constant: f32,
    pub inverse_faraday: f32,
    pub conductance_per_square_cm: f32,
    pub synapse_resistance_ohms: f32,
    pub _pad0: f32,
    pub _pad1: f32,
}
