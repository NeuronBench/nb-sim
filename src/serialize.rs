use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Scene {
    // pub extracellular_solution: Solution,
    pub neurons: Vec<SceneNeuron>,
    pub synapses: Vec<Synapse>,
    /// ONNX models coupled to the simulation.
    #[serde(default)]
    pub models: Vec<SceneModel>,
}

/// An ONNX model that reads simulation state and drives the simulation.
/// Graph inputs and outputs bind, in order, to `inputs` and `outputs`.
#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SceneModel {
    /// The `.onnx` file: a URL, or a path relative to the root scene file.
    pub url: String,
    /// Simulated seconds between evaluations. Models run at most once per frame.
    pub interval_sec: f32,
    /// One entry per graph input, in graph order.
    pub inputs: Vec<ModelInput>,
    /// One entry per graph output, in graph order.
    pub outputs: Vec<ModelOutput>,
}

/// Where a model input's values come from. Each fills the input tensor in
/// row-major order.
#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag="type")]
pub enum ModelInput {
    /// Membrane voltages, as `(v_mv - offset_mv) * scale`.
    Voltages { taps: Vec<SegmentRef>, offset_mv: f32, scale: f32 },
    /// Simulated time in seconds.
    Time,
    /// Fixed values, for example from scene parameters.
    Constant { values: Vec<f32> },
    /// The named model output from the previous run (zeros on the first),
    /// for recurrent state such as a GRU's hidden state: `h0` in, `hn` out.
    /// The input must have a fixed shape.
    Feedback { output: String },
}

/// What a model output drives.
#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag="type")]
pub enum ModelOutput {
    /// Output element i, times `scale_uamps_per_square_cm`, is injected into
    /// `taps[i]` (added to its input current and stimulators).
    InjectCurrent { taps: Vec<SegmentRef>, scale_uamps_per_square_cm: f32 },
    /// Output element i scales the conductance of `taps[i]`'s channels by
    /// `max(0, offset + scale * value)`. `channel` picks one channel by its
    /// index in the segment's `membrane_channels`; without it, every channel
    /// is scaled. Several outputs on one channel multiply.
    ScaleConductance {
        taps: Vec<SegmentRef>,
        #[serde(default)]
        channel: Option<u32>,
        offset: f32,
        scale: f32,
    },
    /// Output element i scales every receptor current of synapse
    /// `synapses[i]` (an index into the scene's `synapses`) by
    /// `max(0, offset + scale * value)`. Several outputs on one synapse multiply.
    SynapticGain { synapses: Vec<u32>, offset: f32, scale: f32 },
    /// The output is computed but not used.
    Ignore,
}

/// A segment in the scene: an index into `neurons`, and the segment's SWC id
/// (as in `stimulator_segments`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SegmentRef {
    pub neuron: u32,
    pub segment: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SceneNeuron {
    pub neuron: Neuron,
    pub location: Location,
    pub stimulator_segments: Vec<StimulatorSegment>,
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Location {
    pub x_mm: f32,
    pub y_mm: f32,
    pub z_mm: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct StimulatorSegment {
    pub stimulator: Stimulator,
    pub segment: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Stimulator {
    pub envelope: Envelope,
    pub current_shape: CurrentShape,
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Envelope {
    pub period_sec: f32,
    pub onset_sec: f32,
    pub offset_sec: f32
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag="type")]
pub enum CurrentShape {
    SquareWave {
        on_current_uamps_per_square_cm: f32,
        off_current_uamps_per_square_cm: f32
    },
    LinearRamp {
        start_current_uamps_per_square_cm: f32,
        end_current_uamps_per_square_cm: f32,
        off_current_uamps_per_square_cm: f32,
    },
    FrequencyRamp {
        on_amplitude_uamps_per_square_cm: f32,
        offset_current_uamps_per_square_cm: f32,
        start_frequency_hz: f32,
        end_frequency_hz: f32,
    }
}


#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Neuron {
    pub segments: Vec<Segment>,
    pub membranes: Vec<Membrane>,
    // pub junctions: Vec<(Uuid, Uuid)>,
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Position {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Segment {

    pub id: i32,

    #[serde(rename="type")]
    pub type_: usize,

    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub r: f32,

    pub parent: i32,

}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Membrane {
    pub membrane_channels: Vec<MembraneChannel>,
    pub capacitance_farads_per_square_cm: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct MembraneChannel {
    pub channel: Channel,
    pub siemens_per_square_cm: f32
}


#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Channel {
  // Parameters for channel activation, and the current magnitude of this parameter.
  pub activation: Option<GatingParameters>,
  // Parameters for channel inactivation, and the current magnitude of this parameter.
  pub inactivation: Option<GatingParameters>,
  // Permiability of the channel to each ion, when activation magnitude is 1 and inactivation magnitude is 0.
  pub ion_selectivity: IonSelectivity,
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct IonSelectivity {
    // Permiability to Na+ ions.
    pub na: f32,
    // Permiability to K+ ions.
    pub k: f32,
    // Permiability to Ca+ ions.
    pub ca: f32,
    // Permiability to Cl+ ions.
    pub cl: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct GatingParameters {
  pub gates: u8,
  pub magnitude: Magnitude,
  pub time_constant: TimeConstant,
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Geometry {
    pub diameter_cm: f32,
    pub length_cm: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Magnitude {
    pub v_at_half_max_mv: f32,
    pub slope: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag="type")]
pub enum TimeConstant {
    Instantaneous,
    Gaussian { v_at_max_tau_mv: f32, c_base: f32, c_amp: f32, sigma: f32 },
    LinearExp { coef: f32, v_offset_mv: f32, inner_coef: f32 },
}


#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Synapse {
    pub pre_neuron: usize,
    pub pre_segment: usize,
    pub post_neuron: usize,
    pub post_segment: usize,
    pub synapse_membranes: SynapseMembranes,
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SynapseMembranes {
    pub cleft_solution: Solution,
    pub transmitter_concentrations: TransmitterConcentrations,
    pub presynaptic_pumps: Vec<TransmitterPump>,
    pub postsynaptic_receptors: Vec<Receptor>,
    pub surface_area_square_mm: f32
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct TransmitterConcentrations {
    pub glutamate_molar: f32,
    pub gaba_molar: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct TransmitterPump {
    pub transmitter: String,
    pub transmitter_pump_params: TransmitterPumpParams,
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct BellFunc {
    pub amplitude: f32,
    pub base: f32,
    pub x_at_max: f32,
    pub sigma: f32,
}


#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Gaussian {
    pub min_molar: f32,
    pub max_molar: f32,
    pub v_at_half_max_mv: f32,
    pub slope: f32,
    // pub log_space: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct TransmitterPumpParams {
    pub target_concentration: Gaussian,
    pub time_constant: TimeConstant,
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Receptor {
    pub membrane_channel: MembraneChannel,
    pub neurotransmitter_sensitivity: Sensitivity,
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Sensitivity {
    pub transmitter: String,
    pub concentration_at_half_max_molar: f32,
    pub slope: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Solution {
    // Na+ concentration (Molars).
    pub na: f32,
    // K+ concentration (Molars).
    pub k: f32,
    // Ca+ concentration (Molars).
    pub ca: f32,
    // Cl+ concentration (Molars).
    pub cl: f32,
}


#[cfg(test)]
pub mod tests {
    use super::*;

}
