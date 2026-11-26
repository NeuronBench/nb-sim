use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Scene {
    // pub extracellular_solution: Solution,
    pub neurons: Vec<SceneNeuron>,
    pub synapses: Vec<Synapse>,
    /// ONNX models coupled to the simulation.
    #[serde(default)]
    pub models: Vec<SceneModel>,
    /// Grids of injected current painted in the Canvas window.
    #[serde(default)]
    pub canvases: Vec<Canvas>,
    /// Read a population's answer, frame by frame, from its spikes.
    #[serde(default)]
    pub readouts: Vec<Readout>,
}

/// Score a population of cells against a canvas's frames: during each frame
/// count every tap's spikes (upward crossings of `spike_threshold_mv`); the
/// tap with the most is the answer, compared with the frame's label. The
/// log shows the running agreement.
#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Readout {
    pub name: String,
    /// One segment per answer: tap i stands for label i.
    pub taps: Vec<SegmentRef>,
    /// The canvas whose frames and labels to score against.
    pub canvas: String,
    pub spike_threshold_mv: f32,
    /// Whether tap i answers label i (false: only compare the taps' rates
    /// with their teachers' target rates, for a hidden layer say).
    #[serde(default = "yes")]
    pub labeled: bool,
}

fn yes() -> bool {
    true
}

/// A grid of current injections, one cell per segment, painted with the
/// mouse in the Canvas window: left-drag raises a cell, right-drag lowers
/// it. Cells range over -1..1; a cell at 1 injects
/// `max_uamps_per_square_cm` into its segment.
#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Canvas {
    /// Shown as the window title; the painting is kept across reloads while
    /// the name and size stay the same.
    pub name: String,
    pub width: u32,
    pub height: u32,
    /// The segment under each cell, row-major (`width * height` of them).
    pub taps: Vec<SegmentRef>,
    pub max_uamps_per_square_cm: f32,
    /// Starting cell values in -1..1, row-major; empty for all zeros.
    #[serde(default)]
    pub initial: Vec<f32>,
    /// A stream of images instead of a painting: frame k (row-major cell
    /// values) shows from `k * frame_sec` of simulated time, cycling.
    #[serde(default)]
    pub frames: Vec<Vec<f32>>,
    #[serde(default)]
    pub frame_sec: f32,
    /// A label per frame (a digit's class, say), for readouts.
    #[serde(default)]
    pub labels: Vec<u32>,
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
    /// Extra sinks fed from intermediate values (hidden layers).
    #[serde(default)]
    pub hidden: Vec<HiddenTap>,
}

/// Drive the simulation from part of an intermediate value of a model.
#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct HiddenTap {
    /// The value's name, as the Models window shows it (`conv2d1`, `relu2`).
    pub value: String,
    /// Flat (row-major) indices into the value: element `elements[i]` drives
    /// target i of `output`.
    pub elements: Vec<u32>,
    pub output: ModelOutput,
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
    /// The cell values of the named canvas (its current frame).
    Canvas { name: String },
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
    /// Output element i, times `scale_hz`, is the firing rate `taps[i]`
    /// should have: the teacher of `supervised` synapses onto that segment.
    /// It injects nothing. Negative values count as 0.
    TargetRate { taps: Vec<SegmentRef>, scale_hz: f32 },
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
    pub surface_area_square_mm: f32,
    /// A Ca2+ pool in the postsynaptic spine, fed by the receptors' calcium
    /// current (an NMDA receptor's, say).
    #[serde(default)]
    pub calcium: Option<SynapticCalcium>,
    /// Calcium-driven change of the synapse's strength. Needs `calcium`.
    #[serde(default)]
    pub plasticity: Option<CalciumPlasticity>,
    /// Pair-based spike-timing-dependent plasticity, an alternative to
    /// `plasticity` that needs no calcium.
    #[serde(default)]
    pub spike_timing: Option<SpikeTimingPlasticity>,
    /// Learning from a teacher's target rate (ReSuMe), the third choice.
    #[serde(default)]
    pub supervised: Option<SupervisedPlasticity>,
}

/// Supervised spike-based learning in the manner of ReSuMe (Ponulak &
/// Kasinski 2010), with a target firing rate as the teacher. Each
/// presynaptic spike adds 1 to a trace that decays with `pre_trace_sec`.
/// While the postsynaptic segment has a target rate (a model's
/// `TargetRate` output), the efficacy `rho` (0..1) grows by
/// `learning_rate * pre_trace * target_hz * dt` and falls by
/// `learning_rate * pre_trace` at each postsynaptic spike, so on average
/// `d rho/dt = learning_rate * pre_trace * (target - actual rate)`: the delta
/// rule, which stops once the cell fires as the teacher asks. Without a
/// target nothing is learned. Receptors marked `plastic` have their
/// conductance scaled by `weight_at_0 + rho (weight_at_1 - weight_at_0)`.
/// For an inhibitory synapse use a negative `learning_rate`: it then
/// strengthens while the cell fires above its target, and weakens while
/// the cell fires below it.
#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SupervisedPlasticity {
    pub spike_threshold_mv: f32,
    pub pre_trace_sec: f32,
    pub learning_rate: f32,
    pub initial_efficacy: f32,
    pub weight_at_0: f32,
    pub weight_at_1: f32,
}

/// Phenomenological, pair-based STDP (nearest spike, soft bounds). A spike
/// is an upward crossing of `spike_threshold_mv` by the pre- or postsynaptic
/// segment. Each spike sets its side's trace to 1, which then decays with
/// `pre_trace_sec` (presynaptic) or `post_trace_sec` (postsynaptic). On a
/// postsynaptic spike the efficacy `rho` (0..1) gains
/// `potentiation * (1 - rho) * pre_trace`; on a presynaptic spike it loses
/// `depression * rho * post_trace`. So a presynaptic spike shortly before a
/// postsynaptic one strengthens the synapse, and one shortly after weakens
/// it, each by an amount that falls off exponentially with the delay.
/// Receptors marked `plastic` have their conductance scaled by
/// `weight_at_0 + rho (weight_at_1 - weight_at_0)`.
#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SpikeTimingPlasticity {
    pub spike_threshold_mv: f32,
    pub pre_trace_sec: f32,
    pub post_trace_sec: f32,
    pub potentiation: f32,
    pub depression: f32,
    pub initial_efficacy: f32,
    pub weight_at_0: f32,
    pub weight_at_1: f32,
}

/// Spine calcium: `d[Ca]/dt = ([Ca]_rest - [Ca]) / decay - I_Ca / (2 F V)`,
/// with `I_Ca` the calcium part of the receptors' current (inward is
/// negative) and `V` the spine volume.
#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SynapticCalcium {
    pub resting_molar: f32,
    pub decay_sec: f32,
    pub spine_volume_cubic_um: f32,
}

/// The Graupner–Brunel (2012) calcium rule. The synapse's efficacy `rho`
/// (0..1) follows
/// `tau d(rho)/dt = -bistability rho (1 - rho) (1/2 - rho)
///   + potentiation_rate (1 - rho) [Ca > potentiation_threshold]
///   - depression_rate rho [Ca > depression_threshold]`,
/// and receptors marked `plastic` have their conductance scaled by
/// `weight_at_0 + rho (weight_at_1 - weight_at_0)`: AMPA receptors inserted
/// or removed. Brief, moderate calcium (a presynaptic spike alone, or a
/// postsynaptic spike before it) depresses; the large calcium of a
/// presynaptic spike followed by a postsynaptic one (NMDA receptors bound
/// and unblocked together) potentiates.
#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct CalciumPlasticity {
    pub depression_threshold_molar: f32,
    pub potentiation_threshold_molar: f32,
    pub depression_rate: f32,
    pub potentiation_rate: f32,
    pub time_constant_sec: f32,
    /// Strength of the pull toward rho = 0 or 1 (0 for none).
    pub bistability: f32,
    pub initial_efficacy: f32,
    pub weight_at_0: f32,
    pub weight_at_1: f32,
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
    /// How fast transmitter binds and unbinds: the bound fraction relaxes
    /// toward the sensitivity curve with these time constants. Without it,
    /// binding follows the cleft at once.
    #[serde(default)]
    pub binding: Option<BindingKinetics>,
    /// Voltage-dependent Mg2+ block of the pore (as in NMDA receptors).
    #[serde(default)]
    pub magnesium_block: Option<MagnesiumBlock>,
    /// Scale this receptor's conductance by the synapse's plastic weight.
    #[serde(default)]
    pub plastic: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct BindingKinetics {
    /// Time constant while the bound fraction grows.
    pub rise_sec: f32,
    /// Time constant while it falls (unbinding).
    pub decay_sec: f32,
}

/// Jahr & Stevens (1990): the conductance is multiplied by
/// `1 / (1 + [Mg2+] / 3.57 mM * exp(-0.062 / mV * V))`.
#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct MagnesiumBlock {
    pub magnesium_molar: f32,
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
