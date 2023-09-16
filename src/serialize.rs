use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Scene {
    // pub extracellular_solution: Solution,
    pub neurons: Vec<SceneNeuron>,
    pub synapses: Vec<Synapse>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneNeuron {
    pub neuron: Neuron,
    pub location: Location,
    pub stimulator_segments: Vec<StimulatorSegment>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Location {
    pub x_mm: f32,
    pub y_mm: f32,
    pub z_mm: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StimulatorSegment {
    pub stimulator: Stimulator,
    pub segment: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Stimulator {
    pub envelope: Envelope,
    pub current_shape: CurrentShape,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Envelope {
    pub period_sec: f32,
    pub onset_sec: f32,
    pub offset_sec: f32
}

#[derive(Clone, Debug, Serialize, Deserialize)]
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


#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Neuron {
    pub segments: Vec<Segment>,
    pub membranes: Vec<Membrane>,
    // pub junctions: Vec<(Uuid, Uuid)>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Position {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Membrane {
    pub membrane_channels: Vec<MembraneChannel>,
    pub capacitance_farads_per_square_cm: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MembraneChannel {
    pub channel: Channel,
    pub siemens_per_square_cm: f32
}


#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Channel {
  // Parameters for channel activation, and the current magnitude of this parameter.
  pub activation: Option<GatingParameters>,
  // Parameters for channel inactivation, and the current magnitude of this parameter.
  pub inactivation: Option<GatingParameters>,
  // Permiability of the channel to each ion, when activation magnitude is 1 and inactivation magnitude is 0.
  pub ion_selectivity: IonSelectivity,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GatingParameters {
  pub gates: u8,
  pub magnitude: Magnitude,
  pub time_constant: TimeConstant,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Geometry {
    pub diameter_cm: f32,
    pub length_cm: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Magnitude {
    pub v_at_half_max_mv: f32,
    pub slope: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag="type")]
pub enum TimeConstant {
    Instantaneous,
    Sigmoid { v_at_max_tau_mv: f32, c_base: f32, c_amp: f32, sigma: f32 },
    LinearExp { coef: f32, v_offset_mv: f32, inner_coef: f32 },
}


#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Synapse {
    pub pre_neuron: usize,
    pub pre_segment: usize,
    pub post_neuron: usize,
    pub post_segment: usize,
    pub synapse_membranes: SynapseMembranes,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SynapseMembranes {
    pub cleft_solution: Solution,
    pub transmitter_concentrations: TransmitterConcentrations,
    pub presynaptic_pumps: Vec<TransmitterPump>,
    pub postsynaptic_receptors: Vec<Receptor>,
    pub surface_area_square_mm: f32
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransmitterConcentrations {
    pub glutamate_molar: f32,
    pub gaba_molar: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransmitterPump {
    pub transmitter: String,
    pub transmitter_pump_params: TransmitterPumpParams,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BellFunc {
    pub amplitude: f32,
    pub base: f32,
    pub x_at_max: f32,
    pub sigma: f32,
}


#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Sigmoid {
    pub min_molar: f32,
    pub max_molar: f32,
    pub v_at_half_max_mv: f32,
    pub slope: f32,
    // pub log_space: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransmitterPumpParams {
    pub target_concentration: Sigmoid,
    pub time_constant: TimeConstant,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Receptor {
    pub membrane_channel: MembraneChannel,
    pub neurotransmitter_sensitivity: Sensitivity,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Sensitivity {
    pub transmitter: String,
    pub concentration_at_half_max_molar: f32,
    pub slope: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
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
