use serde::{Serialize, Deserialize};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use uuid::Uuid;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Scene {
    pub extracellular_solution: Solution,
    pub neurons: Vec<Neuron>,
    pub synapses: Vec<Synapse>,
    pub membranes: Vec<Membrane>,
}


#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Neuron {
    pub id: Uuid,
    pub segments: Vec<Segment>,
    pub junctions: Vec<(Uuid, Uuid)>,
    pub position_cm: Position
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Position {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Segment {
    pub id: Uuid,
    pub geometry: Geometry,
    pub intracellular_solution: Option<Solution>,
    pub position_microns: Position,
    pub membrane: Membrane,
    pub membrane_potential_mv: f32,
    pub stimulator_ids: Vec<Uuid>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Membrane {
    pub id: Uuid,
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
  pub activation: Option<(GatingParameters, f32)>,
  // Parameters for channel inactivation, and the current magnitude of this parameter.
  pub inactivation: Option<(GatingParameters, f32)>,
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
  pub steady_state_magnitude: Magnitude,
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
pub enum TimeConstant {
    Instantaneous,
    Sigmoid { v_at_max_tau: f32, c_base: f32, c_amp: f32, sigma: f32 },
    LinearExp { coef: f32, v_offset_mv: f32, inner_coef: f32 },
}


#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Synapse {
    pre_segment: Uuid,
    post_segment: Uuid,
    cleft_solution: Solution,
    // TODO: other synapse properties.
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

/// A trait for types that are content-addressable.
pub trait ContentAddress: Hash {

    fn content_address<T: ContentAddress>(&self) -> Uuid {
        let mut s = DefaultHasher::new();
        self.hash(&mut s);
        let item_hash: u64 = s.finish();
        Uuid::from_u64_pair(0, item_hash)
    }
}


#[cfg(test)]
pub mod tests {
    use super::*;

}
