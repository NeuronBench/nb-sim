use crate::constants::{gas_constant, inverse_faraday};
use crate::dimension::{Celcius, FaradsPerArea, Siemens, Timestamp, Volts};
use crate::neuron::channel::Channel;
use crate::neuron::solution::Solution;

/// The more static properties of a cell membrane: its permeability to
/// various ions. This may change with the development of the neuron,
/// but it is fairly static, compared to [`MembraneChannelState`].
#[derive(Clone, Debug)]
pub struct Membrane {
    /// The concentration of channels in this membrane.
    pub membrane_channels: Vec<MembraneChannel>,
    pub capacitance: FaradsPerArea,
}

#[derive(Clone, Debug)]
pub struct MembraneChannel {
    /// A chanel in the membrane.
    pub channel: Channel,
    /// The peak conductance of the given channel (what its conductance
    /// would be if all activation and inactivation gates were open).
    pub siemens_per_squere_cm: f32,
}

impl Membrane {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::body_temperature;
    use crate::neuron::solution::tests::*;

    #[test]
    fn example_reversal_potential() {}
}
