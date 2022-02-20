use crate::constants::{gas_constant, inverse_faraday};
use crate::dimension::{Celcius, FaradsPerArea, Siemens, Timestamp, Volts};
use crate::neuron::solution::Solution;

/// The more static properties of a cell membrane: its permeability to
/// various ions. This may change with the development of the neuron,
/// but it is fairly static, compared to [`MembraneChannelState`].
#[derive(Clone, Debug)]
pub struct Membrane {
    pub na_channels_per_square_meter: f32,
    pub k_channels_per_square_meter: f32,
    pub ca_channels_per_square_meter: f32,
    pub capacitance_per_square_meter: f32,
    pub na_leak: Siemens,
    pub k_leak: Siemens,
    pub ca_leak: Siemens,
    pub capacitance: FaradsPerArea,
}

/// The more dynamic state of a membrane: its instantaneous permeability
/// to various ions. This permiability is determined by the properties
/// of the ion channels, which change quickly in response to the voltage
/// across the membrane.
#[derive(Clone, Debug)]
pub struct MembraneChannelState {
    pub na_conductance: Siemens,
    pub k_conductance: Siemens,
    pub ca_conductance: Siemens,

    /// Sodium channels become temporarily closed, and insensitive
    /// to membrane potential, after membrane potential reaches a
    /// certain level. This field tracks whether they are inactivated,
    /// and if so, when the inactivation occurred.
    pub na_inactivated: Option<Timestamp>,
}

impl MembraneChannelState {
    pub fn reversal_potential(
        &self,
        temperature: Celcius,
        internal: &Solution,
        external: &Solution,
    ) -> Volts {
        let total_conductance =
            self.na_conductance.0 + self.k_conductance.0 + self.ca_conductance.0;
        let p_na = self.na_conductance.0 / total_conductance;
        let p_k = self.k_conductance.0 / total_conductance;
        let p_ca = self.ca_conductance.0 / total_conductance;
        let conductances = (p_na * external.na_concentration.0
            + p_k * external.k_concentration.0
            + p_ca * external.ca_concentration.0)
            / (p_na * internal.na_concentration.0
                + p_k * internal.k_concentration.0
                + p_ca * internal.ca_concentration.0);

        Volts(gas_constant * inverse_faraday * conductances)
    }

    pub fn conductance(&self) -> Siemens {
        Siemens(self.na_conductance.0 + self.k_conductance.0 + self.ca_conductance.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::body_temp;
    use crate::neuron::solution::tests::*;

    #[test]
    fn example_reversal_potential() {}
}
