// use crate::constants::{gas_constant, inverse_faraday};
use crate::dimension::{FaradsPerSquareCm, MilliVolts};
use crate::neuron::channel::Channel;

/// The more static properties of a cell membrane: its permeability to
/// various ions. This may change with the development of the neuron,
/// but it is fairly static, compared to [`MembraneChannelState`].
#[derive(Clone, Debug)]
pub struct Membrane {
    /// The concentration of channels in this membrane.
    pub membrane_channels: Vec<MembraneChannel>,
    pub capacitance: FaradsPerSquareCm,
}

impl Membrane {
    pub fn current_per_square_cm(
        &self,
        k_reversal: &MilliVolts,
        na_reversal: &MilliVolts,
        ca_reversal: &MilliVolts,
        cl_reversal: &MilliVolts,
        membrane_potential: &MilliVolts,
    ) -> f32 {
        self.membrane_channels
            .iter()
            .map(|membrane_channel| {
                membrane_channel.channel_current_per_cm(
                    k_reversal,
                    na_reversal,
                    ca_reversal,
                    cl_reversal,
                    membrane_potential,
                )
            })
            .sum()
    }

    // pub fn input_resistance_per_square_cm(
    //     &self,
    //     k_reversal: &MilliVolts,
    //     na_reversal: &MilliVolts,
    //     ca_reversal: &MilliVolts,
    //     cl_reversal: &MilliVolts,
    //     membrane_potential: &MilliVolts,
    // ) -> Siemens {
    //     let current = self.current_per_cm(
    //         k_reversal,
    //         na_reversal,
    //         ca_reversal,
    //         cl_reversal,
    //         membrane_potential,
    //     );
    // }
}

#[derive(Clone, Debug)]
pub struct MembraneChannel {
    /// A chanel in the membrane.
    pub channel: Channel,
    /// The peak conductance of the given channel (what its conductance
    /// would be if all activation and inactivation gates were open).
    pub siemens_per_square_cm: f32,
}

impl MembraneChannel {
    pub fn channel_current_per_cm(
        &self,
        k_reversal: &MilliVolts,
        na_reversal: &MilliVolts,
        ca_reversal: &MilliVolts,
        cl_reversal: &MilliVolts,
        membrane_potential: &MilliVolts,
    ) -> f32 {
        let gating_coefficient = self.channel.conductance_coefficient();
        let k_current = self.channel.ion_selectivity.k
            * gating_coefficient
            * (membrane_potential.0 - k_reversal.0)
            * 0.001;
        let na_current = self.channel.ion_selectivity.na
            * gating_coefficient
            * (membrane_potential.0 - na_reversal.0)
            * 0.001;
        let ca_current = self.channel.ion_selectivity.ca
            * gating_coefficient
            * (membrane_potential.0 - ca_reversal.0)
            * 0.001;
        let cl_current = self.channel.ion_selectivity.cl
            * gating_coefficient
            * (membrane_potential.0 - cl_reversal.0)
            * 0.001;
        let channel_current =
            (k_current + na_current + ca_current + cl_current) * self.siemens_per_square_cm;
        channel_current
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::BODY_TEMPERATURE;

    #[test]
    fn example_reversal_potential() {}
}
