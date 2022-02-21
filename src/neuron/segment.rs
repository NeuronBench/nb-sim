// use crate::constants::BODY_TEMPERATURE;
use crate::dimension::{Diameter, Interval, Kelvin, MilliVolts};
use crate::neuron::channel::{ca_reversal, cl_reversal, k_reversal, na_reversal};
use crate::neuron::membrane::Membrane;
use crate::neuron::solution::Solution;

#[derive(Clone, Debug)]
pub struct Segment {
    /// The ion concentrations inside the segment.
    pub intracellular_solution: Solution,
    /// The segment's shape (cylindrical radius and position).
    pub geometry: Geometry,
    /// The concentration of various channels.
    pub membrane: Membrane,
    pub membrane_potential: MilliVolts,
}

/// A cylindical neuron segment shape.
#[derive(Clone, Debug)]
pub struct Geometry {
    diameter_start: Diameter,
    diameter_end: Diameter,
    length: f32,
}

impl Segment {
    pub fn surface_area(&self) -> f32 {
        (self.geometry.diameter_start.0 + self.geometry.diameter_end.0) / 2.0 * self.geometry.length
    }

    pub fn dv_dt(&self, temperature: &Kelvin, extracellular_solution: &Solution) -> f32 {
        let surface_area = self.surface_area();
        let current = self.membrane.current_per_cm(
            &k_reversal(
                &self.intracellular_solution,
                extracellular_solution,
                temperature,
            ),
            &na_reversal(
                &self.intracellular_solution,
                extracellular_solution,
                temperature,
            ),
            &ca_reversal(
                &self.intracellular_solution,
                extracellular_solution,
                temperature,
            ),
            &cl_reversal(
                &self.intracellular_solution,
                extracellular_solution,
                temperature,
            ),
            &self.membrane_potential,
        ) * self.surface_area();
        let capacitance = self.membrane.capacitance.0 * surface_area;
        current / capacitance
    }

    pub fn step(
        &mut self,
        temperature: &Kelvin,
        extracellular_solution: &Solution,
        interval: &Interval,
    ) {
        // Currents charge the membrane.
        let new_membrane_potential = MilliVolts(
            self.membrane_potential.0
                + self.dv_dt(temperature, extracellular_solution) * interval.0,
        );
        self.membrane_potential = new_membrane_potential.clone();

        // Membrane charge updates voltage-sensitive gates.
        self.membrane
            .membrane_channels
            .iter_mut()
            .for_each(|membrane_channel| {
                membrane_channel
                    .channel
                    .step(&new_membrane_potential, &interval);
            });
    }
}

mod examples {
    use super::*;
    use crate::constants::*;
    use crate::dimension::*;
    use crate::neuron::channel;
    use crate::neuron::membrane::*;

    pub fn giant_squid_axon() -> Segment {
        let initial_membrane_potential = MilliVolts(-80.0);
        Segment {
            intracellular_solution: Solution {
                na_concentration: Molar(0.001),
                k_concentration: Molar(0.140),
                ca_concentration: Molar(0.0000001),
                cl_concentration: Molar(0.004),
            },
            geometry: Geometry {
                diameter_start: Diameter(0.01),
                diameter_end: Diameter(0.01),
                length: 0.1,
            },
            membrane_potential: initial_membrane_potential.clone(),
            membrane: Membrane {
                membrane_channels: vec![
                    MembraneChannel {
                        channel: channel::common_channels::giant_squid::K_CHANNEL
                            .build(&initial_membrane_potential),
                        siemens_per_square_cm: 10.0, // TODO: fix.
                    },
                    MembraneChannel {
                        channel: channel::common_channels::giant_squid::NA_CHANNEL
                            .build(&initial_membrane_potential),
                        siemens_per_square_cm: 10.0, // TODO: fix.
                    },
                ],
                capacitance: FaradsPerArea(1e-6),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::examples::giant_squid_axon;
    use super::*;
    use crate::constants::*;
    use crate::neuron::solution::INTERSTICIAL_FLUID;

    #[test]
    pub fn reaches_steady_state() {
        let mut segment = giant_squid_axon();
        let interval = Interval(0.001);
        for i in 0..100 {
            segment.step(&BODY_TEMPERATURE, &INTERSTICIAL_FLUID, &interval);
            if i % 10 == 0 {
                dbg!(&segment.membrane_potential);
            }
        }
        assert!(false);
    }
}
