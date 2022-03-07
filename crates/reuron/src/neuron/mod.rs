pub mod channel;
pub mod membrane;
pub mod segment;
pub mod solution;

use crate::constants::CONDUCTANCE_PER_SQUARE_CM;
use crate::dimension::{Interval, Kelvin, MilliVolts};
use crate::neuron::solution::Solution;

#[derive(Clone, Debug)]
pub struct Neuron {
    pub segments: Vec<segment::Segment>,
    pub junctions: Vec<(usize, usize)>,
}

impl Neuron {
    pub fn step(
        &mut self,
        temperature: &Kelvin,
        extracellular_solution: &Solution,
        interval: &Interval,
    ) {
        // Take a snapshot of all segment potentials.
        let membrane_potentials: Vec<MilliVolts> = self
            .segments
            .iter()
            .map(|s| s.membrane_potential.clone())
            .collect();

        self.segments
            .iter_mut()
            .for_each(|s| s.step(temperature, extracellular_solution, interval));

        for (m, n) in self.junctions.iter_mut() {
            let (voltage_m, diameter_m, capacitance_m) = {
                let segment_m = &self.segments[m.clone()];
                (
                    segment_m.membrane_potential.clone(),
                    segment_m.geometry.diameter.clone(),
                    segment_m.capacitance(),
                )
            };
            let (voltage_n, diameter_n, capacitance_n) = {
                let segment_n = &self.segments[n.clone()];
                (
                    segment_n.membrane_potential.clone(),
                    segment_n.geometry.diameter.clone(),
                    segment_n.capacitance(),
                )
            };
            let mutual_conductance = diameter_m.0.min(diameter_n.0) * CONDUCTANCE_PER_SQUARE_CM;
            let m_to_n_current = mutual_conductance * (voltage_m.0 - voltage_n.0) * 1e-3;

            self.segments[m.clone()].membrane_potential = MilliVolts(
                self.segments[m.clone()].membrane_potential.0
                    - m_to_n_current / capacitance_m.0 * interval.0,
            );
            self.segments[n.clone()].membrane_potential = MilliVolts(
                self.segments[n.clone()].membrane_potential.0
                    + m_to_n_current / capacitance_n.0 * interval.0,
            );
        }
    }
}

pub mod examples {
    use crate::neuron::segment::examples::{giant_squid_axon, simple_leak};
    use crate::neuron::Neuron;
    pub fn squid_with_passive_attachment() -> Neuron {
        Neuron {
            segments: vec![giant_squid_axon(), simple_leak()],
            junctions: vec![(0, 1)],
        }
    }
}
