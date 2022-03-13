pub mod channel;
pub mod membrane;
pub mod segment;
pub mod solution;
pub mod synapse;
pub mod network;

use crate::constants::CONDUCTANCE_PER_SQUARE_CM;
use crate::dimension::{Diameter, Interval, Kelvin, MilliVolts};
use crate::neuron::solution::Solution;

use std::f32::consts::PI;

#[derive(Clone, Debug)]
pub struct Neuron {
    pub segments: Vec<segment::Segment>,
    pub junctions: Vec<(usize, usize, Diameter)>,
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

        for (m, n, pore_diameter) in self.junctions.iter_mut() {
            let (voltage_m, capacitance_m) = {
                let segment_m = &self.segments[m.clone()];
                (
                    segment_m.membrane_potential.clone(),
                    segment_m.capacitance(),
                )
            };
            let (voltage_n, capacitance_n) = {
                let segment_n = &self.segments[n.clone()];
                (
                    segment_n.membrane_potential.clone(),
                    segment_n.capacitance(),
                )
            };
            let mutual_conductance = pore_diameter.0 * PI * CONDUCTANCE_PER_SQUARE_CM;
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
    use crate::dimension::{Diameter, MicroAmpsPerSquareCm};
    use crate::neuron::segment::examples::{giant_squid_axon, simple_leak};
    use crate::neuron::Neuron;
    pub fn squid_with_passive_attachment() -> Neuron {
        let active_segment = giant_squid_axon();
        let mut active_segment_2 = giant_squid_axon();
        active_segment_2.input_current = MicroAmpsPerSquareCm(-1.0);
        let passive_segment = simple_leak();
        let junction_diameter = active_segment.geometry.diameter.clone();
        let no_junction = Diameter(0.0);
        Neuron {
            segments: vec![
                active_segment,
                passive_segment.clone(),
                active_segment_2.clone(),
                passive_segment,
                active_segment_2,
            ],
            junctions: vec![
                (0, 1, junction_diameter.clone()),
                (1, 2, junction_diameter.clone()),
                (2, 3, junction_diameter.clone()),
                (3, 4, junction_diameter),
            ],
        }
    }
}
