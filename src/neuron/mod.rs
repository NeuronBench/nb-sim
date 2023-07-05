pub mod channel;
pub mod membrane;
pub mod segment;
pub mod solution;
pub mod synapse;
pub mod network;

use crate::constants::CONDUCTANCE_PER_SQUARE_CM;
use crate::dimension::{Diameter, Interval, Kelvin, MilliVolts};
use crate::neuron::solution::Solution;

use bevy::prelude::{Component, Entity};
use std::f32::consts::PI;

#[derive(Clone, Debug)]
pub struct Neuron {
    pub segments: Vec<segment::Segment>,
    pub junctions: Vec<(usize, usize, Diameter)>,
}

pub mod ecs {
    use bevy::prelude::Component;
    #[derive(Component)]
    pub struct Neuron;
}

#[derive(Component)]
pub struct Junction {
    pub first_segment: Entity,
    pub second_segment: Entity,
    pub pore_diameter: Diameter,
}


impl Neuron {
    pub fn step(
        &mut self,
        temperature: &Kelvin,
        extracellular_solution: &Solution,
        interval: &Interval,
    ) {

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
    use crate::dimension::{MicroAmpsPerSquareCm, Diameter};
    use crate::neuron::segment::examples::{giant_squid_axon, simple_leak};
    use crate::neuron::Neuron;
    use crate::neuron::segment::Segment;
    pub fn squid_with_passive_attachment() -> Neuron {
        let active_segment = giant_squid_axon();
        let mut active_segment_2 = giant_squid_axon();
        active_segment_2.input_current = MicroAmpsPerSquareCm(-1.0);
        let passive_segment = simple_leak();
        let junction_diameter = Diameter(1.0);
        // let no_junction = Diameter(0.0);
        let some_segments : Vec<Segment> = vec![
                active_segment,
                passive_segment.clone(),
                active_segment_2.clone(),
                passive_segment,
                active_segment_2,
            ];
        let mut segments = vec![];
        for _ in 0..300 {
            segments.extend(some_segments.clone());
        };
        Neuron {
            segments: segments,
            junctions: vec![
                (0, 1, junction_diameter.clone()),
                (1, 2, junction_diameter.clone()),
                (2, 3, junction_diameter.clone()),
                (3, 4, junction_diameter),
            ],
        }
    }
}
