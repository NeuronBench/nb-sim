use crate::constants::body_temperature;
use crate::dimension::{Diameter, Farads, Interval, Timestamp, Volts};
use crate::neuron::membrane::Membrane;
use crate::neuron::solution::{intersticial_fluid, Solution};

#[derive(Clone, Debug)]
pub struct Segment {
    /// The ion concentrations inside the segment.
    intracellular_solution: Solution,
    /// The segment's shape (cylindrical radius and position).
    geometry: Geometry,
    /// The concentration of various channels.
    membrane: Membrane,
    membrane_potential: Volts,
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
}
