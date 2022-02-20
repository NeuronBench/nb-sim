use crate::constants::body_temperature;
use crate::dimension::{Diameter, Farads, Interval, Timestamp, Volts};
use crate::neuron::membrane::{Membrane, MembraneChannelState};
use crate::neuron::solution::{intersticial_fluid, Solution};

#[derive(Clone, Debug)]
pub struct Geometry {
    diameter_start: Diameter,
    diameter_end: Diameter,
    length: f32,
}

#[derive(Clone, Debug)]
pub struct Segment {
    intracellular_solution: Solution,
    geometry: Geometry,
    membrane: Membrane,
    membrane_channel_state: MembraneChannelState,
    membrane_potential: Volts,
}

impl Segment {
    pub fn step(&mut self, start_time: Timestamp, dt: Interval) {
        // v = ir

        // Current flows according to the reversal potential, charging
        // the membrane capacitance and updating the membrane voltage.
        let reversal_potential = self.membrane_channel_state.reversal_potential(
            body_temperature,
            &self.intracellular_solution,
            &intersticial_fluid,
        );
        let current = self.membrane_potential.0 * self.membrane_channel_state.conductance().0;
        let capacitance = Farads(self.surface_area() * self.membrane.capacitance.0);
        let dv_over_dt = current / capacitance.0;
        self.membrane_potential = Volts(self.membrane_potential.0 + dv_over_dt * dt.0);

        // Channel state is sensitive to the membrane_potential.
        let na_inactivation = self.membrane_channel_state.na_inactivated.clone();

    }

    pub fn surface_area(&self) -> f32 {
        (self.geometry.diameter_start.0 + self.geometry.diameter_end.0) / 2.0 * self.geometry.length
    }
}
