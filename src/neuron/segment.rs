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
        let current =
            -1.0 * self.membrane.current_per_square_cm(
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
    use crate::neuron::channel::{self, ChannelBuilder, CL, K, NA};
    use crate::neuron::membrane::*;
    use crate::neuron::solution::{EXAMPLE_CYTOPLASM, INTERSTICIAL_FLUID};

    pub fn giant_squid_axon() -> Segment {
        let initial_membrane_potential = MilliVolts(-80.0);
        Segment {
            intracellular_solution: Solution {
                na_concentration: Molar(5e-3),
                k_concentration: Molar(140e-3),
                cl_concentration: Molar(4e-3),
                ca_concentration: Molar(0.1e-6),
            },
            geometry: Geometry {
                diameter_start: Diameter(1.0),
                diameter_end: Diameter(1.0),
                length: 3.0,
            },
            membrane_potential: initial_membrane_potential.clone(),
            membrane: Membrane {
                membrane_channels: vec![
                    MembraneChannel {
                        channel: channel::common_channels::giant_squid::K_CHANNEL
                            .build(&initial_membrane_potential),
                        siemens_per_square_cm: 36e-3,
                    },
                    MembraneChannel {
                        channel: channel::common_channels::giant_squid::NA_CHANNEL
                            .build(&initial_membrane_potential),
                        siemens_per_square_cm: 120e-3,
                    },
                    MembraneChannel {
                        channel: channel::common_channels::giant_squid::LEAK_CHANNEL
                            .build(&initial_membrane_potential),
                        siemens_per_square_cm: 0.3e-3,
                    },
                ],
                capacitance: FaradsPerSquareCm(1e-6),
            },
        }
    }

    pub fn simple_leak() -> Segment {
        let initial_membrane_potential = MilliVolts(-80.0);
        Segment {
            intracellular_solution: EXAMPLE_CYTOPLASM,
            geometry: Geometry {
                diameter_start: Diameter(1.0),
                diameter_end: Diameter(1.0),
                length: 3.0,
            },
            membrane_potential: initial_membrane_potential.clone(),
            membrane: Membrane {
                membrane_channels: vec![MembraneChannel {
                    channel: channel::common_channels::giant_squid::LEAK_CHANNEL
                        .build(&initial_membrane_potential),
                    siemens_per_square_cm: 0.3e-3,
                }],
                capacitance: FaradsPerSquareCm(1e-6),
            },
        }
    }

    pub fn k_channels_only() -> Segment {
        let initial_membrane_potential = MilliVolts(-80.0);
        Segment {
            intracellular_solution: Solution {
                na_concentration: Molar(5e-3),
                k_concentration: Molar(140e-3),
                cl_concentration: Molar(4e-3),
                ca_concentration: Molar(0.1e-6),
            },
            geometry: Geometry {
                diameter_start: Diameter(1.0),
                diameter_end: Diameter(1.0),
                length: 3.0,
            },
            membrane_potential: initial_membrane_potential.clone(),
            membrane: Membrane {
                membrane_channels: vec![MembraneChannel {
                    channel: channel::common_channels::giant_squid::K_CHANNEL
                        .build(&initial_membrane_potential),
                    siemens_per_square_cm: 36e-3,
                }],
                capacitance: FaradsPerSquareCm(1e-6),
            },
        }
    }

    pub fn passive_channels(
        na_conductance: Siemens,
        k_conductance: Siemens,
        cl_conductance: Siemens,
    ) -> Segment {
        let initial_membrane_potential = MilliVolts(-80.0);
        Segment {
            intracellular_solution: EXAMPLE_CYTOPLASM,
            geometry: Geometry {
                diameter_start: Diameter(2.0),
                diameter_end: Diameter(2.0),
                length: 2.0,
            },
            membrane_potential: MilliVolts(-80.0),
            membrane: Membrane {
                membrane_channels: vec![
                    MembraneChannel {
                        channel: ChannelBuilder {
                            activation_parameters: None,
                            inactivation_parameters: None,
                            ion_selectivity: CL,
                        }
                        .build(&initial_membrane_potential),
                        siemens_per_square_cm: cl_conductance.0,
                    },
                    MembraneChannel {
                        channel: ChannelBuilder {
                            activation_parameters: None,
                            inactivation_parameters: None,
                            ion_selectivity: K,
                        }
                        .build(&initial_membrane_potential),
                        siemens_per_square_cm: k_conductance.0,
                    },
                    MembraneChannel {
                        channel: ChannelBuilder {
                            activation_parameters: None,
                            inactivation_parameters: None,
                            ion_selectivity: NA,
                        }
                        .build(&initial_membrane_potential),
                        siemens_per_square_cm: na_conductance.0,
                    },
                ],
                capacitance: FaradsPerSquareCm(1e-6),
            },
        }
    }

    #[cfg(test)]
    mod tests {
        use super::examples::{giant_squid_axon, k_channels_only, simple_leak};
        use super::*;
        use crate::constants::*;
        use crate::dimension::*;
        use crate::neuron::channel::{self, ChannelBuilder};
        use crate::neuron::channel::{cl_reversal, CL, K, NA};
        use crate::neuron::membrane::{Membrane, MembraneChannel};
        use crate::neuron::solution::{EXAMPLE_CYTOPLASM, INTERSTICIAL_FLUID};

        #[test]
        pub fn giant_axon_steady_state() {
            let mut segment = giant_squid_axon();
            let interval = Interval(0.001);
            for _ in 0..1000 {
                dbg!(&segment.membrane_potential.0);
                segment.step(&BODY_TEMPERATURE, &INTERSTICIAL_FLUID, &interval);
            }
            // Equilibrium state should be about -76 mV.
            assert!((segment.membrane_potential.0 - (-76.0)).abs() < 1.0);
        }

        #[test]
        pub fn simple_leak_reaches_nearnst_equillibrium() {
            let mut segment = simple_leak();
            let expected_resting_potential = cl_reversal(
                &segment.intracellular_solution,
                &INTERSTICIAL_FLUID,
                &BODY_TEMPERATURE,
            );

            let interval = Interval(0.01);

            for _ in 1..1000 {
                dbg!(&segment.membrane_potential.0);
                segment.step(&BODY_TEMPERATURE, &INTERSTICIAL_FLUID, &interval);
            }
            dbg!(&expected_resting_potential.0);
            assert!((segment.membrane_potential.0 - expected_resting_potential.0).abs() < 1.0);

            segment.membrane_potential = MilliVolts(-160.0);
            for _ in 1..10000 {
                dbg!(&segment.membrane_potential.0);
                segment.step(&BODY_TEMPERATURE, &INTERSTICIAL_FLUID, &interval);
            }
            dbg!(&expected_resting_potential.0);
            assert!((segment.membrane_potential.0 - expected_resting_potential.0).abs() < 1.0);

            segment.membrane_potential = MilliVolts(1.0);
            for _ in 1..10000 {
                dbg!(&segment.membrane_potential.0);
                segment.step(&BODY_TEMPERATURE, &INTERSTICIAL_FLUID, &interval);
            }
            dbg!(&expected_resting_potential.0);
            assert!((segment.membrane_potential.0 - expected_resting_potential.0).abs() < 1.0);
        }

        #[test]
        pub fn k_membrane_reaches_k_reversal_potential() {
            let mut segment = k_channels_only();

            let expected_resting_potential = k_reversal(
                &segment.intracellular_solution,
                &INTERSTICIAL_FLUID,
                &BODY_TEMPERATURE,
            );

            let interval = Interval(0.01);

            // Choose three initial membrane potentials, the segment should
            // equillibrate to the K reversal potential.
            //
            // Note that -100.0 does not work as an initial membrane potential,
            // because at this voltage the K channels are largely deactivated.
            for initial_potential in vec![MilliVolts(-89.5), MilliVolts(0.0), MilliVolts(100.0)] {
                segment.membrane_potential = initial_potential.clone();
                for _ in 1..10000 {
                    dbg!(&segment.membrane_potential.0);
                    segment.step(&BODY_TEMPERATURE, &INTERSTICIAL_FLUID, &interval)
                }
                dbg!(&expected_resting_potential);
                dbg!(initial_potential);
                assert!((segment.membrane_potential.0 - expected_resting_potential.0).abs() < 1.0);
            }
        }

        #[test]
        pub fn resting_potential_follows_ghk_equation() {
            let interval = Interval(0.001);
            fn ghk(g_na: f32, g_k: f32, g_cl: f32) -> MilliVolts {
                let i = &EXAMPLE_CYTOPLASM;
                let o = &INTERSTICIAL_FLUID;
                let e_k = k_reversal(i, o, &BODY_TEMPERATURE);
                let e_na = na_reversal(i, o, &BODY_TEMPERATURE);
                let e_cl = cl_reversal(i, o, &BODY_TEMPERATURE);
                let g_total = g_na + g_k + g_cl;
                MilliVolts((e_k.0 * g_k + e_na.0 * g_na + e_cl.0 * g_cl) / g_total)
            }

            let (na, k, cl) = (1e-3, 2e-3, 3e-3);
            let mut segment = passive_channels(Siemens(na), Siemens(k), Siemens(cl));
            for _ in 1..10000 {
                dbg!(&segment.membrane_potential.0);
                segment.step(&BODY_TEMPERATURE, &INTERSTICIAL_FLUID, &interval);
            }
            let expected_voltage = ghk(na, k, cl);
            dbg!(&expected_voltage);
            assert!((segment.membrane_potential.0 - expected_voltage.0).abs() < 1e-3);
        }
    }
}
