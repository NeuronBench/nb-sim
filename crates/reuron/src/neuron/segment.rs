// use crate::constants::BODY_TEMPERATURE;
use bevy::prelude::Component;
use crate::dimension::{
    Diameter, Farads, Interval, Kelvin, MicroAmps, MicroAmpsPerSquareCm, MilliVolts,
};
use crate::neuron::channel::{ca_reversal, cl_reversal, k_reversal, na_reversal};
use crate::neuron::membrane::Membrane;
use crate::neuron::solution::Solution;

use std::f32::consts::PI;

#[derive(Clone, Debug)]
pub struct Segment {
    /// The ion concentrations inside the segment.
    pub intracellular_solution: Solution,
    /// The segment's shape (cylindrical radius and position).
    pub geometry: Geometry,
    /// The concentration of various channels.
    pub membrane: Membrane,
    pub membrane_potential: MilliVolts,
    pub input_current: MicroAmpsPerSquareCm,
    pub synaptic_current: MicroAmps,
}

/// A cylindical neuron segment shape.
#[derive(Clone, Component, Debug)]
pub struct Geometry {
    pub diameter: Diameter,
    pub length: f32,
}

impl Segment {
    pub fn surface_area(&self) -> f32 {
        (self.geometry.diameter.0) * PI * self.geometry.length
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
                &cl_reversal(
                    &self.intracellular_solution,
                    extracellular_solution,
                    temperature,
                ),
                &ca_reversal(
                    &self.intracellular_solution,
                    extracellular_solution,
                    temperature,
                ),
                &self.membrane_potential,
            ) * self.surface_area()
                - self.synaptic_current.0 * 1e-6
                + self.input_current.0 * 1e-6 * surface_area;
        let capacitance = self.membrane.capacitance.0 * surface_area;
        current / capacitance
    }

    pub fn capacitance(&self) -> Farads {
        Farads(self.membrane.capacitance.0 * self.surface_area())
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
                + self.dv_dt(temperature, extracellular_solution) * 1000.0 * interval.0,
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

pub mod examples {
    use super::*;
    use crate::dimension::*;
    use crate::neuron::channel::{self, ChannelBuilder, CL, K, NA};
    use crate::neuron::membrane::*;
    use crate::neuron::solution::{EXAMPLE_CYTOPLASM};

    pub fn giant_squid_axon() -> Segment {
        let initial_membrane_potential = MilliVolts(-70.0);
        Segment {
            intracellular_solution: Solution {
                na_concentration: Molar(5e-3),
                k_concentration: Molar(140e-3),
                cl_concentration: Molar(4e-3),
                ca_concentration: Molar(0.1e-6),
            },
            geometry: Geometry {
                diameter: Diameter(1.0),
                length: 3.0,
            },
            input_current: MicroAmpsPerSquareCm(0.0),
            synaptic_current: MicroAmps(0.0),
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
            synaptic_current: MicroAmps(0.0),
            geometry: Geometry {
                diameter: Diameter(0.01),
                length: 1000.0,
            },
            input_current: MicroAmpsPerSquareCm(0.0),
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
            input_current: MicroAmpsPerSquareCm(0.0),
            synaptic_current: MicroAmps(0.0),
            intracellular_solution: Solution {
                na_concentration: Molar(5e-3),
                k_concentration: Molar(140e-3),
                cl_concentration: Molar(4e-3),
                ca_concentration: Molar(0.1e-6),
            },
            geometry: Geometry {
                diameter: Diameter(1.0),
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

    /// Build a segment with the passive sodium, potassium and chloride
    /// channels with the parameterized conductances.
    pub fn passive_channels(
        na_conductance: Siemens,
        k_conductance: Siemens,
        cl_conductance: Siemens,
    ) -> Segment {
        let initial_membrane_potential = MilliVolts(-58.0);
        Segment {
            intracellular_solution: EXAMPLE_CYTOPLASM,
            input_current: MicroAmpsPerSquareCm(0.0),
            synaptic_current: MicroAmps(0.0),
            geometry: Geometry {
                diameter: Diameter(2.0),
                length: 2.0,
            },
            membrane_potential: initial_membrane_potential.clone(),
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
        use crate::neuron::channel::cl_reversal;
        use crate::neuron::membrane::{Membrane, MembraneChannel};
        use crate::neuron::solution::{EXAMPLE_CYTOPLASM, INTERSTICIAL_FLUID};
        use std::io;

        #[test]
        // The giant squid axon should settly at a resting membrane potential
        // of -76 mV. (This is a smoke test - I didn't get this number from
        // a book, but should.
        pub fn giant_axon_steady_state() {
            let mut wtr = csv::Writer::from_path("out.csv").unwrap();
            wtr.write_record(&[
                "t", "v_m", "i", "g_k", "g_na", "g_cl", "m_na", "h_na", "m_k",
            ])
            .unwrap();
            let mut write_record = |t: f32, s: &Segment| {
                let (k, na, cl, ca) = s.membrane.conductances();
                wtr.write_record(&[
                    format!("{0:.2}", t * 1000.0),
                    s.membrane_potential.0.to_string(),
                    s.input_current.0.to_string(),
                    k.to_string(),
                    na.to_string(),
                    cl.to_string(),
                    s.membrane.membrane_channels[1]
                        .clone()
                        .channel
                        .activation
                        .unwrap()
                        .magnitude
                        .to_string(),
                    s.membrane.membrane_channels[1]
                        .clone()
                        .channel
                        .inactivation
                        .unwrap()
                        .magnitude
                        .to_string(),
                    s.membrane.membrane_channels[0]
                        .clone()
                        .channel
                        .activation
                        .unwrap()
                        .magnitude
                        .to_string(),
                ])
                .unwrap();
            };
            let mut t = 0.0;

            let mut segment = giant_squid_axon();
            segment.membrane_potential = MilliVolts(-79.0);
            let interval = Interval(0.00001);
            // segment.membrane_potential = MilliVolts(-60.0);

            // 1 ms pre-stim.
            segment.input_current = MicroAmpsPerSquareCm(0.0);
            while t < 0.001 {
                write_record(t, &segment);
                segment.step(&BODY_TEMPERATURE, &INTERSTICIAL_FLUID, &interval);
                t += interval.0;
            }
            // Equilibrium state should be about -76 mV.
            // assert!((segment.membrane_potential.0 - (-76.0)).abs() < 1.0);

            // Now turn on current injection for 0.1 milliseconds.
            segment.input_current = MicroAmpsPerSquareCm(0.0);
            while t < 0.0500 {
                write_record(t, &segment);
                segment.step(&BODY_TEMPERATURE, &INTERSTICIAL_FLUID, &interval);
                t += interval.0;
            }

            // And turn it back off. Run for 100 ms.
            segment.input_current = MicroAmpsPerSquareCm(0.0);
            while t < 0.050 {
                write_record(t, &segment);
                segment.step(&BODY_TEMPERATURE, &INTERSTICIAL_FLUID, &interval);
                t += interval.0;
            }

            assert!(false);
        }

        #[test]
        // A membrane with a leak current (which we model with a passive Cl-
        // channel) should settle at the Cl- reversal potential.
        pub fn simple_leak_reaches_nearnst_equillibrium() {
            let mut segment = simple_leak();
            let expected_resting_potential = cl_reversal(
                &segment.intracellular_solution,
                &INTERSTICIAL_FLUID,
                &BODY_TEMPERATURE,
            );

            let interval = Interval(0.001);

            for _ in 1..10 {
                dbg!(&segment.membrane_potential.0);
                segment.step(&BODY_TEMPERATURE, &INTERSTICIAL_FLUID, &interval);
            }
            dbg!(&expected_resting_potential.0);
            assert!((segment.membrane_potential.0 - expected_resting_potential.0).abs() < 1.0);

            // Instantaneously set the membrane voltage very low. It should
            // recover.
            segment.membrane_potential = MilliVolts(-160.0);
            for _ in 1..10000 {
                dbg!(&segment.membrane_potential.0);
                segment.step(&BODY_TEMPERATURE, &INTERSTICIAL_FLUID, &interval);
            }
            dbg!(&expected_resting_potential.0);
            assert!((segment.membrane_potential.0 - expected_resting_potential.0).abs() < 1.0);

            // Instantaneously set the membrane voltage very high. It should
            // recover.
            segment.membrane_potential = MilliVolts(1.0);
            for _ in 1..10000 {
                dbg!(&segment.membrane_potential.0);
                segment.step(&BODY_TEMPERATURE, &INTERSTICIAL_FLUID, &interval);
            }
            dbg!(&expected_resting_potential.0);
            assert!((segment.membrane_potential.0 - expected_resting_potential.0).abs() < 1.0);
        }

        #[test]
        // A membrane with a leak current should take a certain amount of
        // time to equilibrate.
        pub fn leak_timecourse() {
            let interval = Interval(0.0001);
            let mut segment = simple_leak();
            let mut t = 0.0;

            let target = cl_reversal(
                &segment.intracellular_solution,
                &INTERSTICIAL_FLUID,
                &BODY_TEMPERATURE,
            );

            segment.membrane_potential = MilliVolts(-100.0);
            while (segment.membrane_potential.0 - target.0).abs() > 1.0 && t < 0.5 {
                dbg!(&segment.membrane_potential);
                segment.step(&BODY_TEMPERATURE, &INTERSTICIAL_FLUID, &interval);
                t += interval.0;
            }
            dbg!(&t);

            // It takes about 8ms for the leak current to bring the membrane
            // voltage up from -100mV to -88 mV (1 mV from the Cl reversal
            // potential).
            //
            // This was determined by running the code. Should confirm
            // mathematically.
            assert!(t > 0.001 && t < 0.009)
        }

        #[test]
        // A segment with a single potassium current should settle at
        // the potassium equillibrium potential, no matter what membrane
        // potential it starts at (with some caveats - starting too negative
        // will close all the K channels and the membrane potential will
        // remain constant).
        pub fn k_membrane_reaches_k_reversal_potential() {
            let mut segment = k_channels_only();

            let expected_resting_potential = k_reversal(
                &segment.intracellular_solution,
                &INTERSTICIAL_FLUID,
                &BODY_TEMPERATURE,
            );

            let interval = Interval(0.001);

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
        pub fn giant_squid_one_membrane_voltage_step() {
            let interval = Interval(1e-4);
            let mut segment = giant_squid_axon();
            let area = segment.surface_area();

            let v_m_0 = segment.membrane_potential.clone();
            let e_k = k_reversal(
                &segment.intracellular_solution,
                &INTERSTICIAL_FLUID,
                &BODY_TEMPERATURE,
            );
            let g_k = segment.membrane.membrane_channels[0].siemens_per_square_cm
                * segment.membrane.membrane_channels[0]
                    .channel
                    .conductance_coefficient()
                * area;

            let e_na = na_reversal(
                &segment.intracellular_solution,
                &INTERSTICIAL_FLUID,
                &BODY_TEMPERATURE,
            );
            let g_na = segment.membrane.membrane_channels[1].siemens_per_square_cm
                * segment.membrane.membrane_channels[1]
                    .channel
                    .conductance_coefficient()
                * area;

            let e_cl = cl_reversal(
                &segment.intracellular_solution,
                &INTERSTICIAL_FLUID,
                &BODY_TEMPERATURE,
            );
            let g_cl = segment.membrane.membrane_channels[2].siemens_per_square_cm
                * segment.membrane.membrane_channels[2]
                    .channel
                    .conductance_coefficient()
                * area;
            let ionic_current_amps =
                (g_k * (v_m_0.0 - e_k.0) + g_na * (v_m_0.0 - e_na.0) + g_cl * (v_m_0.0 - e_cl.0))
                    * 1e-3;
            let dv_dt_millivolts =
                -1.0 * ionic_current_amps / (segment.membrane.capacitance.0 * area) * 1000.0;
            dbg!(dv_dt_millivolts);
            let expected_v =
                MilliVolts(segment.membrane_potential.0 + dv_dt_millivolts * interval.0);

            segment.step(&BODY_TEMPERATURE, &INTERSTICIAL_FLUID, &interval);
            dbg!(&segment.membrane_potential);
            assert!((segment.membrane_potential.0 - expected_v.0).abs() < 1e-10);

            for _ in 0..100000 {
                segment.step(&BODY_TEMPERATURE, &INTERSTICIAL_FLUID, &interval);
                dbg!(&segment.membrane_potential);
                let act = segment.membrane.membrane_channels[1]
                    .channel
                    .clone()
                    .activation
                    .unwrap()
                    .magnitude;
                let inact = segment.membrane.membrane_channels[1]
                    .channel
                    .clone()
                    .inactivation
                    .unwrap()
                    .magnitude;
                dbg!(act);
                dbg!(inact);
            }
            assert!(false)
        }

        #[test]
        // A membrane with some combination of passive K, Na and Cl channels
        // should settle at a membrate potential determined by the GHK equation.
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

            dbg!(ghk(00e-3, 36e-3, 0.3e-3).0);
            assert!((ghk(00e-3, 36e-3, 0.3e-3).0 - -89.0) < 1.0);
            // assert!((ghk(1e-3, 2e-3, 3e-3).0 - -58.0) < 1.0);

            // Example 1: Low Na+ conductance, high Cl- conductance.
            let (na, k, cl) = (0e-3, 36e-3, 3e-3);
            let mut segment = passive_channels(Siemens(na), Siemens(k), Siemens(cl));
            for _ in 1..10 {
                dbg!(&segment.membrane_potential.0);
                segment.step(&BODY_TEMPERATURE, &INTERSTICIAL_FLUID, &interval);
            }
            let expected_voltage = ghk(na, k, cl);
            dbg!(&expected_voltage);
            assert!((segment.membrane_potential.0 - expected_voltage.0).abs() < 1e-3);

            // Example 2: High Na+ conductance, low Cl- conductance.
            let (na, k, cl) = (3e-3, 2e-3, 1e-3);
            let mut segment = passive_channels(Siemens(na), Siemens(k), Siemens(cl));
            for _ in 1..10 {
                dbg!(&segment.membrane_potential.0);
                segment.step(&BODY_TEMPERATURE, &INTERSTICIAL_FLUID, &interval);
            }
            let expected_voltage = ghk(na, k, cl);
            dbg!(&expected_voltage);
            assert!((segment.membrane_potential.0 - expected_voltage.0).abs() < 1e-3);
        }
    }

    #[test]
    fn ampa_receptor_reversal_potential_is_zero() {
        let interval = Interval(1e-6);
        let mut ampa_segment = Segment {
            intracellular_solution: EXAMPLE_CYTOPLASM,
            synaptic_current: MicroAmps(0.0),
            geometry: Geometry {
                diameter: Diameter(1e-3),
                length: 1e-3,
            },
            input_current: MicroAmpsPerSquareCm(0.0),
            membrane_potential: MilliVolts(-80.0),
            membrane: Membrane {
                membrane_channels: vec![MembraneChannel {
                    channel: common_channels::AMPA_CHANNEL.build(&MilliVolts(-80.0)),
                    siemens_per_square_cm: 0.3e-3,
                }],
                capacitance: FaradsPerSquareCm(1e-6),
            },
        };
        assert!((ampa_segment.membrane_potential.0 - -80.0).abs() < 1.0);
        for _ in 1..100000 {
            ampa_segment.step(&BODY_TEMPERATURE, &INTERSTICIAL_FLUID, &interval);
        }
        assert!((ampa_segment.membrane_potential.0).abs() < 1.0);
    }
}
