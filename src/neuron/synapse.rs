
use std::str::FromStr;
use bevy::prelude::Component;

use crate::dimension::{
    AreaSquareMillimeters, Interval, Kelvin, MicroAmps, MilliVolts, Molar,
};
use crate::neuron::channel::{ca_reversal, cl_reversal, k_reversal, na_reversal};
use crate::neuron::membrane::MembraneChannel;
use crate::neuron::Solution;
use crate::serialize;

#[derive(Clone, Debug, Component)]
pub struct SynapseMembranes {
    pub cleft_solution: Solution,
    pub transmitter_concentrations: TransmitterConcentrations,
    pub presynaptic_pumps: Vec<TransmitterPump>,
    pub postsynaptic_receptors: Vec<Receptor>,
    pub surface_area: AreaSquareMillimeters,
}

#[derive(Clone, Debug)]
pub struct TransmitterConcentrations {
    pub glutamate: Molar,
    pub gaba: Molar,
}

impl TransmitterConcentrations {
    pub fn serialize(&self) -> serialize::TransmitterConcentrations {
        serialize::TransmitterConcentrations {
            glutamate_molar: self.glutamate.0,
            gaba_molar: self.gaba.0,
        }
    }

    pub fn deserialize(s: &serialize::TransmitterConcentrations) -> Result<TransmitterConcentrations, String> {
        Ok(TransmitterConcentrations {
            glutamate: Molar(s.glutamate_molar),
            gaba: Molar(s.gaba_molar),
        })
    }
}

// TODO: Should the synapse mechanisms be temperature-dependent?
impl SynapseMembranes {
    /// Update the state of the synaptic cleft, and report the current that flows into the
    /// post-synaptic segment.
    pub fn step(
        &mut self,
        _temperature: &Kelvin,
        presynaptic_potential: &MilliVolts,
        postsynaptic_potential: &MilliVolts,
        interval: &Interval,
    ) {
        // First update the concentration of synaptic messengers.
        self.presynaptic_pumps.iter_mut().for_each(|pump| {
            let update_concentration = |initial_concentration: &Molar| {
                let v = &presynaptic_potential;
                let concentration_slope = (pump.target_concentration(v).0
                    - initial_concentration.0)
                    / pump.time_constant(v);
                Molar(initial_concentration.0 + concentration_slope * interval.0)
            };
            match pump.transmitter {
                Transmitter::Glutamate => {
                    self.transmitter_concentrations.glutamate =
                        update_concentration(&self.transmitter_concentrations.glutamate);
                }
                Transmitter::Gaba => {
                    self.transmitter_concentrations.gaba =
                        update_concentration(&self.transmitter_concentrations.gaba);
                }
            };
        });

        // Then update the pump and receptor states.
        self.postsynaptic_receptors.iter_mut().for_each(|receptor| {
            receptor
                .membrane_channel
                .channel
                .step(&postsynaptic_potential, interval)
        });
    }

    pub fn apply_current(
        &self,
        interval: &Interval,
        temperature: &Kelvin,
        postsynaptic_potential: &mut MilliVolts,
        postsynaptic_solution: &Solution
    ) {
        // TODO: Not sure how to handle the I->V conversion
        // for post-synaptic current.
        let synapse_resistance_ohms = 1000000000.0;

        let dv_dt_volts_per_second =
            self.current(temperature, postsynaptic_potential, postsynaptic_solution).0 * 0.000001 * synapse_resistance_ohms * -1.0;

        let delta_mv = MilliVolts(dv_dt_volts_per_second * interval.0 * 1000.0);
        // dbg!(&delta_mv);

        postsynaptic_potential.0 =
            postsynaptic_potential.0 + delta_mv.0;
    }

    pub fn current(
        &self,
        temperature: &Kelvin,
        postsynaptic_potential: &MilliVolts,
        postsynaptic_solution: &Solution
    ) -> MicroAmps {
        let current_per_square_cm = self
            .postsynaptic_receptors
            .iter()
            .map(|receptor| {
                let channel_current_per_cm = receptor.membrane_channel.channel_current_per_cm(
                    &k_reversal(
                        &postsynaptic_solution,
                        &self.cleft_solution,
                        temperature,
                    ),
                    &na_reversal(
                        &postsynaptic_solution,
                        &self.cleft_solution,
                        temperature,
                    ),
                    &cl_reversal(
                        &postsynaptic_solution,
                        &self.cleft_solution,
                        temperature,
                    ),
                    &ca_reversal(
                        &postsynaptic_solution,
                        &self.cleft_solution,
                        temperature,
                    ),
                    &postsynaptic_potential,
                );
                let gating_coefficient = receptor
                    .neurotransmitter_sensitivity
                    .gating_coefficient(&self.transmitter_concentrations);
                // dbg!(channel_current_per_cm);
                // dbg!(gating_coefficient);
                channel_current_per_cm * gating_coefficient
            })
            .sum::<f32>();

        MicroAmps(current_per_square_cm * self.surface_area.0)
    }

    pub fn serialize(&self) -> serialize::SynapseMembranes {
        serialize::SynapseMembranes {
            cleft_solution: self.cleft_solution.serialize(),
            transmitter_concentrations: self.transmitter_concentrations.serialize(),
            presynaptic_pumps: self.presynaptic_pumps.iter().map(|p| p.serialize()).collect(),
            postsynaptic_receptors: self.postsynaptic_receptors.iter().map(|r| r.serialize()).collect(),
            surface_area_square_mm: self.surface_area.0,
        }
    }

    pub fn deserialize(s: &serialize::SynapseMembranes) -> Result<Self, String> {
        Ok(SynapseMembranes {
            cleft_solution: Solution::deserialize(&s.cleft_solution)?,
            transmitter_concentrations: TransmitterConcentrations::deserialize(&s.transmitter_concentrations)?,
            presynaptic_pumps: s.presynaptic_pumps.iter().map(|p| TransmitterPump::deserialize(p)).collect::<Result<_,_>>()?,
            postsynaptic_receptors: s.postsynaptic_receptors.iter().map(|r| Receptor::deserialize(r)).collect::<Result<_,_>>()?,
            surface_area: AreaSquareMillimeters(s.surface_area_square_mm),
        })
    }
}

#[derive(Clone, Debug)]
pub enum Transmitter {
    Glutamate,
    Gaba,
}

impl Transmitter {
    pub fn to_string(&self) -> String {
        match self {
            Transmitter::Glutamate => "glutamate".to_string(),
            Transmitter::Gaba => "gaba".to_string(),
        }
    }
}

impl FromStr for Transmitter {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "glutamate" => Ok(Transmitter::Glutamate),
            "gaba" => Ok(Transmitter::Gaba),
            _ => Err(format!("Unknown transmitter {s}")),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Receptor {
    pub membrane_channel: MembraneChannel,
    pub neurotransmitter_sensitivity: Sensitivity,
}

impl Receptor {
    pub fn serialize(&self) -> serialize::Receptor {
        serialize::Receptor {
            membrane_channel: self.membrane_channel.serialize(),
            neurotransmitter_sensitivity: self.neurotransmitter_sensitivity.serialize(),
        }
    }

    pub fn deserialize(s: &serialize::Receptor) -> Result<Self, String> {
        Ok(Receptor {
            membrane_channel: MembraneChannel::deserialize(&s.membrane_channel),
            neurotransmitter_sensitivity: Sensitivity::deserialize(&s.neurotransmitter_sensitivity)?,
        })
    }
}

#[derive(Clone, Debug)]
pub struct Sensitivity {
    pub transmitter: Transmitter,
    pub concentration_at_half_max: Molar,
    pub slope: f32,
}

impl Sensitivity {
    pub fn gating_coefficient(
        &self,
        transmitter_concentrations: &TransmitterConcentrations,
    ) -> f32 {
        let mk_coefficient = |concentration: &Molar| {
            1.0 / (1.0 + ((-1.0 * (concentration.0 - self.concentration_at_half_max.0) * self.slope)).exp())
        };
        match self.transmitter {
            Transmitter::Glutamate => mk_coefficient(&transmitter_concentrations.glutamate),
            Transmitter::Gaba => mk_coefficient(&transmitter_concentrations.gaba),
        }
    }

    pub fn serialize(&self) -> serialize::Sensitivity {
        serialize::Sensitivity {
            transmitter: self.transmitter.to_string(),
            concentration_at_half_max_molar: self.concentration_at_half_max.0,
            slope: self.slope,
        }
    }

    pub fn deserialize(s: &serialize::Sensitivity) -> Result<Self, String> {
        Ok(Sensitivity {
            transmitter: Transmitter::from_str(&s.transmitter)?,
            concentration_at_half_max: Molar(s.concentration_at_half_max_molar),
            slope: s.slope,
        })
    }
}

#[derive(Clone, Debug)]
pub struct TransmitterPump {
    pub transmitter: Transmitter,
    pub transmitter_pump_params: TransmitterPumpParams,
}

impl TransmitterPump {
    pub fn target_concentration(&self, v: &MilliVolts) -> Molar {
        Molar(
            self.transmitter_pump_params.target_concentration_min.0
                + (self.transmitter_pump_params.target_concentration_max.0 - self.transmitter_pump_params.target_concentration_min.0)
                    / (1.0
                        + ((self.transmitter_pump_params.target_concentration_v_at_half_max.0 - v.0)
                            / self.transmitter_pump_params.target_concentration_v_slope)
                            .exp()),
        )
    }

    pub fn time_constant(&self, v: &MilliVolts) -> f32 {
        let numerator = -1.0 * (self.transmitter_pump_params.time_constant_v_at_max_tau.0 - v.0).powi(2);
        let denominator = self.transmitter_pump_params.time_constant_sigma.powi(2);
        self.transmitter_pump_params.time_constant_c_base
            + self.transmitter_pump_params.time_constant_c_amp * (numerator / denominator).exp()
    }

    pub fn serialize(&self) -> serialize::TransmitterPump {
        serialize::TransmitterPump {
            transmitter: self.transmitter.to_string(),
            transmitter_pump_params: self.transmitter_pump_params.serialize(),
        }
    }

    pub fn deserialize(s: &serialize::TransmitterPump) -> Result<Self, String> {
        Ok(TransmitterPump {
            transmitter: Transmitter::from_str(&s.transmitter)?,
            transmitter_pump_params: TransmitterPumpParams::deserialize(&s.transmitter_pump_params)?,
        })
    }
}

// TODO: The time-constant-as-function-of-voltage doesn't do what I wanted it to do:
// Release transmitter at one rate and remove it at a different rate.
// To achieve that, we need to have two different pumps, one in, one out.
#[derive(Clone, Debug)]
pub struct TransmitterPumpParams {
    pub target_concentration_max: Molar,
    pub target_concentration_min: Molar,
    pub target_concentration_v_at_half_max: MilliVolts,
    pub target_concentration_v_slope: f32,
    pub time_constant_v_at_max_tau: MilliVolts,
    pub time_constant_c_base: f32,
    pub time_constant_c_amp: f32,
    pub time_constant_sigma: f32,
}

impl TransmitterPumpParams {
    pub fn serialize(&self) -> serialize::TransmitterPumpParams {
        serialize::TransmitterPumpParams {
            target_concentration: serialize::Sigmoid {
                max_molar: self.target_concentration_max.0,
                min_molar: self.target_concentration_min.0,
                v_at_half_max_mv: self.target_concentration_v_at_half_max.0,
                slope: self.target_concentration_v_slope,
                // log_space: false,
            },
            time_constant: serialize::TimeConstant::Sigmoid {
                c_base: self.time_constant_c_base,
                c_amp: self.time_constant_c_amp,
                v_at_max_tau_mv: self.time_constant_v_at_max_tau.0,
                sigma: self.time_constant_sigma,
            },
        }
    }

    pub fn deserialize(s: &serialize::TransmitterPumpParams) -> Result<Self, String> {
        let (v_at_max_tau_mv, c_base, c_amp, sigma) = match s.time_constant {
            serialize::TimeConstant::Sigmoid{v_at_max_tau_mv, c_base, c_amp, sigma} => (v_at_max_tau_mv, c_base, c_amp, sigma),
            _ => panic!("TODO: FIXME: Only sigmoid time constants are supported in synapses"),
        };
        Ok(TransmitterPumpParams {
            target_concentration_max: Molar(s.target_concentration.max_molar),
            target_concentration_min: Molar(s.target_concentration.min_molar),
            target_concentration_v_at_half_max: MilliVolts(s.target_concentration.v_at_half_max_mv),
            target_concentration_v_slope: s.target_concentration.slope,
            time_constant_c_base: c_base,
            time_constant_c_amp: c_amp,
            time_constant_v_at_max_tau: MilliVolts(v_at_max_tau_mv),
            time_constant_sigma: sigma,
        })
    }
}

pub mod examples {
    use super::*;
    use crate::dimension::{MilliVolts, Molar};
    use crate::neuron::channel::common_channels::AMPA_CHANNEL;
    use crate::neuron::solution::INTERSTICIAL_FLUID;

    // Note: The numbers here are totally made up.
    pub fn glutamate_release() -> TransmitterPump {
        TransmitterPump {
            transmitter: Transmitter::Glutamate,
            transmitter_pump_params: TransmitterPumpParams {
                target_concentration_max: Molar(1.1e-2),
                target_concentration_min: Molar(1e-4),
                target_concentration_v_at_half_max: MilliVolts(0.0),
                target_concentration_v_slope: 1.0,
                time_constant_c_amp: 1e-6,
                time_constant_c_base: 1e-3,
                time_constant_sigma: 1.0,
                time_constant_v_at_max_tau: MilliVolts(0.0),
            },
        }
    }

    // Note: The numbers here are totally made up.
    pub fn gaba_release() -> TransmitterPump {
        TransmitterPump {
            transmitter: Transmitter::Gaba,
            transmitter_pump_params: TransmitterPumpParams {
                target_concentration_max: Molar(1.1e-2),
                target_concentration_min: Molar(1e-4),
                target_concentration_v_at_half_max: MilliVolts(0.0),
                target_concentration_v_slope: 1.0,
                time_constant_c_amp: 1e-6,
                time_constant_c_base: 1e-3,
                time_constant_sigma: 1.0,
                time_constant_v_at_max_tau: MilliVolts(0.0),
            },
        }
    }

    // Note: The numbers here are totally made up.
    pub fn ampa_receptor(initial_voltage: &MilliVolts) -> Receptor {
        Receptor {
            membrane_channel: MembraneChannel {
                channel: AMPA_CHANNEL.build(initial_voltage),
                siemens_per_square_cm: 1e7,
            },
            neurotransmitter_sensitivity: Sensitivity {
                transmitter: Transmitter::Glutamate,
                concentration_at_half_max: Molar(3e-3), // TODO: determine the right value.
                slope: 10000.0,                            // TODO: determine the right value.
            },
        }
    }
    pub fn excitatory_synapse(initial_voltage: &MilliVolts) -> SynapseMembranes {
        SynapseMembranes {
            cleft_solution: INTERSTICIAL_FLUID,
            transmitter_concentrations: TransmitterConcentrations {
                glutamate: Molar(0.1e-3),
                gaba: Molar(0.1e-3),
            },
            presynaptic_pumps: vec![glutamate_release()],
            postsynaptic_receptors: vec![ampa_receptor(initial_voltage)],
            surface_area: AreaSquareMillimeters(1e-6),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::examples;
    use super::*;
    use crate::constants::BODY_TEMPERATURE;
    use crate::dimension::MicroAmpsPerSquareCm;
    use crate::neuron::solution::INTERSTICIAL_FLUID;

    #[test]
    fn sensitivity_function() {
        let sensitivity = Sensitivity {
            transmitter: Transmitter::Glutamate,
            concentration_at_half_max: Molar(1.0),
            slope: 100.0,
        };
        let epsilon = 1e-9;

        // Low concentrations should have low gating.
        assert!((sensitivity.gating_coefficient(
            &TransmitterConcentrations {
                glutamate: Molar(1e-8),
                gaba: Molar(1e-8),
            }) - 0.0).abs() < epsilon);

        // Half-max concentrations should have 0.5 gating.
        assert!((sensitivity.gating_coefficient(
            &TransmitterConcentrations {
                glutamate: Molar(1.0),
                gaba: Molar(1e-8),
            }) - 0.5).abs() < epsilon);

        // Slighly higher concentrations concentrations should have slightly higher gating.
        assert!((sensitivity.gating_coefficient(
            &TransmitterConcentrations {
                glutamate: Molar(1.01),
                gaba: Molar(1e-8),
            }) - 0.7310584).abs() < epsilon);

        // High concentrations should have high gating.
        assert!((sensitivity.gating_coefficient(
            &TransmitterConcentrations {
                glutamate: Molar(2.0),
                gaba: Molar(1e-8),
            }) - 1.0).abs() < epsilon);

    }

    #[test]
    fn instantaneous_cleft_pereability() {
        let initial_voltage = MilliVolts(-80.0);
        let mut synapse = examples::excitatory_synapse(&initial_voltage);

        // Empty cleft should have low gating.
        assert!(synapse
            .postsynaptic_receptors[0]
            .neurotransmitter_sensitivity
            .gating_coefficient(&TransmitterConcentrations {
                glutamate: Molar(0.0),
                gaba: Molar(0.0),
            }) < 0.1);

        // Full cleft should have high gating.
        assert!(synapse
            .postsynaptic_receptors[0]
            .neurotransmitter_sensitivity
            .gating_coefficient(&TransmitterConcentrations {
                glutamate: Molar(0.5),
                gaba: Molar(0.0),
            }) > 0.95);
    }

    #[test]
    fn excited_synapse_releases_glutamate() {
        let epsilon = 1e-9;
        let mut segment_1 = crate::neuron::segment::examples::giant_squid_axon();
        let mut segment_2 = crate::neuron::segment::examples::giant_squid_axon();
        let initial_voltage = MilliVolts(-70.0);
        segment_1.membrane_potential = initial_voltage.clone();
        segment_2.membrane_potential = initial_voltage.clone();
        segment_2.input_current = MicroAmpsPerSquareCm(-20.0);
        let mut synapse = examples::excitatory_synapse(&initial_voltage);

        // Before glutamate builds up in the synapse, synaptic current should be
        // small.
        dbg!(synapse.current(&BODY_TEMPERATURE, &segment_2));
        assert!(synapse.current(&BODY_TEMPERATURE, &segment_2).0 < 1.0);

        // Run forward by 1.5ms. This is enough time for segment_1 to spike,
        // which should push glutamate into the synapse and trigger some
        // postsynaptic current.
        let interval = Interval(1e-6);
        for n in 0..1500 {
            segment_1.step(&BODY_TEMPERATURE, &INTERSTICIAL_FLUID, &interval);
            segment_2.step(&BODY_TEMPERATURE, &INTERSTICIAL_FLUID, &interval);
            synapse.step(&BODY_TEMPERATURE, &segment_1, &segment_2, &interval);
            synapse.apply_current(&interval, &BODY_TEMPERATURE, &mut segment_2);

            // Pretend there are 1000 of these synapses behaving the same way.
            for n in 0..1000 {
                synapse.apply_current(&interval, &BODY_TEMPERATURE, &mut segment_2);
            }

            if n % 100 == 0 {
                let m_g = &synapse.transmitter_concentrations.glutamate.0;
                let coeff = &synapse.postsynaptic_receptors[0]
                    .neurotransmitter_sensitivity
                    .gating_coefficient(&synapse.transmitter_concentrations);
                let i = synapse.current(&BODY_TEMPERATURE, &segment_2).0;
                let v_1 = &segment_1.membrane_potential;
                let v_2 = &segment_2.membrane_potential;
                dbg!(v_1.0);
                dbg!(v_2.0);
                dbg!(m_g);
                dbg!(coeff);
                dbg!(i);
                dbg!(synapse.current(&BODY_TEMPERATURE, &segment_2).0);
            }
        }

        let glu_pump = &synapse.presynaptic_pumps[0];

        dbg!(synapse.current(&BODY_TEMPERATURE, &segment_2));
        dbg!(glu_pump.target_concentration( &segment_1.membrane_potential ));
        // assert_eq!(synapse.transmitter_concentrations.glutamate, Molar(1.0));
        assert!( (synapse.transmitter_concentrations.glutamate.0 - 0.005202022).abs() < epsilon );
        // assert_eq!(synapse.current(&BODY_TEMPERATURE, &segment_2).0, 1.0);
    }
}
