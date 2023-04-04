use crate::dimension::{
    Diameter, Interval, Kelvin, MicroAmps, MilliVolts, Molar,
};
use crate::neuron::channel::{ca_reversal, cl_reversal, k_reversal, na_reversal};
use crate::neuron::membrane::MembraneChannel;
use crate::neuron::segment::Segment;
use crate::neuron::Solution;

#[derive(Clone, Debug)]
pub struct Synapse {
    pub cleft_solution: Solution,
    pub transmitter_concentrations: TransmitterConcentrations,
    pub presynaptic_pumps: Vec<TransmitterPump>,
    pub postsynaptic_receptors: Vec<Receptor>,
    pub surface_area: Diameter,
}

#[derive(Clone, Debug)]
pub struct TransmitterConcentrations {
    pub glutamate: Molar,
    pub gaba: Molar,
}

impl Synapse {
    /// Update the state of the synaptic cleft, and report the current that flows into the
    /// post-synaptic segment.
    pub fn step(
        &mut self,
        temperature: &Kelvin,
        presynaptic_segment: &Segment,
        postsynaptic_segment: &Segment,
        interval: &Interval,
    ) {
        // First update the concentration of synaptic messengers.
        self.presynaptic_pumps.iter_mut().for_each(|pump| {
            let update_concentration = |initial_concentration: &Molar| {
                let v = &presynaptic_segment.membrane_potential;
                let concentration_slope = (pump.target_concentration(v).0
                    - initial_concentration.0)
                    / pump.time_constant(v);
                Molar(initial_concentration.0 + pump.scale * concentration_slope * interval.0)
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
                .step(&postsynaptic_segment.membrane_potential, interval)
        });
    }

    pub fn current(&self, temperature: &Kelvin, postsynaptic_segment: &Segment) -> MicroAmps {
        let current_per_square_cm = self
            .postsynaptic_receptors
            .iter()
            .map(|receptor| {
                let channel_current_per_cm = receptor.membrane_channel.channel_current_per_cm(
                    &k_reversal(
                        &postsynaptic_segment.intracellular_solution,
                        &self.cleft_solution,
                        temperature,
                    ),
                    &na_reversal(
                        &postsynaptic_segment.intracellular_solution,
                        &self.cleft_solution,
                        temperature,
                    ),
                    &cl_reversal(
                        &postsynaptic_segment.intracellular_solution,
                        &self.cleft_solution,
                        temperature,
                    ),
                    &ca_reversal(
                        &postsynaptic_segment.intracellular_solution,
                        &self.cleft_solution,
                        temperature,
                    ),
                    &postsynaptic_segment.membrane_potential,
                );
                let gating_coefficient = receptor
                    .neurotransmitter_sensitivity
                    .gating_coefficient(&self.transmitter_concentrations);
                dbg!(channel_current_per_cm);
                dbg!(gating_coefficient);
                channel_current_per_cm * gating_coefficient
            })
            .sum::<f32>();

        MicroAmps(current_per_square_cm * self.surface_area.0)
    }
}

#[derive(Clone, Debug)]
pub enum Transmitter {
    Glutamate,
    Gaba,
}

#[derive(Clone, Debug)]
pub struct Receptor {
    pub membrane_channel: MembraneChannel,
    pub neurotransmitter_sensitivity: Sensitivity,
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
            1.0 / (1.0 + ((self.concentration_at_half_max.0 - concentration.0) / self.slope).exp())
        };
        match self.transmitter {
            Transmitter::Glutamate => mk_coefficient(&transmitter_concentrations.glutamate),
            Transmitter::Gaba => mk_coefficient(&transmitter_concentrations.gaba),
        }
    }
}

#[derive(Clone, Debug)]
pub struct TransmitterPump {
    pub scale: f32,
    pub transmitter: Transmitter,
    pub params: TransmitterPumpParams,
}

impl TransmitterPump {
    pub fn target_concentration(&self, v: &MilliVolts) -> Molar {
        Molar(
            self.params.target_concentration_min.0
                + (self.params.target_concentration_max.0 - self.params.target_concentration_min.0)
                    / (1.0
                        + ((self.params.target_concentration_v_at_half_max.0 - v.0)
                            / self.params.target_concentration_v_slope)
                            .exp()),
        )
    }

    pub fn time_constant(&self, v: &MilliVolts) -> f32 {
        let numerator = -1.0 * (self.params.time_constant_v_at_max_tau.0 - v.0).powi(2);
        let denominator = self.params.time_constant_sigma.powi(2);
        self.params.time_constant_c_base
            + self.params.time_constant_c_amp * (numerator / denominator).exp()
    }
}

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

pub mod examples {
    use super::*;
    use crate::dimension::{MilliVolts, Molar};
    use crate::neuron::channel::common_channels::AMPA_CHANNEL;
    use crate::neuron::solution::INTERSTICIAL_FLUID;

    // Note: The numbers here are totally made up.
    pub fn glutamate_removal() -> TransmitterPump {
        TransmitterPump {
            transmitter: Transmitter::Glutamate,
            scale: 1.0,
            params: TransmitterPumpParams {
                target_concentration_max: Molar(1.1e-4),
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
    pub fn glutamate_release() -> TransmitterPump {
        TransmitterPump {
            transmitter: Transmitter::Glutamate,
            scale: 1.0,
            params: TransmitterPumpParams {
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
    pub fn gaba_removal() -> TransmitterPump {
        TransmitterPump {
            transmitter: Transmitter::Gaba,
            scale: 1.0,
            params: TransmitterPumpParams {
                target_concentration_max: Molar(1.1e-4),
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
            scale: 1.0,
            params: TransmitterPumpParams {
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
                siemens_per_square_cm: 100.0,
            },
            neurotransmitter_sensitivity: Sensitivity {
                transmitter: Transmitter::Glutamate,
                concentration_at_half_max: Molar(1e-3), // TODO: determine the right value.
                slope: 1e-3,                            // TODO: determine the right value.
            },
        }
    }
    pub fn excitatory_synapse(initial_voltage: &MilliVolts) -> Synapse {
        Synapse {
            cleft_solution: INTERSTICIAL_FLUID,
            transmitter_concentrations: TransmitterConcentrations {
                glutamate: Molar(0.1e-3),
                gaba: Molar(0.1e-3),
            },
            presynaptic_pumps: vec![glutamate_removal(), glutamate_release()],
            postsynaptic_receptors: vec![ampa_receptor(initial_voltage)],
            surface_area: Diameter(1e-6),
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
    fn excited_synapse_releases_glutamate() {
        let mut segment_1 = crate::neuron::segment::examples::giant_squid_axon();
        let mut segment_2 = crate::neuron::segment::examples::giant_squid_axon();
        let initial_voltage = MilliVolts(-70.0);
        segment_1.membrane_potential = initial_voltage.clone();
        segment_2.membrane_potential = initial_voltage.clone();
        segment_2.input_current = MicroAmpsPerSquareCm(-15.0);
        let mut synapse = examples::excitatory_synapse(&initial_voltage);

        // Before glutamate builds up in the synapse, synaptic current should be
        // small.
        dbg!(synapse.current(&BODY_TEMPERATURE, &segment_2));
        assert!(synapse.current(&BODY_TEMPERATURE, &segment_2).0 < 1.0);

        let interval = Interval(1e-6);
        for n in 0..2000 {
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
            }
            segment_1.step(&BODY_TEMPERATURE, &INTERSTICIAL_FLUID, &interval);
            segment_2.step(&BODY_TEMPERATURE, &INTERSTICIAL_FLUID, &interval);
            synapse.step(&BODY_TEMPERATURE, &segment_1, &segment_2, &interval);
        }

        dbg!(synapse.current(&BODY_TEMPERATURE, &segment_2));
        assert!(synapse.current(&BODY_TEMPERATURE, &segment_2).0 == 1.0);
    }
}
