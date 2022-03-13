use crate::dimension::{
    Diameter, Interval, Kelvin, MicroAmps, MicroAmpsPerSquareCm, MilliVolts, Molar,
};
use crate::neuron::channel::Channel;
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
                let channel_current = receptor.membrane_channel.channel_current_per_cm(
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
                channel_current * gating_coefficient
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
            1.0 / (1.0
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
    pub target_concentration_v_at_half_max: MilliVolts,
    pub target_concentration_v_slope: f32,
    pub time_constant_v_at_max_tau: MilliVolts,
    pub time_constant_c_base: f32,
    pub time_constant_c_amp: f32,
    pub time_constant_sigma: f32,
}
