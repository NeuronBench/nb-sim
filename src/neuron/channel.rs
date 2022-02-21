use crate::dimension::{Interval, MilliVolts, Siemens, Volts};
use crate::neuron::solution::Solution;

/// The relative permeability of a channel to various ions.
/// These should add to 1.0.
#[derive(Clone, Debug)]
pub struct IonSelectivity {
    /// Sodium+.
    na: f32,
    /// Potasium+.
    k: f32,
    /// Calcium2+.
    ca: f32,
    /// Chloride-.
    cl: f32,
}

const K: IonSelectivity = IonSelectivity {
    na: 0.0,
    k: 1.0,
    ca: 0.0,
    cl: 0.0,
};

const NA: IonSelectivity = IonSelectivity {
    na: 1.0,
    k: 0.0,
    ca: 0.0,
    cl: 0.0,
};

const CA: IonSelectivity = IonSelectivity {
    na: 0.0,
    k: 0.0,
    ca: 1.0,
    cl: 0.0,
};

const CL: IonSelectivity = IonSelectivity {
    na: 0.0,
    k: 0.0,
    ca: 0.0,
    cl: 1.0,
};

impl IonSelectivity {
    pub fn normalize(&self) -> IonSelectivity {
        let sum = self.k + self.na + self.ca + self.cl;
        IonSelectivity {
            k: self.k / sum,
            na: self.na / sum,
            ca: self.ca / sum,
            cl: self.cl / sum,
        }
    }
}

/// State of the voltage-gated conductance, such as the conductance of
/// a voltage-gated sodium channel or a voltage-gated potassium channel.
#[derive(Clone, Debug)]
pub struct Channel {
    /// State of the activation gates.
    activation: Option<GateState>,
    /// State of the inactivation gates.
    inactivation: Option<GateState>,
    /// The ion this channel is permeable to.
    ion_selectivity: IonSelectivity,
}

impl Channel {
    /// Advance the channel conduction state for the activation and inactivation
    /// magnitudes.
    pub fn step(&mut self, membrane_potential: &MilliVolts, interval: &Interval) {
        self.activation
            .iter_mut()
            .for_each(|activation| activation.step(membrane_potential, interval));
        self.inactivation
            .iter_mut()
            .for_each(|inactivation| inactivation.step(membrane_potential, interval));
    }

    /// The
    pub fn conductance_coefficient(&self) -> f32 {
        let activation_coefficient = self.activation.as_ref().map_or(1.0, |gate_state| {
            gate_state
                .magnitude
                .powi(gate_state.parameters.gates as i32)
        });
        let inactivation_coefficient = self.inactivation.as_ref().map_or(1.0, |gate_state| {
            gate_state
                .magnitude
                .powi(gate_state.parameters.gates as i32)
        });
        activation_coefficient * inactivation_coefficient
    }
}

#[derive(Clone, Debug)]
pub struct ChannelBuilder {
    activation_parameters: Option<Gating>,
    inactivation_parameters: Option<Gating>,
    ion_selectivity: IonSelectivity,
}

impl ChannelBuilder {
    /// Construct a new conductance state from a set of activation and
    /// inactivation parameters. Choose an initial state for the activation and
    /// inactivation gates by setting them to their steady-state levels.
    pub fn build(self, initial_membrane_potential: &MilliVolts) -> Channel {
        let activation = self.activation_parameters.map(|parameters| {
            let magnitude = parameters
                .steady_state_magnitude
                .steady_state(initial_membrane_potential);
            GateState {
                magnitude,
                parameters: parameters,
            }
        });
        let inactivation = self.inactivation_parameters.map(|parameters| {
            let magnitude = parameters
                .steady_state_magnitude
                .steady_state(initial_membrane_potential);
            GateState {
                magnitude,
                parameters: parameters,
            }
        });
        Channel {
            activation,
            inactivation,
            ion_selectivity: self.ion_selectivity.normalize(),
        }
    }
}

/// The state for a particular type of game (either the activation or
/// inactivation gate).
#[derive(Clone, Debug)]
pub struct GateState {
    /// The current magnitude of tha conductance component. 'm', 'n' or 'h' in
    /// the Hodgkin-Huxley model.
    pub magnitude: f32,
    /// The parameters determining how the magnitutde evolves with time and
    /// membrane voltage.
    pub parameters: Gating,
}

impl GateState {
    /// Update the activation/inactivation state by computing (a) the
    /// steady-state value at the current membrane voltage, and (b) the time
    /// constant, tau, at the current membrane voltage.
    pub fn step(&mut self, membrane_potential: &MilliVolts, interval: &Interval) {
        let v_inf = self
            .parameters
            .steady_state_magnitude
            .steady_state(membrane_potential);
        let tau = self.parameters.time_constant.tau(membrane_potential);
        let df_dt = (v_inf - self.magnitude) / tau;
        self.magnitude = self.magnitude + df_dt * interval.0;
    }
}

/// The confuration for a single type of gate in a single channel.
#[derive(Clone, Debug)]
pub struct Gating {
    /// The number of such gates in each channel. For instance, the 3
    /// activation gates of a potassium channel, or the 1 inactivation
    /// gate of a sodium channel.
    pub gates: u8,
    pub steady_state_magnitude: Magnitude,
    pub time_constant: TimeConstant,
}

#[derive(Clone, Debug)]
pub struct Magnitude {
    pub v_at_half_max: MilliVolts,
    pub slope: f32,
}

impl Magnitude {
    pub fn steady_state(&self, v: &MilliVolts) -> f32 {
        1.0 / (1.0 + ((self.v_at_half_max.0 - v.0) / self.slope).exp())
    }
}

#[derive(Clone, Debug)]
pub struct TimeConstant {
    pub v_at_max_tau: MilliVolts,
    pub c_base: f32,
    pub c_amp: f32,
    pub sigma: f32,
}

impl TimeConstant {
    pub fn tau(&self, v: &MilliVolts) -> f32 {
        self.c_base
            + self.c_amp * ((-1.0 * (self.v_at_max_tau.0 - v.0).powi(2)) / self.sigma.powi(2)).exp()
    }
}

pub mod common_channels {

    pub mod giant_squid {
        use crate::dimension::MilliVolts;
        use crate::neuron::channel::*;

        /// The Giant Squid axon's Na+ channel.
        pub const NA_CHANNEL: ChannelBuilder = ChannelBuilder {
            ion_selectivity: NA,
            activation_parameters: Some(Gating {
                gates: 3,
                steady_state_magnitude: Magnitude {
                    v_at_half_max: MilliVolts(-40.0),
                    slope: 15.0,
                },
                time_constant: TimeConstant {
                    v_at_max_tau: MilliVolts(-38.0),
                    c_base: 0.04,
                    c_amp: 0.46,
                    sigma: 30.0,
                },
            }),
            inactivation_parameters: Some(Gating {
                gates: 1,
                steady_state_magnitude: Magnitude {
                    v_at_half_max: MilliVolts(-62.0),
                    slope: -7.0,
                },
                time_constant: TimeConstant {
                    v_at_max_tau: MilliVolts(-67.0),
                    c_base: 1.2,
                    c_amp: 7.4,
                    sigma: 20.0,
                },
            }),
        };

        /// The Giant Squid axon's K+ rectifying channel.
        pub const K_CHANNEL: ChannelBuilder = ChannelBuilder {
            ion_selectivity: K,
            activation_parameters: Some(Gating {
                gates: 4,
                steady_state_magnitude: Magnitude {
                    v_at_half_max: MilliVolts(-53.0),
                    slope: 15.0,
                },
                time_constant: TimeConstant {
                    v_at_max_tau: MilliVolts(-79.0),
                    c_base: 1.1,
                    c_amp: 4.7,
                    sigma: 50.0,
                },
            }),
            inactivation_parameters: None,
        };
    }
}
