use crate::dimension::{Siemens, Volts, Interval};
use crate::neuron::solution::Solution;

pub fn na_voltage_gated_conductance(
    na_max_conductance: Siemens,
    membrane_potential: Volts,
    na_reversal_potential: Volts,
) -> Siemens {
    let m: f32 = 0.182 * (membrane_potential.0 + 0.035)
        / (1.0 - (-1.0 * (membrane_potential.0 + 0.035) / 9.0).exp());
    let h: f32 = 0.25 * (-1.0 * (membrane_potential.0 * 90.0) / 12.0).exp();
    Siemens(0.0)
}

pub struct ActivationDynamics {
    alpha: f32,
    beta: f32,
}

pub struct Activation {
        n: f32,
        m: f32,
        h: f32,
        n_dynamics: ActivationDynamics,
        m_dynamics: ActivationDynamics,
        h_dynamics: ActivationDynamics,
}

pub struct RelativePermeability {
    na: f32,
    k: f32,
    ca: f32,
}

pub struct Channel {
    name: String,
    activation: Option<Activation>,
    na_max: Siemens,
    k_max: Siemens,
    ca_max: Siemens,
    leak: Siemens,
}

impl Activation {
    pub fn step(&mut self, membrane_potential: Volts, duration: Interval) {
        let v = membrane_potential.0;
        let dn_dt = self.n_dynamics.alpha * v * (1.0 - self.n) - self.n_dynamics.beta * v * self.n;
        let dm_dt = self.m_dynamics.alpha * v * (1.0 - self.m) - self.m_dynamics.beta * v * self.m;
        let dh_dt = self.h_dynamics.alpha * v * (1.0 - self.h) - self.h_dynamics.beta * v * self.h;
        self.n = self.n + dn_dt * duration.0;
        self.m = self.m + dm_dt * duration.0;
        self.h = self.h + dh_dt * duration.0;
    }

    pub fn conductance(mut self, membrane_potential: Volts, internal_solution: &Solution, external_solution: &Solution) {}
}
