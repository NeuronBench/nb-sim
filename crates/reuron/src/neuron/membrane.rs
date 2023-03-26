// use crate::constants::{gas_constant, inverse_faraday};
use bevy::prelude::{Assets, Color, Component, FromWorld, Handle, Resource, StandardMaterial, World};
use uuid::Uuid;
use std::hash::Hash;

use crate::dimension::{FaradsPerSquareCm, MilliVolts};
use crate::neuron::channel::Channel;
use crate::serialize;

/// The more static properties of a cell membrane: its permeability to
/// various ions. This may change with the development of the neuron,
/// but it is fairly static, compared to [`MembraneChannelState`].
#[derive(Clone, Component, Debug, Hash)]
pub struct Membrane {
    /// The concentration of channels in this membrane.
    pub membrane_channels: Vec<MembraneChannel>,
    pub capacitance: FaradsPerSquareCm,
}

#[derive(Component, Hash)]
pub struct MembraneVoltage(pub MilliVolts);

impl Membrane {
    pub fn current_per_square_cm(
        &self,
        k_reversal: &MilliVolts,
        na_reversal: &MilliVolts,
        cl_reversal: &MilliVolts,
        ca_reversal: &MilliVolts,
        membrane_potential: &MilliVolts,
    ) -> f32 {
        self.membrane_channels
            .iter()
            .map(|membrane_channel| {
                membrane_channel.channel_current_per_cm(
                    k_reversal,
                    na_reversal,
                    cl_reversal,
                    ca_reversal,
                    membrane_potential,
                )
            })
            .sum()
    }

    /// A quick snapshot of the per_square_cm conductances of each
    /// ion.
    pub fn conductances(&self) -> (f32, f32, f32, f32) {
        let mut k = 0.0;
        let mut na = 0.0;
        let mut cl = 0.0;
        let mut ca = 0.0;
        self.membrane_channels.iter().for_each(|membrane_channel| {
            let gating_coefficient = membrane_channel.channel.conductance_coefficient();
            k += membrane_channel.siemens_per_square_cm
                * gating_coefficient
                * membrane_channel.channel.ion_selectivity.k;

            na += membrane_channel.siemens_per_square_cm
                * gating_coefficient
                * membrane_channel.channel.ion_selectivity.na;

            ca += membrane_channel.siemens_per_square_cm
                * gating_coefficient
                * membrane_channel.channel.ion_selectivity.ca;

            cl += membrane_channel.siemens_per_square_cm
                * gating_coefficient
                * membrane_channel.channel.ion_selectivity.cl;
        });
        (k, na, cl, ca)
    }

    pub fn serialize(&self) -> serialize::Membrane {
        serialize::Membrane {
            id: Uuid::new_v4(),
            channels: self
                .membrane_channels
                .iter()
                .map(|MembraneChannel {
                    channel,
                    siemens_per_square_cm
                }| serialize::MembraneChannel {
                    channel: channel.serialize(),
                    siemens_per_square_cm: siemens_per_square_cm.clone(),
                }).collect(),
            capacitance_farads_per_square_cm: self.capacitance.0,
        }
    }
}

#[derive(Clone, Debug, Hash)]
pub struct MembraneChannel {
    /// A chanel in the membrane.
    pub channel: Channel,
    /// The peak conductance of the given channel (what its conductance
    /// would be if all activation and inactivation gates were open).
    pub siemens_per_square_cm: f32,
}

// TODO: Return MicroAmpsPerSquareCm.
impl MembraneChannel {
    pub fn channel_current_per_cm(
        &self,
        k_reversal: &MilliVolts,
        na_reversal: &MilliVolts,
        cl_reversal: &MilliVolts,
        ca_reversal: &MilliVolts,
        membrane_potential: &MilliVolts,
    ) -> f32 {
        let gating_coefficient = self.channel.conductance_coefficient();
        let k_current = self.channel.ion_selectivity.k
            * gating_coefficient
            * (membrane_potential.0 - k_reversal.0)
            * 0.001;
        let na_current = self.channel.ion_selectivity.na
            * gating_coefficient
            * (membrane_potential.0 - na_reversal.0)
            * 0.001;
        let ca_current = self.channel.ion_selectivity.ca
            * gating_coefficient
            * (membrane_potential.0 - ca_reversal.0)
            * 0.001;
        let cl_current = self.channel.ion_selectivity.cl
            * gating_coefficient
            * (membrane_potential.0 - cl_reversal.0)
            * 0.001;
        let channel_current =
            (k_current + na_current + ca_current + cl_current) * self.siemens_per_square_cm;
        channel_current
    }
}

/// A collection of segment PBR materials for Bevy rendering.
#[derive(Resource)]
pub struct MembraneMaterials {
    pub handles: Vec<Handle<StandardMaterial>>,
    pub voltage_range: (MilliVolts,MilliVolts),
    pub len: usize,
}

impl FromWorld for MembraneMaterials {
  fn from_world(world: &mut World) -> Self {
      let mut material_assets = world.get_resource_mut::<Assets<StandardMaterial>>().expect("Can get Assets");
      let len = 100;
      let voltage_range = (MilliVolts(-100.0), MilliVolts(100.0));
      let handles = (0..len).map(|i| {
          let intensity_range = 1.0;
          let intensity = i as f32 / len as f32 * intensity_range;
          let color = Color::rgb(intensity, 0.0, 1.0 - intensity);
          let mut material : StandardMaterial = color.clone().into();
          material.emissive = Color::rgb_linear(
              30.0 * intensity,
              30.0 * intensity * intensity,
              30.0 * intensity * intensity
          );
          material.metallic = intensity;
          let handle = material_assets.add(material);
          handle
      }).collect();
      MembraneMaterials { handles, voltage_range, len}
  }
}

impl MembraneMaterials {

    pub fn from_voltage(&self, v: &MilliVolts) -> Handle<StandardMaterial> {
        let v_min = self.voltage_range.0.0;
        let v_max = self.voltage_range.1.0;
        let index = (((v.0 - v_min) / (v_max - v_min)) * self.len as f32) as usize;
        self.handles[index.min(self.len - 1)].clone()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::BODY_TEMPERATURE;

    const K_REVERSAL: MilliVolts = MilliVolts(-89.0);
    const NA_REVERSAL: MilliVolts = MilliVolts(80.0);
    const CL_REVERSAL: MilliVolts = MilliVolts(-80.0);
    const CA_REVERSAL: MilliVolts = MilliVolts(90.0);

    #[test]
    fn example_reversal_potential() {
        let epsilon = 1e-9;

        let initial_membrane_potential = MilliVolts(0.0);
        let na_reversal = MilliVolts(80.0);
        let mut na_channel = crate::neuron::channel::common_channels::giant_squid::NA_CHANNEL
            .build(&initial_membrane_potential);
        na_channel
            .inactivation
            .iter_mut()
            .for_each(|gs| gs.magnitude = 1.0);
        let m = na_channel.activation.as_ref().unwrap().magnitude;
        assert!((m - 0.935).abs() < 1e-3);
        let na_example = MembraneChannel {
            channel: na_channel,
            siemens_per_square_cm: 120e-3,
        };
        let na_current = na_example.channel_current_per_cm(
            &K_REVERSAL,
            &NA_REVERSAL,
            &CL_REVERSAL,
            &CA_REVERSAL,
            &initial_membrane_potential,
        );
        let expected = -0.080 * 120e-3 * m.powi(3);
        assert!((na_current - expected).abs() < 1e-10);
    }

    #[test]
    fn k_current_at_equillibrium_is_zero() {
        let epsilon = 1e-9;
        let initial_membrane_potential = MilliVolts(-89.0);
        let k_reversal = MilliVolts(-89.0);
        let cl_reversal = MilliVolts(-90.0);
        let null_reversal = MilliVolts(0.0);
        let k_example = MembraneChannel {
            channel: crate::neuron::channel::common_channels::giant_squid::K_CHANNEL
                .build(&initial_membrane_potential),
            siemens_per_square_cm: 3e-3,
        };

        // K current when v_m == E(k) should be zero.
        assert!(
            k_example.channel_current_per_cm(
                &K_REVERSAL,
                &NA_REVERSAL,
                &CL_REVERSAL,
                &CA_REVERSAL,
                &K_REVERSAL
            ) < epsilon
        );
    }

    #[test]
    fn cl_current_example() {
        let epsilon = 1e-9;

        let initial_membrane_potential = MilliVolts(-79.0);
        let cl_channel = crate::neuron::channel::common_channels::giant_squid::LEAK_CHANNEL
            .build(&initial_membrane_potential);
        let cl_example = MembraneChannel {
            channel: cl_channel,
            siemens_per_square_cm: 0.3e-3,
        };
        let cl_current = cl_example.channel_current_per_cm(
            &K_REVERSAL,
            &NA_REVERSAL,
            &CL_REVERSAL,
            &CA_REVERSAL,
            &initial_membrane_potential,
        );
        dbg!(cl_current);
        let expected = 0.001 * 0.3e-3;
        dbg!(expected);
        assert!((cl_current - expected).abs() < 1e-6);
    }

    #[test]
    fn k_current_example() {
        let epsilon = 1e-9;

        // K current when v_m is 10mV depolarized from E(k) should be V/R.
        // V is 0.01
        // R is 1/(3e-3 * h), the reciprocal of the max conductance times K activation.
        let initial_membrane_potential = MilliVolts(-53.0);
        let k_channel = crate::neuron::channel::common_channels::giant_squid::K_CHANNEL
            .build(&initial_membrane_potential);
        let n = k_channel.activation.unwrap().magnitude;
        assert_eq!(n, 0.5);

        let k_example = MembraneChannel {
            channel: crate::neuron::channel::common_channels::giant_squid::K_CHANNEL
                .build(&initial_membrane_potential),
            siemens_per_square_cm: 3e-3,
        };

        let expected = (initial_membrane_potential.0 - K_REVERSAL.0) * 0.001 * 3e-3 * n.powi(4);
        dbg!(&expected);
        let k_current = k_example.channel_current_per_cm(
            &K_REVERSAL,
            &NA_REVERSAL,
            &CL_REVERSAL,
            &CA_REVERSAL,
            &initial_membrane_potential,
        );
        dbg!(&k_current);
        assert!((k_current - expected).abs() < epsilon);
    }
}
