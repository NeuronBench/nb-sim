use bevy::prelude::*;
use std::iter::zip;
use std::fmt::{self, Display};

use crate::dimension::{MicroAmpsPerSquareCm, FaradsPerSquareCm, MilliVolts, Diameter, Interval, Kelvin};
use crate::constants::{BODY_TEMPERATURE};
use crate::neuron::segment::Geometry;
use crate::neuron::solution::{Solution, INTERSTICIAL_FLUID, EXAMPLE_CYTOPLASM};
use crate::neuron::membrane;
use crate::neuron::channel::{self, ca_reversal, cl_reversal, k_reversal, na_reversal};

pub struct ReuronPlugin;

impl Plugin for ReuronPlugin {
    fn build(&self, app: &mut App) {
        app
            .insert_resource(default_env())
            .add_system(create_example_neuron)
            .add_system(update_membrane_conductances)
            .add_system(apply_channel_currents)
            .add_system(apply_external_currents)
            .add_system(print_voltages);
    }
}

#[derive(Component)]
pub struct Segment;

#[derive(Bundle)]
pub struct SegmentBundle {
    pub intracellular_solution: Solution,
    pub membrane_voltage: MembraneVoltage,
    pub geometry: Geometry,
    pub input_current: InputCurrent,
}

#[derive(Component)]
pub struct Junction {
    first_segment: Entity,
    second_segment: Entity,
}

#[derive(Component)]
pub struct InputCurrent(MicroAmpsPerSquareCm);

#[derive(Component)]
pub struct MembraneVoltage(MilliVolts);

impl Display for MembraneVoltage {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
      write!(f, "{} mV", self.0.0)
  }
}

#[derive(Component)]
pub struct Neuron;

#[derive(Component)]
pub struct Membrane(membrane::Membrane);

#[derive(Resource)]
pub struct Env {
    pub temperature: Kelvin,
    pub extracellular_solution: Solution,
    pub interval: Interval
}

fn default_env() -> Env {
    Env {
        temperature: BODY_TEMPERATURE,
        extracellular_solution: INTERSTICIAL_FLUID,
        interval: Interval(0.000001)
    }
}

fn create_example_neuron(mut commands: Commands) {
    let v0 = MilliVolts(-70.0);
    let mk_segment = || SegmentBundle {
        intracellular_solution: EXAMPLE_CYTOPLASM,
        membrane_voltage: MembraneVoltage(MilliVolts(-70.0)),
        geometry: Geometry { diameter: Diameter(0.01), length: 1000.0 },
        input_current: InputCurrent(MicroAmpsPerSquareCm(0.0)),
    };
    let membrane = membrane::Membrane {
        capacitance: FaradsPerSquareCm(1e-6),
        membrane_channels: vec![
            membrane::MembraneChannel {
                channel: channel::common_channels::giant_squid::LEAK_CHANNEL
                    .build(&v0),
                siemens_per_square_cm: 36e-3,
            },
            membrane::MembraneChannel {
                channel: channel::common_channels::giant_squid::K_CHANNEL
                    .build(&v0),
                siemens_per_square_cm: 36e-3,
            },
            membrane::MembraneChannel {
                channel: channel::common_channels::giant_squid::NA_CHANNEL
                    .build(&v0),
                siemens_per_square_cm: 120e-3,
            },
        ]
    };
    let segments : Vec<Entity> =
        (0..4)
        .map(|_| {
            let segment = commands.spawn((mk_segment(), Membrane(membrane.clone()))).id();
            segment
        })
        .collect();
    zip(segments.clone(), segments[1..].iter())
        .into_iter()
        .for_each(|(x,y)| {
            commands.spawn(Junction{
                first_segment: x,
                second_segment: y.clone()
            });
        });
}

fn update_membrane_conductances(mut query: Query<(&Segment, &MembraneVoltage, &mut Membrane)>, env: Res<Env>) {

    for (_, membrane_voltage, mut membrane) in &mut query {
        membrane
            .0
            .membrane_channels
            .iter_mut()
            .for_each(|mut membrane_channel| {
              membrane_channel.channel.step(&membrane_voltage.0, &env.interval)
            })
    }
}

fn apply_channel_currents(
    mut query: Query<(
        &Segment,
        &Solution,
        &Geometry,
        &Membrane,
        &mut MembraneVoltage
    )>,
    env: Res<Env>) {
    for (_, solution, geometry, membrane, mut membrane_voltage) in &mut query {
        let surface_area =
            geometry.diameter.0 * std::f32::consts::PI * geometry.length;
        let current = -1.0 * membrane.0.current_per_square_cm(
                &k_reversal(
                    &solution,
                    &env.extracellular_solution,
                    &env.temperature,
                ),
                &na_reversal(
                    &solution,
                    &env.extracellular_solution,
                    &env.temperature,
                ),
                &cl_reversal(
                    &solution,
                    &env.extracellular_solution,
                    &env.temperature,
                ),
                &ca_reversal(
                    &solution,
                    &env.extracellular_solution,
                    &env.temperature,
                ),
                &membrane_voltage.0,
        ) * surface_area;
        let capacitance = membrane.0.capacitance.0 * surface_area;
        let dv_dt : f32 = current / capacitance;

        membrane_voltage.0.0 += (1000.0 * dv_dt * env.interval.0);
    }
}

fn apply_external_currents(
    mut query: Query<(&Segment, &Geometry, &Membrane, &InputCurrent, &mut MembraneVoltage)>,
    env : Res<Env>
){
    for (_, geometry, membrane, input_current, mut membrane_voltage) in &mut query {
        let surface_area =
            geometry.diameter.0 * std::f32::consts::PI * geometry.length;
        let capacitance = membrane.0.capacitance.0 * surface_area;
        let dv_dt = input_current.0.0 * surface_area / capacitance;
        membrane_voltage.0.0 += (1000.0 * dv_dt * env.interval.0);
    }
}


fn print_voltages(
    query: Query<&MembraneVoltage>
) {
    for (membrane_voltage) in &query {
        println!("Voltage: {membrane_voltage}")
    }
}
