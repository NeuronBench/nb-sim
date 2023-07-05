use bevy::prelude::*;
use std::fmt::{self, Display};
use std::time::Duration;

use crate::gui;

use crate::dimension::{
    Interval,
    Kelvin,
    Timestamp,
    SimulationStepSeconds
};
use crate::constants::{BODY_TEMPERATURE, CONDUCTANCE_PER_SQUARE_CM, SIMULATION_STEPS_PER_FRAME};
use crate::stimulator::{StimulatorMaterials, Stimulator, Stimulation};

use crate::neuron::Junction;
use crate::integrations::grace::Synapse;
use crate::neuron::segment::{Geometry, ecs::Segment, ecs::InputCurrent};
use crate::neuron::solution::{Solution, INTERSTICIAL_FLUID};
use crate::neuron::membrane::{Membrane, MembraneMaterials, MembraneVoltage};
use crate::neuron::channel::{ca_reversal, cl_reversal, k_reversal, na_reversal};

pub struct ReuronPlugin;

impl Plugin for ReuronPlugin {
    fn build(&self, app: &mut App) {
            app.insert_resource(default_env())
            .insert_resource(Timestamp(0.0))
            .insert_resource(Stimulator::default())
            .insert_resource(SimulationStepSeconds(5e-7))
            .init_resource::<MembraneMaterials>()
            .init_resource::<StimulatorMaterials>()
            .insert_resource(StdoutRenderTimer {
                timer: Timer::new(Duration::from_millis(2000), TimerMode::Repeating)
            });

            // Because the Bevy frame rate is limited by winit to about 300,
            // if we want to take more than 300 biophysics steps per second,
            // (at 10us steps, this would be 1/333 of realtime), we have to
            // apply the biophysics system multiple times per bevy frame.
            // These 40 repetitions bring us up to nearly 1/10th realtime.
            // TODO, find out how to pass a query to a for loop.
            for _ in 0..SIMULATION_STEPS_PER_FRAME {
              app.add_systems(Update, step_biophysics);
            }

            app
            .add_systems(Update, apply_voltage_to_materials)
            .add_systems(Update, apply_current_to_stimulator_material)

            .add_systems(Update, print_voltages);
            gui::load::setup(app);
    }
}

#[derive(Resource)]
pub struct StdoutRenderTimer {
    timer: Timer,
}



fn step_biophysics(
  env: Res<Env>,
  simulation_step: Res<SimulationStepSeconds>,
  mut timestamp: ResMut<Timestamp>,
  mut segments_query: Query<
          (&Segment,
           &Solution,
           &Geometry,
           &mut Membrane,
           &mut MembraneVoltage,
           Option<&InputCurrent>,
           Option<&Stimulator>
          )>,
  junctions_query: Query<&Junction>,
  mut synapses_query: Query<&mut Synapse>
){
    for (_,
         solution,
         geometry,
         mut membrane,
         mut membrane_voltage,
         maybe_input_current,
         maybe_stimulator
        ) in &mut segments_query {

        // ***********************************
        // ***** Apply channel currents. *****
        // ***********************************
        let surface_area = geometry.surface_area();

        let current = -1.0 * membrane.current_per_square_cm(
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
        let capacitance = membrane.capacitance.0 * surface_area;
        let dv_dt : f32 = current / capacitance;

        membrane_voltage.0.0 += 1000.0 * dv_dt * simulation_step.0;

        // ***********************************
        // ***** Update membrane conductances.
        // ***********************************
        membrane
            .membrane_channels
            .iter_mut()
            .for_each(|membrane_channel| {
            membrane_channel.channel.step(&membrane_voltage.0, &Interval(simulation_step.0))
            });

        // ***************************************************
        // ***** Apply input currents and stimulators. *******
        // ***************************************************
        let input_current = maybe_input_current.map_or(0.0, |i| i.0.0);
        let stimulator_current = maybe_stimulator.map_or(0.0, |stimulator|
                                    stimulator.current(timestamp.clone()
                                    ).0);
        let current_microamps = input_current + stimulator_current;
        let capacitance = membrane.capacitance.0 * surface_area;
        let current = current_microamps * 1e-6 * surface_area;
        let dv_dt = current / capacitance;
        membrane_voltage.0.0 += 1000.0 * dv_dt * simulation_step.0;


    }

    for Junction {first_segment, second_segment, pore_diameter} in &junctions_query {
        let interval_seconds = simulation_step.0;

        let results = segments_query.get_many_mut([first_segment.clone(), second_segment.clone()]);
        match results {
            Ok([(_,_,geom1,membrane1, mut vm1,_,_), (_,_,geom2, membrane2, mut vm2,_,_)]) => {
                let capacitance1 = membrane1.capacitance.0 * geom1.surface_area();
                let capacitance2 = membrane2.capacitance.0 * geom2.surface_area();

                let mutual_conductance = pore_diameter.0 * std::f32::consts::PI * CONDUCTANCE_PER_SQUARE_CM;
                let first_to_second_current = mutual_conductance * (vm1.0.0 - vm2.0.0) * 1e-3;

                vm1.0.0 -= first_to_second_current / capacitance1 * interval_seconds;
                vm2.0.0 += first_to_second_current / capacitance2 * interval_seconds;
            },
            Err(e) => panic!("Other error {e}"),

        }
    }

    for mut synapse in &mut synapses_query {
        // TODO: This fails if the source and target of the synapse are the same Entity.
        let interval_seconds = simulation_step.0;
        let results = segments_query.get_many_mut([synapse.pre_segment.clone(), synapse.post_segment.clone()]);
        match results {
            Ok([(_,_,_,_,vm1,_,_), (_,solution,_,_,mut vm2,_,_)]) => {
                synapse.synapse_membranes.step(
                    &BODY_TEMPERATURE,
                    &vm1.0,
                    &vm2.0,
                    &Interval(interval_seconds)
                );
                synapse.synapse_membranes.apply_current(
                    &Interval(interval_seconds),
                    &BODY_TEMPERATURE,
                    &mut vm2.0,
                    &solution
                );
            }
            Err(e) => {
                eprintln!("Synapse query error: {e}");
            }
        }
    }

    // ***************************************
    // ***** Advance simulation time. *******
    // ***************************************
    timestamp.0 += simulation_step.0;

}

#[derive(Bundle)]
pub struct SegmentBundle {
    pub intracellular_solution: Solution,
    pub membrane_voltage: MembraneVoltage,
    pub geometry: Geometry,

    // #[bundle]
    pub pbr: PbrBundle,
}

impl Display for MembraneVoltage {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
      write!(f, "{} mV", self.0.0)
  }
}

#[derive(Component)]
pub struct Neuron;

#[derive(Resource)]
pub struct Env {
    pub temperature: Kelvin,
    pub extracellular_solution: Solution,
}

fn default_env() -> Env {
    Env {
        temperature: BODY_TEMPERATURE,
        extracellular_solution: INTERSTICIAL_FLUID,
    }
}

fn apply_voltage_to_materials(
    membrane_materials: Res<MembraneMaterials>,
    mut query: Query<(&MembraneVoltage, &mut Handle<StandardMaterial>)>
) {
    for (v, mut material) in &mut query {
        *material = membrane_materials.from_voltage(&v.0);
    }
}

fn apply_current_to_stimulator_material(
    stimulator_materials: Res<StimulatorMaterials>,
    segments_query: Query<(&Segment, &Stimulator)>,
    timestamp: Res<Timestamp>,
    mut stimulations_query: Query<(&Stimulation, &mut Handle<StandardMaterial>)>
) {
    for (Stimulation { stimulation_segment }, mut material) in &mut stimulations_query {
        if let Ok(stimulator) = segments_query.get_component::<Stimulator>(*stimulation_segment) {
            let current = stimulator.current(Timestamp(timestamp.0));
            *material = stimulator_materials.from_selected_and_current(false, &current);
        } else {
            println!("Error, stimulation's segment not found.");
        }
    }
}

fn print_voltages(
    timestamp: Res<Timestamp>,
    mut stdout_render_timer: ResMut<StdoutRenderTimer>,
    query: Query<&MembraneVoltage>,
    time: Res<Time>,
) {
    stdout_render_timer.timer.tick(time.delta());

    if stdout_render_timer.timer.just_finished() {
        if let Some(membrane_voltage) = &query.iter().next() {
            println!("SimulationTime: {} ms. First Voltage: {membrane_voltage}", timestamp.0 * 1000.0);
        }
        if let Some(membrane_voltage) = &query.iter().next() {
            println!("SimulationTime: {} ms. Second Voltage: {membrane_voltage}", timestamp.0 );
        }
        if let Some(membrane_voltage) = &query.iter().next() {
            println!("SimulationTime: {} ms. Third Voltage: {membrane_voltage}", timestamp.0 );
        }
        println!("");
    }
}



// pub fn serialize_simulation (
//     extracellular_solution: &Solution,
//     segments: &[(Membrane, MembraneVoltage, Stimulator)]
// ) -> serialize::Scene {
//     serialize::Scene {
//         extracellular_solution: extracellular_solution.serialize(),
//         membranes: unimplemented!(),
//         neurons: unimplemented!(),
//         synapses: vec![],
//     }
// }
