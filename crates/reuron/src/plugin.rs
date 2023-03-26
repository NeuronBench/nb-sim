use bevy::prelude::*;
use bevy_mod_picking::{PickableBundle, PickingEvent};
use std::iter::zip;
use std::fmt::{self, Display};
use std::time::Duration;

use crate::dimension::{
    MicroAmpsPerSquareCm,
    FaradsPerSquareCm,
    MilliVolts,
    Diameter,
    Interval,
    Kelvin,
    Timestamp,
    SimulationStepSeconds
};
use crate::constants::{BODY_TEMPERATURE, CONDUCTANCE_PER_SQUARE_CM, SIMULATION_STEPS_PER_FRAME};
use crate::stimulator::{StimulatorMaterials, Stimulator};
use crate::serialize;
use crate::neuron::Junction;
use crate::neuron::segment::{Geometry, ecs::Segment, ecs::InputCurrent};
use crate::neuron::solution::{Solution, INTERSTICIAL_FLUID, EXAMPLE_CYTOPLASM};
use crate::neuron::membrane::{self, Membrane, MembraneMaterials, MembraneVoltage};
use crate::neuron::channel::{self, ca_reversal, cl_reversal, k_reversal, na_reversal};

pub struct ReuronPlugin;

impl Plugin for ReuronPlugin {
    fn build(&self, mut app: &mut App) {
            app.insert_resource(default_env())
            .insert_resource(Timestamp(0.0))
            .insert_resource(Stimulator::default())
            .insert_resource(SimulationStepSeconds(1e-8))
            .init_resource::<MembraneMaterials>()
            .init_resource::<StimulatorMaterials>()
            .insert_resource(StdoutRenderTimer {
                timer: Timer::new(Duration::from_millis(100), TimerMode::Repeating)
            })
            // .add_startup_system(create_example_neuron)
            // .add_system(update_timestamp)
            .add_system(stimulate_picked_segments);

            // Because the Bevy frame rate is limited by winit to about 300,
            // if we want to take more than 300 biophysics steps per second,
            // (at 10us steps, this would be 1/333 of realtime), we have to
            // apply the biophysics system multiple times per bevy frame.
            // These 40 repetitions bring us up to nearly 1/10th realtime.
            // TODO, find out how to pass a query to a for loop.
            for _ in 0..SIMULATION_STEPS_PER_FRAME {
              app.add_system(step_biophysics);
            }

            app
            .add_system(apply_voltage_to_materials)
            .add_system(apply_current_to_stimulator_material)

            .add_system(print_voltages);
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
  junctions_query: Query<&Junction>
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
        let mut current_microamps = input_current + stimulator_current;
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
            Ok(_) => panic!("wrong number of results"),
            Err(e) => panic!("Other error {e}"),

        }
    }

    // ***************************************
    // ***** Advance stimulation time. *******
    // ***************************************
    timestamp.0 += simulation_step.0;

}

#[derive(Bundle)]
pub struct SegmentBundle {
    pub intracellular_solution: Solution,
    pub membrane_voltage: MembraneVoltage,
    pub geometry: Geometry,

    #[bundle]
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

fn update_timestamp(
    simulation_step: Res<SimulationStepSeconds>,
    mut timestamp: ResMut<Timestamp>
) {
  timestamp.0 = timestamp.0 + simulation_step.0;
}

fn create_example_neuron(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    materials: Res<MembraneMaterials>,
) {
    let v0 = MilliVolts(-70.0);
    let mut mk_segment = |col:u32, i: u32| SegmentBundle {
        intracellular_solution: EXAMPLE_CYTOPLASM,
        membrane_voltage: MembraneVoltage(v0.clone()),
        geometry: Geometry::Cylinder { diameter: Diameter(1.0), length: 1.0 },
        pbr: PbrBundle {
            mesh: meshes.add(shape::Cylinder {
                radius: 0.5,
                height: 0.95,
                resolution: 12,
                segments:2,
            }.into()),
            material: materials.from_voltage(&MilliVolts(-80.0)),
            transform: Transform::from_xyz(5.0 * col as f32, 1.0 * i as f32, 0.0),
            ..default()
            }
    };
    let membrane = membrane::Membrane {
        capacitance: FaradsPerSquareCm(1e-6),
        membrane_channels: vec![
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
            membrane::MembraneChannel {
                channel: channel::common_channels::giant_squid::LEAK_CHANNEL
                    .build(&v0),
                siemens_per_square_cm: 0.3e-3,
            },
        ]
    };
    let _segments : Vec<Entity> =
        (0..10).map(|col| {
            let col_segments = (0..40)
                .map(|i| {
                    let segment = commands.spawn(
                        (Segment
                            , mk_segment(col, i)
                            , membrane.clone()
                        )).id();
                    let input_current = if i == 0 {30.0 * col as f32 as f32} else {-1.0};
                    commands
                        .entity(segment)
                        .insert(InputCurrent(MicroAmpsPerSquareCm(input_current)));
                    segment
                }).collect::<Vec<_>>();

            zip(col_segments.clone(), col_segments[1..].iter())
                .into_iter()
                .for_each(|(x,y)| {
                    commands.spawn(Junction{
                        first_segment: x,
                        second_segment: y.clone(),
                        pore_diameter: Diameter(1.0),
                    });
                });
            col_segments
        })
        .flatten()
        .collect();
}

fn update_membrane_conductances(
    mut query: Query<(&Segment, &MembraneVoltage, &mut Membrane)>,
    simulation_step: Res<SimulationStepSeconds>,
) {
    for (_, membrane_voltage, mut membrane) in &mut query {
        membrane
            .membrane_channels
            .iter_mut()
            .for_each(|membrane_channel| {
              membrane_channel.channel.step(&membrane_voltage.0, &Interval(simulation_step.0))
            })
    }
}

fn apply_channel_currents(
    mut timestamp: ResMut<Timestamp>,
    simulation_step: Res<SimulationStepSeconds>,
    mut query: Query<(
        &Segment,
        &Solution,
        &Geometry,
        &Membrane,
        &mut MembraneVoltage
    )>,
    env: Res<Env>,
) {
    for (_, solution, geometry, membrane, mut membrane_voltage) in &mut query {
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
    }
}

fn apply_input_currents(
    mut query: Query<(&Segment, &Geometry, &membrane::Membrane, &InputCurrent, &mut MembraneVoltage)>,
    simulation_step : Res<SimulationStepSeconds>,
){
    for (_, geometry, membrane, input_current, mut membrane_voltage) in &mut query {
        let surface_area = geometry.surface_area();
        let capacitance = membrane.capacitance.0 * surface_area;
        let current = input_current.0.0 * 1e-6 * surface_area;
        let dv_dt = current / capacitance;
        membrane_voltage.0.0 += 1000.0 * dv_dt * simulation_step.0;
    }
}

fn apply_junction_currents(
    mut junctions_query: Query<&Junction>,
    mut segments_query: Query<(&Segment, &Geometry, &Membrane, &mut MembraneVoltage)>,
    simulation_step: Res<SimulationStepSeconds>,
) {
    let interval_seconds = simulation_step.0;
    for Junction {first_segment, second_segment, pore_diameter} in &mut junctions_query {
        let results = segments_query.get_many_mut([first_segment.clone(), second_segment.clone()]);
        match results {
            Ok([(_,geom1,membrane1, mut vm1), (_,geom2, membrane2, mut vm2)]) => {
                let capacitance1 = membrane1.capacitance.0 * geom1.surface_area();
                let capacitance2 = membrane2.capacitance.0 * geom2.surface_area();

                let mutual_conductance = pore_diameter.0 * std::f32::consts::PI * CONDUCTANCE_PER_SQUARE_CM;
                let first_to_second_current = mutual_conductance * (vm1.0.0 - vm2.0.0) * 1e-3;

                vm1.0.0 -= first_to_second_current / capacitance1 * interval_seconds;
                vm2.0.0 += first_to_second_current / capacitance2 * interval_seconds;
            },
            Ok(_) => panic!("wrong number of results"),
            Err(e) => panic!("Other error {e}"),

        }
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

#[derive(Component)]
pub struct Stimulation { stimulation_segment: Entity }

fn stimulate_picked_segments(
    mut commands: Commands,
    simulation_step: Res<SimulationStepSeconds>,
    new_stimulators: Res<Stimulator>,
    mut events: EventReader<PickingEvent>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut segments_query: Query<(&Segment, &GlobalTransform)>,
    stimulations_query: Query<&Stimulation>,
) {
    for event in events.iter() {
        match event {
            PickingEvent::Selection(e) => {},
            PickingEvent::Hover(e) => {},
            PickingEvent::Clicked(e) => {

                match segments_query.get(e.clone()) {
                    Ok((_, segment_transform)) => {
                        println!("Adding current");
                        commands.spawn(
                            (Stimulation { stimulation_segment: e.clone() },
                             PbrBundle {
                                mesh: meshes.add(shape::UVSphere{
                                    radius: 7.5,
                                    sectors: 20,
                                    stacks: 20
                                }.into()),
                                material: materials.add(Color::rgb(0.5,0.5,0.5).into()),
                                transform: Transform::from_translation(segment_transform.translation()),
                                ..default()
                              },
                             PickableBundle::default(),
                            ));
                        commands.entity(*e).insert(new_stimulators.clone());
                    },
                    Err(_) => {}
                }
                match stimulations_query.get(e.clone()) {
                    Ok(Stimulation{ stimulation_segment, .. }) => {
                        match segments_query.get( stimulation_segment.clone() ) {
                            Ok(segment) => {
                                commands.entity(stimulation_segment.clone()).remove::<Stimulator>();
                                commands.entity(e.clone()).despawn();
                            }
                            Err(_) => {}
                        }
                    }
                    Err(_) => {}
                }
            }
        }
    }
}


pub fn serialize_simulation (
    extracellular_solution: &Solution,
    segments: &[(Membrane, MembraneVoltage, Stimulator)]
) -> serialize::Scene {
    serialize::Scene {
        extracellular_solution: extracellular_solution.serialize(),
        membranes: unimplemented!(),
        neurons: unimplemented!(),
        synapses: vec![],
    }
}
