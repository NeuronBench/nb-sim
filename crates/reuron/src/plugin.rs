use bevy::prelude::*;
use std::iter::zip;
use std::fmt::{self, Display};
use std::time::Duration;

use crate::dimension::{MicroAmpsPerSquareCm, FaradsPerSquareCm, MilliVolts, Diameter, Interval, Kelvin, Timestamp};
use crate::constants::{BODY_TEMPERATURE, CONDUCTANCE_PER_SQUARE_CM};
use crate::neuron::segment::Geometry;
use crate::neuron::solution::{Solution, INTERSTICIAL_FLUID, EXAMPLE_CYTOPLASM};
use crate::neuron::membrane;
use crate::neuron::channel::{self, ca_reversal, cl_reversal, k_reversal, na_reversal};

pub struct ReuronPlugin;

impl Plugin for ReuronPlugin {
    fn build(&self, app: &mut App) {
        app
            .insert_resource(default_env())
            .insert_resource(Timestamp(0.0))
            .init_resource::<MembraneMaterials>()
            .insert_resource(StdoutRenderTimer {
                timer: Timer::new(Duration::from_millis(100), TimerMode::Repeating)
            })
            .insert_resource(SystemCounts::zero())
            .add_startup_system(create_example_neuron)
            .add_system(update_timestamp)
            .add_system(apply_channel_currents)
            .add_system(update_membrane_conductances)
            .add_system(apply_input_currents)
            .add_system(apply_junction_currents)
            .add_system(apply_voltage_to_materials)
            .add_system(print_voltages);
    }
}

#[derive(Resource)]
pub struct StdoutRenderTimer {
    timer: Timer,
}

#[derive(Component)]
pub struct Segment;


#[derive(Debug, Resource)]
pub struct SystemCounts {
    n_membrane_conductances: u64,
    n_channel_currents: u64,
    n_input_currents: u64,
    n_print: u64,
}

impl SystemCounts {
    pub fn zero() -> SystemCounts {
        SystemCounts { n_membrane_conductances: 0, n_channel_currents: 0, n_input_currents: 0, n_print:0 }
    }
}

#[derive(Bundle)]
pub struct SegmentBundle {
    pub intracellular_solution: Solution,
    pub membrane_voltage: MembraneVoltage,
    pub geometry: Geometry,

    #[bundle]
    pub pbr: PbrBundle,
}

#[derive(Component)]
pub struct Junction {
    first_segment: Entity,
    second_segment: Entity,
    pore_diameter: Diameter,
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
        interval: Interval(20e-6)
    }
}

fn update_timestamp(env: Res<Env>, mut timestamp: ResMut<Timestamp>) {
  timestamp.0 = timestamp.0 + env.interval.0;
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
        geometry: Geometry { diameter: Diameter(1.0), length: 1.0 },
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
    let segments : Vec<Entity> =
        (0..10).map(|col| {
            let col_segments = (0..40)
                .map(|i| {
                    let segment = commands.spawn(
                        (Segment
                            , mk_segment(col, i)
                            , Membrane(membrane.clone())
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
    env: Res<Env>,
    mut counts: ResMut<SystemCounts>
) {
    counts.n_membrane_conductances += 1;
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
    env: Res<Env>,
    mut counts: ResMut<SystemCounts>
) {
    counts.n_channel_currents += 1;
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

        membrane_voltage.0.0 += 1000.0 * dv_dt * env.interval.0;
    }
}

fn apply_input_currents(
    mut query: Query<(&Segment, &Geometry, &Membrane, &InputCurrent, &mut MembraneVoltage)>,
    env : Res<Env>,
    mut counts: ResMut<SystemCounts>

){
    counts.n_input_currents += 1;
    for (_, geometry, membrane, input_current, mut membrane_voltage) in &mut query {
        let surface_area =
            geometry.diameter.0 * std::f32::consts::PI * geometry.length;
        let capacitance = membrane.0.capacitance.0 * surface_area;
        let current = input_current.0.0 * 1e-6 * surface_area;
        let dv_dt = current / capacitance;
        membrane_voltage.0.0 += 1000.0 * dv_dt * env.interval.0;
    }
}

fn apply_junction_currents(
    commands: Commands,
    mut junctions_query: Query<&Junction>,
    mut segments_query: Query<(&Segment, &Geometry, &Membrane, &mut MembraneVoltage)>,
    env: Res<Env>,
) {
    let interval_seconds = env.interval.0;
    for Junction {first_segment, second_segment, pore_diameter} in &mut junctions_query {
        let results = segments_query.get_many_mut([first_segment.clone(), second_segment.clone()]);
        match results {
            Ok([(_,geom1,membrane1, mut vm1), (_,geom2, membrane2, mut vm2)]) => {
                let capacitance1 = membrane1.0.capacitance.0 * geom1.surface_area();
                let capacitance2 = membrane2.0.capacitance.0 * geom2.surface_area();
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

fn print_voltages(
    timestamp: Res<Timestamp>,
    mut stdout_render_timer: ResMut<StdoutRenderTimer>,
    query: Query<&MembraneVoltage>,
    time: Res<Time>,
    mut counts: ResMut<SystemCounts>
) {
    counts.n_print += 1;
    stdout_render_timer.timer.tick(time.delta());

    // let fps = counts.n_print as f64 / time.elapsed_seconds_f64();
    if stdout_render_timer.timer.just_finished() {
        // println!("{:.6} Counts: {counts:?}. FPS: {}", timestamp.0, fps);
        if let Some(membrane_voltage) = &query.iter().next() {
            println!("SimulationTime: {} ms. First Voltage: {membrane_voltage}", counts.n_print / 100 );
        }
        println!("");
    }
}


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
          let color = Color::rgb(intensity, intensity, intensity);
          let handle = material_assets.add(color.into());
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
        self.handles[index].clone()
    }
}
