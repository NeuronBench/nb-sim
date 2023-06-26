use bevy::prelude::*;
use bevy::core_pipeline::bloom::BloomSettings;
use bevy::diagnostic::{LogDiagnosticsPlugin, FrameTimeDiagnosticsPlugin};
use bevy::pbr::CascadeShadowConfigBuilder;
use bevy_egui::{egui, EguiPlugin};
use bevy_mod_picking::prelude::*;
// use bevy_mod_picking::{
//     // DebugCursorPickingPlugin, DebugEventsPickingPlugin, DefaultPickingPlugins,
//     PickableBundle,
//     PickingCameraBundle
// };
use std::f32::consts::PI;

use reuron::plugin::ReuronPlugin;
use reuron::gui::run_gui;
use reuron::gui::load::{handle_loaded_neuron, GraceSceneSource};
use reuron::integrations::swc_file::SwcFile;
use reuron::integrations::grace::{self, GraceScene};
use reuron::neuron::segment::ecs::Segment;
use reuron::neuron::membrane::MembraneMaterials;
use reuron::pan_orbit_camera::{PanOrbitCamera, pan_orbit_camera};

#[derive(Component)]
struct MyCamera;

pub fn main() {
 let mut app = App::new();
 app
    .add_plugins(DefaultPlugins.set(WindowPlugin {
      primary_window: Some(Window {
        fit_canvas_to_parent: true,
        canvas: Some("#bevy".to_string()),
        ..default()
      }),
      ..default()
    }))
        .add_plugin(EguiPlugin)
        .add_plugin(LogDiagnosticsPlugin::default())
        .add_plugin(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(DefaultPickingPlugins.build().disable::<DebugPickingPlugin>())
        // .add_plugin(DebugCursorPickingPlugin)
        // .add_plugin(DebugEventsPickingPlugin)
        .add_plugin(ReuronPlugin)
        .add_system(bevy::window::close_on_esc)
        .add_startup_system(setup_scene)
        // .add_startup_system(setup_swc_neuron)
        .add_startup_system(setup_grace_neuron)
        .insert_resource(ClearColor(Color::rgb(0.2,0.2,0.2)))
        .add_system(pan_orbit_camera)
        .add_system(run_gui)
        .add_system(handle_loaded_neuron);

        app.run();
}

fn setup_swc_neuron(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    segments_query: Query<(&Segment, &GlobalTransform)>,
    mut materials: Res<MembraneMaterials>,
) {
  // let swc_neuron_1 = SwcFile::read_file("/Users/greghale/Downloads/H17.03.010.11.13.06_651089035_m.swc").expect("should parse");
  let swc_neuron_2 = SwcFile::read_str(reuron::integrations::swc_file::sample::neuron()).expect("should parse");

  let location_cm = Vec3::new(0.0, 0.0, 0.0);
  // let soma_entity = swc_neuron_1.clone().simplify().spawn(location_cm, &mut commands, &mut meshes, &mut materials);
  let soma_entity = swc_neuron_2.clone().simplify().spawn(location_cm, &mut commands, &mut meshes, &mut materials);

  // let location_cm = Vec3::new(500.0, 0.0, 0.0);
  // let soma_entity = swc_neuron_2.clone().simplify().spawn(location_cm, &mut commands, &mut meshes, &mut materials);

  // let location_cm = Vec3::new(500.0, 800.0, 0.0);
  // let soma_entity = swc_neuron_1.simplify().spawn(location_cm, &mut commands, &mut meshes, &mut materials);

  // for i in 0..0 {
  //   let location_cm = Vec3::new(500.0, 200.0, -2000.0 + 300.0 * i as f32);
  //   let soma_entity = swc_neuron_2.clone().simplify().spawn(location_cm, &mut commands, &mut meshes, &mut materials);
  // }

  // let soma_entity = SwcFile::sample().spawn(commands, meshes, materials);
  // let soma_transform = segments_query.get_component::<GlobalTransform>(soma_entity).expect("soma exists");
  // println!("Soma translation: {:?}", soma_transform.translation());
  // let (_, mut camera_transform) = camera_query.get_single().expect("just one camera");
  // camera_transform = &camera_transform.looking_at(soma_transform.translation(), Vec3::Y);
}

fn setup_grace_neuron(
  mut commands: Commands,
  mut meshes: ResMut<Assets<Mesh>>,
  mut membrane_materials: Res<MembraneMaterials>,
  mut materials: ResMut<Assets<StandardMaterial>>,
  grace_scene_source: Res<GraceSceneSource>
) {

  if grace_scene_source.0.len == 0 {
    let grace_scene = GraceScene ( grace::sample::scene() );
    grace_scene.spawn(Vec3::new(0.0,0.0,0.0), commands, &mut meshes, membrane_materials, &mut materials);
  }

}


fn setup_scene(
    mut commands: Commands,
) {
    commands.insert_resource(AmbientLight {
        color: Color::rgb(1.0,1.0,1.0),
        brightness: 0.01,
    });
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            shadows_enabled: true,
            ..default()
        },
        transform: Transform {
            translation: Vec3::new(0.0,2.0, 0.0),
            rotation: Quat::from_rotation_x(-PI/ 4.0),
            ..default()
        },
        cascade_shadow_config: CascadeShadowConfigBuilder {
            first_cascade_far_bound: 4.0,
            maximum_distance: 10.0,
            ..default()
        }.into(),
        ..default()
    });

    commands.spawn(
        (Camera3dBundle {
            transform: Transform::from_xyz(-200.0,200.5, 1000.0).looking_at(Vec3::ZERO, Vec3::Y),
            camera: Camera {
                hdr: true,
                ..default()
            },
            ..default()
            },
         MyCamera,

         BloomSettings::default(),
         RaycastPickCamera::default(),

         PanOrbitCamera {
             radius: 500.0,
             ..default()
         }
        ));
}
