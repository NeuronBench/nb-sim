use bevy::prelude::*;
use bevy::core_pipeline::bloom::BloomSettings;
use bevy::diagnostic::{LogDiagnosticsPlugin, FrameTimeDiagnosticsPlugin};
use bevy::pbr::CascadeShadowConfigBuilder;
use bevy_egui::{EguiPlugin, EguiContext};
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
use reuron::integrations::grace::{self, GraceScene};
use reuron::neuron::membrane::MembraneMaterials;
use reuron::pan_orbit_camera::{PanOrbitCamera, pan_orbit_camera};
use reuron::selection::{Selection, Highlight};

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
        .add_systems(Update, bevy::window::close_on_esc)
        .add_systems(Startup, setup_scene)
        // .add_startup_system(setup_swc_neuron)
        .add_systems(Startup, setup_grace_neuron)
        .insert_resource(ClearColor(Color::hex("#0e0e1f").expect("valid hex")))
        .add_systems(Update, pan_orbit_camera.run_if(pan_orbit_condition))
        .add_systems(Update, run_gui)
        .add_systems(Update, handle_loaded_neuron);

        app.run();
}

fn pan_orbit_condition(query: Query<&EguiContext>) -> bool {
  // TODO: Is it safe and fast to clone the egui context?
  // It seemed to be necessary because we are supposed to
  // get the context via `get_mut`, however passing a mutable
  // query to `pan_orbit_condition` fails because
  // "the trait `ReadOnlyWorldQuery` is not implemented for `&mut EguiContext`"
  let mut bevy_egui_context =
    query.get_single().expect("egui should exist").clone();
  let egui_context = bevy_egui_context.get_mut();
  !egui_context.wants_pointer_input()
}

fn setup_grace_neuron(
  commands: Commands,
  mut meshes: ResMut<Assets<Mesh>>,
  membrane_materials: Res<MembraneMaterials>,
  mut materials: ResMut<Assets<StandardMaterial>>,
  grace_scene_source: Res<GraceSceneSource>,
  selections: Query<Entity, With<Selection>>,
  highlights: Query<Entity, With<Highlight>>,
) {
  if grace_scene_source.0.len() == 0 {
    let grace_scene = GraceScene ( grace::sample::scene2() );
    grace_scene.spawn(Vec3::new(0.0,0.0,0.0), commands, &mut meshes, membrane_materials, &mut materials, selections, highlights);
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

    let camera_x : f32 = -100.0;
    let camera_y = 1000.5;
    let camera_z = 2000.0;
    let camera_radius = (camera_x * camera_x + camera_y * camera_y + camera_z * camera_z).sqrt();
    commands.spawn(
        (Camera3dBundle {
            transform: Transform::from_xyz(camera_x,camera_y,camera_z).looking_at(Vec3::ZERO, Vec3::Y),
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
             radius: camera_radius,
             ..default()
         }
        ));

}
