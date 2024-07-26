use bevy::prelude::*;
use bevy::core_pipeline::bloom::{BloomPlugin, BloomSettings};
use bevy::diagnostic::{LogDiagnosticsPlugin, FrameTimeDiagnosticsPlugin};
use bevy::pbr::CascadeShadowConfigBuilder;
use bevy_egui::EguiPlugin;
use bevy_mod_picking::prelude::*;
use bevy_panorbit_camera::{PanOrbitCameraPlugin, PanOrbitCamera};
use std::f32::consts::PI;
use wasm_bindgen::prelude::*;

use crate::plugin::NbSimPlugin;
use crate::gui::run_gui;
use crate::gui::load::{handle_loaded_neuron, GraceSceneSource, InterpreterUrl};
use crate::integrations::grace::{self, GraceScene};
use crate::neuron::membrane::MembraneMaterials;
// use bevy_panorbit_camera::{PanOrbitCamera, pan_orbit_camera};
use crate::selection::{Selection, Highlight};
use crate::gui::external_trigger::ExternalTriggerPlugin;
use crate::onnx::OnnxPlugin;

#[derive(Component)]
struct MyCamera;



#[wasm_bindgen]
pub fn start(
  interpreter_url: String,
  demo: bool,
) {

 let mut app = App::new();
 app
    .add_plugins(DefaultPlugins.set(WindowPlugin {
      primary_window: Some(Window {
        title: "".to_string(),
        canvas: Some("#bevy".to_string()),
        ..default()
      }),
      ..default()
    }))
        .add_plugins(EguiPlugin)
        .add_plugins(LogDiagnosticsPlugin::default())
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(DefaultPickingPlugins.build().disable::<DebugPickingPlugin>())
        // .add_plugin(DebugCursorPickingPlugin)
        // .add_plugin(DebugEventsPickingPlugin)
        .add_plugins(NbSimPlugin)
        .add_plugins(OnnxPlugin)
        .add_plugins(ExternalTriggerPlugin)
        .add_plugins(PanOrbitCameraPlugin)
        .add_systems(Update, bevy::window::close_on_esc)
        .add_systems(Startup, setup_scene)
        .insert_resource(InterpreterUrl(interpreter_url))
        .insert_resource(ClearColor(Color::hex("#0e0e1f").expect("valid hex")))
        .add_systems(Update, run_gui)
        .add_systems(Update, handle_loaded_neuron);

        if demo {
          app.add_systems(Startup, setup_grace_neuron);
        }

        app.run();
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

         BloomSettings::NATURAL,
         PanOrbitCamera {radius: Some(camera_radius), ..default()},  // Set radius to camera_radius
         // RaycastPickCamera::default(),

        ));

}
