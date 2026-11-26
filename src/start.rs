use bevy::prelude::*;
use bevy::picking::mesh_picking::MeshPickingPlugin;
use bevy::post_process::bloom::Bloom;
use bevy::diagnostic::{LogDiagnosticsPlugin, FrameTimeDiagnosticsPlugin};
use bevy::color::Srgba;
use bevy::render::settings::{Backends, RenderCreation, WgpuSettings};
use bevy::render::RenderPlugin;
use bevy_egui::{EguiPlugin, EguiPrimaryContextPass};
use bevy_panorbit_camera::{PanOrbitCameraPlugin, PanOrbitCamera};
use std::f32::consts::PI;
use wasm_bindgen::prelude::*;

use crate::plugin::NbSimPlugin;
use crate::gui::run_gui;
use crate::gui::load::{handle_loaded_neuron, SceneSource};
use crate::gui::model_view::{model_view_tooltip, ModelViewPlugin};
use crate::gui::models_panel::run_models_panel;
use crate::gui::canvas_panel::run_canvas_panel;
use crate::gui::scene_panel::run_scene_panel;
use crate::integrations::grace::{self, GraceScene};
use crate::integrations::neuroml::sample as neuroml_sample;
use crate::neuron::membrane::MembraneMaterials;
use crate::selection::{Selection, Highlight};
use crate::gui::external_trigger::ExternalTriggerPlugin;

#[derive(Component)]
struct MyCamera;



/// Start the simulator.
///
/// `interpreter_url` is accepted for compatibility with older embedders and
/// ignored: scenes are now evaluated in-process. `demo` spawns the built-in
/// sample scene when no scene source is configured.
#[wasm_bindgen]
pub fn start(
  interpreter_url: String,
  demo: bool,
) {
 let _ = interpreter_url;

 let mut app = App::new();
 app
    .add_plugins((DefaultPlugins
      .set(WindowPlugin {
        primary_window: Some(Window {
          title: "".to_string(),
          canvas: Some("#bevy".to_string()),
          ..default()
        }),
        ..default()
      })
      .set(RenderPlugin {
        render_creation: RenderCreation::Automatic(WgpuSettings {
          // The web build picks WebGPU or WebGL2; native uses its own
          // primary API (Metal, Vulkan, DX12).
          backends: Some(if cfg!(target_arch = "wasm32") {
            Backends::BROWSER_WEBGPU | Backends::GL
          } else {
            Backends::PRIMARY
          }),
          ..default()
        }),
        ..default()
      }),
    MeshPickingPlugin))
        .add_plugins(EguiPlugin::default())
        .add_plugins(LogDiagnosticsPlugin::default())
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(NbSimPlugin)
        .add_plugins(ExternalTriggerPlugin)
        .add_plugins(PanOrbitCameraPlugin)
        .add_plugins(ModelViewPlugin)
        .add_systems(Startup, setup_scene)
        .insert_resource(ClearColor(Srgba::hex("#0e0e1f").unwrap().into()))
        .add_systems(EguiPrimaryContextPass, (run_gui, run_scene_panel, run_models_panel, run_canvas_panel, crate::readout::run_readout_panel, model_view_tooltip))
        .add_systems(Update, (handle_loaded_neuron, debug_screenshot));

        if demo {
          app.add_systems(Startup, setup_grace_neuron);
        }

        app.run();
}

/// Diagnostic knob: NB_DEBUG_SCREENSHOT=path saves one screenshot of the
/// window after NB_DEBUG_SCREENSHOT_SECS (default 10) seconds.
fn debug_screenshot(mut commands: Commands, time: Res<Time>, mut done: Local<bool>) {
  use bevy::render::view::screenshot::{save_to_disk, Screenshot};
  if *done {
    return;
  }
  let Ok(path) = std::env::var("NB_DEBUG_SCREENSHOT") else {
    *done = true;
    return;
  };
  let after = std::env::var("NB_DEBUG_SCREENSHOT_SECS").ok().and_then(|s| s.parse().ok()).unwrap_or(10.0);
  if time.elapsed_secs_f64() >= after {
    *done = true;
    info!("NB_DEBUG_SCREENSHOT: saving {path}");
    commands.spawn(Screenshot::primary_window()).observe(save_to_disk(path));
  }
}

fn setup_grace_neuron(
  commands: Commands,
  mut meshes: ResMut<Assets<Mesh>>,
  membrane_materials: Res<MembraneMaterials>,
  mut materials: ResMut<Assets<StandardMaterial>>,
  grace_scene_source: Res<SceneSource>,
  selections: Query<Entity, With<Selection>>,
  highlights: Query<Entity, With<Highlight>>,
) {
  if grace_scene_source.0.len() == 0 {
    let grace_scene = GraceScene ( grace::sample::scene2() );
    grace_scene.spawn(Vec3::new(0.0,0.0,0.0), commands, &mut meshes, membrane_materials, &mut materials, selections, highlights);
  }
}

fn setup_neuroml_neuron(
  commands: Commands,
  mut meshes: ResMut<Assets<Mesh>>,
  membrane_materials: Res<MembraneMaterials>,
  mut materials: ResMut<Assets<StandardMaterial>>,
  // neuroml_scene_source: Res<NeuromlSceneSource>,
  selections: Query<Entity, With<Selection>>,
  highlights: Query<Entity, With<Highlight>>,
) {
  let neuroml_scene = neuroml_sample::scene1();
  neuroml_scene.spawn(Vec3::ZERO, commands, &mut meshes, membrane_materials, &mut materials, selections, highlights);
}


fn setup_scene(
    mut commands: Commands,
) {
    commands.spawn((
        DirectionalLight {
            shadows_enabled: true,
            ..default()
        },
        Transform {
            translation: Vec3::new(0.0,2.0, 0.0),
            rotation: Quat::from_rotation_x(-PI/ 4.0),
            ..default()
        },
    ));

    let camera_x : f32 = -100.0;
    let camera_y = 1000.5;
    let camera_z = 2000.0;
    let camera_radius = (camera_x * camera_x + camera_y * camera_y + camera_z * camera_z).sqrt();
    commands.spawn(
        (Camera3d::default(),
         Transform::from_xyz(camera_x,camera_y,camera_z).looking_at(Vec3::ZERO, Vec3::Y),
         MyCamera,
         Bloom::NATURAL,
         PanOrbitCamera {radius: Some(camera_radius), ..default()},
        ));

}
