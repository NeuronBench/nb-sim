use bevy::prelude::*;
use bevy::diagnostic::{LogDiagnosticsPlugin, FrameTimeDiagnosticsPlugin};
use bevy::pbr::CascadeShadowConfigBuilder;
use std::f32::consts::PI;

use reuron::plugin::ReuronPlugin;

#[derive(Component)]
struct MyCamera;

pub fn main() {
  App::new()
        .add_plugins(DefaultPlugins)
//         .add_plugin(LogDiagnosticsPlugin::default())
//         .add_plugin(FrameTimeDiagnosticsPlugin::default())
        .add_plugin(ReuronPlugin)
        .add_system(bevy::window::close_on_esc)
        .add_startup_system(setup_scene)
        .insert_resource(ClearColor(Color::rgb(0.3,0.2,0.2)))
        .run();
}

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>
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

    commands.spawn((Camera3dBundle {
        transform: Transform::from_xyz(-50.0,50.5, 100.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    }, MyCamera));
}
