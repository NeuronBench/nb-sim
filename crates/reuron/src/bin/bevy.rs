use bevy::prelude::*;
// use bevy::diagnostic::{LogDiagnosticsPlugin, FrameTimeDiagnosticsPlugin};
use bevy::pbr::CascadeShadowConfigBuilder;
use std::f32::consts::PI;

use reuron::plugin::ReuronPlugin;
use reuron::integrations::swc_file::SwcFile;
use reuron::neuron::segment::ecs::Segment;
use reuron::neuron::membrane::MembraneMaterials;

#[derive(Component)]
struct MyCamera;

pub fn main() {
  App::new()
        .add_plugins(DefaultPlugins)
        .add_plugin(ReuronPlugin)
        .add_system(bevy::window::close_on_esc)
        .add_startup_system(setup_scene)
        .add_startup_system(setup_swc_neuron)
        .insert_resource(ClearColor(Color::rgb(0.2,0.2,0.2)))
        .run();
}

fn setup_swc_neuron(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut camera_query: Query<(&MyCamera, &mut Transform)>,
    segments_query: Query<(&Segment, &GlobalTransform)>,
    materials: Res<MembraneMaterials>,
) {
  // let swc_neuron = SwcFile::read_file("/home/greghale/Downloads/H17.03.010.11.13.06_651089035_m.swc").expect("should parse");
  // let soma_entity = swc_neuron.simplify().spawn(commands, meshes, materials);
  // let soma_entity = SwcFile::sample().spawn(commands, meshes, materials);
  // let soma_transform = segments_query.get_component::<GlobalTransform>(soma_entity).expect("soma exists");
  // println!("Soma translation: {:?}", soma_transform.translation());
  // let (_, mut camera_transform) = camera_query.get_single().expect("just one camera");
  // camera_transform = &camera_transform.looking_at(soma_transform.translation(), Vec3::Y);
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

    commands.spawn((Camera3dBundle {
        transform: Transform::from_xyz(-400.0,400.5, 2000.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    }, MyCamera));
}
