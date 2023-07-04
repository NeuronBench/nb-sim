use bevy::prelude::*;
use bevy_mod_picking::{
    prelude::{RaycastPickTarget,ListenedEvent,Bubble, OnPointer},
    PickableBundle,
    events::{Click, Drag}
};

#[derive(Component)]
pub struct Selection;

#[derive(Component)]
pub struct Highlight;

pub fn spawn_highlight(
    mut commands: &mut Commands,
    mut meshes: &mut ResMut<Assets<Mesh>>,
    mut materials: &mut ResMut<Assets<StandardMaterial>>,
    selected_entity: Entity,
) {
    eprintln!("Spawn highlight");
    let highlight_entity = commands.spawn((
        Highlight,
        PbrBundle {
            mesh: meshes.add(shape::UVSphere { radius: 8.5, sectors: 20, stacks: 20 }.into()),
            material: materials.add(StandardMaterial {
                base_color: Color::rgba(1.0,1.0,1.0,0.5),
                ..default()
            }),
            transform: Transform::from_xyz(0.0,0.0,0.0),
            ..default()
        },
        PickableBundle::default(),
        RaycastPickTarget::default(),
        // OnPointer::<Click>::run_callback(deselect_all),
    )).id();
    commands.entity(selected_entity).push_children(&[highlight_entity]);
}
