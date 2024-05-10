use bevy::prelude::*;
use bevy_mod_picking::{
    prelude::*,
    PickableBundle,
};

#[derive(Component)]
pub struct Selection;

#[derive(Component)]
pub struct Highlight;

pub fn spawn_highlight(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    selected_entity: Entity,
) {
    eprintln!("Spawn highlight");
    let highlight_entity = commands.spawn((
        Highlight,
        PbrBundle {
            mesh: meshes.add(Sphere { radius: 8.5}),
            material: materials.add(StandardMaterial {
                base_color: Color::rgba(1.0,1.0,1.0,0.5),
                ..default()
            }),
            transform: Transform::from_xyz(0.0,0.0,0.0),
            ..default()
        },
        PickableBundle::default(),
        // OnPointer::<Click>::run_callback(deselect_all),
    )).id();
    commands.entity(selected_entity).push_children(&[highlight_entity]);
}
