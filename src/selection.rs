use bevy::prelude::*;

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
        Mesh3d(meshes.add(Sphere { radius: 8.5})),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgba(1.0,1.0,1.0,0.5),
            ..default()
        })),
        Transform::from_xyz(0.0,0.0,0.0),
        Pickable::default(),
    )).id();
    commands.entity(selected_entity).add_children(&[highlight_entity]);
}
