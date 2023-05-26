use bevy::prelude::*;
use bevy_mod_picking::{PickableBundle, prelude::OnPointer, events::{Click, Drag}};
use crossbeam::channel::{Sender, Receiver};
// use std::sync::mpsc::{channel, Sender, Receiver};
use std::collections::{HashMap, HashSet};

use crate::dimension::{MilliVolts, Diameter, MicroAmpsPerSquareCm};
use crate::neuron::Junction;
use crate::neuron::membrane::{self, Membrane, MembraneVoltage, MembraneMaterials};
use crate::neuron::solution::EXAMPLE_CYTOPLASM;
use crate::neuron::segment::{ecs::Segment, ecs::InputCurrent, Geometry};
use crate::stimulator;
use crate::serialize;
use crate::neuron::ecs::Neuron;

#[derive(Clone)]
pub struct GraceScene( pub serialize::Scene );

#[derive(Resource, Clone)]
pub struct GraceSceneSender(pub Sender<GraceScene>);

#[derive(Resource)]
pub struct GraceSceneReceiver(pub Receiver<GraceScene>);

impl GraceScene {

    pub fn spawn(
        &self,
        soma_location_cm: Vec3,
        mut commands: &mut Commands,
        mut meshes: &mut ResMut<Assets<Mesh>>,
        membrane_materials: Res<MembraneMaterials>,
        mut materials: &mut ResMut<Assets<StandardMaterial>>
    ) -> Vec<Entity> {
        self.0.neurons.iter().map(move |scene_neuron| {
            spawn_neuron(&scene_neuron, soma_location_cm, commands, meshes, &membrane_materials, materials)
        }).collect()

    }

}

pub fn soma(neuron: &serialize::Neuron) -> Option<&serialize::Segment> {
    neuron.segments.iter().find(|s| s.parent == -1 && s.type_ == 1)
}

/// Determine each segment's children.
pub fn get_children(neuron: &serialize::Neuron) -> HashMap<i32, Vec<i32>> {
    let mut children_map = HashMap::new();
    for segment in neuron.segments.iter() {
        // Modify self's parent, inserting self as a child.
        children_map.entry(segment.parent)
            .and_modify(|mut children: &mut Vec<i32>| (children).push(segment.id))
            .or_insert(vec![segment.id]);
    }
    children_map
}

/// Index the entries by id.
pub fn segments_as_map(neuron: &serialize::Neuron) -> HashMap<i32, &serialize::Segment> {
    neuron.segments.iter().map(|segment| (segment.id, segment)).collect()
}

pub fn simplify(mut neuron: serialize::Neuron) -> serialize::Neuron {

    let segments_copy = neuron.segments.clone();
    let children_map = get_children(&neuron);

    let neuron_for_map = neuron.clone();
    let entries_map = segments_as_map(&neuron_for_map);

    let should_keep : HashSet<i32> = neuron.segments.iter().filter_map(|e| {
        // Keep the soma.
        let is_first = e.id == 1;
        // Keep all branches and leaves (nodes with multiple children or zero children).
        let is_branch_or_leaf = !children_map.get(&e.id).map_or(false, |l| l.len() == 1);
        // Keep 1/10 of all nodes no matter what.
        let is_downsample = e.id % 10 == 0;
        if is_first || is_branch_or_leaf || is_downsample {
            Some(e.id)
        } else {
            None
        }
    }).collect();

    // For each entry, check if its parent is tombstoned.
    // If so, set the entry's parent to its current grandparent.
    // Repeat this process until the current parent is not tombstoned.
    for mut entry in neuron.segments.iter_mut() {
        while !(should_keep.contains(&entry.parent) || entry.parent == -1) {
            entry.parent = entries_map.get(&entry.parent).expect("parent should exist").parent;
        }
    }

    let filtered_entries = neuron
        .segments
        .into_iter()
        .filter(|e| should_keep.contains(&e.id))
        .collect();
    neuron.segments = filtered_entries;
    neuron.clone()
}


pub fn distance_to_segment_cm(source: &serialize::Segment, dest: &serialize::Segment) -> f32 {
    let dist_microns = (
            (source.x - dest.x).powi(2) +
            (source.y - dest.y).powi(2) +
            (source.z - dest.z).powi(2)
    ).sqrt();
    dist_microns * 0.0001
}

pub fn spawn_neuron(
    scene_neuron: &serialize::SceneNeuron,
    soma_location_cm: Vec3,
    mut commands: &mut Commands,
    mut meshes: &mut ResMut<Assets<Mesh>>,
    membrane_materials: &MembraneMaterials,
    materials: &mut ResMut<Assets<StandardMaterial>>,
) -> Entity {
    let neuron = &scene_neuron.neuron;
    let v0 = MilliVolts(-88.0);
    let microns_to_screen = 1.0;
    let entry_map = segments_as_map(neuron);
    let soma = soma(neuron).expect("should have soma");
    let mut entities_and_parents : HashMap<i32, (Entity, i32, Diameter, Transform)> = HashMap::new();
    let mut children_map = get_children(neuron);
    let neuron_entity = commands.spawn(
        (Neuron,
            Transform::from_translation(soma_location_cm),
            GlobalTransform::default(),
            Visibility::default(),
            ComputedVisibility::default(),
        )).id();
    for segment in neuron.segments.iter() {
        let serialize::Segment
                { id,
                type_,
                x,
                y,
                z,
                r,
                parent
                } = segment;
        let x_screen = (x - soma.x) * microns_to_screen;
        let y_screen = (y - soma.y) * microns_to_screen;
        let z_screen = (z - soma.z) * microns_to_screen;
        let default_length_cm = 2.0 * r * 0.0001;
        let length_cm = match (type_, entry_map.get(&parent)) {
            (1, _) => default_length_cm,
            (_, None) => default_length_cm,
            (_, Some(parent_segment)) => distance_to_segment_cm(&segment, parent_segment),
        };
        let length_screen = length_cm * 10000.0 * microns_to_screen;
        let radius_cm = r * 0.0001;
        let radius_screen = radius_cm * 10000.0 * microns_to_screen;
        let shape = match segment.type_ {
            1 => shape::UVSphere {
                radius: length_screen * 0.5,
                sectors: 12,
                stacks: 12,
            }.into(),
            _ => shape::Cylinder {
                        radius: radius_screen * 5.0,
                        height: length_screen,
                        resolution: 12,
                        segments:4,
                    }.into(),
        };

        let membrane_serialized = neuron.membranes.get(segment.type_ - 1).expect("type should be a valid index into membranes");
        let membrane = Membrane::deserialize(membrane_serialized);
        let look_target = match entry_map.get(parent) {
            None => {
                Vec3::ZERO
            },
            Some(p) => {
                let p_x = (p.x - soma.x) * microns_to_screen;
                let p_y = (p.y - soma.y) * microns_to_screen;
                let p_z = (p.z - soma.z) * microns_to_screen;
                Vec3::new(p_x, p_y, p_z)
            }
        };

        let mut transform = Transform::from_xyz(x_screen, y_screen, z_screen);
        transform.look_at(look_target, Vec3::Y);
        transform.rotate_local_x(std::f32::consts::PI / 2.0);
        transform.translation -= transform.local_y() * length_screen * 0.5;

        let input_current = if segment.type_ == 3 {
            MicroAmpsPerSquareCm(-1.8)
        } else {
            MicroAmpsPerSquareCm(-1.8)
        };
        let segment_entity = commands.spawn(
            (Segment,
                EXAMPLE_CYTOPLASM,
                membrane,
                MembraneVoltage(v0.clone()),
                Geometry::Cylinder {
                    diameter: Diameter(1.0),
                    length: 1.0,
                }, // TODO use real geometry. But be careful not to get units wrong,
                // which has caused the model to become unstable

                InputCurrent(input_current),
                PbrBundle {
                    mesh: meshes.add(shape),
                    material: membrane_materials.from_voltage(&v0),
                    transform: transform,
                    ..default()
                },
                PickableBundle::default(),
                     OnPointer::<Click>::target_commands_mut(|_click, target_commands| {
                         eprintln!("CLICK");
                         target_commands.despawn();
                     }),
                     OnPointer::<Drag>::target_component_mut::<Transform>(|drag, transform| {
                         eprintln!("DRAG");
                         transform.rotate_local_y(drag.delta.x / 50.0)
                     }),
            )
        ).id();
        commands.entity(neuron_entity).push_children(&[segment_entity]);
        entities_and_parents.insert(id.clone(), (segment_entity, segment.parent, Diameter(1.0), transform));
    }

    for (entry_id, (entity, parent_id, diameter, _)) in entities_and_parents.iter() {
        match entities_and_parents.get(&parent_id) {
            None => { println!("Entry {:?} with parent {:?} has no parent entry", entry_id, parent_id); },
            Some((parent_entity,_,parent_diameter,_)) => {
                let d = Diameter( diameter.0.min(parent_diameter.0) );
                let junction = commands.spawn(Junction {
                    first_segment: parent_entity.clone(),
                    second_segment: entity.clone(),
                    pore_diameter: d
                }).id();
                commands.entity(neuron_entity).push_children(&[junction]);
            }
        }
    }

    for serialize::StimulatorSegment { segment, stimulator } in scene_neuron.stimulator_segments.iter() {
        match entities_and_parents.get(&(*segment as i32)) {
            None => { println!("Failed to look up segment id {segment:?}") },
            Some((entity,_,_,transform)) => {
                let stim = stimulator::Stimulator::deserialize(stimulator);
                println!("INSERTING A STIMULATOR");
                commands.spawn(
                    (stimulator::Stimulation { stimulation_segment: entity.clone() },
                     PbrBundle {
                        mesh: meshes.add(shape::UVSphere{
                            radius: 7.5,
                            sectors: 20,
                            stacks: 20
                        }.into()),
                        material: materials.add(Color::rgb(0.5,0.5,0.5).into()),
                        transform: Transform::from_translation(transform.translation),
                        ..default()
                     },
                     PickableBundle::default(),
                     OnPointer::<Click>::target_commands_mut(|_click, target_commands| {
                         eprintln!("CLICK");
                         target_commands.despawn();
                     }),
                     OnPointer::<Drag>::target_component_mut::<Transform>(|drag, transform| {
                         eprintln!("DRAG");
                         transform.rotate_local_y(drag.delta.x / 50.0)
                     }),

                    )
                );
                commands.entity(*entity).insert(stim);
            }
        }
    } 

    neuron_entity
}


pub mod sample {
    use std::include_str;
    use crate::serialize;
    pub fn neuron() -> serialize::Neuron {
        let s = include_str!("../../sample_data/swc_neuron.json");
        serde_json::from_str(s).expect("should parse")
    }

    pub fn scene() -> serialize::Scene {
        let n = neuron();
        serialize::Scene {
            neurons: vec![serialize::SceneNeuron {
                neuron: n,
                stimulator_segments: vec![
                    serialize::StimulatorSegment {
                        stimulator: serialize::Stimulator {
                            envelope: serialize::Envelope {
                                period_sec: 0.1,
                                onset_sec: 0.001,
                                offset_sec: 0.07,
                            },
                            current_shape: serialize::CurrentShape::SquareWave {
                                on_current_uamps_per_square_cm: 200.0,
                                off_current_uamps_per_square_cm: -1.0,
                            },
                        },
                        segment: 100,
                    }
                ]
            }],
            // synapses: vec![],
        }
    }
}

#[cfg(test)]
pub mod tests {

    use super::*;
    use crate::serialize;
    use crate::dimension::{Molar};

    #[test]
    pub fn parse_solution() {

        // Parsing warmup.
        let solution : serialize::Solution = serde_json::from_str(
            "{\"k\": 1, \"na\": 1, \"ca\": 0, \"cl\": 0}"
        ).expect("should parse");
        assert!(solution.k - 1.0 < 1e-7);

        let neuron : serialize::Neuron = sample::neuron();
    }

}
