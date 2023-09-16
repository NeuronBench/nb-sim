use bevy::prelude::*;
use bevy_mod_picking::{
    prelude::{RaycastPickTarget,ListenedEvent,Bubble, OnPointer},
    PickableBundle,
    events::{Click}
};
use crossbeam::channel::{Sender, Receiver};
// use std::sync::mpsc::{channel, Sender, Receiver};
use std::collections::{HashMap, HashSet};

use crate::dimension::{MilliVolts, Diameter, MicroAmpsPerSquareCm};
use crate::neuron::Junction;
use crate::neuron::membrane::{Membrane, MembraneVoltage, MembraneMaterials};
use crate::neuron::solution::EXAMPLE_CYTOPLASM;
use crate::neuron::segment::{ecs::Segment, ecs::InputCurrent, Geometry};
use crate::neuron::synapse::{SynapseMembranes};
use crate::stimulator;
use crate::serialize;
use crate::selection::{Selection, Highlight, spawn_highlight};
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
        mut commands: Commands,
        mut meshes: &mut ResMut<Assets<Mesh>>,
        membrane_materials: Res<MembraneMaterials>,
        materials: &mut ResMut<Assets<StandardMaterial>>,
        selections: Query<Entity, With<Selection>>,
        highlights: Query<Entity, With<Highlight>>,
    ) -> Vec<(Entity, Vec<Entity>)> {
        let neuron_entities = self.0.neurons.iter().map(|scene_neuron| {
            spawn_neuron(&scene_neuron, soma_location_cm, &mut commands, &mut meshes, &membrane_materials, materials, &selections, &highlights)
        }).collect();

        for synapse in &self.0.synapses {
            spawn_synapse(&mut commands, &synapse, &neuron_entities, meshes, materials);
        }
        neuron_entities

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
            .and_modify(|children: &mut Vec<i32>| (children).push(segment.id))
            .or_insert(vec![segment.id]);
    }
    children_map
}

/// Index the entries by id.
pub fn segments_as_map(neuron: &serialize::Neuron) -> HashMap<i32, &serialize::Segment> {
    neuron.segments.iter().map(|segment| (segment.id, segment)).collect()
}

pub fn simplify(mut neuron: serialize::Neuron) -> serialize::Neuron {

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
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    membrane_materials: &MembraneMaterials,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    selections:  &Query<Entity, With<Selection>>,
    highlights:  &Query<Entity, With<Highlight>>,
) -> (Entity, Vec<Entity>) {
    let neuron = &scene_neuron.neuron;
    let serialize::Location { x_mm, y_mm, z_mm } = &scene_neuron.location;
    let v0 = MilliVolts(-88.0);
    let microns_to_screen = 1.0;
    let entry_map = segments_as_map(neuron);
    let soma = soma(neuron).expect("should have soma");
    let mut entities_and_parents : HashMap<i32, (Entity, i32, Diameter, Transform)> = HashMap::new();
    let neuron_entity = commands.spawn(
        (Neuron,
            Transform::from_translation(soma_location_cm),
            GlobalTransform::default(),
            Visibility::default(),
            ComputedVisibility::default(),
        )).id();

    // Spawn segments.
    let segment_entities : Vec<Entity> = neuron.segments.iter().map(|segment|  {
        let serialize::Segment
                { id,
                type_,
                x,
                y,
                z,
                r,
                parent
                } = segment;
        let x_screen = (x - soma.x + x_mm*1000.0) * microns_to_screen;
        let y_screen = (y - soma.y + y_mm*1000.0) * microns_to_screen;
        let z_screen = (z - soma.z + z_mm*1000.0) * microns_to_screen;
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
                let p_x = (p.x - soma.x + x_mm*1000.0) * microns_to_screen;
                let p_y = (p.y - soma.y + y_mm*1000.0) * microns_to_screen;
                let p_z = (p.z - soma.z + z_mm*1000.0) * microns_to_screen;
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
                RaycastPickTarget::default(),
                OnPointer::<Click>::run_callback(add_stimulation),
            )
        ).id();
        commands.entity(neuron_entity).push_children(&[segment_entity]);
        entities_and_parents.insert(id.clone(), (segment_entity, segment.parent, Diameter(1.0), transform));
        segment_entity
    }).into_iter().collect();

    // Spawn segment-segment junctions.
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

    // Spawn stimulations.
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
                     RaycastPickTarget::default(),
                     OnPointer::<Click>::run_callback(delete_stimulations),
                    )
                );
                commands.entity(*entity).insert(stim);
                deselect_all(commands, &selections, highlights);
                // commands.entity(*entity).insert(Selection);
                // spawn_highlight(commands, meshes, materials, entity.clone());

            }
        }
    } 

    (neuron_entity, segment_entities)
}

fn deselect_all(
    commands: &mut Commands,
    selections: &Query<Entity, With<Selection>>,
    highlights: &Query<Entity, With<Highlight>>,
) {
    for entity in selections.iter() {
        eprintln!("Removing selection from {}", entity.to_bits());
        commands.entity(entity).remove::<Selection>();
    }
    for entity in highlights.iter() {
        commands.entity(entity).despawn();
    }
}

#[derive(Component)]
pub struct Synapse {
    pub pre_segment: Entity,
    pub post_segment: Entity,
    pub synapse_membranes: SynapseMembranes,
}

// TODO: Meshes for synapse.
pub fn spawn_synapse(
    commands: &mut Commands,
    synapse: &serialize::Synapse,
    neurons_and_segments: &Vec<(Entity, Vec<Entity>)>,
    _meshes: &mut ResMut<Assets<Mesh>>,
    _materials: &mut ResMut<Assets<StandardMaterial>>
) {
    if let Ok(parsed_synapse_membranes) = SynapseMembranes::deserialize(&synapse.synapse_membranes) {
        let pre_segment = neurons_and_segments[synapse.pre_neuron].1[synapse.pre_segment];
        let post_segment = neurons_and_segments[synapse.post_neuron].1[synapse.post_segment];
        commands.spawn(Synapse { pre_segment, post_segment, synapse_membranes: parsed_synapse_membranes});
    } else {
        eprintln!("Parse result: {:?}", SynapseMembranes::deserialize(&synapse.synapse_membranes));
        panic!("TEMPORARY, quit if synapse parsing fails");
    }
}

pub fn add_stimulation(
    In(event): In<ListenedEvent<Click>>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    selections: Query<Entity, With<Selection>>,
    highlights: Query<Entity, With<Highlight>>,
    new_stimulators: Res<stimulator::Stimulator>,
    // selected_stimulators: Query<(&mut stimulator::Stimulator)>,
    segments_query: Query<(&Segment, &GlobalTransform)>
) -> Bubble {
    match segments_query.get(event.target) {
        Ok((_, segment_transform)) => {

          commands.spawn(
              (stimulator::Stimulation { stimulation_segment: event.target },
               PbrBundle {
                   mesh: meshes.add(shape::UVSphere{ radius: 7.5, sectors: 20, stacks: 20 }.into()),
                   material: materials.add(Color::rgb(0.5,0.5,0.5).into()),
                   transform: Transform::from_translation(segment_transform.translation()),
                   ..default()
               },
               PickableBundle::default(),
               RaycastPickTarget::default(),
               OnPointer::<Click>::run_callback(handle_click_stimulator),
              )
          );
          eprintln!("Inserting stimulator into entity {}", event.target.to_bits());
          commands.entity(event.target).insert(new_stimulators.clone());
          select_stimulator(event.target, commands, selections, highlights, meshes, materials);
        },
      Err(_) => {
          eprintln!("No segment found for clicked entity.");
      },
    }
    Bubble::Up
}

pub fn select_stimulator(
    segment_entity: Entity,
    mut commands: Commands,
    selections: Query<Entity, With<Selection>>,
    highlights: Query<Entity, With<Highlight>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) -> Bubble {
    deselect_all(&mut commands, &selections, &highlights);
    spawn_highlight(&mut commands, &mut meshes, &mut materials, segment_entity.clone());
    commands.entity(segment_entity).insert(Selection);
    eprintln!("inserting Selection into entity {}", segment_entity.to_bits());
    commands.entity(segment_entity).insert(Selection);
    Bubble::Up
}

pub fn handle_click_stimulator(
    In(event): In<ListenedEvent<Click>>,
    commands: Commands,
    mut stimulations_query: Query<&stimulator::Stimulation>,
    segments_query: Query<(&Segment, Entity, &stimulator::Stimulator)>,
    selections: Query<Entity, With<Selection>>,
    highlights: Query<Entity, With<Highlight>>,
    meshes: ResMut<Assets<Mesh>>,
    materials: ResMut<Assets<StandardMaterial>>,
) -> Bubble {
    if let Ok(stimulator::Stimulation { stimulation_segment }) = stimulations_query.get_mut(event.target) {
        let results = segments_query.get(stimulation_segment.clone());
        match results {
            Ok((_, segment_entity, _)) => {
                eprintln!("Ok, seeing a stimulator. Selecting its entity.");
                select_stimulator(
                    segment_entity,
                    commands,
                    selections,
                    highlights,
                    meshes,
                    materials,
                );
            },
            Err(e) => {
                eprintln!("Error in select_stimulator: {:?}", e);
            }
        }
    }
    Bubble::Up
}

pub fn delete_stimulations(
    In(event): In<ListenedEvent<Click>>,
    mut commands: Commands,
    mut stimulations_query: Query<&stimulator::Stimulation>,
    segments_query: Query<(&Segment, Entity, &stimulator::Stimulator)>,
) -> Bubble {
  if let Ok(stimulator::Stimulation { stimulation_segment }) = stimulations_query.get_mut(event.target) {

      // Remove stimulation from the segment.
      let results = segments_query.get(stimulation_segment.clone());
      match results {
          Ok((_, segment_entity, _)) => {
              commands.entity(segment_entity.clone()).remove::<stimulator::Stimulator>();
          },
          Err(_) => {
              eprintln!("Missing segment for deleted stimulation.");
          }
      }

      // Despawn the stimulator.
      commands.entity(event.target).despawn();
  }
  Bubble::Up
}


pub mod sample {
    use std::include_str;
    use crate::serialize;
    use crate::dimension::MilliVolts;
    use crate::neuron::synapse;

    pub fn neuron() -> serialize::Neuron {
        let s = include_str!("../../sample_data/swc_neuron.json");
        serde_json::from_str(s).expect("should parse")
    }

    pub fn scene() -> serialize::Scene {
        let stimulator = serialize::Stimulator {
                            envelope: serialize::Envelope {
                                period_sec: 0.1,
                                onset_sec: 0.001,
                                offset_sec: 0.07,
                            },
                            current_shape: serialize::CurrentShape::SquareWave {
                                on_current_uamps_per_square_cm: 200.0,
                                off_current_uamps_per_square_cm: -1.0,
                            },
                        };
        let n = neuron();
        serialize::Scene {
            neurons: vec![serialize::SceneNeuron {
                neuron: n.clone(),
                location: serialize::Location {
                  x_mm: 0.5,
                  y_mm: 0.1,
                  z_mm: 0.0,
                },
                stimulator_segments: vec![
                    serialize::StimulatorSegment {
                        stimulator: stimulator.clone(),
                        segment: 100,
                    }
                ]
            }
            , serialize::SceneNeuron {
                neuron: n.clone(),
                location: serialize::Location {
                    x_mm: -0.4, y_mm: 0.5, z_mm: 0.0
                },
                stimulator_segments: vec![]
            }
            ],

            synapses: vec![ serialize::Synapse {
                pre_neuron: 0,
                pre_segment: 37,
                post_neuron: 1,
                post_segment: 333,
                synapse_membranes: synapse::examples::excitatory_synapse(&MilliVolts(-80.0)).serialize(),
            }],
        }

    }

    pub fn scene2() -> serialize::Scene {
        let s = include_str!("../../sample_data/sample_scene.json");
        serde_json::from_str(s).expect("should parse")
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
