use bevy::prelude::*;
use bevy_mod_picking::PickableBundle;
use std::str::FromStr;
use std::fs;
use std::path::Path;
use std::collections::{HashSet, HashMap};

use crate::dimension::{FaradsPerSquareCm, MilliVolts, Diameter, MicroAmpsPerSquareCm};
use crate::neuron::membrane::{self, Membrane, MembraneVoltage, MembraneMaterials};
use crate::neuron::Junction;
use crate::neuron::segment::{ecs::Segment, ecs::InputCurrent, Geometry};
use crate::neuron::solution::EXAMPLE_CYTOPLASM;
use crate::neuron::channel;
use crate::neuron::ecs::Neuron;

#[derive(Clone, Debug)]
pub struct SwcFile {
    pub entries: Vec<SwcEntry>
}

impl SwcFile {
    pub fn read_file<P: AsRef<Path>>(fp: P) -> Result<Self, ParseError> {
        let contents =
            fs::read_to_string(fp).map_err(|e| ParseError(format!("Error opening file: {e}")))?;
        let swc_lines = contents.lines().map(SwcEntry::from_line).collect::<Result<Vec<Option<_>>,_>>()?;
        Ok(SwcFile {
            entries: swc_lines.into_iter().flatten().collect()
        })
    }

    /// Determine each segment's children.
    pub fn get_children(&self) -> HashMap<i32, Vec<i32>> {
        let mut children_map = HashMap::new();
        for segment in self.entries.iter() {
            // Modify self's parent, inserting self as a child.
            children_map.entry(segment.parent)
                .and_modify(|mut children: &mut Vec<i32>| (children).push(segment.id))
                .or_insert(vec![segment.id]);
        }
        children_map
    }

    /// Index the entries by id.
    pub fn as_map(&self) -> HashMap<i32, &SwcEntry> {
        self.entries.iter().map(|entry| (entry.id, entry)).collect()
    }

    pub fn soma(&self) -> Option<&SwcEntry> {
        self.entries.iter().find(|e| e.segment_type == Some(SegmentType::Soma))
    }

    pub fn spawn(
        &self,
        soma_location_cm: Vec3,
        mut commands: &mut Commands,
        mut meshes: &mut ResMut<Assets<Mesh>>,
        materials: &mut Res<MembraneMaterials>,
    ) -> Entity {
        let v0 = MilliVolts(-80.0);
        let microns_to_screen = 1.0;
        let entry_map = self.as_map();
        let soma = self.soma().expect("Soma should exist");
        let mut entities_and_parents : HashMap<i32, (Entity, i32, Diameter)> = HashMap::new();
        let mut children_map = self.get_children();
        let neuron = commands.spawn(
            (Neuron,
             Transform::from_translation(soma_location_cm),
             GlobalTransform::default(),
             Visibility::default(),
             ComputedVisibility::default(),
            )).id();
        for e in self.entries.iter() {
            let SwcEntry { id,
                           segment_type,
                           x_microns,
                           y_microns,
                           z_microns,
                           radius_microns,
                           parent
            } = e;
            let x_screen = (x_microns - soma.x_microns) * microns_to_screen;
            let y_screen = (y_microns - soma.y_microns) * microns_to_screen;
            let z_screen = (z_microns - soma.z_microns) * microns_to_screen;
            let default_length_cm = 2.0 * radius_microns * 0.0001;
            let length_cm = match (segment_type, entry_map.get(&parent)) {
                (Some(SegmentType::Soma), _) => default_length_cm,
                (_, None) => default_length_cm,
                (_, Some(parent_segment)) => e.distance_to_segment_cm(parent_segment),
            };
            let length_screen = length_cm * 10000.0 * microns_to_screen;
            let radius_cm = radius_microns * 0.0001;
            let radius_screen = radius_cm * 10000.0 * microns_to_screen;
            let membrane = match segment_type {
                Some(SegmentType::Soma) => soma_membrane(),
                Some(SegmentType::Axon) => if parent.clone() == -1 {
                    axon_initial_segment_membrane()
                } else {
                    axon_membrane()
                }
                Some(SegmentType::Dendrite) => basal_dendrite_membrane(),
                Some(SegmentType::ApicalDendrite) => apical_dendrite_membrane(),
                Some(SegmentType::Custom) => basal_dendrite_membrane(),
                None => basal_dendrite_membrane(),
            };
            let look_target = match entry_map.get(parent) {
                None => {
                    Vec3::ZERO
                },
                Some(p) => {
                    let p_x = (p.x_microns - soma.x_microns) * microns_to_screen;
                    let p_y = (p.y_microns - soma.y_microns) * microns_to_screen;
                    let p_z = (p.z_microns - soma.z_microns) * microns_to_screen;
                    Vec3::new(p_x, p_y, p_z)
                }
            };

            let mut transform = Transform::from_xyz(x_screen, y_screen, z_screen);
            transform.look_at(look_target, Vec3::Y);
            transform.rotate_local_x(std::f32::consts::PI / 2.0);
            transform.translation -= transform.local_y() * length_screen * 0.5;


            // shift.mul_transform(transform);

            let input_current = if e.segment_type == Some(SegmentType::ApicalDendrite) {
                MicroAmpsPerSquareCm(-1.0)
            } else {
                MicroAmpsPerSquareCm(-1.0)
            };
            let segment = commands.spawn(
                (Segment,
                 EXAMPLE_CYTOPLASM,
                 membrane,
                 MembraneVoltage(v0.clone()),
                 Geometry {
                     diameter: Diameter(1.0),
                     length: 1.0,
                 },
                 InputCurrent(input_current),
                 PbrBundle {
                     mesh: meshes.add(shape::Cylinder {
                         radius: radius_screen * 5.0,
                         height: length_screen,
                         resolution: 12,
                         segments:4,
                     }.into()),
                     material: materials.from_voltage(&v0),
                     transform: transform,
                     ..default()
                 },
                 PickableBundle::default(),
                )
            ).id();
            commands.entity(neuron).push_children(&[segment]);
            entities_and_parents.insert(id.clone(), (segment, e.parent, Diameter(1.0)));
        }

        for (entry_id, (entity, parent_id, diameter)) in entities_and_parents.iter() {
            match entities_and_parents.get(&parent_id) {
                None => { println!("Entry {:?} with parent {:?} has no parent entry", entry_id, parent_id); },
                Some((parent_entity,_,parent_diameter)) => {
                    let d = Diameter( diameter.0.min(parent_diameter.0) );
                    let junction = commands.spawn(Junction {
                        first_segment: parent_entity.clone(),
                        second_segment: entity.clone(),
                        pore_diameter: d
                    }).id();
                    commands.entity(neuron).push_children(&[junction]);
                }
            }
        }

        entities_and_parents.get(&1).expect("entity 1 should be the soma").0.clone()
    }

    pub fn simplify(mut self) -> Self {

        let entries_copy = self.clone();
        let children_map = entries_copy.get_children();
        let entries_map = entries_copy.as_map();
        let should_keep : HashSet<i32> = self.entries.iter().filter_map(|e| {
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
        for mut entry in self.entries.iter_mut() {
            while !(should_keep.contains(&entry.parent) || entry.parent == -1) {
                entry.parent = entries_map.get(&entry.parent).expect("parent should exist").parent;
            }
        }

        let filtered_entries = self
            .entries
            .into_iter()
            .filter(|e| should_keep.contains(&e.id))
            .collect();
        SwcFile {
            entries: filtered_entries
        }
    }

    pub fn sample() -> Self {
        let mk_entry = |id: i32| -> SwcEntry {
            SwcEntry { id: id,
                       x_microns: 2.0,
                       y_microns: 2.0,
                       z_microns: 3.0 * (id - 2) as f32,
                       radius_microns: 0.1,
                       segment_type: Some(if id == 1 { SegmentType::Soma } else {SegmentType::Axon}),
                       parent: if id == 1 { -1  } else { id - 1 }
                     }
        };
        SwcFile {
            entries: (1..50).map(mk_entry).collect()
        }
    }
}

#[derive(Clone, Debug)]
pub struct SwcEntry {
    pub id: i32,
    pub segment_type: Option<SegmentType>,
    pub x_microns: f32,
    pub y_microns: f32,
    pub z_microns: f32,
    pub radius_microns: f32,
    pub parent: i32,
}

impl SwcEntry {
    pub fn from_line(line: &str) -> Result<Option<Self>, ParseError> {
        match line.chars().next() {
            None => Ok(None),
            Some('#') => Ok(None),
            _ => {
                let words: Vec<&str> = line.split(' ').collect();
                if words.len() == 7 {
                    Ok(Some(SwcEntry {
                        id: parse(words[0], "id")?,
                        segment_type : SegmentType::from_code(parse(words[1], "segment_type")?),
                        x_microns: parse(words[2], "x")?,
                        y_microns: parse(words[3], "y")?,
                        z_microns: parse(words[4], "z")?,
                        radius_microns: parse(words[5], "radius")?,
                        parent: parse(words[6], "parent")?,
                    }))
                } else {
                    Err(ParseError("Incorrect SWC line: too few words".to_string()))
                }
            }
        }
    }

    pub fn distance_to_segment_cm(&self, other: &SwcEntry) -> f32 {
        let dist_microns = (
            (self.x_microns - other.x_microns).powi(2) +
            (self.y_microns - other.y_microns).powi(2) +
            (self.z_microns - other.z_microns).powi(2)
        ).sqrt();
        dist_microns * 0.0001
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum SegmentType {
    Soma,
    Axon,
    Dendrite,
    ApicalDendrite,
    Custom
}

fn  parse<T>(s: &str, context: &'static str) -> Result<T, ParseError>
    where T: FromStr,
          <T as FromStr>::Err: ToString
{
    T::from_str(s).map_err(|e| ParseError(format!("{context}: {}", e.to_string())))
}

impl SegmentType {
    pub fn from_code(code: u8) -> Option<SegmentType> {
        match code {
            1 => Some(SegmentType::Soma),
            2 => Some(SegmentType::Axon),
            3 => Some(SegmentType::Dendrite),
            4 => Some(SegmentType::ApicalDendrite),
            5 => Some(SegmentType::Custom),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ParseError(String);

// pas, Ca_HVA, SKv3_1, SK_E2, Ca_LVAst, Ih, NaTs2_t, CaDynamics_E2
// TODO: implement the above
fn soma_membrane() -> Membrane {
    let v0 = MilliVolts(-88.0);
    Membrane {
        capacitance: FaradsPerSquareCm(1e-6),
        membrane_channels: vec![
            membrane::MembraneChannel {
                channel: channel::common_channels::giant_squid::K_CHANNEL
                    .build(&v0),
                siemens_per_square_cm: 36e-3,
            },
            membrane::MembraneChannel {
                channel: channel::common_channels::giant_squid::NA_CHANNEL
                    .build(&v0),
                siemens_per_square_cm: 120e-3,
            },
            membrane::MembraneChannel {
                channel: channel::common_channels::giant_squid::LEAK_CHANNEL
                    .build(&v0),
                siemens_per_square_cm: 3e-5,
            },
        ]
    }
}

// pas, Ca_HVA, SKv3_1, SK_E2, CaDynamics_E2, Nap_Et2, K_Pst, K_Tst, Ca_LVAst, NaTa_t
fn axon_membrane() -> Membrane {
    let v0 = MilliVolts(-88.0);
    Membrane {
        capacitance: FaradsPerSquareCm(1e-6),
        membrane_channels: vec![
            membrane::MembraneChannel {
                channel: channel::common_channels::giant_squid::K_CHANNEL
                    .build(&v0),
                siemens_per_square_cm: 36e-3,
            },
            membrane::MembraneChannel {
                channel: channel::common_channels::giant_squid::NA_CHANNEL
                    .build(&v0),
                siemens_per_square_cm: 120e-3,
            },
            membrane::MembraneChannel {
                channel: channel::common_channels::giant_squid::LEAK_CHANNEL
                    .build(&v0),
                siemens_per_square_cm: 0.3e-3,
            },
        ]
    }
}

// pas, Ca_HVA, SKv3_1, SK_E2, CaDynamics_E2, Nap_Et2, K_Pst, K_Tst, Ca_LVAst, NaTa_t
fn axon_initial_segment_membrane() -> Membrane {
    let v0 = MilliVolts(-80.0);
    Membrane {
        capacitance: FaradsPerSquareCm(1e-6),
        membrane_channels: vec![
            membrane::MembraneChannel {
                channel: channel::common_channels::giant_squid::K_CHANNEL
                    .build(&v0),
                siemens_per_square_cm: 36e-3,
            },
            membrane::MembraneChannel {
                channel: channel::common_channels::giant_squid::NA_CHANNEL
                    .build(&v0),
                siemens_per_square_cm: 120e-3,
            },
            membrane::MembraneChannel {
                channel: channel::common_channels::giant_squid::LEAK_CHANNEL
                    .build(&v0),
                siemens_per_square_cm: 0.3e-3,
            },
        ]
    }
}

// pas, Ih
fn basal_dendrite_membrane() -> Membrane {
    let v0 = MilliVolts(-88.0);
    Membrane {
        capacitance: FaradsPerSquareCm(2e-6),
        membrane_channels: vec![
            membrane::MembraneChannel {
                channel: channel::common_channels::giant_squid::LEAK_CHANNEL
                    .build(&v0),
                siemens_per_square_cm: 0.03e-3,
            },
            membrane::MembraneChannel {
                channel: channel::common_channels::rat_ca1::HCN_CHANNEL_DENDRITE
                    .build(&v0),
                siemens_per_square_cm: 0.08e-3,
            },
        ]
    }
}

// pas, Im, NaTs2_t, SKv3_1, Ih
fn apical_dendrite_membrane() -> Membrane {
    let v0 = MilliVolts(-88.0);
    Membrane {
        capacitance: FaradsPerSquareCm(2e-6),
        membrane_channels: vec![
            membrane::MembraneChannel {
                channel: channel::common_channels::giant_squid::LEAK_CHANNEL
                    .build(&v0),
                siemens_per_square_cm: 0.03e-3,
            },
            membrane::MembraneChannel {
                channel: channel::common_channels::rat_ca1::HCN_CHANNEL_DENDRITE
                    .build(&v0),
                siemens_per_square_cm: 0.08e-3,
            },
            membrane::MembraneChannel {
                channel: channel::common_channels::rat_thalamocortical::NA_TRANSIENT
                    .build(&v0),
                siemens_per_square_cm: 0.023
            },
            membrane::MembraneChannel {
                channel: channel::common_channels::rat_thalamocortical::K_SLOW
                    .build(&v0),
                siemens_per_square_cm: 0.040
            },
        ]
    }
}
