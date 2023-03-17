use bevy::prelude::*;
use std::str::FromStr;
use std::fs;
use std::path::Path;
use std::collections::{HashSet, HashMap};

use crate::dimension::{FaradsPerSquareCm, MilliVolts, Diameter};
use crate::neuron::membrane::{self, Membrane, MembraneChannel, MembraneVoltage, MembraneMaterials};
use crate::neuron::segment::{ecs::Segment, Geometry};
use crate::neuron::channel;

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
        mut commands: Commands,
        mut meshes: ResMut<Assets<Mesh>>,
        materials: Res<MembraneMaterials>
    ) -> Entity {
        let v0 = MilliVolts(-80.0);
        let microns_to_screen = 1.0;
        let entry_map = self.as_map();
        let soma = self.soma().expect("Soma should exist");
        let mut entities : HashMap<i32, Entity> = HashMap::new();
        let mut children_map = self.get_children();
        // let mut previous_segment = None;
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
                Some(SegmentType::Dendrite) => dendrite_membrane(),
                Some(SegmentType::ApicalDendrite) => apical_dendrite_membrane(),
                Some(SegmentType::Custom) => dendrite_membrane(),
                None => dendrite_membrane(),
            };
            let look_target = match entry_map.get(parent) {
                None => Vec3::ZERO,
                Some(p) => {
                    let p_x = (p.x_microns - soma.x_microns) * microns_to_screen;
                    let p_y = (p.y_microns - soma.y_microns) * microns_to_screen;
                    let p_z = (p.z_microns - soma.z_microns) * microns_to_screen;
                    Vec3::new(p_x, p_y, p_z)
                }
            };
            let entity = commands.spawn(
                (Segment,
                 membrane,
                 MembraneVoltage(v0.clone()),
                 Geometry {
                     diameter: Diameter(radius_cm * 2.0),
                     length: length_cm,
                 },
                 PbrBundle {
                     mesh: meshes.add(shape::Cylinder {
                         radius: radius_screen,
                         height: length_screen,
                         resolution: 12,
                         segments:4,
                     }.into()),
                     material: materials.from_voltage(&v0),
                     transform: Transform::from_xyz(x_screen, y_screen, z_screen).looking_at(
                         look_target,
                         Vec3::Y
                     ),
                     ..default()
                 }
                )
            ).id();
            println!("spawned swc entity: {x_screen} {y_screen} {z_screen}");
            entities.insert(id.clone(), entity);
        }
        entities.get(&1).expect("entity 1 should be the soma").clone()
    }

    pub fn simplify(self) -> Self {

        let children_map = self.get_children();
        let entries_map = self.as_map().clone();
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
        for (mut entry) in self.entries.clone().into_iter() {
            while !(should_keep.contains(&entry.parent) || entry.parent == -1) {
                println!("id {} has parent {} that is not should_keep", entry.id, entry.parent);
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

fn soma_membrane() -> Membrane {
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

fn axon_membrane() -> Membrane {
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

fn dendrite_membrane() -> Membrane {
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

fn apical_dendrite_membrane() -> Membrane {
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
