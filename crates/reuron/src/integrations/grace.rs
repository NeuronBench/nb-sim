use bevy::prelude::*;
use bevy_mod_picking::PickableBundle;
use std::collections::HashMap;

use crate::dimension::{MilliVolts, Diameter, MicroAmpsPerSquareCm};
use crate::neuron::membrane::{self, Membrane, MembraneVoltage, MembraneMaterials};
use crate::neuron::solution::EXAMPLE_CYTOPLASM;
use crate::neuron::segment::{ecs::Segment, ecs::InputCurrent, Geometry};
use crate::serialize;
use crate::neuron::ecs::Neuron;

pub struct GraceNeuron( pub serialize::Neuron );

impl GraceNeuron {

    pub fn spawn(
        &self,
        soma_location_cm: Vec3,
        mut commands: &mut Commands,
        mut meshes: &mut ResMut<Assets<Mesh>>,
        materials: &mut Res<MembraneMaterials>,
    ) -> Entity {
        let v0 = MilliVolts(-88.0);
        let microns_to_screen = 1.0;
        let entry_map = self.segments_as_map();
        let soma = self.soma().expect("should have soma");
        let mut entities_and_parents : HashMap<i32, (Entity, i32, Diameter)> = HashMap::new();
        let mut children_map = self.get_children();
        let neuron_entity = commands.spawn(
            (Neuron,
             Transform::from_translation(soma_location_cm),
             GlobalTransform::default(),
             Visibility::default(),
             ComputedVisibility::default(),
            )).id();
        for segment in self.0.segments.iter() {
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
                (_, Some(parent_segment)) => distance_to_segment_cm(segment, parent_segment),
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

            dbg!(segment.type_);
            let membrane_serialized = self.0.membranes.get(segment.type_ - 1).expect("type should be a valid index into membranes");
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
                     material: materials.from_voltage(&v0),
                     transform: transform,
                     ..default()
                 },
                 PickableBundle::default(),
                )
            ).id();
            commands.entity(neuron_entity).push_children(&[segment_entity]);
            entities_and_parents.insert(id.clone(), (segment_entity, segment.parent, Diameter(1.0)));
        }
        neuron_entity
    }

    pub fn soma(&self) -> Option<&serialize::Segment> {
        self.0.segments.iter().find(|s| s.parent == -1)
    }

    /// Determine each segment's children.
    pub fn get_children(&self) -> HashMap<i32, Vec<i32>> {
        let mut children_map = HashMap::new();
        for segment in self.0.segments.iter() {
            // Modify self's parent, inserting self as a child.
            children_map.entry(segment.parent)
                .and_modify(|mut children: &mut Vec<i32>| (children).push(segment.id))
                .or_insert(vec![segment.id]);
        }
        children_map
    }

    /// Index the entries by id.
    pub fn segments_as_map(&self) -> HashMap<i32, &serialize::Segment> {
        self.0.segments.iter().map(|segment| (segment.id, segment)).collect()
    }

}

pub fn distance_to_segment_cm(source: &serialize::Segment, dest: &serialize::Segment) -> f32 {
    let dist_microns = (
            (source.x - dest.x).powi(2) +
            (source.y - dest.y).powi(2) +
            (source.z - dest.z).powi(2)
    ).sqrt();
    dist_microns + 0.0001
}



pub mod sample {
    use std::include_str;
    use crate::serialize;
    pub fn neuron() -> serialize::Neuron {
        let s = include_str!("../../sample_data/swc_neuron.json");
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
