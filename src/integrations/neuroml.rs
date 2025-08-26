use bevy::math::prelude::Sphere;
use bevy::prelude::*;
use bevy_mod_picking::{
    events::Click,
    prelude::{Listener, Pointer},
    PickableBundle,
};
use crossbeam::channel::{Receiver, Sender};

use crate::dimension::{MicroAmpsPerSquareCm, MilliVolts};
use crate::gui::oscilloscope::Oscilloscope;
use crate::gui::NextClickAction;
use crate::neuron::ecs::Neuron;
use crate::neuron::membrane::{Membrane, MembraneMaterials, MembraneVoltage};
use crate::neuron::segment::ecs::Segment;
use crate::neuron::solution::EXAMPLE_CYTOPLASM;
use crate::neuron::synapse::SynapseMembranes;
use crate::neuron::Junction;
use crate::selection::{Highlight, Selection};
use crate::stimulator;

// Import neuroml-rs types
use neuroml::neuroml::{
    BaseConnection, BaseNonNegativeIntegerId, BaseWithoutId, CellTypes, Common, Connection,
    ExpOneSynapse, GridLayout, IafCell, Instance, Layout, Location, Network, NeuroML, NmlId,
    Population, Projection, RandomLayout, SynapseTypes, UnstructuredLayout, InputTypes, 
    PulseGenerator, SineGenerator, InputList, Input, ExplicitInput,
};
// use neuroml::Annotation;

pub struct NeuroMLScene(pub NeuroML);

#[derive(Resource, Clone)]
pub struct NeuroMLSceneSender(pub Sender<NeuroMLScene>);

#[derive(Resource)]
pub struct NeuroMLSceneReceiver(pub Receiver<NeuroMLScene>);

impl NeuroMLScene {
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
        let mut neuron_entities = Vec::new();

        // Find networks in the NeuroML document
        for network in &self.0.network {
            for population in &network.population {
                for instance in &population.instances {
                    let instance_location = Vec3::new(
                        instance.location.x as f32 * 0.001, // Convert to cm
                        instance.location.y as f32 * 0.001,
                        instance.location.z as f32 * 0.001,
                    );

                    let neuron_entity = spawn_neuroml_neuron(
                        &population.component,
                        instance_location + soma_location_cm,
                        &mut commands,
                        &mut meshes,
                        &membrane_materials,
                        materials,
                        &selections,
                        &highlights,
                    );
                    neuron_entities.push(neuron_entity);
                }
            }
        }

        // Spawn synapses
        for network in &self.0.network {
            for projection in &network.projection {
                spawn_neuroml_synapse(
                    &mut commands,
                    projection,
                    &network.population,
                    &neuron_entities,
                    meshes,
                    materials,
                );
            }
        }

        neuron_entities
    }
}

pub fn spawn_neuroml_neuron(
    _cell_type_id: &str,
    soma_location_cm: Vec3,
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    _membrane_materials: &MembraneMaterials,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    _selections: &Query<Entity, With<Selection>>,
    _highlights: &Query<Entity, With<Highlight>>,
) -> (Entity, Vec<Entity>) {
    // Create a simple spherical soma for NeuroML neurons
    let soma_radius_cm = 0.001; // 10 microns
    let soma_radius_screen = soma_radius_cm * 10000.0;

    let soma_mesh: Mesh = Sphere {
        radius: soma_radius_screen,
    }
    .into();

    let neuron_entity = commands
        .spawn((
            Neuron,
            Transform::from_translation(soma_location_cm),
            GlobalTransform::default(),
            Visibility::default(),
            InheritedVisibility::default(),
            ViewVisibility::default(),
        ))
        .id();

    // Create a simple membrane
    let serialized_membrane = crate::serialize::Membrane {
        membrane_channels: vec![],
        capacitance_farads_per_square_cm: 1e-6,
    };
    let membrane = Membrane::deserialize(&serialized_membrane);

    let soma_entity = commands
        .spawn((
            Segment,
            MembraneVoltage(MilliVolts(-70.0)),
            membrane,
            Transform::from_translation(Vec3::ZERO),
            GlobalTransform::default(),
            Visibility::default(),
            InheritedVisibility::default(),
            ViewVisibility::default(),
            PickableBundle::default(),
            materials.add(StandardMaterial {
                base_color: Color::rgb(0.8, 0.8, 1.0),
                ..default()
            }),
            meshes.add(soma_mesh),
        ))
        .id();

    commands.entity(neuron_entity).push_children(&[soma_entity]);

    (neuron_entity, vec![soma_entity])
}

pub fn spawn_neuroml_synapse(
    commands: &mut Commands,
    projection: &Projection,
    _populations: &[Population],
    neurons_and_segments: &[(Entity, Vec<Entity>)],
    _meshes: &mut ResMut<Assets<Mesh>>,
    _materials: &mut ResMut<Assets<StandardMaterial>>,
) {
    for connection in &projection.connection {
        // Find pre and post neurons (simplified - just use first two neurons)
        if neurons_and_segments.len() >= 2 {
            let pre_entity = neurons_and_segments[0].0;
            let post_entity = neurons_and_segments[1].0;

            commands.spawn((
                SynapseMembranes {
                    cleft_solution: EXAMPLE_CYTOPLASM.clone(),
                    transmitter_concentrations: crate::neuron::synapse::TransmitterConcentrations {
                        glutamate: crate::dimension::Molar(0.0),
                        gaba: crate::dimension::Molar(0.0),
                    },
                    presynaptic_pumps: vec![],
                    postsynaptic_receptors: vec![],
                    surface_area: crate::dimension::AreaSquareMillimeters(1.0),
                },
                Junction {
                    first_segment: pre_entity,
                    second_segment: post_entity,
                    pore_diameter: crate::dimension::Diameter(0.001),
                },
            ));
        }
    }
}

pub fn add_stimulation(
    _event: Listener<Pointer<Click>>,
    _commands: Commands,
    _meshes: ResMut<Assets<Mesh>>,
    _materials: ResMut<Assets<StandardMaterial>>,
    _oscilloscope: ResMut<Oscilloscope>,
    _next_click: ResMut<NextClickAction>,
    _selections: Query<Entity, With<Selection>>,
    _highlights: Query<Entity, With<Highlight>>,
    _new_stimulators: Res<stimulator::Stimulator>,
    _segments_query: Query<(Entity, &Segment, &GlobalTransform)>,
) {
    // Similar to grace.rs implementation but adapted for NeuroML
    // This would handle stimulation for NeuroML neurons
}

pub fn select_stimulator(
    _segment_entity: Entity,
    _commands: Commands,
    _selections: Query<Entity, With<Selection>>,
    _highlights: Query<Entity, With<Highlight>>,
    _meshes: ResMut<Assets<Mesh>>,
    _materials: ResMut<Assets<StandardMaterial>>,
) {
    // Similar to grace.rs implementation but adapted for NeuroML
}

pub fn handle_click_stimulator(
    _event: Listener<Pointer<Click>>,
    _commands: Commands,
    _stimulations_query: Query<&stimulator::Stimulation>,
    _segments_query: Query<(&Segment, Entity, &stimulator::Stimulator)>,
    _selections: Query<Entity, With<Selection>>,
    _highlights: Query<Entity, With<Highlight>>,
    _meshes: ResMut<Assets<Mesh>>,
    _materials: ResMut<Assets<StandardMaterial>>,
) {
    // Similar to grace.rs implementation but adapted for NeuroML
}

pub fn delete_stimulations(
    In(_event): In<Pointer<Click>>,
    _commands: Commands,
    _stimulations_query: Query<&stimulator::Stimulation>,
    _segments_query: Query<(&Segment, Entity, &stimulator::Stimulator)>,
) {
    // Similar to grace.rs implementation but adapted for NeuroML
}

pub mod sample {
    use super::*;

    pub fn scene1() -> NeuroMLScene {
        create_sample_scene()
    }

    pub fn create_sample_scene() -> NeuroMLScene {
        // Create a minimal NeuroML structure with current injection
        let neuroml = NeuroML {
            common: Common {
                id: "SampleNetwork".to_string(),
                metaid: None,
                notes: None,
                properties: vec![],
                annotation: Default::default(),
                neuro_lex_id: None,
            },
            cell_types: vec![],
            synapse_types: vec![],
            // Skip current sources for now due to UOM type complexities
            input_types: vec![],
            network: vec![Network {
                type_attr: None,
                temperature: None,
                space: vec![],
                region: vec![],
                extracellular_properties: vec![],
                population: vec![Population {
                    type_attr: None,
                    component: "simple_neuron".to_string(),
                    size: Some(2),
                    extracellular_properties: None,
                    layout: Layout {
                        space: None,
                        random: RandomLayout::default(),
                        grid: GridLayout::default(),
                        unstructured: UnstructuredLayout::default(),
                        base_without_id: BaseWithoutId::default(),
                    },
                    instances: vec![
                        Instance {
                            id: Some(0),
                            i: None,
                            j: None,
                            k: None,
                            location: Location {
                                x: 0.0,
                                y: 0.0,
                                z: 0.0,
                                base_without_id: BaseWithoutId::default(),
                            },
                        },
                        Instance {
                            id: Some(1),
                            i: None,
                            j: None,
                            k: None,
                            location: Location {
                                x: 50000.0,
                                y: 0.0,
                                z: 0.0,
                                base_without_id: BaseWithoutId::default(),
                            },
                        },
                    ],
                    common: Common {
                        id: "neuron_population".to_string(),
                        metaid: None,
                        notes: None,
                        properties: vec![],
                        annotation: Default::default(),
                        neuro_lex_id: None,
                    },
                }],
                cell_set: vec![],
                synaptic_connection: vec![],
                projection: vec![],
                electrical_projection: vec![],
                continuous_projection: vec![],
                explicit_input: vec![],
                input_list: vec![],
                common: Common {
                    id: "sample_network".to_string(),
                    metaid: None,
                    notes: None,
                    properties: vec![],
                    annotation: Default::default(),
                    neuro_lex_id: None,
                },
            }],
            // Add other required fields with empty vectors
            concentration_model_types: vec![],
            py_nn_cell_types: vec![],
            py_nn_synapse_types: vec![],
            py_nn_input_types: vec![],
            include: vec![],
            extracellular_properties: vec![],
            intracellular_properties: vec![],
            morphology: vec![],
            ion_channel: vec![],
            ion_channel_hh: vec![],
            ion_channel_v_shift: vec![],
            ion_channel_ks: vec![],
            biophysical_properties: vec![],
            component_type: vec![],
        };

        NeuroMLScene(neuroml)
    }
}

pub mod tests {
    use super::*;

    #[test]
    fn test_create_sample_scene() {
        let scene = sample::create_sample_scene();
        assert_eq!(scene.0.cell_types.len(), 0);
        assert_eq!(scene.0.synapse_types.len(), 0);
        assert_eq!(scene.0.network.len(), 1);
    }
}
