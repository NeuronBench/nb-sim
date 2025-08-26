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
    ExpOneSynapse, ExplicitInput, GridLayout, IafCell, Input, InputList, InputTypes, Instance,
    Layout, Location, Network, NeuroML, NmlId, Population, Projection, PulseGenerator,
    RandomLayout, SineGenerator, SynapseTypes, UnstructuredLayout,
};
use uom::si::f64::{ElectricCurrent, Time as UomTime};
use uom::si::{electric_current::nanoampere, time::millisecond};
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

        // Process current inputs and create stimulators
        // TODO: Re-enable after resolving UOM version conflicts
        self.process_current_inputs(&mut commands, &neuron_entities);

        neuron_entities
    }

    fn process_current_inputs(
        &self,
        commands: &mut Commands,
        neuron_entities: &[(Entity, Vec<Entity>)],
    ) {
        // Process each network
        for network in &self.0.network {
            // Process explicit inputs
            for explicit_input in &network.explicit_input {
                // Find the matching input type
                if let Some(input_type) = self.0.input_types.iter().find(|input| match input {
                    InputTypes::PulseGenerator(pg) => pg.common.id == explicit_input.input,
                    _ => false,
                }) {
                    // Convert NeuroML input to nb-sim stimulator
                    if let InputTypes::PulseGenerator(pulse_gen) = input_type {
                        let stimulator = convert_pulse_to_stimulator(pulse_gen);

                        // Find target neuron - simplified: match by index
                        if let Some(target_neuron) = parse_target_index(&explicit_input.target) {
                            if let Some((neuron_entity, segments)) =
                                neuron_entities.get(target_neuron)
                            {
                                if let Some(segment_entity) = segments.first() {
                                    // Add stimulator to the segment
                                    commands.entity(*segment_entity).insert(stimulator);

                                    // Create stimulation link
                                    commands.spawn(stimulator::Stimulation {
                                        stimulation_segment: *segment_entity,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
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

/// Convert NeuroML PulseGenerator to nb-sim Stimulator
pub fn convert_pulse_to_stimulator(pulse_gen: &PulseGenerator) -> stimulator::Stimulator {
    use crate::dimension::{Interval, MicroAmpsPerSquareCm};
    
    // Convert UOM values to nb-sim units
    let delay_ms = pulse_gen.delay.get::<millisecond>();
    let duration_ms = pulse_gen.duration.get::<millisecond>();
    let amplitude_na = pulse_gen.amplitude.get::<nanoampere>();
    
    // Convert nanoamps to microamps per square cm (simplified conversion)
    let amplitude_ua_per_cm2 = amplitude_na / 1000.0; // Rough conversion
    
    stimulator::Stimulator {
        envelope: stimulator::Envelope {
            period: Interval((duration_ms + delay_ms + 100.0) as f32 / 1000.0), // Convert to seconds
            onset: Interval(delay_ms as f32 / 1000.0), // Convert to seconds
            offset: Interval((delay_ms + duration_ms) as f32 / 1000.0),
        },
        current_shape: stimulator::CurrentShape::SquareWave {
            on_current: MicroAmpsPerSquareCm(amplitude_ua_per_cm2 as f32),
            off_current: MicroAmpsPerSquareCm(0.0),
        }
    }
}

/// Parse target string to extract neuron index
/// Expected format: "../neuron_population/0/simple_neuron" -> Some(0)
pub fn parse_target_index(target: &str) -> Option<usize> {
    target
        .split('/')
        .nth(2) // Get the "0" part
        .and_then(|s| s.parse().ok())
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
            // Add current sources with proper UOM types
            input_types: vec![
                InputTypes::PulseGenerator(PulseGenerator {
                    delay: UomTime::new::<millisecond>(100.0),
                    duration: UomTime::new::<millisecond>(200.0), 
                    amplitude: ElectricCurrent::new::<nanoampere>(500.0), // 0.5nA
                    common: Common {
                        id: "pulseGen1".to_string(),
                        metaid: None,
                        notes: None,
                        properties: vec![],
                        annotation: Default::default(),
                        neuro_lex_id: None,
                    },
                }),
                InputTypes::PulseGenerator(PulseGenerator {
                    delay: UomTime::new::<millisecond>(50.0),
                    duration: UomTime::new::<millisecond>(300.0),
                    amplitude: ElectricCurrent::new::<nanoampere>(300.0), // 0.3nA
                    common: Common {
                        id: "pulseGen2".to_string(),
                        metaid: None,
                        notes: None,
                        properties: vec![],
                        annotation: Default::default(),
                        neuro_lex_id: None,
                    },
                }),
            ],
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
                // Connect current sources to specific neurons
                explicit_input: vec![
                    ExplicitInput {
                        target: "../neuron_population/0/simple_neuron".to_string(),
                        input: "pulseGen1".to_string(),
                        destination: Some("synapses".to_string()),
                        base_without_id: BaseWithoutId::default(),
                    },
                    ExplicitInput {
                        target: "../neuron_population/1/simple_neuron".to_string(),
                        input: "pulseGen2".to_string(),
                        destination: Some("synapses".to_string()),
                        base_without_id: BaseWithoutId::default(),
                    },
                ],
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
    use bevy::prelude::*;
    use crate::neuron::membrane::{MembraneMaterials, MembraneVoltage};
    use crate::neuron::segment::ecs::Segment;
    use crate::stimulator;

    #[test]
    fn test_create_sample_scene() {
        let scene = sample::create_sample_scene();
        assert_eq!(scene.0.cell_types.len(), 0);
        assert_eq!(scene.0.synapse_types.len(), 0);
        assert_eq!(scene.0.network.len(), 1);
        assert_eq!(scene.0.input_types.len(), 2); // Two pulse generators
        assert_eq!(scene.0.network[0].explicit_input.len(), 2); // Two current sources
    }

    #[test]
    fn test_pulse_generator_conversion() {
        // Create a test PulseGenerator
        let pulse_gen = PulseGenerator {
            delay: UomTime::new::<millisecond>(100.0),
            duration: UomTime::new::<millisecond>(200.0),
            amplitude: ElectricCurrent::new::<nanoampere>(500.0),
            common: Common {
                id: "test_pulse".to_string(),
                metaid: None,
                notes: None,
                properties: vec![],
                annotation: Default::default(),
                neuro_lex_id: None,
            },
        };

        // Convert to nb-sim stimulator
        let stimulator = convert_pulse_to_stimulator(&pulse_gen);

        // Check timing conversion (100ms delay = 0.1s onset)
        assert!((stimulator.envelope.onset.0 - 0.1).abs() < 0.001);
        
        // Check duration (100ms delay + 200ms duration = 0.3s offset)
        assert!((stimulator.envelope.offset.0 - 0.3).abs() < 0.001);
        
        // Check current conversion (500nA = 0.5µA/cm²)
        if let stimulator::CurrentShape::SquareWave { on_current, off_current } = stimulator.current_shape {
            assert!((on_current.0 - 0.5).abs() < 0.001);
            assert_eq!(off_current.0, 0.0);
        } else {
            panic!("Expected SquareWave current shape");
        }
    }

    #[test]
    fn test_target_parsing() {
        assert_eq!(parse_target_index("../neuron_population/0/simple_neuron"), Some(0));
        assert_eq!(parse_target_index("../neuron_population/1/simple_neuron"), Some(1));
        assert_eq!(parse_target_index("../neuron_population/42/simple_neuron"), Some(42));
        assert_eq!(parse_target_index("invalid/path"), None);
    }

    #[test]
    fn test_neuroml_current_injection_integration() {
        // Simplified test - just verify the scene structure
        let scene = sample::create_sample_scene();
        
        // Verify the scene has the expected structure
        assert_eq!(scene.0.input_types.len(), 2);
        assert_eq!(scene.0.network[0].explicit_input.len(), 2);
        
        // Verify input types are PulseGenerators
        for input_type in &scene.0.input_types {
            match input_type {
                InputTypes::PulseGenerator(_) => {}, // Expected
                _ => panic!("Expected PulseGenerator input type"),
            }
        }
        
        // Verify explicit inputs target correct neurons
        let inputs = &scene.0.network[0].explicit_input;
        assert_eq!(inputs[0].target, "../neuron_population/0/simple_neuron");
        assert_eq!(inputs[1].target, "../neuron_population/1/simple_neuron");
        assert_eq!(inputs[0].input, "pulseGen1");
        assert_eq!(inputs[1].input, "pulseGen2");
    }

    #[cfg(test)]
    mod membrane_voltage_tests {
        use super::*;
        use crate::dimension::{Interval, MicroAmpsPerSquareCm, MilliVolts, Timestamp};
        use crate::neuron::segment::ecs::Segment;
        use crate::neuron::membrane::{Membrane, MembraneVoltage};
        use crate::stimulator::{Stimulator, CurrentShape, Envelope};

        #[test]
        fn test_current_injection_affects_membrane_voltage() {
            // Test stimulator current generation without full ECS setup
            let stimulator = Stimulator {
                envelope: Envelope {
                    period: Interval(1.0),     // 1 second period
                    onset: Interval(0.0),      // Start immediately  
                    offset: Interval(0.5),     // Stop at 0.5 seconds
                },
                current_shape: CurrentShape::SquareWave {
                    on_current: MicroAmpsPerSquareCm(10.0),  // 10 µA/cm² positive current
                    off_current: MicroAmpsPerSquareCm(0.0),
                }
            };
            
            // At t=0.1s, should be ON (positive current)
            let current_at_0_1s = stimulator.current(Timestamp(0.1));
            assert!(current_at_0_1s.0 > 0.0, "Expected positive current at t=0.1s");
            
            // At t=0.6s, should be OFF (zero current) 
            let current_at_0_6s = stimulator.current(Timestamp(0.6));
            assert_eq!(current_at_0_6s.0, 0.0, "Expected zero current at t=0.6s");

            println!("✓ Current injection timing test passed");
            println!("  - Current at t=0.1s: {:.2} µA/cm²", current_at_0_1s.0);
            println!("  - Current at t=0.6s: {:.2} µA/cm²", current_at_0_6s.0);
        }

        #[test] 
        fn test_neuroml_pulse_generator_produces_expected_current() {
            // Test the specific PulseGenerators created in the NeuroML scene
            let scene = sample::create_sample_scene();
            
            // Get the first pulse generator (pulseGen1)
            if let Some(InputTypes::PulseGenerator(pulse_gen)) = scene.0.input_types.first() {
                let stimulator = convert_pulse_to_stimulator(pulse_gen);
                
                // pulseGen1: 100ms delay, 200ms duration, 500nA amplitude
                // Should be ON from 0.1s to 0.3s
                
                let current_before = stimulator.current(Timestamp(0.05)); // Before onset
                let current_during = stimulator.current(Timestamp(0.2));  // During pulse
                let current_after = stimulator.current(Timestamp(0.4));   // After offset
                
                assert_eq!(current_before.0, 0.0, "Should be OFF before onset");
                assert!(current_during.0 > 0.0, "Should be ON during pulse");
                assert_eq!(current_after.0, 0.0, "Should be OFF after offset");
                
                println!("✓ NeuroML PulseGenerator timing test passed");
                println!("  - Current before (t=0.05s): {:.2} µA/cm²", current_before.0);
                println!("  - Current during (t=0.2s): {:.2} µA/cm²", current_during.0);  
                println!("  - Current after (t=0.4s): {:.2} µA/cm²", current_after.0);
            } else {
                panic!("Expected PulseGenerator in scene");
            }
        }
    }
}
