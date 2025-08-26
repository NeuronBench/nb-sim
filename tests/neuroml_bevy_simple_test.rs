use bevy::prelude::*;
use nb_sim::integrations::neuroml::sample;
use nb_sim::neuron::membrane::{MembraneMaterials, MembraneVoltage};
use nb_sim::neuron::segment::ecs::Segment;
use nb_sim::dimension::Timestamp;
use nb_sim::stimulator::{Stimulator, Stimulation};
use nb_sim::selection::{Selection, Highlight};

#[test]
fn test_neuroml_bevy_ecs_integration() {
    println!("🧪 Testing NeuroML integration with Bevy ECS components...");
    
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(AssetPlugin::default());
    
    let world = &mut app.world;
    
    // Initialize resources
    let membrane_materials = MembraneMaterials::from_world(world);
    world.insert_resource(membrane_materials);
    
    // Create and spawn NeuroML scene
    let scene = sample::create_sample_scene();
    
    let mut system_state: bevy::ecs::system::SystemState<(
        Commands,
        ResMut<Assets<Mesh>>,
        Res<MembraneMaterials>,
        ResMut<Assets<StandardMaterial>>,
        Query<Entity, With<Selection>>,
        Query<Entity, With<Highlight>>,
    )> = bevy::ecs::system::SystemState::new(world);

    {
        let (commands, mut meshes, membrane_materials, mut materials, selections, highlights) = 
            system_state.get_mut(world);
        
        let neuron_entities = scene.spawn(
            Vec3::ZERO, 
            commands, 
            &mut meshes, 
            membrane_materials, 
            &mut materials, 
            selections, 
            highlights
        );
        
        println!("  🧬 Spawned {} neuron entities", neuron_entities.len());
    }
    system_state.apply(world);
    drop(system_state);
    
    // Test ECS component structure
    {
        let mut segment_query = world.query::<(Entity, &Segment, &MembraneVoltage)>();
        let segments: Vec<_> = segment_query.iter(world).collect();
        println!("  🧩 Found {} segments with membrane voltage", segments.len());
        assert_eq!(segments.len(), 2, "Should have 2 neuron segments");
        
        // Check initial membrane voltages
        for (entity, _segment, voltage) in &segments {
            println!("    Segment {:?}: {:.2}mV", entity, voltage.0.0);
            assert!(voltage.0.0 < -50.0 && voltage.0.0 > -80.0, 
                   "Membrane voltage should be around resting potential");
        }
    }
    
    {
        let mut stimulator_query = world.query::<(Entity, &Stimulator)>();
        let stimulators: Vec<_> = stimulator_query.iter(world).collect();
        println!("  ⚡ Found {} stimulators", stimulators.len());
        assert_eq!(stimulators.len(), 2, "Should have 2 stimulators");
    }
    
    {
        let mut stimulation_query = world.query::<&Stimulation>();
        let stimulations: Vec<_> = stimulation_query.iter(world).collect();
        println!("  🔗 Found {} stimulation links", stimulations.len());
        assert_eq!(stimulations.len(), 2, "Should have 2 stimulation links");
    }
    
    println!("✅ NeuroML Bevy ECS integration test passed!");
}

#[test] 
fn test_neuroml_stimulator_current_generation() {
    println!("🧪 Testing NeuroML stimulator current generation in Bevy context...");
    
    let mut app = App::new();
    app.add_plugins(MinimalPlugins).add_plugins(AssetPlugin::default());
    
    let world = &mut app.world;
    let membrane_materials = MembraneMaterials::from_world(world);
    world.insert_resource(membrane_materials);
    
    // Create NeuroML scene
    let scene = sample::create_sample_scene();
    
    let mut system_state: bevy::ecs::system::SystemState<(
        Commands,
        ResMut<Assets<Mesh>>,
        Res<MembraneMaterials>,
        ResMut<Assets<StandardMaterial>>,
        Query<Entity, With<Selection>>,
        Query<Entity, With<Highlight>>,
    )> = bevy::ecs::system::SystemState::new(world);

    {
        let (commands, mut meshes, membrane_materials, mut materials, selections, highlights) = 
            system_state.get_mut(world);
        scene.spawn(Vec3::ZERO, commands, &mut meshes, membrane_materials, &mut materials, selections, highlights);
    }
    system_state.apply(world);
    drop(system_state);
    
    // Test stimulator current generation at different times
    let test_cases = [
        (0.05, "Before both pulses", vec![0.0, 0.0]),
        (0.075, "During pulseGen2 only", vec![0.0, 0.3]),  // Only second stimulator active
        (0.15, "During both pulses", vec![0.5, 0.3]),      // Both stimulators active
        (0.32, "During pulseGen1 only", vec![0.5, 0.0]),   // Only first stimulator active  
        (0.4, "After both pulses", vec![0.0, 0.0]),        // Both stimulators off
    ];
    
    println!("  📊 Testing current generation at different time points:");
    
    for (time_s, description, expected_currents) in test_cases {
        println!("    t={:.0}ms ({})", time_s * 1000.0, description);
        
        let timestamp = Timestamp(time_s);
        let mut stimulator_query = world.query::<&Stimulator>();
        let stimulators: Vec<&Stimulator> = stimulator_query.iter(world).collect();
        
        for (i, (stimulator, &expected)) in stimulators.iter().zip(expected_currents.iter()).enumerate() {
            let current = stimulator.current(timestamp.clone());
            println!("      Stimulator {}: {:.1} µA/cm² (expected {:.1})", i, current.0, expected);
            
            // Allow small tolerance for floating point comparison
            assert!((current.0 - expected).abs() < 0.1, 
                   "Stimulator {} current should be ~{} µA/cm² at t={}s", i, expected, time_s);
        }
    }
    
    println!("✅ NeuroML stimulator current generation test passed!");
}

#[test]
fn test_neuroml_stimulation_entity_links() {
    println!("🧪 Testing NeuroML stimulation entity links...");
    
    let mut app = App::new();
    app.add_plugins(MinimalPlugins).add_plugins(AssetPlugin::default());
    
    let world = &mut app.world;
    let membrane_materials = MembraneMaterials::from_world(world);
    world.insert_resource(membrane_materials);
    
    let scene = sample::create_sample_scene();
    
    let mut system_state: bevy::ecs::system::SystemState<(
        Commands,
        ResMut<Assets<Mesh>>,
        Res<MembraneMaterials>,
        ResMut<Assets<StandardMaterial>>,
        Query<Entity, With<Selection>>,
        Query<Entity, With<Highlight>>,
    )> = bevy::ecs::system::SystemState::new(world);

    {
        let (commands, mut meshes, membrane_materials, mut materials, selections, highlights) = 
            system_state.get_mut(world);
        scene.spawn(Vec3::ZERO, commands, &mut meshes, membrane_materials, &mut materials, selections, highlights);
    }
    system_state.apply(world);
    drop(system_state);
    
    // Verify stimulation links connect stimulators to segments correctly
    {
        let mut stimulation_query = world.query::<&Stimulation>();
        let stimulations: Vec<&Stimulation> = stimulation_query.iter(world).collect();
        
        println!("  🔍 Examining {} stimulation links:", stimulations.len());
        
        for (i, stimulation) in stimulations.iter().enumerate() {
            let segment_entity = stimulation.stimulation_segment;
            
            // Verify the linked segment exists and has required components
            let has_segment = world.get::<Segment>(segment_entity).is_some();
            let has_voltage = world.get::<MembraneVoltage>(segment_entity).is_some();
            let has_stimulator = world.get::<Stimulator>(segment_entity).is_some();
            
            assert!(has_segment, "Stimulation {} should link to valid segment", i);
            assert!(has_voltage, "Linked segment should have membrane voltage");
            assert!(has_stimulator, "Linked segment should have stimulator component");
            
            println!("    ✓ Stimulation {}: segment {:?} has all required components", i, segment_entity);
            
            // Test that stimulator on this segment actually produces current
            if let Some(stimulator) = world.get::<Stimulator>(segment_entity) {
                let current_during = stimulator.current(Timestamp(0.15)); // During both pulses
                println!("      Current at t=150ms: {:.2} µA/cm²", current_during.0);
                assert!(current_during.0 > 0.0, "Stimulator should produce positive current");
            }
        }
    }
    
    println!("✅ NeuroML stimulation entity links test passed!");
}

#[test] 
fn test_neuroml_differential_current_injection() {
    println!("🧪 Testing differential current injection between NeuroML neurons...");
    
    let mut app = App::new();
    app.add_plugins(MinimalPlugins).add_plugins(AssetPlugin::default());
    
    let world = &mut app.world;
    let membrane_materials = MembraneMaterials::from_world(world);
    world.insert_resource(membrane_materials);
    
    let scene = sample::create_sample_scene();
    
    let mut system_state: bevy::ecs::system::SystemState<(
        Commands,
        ResMut<Assets<Mesh>>,
        Res<MembraneMaterials>,
        ResMut<Assets<StandardMaterial>>,
        Query<Entity, With<Selection>>,
        Query<Entity, With<Highlight>>,
    )> = bevy::ecs::system::SystemState::new(world);

    {
        let (commands, mut meshes, membrane_materials, mut materials, selections, highlights) = 
            system_state.get_mut(world);
        scene.spawn(Vec3::ZERO, commands, &mut meshes, membrane_materials, &mut materials, selections, highlights);
    }
    system_state.apply(world);
    drop(system_state);
    
    println!("  📊 Verifying different current profiles for each neuron:");
    
    // Collect stimulators and test their different current profiles
    {
        let mut stimulator_query = world.query::<&Stimulator>();
        let stimulators: Vec<&Stimulator> = stimulator_query.iter(world).collect();
        assert_eq!(stimulators.len(), 2, "Should have 2 stimulators");
        
        // Test at a time when both should be active (t=150ms)
        let test_time = Timestamp(0.15);
        let currents: Vec<f32> = stimulators.iter()
            .map(|s| s.current(test_time.clone()).0)
            .collect();
        
        println!("    Neuron 0 current: {:.2} µA/cm² (from pulseGen1: 500nA)", currents[0]);
        println!("    Neuron 1 current: {:.2} µA/cm² (from pulseGen2: 300nA)", currents[1]);
        
        // Verify the currents match expected conversion from NeuroML
        // 500nA → 0.5µA/cm², 300nA → 0.3µA/cm²
        assert!((currents[0] - 0.5).abs() < 0.1, "Neuron 0 should get ~0.5µA/cm²");
        assert!((currents[1] - 0.3).abs() < 0.1, "Neuron 1 should get ~0.3µA/cm²");
        
        // Verify they're different
        assert!((currents[0] - currents[1]).abs() > 0.1, "Neurons should have different current levels");
    }
    
    println!("✅ NeuroML differential current injection test passed!");
    println!("🎉 NeuroML current injection is fully integrated with Bevy ECS!");
}