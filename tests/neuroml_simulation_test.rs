use bevy::prelude::*;
use nb_sim::integrations::neuroml::sample;
use nb_sim::neuron::membrane::{MembraneMaterials, MembraneVoltage};
use nb_sim::neuron::segment::ecs::Segment;
use nb_sim::dimension::{Timestamp, SimulationStepSeconds};
use nb_sim::plugin::NbSimPlugin;
use nb_sim::stimulator::{Stimulator, Stimulation};
use nb_sim::selection::{Selection, Highlight};

#[test]
fn test_neuroml_current_injection_affects_membrane_voltage() {
    println!("🧪 Testing NeuroML current injection with Bevy simulation...");
    
    // Create a minimal Bevy app with the simulation systems
    let mut app = App::new();
    
    // Add minimal plugins needed for the test
    app.add_plugins(MinimalPlugins)
        .add_plugins(AssetPlugin::default())
        .add_plugins(NbSimPlugin)
        .insert_resource(SimulationStepSeconds(5e-7)) // 0.5 microsecond timesteps
        .insert_resource(Timestamp(0.0));

    // Create the NeuroML scene with current injection
    let scene = sample::create_sample_scene();
    
    let world = &mut app.world;
    
    // Initialize MembraneMaterials from world
    let membrane_materials = MembraneMaterials::from_world(world);
    world.insert_resource(membrane_materials);
    
    // Set up system state for spawning
    let mut system_state: bevy::ecs::system::SystemState<(
        Commands,
        ResMut<Assets<Mesh>>,
        Res<MembraneMaterials>,
        ResMut<Assets<StandardMaterial>>,
        Query<Entity, With<Selection>>,
        Query<Entity, With<Highlight>>,
    )> = bevy::ecs::system::SystemState::new(world);

    let (
        mut commands,
        mut meshes,
        membrane_materials,
        mut materials,
        selections,
        highlights,
    ) = system_state.get_mut(world);

    // Spawn the NeuroML scene (includes neurons and stimulators)
    let neuron_entities = scene.spawn(
        Vec3::ZERO,
        commands,
        &mut meshes,
        membrane_materials,
        &mut materials,
        selections,
        highlights,
    );

    system_state.apply(world);
    drop(system_state);

    println!("  📋 Spawned {} neurons with current injection", neuron_entities.len());

    // Record initial membrane voltages
    let mut initial_voltages = Vec::new();
    {
        let voltage_query = world.query::<&MembraneVoltage>();
        for entity in voltage_query.iter(world) {
            initial_voltages.push(entity.0.0);
        }
    }
    
    println!("  🔋 Initial voltages: {:?} mV", initial_voltages);
    
    // Simulate for enough time to see current injection effects
    // We need to reach t=100ms for pulseGen1 to turn on
    let target_time = 0.15; // 150ms - during both current pulses
    let dt = 5e-7; // 0.5 microsecond timestep
    let steps_needed = (target_time / dt) as usize;
    
    println!("  ⏱️  Running {} simulation steps to reach t={}s", steps_needed, target_time);
    
    // Run simulation steps
    for step in 0..steps_needed {
        // Update simulation time
        {
            let mut timestamp = world.resource_mut::<Timestamp>();
            timestamp.0 = step as f32 * dt;
        }
        
        // Run one simulation update
        app.update();
        
        // Print progress every 50,000 steps (25ms of simulation time)
        if step > 0 && step % 50_000 == 0 {
            let sim_time_ms = step as f32 * dt * 1000.0;
            println!("    Step {}: t={:.1}ms", step, sim_time_ms);
            
            // Check a few voltages during simulation
            if step % 100_000 == 0 {
                let voltage_query = world.query::<&MembraneVoltage>();
                let current_voltages: Vec<f32> = voltage_query.iter(world).map(|v| v.0.0).collect();
                println!("      Current voltages: {:?} mV", current_voltages);
            }
        }
    }
    
    // Record final membrane voltages after current injection
    let mut final_voltages = Vec::new();
    {
        let voltage_query = world.query::<&MembraneVoltage>();
        for entity in voltage_query.iter(world) {
            final_voltages.push(entity.0.0);
        }
    }
    
    println!("  🔋 Final voltages: {:?} mV", final_voltages);
    
    // Verify that membrane voltages have changed due to current injection
    assert_eq!(initial_voltages.len(), final_voltages.len(), "Should have same number of neurons");
    
    let mut voltage_increased = false;
    for (i, (&initial, &final_vol)) in initial_voltages.iter().zip(final_voltages.iter()).enumerate() {
        let voltage_change = final_vol - initial;
        println!("  📊 Neuron {}: {:.2}mV → {:.2}mV (Δ={:.2}mV)", i, initial, final_vol, voltage_change);
        
        // With positive current injection, we expect voltage to increase
        // Allow for some tolerance due to numerical integration
        if voltage_change > 0.1 {
            voltage_increased = true;
            println!("    ✅ Significant voltage increase detected!");
        }
    }
    
    // At least one neuron should show voltage increase from current injection
    assert!(voltage_increased, "Expected at least one neuron to show voltage increase from current injection");
    
    // Verify that stimulators are present and active at the target time
    {
        let stimulator_query = world.query::<&Stimulator>();
        let stimulator_count = stimulator_query.iter(world).count();
        println!("  🔌 Active stimulators: {}", stimulator_count);
        assert_eq!(stimulator_count, 2, "Should have 2 stimulators");
        
        // Check that stimulators are producing current at t=150ms
        let test_time = Timestamp(0.15);
        for stimulator in stimulator_query.iter(world) {
            let current = stimulator.current(test_time);
            println!("    Stimulator current at t=150ms: {:.3} µA/cm²", current.0);
            assert!(current.0 > 0.0, "Stimulator should be active at t=150ms");
        }
    }
    
    println!("✅ NeuroML current injection simulation test passed!");
    println!("🎉 LIF neuron dynamics are working with NeuroML PulseGenerators!");
}

#[test] 
fn test_neuroml_stimulator_timing_in_simulation() {
    println!("🧪 Testing NeuroML stimulator timing over simulation...");
    
    // Create app with just the resources needed for stimulator testing
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(AssetPlugin::default());
    
    // Create the NeuroML scene
    let scene = sample::create_sample_scene();
    let world = &mut app.world;
    
    // Initialize MembraneMaterials
    let membrane_materials = MembraneMaterials::from_world(world);
    world.insert_resource(membrane_materials);
    
    let mut system_state: bevy::ecs::system::SystemState<(
        Commands,
        ResMut<Assets<Mesh>>,
        Res<MembraneMaterials>,
        ResMut<Assets<StandardMaterial>>,
        Query<Entity, With<Selection>>,
        Query<Entity, With<Highlight>>,
    )> = bevy::ecs::system::SystemState::new(world);

    let (commands, mut meshes, membrane_materials, mut materials, selections, highlights) = system_state.get_mut(world);
    
    // Spawn NeuroML scene
    scene.spawn(Vec3::ZERO, commands, &mut meshes, membrane_materials, &mut materials, selections, highlights);
    system_state.apply(world);
    drop(system_state);
    
    // Test stimulator timing at different points
    let test_times = [
        (0.05, "Before both pulses"),
        (0.075, "During pulseGen2 only"),  
        (0.125, "During both pulses"),
        (0.25, "During both pulses"), 
        (0.32, "During pulseGen1 only"),
        (0.4, "After both pulses"),
    ];
    
    println!("  📊 Testing stimulator currents at different time points:");
    
    let stimulator_query = world.query::<&Stimulator>();
    let stimulators: Vec<&Stimulator> = stimulator_query.iter(world).collect();
    assert_eq!(stimulators.len(), 2, "Should have 2 stimulators");
    
    for (time_s, description) in test_times {
        let timestamp = Timestamp(time_s);
        println!("    t={:.0}ms ({})", time_s * 1000.0, description);
        
        for (i, stimulator) in stimulators.iter().enumerate() {
            let current = stimulator.current(timestamp);
            println!("      Stimulator {}: {:.3} µA/cm²", i, current.0);
        }
    }
    
    // Verify expected timing patterns
    let stim1_during = stimulators[0].current(Timestamp(0.2)); // pulseGen1: 100-300ms
    let stim1_after = stimulators[0].current(Timestamp(0.4));  // After 300ms
    
    let stim2_during = stimulators[1].current(Timestamp(0.2)); // pulseGen2: 50-350ms  
    let stim2_after = stimulators[1].current(Timestamp(0.4));  // After 350ms
    
    assert!(stim1_during.0 > 0.0, "Stimulator 1 should be active at t=200ms");
    assert_eq!(stim1_after.0, 0.0, "Stimulator 1 should be inactive at t=400ms");
    
    assert!(stim2_during.0 > 0.0, "Stimulator 2 should be active at t=200ms");  
    assert_eq!(stim2_after.0, 0.0, "Stimulator 2 should be inactive at t=400ms");
    
    println!("✅ NeuroML stimulator timing test passed!");
}

#[test]
fn test_neuroml_integration_with_bevy_ecs() {
    println!("🧪 Testing NeuroML integration with Bevy ECS...");
    
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(AssetPlugin::default());
    
    // Create and spawn NeuroML scene
    let scene = sample::create_sample_scene();
    let world = &mut app.world;
    
    // Initialize MembraneMaterials
    let membrane_materials = MembraneMaterials::from_world(world);
    world.insert_resource(membrane_materials);
    
    let mut system_state: bevy::ecs::system::SystemState<(
        Commands,
        ResMut<Assets<Mesh>>,
        Res<MembraneMaterials>, 
        ResMut<Assets<StandardMaterial>>,
        Query<Entity, With<Selection>>,
        Query<Entity, With<Highlight>>,
    )> = bevy::ecs::system::SystemState::new(world);

    let (commands, mut meshes, membrane_materials, mut materials, selections, highlights) = system_state.get_mut(world);
    
    let neuron_entities = scene.spawn(Vec3::ZERO, commands, &mut meshes, membrane_materials, &mut materials, selections, highlights);
    system_state.apply(world);
    drop(system_state);
    
    println!("  🧬 Spawned {} neuron entities", neuron_entities.len());
    
    // Verify ECS component structure
    let segment_query = world.query::<(Entity, &Segment, &MembraneVoltage)>();
    let segments: Vec<_> = segment_query.iter(world).collect();
    println!("  🧩 Found {} segments with membrane voltage", segments.len());
    
    let stimulator_query = world.query::<(Entity, &Stimulator)>();
    let stimulators: Vec<_> = stimulator_query.iter(world).collect();
    println!("  ⚡ Found {} stimulators", stimulators.len());
    
    let stimulation_query = world.query::<&Stimulation>();
    let stimulations: Vec<_> = stimulation_query.iter(world).collect();
    println!("  🔗 Found {} stimulation links", stimulations.len());
    
    // Verify expected ECS structure
    assert_eq!(segments.len(), 2, "Should have 2 neuron segments");
    assert_eq!(stimulators.len(), 2, "Should have 2 stimulators");
    assert_eq!(stimulations.len(), 2, "Should have 2 stimulation links");
    
    // Verify each segment has initial membrane voltage
    for (entity, _segment, voltage) in &segments {
        println!("    Segment {:?}: {:.2}mV", entity, voltage.0.0);
        // Should start around resting potential (~-70mV)
        assert!(voltage.0.0 < -50.0 && voltage.0.0 > -80.0, 
               "Membrane voltage should be around resting potential");
    }
    
    // Verify stimulation links connect stimulators to segments
    for stimulation in &stimulations {
        let segment_entity = stimulation.stimulation_segment;
        
        // Verify the linked segment exists and has a stimulator
        let has_segment = world.get::<Segment>(segment_entity).is_some();
        let has_stimulator = world.get::<Stimulator>(segment_entity).is_some();
        
        assert!(has_segment, "Stimulation should link to valid segment");
        assert!(has_stimulator, "Linked segment should have stimulator component");
        
        println!("    ✓ Stimulation link verified for segment {:?}", segment_entity);
    }
    
    println!("✅ NeuroML Bevy ECS integration test passed!");
}