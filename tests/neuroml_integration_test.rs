use nb_sim::integrations::neuroml::{sample, convert_pulse_to_stimulator, parse_target_index};
use nb_sim::dimension::{Interval, MicroAmpsPerSquareCm, Timestamp};
use nb_sim::stimulator::{CurrentShape, Envelope};
use uom::si::f64::{ElectricCurrent, Time as UomTime};
use uom::si::{electric_current::nanoampere, time::millisecond};
use neuroml::neuroml::{InputTypes, PulseGenerator, Common};

#[test]
fn test_neuroml_scene_has_current_sources() {
    println!("🧪 Testing NeuroML scene structure...");
    
    let scene = sample::create_sample_scene();
    
    // Verify the scene has the expected structure
    assert_eq!(scene.0.input_types.len(), 2, "Should have 2 pulse generators");
    assert_eq!(scene.0.network[0].explicit_input.len(), 2, "Should have 2 current connections");
    
    // Verify input types are PulseGenerators
    for (i, input_type) in scene.0.input_types.iter().enumerate() {
        match input_type {
            InputTypes::PulseGenerator(pg) => {
                println!("  ✓ Found PulseGenerator {}: {}", i, pg.common.id);
            },
            _ => panic!("Expected PulseGenerator input type at index {}", i),
        }
    }
    
    // Verify explicit inputs target correct neurons
    let inputs = &scene.0.network[0].explicit_input;
    assert_eq!(inputs[0].target, "../neuron_population/0/simple_neuron");
    assert_eq!(inputs[1].target, "../neuron_population/1/simple_neuron");
    assert_eq!(inputs[0].input, "pulseGen1");
    assert_eq!(inputs[1].input, "pulseGen2");
    
    println!("✅ NeuroML scene structure test passed");
}

#[test]
fn test_pulse_generator_conversion() {
    println!("🧪 Testing PulseGenerator → Stimulator conversion...");
    
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
    assert!((stimulator.envelope.onset.0 - 0.1).abs() < 0.001, 
           "Onset should be ~0.1s, got {}", stimulator.envelope.onset.0);
    
    // Check duration (100ms delay + 200ms duration = 0.3s offset)  
    assert!((stimulator.envelope.offset.0 - 0.3).abs() < 0.001,
           "Offset should be ~0.3s, got {}", stimulator.envelope.offset.0);
    
    // Check current conversion (500nA = 0.5µA/cm²)
    if let CurrentShape::SquareWave { ref on_current, ref off_current } = stimulator.current_shape {
        assert!((on_current.0 - 0.5).abs() < 0.001,
               "On current should be ~0.5 µA/cm², got {}", on_current.0);
        assert_eq!(off_current.0, 0.0);
        
        println!("  ✓ Timing: 100ms delay → 0.1s onset");
        println!("  ✓ Duration: 200ms duration → 0.3s offset");
        println!("  ✓ Current: 500nA → 0.5µA/cm²");
    } else {
        panic!("Expected SquareWave current shape");
    }
    
    println!("✅ PulseGenerator conversion test passed");
}

#[test]
fn test_current_injection_timing() {
    println!("🧪 Testing current injection timing...");
    
    // Test both pulse generators from the sample scene
    let scene = sample::create_sample_scene();
    
    for (i, input_type) in scene.0.input_types.iter().enumerate() {
        if let InputTypes::PulseGenerator(pulse_gen) = input_type {
            let stimulator = convert_pulse_to_stimulator(pulse_gen);
            
            println!("  Testing {}: {}", i, pulse_gen.common.id);
            
            match pulse_gen.common.id.as_str() {
                "pulseGen1" => {
                    // pulseGen1: 100ms delay, 200ms duration
                    // Should be ON from 0.1s to 0.3s
                    let current_before = stimulator.current(Timestamp(0.05));
                    let current_during = stimulator.current(Timestamp(0.2)); 
                    let current_after = stimulator.current(Timestamp(0.4));
                    
                    assert_eq!(current_before.0, 0.0, "Should be OFF before onset");
                    assert!(current_during.0 > 0.0, "Should be ON during pulse");
                    assert_eq!(current_after.0, 0.0, "Should be OFF after offset");
                    
                    println!("    ✓ OFF-ON-OFF timing pattern correct");
                },
                "pulseGen2" => {
                    // pulseGen2: 50ms delay, 300ms duration  
                    // Should be ON from 0.05s to 0.35s
                    let current_before = stimulator.current(Timestamp(0.02));
                    let current_during = stimulator.current(Timestamp(0.2));
                    let current_after = stimulator.current(Timestamp(0.4));
                    
                    assert_eq!(current_before.0, 0.0, "Should be OFF before onset");
                    assert!(current_during.0 > 0.0, "Should be ON during pulse");
                    assert_eq!(current_after.0, 0.0, "Should be OFF after offset");
                    
                    println!("    ✓ OFF-ON-OFF timing pattern correct");
                },
                _ => panic!("Unexpected pulse generator id: {}", pulse_gen.common.id),
            }
        }
    }
    
    println!("✅ Current injection timing test passed");
}

#[test]
fn test_target_parsing() {
    println!("🧪 Testing NeuroML target parsing...");
    
    assert_eq!(parse_target_index("../neuron_population/0/simple_neuron"), Some(0));
    assert_eq!(parse_target_index("../neuron_population/1/simple_neuron"), Some(1));
    assert_eq!(parse_target_index("../neuron_population/42/simple_neuron"), Some(42));
    assert_eq!(parse_target_index("invalid/path"), None);
    
    println!("  ✓ Valid targets parse correctly");
    println!("  ✓ Invalid targets return None");
    println!("✅ Target parsing test passed");
}

#[test]
fn test_neuroml_stimulator_integration_workflow() {
    println!("🧪 Testing complete NeuroML → nb-sim workflow...");
    
    // This test demonstrates the complete integration pipeline:
    // NeuroML PulseGenerator → nb-sim Stimulator → Current injection
    
    let scene = sample::create_sample_scene();
    
    println!("  📋 Created NeuroML scene with {} inputs", scene.0.input_types.len());
    
    for (i, input_type) in scene.0.input_types.iter().enumerate() {
        if let InputTypes::PulseGenerator(pulse_gen) = input_type {
            println!("  🔌 Processing input {}: {}", i, pulse_gen.common.id);
            
            // Step 1: Convert NeuroML to nb-sim
            let stimulator = convert_pulse_to_stimulator(pulse_gen);
            println!("    ✓ Converted to nb-sim Stimulator");
            
            // Step 2: Parse target neuron
            let target_str = &scene.0.network[0].explicit_input[i].target;
            let target_idx = parse_target_index(target_str);
            println!("    ✓ Parsed target: {} → {:?}", target_str, target_idx);
            
            // Step 3: Verify current generation works
            let test_time = Timestamp(0.15); // 150ms - should be during both pulses
            let current = stimulator.current(test_time);
            println!("    ✓ Current at t=150ms: {:.2} µA/cm²", current.0);
            
            // Different pulses should have different amplitudes
            match pulse_gen.common.id.as_str() {
                "pulseGen1" => assert!((current.0 - 0.5).abs() < 0.01), // 500nA → 0.5µA/cm²
                "pulseGen2" => assert!((current.0 - 0.3).abs() < 0.01), // 300nA → 0.3µA/cm²
                _ => panic!("Unexpected pulse generator"),
            }
        }
    }
    
    println!("✅ Complete NeuroML integration workflow test passed");
    println!("🎉 NeuroML current injection is working correctly!");
}