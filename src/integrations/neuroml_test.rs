use super::neuroml::*;
use crate::dimension::{Interval, MicroAmpsPerSquareCm, Timestamp};
use crate::stimulator::{Stimulator, CurrentShape, Envelope};
use uom::si::f64::{ElectricCurrent, Time as UomTime};
use uom::si::{electric_current::nanoampere, time::millisecond};
use neuroml::neuroml::{InputTypes, PulseGenerator, Common};

#[test]
fn test_neuroml_scene_structure() {
    let scene = sample::create_sample_scene();
    
    // Verify the scene has the expected structure
    assert_eq!(scene.0.input_types.len(), 2, "Should have 2 pulse generators");
    assert_eq!(scene.0.network[0].explicit_input.len(), 2, "Should have 2 current connections");
    
    // Verify input types are PulseGenerators
    for (i, input_type) in scene.0.input_types.iter().enumerate() {
        match input_type {
            InputTypes::PulseGenerator(pg) => {
                println!("✓ PulseGenerator {}: id={}", i, pg.common.id);
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
    
    println!("✓ NeuroML scene structure test passed");
}

#[test]
fn test_pulse_generator_to_stimulator_conversion() {
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
        
        println!("✓ PulseGenerator conversion test passed");
        println!("  - Delay: {}ms → Onset: {}s", 100.0, stimulator.envelope.onset.0);
        println!("  - Duration: {}ms → Offset: {}s", 200.0, stimulator.envelope.offset.0);
        println!("  - Amplitude: {}nA → Current: {}µA/cm²", 500.0, on_current.0);
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
    
    println!("✓ Target parsing test passed");
}

#[test]
fn test_stimulator_current_timing() {
    // Create a stimulator that mimics the NeuroML conversion
    let stimulator = Stimulator {
        envelope: Envelope {
            period: Interval(1.0),     // 1 second period
            onset: Interval(0.1),      // Start at 100ms  
            offset: Interval(0.3),     // Stop at 300ms
        },
        current_shape: CurrentShape::SquareWave {
            on_current: MicroAmpsPerSquareCm(0.5),  // 0.5 µA/cm² (converted from 500nA)
            off_current: MicroAmpsPerSquareCm(0.0),
        }
    };
    
    // Test current at different time points
    let current_before = stimulator.current(Timestamp(0.05)); // Before onset (50ms)
    let current_during = stimulator.current(Timestamp(0.2));  // During pulse (200ms)
    let current_after = stimulator.current(Timestamp(0.4));   // After offset (400ms)
    
    assert_eq!(current_before.0, 0.0, "Should be OFF before onset");
    assert!(current_during.0 > 0.0, "Should be ON during pulse");
    assert_eq!(current_after.0, 0.0, "Should be OFF after offset");
    
    println!("✓ Stimulator timing test passed");
    println!("  - Current before (t=50ms): {:.2} µA/cm²", current_before.0);
    println!("  - Current during (t=200ms): {:.2} µA/cm²", current_during.0);  
    println!("  - Current after (t=400ms): {:.2} µA/cm²", current_after.0);
}

#[test]
fn test_neuroml_sample_scene_current_profiles() {
    // Test the specific PulseGenerators created in the sample scene
    let scene = sample::create_sample_scene();
    
    // Test both pulse generators
    for (i, input_type) in scene.0.input_types.iter().enumerate() {
        if let InputTypes::PulseGenerator(pulse_gen) = input_type {
            let stimulator = convert_pulse_to_stimulator(pulse_gen);
            
            println!("Testing PulseGenerator {}: {}", i, pulse_gen.common.id);
            
            match pulse_gen.common.id.as_str() {
                "pulseGen1" => {
                    // pulseGen1: 100ms delay, 200ms duration, 500nA amplitude
                    // Should be ON from 0.1s to 0.3s
                    let current_before = stimulator.current(Timestamp(0.05));
                    let current_during = stimulator.current(Timestamp(0.2)); 
                    let current_after = stimulator.current(Timestamp(0.4));
                    
                    assert_eq!(current_before.0, 0.0, "pulseGen1 should be OFF before onset");
                    assert!(current_during.0 > 0.0, "pulseGen1 should be ON during pulse");
                    assert_eq!(current_after.0, 0.0, "pulseGen1 should be OFF after offset");
                    
                    println!("  ✓ pulseGen1 timing: OFF-ON-OFF pattern correct");
                },
                "pulseGen2" => {
                    // pulseGen2: 50ms delay, 300ms duration, 300nA amplitude  
                    // Should be ON from 0.05s to 0.35s
                    let current_before = stimulator.current(Timestamp(0.02));
                    let current_during = stimulator.current(Timestamp(0.2));
                    let current_after = stimulator.current(Timestamp(0.4));
                    
                    assert_eq!(current_before.0, 0.0, "pulseGen2 should be OFF before onset");
                    assert!(current_during.0 > 0.0, "pulseGen2 should be ON during pulse");
                    assert_eq!(current_after.0, 0.0, "pulseGen2 should be OFF after offset");
                    
                    println!("  ✓ pulseGen2 timing: OFF-ON-OFF pattern correct");
                },
                _ => panic!("Unexpected pulse generator id: {}", pulse_gen.common.id),
            }
        }
    }
    
    println!("✓ NeuroML sample scene current profiles test passed");
}