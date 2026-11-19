//! Runs the fixtures in `sample_data/onnx/` (made by `make_fixtures.py`) and
//! compares against outputs from onnx's reference evaluator.

use std::path::PathBuf;

use burn::tensor::Device;
use nb_sim::onnx::{HostTensor, Program};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    inputs: Vec<Named>,
    outputs: Vec<Named>,
}

#[derive(Deserialize)]
struct Named {
    name: String,
    shape: Vec<usize>,
    data: Vec<f32>,
}

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("sample_data/onnx").join(name)
}

fn compile(name: &str) -> Result<Program, nb_sim::onnx::CompileError> {
    let bytes = std::fs::read(fixture_path(&format!("{name}.onnx"))).expect("fixture exists");
    Program::from_bytes(&bytes, Device::default())
}

fn check_against_reference(name: &str) {
    let program = compile(name).unwrap_or_else(|e| panic!("{name} should compile: {e}"));
    let fixture: Fixture =
        serde_json::from_slice(&std::fs::read(fixture_path(&format!("{name}.json"))).unwrap()).unwrap();

    let names: Vec<_> = program.inputs().iter().map(|p| p.name.clone()).collect();
    assert_eq!(names, fixture.inputs.iter().map(|i| i.name.clone()).collect::<Vec<_>>());

    let inputs: Vec<HostTensor> =
        fixture.inputs.iter().map(|i| HostTensor::new(i.shape.clone(), i.data.clone())).collect();
    let outputs = program.run(&inputs).unwrap_or_else(|e| panic!("{name} should run: {e}"));

    assert_eq!(outputs.len(), fixture.outputs.len());
    for (got, want) in outputs.iter().zip(&fixture.outputs) {
        assert_eq!(got.shape, want.shape, "{name}: shape of output `{}`", want.name);
        for (i, (g, w)) in got.data.iter().zip(&want.data).enumerate() {
            assert!(
                (g - w).abs() <= 1e-5 + 1e-5 * w.abs(),
                "{name}: output `{}`[{i}] = {g}, reference {w}",
                want.name
            );
        }
    }
}

#[test]
fn mlp_matches_reference() {
    check_against_reference("mlp");
}

#[test]
fn every_supported_op_matches_reference() {
    check_against_reference("ops");
}

#[test]
fn batched_matmul_and_gemm_options_match_reference() {
    check_against_reference("matmul");
}

#[test]
fn unsupported_op_is_a_compile_error_naming_the_op() {
    let err = compile("unsupported").err().expect("Cos is not supported");
    assert!(err.to_string().contains("Cos"), "error should name the op: {err}");
}

#[test]
fn input_shape_mismatch_is_a_run_error() {
    let program = compile("mlp").unwrap();
    assert_eq!(program.inputs()[0].shape, vec![Some(1), Some(4)]);
    let err = program.run(&[HostTensor::new(vec![1, 3], vec![0.0; 3])]).unwrap_err();
    assert!(err.to_string().contains("expects shape"), "{err}");
}

#[test]
fn garbage_bytes_are_a_compile_error() {
    assert!(Program::from_bytes(b"not an onnx model", Device::default()).is_err());
}
