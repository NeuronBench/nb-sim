//! Binding a compiled [`Program`] to a scene: turning simulation state into
//! input tensors and output tensors into injected currents. Plain functions,
//! independent of Bevy, so they can be tested directly.

use std::sync::Arc;

use crate::onnx::{Activation, HostTensor, Port, Program};
use crate::serialize::{ModelInput, ModelOutput, SceneModel, SegmentRef};

/// A model whose bindings have been checked against its graph.
pub struct ModelInstance {
    pub program: Arc<Program>,
    pub spec: SceneModel,
    /// Tensor shape for each input, with dynamic dims resolved.
    input_shapes: Vec<Vec<usize>>,
}

/// Simulation state a model can read.
pub trait SimState {
    /// Membrane voltage of a segment in mV, or `None` if it doesn't exist.
    fn voltage_mv(&self, segment: SegmentRef) -> Option<f32>;
    /// Simulated time in seconds.
    fn time_sec(&self) -> f32;
}

/// The outcome of one evaluation.
#[derive(Debug, PartialEq)]
pub enum Step {
    /// Currents to inject, in µA/cm², per segment. A segment named by several
    /// outputs gets the sum.
    Inject(Vec<(SegmentRef, f32)>),
    /// Some input was not a finite number yet (for example GPU voltages
    /// before the first readback); the model was not run.
    NotReady,
}

impl ModelInstance {
    /// Check `spec` against the program's graph.
    pub fn bind(program: Arc<Program>, spec: SceneModel) -> Result<ModelInstance, String> {
        let ports = program.inputs();
        if spec.inputs.len() != ports.len() {
            return Err(format!(
                "the model has {} inputs ({}), but the scene binds {}",
                ports.len(),
                port_names(ports),
                spec.inputs.len()
            ));
        }
        let outputs = program.outputs();
        if spec.outputs.len() != outputs.len() {
            return Err(format!(
                "the model has {} outputs ({}), but the scene binds {}",
                outputs.len(),
                port_names(outputs),
                spec.outputs.len()
            ));
        }
        if !(spec.interval_sec >= 0.0) {
            return Err(format!("interval_sec must be at least 0, got {}", spec.interval_sec));
        }

        let input_shapes = ports
            .iter()
            .zip(&spec.inputs)
            .map(|(port, input)| fit_shape(port, input_len(input)))
            .collect::<Result<Vec<_>, _>>()?;

        for (port, output) in outputs.iter().zip(&spec.outputs) {
            if let ModelOutput::InjectCurrent { taps, .. } = output {
                // Only checkable now when the model fixes the output shape.
                if let Some(len) = static_len(port) {
                    if len != taps.len() {
                        return Err(format!(
                            "output `{}` has {len} values, but InjectCurrent names {} taps",
                            port.name,
                            taps.len()
                        ));
                    }
                }
            }
        }

        Ok(ModelInstance { program, spec, input_shapes })
    }

    /// Every segment this model reads or writes.
    pub fn taps(&self) -> impl Iterator<Item = SegmentRef> + '_ {
        let ins = self.spec.inputs.iter().flat_map(|i| match i {
            ModelInput::Voltages { taps, .. } => taps.as_slice(),
            _ => &[],
        });
        let outs = self.spec.outputs.iter().flat_map(|o| match o {
            ModelOutput::InjectCurrent { taps, .. } => taps.as_slice(),
            ModelOutput::Ignore => &[],
        });
        ins.chain(outs).copied()
    }

    /// Gather inputs from the simulation, run the model, and map its outputs
    /// to currents.
    pub fn step(&self, sim: &impl SimState) -> Result<Step, String> {
        self.evaluate(sim, false).map(|(step, _)| step)
    }

    /// Like [`ModelInstance::step`], and also return every layer's values.
    /// The trace is empty when the model didn't run.
    pub fn step_traced(&self, sim: &impl SimState) -> Result<(Step, Vec<Activation>), String> {
        self.evaluate(sim, true)
    }

    fn evaluate(&self, sim: &impl SimState, trace: bool) -> Result<(Step, Vec<Activation>), String> {
        let mut inputs = Vec::with_capacity(self.spec.inputs.len());
        for (input, shape) in self.spec.inputs.iter().zip(&self.input_shapes) {
            let data = match input {
                ModelInput::Voltages { taps, offset_mv, scale } => taps
                    .iter()
                    .map(|tap| {
                        sim.voltage_mv(*tap)
                            .map(|v| (v - offset_mv) * scale)
                            .ok_or_else(|| format!("no segment {} in neuron {}", tap.segment, tap.neuron))
                    })
                    .collect::<Result<Vec<_>, _>>()?,
                ModelInput::Time => vec![sim.time_sec()],
                ModelInput::Constant { values } => values.clone(),
            };
            if data.iter().any(|v| !v.is_finite()) {
                return Ok((Step::NotReady, Vec::new()));
            }
            inputs.push(HostTensor::new(shape.clone(), data));
        }

        let (outputs, activations) = if trace {
            self.program.run_traced(&inputs)
        } else {
            self.program.run(&inputs).map(|o| (o, Vec::new()))
        }
        .map_err(|e| e.to_string())?;

        let mut currents: Vec<(SegmentRef, f32)> = Vec::new();
        for ((output, port), tensor) in self.spec.outputs.iter().zip(self.program.outputs()).zip(&outputs) {
            let ModelOutput::InjectCurrent { taps, scale_uamps_per_square_cm } = output else { continue };
            if tensor.data.len() != taps.len() {
                return Err(format!(
                    "output `{}` has {} values, but InjectCurrent names {} taps",
                    port.name,
                    tensor.data.len(),
                    taps.len()
                ));
            }
            for (tap, value) in taps.iter().zip(&tensor.data) {
                if !value.is_finite() {
                    return Err(format!("output `{}` produced {value}", port.name));
                }
                let current = value * scale_uamps_per_square_cm;
                match currents.iter_mut().find(|(t, _)| t == tap) {
                    Some((_, total)) => *total += current,
                    None => currents.push((*tap, current)),
                }
            }
        }
        Ok((Step::Inject(currents), activations))
    }
}

fn input_len(input: &ModelInput) -> usize {
    match input {
        ModelInput::Voltages { taps, .. } => taps.len(),
        ModelInput::Time => 1,
        ModelInput::Constant { values } => values.len(),
    }
}

fn static_len(port: &Port) -> Option<usize> {
    port.shape.iter().copied().product::<Option<usize>>()
}

/// The tensor shape for `len` values fed into `port`: fixed dims as given,
/// and at most one dynamic dim inferred (several are all taken as 1, except
/// that a lone dynamic dim absorbs the remainder).
pub fn fit_shape(port: &Port, len: usize) -> Result<Vec<usize>, String> {
    let fixed: usize = port.shape.iter().map(|d| d.unwrap_or(1)).product();
    let dynamic: Vec<usize> = port.shape.iter().enumerate().filter(|(_, d)| d.is_none()).map(|(i, _)| i).collect();
    let mut shape: Vec<usize> = port.shape.iter().map(|d| d.unwrap_or(1)).collect();
    match dynamic.as_slice() {
        [i] if fixed > 0 && len % fixed == 0 => shape[*i] = len / fixed,
        _ if fixed == len => {}
        _ => {
            return Err(format!(
                "input `{}` has shape {}, which doesn't fit the {len} values bound to it",
                port.name,
                shape_text(&port.shape)
            ));
        }
    }
    Ok(shape)
}

fn shape_text(shape: &[Option<usize>]) -> String {
    let dims: Vec<String> = shape.iter().map(|d| d.map_or("?".to_string(), |n| n.to_string())).collect();
    format!("[{}]", dims.join(", "))
}

fn port_names(ports: &[Port]) -> String {
    ports.iter().map(|p| format!("`{}` {}", p.name, shape_text(&p.shape))).collect::<Vec<_>>().join(", ")
}
