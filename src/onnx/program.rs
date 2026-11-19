//! Runtime ONNX evaluation: `onnx-ir` parses the model, [`Program::compile`]
//! turns its typed nodes into a flat list of steps, and [`Program::run`]
//! evaluates them with burn tensor ops.
//!
//! burn tensors carry their rank in the type (`Tensor<D>`), but an
//! interpreter only learns ranks at runtime. Every value is therefore stored
//! as a `Tensor<4>` with leading 1s plus its logical rank. Left-padding with
//! ones is exactly ONNX (numpy) broadcasting, and batched matmul over the two
//! leading dims matches ONNX MatMul, so most ops need no rank bookkeeping at
//! all. Models with tensors above rank 4 are rejected at compile time.

use std::collections::HashMap;
use std::fmt;

use burn::tensor::{activation, DType, Device, Tensor, TensorData};
use onnx_ir::{Argument, Node, OnnxGraph, OnnxGraphBuilder};

/// Highest tensor rank a program may use.
pub const MAX_RANK: usize = 4;

/// A host-side tensor: row-major `f32` data and its shape. Inputs and
/// outputs of [`Program::run`] use this, so callers never touch burn types.
#[derive(Clone, Debug, PartialEq)]
pub struct HostTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl HostTensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        Self { shape, data }
    }
}

/// A graph input or output: its name, and its shape where the model fixes it
/// (`None` for a dynamic dimension such as a batch axis).
#[derive(Clone, Debug, PartialEq)]
pub struct Port {
    pub name: String,
    pub shape: Vec<Option<usize>>,
}

/// Why a model could not be compiled.
#[derive(Clone, Debug, PartialEq)]
pub struct CompileError {
    /// The ONNX node at fault, when there is one.
    pub node: Option<String>,
    pub message: String,
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.node {
            Some(node) => write!(f, "node `{node}`: {}", self.message),
            None => f.write_str(&self.message),
        }
    }
}

impl std::error::Error for CompileError {}

/// Why a compiled program could not run on the given inputs.
#[derive(Clone, Debug, PartialEq)]
pub struct RunError(pub String);

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for RunError {}

/// A compiled ONNX model, ready to run.
pub struct Program {
    device: Device,
    inputs: Vec<Port>,
    outputs: Vec<Port>,
    /// Slot of each graph input, in `inputs` order.
    input_slots: Vec<Slot>,
    /// Slot of each graph output, in `outputs` order.
    output_slots: Vec<Slot>,
    /// Constants (weights), materialized once at compile time.
    constants: Vec<(Slot, Value)>,
    steps: Vec<Step>,
    slot_count: usize,
}

type Slot = usize;

#[derive(Clone)]
struct Value {
    tensor: Tensor<MAX_RANK>,
    rank: usize,
}

impl Value {
    /// The logical (unpadded) shape.
    fn shape(&self) -> Vec<usize> {
        self.tensor.dims()[MAX_RANK - self.rank..].to_vec()
    }

    fn with(&self, tensor: Tensor<MAX_RANK>) -> Value {
        Value { tensor, rank: self.rank }
    }
}

struct Step {
    node: String,
    op: Op,
    inputs: Vec<Slot>,
    output: Slot,
    /// Logical rank of the output.
    rank: usize,
}

enum Op {
    Identity,
    Add,
    Sub,
    Mul,
    Div,
    Relu,
    Tanh,
    Sigmoid,
    LeakyRelu { alpha: f64 },
    Clip { min: f32, max: f32 },
    /// `axis` is in the padded rank-4 frame.
    Softmax { axis: usize },
    /// `axis` is in the logical frame of the input.
    Flatten { axis: usize },
    /// ONNX reshape spec: 0 copies the input dim, -1 is inferred.
    Reshape { shape: Vec<i64> },
    /// `axis` is in the padded rank-4 frame.
    Concat { axis: usize },
    MatMul,
    Gemm { alpha: f32, beta: f32, trans_a: bool, trans_b: bool },
    Linear { transpose_weight: bool },
}

impl Program {
    /// Parse and compile an ONNX model from its bytes.
    pub fn from_bytes(bytes: &[u8], device: Device) -> Result<Program, CompileError> {
        let graph = OnnxGraphBuilder::new().parse_bytes(bytes).map_err(|e| CompileError {
            node: None,
            message: format!("could not parse the ONNX model: {e}"),
        })?;
        Program::compile(graph, device)
    }

    /// Compile a parsed graph, rejecting anything this interpreter can't run.
    pub fn compile(graph: OnnxGraph, device: Device) -> Result<Program, CompileError> {
        let mut compiler = Compiler { device: device.clone(), slots: HashMap::new(), constants: Vec::new(), slot_count: 0 };

        let mut inputs = Vec::new();
        let mut input_slots = Vec::new();
        for arg in &graph.inputs {
            check_float_tensor(arg, None)?;
            inputs.push(port(arg));
            input_slots.push(compiler.define(&arg.name));
        }

        let mut steps = Vec::new();
        for node in &graph.nodes {
            if let Some(step) = compiler.step(node)? {
                steps.push(step);
            }
        }

        let mut outputs = Vec::new();
        let mut output_slots = Vec::new();
        for arg in &graph.outputs {
            check_float_tensor(arg, None)?;
            outputs.push(port(arg));
            output_slots.push(compiler.input(arg, "graph output")?);
        }

        Ok(Program {
            device,
            inputs,
            outputs,
            input_slots,
            output_slots,
            constants: compiler.constants,
            steps,
            slot_count: compiler.slot_count,
        })
    }

    pub fn inputs(&self) -> &[Port] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[Port] {
        &self.outputs
    }

    /// Evaluate the program. `inputs` are in [`Program::inputs`] order; the
    /// result is in [`Program::outputs`] order.
    pub fn run(&self, inputs: &[HostTensor]) -> Result<Vec<HostTensor>, RunError> {
        if inputs.len() != self.inputs.len() {
            return Err(RunError(format!(
                "the model takes {} inputs, got {}",
                self.inputs.len(),
                inputs.len()
            )));
        }

        let mut values: Vec<Option<Value>> = vec![None; self.slot_count];
        for (slot, value) in &self.constants {
            values[*slot] = Some(value.clone());
        }
        for ((port, host), slot) in self.inputs.iter().zip(inputs).zip(&self.input_slots) {
            check_input_shape(port, host)?;
            values[*slot] = Some(to_value(host, &self.device));
        }

        for step in &self.steps {
            let args: Vec<&Value> = step
                .inputs
                .iter()
                .map(|slot| values[*slot].as_ref().expect("compile order guarantees inputs are set"))
                .collect();
            let out = eval(step, &args).map_err(|e| RunError(format!("node `{}`: {e}", step.node)))?;
            values[step.output] = Some(out);
        }

        self.output_slots
            .iter()
            .map(|slot| to_host(values[*slot].as_ref().expect("outputs are set")))
            .collect()
    }
}

struct Compiler {
    device: Device,
    slots: HashMap<String, Slot>,
    constants: Vec<(Slot, Value)>,
    slot_count: usize,
}

impl Compiler {
    fn define(&mut self, name: &str) -> Slot {
        let slot = self.slot_count;
        self.slot_count += 1;
        self.slots.insert(name.to_string(), slot);
        slot
    }

    /// The slot holding an argument: a value computed earlier, or a constant
    /// materialized on first use.
    fn input(&mut self, arg: &Argument, node: &str) -> Result<Slot, CompileError> {
        if let Some(slot) = self.slots.get(&arg.name) {
            return Ok(*slot);
        }
        let Some(data) = arg.value() else {
            return Err(err(node, format!("input `{}` is neither computed nor constant", arg.name)));
        };
        let shape = data.shape().to_vec();
        if shape.len() > MAX_RANK {
            return Err(err(node, format!("constant `{}` has rank {}; at most {MAX_RANK} is supported", arg.name, shape.len())));
        }
        let floats: Vec<f32> = data
            .convert::<f32>()
            .try_to_vec()
            .map_err(|e| err(node, format!("constant `{}` is not numeric: {e:?}", arg.name)))?;
        let value = to_value(&HostTensor::new(shape, floats), &self.device);
        // Anonymous constants (embedded, empty name) get a fresh slot each use.
        let slot = if arg.name.is_empty() { self.fresh() } else { self.define(&arg.name) };
        self.constants.push((slot, value));
        Ok(slot)
    }

    fn fresh(&mut self) -> Slot {
        let slot = self.slot_count;
        self.slot_count += 1;
        slot
    }

    fn step(&mut self, node: &Node) -> Result<Option<Step>, CompileError> {
        let name = node.name().to_string();
        let ins = node.inputs();
        let outs = node.outputs();

        // Initializers become Constant nodes; consumers read them through
        // `Argument::value()`, so there is nothing to evaluate.
        if matches!(node, Node::Constant(_)) {
            return Ok(None);
        }

        let in_rank = |i: usize| ins[i].ty.rank();
        let padded = |axis: usize, rank: usize| axis + MAX_RANK - rank;

        let (op, operands): (Op, &[Argument]) = match node {
            Node::Identity(_) => (Op::Identity, &ins[..1]),
            Node::Add(_) => (Op::Add, &ins[..2]),
            Node::Sub(_) => (Op::Sub, &ins[..2]),
            Node::Mul(_) => (Op::Mul, &ins[..2]),
            Node::Div(_) => (Op::Div, &ins[..2]),
            Node::Relu(_) => (Op::Relu, &ins[..1]),
            Node::Tanh(_) => (Op::Tanh, &ins[..1]),
            Node::Sigmoid(_) => (Op::Sigmoid, &ins[..1]),
            Node::LeakyRelu(n) => (Op::LeakyRelu { alpha: n.config.alpha }, &ins[..1]),
            Node::Clip(n) => {
                let bound = |b: &Option<onnx_ir::clip::ClipInput>, default: f32| match b {
                    None => Ok(default),
                    Some(onnx_ir::clip::ClipInput::Static(v)) => Ok(*v as f32),
                    Some(onnx_ir::clip::ClipInput::Runtime(_)) => {
                        Err(err(&name, "Clip bounds computed at runtime are not supported".into()))
                    }
                };
                let op = Op::Clip { min: bound(&n.config.min, f32::MIN)?, max: bound(&n.config.max, f32::MAX)? };
                (op, &ins[..1])
            }
            Node::Softmax(n) => (Op::Softmax { axis: padded(n.config.axis, in_rank(0)) }, &ins[..1]),
            Node::Flatten(n) => (Op::Flatten { axis: n.config.axis }, &ins[..1]),
            Node::Reshape(n) => match &n.config.shape {
                onnx_ir::reshape::ReshapeInput::Static(shape) => (Op::Reshape { shape: shape.clone() }, &ins[..1]),
                onnx_ir::reshape::ReshapeInput::Runtime(_) => {
                    return Err(err(&name, "Reshape to a shape computed at runtime is not supported".into()));
                }
            },
            Node::Concat(n) => (Op::Concat { axis: padded(n.config.axis, in_rank(0)) }, ins),
            Node::MatMul(_) => {
                if in_rank(1) < 2 {
                    return Err(err(&name, "MatMul with a rank-1 right operand is not supported".into()));
                }
                (Op::MatMul, &ins[..2])
            }
            Node::Gemm(n) => {
                let c = &n.config;
                let op = Op::Gemm { alpha: c.alpha, beta: c.beta, trans_a: c.trans_a != 0, trans_b: c.trans_b != 0 };
                (op, ins)
            }
            Node::Linear(n) => (Op::Linear { transpose_weight: n.config.transpose_weight }, ins),
            other => {
                return Err(err(
                    &name,
                    format!("ONNX op `{}` is not supported yet", other.node_type()),
                ));
            }
        };

        // Optional inputs that weren't provided show up with an empty name and no value.
        let operands: Vec<&Argument> = operands
            .iter()
            .filter(|a| !(a.name.is_empty() && a.value().is_none()))
            .collect();

        let mut slots = Vec::with_capacity(operands.len());
        for arg in operands {
            check_float_tensor(arg, Some(&name))?;
            slots.push(self.input(arg, &name)?);
        }

        let [out] = outs else {
            return Err(err(&name, format!("expected one output, found {}", outs.len())));
        };
        check_float_tensor(out, Some(&name))?;
        let rank = out.ty.rank();
        let output = self.define(&out.name);

        Ok(Some(Step { node: name, op, inputs: slots, output, rank }))
    }
}

fn eval(step: &Step, args: &[&Value]) -> Result<Value, String> {
    let rank = step.rank;
    let unary = |f: fn(Tensor<MAX_RANK>) -> Tensor<MAX_RANK>| {
        Ok::<_, String>(Value { tensor: f(args[0].tensor.clone()), rank })
    };
    let binary = |f: fn(Tensor<MAX_RANK>, Tensor<MAX_RANK>) -> Tensor<MAX_RANK>| {
        Ok::<_, String>(Value { tensor: f(args[0].tensor.clone(), args[1].tensor.clone()), rank })
    };

    match &step.op {
        Op::Identity => Ok(args[0].clone()),
        Op::Add => binary(|a, b| a + b),
        Op::Sub => binary(|a, b| a - b),
        Op::Mul => binary(|a, b| a * b),
        Op::Div => binary(|a, b| a / b),
        Op::Relu => unary(activation::relu),
        Op::Tanh => unary(activation::tanh),
        Op::Sigmoid => unary(activation::sigmoid),
        Op::LeakyRelu { alpha } => Ok(args[0].with(activation::leaky_relu(args[0].tensor.clone(), *alpha))),
        Op::Clip { min, max } => Ok(args[0].with(args[0].tensor.clone().clamp(*min, *max))),
        Op::Softmax { axis } => Ok(args[0].with(activation::softmax(args[0].tensor.clone(), *axis))),
        Op::Flatten { axis } => {
            let shape = args[0].shape();
            let outer: usize = shape[..*axis].iter().product();
            let inner: usize = shape[*axis..].iter().product();
            reshape(args[0], &[outer, inner])
        }
        Op::Reshape { shape: spec } => {
            let target = resolve_reshape(&args[0].shape(), spec)?;
            reshape(args[0], &target)
        }
        Op::Concat { axis } => {
            let tensors = args.iter().map(|v| v.tensor.clone()).collect();
            Ok(Value { tensor: Tensor::cat(tensors, *axis), rank })
        }
        Op::MatMul => binary(|a, b| a.matmul(b)),
        Op::Gemm { alpha, beta, trans_a, trans_b } => {
            let a = transpose_if(args[0].tensor.clone(), *trans_a);
            let b = transpose_if(args[1].tensor.clone(), *trans_b);
            let mut y = a.matmul(b);
            if *alpha != 1.0 {
                y = y * *alpha;
            }
            if let Some(c) = args.get(2) {
                y = y + c.tensor.clone() * *beta;
            }
            Ok(Value { tensor: y, rank })
        }
        Op::Linear { transpose_weight } => {
            let w = transpose_if(args[1].tensor.clone(), *transpose_weight);
            let mut y = args[0].tensor.clone().matmul(w);
            if let Some(bias) = args.get(2) {
                y = y + bias.tensor.clone();
            }
            Ok(Value { tensor: y, rank })
        }
    }
}

fn transpose_if(t: Tensor<MAX_RANK>, transpose: bool) -> Tensor<MAX_RANK> {
    if transpose { t.swap_dims(MAX_RANK - 2, MAX_RANK - 1) } else { t }
}

fn reshape(value: &Value, shape: &[usize]) -> Result<Value, String> {
    if shape.len() > MAX_RANK {
        return Err(format!("reshape to rank {} exceeds the supported {MAX_RANK}", shape.len()));
    }
    Ok(Value { tensor: value.tensor.clone().reshape(pad(shape)), rank: shape.len() })
}

/// Apply ONNX reshape rules: 0 copies the input dim, -1 takes the remainder.
fn resolve_reshape(input: &[usize], spec: &[i64]) -> Result<Vec<usize>, String> {
    let total: usize = input.iter().product();
    let mut out = Vec::with_capacity(spec.len());
    let mut infer = None;
    for (i, &d) in spec.iter().enumerate() {
        match d {
            0 => out.push(*input.get(i).ok_or_else(|| format!("reshape dim {i} copies a missing input dim"))?),
            -1 if infer.is_none() => {
                infer = Some(i);
                out.push(1);
            }
            d if d > 0 => out.push(d as usize),
            d => return Err(format!("invalid reshape dim {d}")),
        }
    }
    if let Some(i) = infer {
        let known: usize = out.iter().product();
        if known == 0 || total % known != 0 {
            return Err(format!("cannot reshape {input:?} to {spec:?}"));
        }
        out[i] = total / known;
    }
    if out.iter().product::<usize>() != total {
        return Err(format!("cannot reshape {input:?} to {spec:?}"));
    }
    Ok(out)
}

fn pad(shape: &[usize]) -> [usize; MAX_RANK] {
    let mut padded = [1; MAX_RANK];
    padded[MAX_RANK - shape.len()..].copy_from_slice(shape);
    padded
}

fn to_value(host: &HostTensor, device: &Device) -> Value {
    let data = TensorData::new(host.data.clone(), pad(&host.shape).to_vec());
    Value { tensor: Tensor::from_data(data, device), rank: host.shape.len() }
}

fn to_host(value: &Value) -> Result<HostTensor, RunError> {
    let data = value
        .tensor
        .to_data()
        .try_to_vec::<f32>()
        .map_err(|e| RunError(format!("could not read back an output: {e:?}")))?;
    Ok(HostTensor::new(value.shape(), data))
}

fn port(arg: &Argument) -> Port {
    let shape = match arg.ty.static_shape() {
        Some(dims) => dims.clone(),
        None => vec![None; arg.ty.rank()],
    };
    Port { name: arg.name.clone(), shape }
}

fn check_float_tensor(arg: &Argument, node: Option<&str>) -> Result<(), CompileError> {
    let fail = |message: String| CompileError { node: node.map(str::to_string), message };
    if arg.ty.is_shape() {
        return Err(fail(format!("`{}` is a shape value; shape arithmetic is not supported", arg.name)));
    }
    if arg.ty.rank() > MAX_RANK {
        return Err(fail(format!("`{}` has rank {}; at most {MAX_RANK} is supported", arg.name, arg.ty.rank())));
    }
    // Constants of any numeric type are converted to f32 when materialized;
    // computed values must already be float.
    let dtype = arg.ty.elem_type();
    if arg.value().is_none() && !matches!(dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16) {
        return Err(fail(format!("`{}` has element type {dtype:?}; only float tensors are supported", arg.name)));
    }
    Ok(())
}

fn check_input_shape(port: &Port, host: &HostTensor) -> Result<(), RunError> {
    let fits = port.shape.len() == host.shape.len()
        && port.shape.iter().zip(&host.shape).all(|(want, got)| want.map_or(true, |w| w == *got));
    if !fits {
        return Err(RunError(format!("input `{}` expects shape {:?}, got {:?}", port.name, port.shape, host.shape)));
    }
    if host.data.len() != host.shape.iter().product::<usize>() {
        return Err(RunError(format!(
            "input `{}` has {} values but shape {:?}",
            port.name,
            host.data.len(),
            host.shape
        )));
    }
    Ok(())
}

fn err(node: &str, message: String) -> CompileError {
    CompileError { node: Some(node.to_string()), message }
}
