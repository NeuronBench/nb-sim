//! ONNX models in scenes. See docs/onnx-integration.md.

pub mod program;

pub use program::{CompileError, HostTensor, Port, Program, RunError};
