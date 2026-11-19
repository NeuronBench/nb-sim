//! ONNX models in scenes. See docs/onnx-integration.md.

pub mod coupling;
pub mod plugin;
pub mod program;

pub use plugin::OnnxPlugin;
pub use program::{CompileError, HostTensor, Port, Program, RunError};
