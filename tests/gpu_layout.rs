//! The shader's structs must match the `#[repr(C)]` structs uploaded to
//! them byte for byte, or the GPU reads garbage without any error.

use std::mem::size_of;

use nb_sim::gpu::buffers::*;

fn shader() -> naga::Module {
    let source = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/shaders/biophysics.wgsl")).unwrap();
    naga::front::wgsl::parse_str(&source).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)))
}

#[test]
fn shader_validates() {
    let module = shader();
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|e| panic!("{e:?}"));
}

#[test]
fn shader_structs_match_the_uploaded_structs() {
    let module = shader();
    let mut layouter = naga::proc::Layouter::default();
    layouter.update(module.to_ctx()).unwrap();
    let wgsl = |name: &str| {
        let (handle, ty) = module.types.iter().find(|(_, t)| t.name.as_deref() == Some(name)).unwrap_or_else(|| panic!("no struct {name}"));
        let naga::TypeInner::Struct { members, .. } = &ty.inner else { panic!("{name} is not a struct") };
        (layouter[handle].size as usize, members.len())
    };
    // Every field is 4 bytes or an array/struct of them.
    let cases: [(&str, usize); 9] = [
        ("GpuGateState", size_of::<GpuGateState>()),
        ("GpuMembraneChannel", size_of::<GpuMembraneChannel>()),
        ("GpuSegmentData", size_of::<GpuSegmentData>()),
        ("GpuJunctionData", size_of::<GpuJunctionData>()),
        ("JunctionNeighbor", size_of::<GpuJunctionNeighbor>()),
        ("GpuTransmitterPump", size_of::<GpuTransmitterPump>()),
        ("GpuReceptor", size_of::<GpuReceptor>()),
        ("GpuSynapseData", size_of::<GpuSynapseData>()),
        ("SimParams", size_of::<SimParams>()),
    ];
    for (name, rust) in cases {
        let (size, _) = wgsl(name);
        assert_eq!(size, rust, "{name}: WGSL {size} bytes, Rust {rust}");
    }
    // The fields added for plasticity are where both sides expect them.
    assert_eq!(wgsl("GpuReceptor").1, 13);
    assert_eq!(wgsl("GpuSynapseData").1, 35);
}
