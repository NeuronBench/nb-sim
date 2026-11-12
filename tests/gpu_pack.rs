//! The packed `adj_index` buffer layout must match the accessors in
//! `assets/shaders/biophysics.wgsl`.

use nb_sim::gpu::pack_adj_index;

/// Mirrors the shader's accessors, so a layout change on one side fails here.
#[test]
fn packed_layout_matches_shader_accessors() {
    // 3 segments; segment 1 has two incoming synapses (7, 9), segment 2 has one (4).
    let junction_offsets = [0, 1, 3, 4];
    let synapse_offsets = [0, 0, 2, 3];
    let synapse_adj = [7, 9, 4];
    let packed = pack_adj_index(&junction_offsets, &synapse_offsets, &synapse_adj);

    let n = 3u32;
    let junction_adj_offset = |i: u32| packed[i as usize];
    let synapse_adj_offset = |i: u32| packed[(n + 1 + i) as usize];
    let synapse_adj_at = |i: u32| packed[(2 * (n + 1) + i) as usize];

    for i in 0..=n {
        assert_eq!(junction_adj_offset(i), junction_offsets[i as usize]);
        assert_eq!(synapse_adj_offset(i), synapse_offsets[i as usize]);
    }
    let seg1: Vec<u32> = (synapse_adj_offset(1)..synapse_adj_offset(2)).map(synapse_adj_at).collect();
    assert_eq!(seg1, vec![7, 9]);
    assert_eq!(synapse_adj_at(synapse_adj_offset(2)), 4);
}
