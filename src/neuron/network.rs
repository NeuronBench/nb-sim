// use crate::dimension::{Interval, Kelvin, MicroAmps};
// use crate::neuron::synapse::Synapse;
// use crate::neuron::Neuron;
// use crate::neuron::Solution;
//
// // TODO: Give Neurons and Segments Id's, and index by Id.
// pub struct NetworkSegmentIndex {
//     pub neuron: usize,
//     pub segment: usize,
// }
//
// pub struct Network {
//     pub neurons: Vec<Neuron>,
//     pub synapses: Vec<(NetworkSegmentIndex, NetworkSegmentIndex, Synapse)>,
//     pub extracellular_solution: Solution,
// }
//
// impl Network {
//     pub fn step(&mut self, temperature: &Kelvin, interval: &Interval) {
//         // First apply the sytaptic currents to their respective neurons.
//         self.neurons.iter_mut().for_each(|neuron| {
//             neuron
//                 .segments
//                 .iter_mut()
//                 .for_each(|segment| segment.synaptic_current = MicroAmps(0.0))
//         });
//         self.synapses
//             .iter()
//             .for_each(|(_, NetworkSegmentIndex { neuron, segment }, synapse)| {
//                 let mut postsynaptic_segment =
//                     &mut self.neurons[neuron.clone()].segments[segment.clone()];
//                 let current = synapse.current(temperature, postsynaptic_segment);
//                 postsynaptic_segment.synaptic_current =
//                     MicroAmps(postsynaptic_segment.synaptic_current.0 + current.0);
//             });
//
//         // Then step the neurons.
//         self.neurons
//             .iter_mut()
//             .for_each(|neuron| neuron.step(temperature, &self.extracellular_solution, interval));
//
//         // Finally step the synapses.
//         self.synapses
//             .iter_mut()
//             .for_each(|(presynaptic_index, postsynaptic_index, synapse)| {
//                 let presynaptic_segment = &self.neurons[presynaptic_index.neuron.clone()].segments
//                     [presynaptic_index.segment.clone()];
//
//                 let postsynaptic_segment = &self.neurons[postsynaptic_index.neuron.clone()]
//                     .segments[postsynaptic_index.segment.clone()];
//
//                 synapse.step(
//                     temperature,
//                     presynaptic_segment,
//                     postsynaptic_segment,
//                     interval,
//                 );
//             })
//     }
// }
