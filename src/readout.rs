//! Readouts (see [`crate::serialize::Readout`]): score a population's spikes
//! against a streaming canvas's labels, frame by frame, and show the running
//! agreement in the Readouts window and the log.

use std::collections::{HashMap, VecDeque};

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use crate::canvas::SceneCanvases;
use crate::dimension::Timestamp;
use crate::neuron::segment::ecs::{ModelTargetRate, SegmentAddress, SpikeCount};
use crate::serialize::{Readout, SegmentRef};

/// How many recent frames the running agreement covers.
const RECENT: usize = 50;

#[derive(Resource, Default)]
pub struct SceneReadouts {
    pub readouts: Vec<ReadoutState>,
}

pub struct ReadoutState {
    pub spec: Readout,
    /// Spikes per tap during the frame on show.
    pub counts: Vec<u32>,
    pub frame: Option<usize>,
    /// Whether each recent frame's answer matched its label, newest last.
    pub recent: VecDeque<bool>,
    pub frames: u32,
    pub correct: u32,
    /// The last finished frame: its label, the answer (None if no tap
    /// spiked) and the counts.
    pub last: Option<(u32, Option<usize>, Vec<u32>)>,
    /// Spikes each tap's teacher asked for during the frame on show (target
    /// rate times time), while one did.
    pub expected: Vec<f32>,
    /// Recent frames' (spikes, expected spikes) per tap, for frames with a
    /// teacher.
    pub taught: VecDeque<(Vec<u32>, Vec<f32>)>,
    /// Of the recent frames with a teacher, whether the cells' answer was
    /// the teacher's.
    pub agreement: VecDeque<bool>,
}

impl SceneReadouts {
    /// Start over with the scene's readouts.
    pub fn load(&mut self, specs: &[Readout]) {
        self.readouts = specs
            .iter()
            .map(|spec| ReadoutState {
                spec: spec.clone(),
                counts: vec![0; spec.taps.len()],
                frame: None,
                recent: VecDeque::new(),
                frames: 0,
                correct: 0,
                last: None,
                expected: vec![0.0; spec.taps.len()],
                taught: VecDeque::new(),
                agreement: VecDeque::new(),
            })
            .collect();
    }
}

impl ReadoutState {
    /// The tap with the most spikes, if any spiked; ties go to the first.
    pub fn answer(counts: &[u32]) -> Option<usize> {
        let best = counts.iter().copied().max().filter(|m| *m > 0)?;
        counts.iter().position(|c| *c == best)
    }

    /// The frame on show is now `frame` with `label`: close the previous
    /// frame if it changed.
    pub fn observe(&mut self, frame: usize, previous_label: Option<u32>) {
        if self.frame == Some(frame) {
            return;
        }
        if self.frame.is_some() && self.expected.iter().any(|e| *e > 0.0) {
            push_recent(&mut self.taught, (self.counts.clone(), self.expected.clone()));
            let teacher = self
                .expected
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i);
            push_recent(&mut self.agreement, Self::answer(&self.counts) == teacher);
        }
        self.expected.iter_mut().for_each(|e| *e = 0.0);
        if let (Some(_), Some(label), true) = (self.frame, previous_label, self.spec.labeled) {
            let answer = Self::answer(&self.counts);
            let right = answer == Some(label as usize);
            self.frames += 1;
            self.correct += right as u32;
            self.recent.push_back(right);
            if self.recent.len() > RECENT {
                self.recent.pop_front();
            }
            self.last = Some((label, answer, self.counts.clone()));
        }
        self.counts.iter_mut().for_each(|c| *c = 0);
        self.frame = Some(frame);
    }

    pub fn recent_accuracy(&self) -> Option<f32> {
        fraction(&self.recent)
    }

    /// How often the cells' answer was the teacher's, over recent taught frames.
    pub fn teacher_agreement(&self) -> Option<f32> {
        fraction(&self.agreement)
    }

    /// Correlation of spikes with the teacher's expected spikes, pooled over
    /// taps and recent taught frames; and the mean of each.
    pub fn rate_match(&self) -> Option<(f32, f32, f32)> {
        let pairs: Vec<(f32, f32)> = self
            .taught
            .iter()
            .flat_map(|(c, e)| c.iter().zip(e).map(|(c, e)| (*c as f32, *e)))
            .collect();
        if pairs.len() < 2 {
            return None;
        }
        let n = pairs.len() as f32;
        let (mx, my) = (pairs.iter().map(|p| p.0).sum::<f32>() / n, pairs.iter().map(|p| p.1).sum::<f32>() / n);
        let cov: f32 = pairs.iter().map(|(x, y)| (x - mx) * (y - my)).sum();
        let vx: f32 = pairs.iter().map(|(x, _)| (x - mx).powi(2)).sum();
        let vy: f32 = pairs.iter().map(|(_, y)| (y - my).powi(2)).sum();
        let r = if vx > 0.0 && vy > 0.0 { cov / (vx * vy).sqrt() } else { 0.0 };
        Some((r, mx, my))
    }
}

fn push_recent<T>(q: &mut VecDeque<T>, item: T) {
    q.push_back(item);
    if q.len() > RECENT {
        q.pop_front();
    }
}

fn fraction(q: &VecDeque<bool>) -> Option<f32> {
    (!q.is_empty()).then(|| q.iter().filter(|r| **r).count() as f32 / q.len() as f32)
}

/// After each simulation step, add the taps' new spikes (from either
/// backend's `SpikeCount`) to their readouts and close frames as the
/// canvases move on.
pub fn score_readouts(
    readouts: Option<ResMut<SceneReadouts>>,
    canvases: Option<Res<SceneCanvases>>,
    addresses: Query<(Entity, &SegmentAddress, &SpikeCount, Option<&ModelTargetRate>)>,
    mut seen: Local<HashMap<Entity, f32>>,
    time: Res<Time>,
    timestamp: Res<Timestamp>,
    mut since_log: Local<f32>,
    mut last_time: Local<f32>,
) {
    let elapsed = (timestamp.0 - *last_time).max(0.0);
    *last_time = timestamp.0;
    let (Some(mut readouts), Some(canvases)) = (readouts, canvases) else { return };
    if readouts.readouts.is_empty() {
        return;
    }
    // New spikes per segment since the last frame; a count that went down
    // was restarted (GPU buffers rebuilt), so all of it is new.
    let index: HashMap<SegmentRef, (u32, f32)> = addresses
        .iter()
        .map(|(e, a, count, target)| {
            let before = seen.insert(e, count.0).unwrap_or(count.0);
            let new = if count.0 >= before { count.0 - before } else { count.0 };
            (SegmentRef { neuron: a.neuron, segment: a.segment }, (new as u32, target.map_or(-1.0, |t| t.0)))
        })
        .collect();
    if seen.len() > 2 * index.len() {
        seen.retain(|e, _| addresses.contains(*e));
    }
    for r in &mut readouts.readouts {
        let Some((frame, _)) = canvases.label(&r.spec.canvas) else { continue };
        let previous = r.frame.and_then(|f| canvases.label_of(&r.spec.canvas, f));
        r.observe(frame, previous);
        for ((count, expected), tap) in r.counts.iter_mut().zip(r.expected.iter_mut()).zip(&r.spec.taps) {
            let Some((new, target)) = index.get(tap) else { continue };
            *count += new;
            if *target >= 0.0 {
                *expected += target * elapsed;
            }
        }
    }
    *since_log += time.delta_secs();
    if *since_log >= 5.0 {
        *since_log = 0.0;
        for r in &readouts.readouts {
            if let Some((corr, spikes, wanted)) = r.rate_match() {
                info!(
                    "readout {}: spikes per frame vs. the teacher's: r = {corr:.2}, mean {spikes:.1} vs. {wanted:.1}; answers agree with the teacher's {:.0}%",
                    r.spec.name,
                    100.0 * r.teacher_agreement().unwrap_or(0.0)
                );
            }
            if let (Some(acc), Some((label, answer, counts))) = (r.recent_accuracy(), &r.last) {
                info!(
                    "readout {}: {:.0}% of the last {} frames ({}/{} overall); last: label {label}, answer {}, spikes {counts:?}",
                    r.spec.name,
                    100.0 * acc,
                    r.recent.len(),
                    r.correct,
                    r.frames,
                    answer.map_or("none".to_string(), |a| a.to_string()),
                );
            }
        }
    }
}

/// The Readouts window.
pub fn run_readout_panel(mut contexts: EguiContexts, readouts: Option<Res<SceneReadouts>>) {
    let Some(readouts) = readouts else { return };
    if readouts.readouts.is_empty() {
        return;
    }
    let Ok(ctx) = contexts.ctx_mut() else { return };
    egui::Window::new("Readouts").default_pos([20.0, 600.0]).resizable(false).show(ctx, |ui| {
        for r in &readouts.readouts {
            ui.strong(&r.spec.name);
            if let Some((corr, spikes, wanted)) = r.rate_match() {
                ui.label(format!(
                    "vs. teacher: r = {corr:.2}, {spikes:.1} spikes per frame vs. {wanted:.1} asked; same answer {:.0}%",
                    100.0 * r.teacher_agreement().unwrap_or(0.0)
                ));
            }
            if !r.spec.labeled {
                continue;
            }
            match r.recent_accuracy() {
                Some(acc) => ui.label(format!(
                    "{:.0}% of the last {} frames, {}/{} overall",
                    100.0 * acc,
                    r.recent.len(),
                    r.correct,
                    r.frames
                )),
                None => ui.label("waiting for the first frame to finish"),
            };
            if let Some((label, answer, counts)) = &r.last {
                ui.label(format!(
                    "last frame: label {label}, answer {}",
                    answer.map_or("none".to_string(), |a| a.to_string())
                ));
                ui.horizontal(|ui| {
                    let most = counts.iter().copied().max().unwrap_or(0).max(1);
                    for (i, c) in counts.iter().enumerate() {
                        let strength = *c as f32 / most as f32;
                        let color = if Some(i) == *answer {
                            egui::Color32::from_rgb(80, (120.0 + 135.0 * strength) as u8, 80)
                        } else {
                            egui::Color32::from_gray((60.0 + 150.0 * strength) as u8)
                        };
                        ui.colored_label(color, format!("{i}:{c}"));
                    }
                });
            }
        }
    });
}
