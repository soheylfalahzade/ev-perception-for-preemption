# Changelog

All notable changes to this project are documented here. The format follows Keep a Changelog and
the project uses semantic versioning.

## [0.1.0]

### Added
- Flashing-light cue (`FlashDetector`, `light_feature`), 2-D constant-velocity motion filter,
  online HMM map matching on a road graph (including a SUMO `.net.xml` loader), logistic cue fusion
  with consecutive-frame confirmation, and the `EVEvent` contract with a JSON schema.
- Perception-level emulator (`synth`) with exact ground truth; evaluation of cue ablations,
  operating points, lead distance, map matching and latency, with Wilson intervals.
- Closed-loop preemption study in SUMO (`evperc-sumo`) with paired seeds.
- Data-driven scenario animation (`evperc-gif`), static typing (`mypy`), continuous integration.
