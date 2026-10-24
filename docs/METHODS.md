# Methods

Notation: a frame at time `t` contains observations `o_i = (id_i, x_i, c_i, l_i)` with ground-plane
position `x_i` in metres, detector emergency-class score `c_i` in [0, 1] and light feature `l_i`
(fraction of saturated red/blue pixels in the roof region).

## 1. Flashing-light score

For each track the last `W = 2 s` of `l` is resampled to a uniform 20 Hz grid, mean-removed and
Hann-windowed. With power spectrum `P_k`,

- peak concentration `rho = max_{f in [1, 6] Hz} P_f / sum_{k >= 1} P_k`,
- modulation depth `delta = p95(l) - p05(l)`,
- score `s = sigmoid(14 (rho - 0.30)) * sigmoid(60 (delta - 0.10))`.

`s` is undefined (None) until at least 75% of the window is covered (and 8 samples exist). A steady
light has `rho` near the white-noise level and a slow drift has its power below 1 Hz; both give low
`s`. A hazard-light blink is periodic and gives high `s` (modelled as a confounder in the emulator).

## 2. Motion

A per-track constant-velocity Kalman filter with state `[x, y, v_x, v_y]`, white-noise acceleration
(`sigma_a = 2 m/s^2`) and position noise `sigma = 2.5 m`. Speed is `||v||`; the heading
`atan2(v_y, v_x)` is undefined below 1.5 m/s.

## 3. Cue fusion and confirmation

Feature vector `z = [logit(c), logit(s) or 0, 1[s defined], 1]` (probabilities clipped to
[1e-3, 1 - 1e-3]). The posterior is `p = sigmoid(w . z)`, with `w` fitted by L2-regularised
Newton/IRLS (`lambda = 0.01`, bias unregularised) on development frames. The smoothed posterior is
`q_t = 0.4 p_t + 0.6 q_{t-1}`. A track is *confirmed* once `q_t >= theta` for `N = 5` consecutive
frames. `theta` is the smallest value on a fixed grid whose development false-alarm rate (fraction
of ordinary tracks that alarm) is at most 1%.

## 4. Map matching

Candidate edges are those within 25 m of the filtered position. For each track a normalised forward
filter over candidates is maintained:

`log a_t(e) = logsumexp_{e'} [ log a_{t-1}(e') + log T(e' -> e) ] - d(e)^2 / (2 sigma_p^2) - dtheta(e)^2 / (2 sigma_h^2)`

with `T = 0.97` (stay), `0.025` (successor edge), `1e-4` (any other), `sigma_p = 5 m`,
`sigma_h = 0.5 rad`; the heading term is omitted when the heading is undefined. The reported edge is
the arg-max; `match_prob` is its normalised posterior.

## 5. Event

On confirmation the pipeline emits one `EVEvent` per (track, edge) pair with
`dist_to_stop_line_m = L_e - s` (edge length minus arc position), `eta_s = dist / max(speed, 1)`,
`confidence = q_t`, `match_prob`, and the processing time of the confirming frame. The schema is in
`configs/event_schema.json`.

## 6. Metrics

- *Detection rate*: emergency-vehicle tracks with an alarm / emergency-vehicle tracks.
- *False-alarm rate*: ordinary tracks with an alarm / ordinary tracks. Both with Wilson 95% intervals.
- *Lead distance*: true distance to the end of the true edge at the first alarm frame (median over
  detected emergency tracks).
- *AUC*: track-level, using the maximum smoothed posterior of each track (ties count 0.5).
- *Edge accuracy*: fraction of frames whose matched edge equals the true edge; frames on junction
  connectors are excluded. Baseline: geometric nearest edge to the filtered position.
- *Latency*: wall-clock time per `process` call on one CPU core; detection and tracking excluded.
