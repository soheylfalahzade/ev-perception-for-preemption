"""Flashing-light cue: pixel feature and periodicity detector."""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


def light_feature(crop_bgr: np.ndarray, top_fraction: float = 0.45) -> float:
    """Fraction of strongly saturated red or blue pixels in the upper part of a vehicle crop.

    Emergency light bars sit on the roof, so only the top ``top_fraction`` of the box is used.
    Returns the larger of the red and blue fractions (they alternate in many light bars).
    """
    h = max(1, int(crop_bgr.shape[0] * top_fraction))
    px = crop_bgr[:h].reshape(-1, 3).astype(np.int32)
    if px.size == 0:
        return 0.0
    b, g, r = px[:, 0], px[:, 1], px[:, 2]
    red = (r > 200) & (g < 110) & (b < 110)
    blue = (b > 200) & (r < 110) & (g < 170)
    return float(max(red.mean(), blue.mean()))


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


class FlashDetector:
    """Scores how strongly a per-track brightness/colour time series is periodic in 1-6 Hz.

    The series is resampled to a uniform grid, windowed, and transformed. The score combines the
    *peak concentration* (largest in-band spectral line over total non-DC power) with the
    modulation depth (5-95 percentile range), so steady lights, slow drifts and broadband noise
    all score low while a 1-6 Hz blinking pattern scores high.
    """

    def __init__(
        self,
        window_s: float = 2.0,
        grid_hz: float = 20.0,
        band_hz: tuple[float, float] = (1.0, 6.0),
        min_coverage: float = 0.75,
        peak_midpoint: float = 0.30,
        depth_midpoint: float = 0.10,
    ) -> None:
        self.window_s, self.grid_hz, self.band = window_s, grid_hz, band_hz
        self.min_coverage = min_coverage
        self.peak_mid, self.depth_mid = peak_midpoint, depth_midpoint
        self._buf: dict[int, deque[tuple[float, float]]] = defaultdict(deque)

    def add(self, track_id: int, t: float, value: float) -> None:
        """Append a sample and drop samples older than the window."""
        buf = self._buf[track_id]
        buf.append((t, value))
        while buf and t - buf[0][0] > self.window_s:
            buf.popleft()

    def drop(self, track_id: int) -> None:
        """Forget a track."""
        self._buf.pop(track_id, None)

    def score(self, track_id: int) -> float | None:
        """Flashing score in [0, 1], or ``None`` if the window is not yet sufficiently covered."""
        buf = self._buf.get(track_id)
        if not buf or len(buf) < 8:
            return None
        arr = np.asarray(buf)
        span = arr[-1, 0] - arr[0, 0]
        if span < self.min_coverage * self.window_s:
            return None
        n = int(self.grid_hz * span)
        grid = np.linspace(arr[0, 0], arr[-1, 0], n)
        x = np.interp(grid, arr[:, 0], arr[:, 1])
        depth = float(np.percentile(x, 95) - np.percentile(x, 5))
        x = (x - x.mean()) * np.hanning(n)
        power = np.abs(np.fft.rfft(x)) ** 2
        freqs = np.fft.rfftfreq(n, 1.0 / self.grid_hz)
        total = float(power[1:].sum())
        if total < 1e-12:
            return 0.0
        in_band = (freqs >= self.band[0]) & (freqs <= self.band[1])
        peak = float(power[in_band].max() / total)
        return _sigmoid(14.0 * (peak - self.peak_mid)) * _sigmoid(60.0 * (depth - self.depth_mid))
