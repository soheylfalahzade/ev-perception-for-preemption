"""Constant-velocity Kalman filter in the ground plane: smoothed position, speed and heading."""
from __future__ import annotations

import math

import numpy as np


class MotionFilter:
    """2-D constant-velocity filter; heading is undefined below ``v_min`` (vehicle ~stationary)."""

    def __init__(self, meas_sigma_m: float = 2.5, accel_sigma: float = 2.0, v_min: float = 1.5):
        self._r = meas_sigma_m**2
        self._qa = accel_sigma**2
        self.v_min = v_min
        self.x: np.ndarray | None = None
        self.p = np.eye(4)
        self.t_last = 0.0

    def update(self, t: float, xy: tuple[float, float], meas_sigma_m: float | None = None) -> None:
        """Fuse a position fix taken at time ``t``."""
        r = self._r if meas_sigma_m is None else meas_sigma_m**2
        if self.x is None:
            self.x = np.array([xy[0], xy[1], 0.0, 0.0])
            self.p = np.diag([r, r, 100.0, 100.0])
            self.t_last = t
            return
        dt = max(t - self.t_last, 1e-3)
        self.t_last = t
        f = np.eye(4)
        f[0, 2] = f[1, 3] = dt
        q = self._qa * np.block([
            [np.eye(2) * dt**4 / 4, np.eye(2) * dt**3 / 2],
            [np.eye(2) * dt**3 / 2, np.eye(2) * dt**2],
        ])
        self.x = f @ self.x
        self.p = f @ self.p @ f.T + q
        h = np.hstack([np.eye(2), np.zeros((2, 2))])
        s = h @ self.p @ h.T + np.eye(2) * r
        k = self.p @ h.T @ np.linalg.inv(s)
        self.x = self.x + k @ (np.asarray(xy) - h @ self.x)
        self.p = (np.eye(4) - k @ h) @ self.p

    @property
    def position(self) -> tuple[float, float]:
        assert self.x is not None
        return float(self.x[0]), float(self.x[1])

    @property
    def speed(self) -> float:
        return 0.0 if self.x is None else float(math.hypot(self.x[2], self.x[3]))

    @property
    def heading_rad(self) -> float | None:
        """Direction of travel (math convention, radians) or ``None`` if nearly stationary."""
        if self.x is None or self.speed < self.v_min:
            return None
        return float(math.atan2(self.x[3], self.x[2]))
