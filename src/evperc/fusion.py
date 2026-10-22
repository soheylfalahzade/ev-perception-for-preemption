"""Cue fusion (logistic) and event confirmation with hysteresis."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def logit(p: float, eps: float = 1e-3) -> float:
    p = min(max(p, eps), 1 - eps)
    return float(np.log(p / (1 - p)))


def features(class_score: float, flash_score: float | None) -> np.ndarray:
    """Feature vector ``[logit(class), logit(flash) or 0, flash_available, 1]``."""
    has = flash_score is not None
    return np.array([logit(class_score), logit(flash_score) if has else 0.0, float(has), 1.0])


@dataclass
class FusionModel:
    """Logistic regression over detector and flashing-light evidence."""

    weights: np.ndarray = field(default_factory=lambda: np.array([0.8, 0.8, 0.0, -2.0]))

    def predict(self, x: np.ndarray) -> float:
        return float(1.0 / (1.0 + np.exp(-float(x @ self.weights))))

    def fit(self, x: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 50) -> FusionModel:
        """Newton / IRLS fit with L2 regularisation (bias unregularised)."""
        w = np.zeros(x.shape[1])
        reg = np.eye(x.shape[1]) * l2
        reg[-1, -1] = 0.0
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-(x @ w)))
            grad = x.T @ (p - y) + reg @ w
            hess = (x * (p * (1 - p))[:, None]).T @ x + reg + 1e-9 * np.eye(len(w))
            step = np.linalg.solve(hess, grad)
            w = w - step
            if np.abs(step).max() < 1e-8:
                break
        self.weights = w
        return self


@dataclass
class ConfirmConfig:
    """Operating point: smoothed posterior threshold and number of confirming frames."""

    threshold: float = 0.8
    confirm_frames: int = 5
    ema: float = 0.4


class Confirmer:
    """Per-track EMA of the posterior with a consecutive-frames confirmation rule."""

    def __init__(self, cfg: ConfirmConfig) -> None:
        self.cfg = cfg
        self._ema: dict[int, float] = {}
        self._run: dict[int, int] = {}

    def update(self, track_id: int, posterior: float) -> tuple[float, bool]:
        """Return ``(smoothed_posterior, confirmed)``."""
        prev = self._ema.get(track_id, posterior)
        cur = self.cfg.ema * posterior + (1 - self.cfg.ema) * prev
        self._ema[track_id] = cur
        run = self._run.get(track_id, 0) + 1 if cur >= self.cfg.threshold else 0
        self._run[track_id] = run
        return cur, run >= self.cfg.confirm_frames

    def drop(self, track_id: int) -> None:
        self._ema.pop(track_id, None)
        self._run.pop(track_id, None)
