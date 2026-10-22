"""Bridge from detector output (boxes + pixels) to :class:`TrackObservation`."""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from evperc.flash import light_feature
from evperc.types import TrackObservation

EMERGENCY_CLASSES = {"ambulance", "emergency", "fire_truck", "police"}


def observations_from_detections(
    t: float,
    frame_bgr: np.ndarray,
    detections: Sequence[tuple[int, str, tuple[float, float, float, float], float]],
    image_to_ground: Callable[[np.ndarray], np.ndarray],
) -> list[TrackObservation]:
    """Convert ``(track_id, class_name, xyxy, score)`` detections of one frame.

    ``class_score`` is the detector confidence if the class is an emergency class (a fine-tuned
    or open-vocabulary detector), otherwise ``1 - score`` is *not* used: it is set to 0.
    ``image_to_ground`` maps pixel points ``(N, 2)`` to metric ground points ``(N, 2)``.
    """
    out: list[TrackObservation] = []
    for tid, cls, (x1, y1, x2, y2), score in detections:
        xi1, yi1, xi2, yi2 = (int(round(v)) for v in (x1, y1, x2, y2))
        crop = frame_bgr[max(yi1, 0):max(yi2, 0), max(xi1, 0):max(xi2, 0)]
        feat = light_feature(crop) if crop.size else 0.0
        g = image_to_ground(np.array([[0.5 * (x1 + x2), y2]]))[0]
        out.append(TrackObservation(tid, t, (float(g[0]), float(g[1])),
                                    float(score) if cls in EMERGENCY_CLASSES else 0.0, feat))
    return out
