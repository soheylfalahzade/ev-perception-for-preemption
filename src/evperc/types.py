"""Input records of the perception pipeline."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackObservation:
    """One tracked vehicle in one frame, after localisation.

    Attributes:
        track_id: tracker identity (e.g. ByteTrack).
        t: capture time in seconds.
        xy_m: ground-plane position in the road-graph frame (metres).
        class_score: detector probability that the vehicle is an emergency vehicle.
        light_feature: fraction of saturated red/blue light pixels (see ``flash.light_feature``).
    """

    track_id: int
    t: float
    xy_m: tuple[float, float]
    class_score: float
    light_feature: float
