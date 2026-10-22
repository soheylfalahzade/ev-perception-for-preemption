"""Event contract between perception and the signal-control layer."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

SCHEMA_VERSION = "1.0"

EVENT_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "EmergencyVehicleEvent",
    "type": "object",
    "required": ["schema_version", "timestamp_s", "track_id", "edge_id", "confidence"],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "timestamp_s": {"type": "number", "description": "capture time of the confirming frame"},
        "track_id": {"type": "integer"},
        "edge_id": {"type": "string", "description": "road-graph edge the vehicle is matched to"},
        "s_m": {"type": "number", "description": "arc-length position along the edge"},
        "dist_to_stop_line_m": {"type": "number", "description": "distance to the edge end"},
        "eta_s": {"type": "number", "description": "dist_to_stop_line_m / max(speed, 1 m/s)"},
        "heading_deg": {"type": ["number", "null"], "description": "compass-free math angle, deg"},
        "speed_mps": {"type": "number"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "match_prob": {"type": "number", "minimum": 0, "maximum": 1},
        "latency_ms": {"type": "number", "description": "processing time of the confirming frame"},
    },
}


@dataclass(frozen=True)
class EVEvent:
    """A confirmed emergency-vehicle sighting, expressed on the road graph."""

    timestamp_s: float
    track_id: int
    edge_id: str
    s_m: float
    dist_to_stop_line_m: float
    eta_s: float
    heading_deg: float | None
    speed_mps: float
    confidence: float
    match_prob: float
    latency_ms: float = 0.0
    schema_version: str = SCHEMA_VERSION

    def to_json(self) -> str:
        """Serialise to a single JSON line following :data:`EVENT_SCHEMA`."""
        return json.dumps(asdict(self), separators=(",", ":"))
