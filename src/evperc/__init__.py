"""Emergency-vehicle perception for signal preemption."""
from evperc.events import EVENT_SCHEMA, EVEvent
from evperc.flash import FlashDetector, light_feature
from evperc.fusion import ConfirmConfig, FusionModel
from evperc.mapmatch import MapMatcher, RoadGraph
from evperc.pipeline import EVPipeline
from evperc.types import TrackObservation

__version__ = "0.1.0"
__all__ = ["EVENT_SCHEMA", "EVEvent", "EVPipeline", "FlashDetector", "FusionModel",
           "ConfirmConfig", "MapMatcher", "RoadGraph", "TrackObservation", "light_feature"]
