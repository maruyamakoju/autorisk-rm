"""Mining module: signal scorers and fusion for candidate extraction."""

from autorisk.mining.audio import AudioScorer
from autorisk.mining.fuse import SignalFuser
from autorisk.mining.motion import MotionScorer
from autorisk.mining.proximity import ProximityScorer

__all__ = ["AudioScorer", "MotionScorer", "ProximityScorer", "SignalFuser"]
