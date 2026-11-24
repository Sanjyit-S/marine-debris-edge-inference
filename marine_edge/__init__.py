"""
Marine Debris Edge Inference Package
Clean, modular edge-vision pipeline for automated UAV marine debris detection and payload deployment.
"""

from .actuator import ServoActuator
from .camera import FrameSource
from .detector import ONNXDetector, Detection
from .pipeline import SurveillancePipeline

__all__ = [
    "ServoActuator",
    "FrameSource",
    "ONNXDetector",
    "Detection",
    "SurveillancePipeline",
]
