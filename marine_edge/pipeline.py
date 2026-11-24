"""
Surveillance Pipeline Orchestrator
Coordinates frame intake, edge detection, visual overlays, and actuation triggering.
"""

from typing import Optional, Callable, Tuple, List
import time
import logging
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

from .camera import FrameSource
from .detector import ONNXDetector, Detection
from .actuator import ServoActuator

logger = logging.getLogger(__name__)

class SurveillancePipeline:
    """
    Main execution loop coupling video intake, inference, and hardware payload actuation.
    """

    def __init__(
        self,
        camera: FrameSource,
        detector: ONNXDetector,
        actuator: ServoActuator,
        auto_trigger: bool = True,
        trigger_cooldown_seconds: float = 3.0,
    ) -> None:
        self.camera = camera
        self.detector = detector
        self.actuator = actuator
        self.auto_trigger = auto_trigger
        self.trigger_cooldown = trigger_cooldown_seconds
        self.last_trigger_time: float = 0.0

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Detection], bool]:
        """
        Executes detection on a single frame, draws overlays, and evaluates trigger condition.
        """
        detections = self.detector.detect(frame)
        triggered = False
        annotated_frame = frame.copy()

        now = time.time()
        if detections and (now - self.last_trigger_time >= self.trigger_cooldown):
            if self.auto_trigger:
                self.actuator.trigger_release()
                self.last_trigger_time = now
                triggered = True

        if cv2 is not None:
            for det in detections:
                x1, y1, x2, y2 = det.box
                color = (0, 0, 255) if triggered else (0, 255, 0)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{det.class_name}: {det.confidence:.2f}"
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            status_text = "STATUS: TRIGGERED PAYLOAD" if triggered else "STATUS: PATROLLING"
            cv2.putText(
                annotated_frame,
                status_text,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255) if triggered else (0, 255, 0),
                2,
            )

        return annotated_frame, detections, triggered

    def run(self, max_frames: Optional[int] = None, callback: Optional[Callable] = None) -> None:
        """
        Executes continuous processing until stream exhausts or maximum frames processed.
        """
        logger.info("Initiating surveillance pipeline loop...")
        for frame in self.camera.stream(max_frames=max_frames):
            annotated, detections, triggered = self.process_frame(frame)
            if callback:
                callback(annotated, detections, triggered)

    def close(self) -> None:
        """
        Releases pipeline hardware resources cleanly.
        """
        self.camera.close()
        self.actuator.close()
        logger.info("Surveillance pipeline stopped cleanly.")
