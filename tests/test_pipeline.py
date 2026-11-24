"""
Unit tests for end-to-end surveillance pipeline execution and trigger cooldown.
"""

from marine_edge.camera import FrameSource
from marine_edge.detector import ONNXDetector
from marine_edge.actuator import ServoActuator
from marine_edge.pipeline import SurveillancePipeline

def test_pipeline_execution():
    camera = FrameSource(mock_mode=True)
    detector = ONNXDetector(mock_mode=True)
    actuator = ServoActuator(dry_run=True)

    pipeline = SurveillancePipeline(
        camera=camera,
        detector=detector,
        actuator=actuator,
        auto_trigger=True,
        trigger_cooldown_seconds=1.0,
    )

    processed_frames = []

    def callback(annotated, detections, triggered):
        processed_frames.append((len(detections), triggered))

    pipeline.run(max_frames=3, callback=callback)
    pipeline.close()

    assert len(processed_frames) == 3
    # First frame should trigger release
    assert processed_frames[0][1] is True
    # Subsequent immediate frames should respect cooldown
    assert processed_frames[1][1] is False
