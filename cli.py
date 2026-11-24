#!/usr/bin/env python3
"""
Marine Debris Edge Inference CLI
Production entrypoint for real-time UAV surveillance and payload release actuation.
"""

import argparse
import sys
import logging

try:
    import cv2
except Exception:
    cv2 = None

from marine_edge.actuator import ServoActuator
from marine_edge.camera import FrameSource
from marine_edge.detector import ONNXDetector
from marine_edge.pipeline import SurveillancePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("marine_edge_cli")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Edge AI Coastal Surveillance & Automated Payload Deployment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video capture source: camera index (e.g. '0') or path to video file",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to trained object detection weights (.onnx format). If omitted, mock mode activates.",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.40,
        help="Confidence threshold for bounding box filtering",
    )
    parser.add_argument(
        "--servo-pin",
        type=int,
        default=18,
        help="Hardware PWM GPIO pin connected to payload servo",
    )
    parser.add_argument(
        "--neutral-angle",
        type=float,
        default=60.0,
        help="Holding / neutral angle for release servo (degrees)",
    )
    parser.add_argument(
        "--release-angle",
        type=float,
        default=140.0,
        help="Trigger / release angle for marker deployment (degrees)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without hardware servo actuation (safe testing mode)",
    )
    parser.add_argument(
        "--mock-inference",
        action="store_true",
        help="Use synthetic detections for validation without requiring model weights",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process before terminating",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying GUI viewport windows",
    )
    return parser

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logger.info("Initializing Edge Surveillance System...")
    logger.info("Source: %s | Weights: %s | Dry-Run: %s", args.source, args.weights, args.dry_run)

    camera = FrameSource(
        source=args.source,
        mock_mode=args.mock_inference and not args.source.isdigit(),
    )
    detector = ONNXDetector(
        weights_path=args.weights,
        conf_threshold=args.conf_threshold,
        mock_mode=args.mock_inference or (args.weights is None),
    )
    actuator = ServoActuator(
        pin=args.servo_pin,
        neutral_angle=args.neutral_angle,
        release_angle=args.release_angle,
        dry_run=args.dry_run,
    )

    pipeline = SurveillancePipeline(
        camera=camera,
        detector=detector,
        actuator=actuator,
    )

    frame_count = 0

    def frame_callback(annotated_frame, detections, triggered):
        nonlocal frame_count
        frame_count += 1
        if detections:
            logger.info(
                "Frame %d: Detected %d target(s) | Triggered: %s",
                frame_count,
                len(detections),
                triggered,
            )

        if not args.headless and cv2 is not None:
            cv2.imshow("Marine Surveillance Edge Viewport", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("User requested termination.")
                sys.exit(0)

    try:
        pipeline.run(max_frames=args.max_frames, callback=frame_callback)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        pipeline.close()
        if not args.headless and cv2 is not None:
            cv2.destroyAllWindows()

    logger.info("Session complete. Processed %d frames.", frame_count)
    return 0

if __name__ == "__main__":
    sys.exit(main())
