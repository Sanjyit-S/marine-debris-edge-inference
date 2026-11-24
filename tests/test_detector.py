"""
Unit tests for ONNXDetector preprocessing, NMS algorithm, and mock inference.
"""

import numpy as np
from marine_edge.detector import ONNXDetector

def test_detector_mock_mode():
    detector = ONNXDetector(weights_path=None, conf_threshold=0.5, mock_mode=True)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(frame)

    assert len(detections) == 1
    assert detections[0].confidence >= 0.5
    assert detections[0].class_name == "marine_debris"
    assert len(detections[0].box) == 4

def test_preprocessing_tensor_shape():
    detector = ONNXDetector(mock_mode=True, target_size=(320, 320))
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    tensor = detector.preprocess(frame)

    assert tensor.shape == (1, 3, 320, 320)
    assert tensor.dtype == np.float32
    assert tensor.max() <= 1.0

def test_nms_filtering():
    # Two heavily overlapping boxes, higher confidence should suppress lower
    boxes = np.array([
        [10.0, 10.0, 100.0, 100.0],
        [12.0, 12.0, 98.0, 98.0],
        [200.0, 200.0, 300.0, 300.0],
    ], dtype=np.float32)
    scores = np.array([0.90, 0.75, 0.85], dtype=np.float32)

    keep = ONNXDetector.compute_nms(boxes, scores, iou_threshold=0.40)
    assert 0 in keep  # Best box in cluster 1
    assert 1 not in keep  # Suppressed box
    assert 2 in keep  # Independent box
