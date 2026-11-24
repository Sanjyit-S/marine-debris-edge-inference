"""
ONNX Edge Inference Engine & Preprocessing
Executes optimized neural object detection with NMS and mock inference support.
"""

from typing import List, Tuple, Optional, NamedTuple
import numpy as np
import logging

try:
    import cv2
except Exception:
    cv2 = None

logger = logging.getLogger(__name__)

class Detection(NamedTuple):
    """
    Standardized bounding box detection representation.
    """
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str


class ONNXDetector:
    """
    Generic edge inference wrapper for ONNX object detection architectures.
    Provides frame preprocessing, non-maximum suppression, and simulated inference.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        conf_threshold: float = 0.40,
        iou_threshold: float = 0.45,
        target_size: Tuple[int, int] = (640, 640),
        mock_mode: bool = False,
    ) -> None:
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_size = target_size
        self.mock_mode = mock_mode or (weights_path is None)
        self.session = None
        self.input_name = None

        if not self.mock_mode and weights_path:
            try:
                import onnxruntime as ort  # type: ignore
                self.session = ort.InferenceSession(
                    weights_path,
                    providers=["CPUExecutionProvider"],
                )
                self.input_name = self.session.get_inputs()[0].name
                logger.info("ONNX inference session initialized with model: %s", weights_path)
            except Exception as err:
                logger.warning("Unable to initialize ONNX session (%s). Activating mock mode.", err)
                self.mock_mode = True

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Resizes, converts color space, and normalizes a frame to NCHW float32 format.
        """
        if cv2 is not None:
            resized = cv2.resize(frame, self.target_size)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        else:
            # Fallback numpy resizing
            rgb = frame[:self.target_size[1], :self.target_size[0]]
            if rgb.shape[0] != self.target_size[1] or rgb.shape[1] != self.target_size[0]:
                padded = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=frame.dtype)
                padded[:rgb.shape[0], :rgb.shape[1]] = rgb
                rgb = padded
        normalized = rgb.astype(np.float32) / 255.0
        transposed = np.transpose(normalized, (2, 0, 1))
        batched = np.expand_dims(transposed, axis=0)
        return batched

    @staticmethod
    def compute_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """
        Standard greedy Non-Maximum Suppression algorithm.
        """
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep: List[int] = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h

            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / (union + 1e-8)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Runs inference on an input frame and returns filtered detections.
        """
        if self.mock_mode:
            # Generate a deterministic mock detection for verification
            h, w = frame.shape[:2]
            return [
                Detection(
                    box=(int(w * 0.3), int(h * 0.3), int(w * 0.6), int(h * 0.6)),
                    confidence=0.88,
                    class_id=0,
                    class_name="marine_debris",
                )
            ]

        if self.session is None or self.input_name is None:
            return []

        tensor = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: tensor})
        raw_preds = outputs[0]  # Shape: [1, num_classes+4, num_proposals] or [1, num_proposals, num_classes+4]

        # Squeeze batch dimension
        preds = np.squeeze(raw_preds)
        if preds.shape[0] < preds.shape[1]:
            preds = np.transpose(preds)

        boxes = []
        scores = []
        class_ids = []

        orig_h, orig_w = frame.shape[:2]
        scale_x = orig_w / self.target_size[0]
        scale_y = orig_h / self.target_size[1]

        for pred in preds:
            class_scores = pred[4:]
            max_idx = int(np.argmax(class_scores))
            conf = float(class_scores[max_idx])

            if conf >= self.conf_threshold:
                cx, cy, bw, bh = pred[:4]
                x1 = int((cx - bw / 2.0) * scale_x)
                y1 = int((cy - bh / 2.0) * scale_y)
                x2 = int((cx + bw / 2.0) * scale_x)
                y2 = int((cy + bh / 2.0) * scale_y)

                boxes.append([x1, y1, x2, y2])
                scores.append(conf)
                class_ids.append(max_idx)

        if not boxes:
            return []

        boxes_arr = np.array(boxes, dtype=np.float32)
        scores_arr = np.array(scores, dtype=np.float32)
        keep_indices = self.compute_nms(boxes_arr, scores_arr, self.iou_threshold)

        detections = []
        for idx in keep_indices:
            detections.append(
                Detection(
                    box=tuple(int(v) for v in boxes[idx]),
                    confidence=scores[idx],
                    class_id=class_ids[idx],
                    class_name="marine_target",
                )
            )

        return detections
