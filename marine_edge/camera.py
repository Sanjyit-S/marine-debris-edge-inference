"""
Video & Camera Frame Acquisition Pipeline
Supports physical camera modules, recorded video streams, and synthetic frames for testing.
"""

from typing import Generator, Tuple, Optional, Union
import numpy as np
import logging

try:
    import cv2
except Exception:
    cv2 = None

logger = logging.getLogger(__name__)

class FrameSource:
    """
    Manages video frame acquisition with robust error handling and mock stream generation.
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        target_size: Tuple[int, int] = (640, 640),
        mock_mode: bool = False,
    ) -> None:
        self.source = source
        self.target_size = target_size
        self.mock_mode = mock_mode
        self._cap: Optional[cv2.VideoCapture] = None

        if not self.mock_mode:
            try:
                # If numeric string passed, convert to int
                cam_id = int(self.source) if isinstance(self.source, str) and self.source.isdigit() else self.source
                self._cap = cv2.VideoCapture(cam_id)
                if not self._cap.isOpened():
                    logger.warning("Unable to open video source '%s'. Falling back to mock generator.", source)
                    self.mock_mode = True
            except Exception as err:
                logger.warning("Error initializing video source '%s': %s. Using mock mode.", source, err)
                self.mock_mode = True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Reads a single raw frame from the configured video source.
        """
        if self.mock_mode:
            # Generate a clean 640x640 mock frame representing water surface
            frame = np.full((self.target_size[1], self.target_size[0], 3), 120, dtype=np.uint8)
            if cv2 is not None:
                cv2.putText(
                    frame,
                    "MOCK MARINE SURVEILLANCE FEED",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
            return True, frame

        if self._cap is not None and self._cap.isOpened():
            ret, frame = self._cap.read()
            return ret, frame

        return False, None

    def stream(self, max_frames: Optional[int] = None) -> Generator[np.ndarray, None, None]:
        """
        Yields frames sequentially until end of stream or maximum frame limit is reached.
        """
        count = 0
        while True:
            if max_frames is not None and count >= max_frames:
                break

            ret, frame = self.read_frame()
            if not ret or frame is None:
                break

            count += 1
            yield frame

    def close(self) -> None:
        """
        Releases the underlying video capture hardware.
        """
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Video capture source released.")
