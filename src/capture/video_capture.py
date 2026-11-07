import time as time_module

import cv2
import numpy as np

# Управление одним источником видеопотока
class VideoSource:
    def __init__(self, source: str | int, reconnect_sec: float = 2.0, max_attempts: int = 5):
        self.source = source
        self.reconnect_sec = reconnect_sec
        self.max_attempts = max_attempts
        self.cap: cv2.VideoCapture | None = None # Поле хранящее объект OpenCV, управляющий открытым видеоисточником

    def open(self) -> None:
        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            raise RuntimeError(f"Could not open video source {self.source}")


    def read(self) -> np.ndarray:
        for attempt in range(1, self.max_attempts + 1):
            try:
                if self.cap is None or not self.cap.isOpened():
                    self.open()

                if self.cap is None:
                    raise RuntimeError("Video capture is not initialized")

                ok, frame = self.cap.read()

                if ok and frame is not None:
                    return frame

                raise RuntimeError("Failed to receive video frame")

            except RuntimeError as error:
                self.close()

                if attempt == self.max_attempts:
                    raise RuntimeError(f"Video source unavailable after "
                                       f"{self.max_attempts} attempts"
                                       ) from error

                print(
                    f"Video source unavailable. "
                    f"Attempt {attempt}/{self.max_attempts}. "
                    f"Retrying in {self.reconnect_sec} seconds..."
                )

                time_module.sleep(self.reconnect_sec)


    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None