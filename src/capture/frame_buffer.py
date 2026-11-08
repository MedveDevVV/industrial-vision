import threading

import numpy as np


class FrameBuffer:
    def __init__(self):
        self.frame: np.ndarray | None = None
        self.frame_id = 0
        self.lock = threading.Lock()

    def update(self, frame: np.ndarray) -> None:
        with self.lock:
            self.frame = frame
            self.frame_id += 1

    def get_latest(self) -> tuple[np.ndarray | None, int]:
        with self.lock:
            return self.frame, self.frame_id