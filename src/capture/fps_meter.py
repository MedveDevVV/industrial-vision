import time
from collections import deque


class FPSMeter:
    def __init__(self, window: int = 30):
        self.capture_times = deque(maxlen=window)
        self.display_times = deque(maxlen=window)

    def _calculate_fps(self, times: deque) -> float:
        if len(times) < 2:
            return 0.0

        dt = times[-1] - times[0]

        return (len(times) - 1) / dt if dt > 0 else 0.0

    def tick_capture(self) -> float:
        self.capture_times.append(time.perf_counter())
        return self._calculate_fps(self.capture_times)

    def tick_display(self) -> float:
        self.display_times.append(time.perf_counter())
        return self._calculate_fps(self.display_times)