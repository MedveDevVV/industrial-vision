import threading
import time

from src.capture.video_capture import VideoSource
from src.capture.frame_buffer import FrameBuffer
from src.capture.fps_meter import FPSMeter


class CaptureWorker:
    def __init__(
        self,
        video: VideoSource,
        frame_buffer: FrameBuffer
    ):
        self.video = video
        self.frame_buffer = frame_buffer

        self.stop_event = threading.Event()
        self.thread = None

        self.fps_meter = FPSMeter()
        self.capture_fps = 0.0

        self.finished = False
        self.error = None

    def start(self) -> None:
        self.thread = threading.Thread(
            target=self._run,
            name="capture-worker"
        )
        self.thread.start()

    def _run(self) -> None:
        try:
            self.video.open()

            print("Видеопоток успешно открыт.")
            print(f"FPS источника: {self.video.fps}")

            next_frame_time = time.perf_counter()

            while not self.stop_event.is_set():
                frame = self.video.read()

                if frame is None:
                    print("Воспроизведение видео завершено.")
                    break

                self.frame_buffer.update(frame)
                self.capture_fps = self.fps_meter.tick_capture()

                # Синхронизация воспроизведения видеофайла.
                if self.video.is_file and self.video.fps > 0:
                    frame_interval = 1.0 / self.video.fps

                    next_frame_time += frame_interval

                    remaining_time = next_frame_time - time.perf_counter()

                    if remaining_time > 0:
                        if self.stop_event.wait(remaining_time):
                            break
                    else:
                        next_frame_time = time.perf_counter()

        except Exception as error:
            self.error = error

        finally:
            self.video.close()
            self.finished = True

    def stop(self) -> None:
        self.stop_event.set()

    def join(self) -> None:
        if self.thread is not None:
            self.thread.join()