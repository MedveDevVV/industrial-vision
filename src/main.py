import cv2
import time

from src.capture.video_capture import VideoSource
from src.capture.fps_meter import FPSMeter


# 0 — первая доступная камера.
# Вместо 0 можно указать путь к видеофайлу, например:
# VIDEO_SOURCE = "data/test_video.mp4"
VIDEO_SOURCE = 0


def main():
    video = VideoSource(VIDEO_SOURCE)
    fps_meter = FPSMeter()

    actual_frame = None
    last_update = time.perf_counter()
    update_interval = 0.33

    try:
        video.open()
        print("Видеопоток успешно открыт.")
        while True:
            frame = video.read()
            capture_fps = fps_meter.tick_capture()

            current_time = time.perf_counter()

            if actual_frame is None or current_time - last_update >= update_interval:
                actual_frame = frame.copy()
                last_update = current_time

                display_fps = fps_meter.tick_display()

                cv2.putText(
                    actual_frame,
                    f"Capture FPS: {capture_fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    actual_frame,
                    f"Display FPS: {display_fps:.1f}",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.imshow("Industrial Vision", actual_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except RuntimeError as error:
        print(f"Ошибка видеозахвата: {error}")

    finally:
        video.close()
        cv2.destroyAllWindows()
        print("Видеозахват остановлен.")


if __name__ == "__main__":
    main()