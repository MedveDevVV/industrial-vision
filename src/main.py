import cv2
import time
import argparse

from src.capture.video_capture import VideoSource
from src.capture.frame_buffer import FrameBuffer
from src.capture.capture_worker import CaptureWorker
from src.capture.fps_meter import FPSMeter

from src.inference.detector import Detector

from src.visualization.drawing import (
    draw_detections,
    draw_metrics
)


MODEL_PATH = "yolo11n.pt"


def main():
    args = parse_arguments()

    video = VideoSource(args.source)
    frame_buffer = FrameBuffer()

    update_interval = args.update_interval / 1000.0

    try:
        detector = Detector(MODEL_PATH, confidence=0.5)
        print("Модель YOLO успешно загружена.")
    except Exception as error:
        print(f"Ошибка загрузки модели YOLO: {error}")
        return

    fps_meter = FPSMeter()
    capture_worker = CaptureWorker(video, frame_buffer)

    actual_frame = None
    last_processed_frame_id = -1

    next_update_time = 0.0
    pipeline_fps = 0.0

    try:
        capture_worker.start()

        while True:
            frame, frame_id = frame_buffer.get_latest()

            current_time = time.perf_counter()

            # Проверяем наличие нового кадра и интервал обработки.
            if (
                frame is not None
                and frame_id != last_processed_frame_id
                and current_time >= next_update_time
            ):
                actual_frame = frame.copy()

                # Распознаём объекты.
                detections = detector.predict(actual_frame)

                # Отрисовываем результаты распознавания.
                draw_detections(actual_frame, detections)

                last_processed_frame_id = frame_id
                next_update_time = current_time + update_interval

                pipeline_fps = fps_meter.tick_display()

            # Отображаем последний обработанный кадр.
            if actual_frame is not None:
                display_frame = actual_frame.copy()

                draw_metrics(
                    display_frame,
                    capture_worker.capture_fps,
                    pipeline_fps
                )

                cv2.imshow("Industrial Vision", display_frame)

            # Проверяем ошибки захвата.
            if capture_worker.error is not None:
                print(f"Ошибка видеозахвата: {capture_worker.error}")
                break

            # Проверяем завершение видеопотока.
            if capture_worker.finished:
                if frame is None or frame_id == last_processed_frame_id:
                    break

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        capture_worker.stop()
        capture_worker.join()

        cv2.destroyAllWindows()
        print("Видеозахват остановлен.")


def parse_source(value: str) -> str | int:
    try:
        return int(value)
    except ValueError:
        return value


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Industrial Vision — система технического зрения"
    )

    parser.add_argument(
        "--source",
        type=parse_source,
        default=0,
        help="Источник видео: номер камеры или путь к видеофайлу (default=0)"
    )

    parser.add_argument(
        "--update-interval",
        type=float,
        default=200,
        help="Интервал между запусками YOLO в миллисекундах (default=200)"
    )

    args = parser.parse_args()

    if args.update_interval <= 0:
        parser.error("--update-interval должен быть больше нуля")

    return args


if __name__ == "__main__":
    main()