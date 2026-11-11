import cv2
import time
import argparse

from src.config import load_config

from src.capture.video_capture import VideoSource
from src.capture.frame_buffer import FrameBuffer
from src.capture.capture_worker import CaptureWorker
from src.capture.fps_meter import FPSMeter
from src.logger import setup_logger
from src.analytics.event_engine import EventEngine

from src.inference.detector import Detector

from src.visualization.drawing import (
    draw_detections,
    draw_metrics,
    draw_roi
)
from src.analytics.roi import (
    normalized_polygon_to_pixels,
    is_inside_roi
)


def main():
    config = load_config()
    logger = setup_logger(
        config["app"]["log_level"]
    )
    args = parse_arguments()

    # Настройки приложения.
    app_config = config["app"]
    video_config = config["video"]
    model_config = config["model"]
    inference_config = config["inference"]
    roi_config = config["roi"]

    # Аргументы командной строки имеют приоритет над YAML.
    source = (
        args.source
        if args.source is not None
        else video_config["source"]
    )

    update_interval = (
        args.update_interval
        if args.update_interval is not None
        else inference_config["update_interval_ms"]
    )

    update_interval /= 1000.0

    logger.info(f"Запуск приложения: {app_config['name']}")
    logger.info(f"Источник видео: {source}")

    video = VideoSource(
        source,
        reconnect_sec=video_config["reconnect_sec"],
        max_attempts=video_config["max_attempts"]
    )

    frame_buffer = FrameBuffer()

    try:
        detector = Detector(
            model_path=config["model"]["path"],
            confidence=config["model"]["confidence"],
            tracker_type=config["tracking"]["tracker"]
        )
        print("Модель YOLO успешно загружена.")
    except Exception:
        logger.exception("Ошибка загрузки модели YOLO")
        return

    fps_meter = FPSMeter()
    capture_worker = CaptureWorker(video, frame_buffer)
    event_engine = EventEngine(confirmation_frames=7)

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

                # Переводим координаты ROI в пиксели кадра.
                if roi_config["enabled"]:
                    height, width = actual_frame.shape[:2]

                    roi_polygon = normalized_polygon_to_pixels(
                        roi_config["points"],
                        width,
                        height
                    )

                    # Проверяем положение каждого человека.
                    for detection in detections:
                        detection.in_roi = is_inside_roi(
                            detection,
                            roi_polygon
                        )

                    draw_roi(actual_frame, roi_polygon)

                    # Проверяем условия возникновения событий.
                    events = event_engine.update(detections)

                    # Регистрируем подтверждённые нарушения.
                    for track_id in events:
                        logger.warning(
                            f"Нарушение: человек с Track ID {track_id} "
                            f"находится в запретной зоне."
                        )

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
                logger.error(f"Ошибка видеозахвата: {capture_worker.error}")
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
        logger.info("Видеозахват остановлен.")


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
