import cv2
import time

from src.capture.video_capture import VideoSource
from src.capture.frame_buffer import FrameBuffer
from src.capture.capture_worker import CaptureWorker
from src.capture.fps_meter import FPSMeter

from src.analytics.event_engine import EventEngine
from src.analytics.ppe_assessor import PPEAssessor
from src.analytics.ppe_event_engine import PPEEventEngine
from src.analytics.roi import (
    normalized_polygon_to_pixels,
    is_inside_roi
)

from src.inference.detector import Detector
from src.inference.ppe_detector import PPEDetector

from src.visualization.drawing import (
    draw_detections,
    draw_metrics,
    draw_roi,
    draw_ppe
)


class Application:
    def __init__(self, config, source, update_interval, logger):
        self.config = config
        self.source = source
        self.update_interval = update_interval / 1000.0
        self.logger = logger

        video_config = config["video"]
        model_config = config["model"]
        ppe_config = config["ppe"]

        self.video = VideoSource(
            source,
            reconnect_sec=video_config["reconnect_sec"],
            max_attempts=video_config["max_attempts"]
        )

        self.frame_buffer = FrameBuffer()

        try:
            self.detector = Detector(
                model_path=model_config["path"],
                confidence=model_config["confidence"],
                tracker_type=config["tracking"]["tracker"]
            )
            self.logger.info("Модель YOLO успешно загружена.")

            if ppe_config["enabled"]:
                self.ppe_detector = PPEDetector(
                    model_path=ppe_config["model_path"],
                    confidence=ppe_config["confidence"]
                )
                self.logger.info("PPE-модель успешно загружена.")
            else:
                self.ppe_detector = None

        except Exception:
            self.logger.exception("Ошибка загрузки моделей")
            raise

        self.fps_meter = FPSMeter()
        self.capture_worker = CaptureWorker(
            self.video,
            self.frame_buffer
        )

        self.event_engine = EventEngine(confirmation_frames=7)
        self.ppe_assessor = PPEAssessor()

        self.ppe_event_engine = PPEEventEngine(
            confirmation_frames=ppe_config["confirmation_frames"]
        )

        self.actual_frame = None
        self.last_processed_frame_id = -1
        self.next_update_time = 0.0
        self.pipeline_fps = 0.0

    def _process_frame(self, frame):
        actual_frame = frame.copy()

        # Распознаём объекты.
        detections = self.detector.predict(actual_frame)

        # Проверяем средства индивидуальной защиты.
        ppe_detections, ppe_assessments = self._process_ppe(
            actual_frame,
            detections
        )

        # Проверяем положение людей в ROI.
        self._process_roi(actual_frame, detections)

        # Отрисовываем результаты.
        draw_detections(actual_frame, detections)
        draw_ppe(actual_frame, ppe_detections, ppe_assessments)

        return actual_frame

    def _process_ppe(self, frame, detections):
        ppe_detections = []
        ppe_assessments = {}

        if self.ppe_detector is None:
            return ppe_detections, ppe_assessments

        min_person_height = self.config["ppe"]["min_person_height"]

        for person in detections:
            if person.y2 - person.y1 < min_person_height:
                continue

            person_ppe = self.ppe_detector.detect(frame, person)
            ppe_detections.extend(person_ppe)

            if person.track_id is not None:
                ppe_assessments[person.track_id] = (
                    self.ppe_assessor.assess(person_ppe)
                )

        ppe_events = self.ppe_event_engine.update(
            detections,
            ppe_assessments
        )

        for track_id, event_type in ppe_events:
            self.logger.warning(
                f"Нарушение СИЗ: {event_type}, Track ID: {track_id}"
            )

        return ppe_detections, ppe_assessments

    def _process_roi(self, frame, detections):
        roi_config = self.config.get("roi", {})

        if not roi_config.get("enabled", False) or not roi_config.get("points"):
            return

        height, width = frame.shape[:2]

        roi_polygon = normalized_polygon_to_pixels(
            roi_config["points"],
            width,
            height
        )

        for detection in detections:
            detection.in_roi = is_inside_roi(
                detection,
                roi_polygon
            )

        draw_roi(frame, roi_polygon)

        events = self.event_engine.update(detections)

        for track_id in events:
            self.logger.warning(
                f"Нарушение: человек с Track ID {track_id} "
                f"находится в запретной зоне."
            )

    def run(self):
        roi_config = self.config.get("roi", {})

        try:
            self.capture_worker.start()

            while True:
                frame, frame_id = self.frame_buffer.get_latest()
                current_time = time.perf_counter()

                # Проверяем, можно ли обработать новый кадр.
                if (
                        frame is not None  # Кадр успешно получен из буфера.
                        and frame_id != self.last_processed_frame_id  # Этот кадр ещё не обрабатывался.
                        and current_time >= self.next_update_time  # Наступило время следующей обработки.
                ):
                    # Запускаем полный цикл обработки кадра:
                    # распознавание объектов, анализ СИЗ, проверку ROI и отрисовку результатов.
                    self.actual_frame = self._process_frame(frame)

                    # Запоминаем ID обработанного кадра, чтобы не обрабатывать его повторно.
                    self.last_processed_frame_id = frame_id

                    # Назначаем время следующей обработки с учётом заданного интервала.
                    self.next_update_time = current_time + self.update_interval

                    # Обновляем FPS конвейера — количество обработанных кадров в секунду.
                    self.pipeline_fps = self.fps_meter.tick_display()

                # Отображаем последний обработанный кадр.
                if self.actual_frame is not None:
                    display_frame = self.actual_frame.copy()

                    draw_metrics(display_frame, self.capture_worker.capture_fps, self.pipeline_fps)

                    cv2.imshow("Industrial Vision", display_frame)

                # Проверяем ошибки захвата.
                if self.capture_worker.error is not None:
                    self.logger.error(
                        f"Ошибка видеозахвата: {self.capture_worker.error}"
                    )
                    break

                # Проверяем завершение видеопотока.
                if self.capture_worker.finished:
                    if frame is None or frame_id == self.last_processed_frame_id:
                        break

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            self.capture_worker.stop()
            self.capture_worker.join()

            cv2.destroyAllWindows()
            self.logger.info("Видеозахват остановлен.")
