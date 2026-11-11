import cv2
import numpy as np

from src.inference.types import Detection


def draw_detections(frame, detections: list[Detection]):
    for detection in detections:
        color = (0, 0, 255) if detection.in_roi else (0, 255, 0)

        cv2.rectangle(
            frame,
            (detection.x1, detection.y1),
            (detection.x2, detection.y2),
            color,
            2
        )

        point = (
            (detection.x1 + detection.x2) // 2,
            detection.y2
        )

        cv2.circle(frame, point, 5, color, -1)

        label = (
            f"{detection.class_name}: "
            f"ID:{detection.track_id} "
            f"{detection.confidence:.2f}"
        )

        cv2.putText(
            frame,
            label,
            (detection.x1, detection.y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return frame


def draw_metrics(
    frame,
    capture_fps: float,
    pipeline_fps: float
):
    cv2.putText(
        frame,
        f"Capture FPS: {capture_fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Pipeline FPS: {pipeline_fps:.1f}",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    return frame

def draw_roi(frame, polygon: list[tuple[int, int]]):
    points = np.array(polygon, dtype=np.int32)

    cv2.polylines(
        frame,
        [points],
        isClosed=True,
        color=(0, 0, 255),
        thickness=2
    )

    return frame