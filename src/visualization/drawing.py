import cv2
import numpy as np

from src.inference.types import Detection
from src.inference.types import PPEDetection, PPEAssessment, PPEStatus


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


def draw_ppe(
    frame,
    detections: list[PPEDetection],
    assessments: dict[int, PPEAssessment]
):
    colors = {
        "helmet": (0, 255, 0),
        "no_helmet": (0, 0, 255),
        "vest": (0, 255, 0),
        "no_vest": (0, 0, 255)
    }

    for detection in detections:
        color = colors.get(detection.class_name, (255, 255, 255))

        cv2.rectangle(
            frame,
            (detection.x1, detection.y1),
            (detection.x2, detection.y2),
            color,
            2
        )

        label = (
            f"{detection.class_name}: "
            f"{detection.confidence:.2f}"
        )

        cv2.putText(
            frame,
            label,
            (detection.x1, max(20, detection.y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    for track_id, assessment in assessments.items():
        if assessment.helmet == PPEStatus.VIOLATION:
            label = "NO HELMET"
        elif assessment.helmet == PPEStatus.UNKNOWN:
            label = "HELMET: CHECK"
        else:
            label = "HELMET: OK"

        if assessment.vest == PPEStatus.VIOLATION:
            label += " | NO VEST"
        elif assessment.vest == PPEStatus.UNKNOWN:
            label += " | VEST: CHECK"
        else:
            label += " | VEST: OK"

        cv2.putText(
            frame,
            f"ID {track_id}: {label}",
            (10, 100 + track_id * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255)
            if (
                assessment.helmet == PPEStatus.VIOLATION
                or assessment.vest == PPEStatus.VIOLATION
            )
            else (0, 255, 0),
            2
        )

    return frame