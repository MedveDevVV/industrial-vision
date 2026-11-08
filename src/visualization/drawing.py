import cv2

from src.inference.types import Detection


def draw_detections(frame, detections: list[Detection]):
    for detection in detections:
        cv2.rectangle(
            frame,
            (detection.x1, detection.y1),
            (detection.x2, detection.y2),
            (0, 255, 0),
            2
        )

        label = (
            f"{detection.class_name}: "
            f"{detection.confidence:.2f}"
        )

        cv2.putText(
            frame,
            label,
            (detection.x1, detection.y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
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