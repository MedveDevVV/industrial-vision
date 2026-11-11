import cv2
import numpy as np

from src.inference.types import Detection


def normalized_polygon_to_pixels(
    points: list[list[float]],
    width: int,
    height: int
) -> list[tuple[int, int]]:
    return [
        (int(x * width), int(y * height))
        for x, y in points
    ]


def bottom_center(detection: Detection) -> tuple[int, int]:
    return (
        (detection.x1 + detection.x2) // 2,
        detection.y2
    )


def is_inside_roi(
    detection: Detection,
    polygon: list[tuple[int, int]]
) -> bool:
    contour = np.array(polygon, dtype=np.int32)
    point = bottom_center(detection)

    return cv2.pointPolygonTest(contour, point, False) >= 0
