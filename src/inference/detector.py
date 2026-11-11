from pathlib import Path
from ultralytics import YOLO

from .types import Detection


class Detector:
    def __init__(self, model_path: str, confidence: float = 0.3):
        project_root = Path(__file__).resolve().parents[2]
        model_file = Path(model_path)

        if not model_file.is_absolute():
            model_file = project_root / model_file

        if not model_file.is_file():
            raise FileNotFoundError(
                f"Файл модели не найден: {model_file}"
            )

        self.model = YOLO(str(model_file))
        self.confidence = confidence

    def predict(self, frame) -> list[Detection]:
        result = self.model.track(
            frame,
            conf=self.confidence,
            classes=[0],
            persist=True,               # сохранение состояния трека между кадрами
            tracker="bytetrack.yaml",
            verbose=False
        )[0]

        names = result.names
        detections = []

        for index, box in enumerate(result.boxes):
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            track_id = (
                int(result.boxes.id[index].item())
                if result.boxes.id is not None
                else None
            )

            detection = Detection(
                class_id=class_id,
                class_name=str(names[class_id]),
                confidence=confidence,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                track_id=track_id
            )

            detections.append(detection)

        return detections