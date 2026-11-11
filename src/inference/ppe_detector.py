from pathlib import Path

from ultralytics import YOLO

from src.inference.types import Detection, PPEDetection


class PPEDetector:
    def __init__(
        self,
        model_path: str,
        confidence: float = 0.25
    ):
        project_root = Path(__file__).resolve().parents[2]

        model_file = Path(model_path)
        if not model_file.is_absolute():
            model_file = project_root / model_file

        if not model_file.is_file():
            raise FileNotFoundError(
                f"PPE-модель не найдена: {model_file}"
            )

        self.model = YOLO(str(model_file))
        self.confidence = confidence

    def detect(
        self,
        frame,
        person: Detection
    ) -> list[PPEDetection]:

        height, width = frame.shape[:2]

        x1 = max(0, person.x1)
        y1 = max(0, person.y1)
        x2 = min(width, person.x2)
        y2 = min(height, person.y2)

        if x2 <= x1 or y2 <= y1:
            return []

        person_crop = frame[y1:y2, x1:x2]

        result = self.model.predict(
            person_crop,
            conf=self.confidence,
            verbose=False
        )[0]

        detections = []

        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())

            crop_x1, crop_y1, crop_x2, crop_y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            detection = PPEDetection(
                class_id=class_id,
                class_name=str(result.names[class_id]),
                confidence=confidence,
                x1=crop_x1 + x1,
                y1=crop_y1 + y1,
                x2=crop_x2 + x1,
                y2=crop_y2 + y1,
                person_track_id=person.track_id
            )

            detections.append(detection)

        return detections