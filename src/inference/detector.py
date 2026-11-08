from ultralytics import YOLO

from .types import Detection


class Detector:
    def __init__(self, model_path: str, confidence: float = 0.5):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def predict(self, frame) -> list[Detection]:
        result = self.model.predict(
            frame,
            conf=self.confidence,
            verbose=False
        )[0]

        names = result.names
        detections = []

        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            detection = Detection(
                class_id=class_id,
                class_name=str(names[class_id]),
                confidence=confidence,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2
            )

            detections.append(detection)

        return detections