
from src.inference.types import (
    PPEDetection,
    PPEAssessment,
    PPEStatus
)


class PPEAssessor:

    def assess(
        self,
        detections: list[PPEDetection]
    ) -> PPEAssessment:

        classes = {
            detection.class_name
            for detection in detections
        }

        # Определяем состояние каски.
        if "no_helmet" in classes:
            helmet_status = PPEStatus.VIOLATION
        elif "helmet" in classes:
            helmet_status = PPEStatus.COMPLIANT
        else:
            helmet_status = PPEStatus.UNKNOWN

        # Определяем состояние жилета.
        if "no_vest" in classes:
            vest_status = PPEStatus.VIOLATION
        elif "vest" in classes:
            vest_status = PPEStatus.COMPLIANT
        else:
            vest_status = PPEStatus.UNKNOWN

        return PPEAssessment(
            helmet=helmet_status,
            vest=vest_status
        )
