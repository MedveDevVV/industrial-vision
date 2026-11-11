
from src.inference.types import Detection, PPEAssessment, PPEStatus


class PPEEventEngine:
    def __init__(self, confirmation_frames: int = 3):
        self.confirmation_frames = confirmation_frames

        self.frame_counts: dict[tuple[int, str], int] = {}
        self.active_events: set[tuple[int, str]] = set()

    def update(
        self,
        detections: list[Detection],
        assessments: dict[int, PPEAssessment]
    ) -> list[tuple[int, str]]:
        events = []
        observed_tracks = set()

        for detection in detections:
            track_id = detection.track_id

            if track_id is None:
                continue

            observed_tracks.add(track_id)
            assessment = assessments.get(track_id)

            if assessment is None:
                helmet_status = PPEStatus.UNKNOWN
                vest_status = PPEStatus.UNKNOWN
            else:
                helmet_status = assessment.helmet
                vest_status = assessment.vest

            checks = {
                "ppe_helmet_missing": helmet_status,
                "ppe_vest_missing": vest_status
            }

            for event_type, status in checks.items():
                key = (track_id, event_type)

                if status == PPEStatus.VIOLATION:
                    self.frame_counts[key] = (
                        self.frame_counts.get(key, 0) + 1
                    )

                    if (
                        self.frame_counts[key] >= self.confirmation_frames
                        and key not in self.active_events
                    ):
                        events.append((track_id, event_type))
                        self.active_events.add(key)
                else:
                    self.frame_counts[key] = 0
                    self.active_events.discard(key)

        missing_tracks = (
            {track_id for track_id, _ in self.frame_counts}
            - observed_tracks
        )

        for track_id in missing_tracks:
            for key in list(self.frame_counts):
                if key[0] == track_id:
                    self.frame_counts[key] = 0
                    self.active_events.discard(key)

        return events
