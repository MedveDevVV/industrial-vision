
from src.inference.types import Detection


class EventEngine:
    def __init__(self, confirmation_frames: int = 3):
        self.confirmation_frames = confirmation_frames
        self.frame_counts: dict[int, int] = {}
        self.active_tracks: set[int] = set()

    def update(self, detections: list[Detection]) -> list[int]:
        events = []
        observed_tracks = set()

        for detection in detections:
            track_id = detection.track_id

            if track_id is None:
                continue

            observed_tracks.add(track_id)

            if not detection.in_roi:
                self.frame_counts[track_id] = 0
                self.active_tracks.discard(track_id)
                continue

            self.frame_counts[track_id] = (
                self.frame_counts.get(track_id, 0) + 1
            )

            if (
                self.frame_counts[track_id] >= self.confirmation_frames
                and track_id not in self.active_tracks
            ):
                events.append(track_id)
                self.active_tracks.add(track_id)

        # Сбрасываем подтверждение для пропавших с текущего кадра треков.
        missing_tracks = set(self.frame_counts) - observed_tracks

        for track_id in missing_tracks:
            self.frame_counts[track_id] = 0
            self.active_tracks.discard(track_id)

        return events
