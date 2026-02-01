import cv2
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, '..', 'hand_landmarker.task')

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

class HandTracker:
    def __init__(self, num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def process(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.detector.detect(mp_image)
        return results

    def draw_landmarks(self, frame, hand_landmarks):
        """
        Draw hand landmarks on the frame.
        """
        h, w, _ = frame.shape

        landmarks = hand_landmarks  # already a list of 21 points

        # Draw connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            start_point = landmarks[start_idx]
            end_point = landmarks[end_idx]

            start_x, start_y = int(start_point.x * w), int(start_point.y * h)
            end_x, end_y = int(end_point.x * w), int(end_point.y * h)

            cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)

        # Draw landmark dots
        for landmark in landmarks:
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        return frame

    def close(self):
        self.detector.close()
