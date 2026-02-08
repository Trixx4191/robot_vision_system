from ultralytics import YOLO

class YOLODetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")  # lightweight & fast run

    def detect(self, frame):
        results = self.model(frame, verbose=False)
        return results[0]
