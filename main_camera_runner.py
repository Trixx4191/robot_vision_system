import cv2
from object_detection.yolo_detector import YOLODetector
from hand_gesture.hand_tracker import HandTracker
from face_id.face_module import FaceRecognition

yolo = YOLODetector()
hands = HandTracker()
faces = FaceRecognition()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #  Object Detection 
    yolo_results = yolo.detect(frame)
    annotated_frame = yolo_results.plot()
    
    #  Hand Tracking 
    hand_results = hands.process(frame)
    if hand_results.hand_landmarks is not None:
        for hand_landmarks in hand_results.hand_landmarks:
            annotated_frame = hands.draw_landmarks(annotated_frame, hand_landmarks)

    #  Face Recognition 
    locations, names = faces.recognize(frame)
    for (top, right, bottom, left), name in zip(locations, names):
        cv2.rectangle(annotated_frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(annotated_frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Robot Vision System", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
