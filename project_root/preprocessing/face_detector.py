import cv2
import mediapipe as mp

class FaceDetector:
    def __init__(self, min_conf=0.5):
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(
            model_selection=0,  # short-range model (best for emotion)
            min_detection_confidence=min_conf
        )

    def detect(self, img):
        """
        Detects face bounding box.
        Returns [x1, y1, x2, y2] or None.
        """
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)

        if not results.detections:
            return None

        d = results.detections[0]
        box = d.location_data.relative_bounding_box

        x1 = int(box.xmin * w)
        y1 = int(box.ymin * h)
        x2 = int((box.xmin + box.width) * w)
        y2 = int((box.ymin + box.height) * h)

        # clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        return [x1, y1, x2, y2]
