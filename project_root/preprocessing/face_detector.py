import cv2
import mediapipe as mp

class FaceDetector:
    def __init__(self, min_conf=0.5):
        # Load MediaPipe's Face Detection module
        self.mp_face = mp.solutions.face_detection
        
        # Create a face detector object using MediaPipe's short-range model (model_selection=0)
        # min_detection_confidence controls how confident the detector must be before returning a face
        self.detector = self.mp_face.FaceDetection(
            model_selection=0,  # short-range model ideal for faces close to the camera
            min_detection_confidence=min_conf
        )

    def detect(self, img):
        """
        Detects face bounding box.
        Returns [x1, y1, x2, y2] or None.
        """

        # Extract image height and width
        # MediaPipe returns bounding boxes in *relative* coordinates, so we need absolute pixel values
        h, w, _ = img.shape

        # MediaPipe expects RGB images, but OpenCV loads images in BGR
        # So convert BGR → RGB before processing
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run face detection model on the image
        results = self.detector.process(rgb)

        # If no face detections, return None
        if not results.detections:
            return None

        # Take the first detected face (MediaPipe may detect multiple)
        d = results.detections[0]

        # Extract the bounding box in relative format (values between 0 and 1)
        box = d.location_data.relative_bounding_box

        # Convert relative bounding box coordinates to absolute pixel coordinates
        x1 = int(box.xmin * w)
        y1 = int(box.ymin * h)
        x2 = int((box.xmin + box.width) * w)
        y2 = int((box.ymin + box.height) * h)

        # Ensure the coordinates do not go outside image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Return bounding box as [left, top, right, bottom]
        return [x1, y1, x2, y2]
