import cv2
import numpy as np

from preprocessing.face_detector import FaceDetector
from preprocessing.face_align import FaceAligner

class FacePreprocessor:
    def __init__(self):
        self.detector = FaceDetector()
        self.aligner = FaceAligner()

    def preprocess(self, img):
        """
        Full pipeline:
        1. Face detect
        2. Face align
        3. Crop
        4. Resize to 224x224
        5. Normalize
        """
        box = self.detector.detect(img)
        if box is None:
            return None

        x1, y1, x2, y2 = box
        face = img[y1:y2, x1:x2]
        box = self.detector.detect(img)
        if box is None:

    # fallback: use entire image
            resized = cv2.resize(img, (224, 224))
            normalized = resized.astype(np.float32) / 255.0
            return normalized


        

        
