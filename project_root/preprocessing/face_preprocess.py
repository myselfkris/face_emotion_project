# preprocessing/face_preprocess.py
import cv2
import numpy as np
import os

from preprocessing.face_detector import FaceDetector
from preprocessing.face_align import FaceAligner

class FacePreprocessor:
    def __init__(self, upsample_size=224):
        self.detector = FaceDetector()
        self.aligner = FaceAligner()
        self.upsample_size = upsample_size

    def _ensure_color_and_size(self, img):
        # If grayscale -> convert to RGB
        if img is None:
            return None
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # If very small, upscale first to help detection
        h, w = img.shape[:2]
        if h < 100 or w < 100:
            img = cv2.resize(img, (self.upsample_size, self.upsample_size), interpolation=cv2.INTER_LINEAR)
        return img

    def preprocess(self, img):
        """
        Full pipeline:
         1. ensure color + size
         2. face detect (if possible)
         3. align (if possible)
         4. crop / fallback to full image
         5. resize to 224x224 and normalize
        Always returns a 224x224 RGB float32 image in [0,1] or None if input None.
        """
        if img is None:
            return None

        # Step 0: ensure color and reasonable size
        img = self._ensure_color_and_size(img)
        if img is None:
            return None

        # Step 1: try detection
        box = self.detector.detect(img)
        if box is not None:
            x1, y1, x2, y2 = box
            # safety clamp (again)
            h, w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 <= 10 or y2 - y1 <= 10:
                face = None
            else:
                face = img[y1:y2, x1:x2]
        else:
            face = None

        # Step 2: try align if we have a face region
        aligned = None
        if face is not None:
            aligned = self.aligner.align_face(face)

        # Step 3: fallbacks
        if aligned is None:
            # If face failed or alignment failed, try aligning full image (sometimes works)
            aligned_full = self.aligner.align_face(img)
            if aligned_full is not None:
                final = aligned_full
            elif face is not None:
                final = face
            else:
                # Last resort: use the original (upsampled) image as the crop
                final = img
        else:
            final = aligned

        # Final resize and normalize
        try:
            face_resized = cv2.resize(final, (224, 224), interpolation=cv2.INTER_LINEAR)
        except Exception:
            return None
        face_norm = face_resized.astype(np.float32) / 255.0

        return face_norm
