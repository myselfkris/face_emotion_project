import cv2
import mediapipe as mp
import numpy as np

class FaceAligner:
    def __init__(self):
        self.mp_mesh = mp.solutions.face_mesh
        self.mesh = self.mp_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
            max_num_faces=1
        )

        self.left_eye_id = 33
        self.right_eye_id = 263

    def align_face(self, img):
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.mesh.process(rgb)

        if not result.multi_face_landmarks:
            return None

        lm = result.multi_face_landmarks[0].landmark

        left_eye = (int(lm[self.left_eye_id].x * w), int(lm[self.left_eye_id].y * h))
        right_eye = (int(lm[self.right_eye_id].x * w), int(lm[self.right_eye_id].y * h))

        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))

        center = (w // 2, h // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(img, rot_matrix, (w, h))

        return aligned
