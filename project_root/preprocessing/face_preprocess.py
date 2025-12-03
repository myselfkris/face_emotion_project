def preprocess(self, img):
        # 1. Convert grayscale → RGB
    if img.ndim == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # 2. Upscale tiny 48x48 images → 224x224 before detection
    if img.shape[0] < 100 or img.shape[1] < 100:
        img = cv2.resize(img, (224, 224))

        # 3. Face detection (MediaPipe)
    box = self.detector.detect(img)

        # 4. If detection fails → fallback: use whole image
    if box is None:
        resized = cv2.resize(img, (224, 224))
        normalized = resized.astype(np.float32) / 255.0
        return normalized

        # 5. Crop detected face
    x1, y1, x2, y2 = box
    face = img[y1:y2, x1:x2]

        # 6. Alignment (may fail for FER)
    aligned = self.aligner.align_face(face)
    if aligned is None:
        aligned = face

        # 7. Resize & normalize
    resized = cv2.resize(aligned, (224, 224))
    normalized = resized.astype(np.float32) / 255.0

    return normalized
