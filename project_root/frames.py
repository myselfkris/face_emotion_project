import cv2
import os

# -------- CONFIG --------
VIDEO_PATH =r"C:\Users\krish\Desktop\NN_learning\project_root\video\sample.mp4"
OUTPUT_DIR = "frames_output"
SECONDS_PER_FRAME = 2
# ------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("❌ Cannot open video")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps * SECONDS_PER_FRAME)

frame_count = 0
saved_count = 0

print(f"FPS: {fps}")
print(f"Saving 1 frame every {SECONDS_PER_FRAME} seconds")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        frame_name = f"frame_{saved_count:04d}.jpg"
        save_path = os.path.join(OUTPUT_DIR, frame_name)

        cv2.imwrite(save_path, frame)
        saved_count += 1

    frame_count += 1

cap.release()

print(f"✅ Done. Saved {saved_count} frames.")
