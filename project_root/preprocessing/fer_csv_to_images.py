import pandas as pd
import numpy as np
import cv2
import os

# Load CSV
df = pd.read_csv("face_emotion_v1/data/raw/fer2013.csv")

# Create folder structure
output_dir = "data"
splits = ["train", "val", "test"]
emotions = list(range(7))

for split in splits:
    for emotion in emotions:
        path = os.path.join(output_dir, split, str(emotion))
        os.makedirs(path, exist_ok=True)

# Convert each row into an image file
for index, row in df.iterrows():
    pixels = np.array(row["pixels"].split(), dtype=np.uint8)
    img = pixels.reshape(48, 48)

    emotion = int(row["emotion"])
    usage = row["Usage"]

    # Decide folder
    if usage == "Training":
        folder = "train"
    elif usage == "PublicTest":
        folder = "val"
    else:
        folder = "test"

    save_path = f"data/{folder}/{emotion}/{index}.jpg"
    cv2.imwrite(save_path, img)

print("Done creating images.")
