import os
import numpy as np
import cv2
from tqdm import tqdm

from preprocessing.face_preprocess import FacePreprocessor

IMG_SIZE = 224
NUM_CLASSES = 7

# paths
RAW_DATA_PATH = "data"                  # original FER structure: data/train/0/*.jpg ...
OUT_PATH = "data_npy"                   # new npy dataset

pp = FacePreprocessor()

os.makedirs(OUT_PATH, exist_ok=True)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def preprocess_split(split_name):
    """
    split_name: "train", "val", "test"
    Saves preprocessed npy files:
    data_npy/train/0/xxx.npy
    """
    input_dir = os.path.join(RAW_DATA_PATH, split_name)
    output_dir = os.path.join(OUT_PATH, split_name)

    ensure_dir(output_dir)

    print(f"\nProcessing split: {split_name}")

    for label in range(NUM_CLASSES):
        class_in = os.path.join(input_dir, str(label))
        class_out = os.path.join(output_dir, str(label))
        ensure_dir(class_out)

        if not os.path.exists(class_in):
            continue

        for fname in tqdm(os.listdir(class_in), desc=f"Label {label}"):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(class_in, fname)
            img = cv2.imread(img_path)

            if img is None:
                continue

            processed = pp.preprocess(img)
            if processed is None:
                # fallback — zero tensor
                processed = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

            # Save as npy
            out_name = fname.split(".")[0] + ".npy"
            out_path = os.path.join(class_out, out_name)
            np.save(out_path, processed.astype(np.float32))


if __name__ == "__main__":
    preprocess_split("train")
    preprocess_split("val")
    preprocess_split("test")

    print("\n✔ Preprocessing complete!")
    print(f"Saved to: {OUT_PATH}")
