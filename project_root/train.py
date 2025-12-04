import tensorflow as tf
import numpy as np
import os
import cv2
from tensorflow.keras.utils import to_categorical, Sequence

from preprocessing.face_preprocess import FacePreprocessor
from model.mobilenet_emotion import build_mobilenet_emotion

# ---------------- CONFIG ----------------
NUM_CLASSES = 7
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS_HEAD = 8
EPOCHS_FINE = 3
MODEL_SAVE_PATH = "face_emotion_v1.keras"

preprocessor = FacePreprocessor()

# ------------- DATA LOADING -------------
def list_files(folder):
    items = []
    for label in range(NUM_CLASSES):
        class_dir = os.path.join(folder, str(label))
        if not os.path.exists(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.jpg','.jpeg','.png')):
                items.append((os.path.join(class_dir, fname), label))
    return items


class FaceSequence(Sequence):
    def __init__(self, file_label_list, batch_size=BATCH_SIZE, shuffle=True):
        self.data = file_label_list
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch_size)))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.data)

    def __getitem__(self, idx):
        batch = self.data[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        images, labels = [], []
        for path, label in batch:
            img = cv2.imread(path)

            # Preprocess safely
            if img is not None:
                img = preprocessor.preprocess(img)
            if img is None:
                img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

            images.append(img)
            labels.append(label)

        x = np.array(images, dtype=np.float32)
        y = to_categorical(labels, NUM_CLASSES)
        return x, y


# ---------------- TRAINING ----------------
if __name__ == "__main__":
    train_files = list_files("data/train")
    val_files   = list_files("data/val")

    print("Train files:", len(train_files))
    print("Val files:", len(val_files))

    if len(train_files) == 0:
        raise RuntimeError("No training files found. Check data/train path.")

    train_seq = FaceSequence(train_files, batch_size=BATCH_SIZE, shuffle=True)
    val_seq   = FaceSequence(val_files, batch_size=BATCH_SIZE, shuffle=False)

    # Build model
    model = build_mobilenet_emotion(num_classes=NUM_CLASSES, embedding_dim=128)

    # -------- Stage 1: train head --------
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\nTraining head (backbone frozen)...")
    model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=EPOCHS_HEAD
    )

    # -------- Stage 2: fine-tune --------
    for layer in model.layers[-20:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\nFine-tuning top layers...")
    model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=EPOCHS_FINE
    )

    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved at {MODEL_SAVE_PATH}")
