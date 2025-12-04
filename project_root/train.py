import tensorflow as tf
import numpy as np
import os
import cv2
from tensorflow.keras.utils import to_categorical, Sequence

from preprocessing.face_preprocess import FacePreprocessor
from model.mobilenet_emotion import build_mobilenet_emotion

# ---------- CONFIG ----------
NUM_CLASSES = 7
IMG_SIZE = 224
BATCH_SIZE = 8       # small batch to avoid OOM; increase later if GPU available
EPOCHS_HEAD = 8
EPOCHS_FINE = 3
MODEL_SAVE_PATH = "face_emotion_v1.keras"

preprocessor = FacePreprocessor()

def list_files_in_folder(folder):
    files = []
    for label in range(NUM_CLASSES):
        class_dir = os.path.join(folder, str(label))
        if not os.path.exists(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.jpg','.jpeg','.png')):
                files.append((os.path.join(class_dir, fname), label))
    return files

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
        batch = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        images = []
        labels = []
        for fp, label in batch:
            img = cv2.imread(fp)
            if img is None:
                # fallback zero image if file unreadable
                img_proc = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
            else:
                img_proc = preprocessor.preprocess(img)
                if img_proc is None:
                    img_proc = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
            images.append(img_proc)
            labels.append(label)
        x = np.array(images, dtype=np.float32)
        y = to_categorical(labels, NUM_CLASSES)
        return x, {"emotion": y}

if __name__ == "__main__":
    train_files = list_files_in_folder("data/train")
    val_files = list_files_in_folder("data/val")

    print("Train files:", len(train_files), "Val files:", len(val_files))
    if len(train_files) == 0:
        raise RuntimeError("No training files found in data/train - check your dataset path.")

    train_seq = FaceSequence(train_files, batch_size=BATCH_SIZE, shuffle=True)
    val_seq   = FaceSequence(val_files, batch_size=BATCH_SIZE, shuffle=False)

    # Build model
    model = build_mobilenet_emotion(num_classes=NUM_CLASSES, embedding_dim=128)

    # Stage 1: train head only
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={"emotion": "categorical_crossentropy"},
        metrics={"emotion": "accuracy"}
    )

    print("Training head (backbone frozen)...")
    model.fit(
        train_seq,
        validation_data=val_seq if len(val_files) > 0 else None,
        epochs=EPOCHS_HEAD,
        
    )

    # Stage 2: unfreeze top layers
    for layer in model.layers[-20:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss={"emotion": "categorical_crossentropy"},
        metrics={"emotion": "accuracy"}
    )

    print("Fine-tuning top layers...")
    model.fit(
        train_seq,
        validation_data=val_seq if len(val_files) > 0 else None,
        epochs=EPOCHS_FINE,
        
    )

    model.save(MODEL_SAVE_PATH)
    print("Saved model:", MODEL_SAVE_PATH)

