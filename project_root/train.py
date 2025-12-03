import os
import cv2
import numpy as np
import tensorflow as tf

from preprocessing.face_preprocess import FacePreprocessor
from model.mobilenet_emotion import build_mobilenet_emotion

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_HEAD = 10
EPOCHS_FINE = 5
NUM_CLASSES = 7
MODEL_SAVE_PATH = "face_emotion_v1.keras"

preprocessor = FacePreprocessor()

# -----------------------------
# LOAD DATASET
# -----------------------------
def load_dataset(folder):
    images = []
    labels = []

    # expects folder structure: folder/0/, folder/1/, ..., folder/6/
    for label in range(NUM_CLASSES):
        class_path = os.path.join(folder, str(label))
        if not os.path.exists(class_path):
            continue

        for fname in os.listdir(class_path):
            img_path = os.path.join(class_path, fname)

            img = cv2.imread(img_path)
            if img is None:
                continue

            processed = preprocessor.preprocess(img)
            if processed is None:
                continue

            # ensure shape (IMG_SIZE, IMG_SIZE, 3) and float32 normalized [0,1]
            if processed.dtype != np.float32:
                processed = processed.astype('float32')
            if processed.max() > 1.0:
                processed = processed / 255.0

            if processed.shape[0] != IMG_SIZE or processed.shape[1] != IMG_SIZE:
                processed = cv2.resize((processed * 255).astype('uint8'), (IMG_SIZE, IMG_SIZE))
                processed = processed.astype('float32') / 255.0

            images.append(processed)
            labels.append(label)

    images = np.array(images, dtype='float32')
    labels = tf.keras.utils.to_categorical(labels, NUM_CLASSES).astype('float32')

    print(f"Loaded {len(images)} samples from {folder}")
    return images, labels

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    trainX, trainY = load_dataset("data/train")
    valX, valY = load_dataset("data/val")

    if len(trainX) == 0:
        raise RuntimeError("No training images found. Check data/train folder structure.")

    # ensure float32 (safety)
    trainX = trainX.astype('float32')
    trainY = trainY.astype('float32')
    valX = valX.astype('float32') if len(valX) > 0 else np.array([], dtype='float32')
    valY = valY.astype('float32') if len(valY) > 0 else np.array([], dtype='float32')

    # build model (single-output: emotion)
    model = build_mobilenet_emotion(num_classes=NUM_CLASSES, embedding_dim=128)

    # Stage 1 — train head only
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("Training head (backbone frozen)...")
    model.fit(
        trainX, trainY,
        validation_data=(valX, valY) if len(valX) > 0 else None,
        epochs=EPOCHS_HEAD,
        batch_size=BATCH_SIZE
    )

    # Stage 2 — fine-tune top layers
    # unfreeze last N layers (tune N as needed)
    for layer in model.layers[-20:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("Fine-tuning top layers...")
    model.fit(
        trainX, trainY,
        validation_data=(valX, valY) if len(valX) > 0 else None,
        epochs=EPOCHS_FINE,
        batch_size=BATCH_SIZE
    )

    # save model
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved as {MODEL_SAVE_PATH}")

    # quick tip: extract embeddings later with:
    # embedding_model = tf.keras.Model(inputs=model.input,
    #                                  outputs=model.get_layer('face_embedding').output)
    # embs = embedding_model.predict(some_images)
