import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.utils import to_categorical, Sequence

# -------------------
# CONFIG
# -------------------
NUM_CLASSES = 7
IMG_SIZE = 224
BATCH_SIZE = 32         # now we can increase batch size
EPOCHS_HEAD = 8
EPOCHS_FINE = 3
DATA_PATH = "data_npy"  # << 🔥 use preprocessed npy data
MODEL_SAVE_PATH = "face_emotion_v1.keras"

from model.mobilenet_emotion import build_mobilenet_emotion  


# -------------- DATA SEQUENCE ----------------
class FastNpySequence(Sequence):
    def __init__(self, file_label_list, batch_size=BATCH_SIZE, shuffle=True):
        self.data = file_label_list
        self.batch_size = batch_size   
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:  
            np.random.shuffle(self.data)

    def __getitem__(self, idx):   
        batch = self.data[idx*self.batch_size:(idx+1)*self.batch_size]
        images, labels = [], []

        for fp, label in batch:
            arr = np.load(fp)   # 🔥 super fast
            images.append(arr)
            labels.append(label)

        x = np.stack(images).astype(np.float32)
        y = to_categorical(labels, NUM_CLASSES)
        return x, y


def load_file_list(split):
    files = []
    root = os.path.join(DATA_PATH, split)
    for label in range(NUM_CLASSES):
        class_dir = os.path.join(root, str(label))
        if not os.path.exists(class_dir):
            continue

        for fname in os.listdir(class_dir):
            if fname.endswith(".npy"):
                files.append((os.path.join(class_dir, fname), label))   
    return files  


# ---------------- TRAIN ----------------
if __name__ == "__main__":
    train_files = load_file_list("train")
    val_files = load_file_list("val")

    print("Train samples:", len(train_files))
    print("Val samples:", len(val_files))

    train_seq = FastNpySequence(train_files)
    val_seq = FastNpySequence(val_files, shuffle=False)

    model = build_mobilenet_emotion(num_classes=NUM_CLASSES, embedding_dim=128)

    # stage 1
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])  

    print("\nTraining head...")
    model.fit(train_seq, validation_data=val_seq, epochs=EPOCHS_HEAD)  

    # stage 2
    for layer in model.layers[-20:]: 
        layer.trainable = True 

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    print("\nFine-tuning...")
    model.fit(train_seq, validation_data=val_seq, epochs=EPOCHS_FINE)

    model.save(MODEL_SAVE_PATH)
    print("✔ Model saved!")
    print("Training complete.")