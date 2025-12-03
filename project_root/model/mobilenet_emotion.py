import tensorflow as tf
from tensorflow.keras import layers, Model

def build_mobilenet_emotion(num_classes=7, embedding_dim=128):
    """
    MobileNetV2 backbone with:
    - internal 128-d embedding layer (name: 'face_embedding')
    - single emotion softmax output (model output)
    """
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        pooling="avg"
    )

    # freeze backbone initially
    base.trainable = False

    x = layers.Dense(256, activation='relu')(base.output)
    x = layers.Dropout(0.3)(x)

    # embedding layer (internal; will be trained via emotion loss)
    embedding = layers.Dense(embedding_dim, activation=None, name="face_embedding")(x)

    # emotion output (model's only output)
    emotion_output = layers.Dense(num_classes, activation='softmax', name="emotion")(embedding)

    # single-output model
    model = Model(inputs=base.input, outputs=emotion_output)
    return model
