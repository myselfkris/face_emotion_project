import tensorflow as tf
from tensorflow.keras import layers, Model

def build_mobilenet_emotion(num_classes=7, embedding_dim=128):
    """
    MobileNetV2 backbone
    Output:
        - emotion softmax (used for training)
    The embedding layer is still created internally and can be extracted later.
    """

    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        pooling="avg"
    )

    # freeze backbone for head training
    base.trainable = False

    x = layers.Dense(256, activation='relu')(base.output)
    x = layers.Dropout(0.3)(x)

    # Embedding head (not used as training output)
    embedding = layers.Dense(embedding_dim, activation=None, name="face_embedding")(x)

    # Emotion classification head (main output)
    emotion_output = layers.Dense(num_classes, activation="softmax", name="emotion")(embedding)

    # Training model → ONLY emotion output
    model = Model(inputs=base.input, outputs=emotion_output, name="face_emotion_model")

    return model


def build_embedding_model(trained_model):
    """
    Extracts an embedding-only model after training.
    """
    embedding_layer = trained_model.get_layer("face_embedding").output
    embedding_model = Model(inputs=trained_model.input, outputs=embedding_layer)
    return embedding_model
