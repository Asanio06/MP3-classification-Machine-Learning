from tensorflow import keras
from tensorflow.keras import layers


def getModel(class_labels):
    model = keras.Sequential(
        [
            layers.Conv2D(4, kernel_size=(4, 4), activation="relu", input_shape=(220, 350, 3)),
            layers.Dropout(0.2),

            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(8, kernel_size=(3, 3), activation="relu",padding="valid"),
            layers.Dropout(0.2),

            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),

            layers.Flatten(),
            layers.Dropout(0.4),
            layers.Dense(len(class_labels), activation="softmax"),
        ]
    )
    return model
