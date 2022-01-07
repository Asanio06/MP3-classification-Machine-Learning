from keras.applications.xception import Xception
from tensorflow import keras
from tensorflow.keras import layers


def getModel(class_labels):
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(150,200,3))
    base_model.trainable = False
    model = keras.Sequential(
        [
            base_model,

            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(len(class_labels), activation="softmax"),
        ]
    )
    return model