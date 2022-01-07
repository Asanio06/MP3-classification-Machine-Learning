from tensorflow.keras import layers
import tensorflow.keras as keras





def getModel(class_labels):

    model = keras.Sequential(
        [
            layers.Conv2D(32, (3, 3), padding='same',
                          input_shape=(100, 100, 4)),
            layers.Activation('relu'),
            layers.Conv2D(64, (3, 3)),
            layers.Activation('relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.Activation('relu'),
            layers.Conv2D(64, (3, 3)),
            layers.Activation('relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.5),
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.Activation('relu'),
            layers.Conv2D(128, (3, 3)),
            layers.Activation('relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.5),
            layers.Flatten(),
            layers.Dense(512),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            layers.Dense(len(class_labels), activation='softmax'),
        ]
    )

    return model
