from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras as keras
from training import training_86_100x100x4 as training_model

train_dir = "./content/spectrograms3sec/pathtrain/"
train_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(100, 100), color_mode="rgba",
                                                    class_mode='categorical', batch_size=128)

validation_dir = "./content/spectrograms3sec/pathtest/"
vali_datagen = ImageDataGenerator(rescale=1. / 255)
vali_generator = vali_datagen.flow_from_directory(validation_dir, target_size=(100, 100), color_mode='rgba',
                                                  class_mode='categorical', batch_size=128)

class_labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'metal', 'pop', 'reggae', 'rock', 'jazz']

model = training_model.getModel(class_labels=class_labels)

model.summary()
opt = Adam(learning_rate=0.0005)
metric_top2 = keras.metrics.TopKCategoricalAccuracy(k=2)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', metric_top2])

model.fit_generator(train_generator, epochs=40, validation_data=vali_generator)

model.save('lln_86_40epoch_100x100.h5')
