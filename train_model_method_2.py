import random
import numpy as np
import seaborn as sns
from keras.applications.xception import Xception
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from training import training_60_150x200x3 as training_model

def load_training_dataset(dataset_location='./dataset/',
                          return_format='numpy',
                          image_size=(100, 100),
                          batch_size=32,
                          shuffle=True):
    '''
    Charge et retourne un dataset à partir d’un dossier contenant
    des images où chaque classe est dans un sous-dossier.
    Le dataset est peut être renvoyé comme deux tableaux NumPy, sous
    la forme d’un couple (features, label) ; ou comme un Dataset
    TensorFlow (déjà découpé en batch).
    # Arguments
        dataset_location: chemin vers le dossier contenant les images
            réparties dans des sous-dossiers représentants les
            classes.
        return_format: soit `numpy` (le retour sera un couple de
            tableaux NumPy (features, label)), soit `tf` (le
            retour sera un Dataset TensorFlow).
        image_size: la taille dans laquelle les images seront
            redimensionnées après avoir été chargée du disque.
        batch_size: la taille d’un batch, cette valeur n’est utilisée
            que si `return_format` est égale à `tf`.
        shuffle: indique s’il faut mélanger les données. Si défini à
            `False` les données seront renvoyées toujours dans le
            même ordre.
    # Retourne
        Un couple de tableaux NumPy (features, label) si
        `return_format` vaut `numpy`.
        Un Dataset TensorFlow si `return_format` vaut `tf`.
    '''
    ds = image_dataset_from_directory(
        dataset_location,
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        shuffle=shuffle if return_format == 'tf' else False,
        image_size=image_size,
        color_mode='rgb',
        interpolation='bilinear'
    )
    if return_format == 'tf':
        return ds
    elif return_format == 'numpy':
        X = np.concatenate([images.numpy() for images, labels in ds])
        y = np.concatenate([labels.numpy() for images, labels in ds])
        if shuffle:
            idx = list(range(len(X)))
            random.shuffle(idx)
            X = X[idx]
            y = y[idx]
        return (X, y)
    else:
        raise ValueError(
            'The `return_format` argument should be either `numpy` (NumPy arrays) or `tf` (TensorFlow dataset).')


if __name__ == "__main__":
    class_labels = ['blues', 'classical', 'country', 'disco', 'pop', 'hiphop', 'metal', 'reggae', 'rock', 'jazz']

    image_size = (150, 200) # TODO: CONFIGURE
    (X, Y) = load_training_dataset(dataset_location='./images', image_size=image_size)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    X_train = np.array(X_train) / 255.0
    X_test = np.array(X_test) / 255.0

    model = training_model.getModel(class_labels=class_labels)

    metric_top2 = keras.metrics.TopKCategoricalAccuracy(k=2)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=["accuracy", metric_top2])
    model.summary()
    model.fit(X_train, y_train, batch_size=10, epochs=150, validation_data=(X_test, y_test))

    model.save('model.h5')

    prediction_probas = model.evaluate(X_test, y_test)
    print(prediction_probas)

    y_pred = model.predict(X_test)
    con_mat = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

    con_mat_df = pd.DataFrame(con_mat_norm, index=class_labels, columns=class_labels)
    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.savefig('confusion_matrix.png')
