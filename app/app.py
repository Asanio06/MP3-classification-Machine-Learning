import streamlit as st
import pydub
from PIL import Image
import librosa
import numpy as np
import librosa.display
from pydub import AudioSegment
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from tensorflow import keras
import keras.backend as K
import pandas as pd

st.write("Music Genre Recognition App")
st.write("This is a Web App to predict genre of music")
file = st.sidebar.file_uploader("Please Upload Mp3 Audio File Here or Use Demo Of App Below using Preloaded Music",
                                type=["mp3"])

# Configuration
image_size = (100, 100)
song_duration = 30
color_mode = "rgba"  # rgb or rgba
model_path = "model.h5"  # Path for your model
class_labels = ['blues', 'classical', 'country', 'disco', 'pop', 'hiphop', 'metal', 'reggae', 'rock', 'jazz']


# class_labels = ['blues', 'country', 'hiphop', 'jazz']


def get_f1(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


def convert_mp3_to_wav(music_file):
    pydub.AudioSegment.ffmpeg = "C:/ffmpeg/bin"
    sound = AudioSegment.from_mp3(music_file)
    sound.export("music_file.wav", format="wav")


def extract(wav_file, t1, t2):
    wav = AudioSegment.from_wav(wav_file)
    wav = wav[1000 * t1:1000 * t2]
    wav.export("extracted.wav", format='wav')


def create_melspectrogram(wav_file):
    y, sr = librosa.load(wav_file, duration=song_duration)  # TODO : En fonction du temps mis sur les entrainement
    mels = librosa.feature.melspectrogram(y=y, sr=sr)
    fig = plt.Figure()
    # canvas = FigureCanvas(fig)
    p = plt.imshow(librosa.power_to_db(mels, ref=np.max))

    plt.savefig('melspectrogram.png')


def predict(image_data, model):
    image = img_to_array(image_data) / 255.0
    image = np.reshape(image, (
        1, image_size[0], image_size[1], 3 if color_mode == "rgb" else 4))  # TODO :  en fonction de son model

    prediction = model.predict(image)
    top2_index = np.argpartition(prediction, -2, axis=1)[:, -2:]
    prediction = prediction.reshape((len(class_labels),))  # TODO: En fonction du nombre de class
    class_label = np.argmax(prediction)
    return class_label, prediction, top2_index[0]


if file is None:
    st.text("Please upload an mp3 file")
else:
    convert_mp3_to_wav(file)
    extract("music_file.wav", 30, 40)
    create_melspectrogram("extracted.wav")
    image_data = load_img('melspectrogram.png', color_mode=color_mode,
                          target_size=image_size)

    model = keras.models.load_model(model_path, custom_objects={"get_f1": get_f1})
    class_label, prediction, top2_index = predict(image_data, model)
    st.write(f"## The Genre of Song is {class_labels[top2_index[0]]} or  {class_labels[top2_index[1]]} ")
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    genre = [class_labels[top2_index[0]], class_labels[top2_index[1]]]
    proba = [prediction[top2_index[0]] * 100, prediction[top2_index[1]] * 100]
    ax.bar(genre, proba, color=['red', 'blue'])
    ax.legend(['Probabilités en %'])
    st.pyplot(fig)
