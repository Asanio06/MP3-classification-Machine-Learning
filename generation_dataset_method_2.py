import os
import librosa, librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_template import FigureCanvas
import numpy as np
from pydub import AudioSegment


def generate_sample(filePath, directoryOfMultiSongs):
    isExist = os.path.exists(directoryOfMultiSongs + '/' + filePath.split('\\')[-2])

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(directoryOfMultiSongs + '/' + filePath.split('\\')[-2])
        print("The new directory is created!")

    for w in range(0, 10):
        # print(i)
        t1 = 3 * (w) * 1000
        t2 = 3 * (w + 1) * 1000
        newAudio = AudioSegment.from_mp3(filePath)
        new = newAudio[t1:t2]

        new.export(directoryOfMultiSongs + '/' + '/'.join(
            ('.'.join(filePath.split('.')[:-1])).split('\\')[1:]) + f'{w + 1}' + '.wav',
                   format="wav")


def generate_images(filePath, directoryOfMelSpectrogram):
    isExist = os.path.exists(directoryOfMelSpectrogram + '/' + filePath.split('\\')[-2])

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(directoryOfMelSpectrogram + '/' + filePath.split('\\')[-2])
        print("The new directory is created!")

    y, sr = librosa.load(filePath, duration=3)
    # print(sr)
    mels = librosa.feature.melspectrogram(y=y, sr=sr)
    fig = plt.Figure()
    canvas = FigureCanvas(fig)
    p = plt.imshow(librosa.power_to_db(mels, ref=np.max))
    plt.savefig(
        directoryOfMelSpectrogram + '/' + '/'.join(('.'.join(filePath.split('.')[:-1])).split('\\')[1:]) + '.png')



if __name__ == '__main__':

    # TODO: CONFIGURE
    directoryOfSongs = "dataset"
    directoryOfMultiSongs = "data"
    directoryOfMelSpectrogram = "images"

    for subdir, dirs, files in os.walk(directoryOfSongs):
        for file in files:
            try:
                generate_sample(os.path.join(subdir, file), directoryOfMultiSongs)

            except:
                pass

    for subdir, dirs, files in os.walk(directoryOfMultiSongs):
        for file in files:
            try:
                generate_images(os.path.join(subdir, file), directoryOfMelSpectrogram)

            except:
                pass
