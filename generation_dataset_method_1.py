import numpy as np
import os
import matplotlib.pyplot as plt
import librosa
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pydub import AudioSegment
import shutil
import random



#TODO: CONFIGURATION

directoryOfSongs= "./genres"


os.makedirs('./content/spectrograms3sec')
os.makedirs('./content/spectrograms3sec/train')
os.makedirs('./content/spectrograms3sec/test')
genres = 'blues classical country disco pop hiphop metal reggae rock'
genres = genres.split()




for g in genres:
    path_audio = os.path.join('./content/audio3sec', f'{g}')
    os.makedirs(path_audio)
    path_train = os.path.join('./content/spectrograms3sec/pathtrain', f'{g}')
    path_test = os.path.join('./content/spectrograms3sec/pathtest', f'{g}')
    os.makedirs(path_train)
    os.makedirs(path_test)

i = 0
for g in genres:
    j = 0
    print(f"{g}")

    for filename in os.listdir(os.path.join(directoryOfSongs, f"{g}")):

        song = os.path.join(f'{directoryOfSongs}/{g}', f'{filename}')
        j = j + 1
        for w in range(0, 10):
            i = i + 1
            # print(i)
            t1 = 3 * (w) * 1000
            t2 = 3 * (w + 1) * 1000
            newAudio = AudioSegment.from_wav(song)
            new = newAudio[t1:t2]
            new.export(f'./content/audio3sec/{g}/{g + str(j) + str(w)}.wav', format="wav")



for g in genres:
    j = 0
    print(g)
    for filename in os.listdir(os.path.join('./content/audio3sec', f"{g}")):
        song = os.path.join(f'./content/audio3sec/{g}', f'{filename}')
        j = j + 1

        y, sr = librosa.load(song, duration=3)
        # print(sr)
        mels = librosa.feature.melspectrogram(y=y, sr=sr)
        fig = plt.Figure()
        canvas = FigureCanvas(fig)
        p = plt.imshow(librosa.power_to_db(mels, ref=np.max))
        plt.savefig(f'./content/spectrograms3sec/pathtrain/{g}/{g + str(j)}.png')


directory = "./content/spectrograms3sec/pathtrain/"
for g in genres:
    filenames = os.listdir(os.path.join(directory, f"{g}"))
    random.shuffle(filenames)
    test_files = filenames[0:100]

    for f in test_files:
        shutil.move(directory + f"{g}" + "/" + f, "./content/spectrograms3sec/pathtest/" + f"{g}")
