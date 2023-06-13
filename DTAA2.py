import numpy as np
import os
import librosa
import librosa.display
from scipy.io import wavfile
import scipy.signal as signal
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
from PIL import Image
import csv

aud_path = './origin/audio'
Out_path = './output'  # carpeta en la que se generarán las carpetas clasificadas
csvpath = './origin/esc50.csv'  # ruta de archivo con dataset escXX
names = []
cats = []

# Generar un diccionario que de la categoria para un nombre de archivo


def mapcat(path):
    with open(path, 'r') as archivo_csv:
        lector_csv = csv.reader(archivo_csv)
        # Itera sobre las filas del archivo CSV
        for fila in lector_csv:
            # Accede al valor de la columna deseada en cada fila
            name = fila[0]
            cat = fila[3]
            # Agrega el valor al vector
            names.append(name)
            cats.append(cat)
    ntoc = dict(zip(names, cats))
    return ntoc

# Generar el espectrograma de un elemento en especifico


def spect(audioname, cat):
    # Load the WAV file
    path = './origin/audio/' + audioname
    sample_rate, waveform = wavfile.read(path)

    # Compute the spectrogram
    frequencies, times, spectrogram = signal.spectrogram(
        waveform, fs=sample_rate)

    # Apply a small offset to avoid zeros
    offset = 1e-10
    spectrogram += offset

    # Convert the spectrogram to decibels
    spectrogram_db = 10 * np.log10(spectrogram)

    # Plot the spectrogram
    plt.figure(figsize=(10, 4))
    plt.imshow(np.log(spectrogram), aspect='auto',
               cmap='inferno', origin='lower')
    plt.title('Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.tight_layout()

    # Save the spectrogram as an image
    plt.savefig('./output/' + cat + '/' + os.path.splitext(audioname)
                [0] + '.png', bbox_inches='tight', pad_inches=0, format='png', dpi=80)
    plt.close()


catdic = mapcat(csvpath)
names = catdic.keys()
usedcat = []

for name in names:
    if catdic[name] not in usedcat:
        newpath = os.path.join(Out_path, catdic[name])
        print(newpath)
        os.makedirs(newpath, exist_ok=True)
        usedcat.append(catdic[name])

    if name != 'filename':
        spect(name, catdic[name])
