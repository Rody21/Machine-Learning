import librosa
import numpy as np
import pandas as pd
import os

# Ruta de la carpeta que contiene los archivos de audio
carpeta_audio = "../Unlabeled"

# Obtener una lista de todos los archivos de audio en la carpeta
archivos_audio = os.listdir(carpeta_audio)

# Lista para almacenar las características extraídas
caracteristicas = []

# Iterar sobre los archivos de audio
for archivo in archivos_audio:
    # Construir la ruta completa del archivo de audio
    ruta_audio = os.path.join(carpeta_audio, archivo)

    # Cargar el archivo de audio
    audio, sr = librosa.load(ruta_audio)

    # Extraer las características
    rms = np.sqrt(np.mean(audio**2))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio))
    mfcc = librosa.feature.mfcc(y=audio, sr=sr)

    # Crear un diccionario para almacenar las características
    diccionario_caracteristicas = {
        'Nombre de archivo': archivo,
        'RMS': rms,
        'Zero Crossing Rate': zero_crossing_rate
    }

    # Agregar cada coeficiente de los MFCC como una columna en el diccionario
    for i, coeficiente in enumerate(mfcc):
        nombre_columna = f'MFCC{i+1}'
        diccionario_caracteristicas[nombre_columna] = np.mean(coeficiente)

    # Agregar el diccionario de características a la lista
    caracteristicas.append(diccionario_caracteristicas)

# Crear un DataFrame con las características extraídas
df = pd.DataFrame(caracteristicas)

# Especificar el nombre del archivo CSV de salida
archivo_salida_csv = "caracteristicas_audio.csv"

# Guardar el DataFrame en un archivo CSV
df.to_csv(archivo_salida_csv, index=False)

# Imprimir los resultados
print(df)
