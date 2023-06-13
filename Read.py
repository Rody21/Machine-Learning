import librosa
import numpy as np
import pandas as pd
import os

# Ruta de la carpeta que contiene los archivos de audio
audio_folder = "../ML/audio"

# Ruta del archivo CSV que contiene los nombres de los archivos de audio
csv_file = "Seteo1.csv"

# Leer el archivo CSV
data = pd.read_csv(csv_file)

# Obtener los nombres de los archivos de audio desde el CSV
audio_files = data["filename"].tolist()

# Listas para almacenar las características extraídas
rms_values = []
zero_crossing_rates = []
mfccs = []

# Iterar sobre los archivos de audio
for file in audio_files:
    # Construir la ruta completa del archivo de audio
    audio_path = os.path.join(audio_folder, file)

    # Cargar el archivo de audio
    audio, sr = librosa.load(audio_path)

    # Calcular el RMS
    rms = np.sqrt(np.mean(audio**2))
    rms_values.append(rms)

    # Calcular el Zero Crossing Rate
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio))
    zero_crossing_rates.append(zero_crossing_rate)

    # Calcular los MFCCs
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr), axis=1)
    mfccs.append(mfcc)

# Crear un DataFrame con las características individuales
df_rms = pd.DataFrame({
    'Filename': audio_files,
    'RMS': rms_values
})
df_zero_crossing_rate = pd.DataFrame({
    'Filename': audio_files,
    'Zero Crossing Rate': zero_crossing_rates
})
df_mfccs = pd.DataFrame({
    'Filename': audio_files
})
for i in range(len(mfccs[0])):
    df_mfccs[f'MFCC{i+1}'] = [mfcc[i] for mfcc in mfccs]

# Especificar los nombres de los archivos CSV de salida para las características individuales
csv_output_file_rms = "rms_individual.csv"
csv_output_file_zero_crossing_rate = "zero_crossing_rate_individual.csv"
csv_output_file_mfccs = "mfccs_individual.csv"

# Guardar los DataFrames en archivos CSV para las características individuales
df_rms.to_csv(csv_output_file_rms, index=False)
df_zero_crossing_rate.to_csv(csv_output_file_zero_crossing_rate, index=False)
df_mfccs.to_csv(csv_output_file_mfccs, index=False)

# Guardar las características individuales en archivos NPY
np.save("rms_individual.npy", rms_values)
np.save("zero_crossing_rate_individual.npy", zero_crossing_rates)
for i, mfcc in enumerate(mfccs):
    np.save(f"mfcc{i+1}_individual.npy", mfcc)

# Combinar las características en un solo DataFrame
combined_df = pd.DataFrame({
    'Filename': audio_files,
    'RMS': rms_values,
    'Zero Crossing Rate': zero_crossing_rates,
})
for i in range(len(mfccs[0])):
    combined_df[f'MFCC{i+1}'] = [mfcc[i] for mfcc in mfccs]

# Especificar el nombre del archivo CSV de salida para las características combinadas
csv_output_file_combined = "caracteristicas_combinadas.csv"

# Guardar el DataFrame combinado en un archivo CSV
combined_df.to_csv(csv_output_file_combined, index=False)

# Guardar las características combinadas en un archivo NPY
np.save("caracteristicas_combinadas.npy", combined_df.values)

# Imprimir los resultados
print("RMS:")
print(rms_values)
print("Zero Crossing Rates:")
print(zero_crossing_rates)
print("MFCCs:")
print(mfccs)
