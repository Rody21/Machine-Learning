import pandas as pd
import numpy as np
import joblib

# Cargar el modelo desde el archivo .hf
modelo = joblib.load("best_model.h5")

# Cargar los datos de entrada desde un archivo CSV
datos_entrada = pd.read_csv("caracteristicas_audio.csv")

# Omitir la primera columna del DataFrame
datos_entrada = datos_entrada.iloc[:, 1:]

# Realizar predicciones utilizando el modelo cargado
predicciones = modelo.predict(datos_entrada)

# Definir el diccionario de mapeo de etiquetas
etiquetas_dict = {
    1: "Animals",
    2: "Natural",
    3: "Human",
    4: "Interior",
    5: "Exterior"
}

# Mapear las predicciones a etiquetas de texto condicionadas
etiquetas = np.where(predicciones <= 5, [etiquetas_dict.get(
    num) for num in predicciones], "Etiqueta no definida")

# Imprimir los resultados de las predicciones con etiquetas
for etiqueta in etiquetas:
    print(etiqueta)
