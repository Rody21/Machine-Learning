import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

# Establecer una semilla global para la reproducibilidad
tf.random.set_seed(21)

# Cargar el archivo CSV con las etiquetas
etiquetas = pd.read_csv('Seteo1.csv')

# Obtener los nombres completos de las imágenes y las etiquetas correspondientes
nombres_imagenes = etiquetas.iloc[:, 0].tolist()  # Primera columna
etiquetas_imagenes = etiquetas.iloc[:, -2].tolist()  # Penúltima columna

# Directorio donde se encuentran las imágenes
directorio_imagenes = 'Data'

# Crear listas para almacenar las imágenes y etiquetas procesadas
imagenes_procesadas = []
etiquetas_procesadas = []

# Procesar cada imagen
for nombre_imagen, etiqueta in zip(nombres_imagenes, etiquetas_imagenes):
    # Obtener la ruta completa de la imagen
    ruta_completa = os.path.join(directorio_imagenes, nombre_imagen)

    # Abrir la imagen utilizando PIL y convertir a un array de NumPy
    imagen = Image.open(ruta_completa).convert('RGB')
    imagen_array = np.array(imagen)

    # Convertir el array a float32 y normalizar los valores de píxeles entre 0 y 1
    imagen_float32 = imagen_array.astype(np.float32) / 255.0

    # Agregar la imagen y la etiqueta a las listas procesadas
    imagenes_procesadas.append(imagen_float32)
    etiquetas_procesadas.append(etiqueta)

# Convertir las listas de imágenes y etiquetas a tensores
x = np.array(imagenes_procesadas)
y = np.array(etiquetas_procesadas)

# Dividir los datos en conjuntos de entrenamiento y prueba
x_entrenamiento, x_prueba, y_entrenamiento, y_prueba = train_test_split(
    x, y, test_size=0.2, random_state=42)

# Definir y configurar el modelo de CNN
model = tf.keras.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
          input_shape=x_entrenamiento[0].shape))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(6, activation='softmax'))
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(x_entrenamiento, y_entrenamiento, epochs=10,
          validation_data=(x_prueba, y_prueba))

# Evaluar el modelo con los datos de prueba
test_loss, test_acc = model.evaluate(x_prueba, y_prueba)
print('Precisión en los datos de prueba:', test_acc)

# Save the model
model.save('trained_model.h5')
print('Trained model saved as "trained_model.h5"')
