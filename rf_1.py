import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, GroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.tree import plot_tree
import joblib

# Cargar los datos de características desde CSV
data_x = pd.read_csv("caracteristicas_combinadas.csv").iloc[:, 1:]

# Cargar las etiquetas y los IDs desde CSV
data = pd.read_csv("Seteo1.csv")
y = data.iloc[:, 4]
ids = data.iloc[:, 1]

# Establecer la semilla para reproducibilidad
seed = 21
np.random.seed(seed)

# Obtener las características (X)
X = data_x.values

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X, y, ids, test_size=0.2, random_state=seed)

# Definir los hiperparámetros para buscar
param_grid = {
    'n_estimators': [50, 250, 500],
    'max_depth': [5, 10, 20],
    'min_samples_leaf': [5, 10, 20],
    'max_features': ["sqrt", "log2"]
}

# Crear el objeto GroupKFold utilizando los IDs
group_kfold = GroupKFold(n_splits=5)

# Realizar búsqueda de hiperparámetros utilizando validación cruzada
model = RandomForestClassifier(random_state=seed)
grid_search = GridSearchCV(
    model, param_grid, cv=group_kfold.split(X_train, y_train, ids_train))
grid_search.fit(X_train, y_train)

# Obtener el mejor modelo según la búsqueda de hiperparámetros
best_model = grid_search.best_estimator_

# Realizar predicciones en el conjunto de prueba
y_pred = best_model.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)

# Calcular el reporte de clasificación (precision, recall, F1-score)
classification_rep = classification_report(y_test, y_pred)

# Calcular la matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Imprimir los resultados
print("Mejores hiperparámetros:", grid_search.best_params_)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_rep)
print("Confusion Matrix:")
print(cm)

# Visualizar la matriz de confusión
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Visualizar un árbol del modelo final
plt.figure(figsize=(10, 10))
plot_tree(best_model.estimators_[
          0], feature_names=data_x.columns, class_names=np.unique(y).astype(str), filled=True)
plt.show()

joblib.dump(best_model, "best_model.h5")