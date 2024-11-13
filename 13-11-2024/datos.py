import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Paso 1: Carga de Datos
# Cargar los datos de phishing desde un archivo CSV
data = pd.read_csv('Website Phishing.csv')

# Mostrar las primeras filas del DataFrame
print(data.head())

# Paso 2: Preprocesamiento de Datos
# Convertir la columna 'Result' a categórica
data['Result'] = data['Result'].astype('category')

# Separar características y etiquetas
X = data.drop('Result', axis=1)
y = data['Result']

# Normalizar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Paso 3: Creación del Modelo MLP
# Crear el modelo MLP
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)

# Entrenar el modelo
mlp.fit(X_train, y_train)

# Paso 4: Evaluación del Modelo
# Realizar predicciones
y_pred = mlp.predict(X_test)

# Evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Precisión: {accuracy}')
print('Reporte de Clasificación:')
print(report)
print('Matriz de Confusión:')
print(conf_matrix)

# Paso 5: Visualización de Resultados
# Visualizar la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Legítimo', 'Phishing'], yticklabels=['Legítimo', 'Phishing'])
plt.ylabel('Etiqueta Verdadera')
plt.xlabel('Etiqueta Predicha')
plt.title('Matriz de Confusión')
plt.show()