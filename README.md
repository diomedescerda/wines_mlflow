# 🍷 Predicción de Calidad de Vino con MLflow

Pipeline de MLOps completo para entrenar, registrar y servir un modelo de predicción de calidad de vino blanco, con explicaciones generadas por IA.


## ⚙️ Requisitos Previos

Asegúrate de tener instalados:
- Python 3.10 o superior  
- MLflow  
- Conda

Inicia la interfaz de MLflow para visualizar experimentos y modelos:
```bash
mlflow ui --port 5000
```

## 🚀 Entrenamiento y Registro del Modelo

Ejecuta el pipeline de entrenamiento, que incluye:

- Entrenamiento de un modelo Random Forest
- Registro automático del mejor modelo en el Model Registry
- Asignación de alias (staging, production)
```bash
mlflow run project --experiment-name wines_rf_tuning -e train
```
Durante este proceso, los resultados del tuning y las métricas se guardan en MLflow, y el mejor modelo se registra como WineQuality.

## 🧩 Despliegue del Modelo (App)
Una vez entrenado y registrado el modelo, puedes ejecutar la aplicación de predicción de cualquiera de las siguientes formas:
```bash
mlflow run project 
```
```bash
mlflow run project -e serve
```
> 💡 **Nota:** Si el modelo aún no está entrenado, asegúrate de ejecutar el paso de entrenamiento antes.