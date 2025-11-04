# 🍷 Predicción de Calidad de Vino con MLflow

Pipeline de MLOps completo para entrenar, registrar y servir un modelo de predicción de calidad de vino blanco, con explicaciones generadas por IA.

## 🚀 Entrenamiento y Registro del Modelo

Para entrenar y registrar el modelo
```bash
mlflow run project --experiment-name wines_rf_tuning -e train
```
Para ejecutar la app
```bash
mlflow run project -e serve
mlflow run project 
```