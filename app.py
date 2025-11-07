import gradio as gr
import mlflow
import pandas as pd

ENGLISH_FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol"
]

SPANISH_LABELS = [
    "Acidez fija", "Acidez volátil", "Ácido cítrico", "Azúcar residual",
    "Cloruros", "Dióxido de azufre libre", "Dióxido de azufre total", "Densidad",
    "pH", "Sulfatos", "Alcohol"
]

def load_model():
    model_uri = f"models:/WineQuality@champion"
    model = mlflow.sklearn.load_model(model_uri)
    return model

def predict_quality(*manual_inputs):
    model = load_model()
    if all(val is not None for val in manual_inputs):
        input_df = pd.DataFrame([manual_inputs], columns=ENGLISH_FEATURES)
        pred = model.predict(input_df.values)[0]
        return pred
    else:
        return ("Por favor completa los campos.")

def get_model_metrics(stage_or_version="champion"):
    client = mlflow.tracking.MlflowClient()
    model_versions = client.search_model_versions("name='WineQuality'")
    for mv in model_versions:
        if mv.aliases[0] == stage_or_version or mv.version == stage_or_version:
            run_id = mv.run_id
            run = client.get_run(run_id)
            return run.data.metrics
    return {}

metrics = get_model_metrics("champion")
if metrics:
    metrics_md = "## 📊Métricas del Modelo Champion\n" + "\n".join(
        [f"- **{k.upper()}**: {v:.4f}" for k, v in metrics.items()]
    )
else:
    metrics_md = "No se encontraron métricas para el modelo."

with gr.Blocks(title="Predicción de Calidad de Vino") as demo:
    gr.Markdown("# Calidad de vinos")
    
    with gr.Tab("Entrada Manual"):
        inputs = []
        with gr.Row():
            for label in SPANISH_LABELS:
                inputs.append(gr.Number(label=label))
    
    with gr.Tab("Subir CSV"):
        gr.Markdown("Sube un archivo CSV con las columnas del dataset de vinos.")
        csv_input = gr.File(label="Archivo CSV", file_types=[".csv"])
    
    with gr.Tab("Comparar Versiones"):
        gr.Markdown("## Comparación entre Modelos")
        version_selector = gr.Dropdown(
            choices=["v1.0", "v2.0", "v3.0"], 
            label="Seleccionar Versiones a Comparar",
            multiselect=True
        )
        comparison_table = gr.Dataframe(label="Comparación de Métricas")
        comparison_plot = gr.Plot(label="Gráfica de Comparación")
    
    # Common components that appear below ALL tabs
    predict_btn = gr.Button("🎯 Predecir Calidad", variant="primary")
    
    with gr.Row():
        with gr.Column():
            output_pred = gr.Textbox(label="Resultado")
    gr.Markdown(metrics_md)
    
    predict_btn.click(
        fn=predict_quality,
        inputs = inputs,
        outputs = output_pred
    )

demo.launch()