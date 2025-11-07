import gradio as gr
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import plotly.express as px

MODEL_NAME = "WineQuality"

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

def get_model_metrics(model_name=MODEL_NAME, stage_or_version="champion"):
    client = mlflow.tracking.MlflowClient()
    model_versions = client.search_model_versions(f"name='{model_name}'")
    for mv in model_versions:
        if mv.aliases and mv.aliases[0] == stage_or_version or mv.version == stage_or_version:
            run_id = mv.run_id
            run = client.get_run(run_id)
            return run.data.metrics
    return {}

metrics = get_model_metrics(stage_or_version="champion")
if metrics:
    metrics_md = "## 📊Métricas del Modelo Champion\n" + "\n".join(
        [f"- **{k.upper()}**: {v:.4f}" for k, v in metrics.items()]
    )
else:
    metrics_md = "No se encontraron métricas para el modelo."

def list_registered_versions(model_name):
    client = MlflowClient()
    mvs = client.search_model_versions(f"name='{model_name}'")

    choices = []
    for mv in mvs:
        stage = getattr(mv, "current_stage", None) or "None"
        aliases = getattr(mv, "aliases", []) or []
        alias_str = f" | aliases: {','.join(aliases)}" if aliases else ""
        label = f"{mv.version} ({stage}){alias_str}"
        choices.append((label, mv.version))
    def sort_key(item):
        try:
            return int(item[1])
        except Exception:
            return item[1]
    choices = sorted(choices, key=sort_key)
    return choices

def compare_versions(versions):
    if not versions:
        return pd.DataFrame(), None

    rows = []
    metric_keys = set()

    for v in versions:
        metrics = get_model_metrics(MODEL_NAME, v)
        clean_metrics = {}
        for k, val in (metrics or {}).items():
            try:
                numeric = float(val)
            except Exception:
                numeric = val
            clean_metrics[k] = numeric
            if isinstance(numeric, (int, float)):
                metric_keys.add(k)
        rows.append({"version": str(v), **clean_metrics})

    df = pd.DataFrame(rows).fillna("N/A")

    # Reorder columns: version first, then sorted metric keys
    ordered_cols = ["version"] + sorted(k for k in df.columns if k != "version")
    df = df[ordered_cols]

    # If no numeric metrics to plot, return only table
    numeric_metric_cols = [c for c in df.columns if c != "version" and pd.to_numeric(df[c], errors="coerce").notna().any()]
    if not numeric_metric_cols:
        return df, None

    # Prepare long format for plotting
    melted = df.melt(id_vars="version", value_vars=numeric_metric_cols, var_name="metric", value_name="value")
    melted["value"] = pd.to_numeric(melted["value"], errors="coerce")

    fig = px.bar(melted, x="version", y="value", color="metric", barmode="group",
                 title=f"Comparación de métricas para {MODEL_NAME}")

    fig.update_layout(legend_title_text="Métrica")
    return df, fig

version_choices = list_registered_versions(MODEL_NAME)
if not version_choices:
    version_choice_labels = ["1 (Production)", "2 (Staging)", "3 (None)"]
    version_choice_values = ["1", "2", "3"]
else:
    label_to_value = {lbl: val for (lbl, val) in version_choices}
    version_choice_labels = [lbl for (lbl, _) in version_choices]
    version_choice_values = [label_to_value[lbl] for lbl in version_choice_labels]

with gr.Blocks(title="Predicción de Calidad de Vino") as demo:
    gr.Markdown("# Calidad de vinos")
    
    with gr.Tab("Entrada Manual"):
        inputs = []
        with gr.Row():
            for label in SPANISH_LABELS:
                inputs.append(gr.Number(label=label))
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
    
    with gr.Tab("Subir CSV"):
        gr.Markdown("Sube un archivo CSV con las columnas del dataset de vinos.")
        csv_input = gr.File(label="Archivo CSV", file_types=[".csv"])
    
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

    with gr.Tab("Comparar Versiones"):
        gr.Markdown("## Comparación entre Modelos")
        version_selector = gr.Dropdown(choices=version_choice_labels, label="Selecciona versiones (múltiple)", multiselect=True)
        compare_btn = gr.Button("🔍 Comparar versiones seleccionadas")
        comparison_table = gr.Dataframe(headers=["version"], label="Tabla de métricas por versión")
        comparison_plot = gr.Plot(label="Gráfica comparativa")

        def compare_wrapper(selected_labels):
            if not selected_labels:
                return pd.DataFrame(), None
            selected_versions = [label_to_value.get(lbl, lbl) if 'label_to_value' in globals() else lbl for lbl in selected_labels]
            df, fig = compare_versions(selected_versions)
            return df, fig

        compare_btn.click(fn=compare_wrapper, inputs=[version_selector], outputs=[comparison_table, comparison_plot])

demo.launch()