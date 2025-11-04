import gradio as gr

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

with gr.Blocks(title="Predicción de Calidad de Vino") as demo:
    gr.Markdown("# Calidad de Vinos")
    
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
            quality_gauge = gr.Label(label="Calidad")
        with gr.Column():
            output_metrics = gr.Textbox(label="Métricas")
            output_prob = gr.Number(label="Confianza")

demo.launch()