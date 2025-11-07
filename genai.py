import mlflow
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
gemini_key = os.environ.get("GEMINI_API_KEY")
gemini = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=gemini_key)

def genai_explanations(mode, features, pred):
    try:
        if isinstance(features, dict):
            features_str = ", ".join(f"{k}: {v}" for k, v in features.items())
        else:
            features_str = str(features)
 
        if mode == "individual":
            prompt = f"""Eres un asistente experto en análisis sensorial de vinos.  
                Te daré los valores de las características químicas de un vino y la calidad que el modelo predijo. Tu tarea es generar una breve explicación en español (1 o 2 oraciones) que justifique esa predicción, mencionando las características más influyentes y si sus valores son altos o bajos. Usa un tono natural, claro y conciso. Evita tecnicismos.
                Valores del vino: {features_str}
                Predicción del modelo (calidad): {pred}
                Ejemplo de salida: “Este vino tiene alta calidad porque presenta alta acidez fija y bajo contenido de azúcar residual.”
                """
        elif mode == "csv":
            prompt = f"""Eres un asistente experto en análisis de vinos.  
                Se te entregarán datos de varios vinos, junto con sus predicciones de calidad hechas por un modelo de machine learning. Tu tarea es generar una breve explicación general (2 a 3 oraciones) en español, describiendo las tendencias observadas entre las características químicas y la calidad promedio.
                No analices vinos individuales, sino patrones globales. Menciona qué características parecen asociarse con mayor calidad y cuáles con menor, usando un lenguaje natural.
                Valores del vino: {features}
                Predicción del modelo (calidad): {pred}
                Ejemplo de salida: “En general, los vinos con mayor calidad presentan más alcohol y menor cantidad de azúcares residuales, mientras que un pH bajo y altos sulfatos también se asocian a calidades superiores.”
            """
        else:
            raise ValueError("Modo no válido: usa 'individual' o 'csv'.")

        resp = gemini.chat.completions.create(
            model="models/gemini-2.5-flash-lite",
            messages=[{"role" : "user", "content" : prompt}],
        )

        explanation = resp.choices[0].message.content
        mlflow.log_text(explanation, artifact_file=f"explanations/{mode}_explanation.txt")
        return explanation

    except Exception as e:
        print(f"Error: {e}")
        return f"Error: {e}"