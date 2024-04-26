import openai
from dotenv import load_dotenv
import os
import numpy as np
from scipy.spatial.distance import cosine

# Carga la configuración del archivo .env
load_dotenv()

# Establece la clave API desde la variable de entorno
openai.api_key = os.getenv("OPENAI_API_KEY")

def solve_analogy(a, b, c):
    """Resuelve analogías utilizando GPT-3, intentando devolver solo una palabra."""
    prompt = f"solve this analogy with the embedding closest to the word is to say if '{a}' is '{b}', what is '{c}'? excluding the words '{a}' '{b}' and '{c}'"
    response = openai.Completion.create(
        engine="text-embedding-ada-002",  # Usar el modelo más adecuado disponible
        prompt=prompt,
        max_tokens=1,  # Limitar a solo un token de salida
        n=1,  # Una sola respuesta
        stop=["\n", "."],  # Detiene la generación en el primer espacio o punto
        temperature=0.3,  # Baja temperatura para respuestas más predecibles
        top_p=1.0  # Usar el modo núcleo para una respuesta más enfocada
    )
    return response.choices[0].text.strip()

# Ejemplo de uso de la función de analogía
a = 'france'
b = 'paris'
c = 'londom'
result = solve_analogy(a, b, c)
print(f"{a} is to {b} as {c} is to {result}")

