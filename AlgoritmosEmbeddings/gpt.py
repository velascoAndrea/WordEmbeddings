import openai
from dotenv import load_dotenv
import os
import numpy as np
from scipy.spatial.distance import cosine

# Carga la configuración del archivo .env
load_dotenv()

# Establece la clave API desde la variable de entorno
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_gpt3_embedding(text):
    """Obtiene el embedding de un texto utilizando el modelo de OpenAI."""
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"  # Asegúrate de que este modelo sea el adecuado para obtener embeddings.
    )
    embedding = response['data'][0]['embedding']
    return np.array(embedding)

def solve_analogy_gpt3(a, b, c):
    """Resuelve analogías utilizando GPT-3, calculando el embedding más cercano."""
    # Obtiene embeddings para cada palabra
    embedding_a = get_gpt3_embedding(a)
    embedding_b = get_gpt3_embedding(b)
    embedding_c = get_gpt3_embedding(c)
    
    # Calcula el vector resultante de la analogía
    result_vector = embedding_c + (embedding_b - embedding_a)

    # Aquí necesitarías una función que calcule cuál es el embedding más cercano en tu dataset a result_vector
    # Esto implica tener una lista de palabras predefinidas y sus embeddings, similar a GloVe.
    # Como GPT-3 no devuelve directamente un diccionario de palabras y embeddings, esta parte es conceptual
    # y necesitarías ajustar según tu conjunto de datos específico o hacer una solicitud adicional para encontrar
    # el embedding más cercano usando otro método o servicio.

    closest_word = "queen,apple,tree,man,he,she"  # Esta parte necesita ser implementada según tus necesidades.
    return closest_word

# Ejemplo de uso de la función de analogía
a = 'king'
b = 'man'
c = 'woman'
result = solve_analogy_gpt3(a, b, c)
print(f"{a} is to {b} as {c} is to {result}")
