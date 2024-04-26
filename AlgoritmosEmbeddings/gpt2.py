import openai
from dotenv import load_dotenv
import os
import numpy as np
from scipy.spatial.distance import cosine

# Carga la configuración del archivo .env
load_dotenv()

# Establece la clave API desde la variable de entorno
openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_and_store_embeddings(word_list):
    embeddings = {}
    for word in word_list:
        response = openai.Embedding.create(
            input=word,
            model="text-embedding-ada-002"
        )
        embedding = np.array(response['data'][0]['embedding'])
        embeddings[word] = embedding
    return embeddings

def find_closest_embedding(result_vector, embeddings, exclude_words):
    closest_word = None
    min_distance = float('inf')
    for word, embedding in embeddings.items():
        if word.lower() in exclude_words:
            continue  # Salta esta palabra si debe ser excluida
        dist = cosine(result_vector, embedding)
        if dist < min_distance:
            min_distance = dist
            closest_word = word
    return closest_word

def solve_analogy_gpt3(a, b, c, embeddings):
    """Resuelve analogías utilizando GPT-3, excluyendo palabras específicas y calculando el embedding más cercano."""
    # Convertir a minúsculas y calcular embeddings
    a, b, c = a.lower(), b.lower(), c.lower()
    embedding_a = embeddings[a]
    embedding_b = embeddings[b]
    embedding_c = embeddings[c]

    # Lista de palabras a excluir
    exclude_words = [a, b, c]

    # Calcula el vector resultante de la analogía
    result_vector =embedding_a -embedding_b + embedding_c 

    # Encuentra el embedding más cercano al vector resultante, excluyendo las palabras dadas
    closest_word = find_closest_embedding(result_vector, embeddings, exclude_words)
    return closest_word

# Ejemplo de uso de la función para resolver una analogía
a = 'germany' 
b = 'berlin'
c = 'spain'
embeddings = generate_and_store_embeddings(["king", "queen", "man", "woman", "scientist", "painter", "actress", "mother", "heir","heiress","princess","nephew","niece", "aunt",
                                            "french","english","japanese","chinese","italian","australian",
                                            "france","paris","london","rome","italy","japan","tokyo","beijing","miami","florida","texas","houston","spain","madrid"
                                            "china","chinese","she","he","her","his","sister","brother",
                                            "january","february","march","april","may","june","july","august","september","octover","november","december",

                                              ])  # Si se desea analizar otras analogias en necesario incluirlas en el arreglo.
result = solve_analogy_gpt3(a, b, c, embeddings)
print(f"'{a}' is to '{b}' as '{c}' is to {result}")
