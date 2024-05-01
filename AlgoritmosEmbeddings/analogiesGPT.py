import openai
import numpy as np
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
import os


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

# Arreglo de palabras
embeddings = generate_and_store_embeddings(["king", "queen", "man", "woman", "prince",
                                            "scientist", "painter", "actress", "mother",
                                            "heir","heiress","princess","nephew","niece","husband",
                                            "aunt","french","english","japanese","chinese","uncle","wife",
                                            "italian","australian","france","paris","london","rome","italy","england","mexico","india","thailand",
                                            "japan","tokyo","beijing","miami",
                                            "florida","texas","houston","spain","madrid","australia","actor",
                                             "china","chinese","she","he","her","his","sister","brother","picasso","einstein",
                                            "january","february","march","april","may","june","july","august","september","octover","november","december",
                                            "pizza","tacos","sushi","curry","paella","bread","rice"])  # Si se desea analizar otras analogias en necesario incluirlas en el arreglo.




print("king is to man as queen is to  ->",solve_analogy_gpt3('king', 'man', 'queen', embeddings))
print("France is to Paris as london is to ->", solve_analogy_gpt3("france", "paris", "london", embeddings))
print("France is to Paris as rome is to ->", solve_analogy_gpt3('france', 'paris', 'rome', embeddings))
print("Paris is to France as italy is to ->", solve_analogy_gpt3('paris', 'france', 'italy', embeddings))
print("France is to French as english is to ->", solve_analogy_gpt3('france', 'french', 'english', embeddings))
print("France is to French as english is to ->", solve_analogy_gpt3('france', 'french', 'english', embeddings))
print("Japan is to Japanese as chinese is to ->", solve_analogy_gpt3('japan', 'japanese', 'chinese', embeddings))
print("Japan is to Japanese as italian is to ->", solve_analogy_gpt3('japan', 'japanese', 'italian', embeddings))
print("Japan is to Japanese as australian is to ->", solve_analogy_gpt3('japan', 'japanese', 'australian', embeddings))
print("December is to November as june is to ->", solve_analogy_gpt3('december', 'november', 'june', embeddings))
print("Miami is to Florida as texas is to ->", solve_analogy_gpt3('miami', 'florida', 'texas', embeddings))
print("Einstein is to scientist as painter is to ->", solve_analogy_gpt3('einstein', 'scientist', 'painter', embeddings))
print("China is to rice as bread is to ->", solve_analogy_gpt3('china', 'rice', 'bread', embeddings))
print("Man is to woman as she is to ->", solve_analogy_gpt3('man', 'woman', 'she', embeddings))
print("Man is to woman as aunt is to ->", solve_analogy_gpt3('man', 'woman', 'aunt', embeddings))
print("Man is to woman as sister is to ->", solve_analogy_gpt3('man', 'woman', 'sister', embeddings))
print("Man is to woman as wife is to ->", solve_analogy_gpt3('man', 'woman', 'wife', embeddings))
print("Man is to woman as actress is to ->", solve_analogy_gpt3('man', 'woman', 'actress', embeddings))
print("Man is to woman as mother is to ->", solve_analogy_gpt3('man', 'woman', 'mother', embeddings))
print("Heir is to heiress as princess is to ->", solve_analogy_gpt3('heir', 'heiress', 'princess', embeddings))
print("Nephew is to niece as aunt is to ->", solve_analogy_gpt3('nephew', 'niece', 'aunt', embeddings))
print("France is to Paris as tokyo is to ->", solve_analogy_gpt3('france', 'paris', 'tokyo', embeddings))
print("France is to Paris as beijing is to ->", solve_analogy_gpt3('france', 'paris', 'beijing', embeddings))
print("February is to January as november is to ->", solve_analogy_gpt3('february', 'january', 'november', embeddings))
print("Paris is to France as italy is to ->", solve_analogy_gpt3('paris', 'france', 'italy', embeddings))
