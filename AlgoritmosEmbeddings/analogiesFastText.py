import numpy as np
import gensim
from gensim.models import FastText, KeyedVectors
from scipy.spatial.distance import cosine

# Función para cargar el modelo FastText desde un archivo binario
def load_fasttext_model (fasttext_file_path, load_full_model=False):
    print("Cargando FastText Model")
    if load_full_model:
        # Carga el modelo completo de FastText
        model = FastText.load_facebook_model(fasttext_file_path)
    else:
        # Carga solo los vectores de palabras (más eficiente en memoria)
        model = KeyedVectors.load(fasttext_file_path)
    print(f"Modelo cargado. Se cargaron {len(model.wv.key_to_index)} palabras.")
    return model

# Función para encontrar el vector más cercano
def find_closest_embeddings(embedding, embeddings_model, exclude_words=[]):
    closest_word = None
    min_distance = float('inf')
    for word, idx in embeddings_model.wv.key_to_index.items():
        if word.lower() in exclude_words:
            continue
        dist = cosine(embedding, embeddings_model.wv[word])
        if dist < min_distance:
            min_distance = dist
            closest_word = word
    return closest_word

# Función para realizar analogías
def analogy(a, b, c, embeddings_model):
    a, b, c = a.lower(), b.lower(), c.lower()
    exclude_words = [a, b, c]
    if all(word in embeddings_model.wv.key_to_index for word in [a, b, c]):
        result_vector = embeddings_model.wv[c] + (embeddings_model.wv[b] - embeddings_model.wv[a])
        closest_word = find_closest_embeddings(result_vector, embeddings_model, exclude_words)
        return closest_word
    else:
        return "Una o más palabras no están en el vocabulario."

# Función para encontrar vecinos cercanos
def nearest_neighbors(word, embeddings_model, n=5):
    word = word.lower()
    if word in embeddings_model.wv.key_to_index:
        neighbors = embeddings_model.wv.most_similar(word, topn=n)
        return [neighbor[0] for neighbor in neighbors]
    else:
        return "La palabra no está en el vocabulario."

# Carga el modelo de FastText
fasttext_path = '/media/escar/Escarleth/DatosEmbeddings/cc.en.300.bin'
embeddings = load_fasttext_model(fasttext_path, load_full_model=False)

# Ejemplos de uso
print("king is to man as queen is to  ->", analogy('king', 'man', 'queen', embeddings))
print("Vecinos cercanos de 'king':", nearest_neighbors('king', embeddings))
