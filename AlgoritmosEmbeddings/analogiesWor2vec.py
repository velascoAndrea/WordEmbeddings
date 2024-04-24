import numpy as np
import gensim
from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine

# Función para cargar el modelo Word2Vec desde un archivo
def load_word2vec_model(word2vec_file_path):
    print("Cargando Word2Vec Model")
    model = KeyedVectors.load_word2vec_format(word2vec_file_path, binary=True)  # Cambiar a True si es binario
    print(f"Modelo cargado. Se cargaron {len(model.key_to_index)} palabras.")
    return model

# Función para encontrar el vector más cercano (utilizado para analogías)
def find_closest_embeddings(embedding, embeddings_model, exclude_words=[]):
    return sorted(embeddings_model.key_to_index.keys(), 
                  key=lambda word: cosine(embeddings_model[word], embedding) if word not in exclude_words else 2)

# Función para realizar analogías de la forma a es a b como c es a __
def analogy(a, b, c, embeddings_model):
    a, b, c = a.lower(), b.lower(), c.lower()
    # Encontramos los vectores para cada palabra
    closest_words = find_closest_embeddings(embeddings_model[b] - embeddings_model[a] + embeddings_model[c], embeddings_model, exclude_words=[a, b, c])
    # Excluimos las palabras originales y devolvemos la más cercana
    return closest_words[0]

# Función para encontrar vecinos cercanos de una palabra
def nearest_neighbors(word, embeddings_model, n=5):
    word = word.lower()
    nearest = sorted(embeddings_model.key_to_index.keys(), key=lambda x: cosine(embeddings_model[x], embeddings_model[word]))
    nearest = [x for x in nearest if x != word]
    return nearest[:n]

# Carga el modelo de Word2Vec desde el archivo
word2vec_path = '/home/escar/Descargas/glove/GoogleNews-vectors-negative300.bin'  # Cambia esto por la ruta de tu archivo de Word2Vec
embeddings = load_word2vec_model(word2vec_path)

# Analiza analogías
#print("Analogía: king is to queen as man is to __")
print(analogy('king', 'queen', 'man', embeddings))
print(analogy('france', 'paris', 'london', embeddings))
print(analogy('france', 'paris', 'rome', embeddings))
print(analogy('paris', 'france', 'italy', embeddings))
print(analogy('france', 'french', 'english', embeddings))
print(analogy('japan', 'japanese', 'chinese', embeddings))
print(analogy('japan', 'japanese', 'italian', embeddings))
print(analogy('japan', 'japanese', 'australian', embeddings))
print(analogy('december', 'november', 'june', embeddings))
print(analogy('miami', 'florida', 'texas', embeddings))
print(analogy('einstein', 'scientist', 'painter', embeddings))
print(analogy('china', 'rice', 'bread', embeddings))
print(analogy('man', 'woman', 'she', embeddings))
print(analogy('man', 'woman', 'aunt', embeddings))
print(analogy('man', 'woman', 'sister', embeddings))
print(analogy('man', 'woman', 'wife', embeddings))
print(analogy('man', 'woman', 'actress', embeddings))
print(analogy('man', 'woman', 'mother', embeddings))
print(analogy('heir', 'heiress', 'princess', embeddings))
print(analogy('nephew', 'niece', 'aunt', embeddings))
print(analogy('france', 'paris', 'tokyo', embeddings))
print(analogy('france', 'paris', 'beijing', embeddings))
print(analogy('february', 'january', 'november', embeddings))
print(analogy('france', 'paris', 'rome', embeddings))
print(analogy('paris', 'france', 'italy', embeddings))

# Encuentra vecinos cercanos
print("Vecinos cercanos de 'king':")
print(nearest_neighbors('king', embeddings))
