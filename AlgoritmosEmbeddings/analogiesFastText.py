import numpy as np
from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine

# Función para cargar el modelo de FastText
def load_fasttext_model(fasttext_file_path):
    print("Cargando modelo de FastText")
    model = KeyedVectors.load(fasttext_file_path)
    print(f"Modelo cargado. Se cargaron {len(model.key_to_index)} palabras.")
    return model

# Función para encontrar el vector más cercano (utilizado para analogías)
def find_closest_embeddings(embedding, embeddings_model, exclude_words=[]):
    return sorted(embeddings_model.key_to_index.keys(), key=lambda word: cosine(embeddings_model[word], embedding) if word not in exclude_words else 2)

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

# Carga el modelo de FastText desde el archivo
fasttext_path = './/home/escar/Descargas/glove/cc.en.300.vec' 
embeddings = load_fasttext_model(fasttext_path)

# Ejemplo de analogías
print("king is to man as queen is to  ->",analogy('king', 'man', 'queen', embeddings))
print("France is to Paris as london is to ->", analogy('france', 'paris', 'london', embeddings))
print("France is to Paris as rome is to ->", analogy('france', 'paris', 'rome', embeddings))
print("Paris is to France as italy is to ->", analogy('paris', 'france', 'italy', embeddings))
print("France is to French as english is to ->", analogy('france', 'french', 'english', embeddings))
print("Japan is to Japanese as chinese is to ->", analogy('japan', 'japanese', 'chinese', embeddings))
print("Japan is to Japanese as italian is to ->", analogy('japan', 'japanese', 'italian', embeddings))
print("Japan is to Japanese as australian is to ->", analogy('japan', 'japanese', 'australian', embeddings))
print("December is to November as june is to ->", analogy('december', 'november', 'june', embeddings))
print("Miami is to Florida as texas is to ->", analogy('miami', 'florida', 'texas', embeddings))
print("Einstein is to scientist as painter is to ->", analogy('einstein', 'scientist', 'painter', embeddings))
print("China is to rice as bread is to ->", analogy('china', 'rice', 'bread', embeddings))
print("Man is to woman as she is to ->", analogy('man', 'woman', 'she', embeddings))
print("Man is to woman as aunt is to ->", analogy('man', 'woman', 'aunt', embeddings))
print("Man is to woman as sister is to ->", analogy('man', 'woman', 'sister', embeddings))
print("Man is to woman as wife is to ->", analogy('man', 'woman', 'wife', embeddings))
print("Man is to woman as actress is to ->", analogy('man', 'woman', 'actress', embeddings))
print("Man is to woman as mother is to ->", analogy('man', 'woman', 'mother', embeddings))
print("Heir is to heiress as princess is to ->", analogy('heir', 'heiress', 'princess', embeddings))
print("Nephew is to niece as aunt is to ->", analogy('nephew', 'niece', 'aunt', embeddings))
print("France is to Paris as tokyo is to ->", analogy('france', 'paris', 'tokyo', embeddings))
print("France is to Paris as beijing is to ->", analogy('france', 'paris', 'beijing', embeddings))
print("February is to January as november is to ->", analogy('february', 'january', 'november', embeddings))
print("Paris is to France as italy is to ->", analogy('paris', 'france', 'italy', embeddings))
# Ejemplo de vecinos cercanos
#print("Vecinos cercanos de 'king':", nearest_neighbors('king', embeddings))
