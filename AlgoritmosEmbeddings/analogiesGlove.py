import numpy as np
from scipy.spatial.distance import cosine

# Función para cargar los embeddings de GloVe desde un archivo
def load_glove_model(glove_file_path):
    print("Cargando GloVe Model")
    with open(glove_file_path, 'r', encoding='utf8') as f:
        model = {}
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            model[word] = embedding
        print(f"Modelo cargado. Se cargaron {len(model)} palabras.")
    return model

# Función para encontrar el vector más cercano (utilizado para analogías)
def find_closest_embeddings(embedding, embeddings_matrix, exclude_words=[]):
    return sorted(embeddings_matrix.keys(), key=lambda word: cosine(embeddings_matrix[word], embedding) if word not in exclude_words else 2)

# Función para realizar analogías de la forma a es a b como c es a __
def analogy(a, b, c, embeddings_matrix):
    a, b, c = a.lower(), b.lower(), c.lower()
    # Encontramos los vectores para cada palabra
    closest_words = find_closest_embeddings(embeddings_matrix[b] - embeddings_matrix[a] + embeddings_matrix[c], embeddings_matrix, exclude_words=[a, b, c])
    # Excluimos las palabras originales y devolvemos la más cercana
    return closest_words[0]

# Función para encontrar vecinos cercanos de una palabra
def nearest_neighbors(word, embeddings_matrix, n=5):
    word = word.lower()
    nearest = sorted(embeddings_matrix.keys(), key=lambda x: cosine(embeddings_matrix[x], embeddings_matrix[word]))
    nearest = [x for x in nearest if x != word]
    return nearest[:n]

# Carga el modelo de GloVe desde el archivo
glove_path = './glove/glove.6B.50d.txt' # Cambia esto por la ruta de tu archivo de GloVe
embeddings = load_glove_model(glove_path)

# Analiza analogías
print("Analogía: rey es a reina como hombre es a __")
print(analogy('king', 'man', 'queen', embeddings))

# Encuentra vecinos cercanos
print("Vecinos cercanos de 'python':")
print(nearest_neighbors('king', embeddings))
