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
    closest_words = find_closest_embeddings(embeddings_matrix[a] - embeddings_matrix[b]  + embeddings_matrix[c], embeddings_matrix, exclude_words=[a, b, c])
    # Excluimos las palabras originales y devolvemos la más cercana
    return closest_words[0]

# Función para encontrar vecinos cercanos de una palabra
def nearest_neighbors(word, embeddings_matrix, n=5):
    word = word.lower()
    nearest = sorted(embeddings_matrix.keys(), key=lambda x: cosine(embeddings_matrix[x], embeddings_matrix[word]))
    nearest = [x for x in nearest if x != word]
    return nearest[:n]

# Carga el modelo de GloVe desde el archivo
glove_path = '/media/escar/Escarleth/DatosEmbeddings/glove.6B/glove.6B.50d.txt' # Cambia esto por la ruta de tu archivo de GloVe
embeddings = load_glove_model(glove_path)

# Analiza analogías
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


print("Vecinos cercanos")
print("Vecinos cercanos de king",nearest_neighbors('king',embeddings))
print("Vecinos cercanos de france",nearest_neighbors('france',embeddings))
print("Vecinos cercanos de japan",nearest_neighbors('japan',embeddings))
print("Vecinos cercanos de einstein",nearest_neighbors('einstein',embeddings))
print("Vecinos cercanos de woman",nearest_neighbors('woman',embeddings))
print("Vecinos cercanos de nephew",nearest_neighbors('nephew',embeddings))
print("Vecinos cercanos de february",nearest_neighbors('february',embeddings))
print("Vecinos cercanos de rome",nearest_neighbors('rome',embeddings))
print("Vecinos cercanos de italy",nearest_neighbors('italy',embeddings))
print("Vecinos cercanos de he",nearest_neighbors('he',embeddings))
print("Vecinos cercanos de she",nearest_neighbors('she',embeddings))
print("Vecinos cercanos de january",nearest_neighbors('january',embeddings))
print("Vecinos cercanos de queen",nearest_neighbors('queen',embeddings))
print("Vecinos cercanos de china",nearest_neighbors('china',embeddings))
print("Vecinos cercanos de tree",nearest_neighbors('tree',embeddings))
print("Vecinos cercanos de car",nearest_neighbors('car',embeddings))
print("Vecinos cercanos de hapiness",nearest_neighbors('happiness',embeddings))
print("Vecinos cercanos de technology",nearest_neighbors('technology',embeddings))
print("Vecinos cercanos de beautiful",nearest_neighbors('beautiful',embeddings))
print("Vecinos cercanos de dangerous",nearest_neighbors('dangerous',embeddings))
print("Vecinos cercanos de quiet",nearest_neighbors('quiet',embeddings))
print("Vecinos cercanos de amazon",nearest_neighbors('amazon',embeddings))
print("Vecinos cercanos de algorithm",nearest_neighbors('algorithm',embeddings))
print("Vecinos cercanos de bank",nearest_neighbors('bank',embeddings))
print("Vecinos cercanos de people",nearest_neighbors('people',embeddings))









