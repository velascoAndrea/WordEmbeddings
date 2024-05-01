import gensim
#from gensim.models import FastText
from scipy.spatial.distance import cosine
import fasttext
#import fasttext.util

# Función para cargar el modelo FastText desde un archivo binario
def load_fasttext_model(fasttext_file_path):
    print("Cargando modelo FastText")
    model =  fasttext.load_model(fasttext_file_path)
    #print(f"Modelo cargado. Se cargaron {len(model.wv.key_to_index)} palabras.")
    return model

# Función para encontrar el vector más cercano (utilizado para analogías)
def find_closest_embeddings(embedding, embeddings_model, exclude_words=[]):
    closest_word = None
    min_distance = float('inf')
    for word in embeddings_model.wv.key_to_index:
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

# Carga el modelo de FastText
fasttext_path = '/media/escar/Escarleth/DatosEmbeddings/cc.en.300.bin'  # Cambia esto por la ruta de tu archivo de FastText
embeddings = load_fasttext_model(fasttext_path)

# Ejemplos de uso
#print("king is to man as queen is to  ->", analogy('king', 'man', 'queen', embeddings))
#print("France is to Paris as london is to ->", analogy('france', 'paris', 'london', embeddings))
