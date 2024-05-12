import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Función para cargar GloVe desde un archivo de texto
def load_glove_model(glove_file_path):
    print("Cargando GloVe Model")
    with open(glove_file_path, 'r', encoding='utf8') as f:
        model = {}
        for line in f:
            parts = line.split()
            word = parts[0]
            embedding = np.array([float(val) for val in parts[1:]])
            model[word] = embedding
        print(f"Modelo cargado. Se cargaron {len(model)} palabras.")
    return model

# Ruta al archivo GloVe (modificar según sea necesario)
glove_path = '/content/drive/MyDrive/ModelosEmbeddings/ModelosGlove/glove.6B.100d.txt'
embeddings = load_glove_model(glove_path)

# Seleccionar un subconjunto de palabras para visualizar
words = ['smartphone', 'tablet', 'laptop', 'desktop', 'AI', 'IoT', 'cloud', 'Business', 'marketing', 'finance', 'strategy', 'entrepreneurship', 'startup', 'innovation',
         'coffee', 'apple', 'car', 'park', 'house', 'book', 'phone', 'happy', 'sad', 'beautiful', 'dangerous', 'effective', 'innovative', 'reliable',
         'run', 'jump', 'speak', 'write', 'listen', 'drive', 'sustainability', 'environment', 'recycling', 'green', 'pollution', 'energy',
         'health', 'medicine', 'fitness', 'disease', 'vaccine', 'therapy']

# Filtrar palabras que existen en el modelo y extraer los embeddings correspondientes
filtered_words = [word for word in words if word in embeddings]
word_vectors = np.array([embeddings[word] for word in filtered_words])

# Usar t-SNE para reducir la dimensionalidad
tsne = TSNE(n_components=2, random_state=0, perplexity=min(30, len(word_vectors) - 1))
embeddings_2d = tsne.fit_transform(word_vectors)

# Graficar los embeddings en 2D
plt.figure(figsize=(10, 10))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])

# Anotar cada punto con su respectiva palabra
for i, word in enumerate(filtered_words):
    plt.annotate(word, xy=(embeddings_2d[i, 0], embeddings_2d[i, 1]), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

plt.show()
