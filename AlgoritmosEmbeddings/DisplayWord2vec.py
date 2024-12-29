import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import KeyedVectors

# Cargar el modelo Word2Vec
model_path = '/content/drive/MyDrive/ModelosEmbeddings/GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

# Seleccionar un subconjunto de palabras para visualizar
words = ['smartphone', 'tablet', 'laptop', 'desktop', 'AI', 'IoT', 'cloud', 'Business', 'marketing', 'finance', 'strategy', 'entrepreneurship', 'startup', 'innovation',
         'coffee', 'apple', 'car', 'park', 'house', 'book', 'phone', 'happy', 'sad', 'beautiful', 'dangerous', 'effective', 'innovative', 'reliable',
         'run', 'jump', 'speak', 'write', 'listen', 'drive', 'sustainability', 'environment', 'recycling', 'green', 'pollution', 'energy',
         'health', 'medicine', 'fitness', 'disease', 'vaccine', 'therapy']

# Extraer los embeddings correspondientes
word_vectors = np.array([model[word] for word in words if word in model])

# Usar t-SNE para reducir la dimensionalidad, ajustar la perplexity adecuadamente
tsne = TSNE(n_components=2, random_state=0, perplexity=min(30, len(word_vectors) - 1))  # Perplexity no puede ser mayor que n_samples - 1
embeddings_2d = tsne.fit_transform(word_vectors)

# Graficar los embeddings en 2D
plt.figure(figsize=(10, 10))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])

# Anotar cada punto con su respectiva palabra
for i, word in enumerate(words):
    plt.annotate(word, xy=(embeddings_2d[i, 0], embeddings_2d[i, 1]), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

plt.show()
