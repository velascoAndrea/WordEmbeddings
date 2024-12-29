import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import openai
from dotenv import load_dotenv
import os

# Carga la configuración del archivo .env
load_dotenv()

# Establece la clave API desde la variable de entorno
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_gpt_embeddings(texts):
    # Preparar los embeddings
    embeddings = []
    for text in texts:
        if not text.strip():
            embeddings.append(np.zeros(1024))  # Asumiendo que la dimensión del embedding es 1024
            continue
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        embedding_vector = np.array(response['data'][0]['embedding'])
        norm = np.linalg.norm(embedding_vector)
        if norm > 0:
            normalized_vector = embedding_vector / norm
        else:
            normalized_vector = np.zeros_like(embedding_vector)
        embeddings.append(normalized_vector)
    return np.array(embeddings)

# Lista de descripciones para visualizar
texts = [
    'smartphone', 'tablet', 'laptop', 'desktop', 'AI', 'IoT', 'cloud', 'Business', 'marketing', 'finance', 'strategy', 'entrepreneurship', 'startup', 'innovation',
         'coffee', 'apple', 'car', 'park', 'house', 'book', 'phone', 'happy', 'sad', 'beautiful', 'dangerous', 'effective', 'innovative', 'reliable',
         'run', 'jump', 'speak', 'write', 'listen', 'drive', 'sustainability', 'environment', 'recycling', 'green', 'pollution', 'energy',
         'health', 'medicine', 'fitness', 'disease', 'vaccine', 'therapy'
]

# Generar embeddings GPT para las descripciones dadas
embeddings = generate_gpt_embeddings(texts)

# Usar t-SNE para reducir la dimensionalidad a 2 componentes
tsne = TSNE(n_components=2, random_state=0, perplexity=min(30, len(embeddings) - 1))
embeddings_2d = tsne.fit_transform(embeddings)

# Graficar los embeddings en un espacio 2D
plt.figure(figsize=(12, 12))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='blue', edgecolors='k', s=50)
for i, text in enumerate(texts):
    plt.annotate(text, (embeddings_2d[i, 0], embeddings_2d[i, 1]), textcoords="offset points", xytext=(0,10), ha='center')
plt.title("Visualización de Embeddings GPT")
plt.show()