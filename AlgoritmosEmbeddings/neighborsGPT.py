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
    """Genera y almacena embeddings de una lista de palabras utilizando GPT-3."""
    embeddings = {}
    for word in word_list:
        response = openai.Embedding.create(
            input=word,
            model="text-embedding-ada-002"
        )
        embedding = np.array(response['data'][0]['embedding'])
        embeddings[word] = embedding
    return embeddings

def find_closest_embeddings(query_embedding, embeddings, n=5, query_word=None):
    """Encuentra los n vecinos más cercanos de un embedding dado, excluyendo la palabra de consulta."""
    distances = []
    for word, embedding in embeddings.items():
        if word != query_word:  # Excluir la palabra de consulta del resultado
            dist = cosine(query_embedding, embedding)
            distances.append((word, dist))
    distances.sort(key=lambda x: x[1])
    return [word for word, _ in distances[:n]]

def nearest_neighbors(word, embeddings, n=5):
    """Devuelve los n vecinos más cercanos para una palabra dada, excluyéndola de los resultados."""
    word = word.lower()
    if word in embeddings:
        query_embedding = embeddings[word]
        return find_closest_embeddings(query_embedding, embeddings, n, query_word=word)
    else:
        return "La palabra no está en los embeddings generados."
def generate_and_store_embeddings(word_list):
    """Genera y almacena embeddings de una lista de palabras utilizando GPT-3."""
    embeddings = {}
    for word in word_list:
        response = openai.Embedding.create(
            input=word,
            model="text-embedding-ada-002"
        )
        embedding = np.array(response['data'][0]['embedding'])
        embeddings[word] = embedding
    return embeddings

def find_closest_embeddings(query_embedding, embeddings, n=5, query_word=None):
    """Encuentra los n vecinos más cercanos de un embedding dado, excluyendo la palabra de consulta."""
    distances = []
    for word, embedding in embeddings.items():
        if word != query_word:  # Excluir la palabra de consulta del resultado
            dist = cosine(query_embedding, embedding)
            distances.append((word, dist))
    distances.sort(key=lambda x: x[1])
    return [word for word, _ in distances[:n]]

def nearest_neighbors(word, embeddings, n=5):
    """Devuelve los n vecinos más cercanos para una palabra dada, excluyéndola de los resultados."""
    word = word.lower()
    if word in embeddings:
        query_embedding = embeddings[word]
        return find_closest_embeddings(query_embedding, embeddings, n, query_word=word)
    else:
        return "La palabra no está en los embeddings generados."


# Generar embeddings para un conjunto de palabras
word_list = ["queen","monarch","crown","throne","prince","royal","sovereign","realm","palace","regent","dynasty","heir","coronation","empire","noble",
             "paris","french","europe","eiffel","loire","versailles","lyon","marseille","normandy","alsace","napoleon","riviera","baguette","cheese",
             "tokyo","kyoto","sushi","samurai","japanese","physicist","relativity","science","newton","quantum",
             "man","girl","lady","mother","female","niece","uncle","cousin","brother","family",
             "january","february","march","april","may","june","july","august","september","octover","november","december",
             "winter","valentine","leap","italy","vatican","ancient","colosseum","roman","rome","florence","pasta","venice","italian",
             "she","he","him","his","man","they","he","her","woman","hers","month","king","elizabeth",
             "beijing","shanghai","chinese","asia","dragon","forest","leaf","wood","branch","oak",
             "vehicle","drive","engine","road","wheel","joy","happy","contentment","smile","pleasure",
             "innovation","science","computers","internet","tech","pretty",
             "lovely","gorgeous","beauty","stunning","risky","safe","threat","hazardous","perilous",
             "silent","calm","peaceful","noiseless","soft","amazon.com","online","store","ecommerce","kindle","books",
             "jungle","code","programming","software","data","computation",
             "money","finance","account","loan","financial","person","society","community","individuals","humans",
             "france","japan","einstein","nephew","china","tree","car","happiness","technology","beautiful","dangerous","quiet","amazon","algorithm","bank","people"]

embeddings = generate_and_store_embeddings(word_list)

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