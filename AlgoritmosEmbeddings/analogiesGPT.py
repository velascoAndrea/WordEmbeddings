import openai
from dotenv import load_dotenv
import os

# Acceder a la clave de API desde la variable de entorno
api_key = os.getenv('OPENAI_API_KEY')

if api_key is None:
    raise ValueError("API key no encontrada. Asegúrate de que la variable de entorno OPENAI_API_KEY esté configurada correctamente en el archivo .env.")

openai.api_key = api_key

def gpt3_analogy(a, b, c):
    prompt = f"{a} is to {b} as {c} is to"
    response = openai.Completion.create(
        engine="text-davinci-002",  # Asegúrate de elegir el engine correcto
        prompt=prompt,
        max_tokens=50
    )
    return response.choices[0].text.strip()

# Ejemplo de uso de la función de analogía con GPT-3
a = 'king'
b = 'man'
c = 'queen'
result = gpt3_analogy(a, b, c)
print(f"{a} is to {b} as {c} is to {result}")
