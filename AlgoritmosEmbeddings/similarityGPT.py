import openai
import numpy as np
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
import os
# Carga la configuración del archivo .env
load_dotenv()

# Establece la clave API desde la variable de entorno
openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_gpt_embeddings(text):
    if not text.strip():  # Comprueba si el texto está vacío o solo contiene espacios en blanco
        return np.zeros(1024)  # Asumiendo que la dimensión del embedding es 1024
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    embedding_vector = np.array(response['data'][0]['embedding'])
    norm = np.linalg.norm(embedding_vector)
    if norm > 0:
        normalized_vector = embedding_vector / norm
        return normalized_vector
    return np.zeros_like(embedding_vector)

def cosine_similarity(vec1, vec2):
    # Verifica si alguno de los vectores es completamente cero
    if np.any(vec1) and np.any(vec2):
        return np.dot(vec1, vec2)
    else:
        # Retorna 0 si alguno de los vectores es completamente cero
        return 0
#-----------------------------------------------------------------------------------------------1----------------------------------------------------------------------------------------------------------------

# Descripción escrita por humanos
human_description = "Latest generation smartphone with 5G technology"

# Descripción generada por LLM Smartphone
llm_description = "Experience cutting-edge technology with our latest smartphone, featuring a seamless OLED display, 5G connectivity, and an AI-powered camera that captures perfect shots every time."

embeddings_human = generate_gpt_embeddings(human_description)
embeddings_llm = generate_gpt_embeddings(llm_description)

similarity_score = cosine_similarity(embeddings_human, embeddings_llm)
print(f"Similitud del coseno entre la descripción escrita por humanos y la generada por LLM Smartphone es: {similarity_score}")

#--------------------------------------------------------------------------------------------2-------------------------------------------------------------------------------------------------------------------

human_description2 = "beautiful day we spent today"
#Smartwatch
llm_description2 = "Keep your health in check with our smartwatch that monitors heart rate, tracks sleep patterns, and offers personalized fitness coaching, all wrapped in a sleek, water-resistant design."

embeddings_human2 = generate_gpt_embeddings(human_description2)
embeddings_llm2 = generate_gpt_embeddings(llm_description2)

similarity_score2 = cosine_similarity(embeddings_human2, embeddings_llm2)
print(f"Similitud del coseno entre la descripción escrita por humanos y la generada por LLM  Smartwatch es: {similarity_score2}")

#--------------------------------------------------------------------------------------------3-------------------------------------------------------------------------------------------------------------------
human_description3 = "Latest generation VR with 4K technology"
#VR Headset
llm_description3 = "Dive into new worlds with our VR headset, providing an immersive experience with high-resolution graphics, 360-degree audio, and intuitive motion tracking."

embeddings_human3 = generate_gpt_embeddings(human_description3)
embeddings_llm3 = generate_gpt_embeddings(llm_description3)

similarity_score3 = cosine_similarity(embeddings_human3, embeddings_llm3)
print(f"Similitud del coseno entre la descripción escrita por humanos y la generada por LLM VR Headset es: {similarity_score3}")

#------------------------------------------------------------------------------------------4---------------------------------------------------------------------------------------------------------------------
human_description4 = "Latest generation drone with 4K camera and long-lasting battery"
#Drone
llm_description4 = "Capture stunning aerial footage with our drone, equipped with a 4K camera, obstacle avoidance technology, and a long-lasting battery for extended flight times."

embeddings_human4 = generate_gpt_embeddings(human_description4)
embeddings_llm4 = generate_gpt_embeddings(llm_description4)

similarity_score4 = cosine_similarity(embeddings_human4, embeddings_llm4)
print(f"Similitud del coseno entre la descripción escrita por humanos y la generada por LLM Drone es: {similarity_score4}")

#------------------------------------------------------------------------------------------5---------------------------------------------------------------------------------------------------------------------
human_description5 = "latest generation computer with SSD hard drive"
#Gaming Console
llm_description5 = "Unleash the ultimate gaming experience with our console, offering exclusive games, ultra-high-speed SSD for faster load times, and an innovative controller that brings gameplay to life"

embeddings_human5 = generate_gpt_embeddings(human_description5)
embeddings_llm5 = generate_gpt_embeddings(llm_description5)

similarity_score5 = cosine_similarity(embeddings_human5, embeddings_llm5)
print(f"Similitud del coseno entre la descripción escrita por humanos y la generada por LLM  Gaming Console es: {similarity_score5}")


#------------------------------------------------------------------------------------------6---------------------------------------------------------------------------------------------------------------------
human_description6 = "color typewriter"
#Laptop
llm_description6 = "Our laptop combines portability with power, featuring a lightweight design, robust processing capabilities, and a battery that lasts all day on a single charge."

embeddings_human6 = generate_gpt_embeddings(human_description6)
embeddings_llm6 = generate_gpt_embeddings(llm_description6)

similarity_score6 = cosine_similarity(embeddings_human6, embeddings_llm6)
print(f"Similitud del coseno entre la descripción escrita por humanos y la generada por LLM Laptop es: {similarity_score6}")

#------------------------------------------------------------------------------------------7---------------------------------------------------------------------------------------------------------------------
human_description7 = "smart home for those who want to automate their home"
#Smart Home Device
llm_description7 = "Enhance your home's intelligence with our smart device that automates lighting, security, and temperature controls, easily managed via voice commands or your smartphone."

embeddings_human7 = generate_gpt_embeddings(human_description7)
embeddings_llm7 = generate_gpt_embeddings(llm_description7)

similarity_score7 = cosine_similarity(embeddings_human7, embeddings_llm7)
print(f"Similitud del coseno entre la descripción escrita por humanos y la generada por LLM  Smart Home Device es: {similarity_score7}")


#------------------------------------------------------------------------------------------8---------------------------------------------------------------------------------------------------------------------
human_description8 = "electric car with fast charging and elegant features that will make you love it"
#Electric Vehicle
llm_description8 = "Drive into the future with our electric vehicle, offering zero emissions, exceptional mileage range, and a fast-charging battery system."

embeddings_human8 = generate_gpt_embeddings(human_description8)
embeddings_llm8 = generate_gpt_embeddings(llm_description8)

similarity_score8 = cosine_similarity(embeddings_human8, embeddings_llm8)
print(f"Similitud del coseno entre la descripción escrita por humanos y la generada por LLM  Electric Vehicle es: {similarity_score8}")

#------------------------------------------------------------------------------------------9---------------------------------------------------------------------------------------------------------------------
human_description9 = "Now you can print what you imagine with the high precision 3D printer"
#3D Printer
llm_description9 = "Bring your creations to life with our 3D printer that offers precision printing, a variety of material compatibilities, and user-friendly software."

embeddings_human9 = generate_gpt_embeddings(human_description9)
embeddings_llm9 = generate_gpt_embeddings(llm_description9)

similarity_score9 = cosine_similarity(embeddings_human9, embeddings_llm9)
print(f"Similitud del coseno entre la descripción escrita por humanos y la generada por LLM  3D Printer es: {similarity_score9}")

#------------------------------------------------------------------------------------------10---------------------------------------------------------------------------------------------------------------------
human_description10 = "IDEAL ELECTRONIC VACUUM CLEANER FOR THOSE WHO HAVE LITTLE TIME"
#Robotic Vacuum Cleaner
llm_description10 = "Effortlessly keep your floors pristine with our robotic vacuum cleaner, featuring smart room navigation, multiple cleaning modes, and compatibility with voice assistant technologies."

embeddings_human10 = generate_gpt_embeddings(human_description10)
embeddings_llm10 = generate_gpt_embeddings(llm_description10)

similarity_score10 = cosine_similarity(embeddings_human10, embeddings_llm10)
print(f"Similitud del coseno entre la descripción escrita por humanos y la generada por LLM Robotic Vacuum Cleaneres: {similarity_score10}")
