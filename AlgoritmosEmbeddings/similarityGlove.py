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

# Función para generar embeddings para una descripción
def generate_description_embeddings(description, embeddings):
    words = description.lower().split()
    vectors = [embeddings[word] for word in words if word in embeddings]
    if vectors:
        mean_vector = np.mean(vectors, axis=0)
        # Normalizar el vector resultante
        norm = np.linalg.norm(mean_vector)
        if norm > 0:
            normalized_vector = mean_vector / norm
            return normalized_vector
        else:
            return np.zeros_like(mean_vector)  # Devuelve un vector cero si la norma es cero
    else:
        return np.zeros(next(iter(embeddings.values())).shape)  # Devuelve un vector cero si no hay embeddings disponibles


# Función para calcular la similitud del coseno
def cosine_similarity(vec1, vec2):
    if np.any(vec1) and np.any(vec2):
        return np.dot(vec1, vec2)  # Producto punto de dos vectores normalizados
    else:
        return 0  # Retorna 0 similitud si alguno de los vectores es cero

# Carga el modelo de GloVe
glove_path = '/content/drive/MyDrive/ModelosEmbeddings/ModelosGlove/glove.6B.100d.txt' # Cambia esto por la ruta de tu archivo de GloVe
embeddings = load_glove_model(glove_path)

#-----------------------------------------------------------------------------------------------1----------------------------------------------------------------------------------------------------------------

# Descripción escrita por humanos
human_description = "Latest generation smartphone with 5G technology"

# Descripción generada por LLM Smartphone
llm_description = "Experience cutting-edge technology with our latest smartphone, featuring a seamless OLED display, 5G connectivity, and an AI-powered camera that captures perfect shots every time."

# Generar incrustaciones para ambas descripciones
embeddings_human = generate_description_embeddings(human_description, embeddings)
embeddings_llm = generate_description_embeddings(llm_description, embeddings)

# Calcular la similitud del coseno
similarity_score = cosine_similarity(embeddings_human, embeddings_llm)
print(f"Similitud del coseno entre la descripción escrita por humanos y la generada por LLM Smartphone es: {similarity_score}")

#--------------------------------------------------------------------------------------------2-------------------------------------------------------------------------------------------------------------------

human_description2 = "beautiful day we spent today"
#Smartwatch
llm_description2 = "Keep your health in check with our smartwatch that monitors heart rate, tracks sleep patterns, and offers personalized fitness coaching, all wrapped in a sleek, water-resistant design."

embeddings_human2 = generate_description_embeddings(human_description2, embeddings)
embeddings_llm2 = generate_description_embeddings(llm_description2, embeddings)

similarity_score2 = cosine_similarity(embeddings_human2, embeddings_llm2)
print(f"Similitud del coseno entre la descripción escrita por humanos y la generada por LLM  Smartwatch es: {similarity_score2}")

#--------------------------------------------------------------------------------------------3-------------------------------------------------------------------------------------------------------------------
human_description3 = "Latest generation VR with 4K technology"
#VR Headset
llm_description3 = "Dive into new worlds with our VR headset, providing an immersive experience with high-resolution graphics, 360-degree audio, and intuitive motion tracking."

embeddings_human3 = generate_description_embeddings(human_description3, embeddings)
embeddings_llm3 = generate_description_embeddings(llm_description3, embeddings)

similarity_score3 = cosine_similarity(embeddings_human3, embeddings_llm3)
print(f"Similitud del coseno entre la descripción escrita por humanos y la generada por LLM VR Headset es: {similarity_score3}")

#------------------------------------------------------------------------------------------4---------------------------------------------------------------------------------------------------------------------
human_description4 = "Latest generation drone with 4K camera and long-lasting battery"
#Drone
llm_description4 = "Capture stunning aerial footage with our drone, equipped with a 4K camera, obstacle avoidance technology, and a long-lasting battery for extended flight times."

embeddings_human4 = generate_description_embeddings(human_description4, embeddings)
embeddings_llm4 = generate_description_embeddings(llm_description4, embeddings)

similarity_score4 = cosine_similarity(embeddings_human4, embeddings_llm4)
print(f"Similitud del coseno entre la descripción escrita por humanos y la generada por LLM Drone es: {similarity_score4}")

#------------------------------------------------------------------------------------------5---------------------------------------------------------------------------------------------------------------------
human_description5 = "latest generation computer with SSD hard drive"
#Gaming Console
llm_description5 = "Unleash the ultimate gaming experience with our console, offering exclusive games, ultra-high-speed SSD for faster load times, and an innovative controller that brings gameplay to life"

embeddings_human5 = generate_description_embeddings(human_description5, embeddings)
embeddings_llm5 = generate_description_embeddings(llm_description5, embeddings)

similarity_score5 = cosine_similarity(embeddings_human5, embeddings_llm5)
print(f"Similitud del coseno entre la descripción escrita por humanos y la generada por LLM  Gaming Console es: {similarity_score5}")


#------------------------------------------------------------------------------------------6---------------------------------------------------------------------------------------------------------------------
human_description6 = "color typewriter"
#Laptop
llm_description6 = "Our laptop combines portability with power, featuring a lightweight design, robust processing capabilities, and a battery that lasts all day on a single charge."

embeddings_human6 = generate_description_embeddings(human_description6, embeddings)
embeddings_llm6 = generate_description_embeddings(llm_description6, embeddings)

similarity_score6 = cosine_similarity(embeddings_human6, embeddings_llm6)
print(f"Similitud del coseno entre la descripción escrita por humanos y la generada por LLM Laptop es: {similarity_score6}")

#------------------------------------------------------------------------------------------7---------------------------------------------------------------------------------------------------------------------
human_description7 = "smart home for those who want to automate their home"
#Smart Home Device
llm_description7 = "Enhance your home's intelligence with our smart device that automates lighting, security, and temperature controls, easily managed via voice commands or your smartphone."

embeddings_human7 = generate_description_embeddings(human_description7, embeddings)
embeddings_llm7 = generate_description_embeddings(llm_description7, embeddings)

similarity_score7 = cosine_similarity(embeddings_human7, embeddings_llm7)
print(f"Similitud del coseno entre la descripción escrita por humanos y la generada por LLM  Smart Home Device es: {similarity_score7}")


#------------------------------------------------------------------------------------------8---------------------------------------------------------------------------------------------------------------------
human_description8 = "electric car with fast charging and elegant features that will make you love it"
#Electric Vehicle
llm_description8 = "Drive into the future with our electric vehicle, offering zero emissions, exceptional mileage range, and a fast-charging battery system."

embeddings_human8 = generate_description_embeddings(human_description8, embeddings)
embeddings_llm8 = generate_description_embeddings(llm_description8, embeddings)

similarity_score8 = cosine_similarity(embeddings_human8, embeddings_llm8)
print(f"Similitud del coseno entre la descripción escrita por humanos y la generada por LLM  Electric Vehicle es: {similarity_score8}")

#------------------------------------------------------------------------------------------9---------------------------------------------------------------------------------------------------------------------
human_description9 = "Now you can print what you imagine with the high precision 3D printer"
#3D Printer
llm_description9 = "Bring your creations to life with our 3D printer that offers precision printing, a variety of material compatibilities, and user-friendly software."

embeddings_human9 = generate_description_embeddings(human_description9, embeddings)
embeddings_llm9 = generate_description_embeddings(llm_description9, embeddings)

similarity_score9 = cosine_similarity(embeddings_human9, embeddings_llm9)
print(f"Similitud del coseno entre la descripción escrita por humanos y la generada por LLM  3D Printer es: {similarity_score9}")

#------------------------------------------------------------------------------------------10---------------------------------------------------------------------------------------------------------------------
human_description10 = "IDEAL ELECTRONIC VACUUM CLEANER FOR THOSE WHO HAVE LITTLE TIME"
#Robotic Vacuum Cleaner
llm_description10 = "Effortlessly keep your floors pristine with our robotic vacuum cleaner, featuring smart room navigation, multiple cleaning modes, and compatibility with voice assistant technologies."

embeddings_human10 = generate_description_embeddings(human_description10, embeddings)
embeddings_llm10 = generate_description_embeddings(llm_description10, embeddings)

similarity_score10 = cosine_similarity(embeddings_human10, embeddings_llm10)
print(f"Similitud del coseno entre la descripción escrita por humanos y la generada por LLM Robotic Vacuum Cleaneres: {similarity_score10}")