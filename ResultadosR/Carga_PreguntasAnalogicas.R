# Cargar datos para cada modelo
preguntas_glove_analogicas <- read.csv("/home/velasco/Descargas/ResultadoEmbeddings/Preguntas_Analogicas_Glove.csv")
preguntas_gpt_analogicas <- read.csv("/home/velasco/Descargas/ResultadoEmbeddings/Preguntas_Analogicas_GPT.csv")
preguntas_word2vec_analogicas <- read.csv("/home/velasco/Descargas/ResultadoEmbeddings/Preguntas_Analogicas_Word2vec.csv")

# Agregar columna para identificar el modelo
preguntas_glove_analogicas$Modelo <- "GloVe"
preguntas_gpt_analogicas$Modelo <- "GPT"
preguntas_word2vec_analogicas$Modelo <- "Word2Vec"

# Combinar los datos en un solo dataframe
library(dplyr)
preguntasAnalogicas <- bind_rows(preguntas_glove_analogicas, preguntas_gpt_analogicas, preguntas_word2vec_analogicas)

# Inspeccionar los datos
head(preguntasAnalogicas)

