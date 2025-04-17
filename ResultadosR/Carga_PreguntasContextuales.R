# Cargar datos para cada modelo
preguntas_glove_contex <- read.csv("/home/velasco/Descargas/ResultadoEmbeddings/Preguntas_contextuales_Glove.csv")
preguntas_gpt_contex <- read.csv("/home/velasco/Descargas/ResultadoEmbeddings/Preguntas_contextuales_GPT.csv")
preguntas_word2vec_contex <- read.csv("/home/velasco/Descargas/ResultadoEmbeddings/Preguntas_contextuales_Word2vec.csv")

# Agregar columna para identificar el modelo
preguntas_glove_contex$Modelo <- "GloVe"
preguntas_gpt_contex$Modelo <- "GPT"
preguntas_word2vec_contex$Modelo <- "Word2Vec"

# Combinar los datos en un solo dataframe
library(dplyr)
preguntasContextuales <- bind_rows(preguntas_glove_contex, preguntas_gpt_contex, preguntas_word2vec_contex)

# Inspeccionar los datos
head(preguntasContextuales)

