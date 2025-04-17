# Cargar datos para cada modelo
preguntas_glove <- read.csv("/home/velasco/Descargas/ResultadoEmbeddings/PreguntasFactuales_Glove.csv")
preguntas_gpt <- read.csv("/home/velasco/Descargas/ResultadoEmbeddings/PreguntasFactuales_GPT.csv")
preguntas_word2vec <- read.csv("/home/velasco/Descargas/ResultadoEmbeddings/PreguntasFactuales_Word2vec.csv")

# Agregar columna para identificar el modelo
preguntas_glove$Modelo <- "GloVe"
preguntas_gpt$Modelo <- "GPT"
preguntas_word2vec$Modelo <- "Word2Vec"

# Combinar los datos en un solo dataframe
library(dplyr)
preguntas <- bind_rows(preguntas_glove, preguntas_gpt, preguntas_word2vec)

# Inspeccionar los datos
head(preguntas)

