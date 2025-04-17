# Cargar los datos de los tres modelos
similitud_glove <- read.csv("/home/velasco/Descargas/ResultadoEmbeddings/Similitud_Semantica_glove.csv")
similitud_gpt <- read.csv("/home/velasco/Descargas/ResultadoEmbeddings/Similitud_Semantica_GPT.csv")
similitud_word2vec <- read.csv("/home/velasco/Descargas/ResultadoEmbeddings/Similitud_Semantica_Wor2vec.csv")

# Agregar columna de modelo
similitud_glove$Modelo <- "GloVe"
similitud_gpt$Modelo <- "GPT"
similitud_word2vec$Modelo <- "Word2Vec"

# Combinar los datasets
library(dplyr)
similitud <- bind_rows(similitud_glove, similitud_gpt, similitud_word2vec)

# Inspeccionar los datos
head(similitud)

