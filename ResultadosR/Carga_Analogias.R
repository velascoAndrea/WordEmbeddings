# Cargar datos para los tres modelos
analogias_glove <- read.csv("/home/velasco/Descargas/ResultadoEmbeddings/Analogias_Glove.csv")
analogias_gpt <- read.csv("/home/velasco/Descargas/ResultadoEmbeddings/Analogias_GPT.csv")
analogias_word2vec <- read.csv("/home/velasco/Descargas/ResultadoEmbeddings/Analogias_Wor2vec.csv")

# Añadir columna de modelo
analogias_glove$Modelo <- "GloVe"
analogias_gpt$Modelo <- "GPT"
analogias_word2vec$Modelo <- "Word2Vec"

# Combinar los tres datasets
library(dplyr)
analogias <- bind_rows(analogias_glove, analogias_gpt, analogias_word2vec)

# Inspeccionar los datos combinados
head(analogias)

