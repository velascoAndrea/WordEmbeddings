library(ggplot2)

ggplot(resumen_f1, aes(x = Modelo, y = Porcentaje_Correctas, fill = Modelo)) +
  geom_bar(stat = "identity", color = "black") +
  labs(
    title = "Porcentaje de Respuestas Correctas (F1) por Modelo",
    x = "Modelo",
    y = "Porcentaje Correctas"
  ) +
  theme_minimal()

ggplot(resumen_likert, aes(x = Modelo, y = Media_Likert, fill = Modelo)) +
  geom_bar(stat = "identity", color = "black") +
  labs(
    title = "Media de Escala de Likert por Modelo",
    x = "Modelo",
    y = "Media Likert"
  ) +
  theme_minimal()

ggplot(preguntas, aes(x = Modelo, y = Escala.de.likert, fill = Modelo)) +
  geom_boxplot() +
  labs(
    title = "Distribución de Puntuación Likert por Modelo",
    x = "Modelo",
    y = "Puntuación Likert"
  ) +
  theme_minimal()


