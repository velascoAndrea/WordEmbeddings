library(ggplot2)

# Histograma de similitudes por modelo
ggplot(similitud, aes(x = Resultado, fill = Modelo)) +
  geom_histogram(binwidth = 0.02, position = "dodge", alpha = 0.7, color = "black") +
  labs(
    title = "Distribución de Similitud Semántica por Modelo",
    x = "Similitud Semántica (Coseno)",
    y = "Frecuencia"
  ) +
  theme_minimal()

# Boxplot de resultados por modelo
ggplot(similitud, aes(x = Modelo, y = Resultado, fill = Modelo)) +
  geom_boxplot() +
  labs(
    title = "Boxplot de Similitud Semántica por Modelo",
    x = "Modelo",
    y = "Similitud Semántica"
  ) +
  theme_minimal()

# ANOVA para comparar modelos
aov_similitud <- aov(Resultado ~ Modelo, data = similitud)
summary(aov_similitud)

# Prueba post-hoc de Tukey
TukeyHSD(aov_similitud)


