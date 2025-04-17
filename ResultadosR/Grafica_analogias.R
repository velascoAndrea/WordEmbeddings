library(ggplot2)

ggplot(resumen_analogias, aes(x = Modelo, y = Porcentaje_Correcto, fill = Modelo)) +
  geom_bar(stat = "identity", color = "black") +
  labs(
    title = "Porcentaje de Analogías Correctas por Modelo",
    x = "Modelo",
    y = "Porcentaje Correcto"
  ) +
  theme_minimal()


