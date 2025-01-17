# Resumen estadístico por modelo
library(dplyr)

resumen_analogias <- analogias %>%
  group_by(Modelo) %>%
  summarise(
    Total_Analogias = n(),
    Correctas = sum(Es_correcto),
    Porcentaje_Correcto = mean(Es_correcto) * 100
  )

# Mostrar el resumen
print(resumen_analogias)

