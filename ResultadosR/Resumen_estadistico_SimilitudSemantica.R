# Resumen estadístico por modelo
resumen_similitud <- similitud %>%
  group_by(Modelo) %>%
  summarise(
    Media = mean(Resultado),
    Mediana = median(Resultado),
    Desviacion_Estandar = sd(Resultado),
    Minimo = min(Resultado),
    Maximo = max(Resultado)
  )

# Mostrar el resumen
print(resumen_similitud)

