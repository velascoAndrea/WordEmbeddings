# Resumen estadístico para F1
resumen_f1_analogicas <- preguntasAnalogicas %>%
  group_by(Modelo) %>%
  summarise(
    Total_Preguntas = n(),
    Correctas = sum(F1.correcta.incorrecta.),
    Porcentaje_Correctas = mean(F1.correcta.incorrecta.) * 100
  )

# Mostrar el resumen de F1
print(resumen_f1_analogicas)

# Resumen estadístico para Escala de Likert
resumen_likert_analogicas <- preguntasAnalogicas %>%
  group_by(Modelo) %>%
  summarise(
    Media_Likert = mean(Escala.de.likert),
    Desviacion_Likert = sd(Escala.de.likert)
  )

# Mostrar el resumen de Likert
print(resumen_likert_analogicas)

