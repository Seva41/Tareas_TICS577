# Limpieza del entorno
rm(list = ls())

# Carga de datos desde el archivo CSV
datos = read.csv(file.choose(), header = TRUE)

# Creación de una variable lógica para identificar la categoría 'Breakfast'
clase = datos$Category == "Breakfast"

# Conversión de la clase a valores numéricos (1 para 'Breakfast', -1 para otros)
datos$clase_num = 2 * clase - 1

# Identificación de índices para la categoría 'Breakfast'
indices_desayuno = which(datos$Category == "Breakfast")

# Separación de los datos en dos subconjuntos: Desayuno y Otros
datos_desayuno = datos[indices_desayuno, ]
datos_otros = datos[-indices_desayuno, ]

# Creación de particiones k-fold para cada subconjunto
kfold_desayuno = sample(1:4, nrow(datos_desayuno), replace = TRUE)
kfold_otros = sample(1:4, nrow(datos_otros), replace = TRUE)

# Asignación de las particiones k-fold al conjunto de datos principal
datos$Kfold <- rep(NA, nrow(datos))
datos$Kfold[indices_desayuno] <- kfold_desayuno
datos$Kfold[-indices_desayuno] <- kfold_otros

# Selección de columnas relevantes para el análisis
columnas = setdiff(names(datos), c("Category", "Item", "Serving.Size", "Kfold", "clase_num"))

# Carga de librerías necesarias para el modelado
library("caret")
library("kernlab")

# Definición de la secuencia de valores de C para la optimización
grilla_C = seq(1, 1000, 1)

# Inicialización de vectores para almacenar métricas de rendimiento
acuracias = numeric(length(grilla_C))
f1_medias = numeric(length(grilla_C))

for (j in grilla_C) {
  print(j) # Imprime el valor actual de C en la optimización
  acuracias_temp = numeric(4)
  f1_temp = numeric(4)
  
  for (i in 1:4) {
    # Creación de conjuntos de entrenamiento y prueba
    indices_prueba = datos$Kfold == i
    datos_prueba = datos[indices_prueba, ]
    datos_entrenamiento = datos[!indices_prueba, ]
    
    # Estandarización de los datos (excluyendo la columna objetivo y Kfold)
    escalador = preProcess(datos_entrenamiento[, columnas], method = c("center", "scale"))
    datos_entrenamiento_escalados = predict(escalador, datos_entrenamiento[, columnas])
    datos_prueba_escalados = predict(escalador, datos_prueba[, columnas])
    
    # Entrenamiento del modelo SVM
    modelo_svm = ksvm(y = as.factor(datos_entrenamiento$clase_num), 
                      x = as.matrix(datos_entrenamiento_escalados), 
                      kernel = 'vanilladot', scaled = FALSE, C = j)
    
    # Predicción y evaluación del modelo
    predicciones = predict(modelo_svm, datos_prueba_escalados)
    tabla_confusion = table(datos_prueba$clase_num, predicciones)
    
    # Cálculo de la precisión y el puntaje F1
    acuracia = (tabla_confusion[1, 1] + tabla_confusion[2, 2]) / sum(tabla_confusion)
    precision = tabla_confusion[2, 2] / (tabla_confusion[2, 1] + tabla_confusion[2, 2])
    recall = tabla_confusion[2, 2] / (tabla_confusion[1, 2] + tabla_confusion[2, 2])
    f1 = 2 * (recall * precision) / (recall + precision)
    
    # Almacenamiento de métricas
    acuracias_temp[i] = acuracia
    f1_temp[i] = f1
  }
  
  # Cálculo de la media de las métricas para cada valor de C
  acuracias[j] = mean(acuracias_temp)
  f1_medias[j] = mean(f1_temp)
}

# Creación de un dataframe para visualizar los resultados
resultados = data.frame("C" = grilla_C, "Acuracia" = acuracias, "F1" = f1_medias)

# Identificación del mejor valor de C para cada métrica
mejor_C_acuracia = which.max(resultados$Acuracia)
mejor_C_f1 = which.max(resultados$F1)

# Visualización de los mejores resultados
resultados[mejor_C_acuracia, ]
resultados[mejor_C_f1, ]

# -------------------------------------------------------------------------------------------

# Definición de la grilla de parámetros C y l para la optimización
grilla_C_b = seq(1, 10, 1)
grilla_l = seq(1, 100, 1)
combinaciones = expand.grid(C = grilla_C_b, l = grilla_l)
num_combinaciones = length(parm[,1])

# Inicialización de vectores para almacenar métricas de rendimiento
acuracias_b = numeric(num_combinaciones)
f1_medias_b = numeric(num_combinaciones)
Kfun = function(x,y){
  norm2 = sum((x-y)^2)
  exp(-norm2/l^2)
}
class(Kfun)='kernel'

for (j in grilla_C) {
  print(j) # Imprime el valor actual de C en la optimización
  acuracias_temp = numeric(4)
  f1_temp = numeric(4)
  c<-combinaciones[j,1]
  l<-combinaciones[j,2]
  
  for (i in 1:4) {
    # Creación de conjuntos de entrenamiento y prueba
    indices_prueba = datos$Kfold == i
    datos_prueba = datos[indices_prueba, ]
    datos_entrenamiento = datos[!indices_prueba, ]
    
    # Estandarización de los datos (excluyendo la columna objetivo y Kfold)
    escalador = preProcess(datos_entrenamiento[, columnas], method = c("center", "scale"))
    datos_entrenamiento_escalados = predict(escalador, datos_entrenamiento[, columnas])
    datos_prueba_escalados = predict(escalador, datos_prueba[, columnas])
    
    # Entrenamiento del modelo SVM
    modelo_svm = ksvm(y = as.factor(datos_entrenamiento$clase_num), 
                  x = as.matrix(datos_entrenamiento_escalados),
                  kernel = Kfun, l=l, scaled = FALSE, C = c)
    
    
    # Predicción y evaluación del modelo
    predicciones = predict(modelo_svm, datos_prueba_escalados)
    tabla_confusion = table(datos_prueba$clase_num, predicciones)
    
    # Cálculo de la precisión y el puntaje F1
    acuracia = (tabla_confusion[1, 1] + tabla_confusion[2, 2]) / sum(tabla_confusion)
    precision = tabla_confusion[2, 2] / (tabla_confusion[2, 1] + tabla_confusion[2, 2])
    recall = tabla_confusion[2, 2] / (tabla_confusion[1, 2] + tabla_confusion[2, 2])
    f1 = 2 * (recall * precision) / (recall + precision)
    
    # Almacenamiento de métricas
    acuracias_temp[i] = acuracia
    f1_temp[i] = f1
  }
  
  # Cálculo de la media de las métricas para cada valor de C
  acuracias_b[j] = mean(acuracias_temp)
  f1_medias_b[j] = mean(f1_temp)
}

# Creación de un dataframe para visualizar los resultados
resultados_b = data.frame(combinaciones, "Acuracia" = acuracias_b, "F1" = f1_medias_b)

# Identificación del mejor conjunto de parámetros para cada métrica
mejor_combinacion_acuracia = which.max(resultados_b$Acuracia)
mejor_combinacion_f1 = which.max(resultados_b$F1)

# Visualización de los mejores resultados
resultados_b[mejor_combinacion_acuracia, ]
resultados_b[mejor_combinacion_f1, ]







# Definición de la grilla de parámetros para la optimización
grilla_C_lp = seq(1, 5, 1)
grilla_ls = seq(1, 5, 1)
grilla_sigma = seq(1, 5, 1)
grilla_p = seq(1, 5, 1)
combinaciones_lp = expand.grid(C = grilla_C_lp, ls = grilla_ls, sigma = grilla_sigma, p = grilla_p)
num_combinaciones_lp = nrow(combinaciones_lp)

# Inicialización de vectores para almacenar métricas de rendimiento
acuracias_lp = numeric(num_combinaciones_lp)
f1_medias_lp = numeric(num_combinaciones_lp)

# Función kernel personalizada para 'locally periodic'
Kfun_lp = function(x, y) {
  exp(-2 * (sin(pi * sum(abs(x - y)) / p)^2) / ls^2) * exp(-sum((x - y)^2) / sigma^2)
}
class(Kfun_lp) = 'kernel'

for (k in 1:num_combinaciones_lp) {
  c_actual_lp = combinaciones_lp$C[k]
  ls_actual = combinaciones_lp$ls[k]
  sigma_actual = combinaciones_lp$sigma[k]
  p_actual = combinaciones_lp$p[k]
  acuracias_temp_lp = numeric(4)
  f1_temp_lp = numeric(4)
  
  for (i in 1:4) {
    # Creación de conjuntos de entrenamiento y prueba (repetir como en los casos anteriores)
    
    # Entrenamiento del modelo SVM con kernel personalizado 'locally periodic'
    modelo_svm_lp = ksvm(y = as.factor(datos_entrenamiento_escalados[, target]),
                         x = as.matrix(datos_entrenamiento_escalados[, columnas_modelo]), 
                         kernel = Kfun_lp, ls = ls_actual, sigma = sigma_actual, p = p_actual, 
                         scaled = FALSE, C = c_actual_lp)
    
    # Predicción y evaluación del modelo (repetir como en los casos anteriores)
  }
  
  # Cálculo de la media de las métricas para cada combinación de parámetros
  acuracias_lp[k] = mean(acuracias_temp_lp)
  f1_medias_lp[k] = mean(f1_temp_lp)
}

# Creación de un dataframe para visualizar los resultados
resultados_lp = data.frame(combinaciones_lp, "Acuracia" = acuracias_lp, "F1" = f1_medias_lp)

# Identificación del mejor conjunto de parámetros para cada métrica
mejor_combinacion_acuracia_lp = which.max(resultados_lp$Acuracia)
mejor_combinacion_f1_lp = which.max(resultados_lp$F1)

# Visualización de los mejores resultados
resultados_lp[mejor_combinacion_acuracia_lp, ]
resultados_lp[mejor_combinacion_f1_lp, ]