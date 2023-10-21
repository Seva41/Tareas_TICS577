# Cargar las bibliotecas necesarias
library(openintro)
library(ggplot2)

data(starbucks)

# EJERCICIO 1
# Estandarizar la variable independiente
starbucks$protein_std <- scale(starbucks$protein)

# Definir una función de kernel personalizada
kernel_personalizado <- function(x, y, c) {
  return(exp(-((x - y)^2) / c^2))
}

regresion_kernel <- function(tipo_kernel, nombre_kernel) {
  # Definir el rango de valores de c que se probarán
  c_values <- seq(0.001, 1, length.out = 100)
  
  # Inicializar variables para el seguimiento del mejor modelo
  best_mse <- Inf
  best_c <- NULL
  best_model <- NULL
  
  for (c in c_values) {
    # Crear una matriz de kernel personalizada
    K <- outer(starbucks$protein_std, starbucks$protein_std, tipo_kernel, c)
    
    # Ajustar el modelo de regresión con el kernel personalizado
    model <- lm(carb ~ K, data = starbucks)
    
    # Calcular el error cuadrático medio
    mse <- mean((predict(model, starbucks) - starbucks$carb)^2)
    
    # Actualizar el mejor modelo si se encontró uno con menor MSE
    if (mse < best_mse) {
      best_mse <- mse
      best_c <- c
      best_model <- model
    }
  }
  
  # Resultados
  cat("Mejor valor de c para", nombre_kernel, ":", best_c, "\n")
  cat("Error cuadrático medio mínimo para", nombre_kernel, ":", best_mse, "\n")
  
  # Gráfico de dispersión de los valores observados vs. valores predichos
  plot(starbucks$protein_std, starbucks$carb,
       main = paste("Valores Observados vs. Valores Predichos (", nombre_kernel, ")"),
       xlab = "Protein (Estandarizado)", ylab = "Carb", pch = 20, col = "blue"
  )
  points(starbucks$protein_std, predict(best_model, starbucks), pch = 20, col = "red")
  legend("topright", legend = c("Observados", "Predichos"), col = c("blue", "red"), pch = 20)
}

# Utilizar la función de kernel personalizada
regresion_kernel(kernel_personalizado, "Personalizado")

