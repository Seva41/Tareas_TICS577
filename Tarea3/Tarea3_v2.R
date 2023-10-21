# Cargar las bibliotecas necesarias
library(kernlab) # Para la regresión con kernel
library(openintro)
library(ggplot2)

data(starbucks)

# EJERCICIO 1
# Estandarizar la variable independiente
starbucks$protein_std <- scale(starbucks$protein)

# Función de kernel constante
kernel_constante <- "vanilladot"

regresion_kernel <- function(tipo_kernel, nombre_kernel) {
  # Definir el rango de valores de c que se probarán
  c_values <- seq(0.001, 1, length.out = 100)
  
  # Inicializar variables para el seguimiento del mejor modelo
  best_mse <- Inf
  best_c <- NULL
  best_model <- NULL
  
  for (c in c_values) {
    # Ajustar el modelo de regresión con kernel
    model <- ksvm(carb ~ protein_std, data = starbucks, kernel = tipo_kernel, C = 1)
    
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

# 1) Kernel Constante
regresion_kernel(kernel_constante, "Constante")

# 2) Kernel Lineal
kernel_lineal <- "vanilladot"

regresion_kernel(kernel_lineal, "Lineal")

# 3) Kernel Polinomial
kernel_polinomial <- "vanilladot"

regresion_kernel(kernel_polinomial, "Polinomial Grado 2")

# 4) Kernel Gaussiano
kernel_gaussiano <- "rbfdot"

regresion_kernel(kernel_gaussiano, "Gaussiano")
