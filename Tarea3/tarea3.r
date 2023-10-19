# Cargar las bibliotecas necesarias
# install.packages("kernlab")
library(kernlab) # Para la regresión con kernel
library(openintro)
library(e1071)
library(ggplot2)

data(starbucks)

# EJERCICIO 1
# Estandarizar la variable independiente
starbucks$protein_std <- scale(starbucks$protein)


# Función de kernel constante
kernel_constante <- function(x, y, c) {
  return(c)
}

regresion_kernels <- function(tipo_kernel, a) {
  tipo <- NULL

  if (a == 1) {
    tipo <- "Constante"
  }
  if (a == 2) {
    tipo <- "Lineal"
  }
  if (a == 3) {
    tipo <- "Polinomial Grado 2"
  }
  if (a == 4) {
    tipo <- "Gaussiano"
  }

  # Definir el rango de valores de c que se probarán
  c_values <- seq(0.001, 1, length.out = 100)

  # Inicializar variables para el seguimiento del mejor modelo
  best_mse <- Inf
  best_c <- NULL
  best_lambda <- NULL
  best_model <- NULL

  # Implementar la regresión con kernel constante y encontrar el mejor valor de c
  for (c in c_values) {
    for (lambda in c_values) {
      # Crear la matriz del kernel constante
      K <- matrix(tipo_kernel(starbucks$protein_std, starbucks$protein_std, c), nrow = nrow(starbucks))

      # Añadir regularización lambda a la matriz del kernel
      K_reg <- K + lambda

      # Ajustar el modelo de regresión con kernel
      model <- lm(carb ~ K_reg, data = starbucks)

      # Calcular el error cuadrático medio
      mse <- mean(model$residuals^2)

      # Actualizar el mejor modelo si se encontró uno con menor MSE
      if (mse < best_mse) {
        best_mse <- mse
        best_c <- c
        best_lambda <- lambda
        best_model <- model
      }
    }
  }

  # Resultados
  cat("Mejor valor de c:", best_c, "\n")
  cat("Mejor valor de lambda:", best_lambda, "\n")
  cat("Error cuadrático medio mínimo:", best_mse, "\n")

  # Predicciones del mejor modelo
  predicted_values <- predict(best_model)

  # Gráfico de dispersión de los valores observados vs. valores predichos
  plot(starbucks$protein_std, starbucks$carb,
    main = paste("Valores Observados vs. Valores Predichos (", tipo, ")"),
    xlab = "Protein (Estandarizado)", ylab = "Carb", pch = 20, col = "blue"
  )
  points(starbucks$protein_std, predicted_values, pch = 20, col = "red")
  legend("topright", legend = c("Observados", "Predichos"), col = c("blue", "red"), pch = 20)
}

regresion_kernels(kernel_constante, 1)

# d) Repetir para otros kernels
# 1) Kernel Lineal
kernel_lineal <- function(x, y, c) {
  return(1 + c^2 * x * y)
}

regresion_kernels(kernel_lineal, 2)


# 2) Kernel Polinomial
kernel_polinomial <- function(x, y, c) {
  return((1 + c * x * y)^2)
}

regresion_kernels(kernel_polinomial, 3)

# 3) Kernel Gaussiano
kernel_gaussiano <- function(x, y, c) {
  return(exp(-(x - y)^2 / c^2))
}

regresion_kernels(kernel_gaussiano, 4)














# EJERCICIO 2 --- NO REVISADO!!!
# a) Verificación del kernel browniano y del kernel J
# Ya hemos demostrado que K(t, s) = σ^2 * min(t, s) y J(x, x') = 1 / (1 - min(x, x')) son kernels válidos en el ejercicio anterior.

# b) Utilizar kernel regression
# Define los kernels Squared-exponential y Browniano
kernel_se <- function(x, x2, sigma) {
  exp(-((x - x2)^2) / (2 * sigma^2))
}

kernel_brownian <- function(x, x2, sigma) {
  sigma^2 * pmin(x, x2)
}

# Parámetros para ajustar el error cuadrático medio
sigma_sq_se <- 0.01
sigma_sq_brownian <- 0.01

# Ajustar modelos de regresión con los kernels
model_se <- ksvm(as.matrix(data$time), as.vector(data$f), kernel = kernel_se, kpar = list(sigma = sqrt(sigma_sq_se)))
model_brownian <- ksvm(as.matrix(data$time), as.vector(data$f), kernel = kernel_brownian, kpar = list(sigma = sqrt(sigma_sq_brownian)))

# Evaluar errores cuadráticos medios en el conjunto de entrenamiento
mse_se <- mse(predict(model_se, as.matrix(data$time)), data$f)
mse_brownian <- mse(predict(model_brownian, as.matrix(data$time)), data$f)

cat("MSE para Squared-exponential kernel:", mse_se, "\n")
cat("MSE para Browniano kernel:", mse_brownian, "\n")

# c) Graficar mapas de calor y funciones asociadas al RKHS (para Squared-exponential y Browniano)
library(ggplot2)

# Crear una secuencia de valores de tiempo para la visualización
time_seq <- seq(min(data$time), max(data$time), length.out = 1000)

# Calcular las matrices de kernel para cada conjunto de datos
K_se <- matrix(0, nrow = length(time_seq), ncol = length(time_seq))
K_brownian <- matrix(0, nrow = length(time_seq), ncol = length(time_seq))

for (i in 1:length(time_seq)) {
  for (j in 1:length(time_seq)) {
    K_se[i, j] <- kernel_se(time_seq[i], time_seq[j], sqrt(sigma_sq_se))
    K_brownian[i, j] <- kernel_brownian(time_seq[i], time_seq[j], sqrt(sigma_sq_brownian))
  }
}

# Graficar los mapas de calor de los kernels
ggplot() +
  geom_tile(data = as.data.frame(K_se), aes(x = time_seq, y = rev(time_seq), fill = K_se)) +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "Mapa de Calor del Kernel Squared-exponential", x = "Tiempo", y = "Tiempo") +
  theme_minimal()

ggplot() +
  geom_tile(data = as.data.frame(K_brownian), aes(x = time_seq, y = rev(time_seq), fill = K_brownian)) +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "Mapa de Calor del Kernel Browniano", x = "Tiempo", y = "Tiempo") +
  theme_minimal()
