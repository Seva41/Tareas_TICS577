# Cargar las bibliotecas necesarias
# install.packages("kernlab")
library(kernlab) # Para la regresión con kernel

# Se cargan los datos (reemplazar la ruta por la ruta en su computador)
data <- read.table("C:/Users/sebad/OneDrive - Universidad Adolfo Ibanez/Code/Métodos basados en Kernel/Tareas/Tarea3/datos.txt", header = TRUE)

# EJERCICIO 1
# Variables
data$x <- as.numeric(data$x)
protein <- scale(data$protein) # Estandarizar variable independiente
carb <- data$carb

# a) Justificación del kernel constante

# Propiedades requeridas para que sea un kernel válido:
# 1. Simetría: Kc(x, y) = c es simétrico ya que Kc(x, y) = Kc(y, x) para todos x e y.
# 2. Semidefinición positiva: La matriz de Gram K debe ser semidefinida positiva.

# Demostración de semidefinición positiva:
# Para cualquier conjunto de puntos {x1, x2, ..., xn}, la matriz de Gram K con K_ij = Kc(xi, xj) será una matriz de unos multiplicada por c.
# Esta matriz es semidefinida positiva siempre que c sea mayor que o igual a cero.

# Se elige c = 1 para el kernel constante
c <- 1

# Se crea una matriz de Gram K con K_ij = Kc(xi, xj)
n <- 10 # Número de puntos
K <- matrix(c, n, n)

# Verifica si K es semidefinida positiva
is_positive_semidefinite <- all(eigen(K)$values >= 0) # True si todos los valores propios son mayores que o iguales a cero

if (is_positive_semidefinite) {
    cat("El kernel constante Kc(x, y) = c es un kernel válido debido a su simetría y semidefinición positiva.\n")
} else {
    cat("El kernel constante Kc(x, y) = c no es un kernel válido debido a su falta de semidefinición positiva.\n")
}


# b) Ajuste del modelo de regresión con kernel constante
# Define una función de error cuadrático medio
mse <- function(predicted, actual) {
    mean((predicted - actual)^2)
}

# Encuentra el valor óptimo de c que minimiza el error cuadrático medio
c_values <- seq(0.01, 1, length.out = 100) # Valores candidatos de c
mse_values <- numeric(length(c_values)) # Valores de error cuadrático medio

# NO FUNCIONA
for (i in 1:length(c_values)) {
    tryCatch(
        {
            # Ajusta el modelo de regresión con kernel constante
            kernel_model <- ksvm(as.matrix(protein), as.vector(carb), kernel = "vanilladot", kpar = list(sigma = sqrt(c_values[i])))
            predicted_carb <- predict(kernel_model, as.matrix(protein)) # Valores predichos de carb
            mse_values[i] <- mse(predicted_carb, carb) # Se guarda el error cuadrático medio
        },
        error = function(e) {
            cat("Error en el ajuste del modelo con c =", c_values[i], "\n")
        }
    )
}


optimal_c <- c_values[which.min(mse_values)]
cat("El valor óptimo de c para el kernel constante es:", optimal_c, "\n")

# c) Graficar los valores ajustados para el kernel constante
kernel_model <- ksvm(as.matrix(protein), as.vector(carb), kernel = "vanilladot", kpar = list(sigma = sqrt(optimal_c)))
predicted_carb <- predict(kernel_model, as.matrix(protein))
plot(protein, carb, main = "Kernel Constante", xlab = "Proteína (Estandarizada)", ylab = "Carbohidratos")
lines(protein, predicted_carb, col = "red")

# d) Repetir para otros kernels
# Define los kernels lineal, polinomial y gaussiano
kernels <- list("Lineal" = "vanilladot", "Polinomial" = "polydot", "Gaussiano" = "rbfdot")
kernel_parameters <- list("Lineal" = c(0.01, 1), "Polinomial" = c(0.01, 2), "Gaussiano" = c(0.01, 1))

for (kernel_name in names(kernels)) {
    cat("\nKernel:", kernel_name, "\n")
    kernel_model <- ksvm(as.matrix(protein), as.vector(carb), kernel = kernels[[kernel_name]], kpar = list(sigma = sqrt(kernel_parameters[[kernel_name]][1])), degree = kernel_parameters[[kernel_name]][2])
    predicted_carb <- predict(kernel_model, as.matrix(protein))
    mse_value <- mse(predicted_carb, carb)
    cat("MSE para el kernel", kernel_name, "es:", mse_value, "\n")
    plot(protein, carb, main = paste("Kernel", kernel_name), xlab = "Proteína (Estandarizada)", ylab = "Carbohidratos")
    lines(protein, predicted_carb, col = "red")
}

# EJERCICIO 2
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
