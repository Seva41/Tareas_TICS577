#install.packages("caret")
#install.packages("kernlab")
#install.packages("doParallel")
rm(list=ls())
library(caret)
library(kernlab)
library(doParallel)

# Leer los datos
df <- read.csv(file.choose(), header = TRUE)
df$clase <- ifelse(df$Category == 'Breakfast', 1, 0)  # Variable binaria para clase

# Preprocesamiento: Estandarización de las variables numéricas
columnas <- c('Calories', 'Total.Fat')
scaler <- preProcess(df[, columnas], method = c('center', 'scale'))
df_scaled <- predict(scaler, df)
df_scaled$clase <- as.factor(df$clase)  # Convertir a factor

# Definir el kernel locally periodic
locally_periodic_kernel <- function(sigma, p, x, y) {
  # Calcula la matriz de distancias
  distance_matrix <- as.matrix(dist(rbind(x, y)))
  # Calcula el kernel RBF part
  rbf_part <- exp(-1 / (2 * sigma^2) * distance_matrix^2)
  # Calcula el kernel periodic part
  periodic_part <- exp(-2 / p^2 * sin(pi * distance_matrix / p)^2)
  # Combina ambos kernels
  kernel_matrix <- rbf_part * periodic_part
  # Devuelve la parte de la matriz que corresponde a las combinaciones de x y y
  kernel_matrix[1:nrow(x), (nrow(x) + 1):(nrow(x) + nrow(y))]
}

# Especificar la cuadrícula de hiperparámetros
sigma_grid <- seq(0.1, 2, length.out = 5)  # Ejemplo con 5 valores para ilustrar
p_grid <- seq(0.1, 2, length.out = 5)
C_grid <- seq(0.01, 5, length.out = 5)

# Configuración para la paralelización
registerDoParallel(cores = detectCores() - 1)  # Registra el backend paralelo, reserva un núcleo

# Configuración de la búsqueda en cuadrícula para un kernel lineal
tuneGrid_1 <- expand.grid(C = seq(0.01, 5, length.out = 30))
tuneGrid_2 <- expand.grid(C = seq(0.01, 5, length.out = 30),
                        sigma = seq(0.1, 2, length.out = 30))  # sigma es el parámetro de escala del kernel

# Configuración del control de entrenamiento
trainControl <- trainControl(method = "cv", number = 4, 
                             summaryFunction = defaultSummary, 
                             savePredictions = "final",
                             allowParallel = TRUE)

# Entrenamiento del modelo SVM con kernel lineal (vanilla)
set.seed(123)  # Para reproducibilidad
svm_model_vanilla <- train(clase ~ Calories + Total.Fat, data = df_scaled, 
                   method = "svmLinear",
                   trControl = trainControl,
                   tuneGrid = tuneGrid_1)

# SVM con kernel squared exponential
svm_model_sqexp <- train(clase ~ Calories + Total.Fat, data = df_scaled, 
                   method = "svmRadial",
                   trControl = trainControl,
                   tuneGrid = tuneGrid_2)

#SVM con kernel locally periodic
svm_model_locally <- foreach(sigma = sigma_grid, .combine = rbind, .packages = 'kernlab') %:%
  foreach(p = p_grid, .combine = rbind) %:%
  foreach(C = C_grid, .combine = rbind) %dopar% {
    # Aquí definimos 'clase' y 'df_scaled' para que estén disponibles en el entorno paralelo
    clase <- df_scaled$clase
    x <- df_scaled[, columnas, drop = FALSE]
    y <- df_scaled[, columnas, drop = FALSE]
    # Precalcula la matriz del kernel
    kernel_matrix <- locally_periodic_kernel(sigma, p, x, x)
    # Entrenamiento del modelo SVM
    model <- ksvm(y = clase, x = as.kernelMatrix(kernel_matrix), kernel = "matrix", C = C)
    # Predicción y cálculo de la matriz de confusión
    pred <- predict(model, newdata = as.kernelMatrix(locally_periodic_kernel(sigma, p, x, y)))
    cm <- confusionMatrix(pred, clase)
    # Devuelve los resultados
    c(sigma = sigma, p = p, C = C, Accuracy = cm$overall['Accuracy'], Kappa = cm$overall['Kappa'])
  }

# Detener el backend paralelo después del uso
stopImplicitCluster()

# Resultados
print(svm_model_vanilla)
print(svm_model_sqexp)
svm_model_locally <- as.data.frame(svm_model_locally)
print(svm_model_locally)

# Graficamos los resultados
ggplot(svm_model_vanilla$results, aes(x = C, y = Accuracy)) + 
  geom_line() + 
  labs(title = "Accuracy vs C - Vanilla", y = "Accuracy", x = "C")

ggplot(svm_model_sqexp$results, aes(x = C, y = Accuracy)) +
  geom_line() +
  labs(title = "Accuracy vs C - Squared Exponential", y = "Accuracy", x = "C")
