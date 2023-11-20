#install.packages("caret")
#install.packages("kernlab")
#install.packages("doParallel")
rm(list=ls())
library(caret)
library(kernlab)
library(doParallel)

# Leer los datos
df <- read.csv(file.choose(), header = TRUE)
# Convertir 'Category' en una variable binaria para clase
df$clase <- as.factor(ifelse(df$Category == 'Breakfast', 1, 0))

# Preprocesamiento: Estandarización de todas las variables numéricas
columnas_numericas <- sapply(df, is.numeric)
scaler <- preProcess(df[, columnas_numericas], method = c('center', 'scale'))
df_scaled <- predict(scaler, df)
df_scaled$clase <- as.factor(ifelse(df$Category == 'Breakfast', 1, 0))

df$Kfold <- sample(1:4, nrow(df), replace = TRUE)
df_scaled$Kfold <- df$Kfold

# Definir el kernel locally periodic
locally_periodic_kernel <- function(sigma, p, x, y) {
  # Asegúrate de que las dimensiones de x y y son correctas
  distance_matrix <- as.matrix(dist(rbind(x, y)))
  # Calcula las partes del kernel
  rbf_part <- exp(-1 / (2 * sigma^2) * distance_matrix^2)
  periodic_part <- exp(-2 / p^2 * sin(pi * distance_matrix / p)^2)
  # Combina ambos kernels
  kernel_matrix <- rbf_part * periodic_part
  # Devuelve solo la parte de la matriz que corresponde a las combinaciones de x y y
  return(kernel_matrix[1:nrow(x), (nrow(x) + 1):(nrow(x) + nrow(y))])
}


# Especificar la cuadrícula de hiperparámetros
sigma_grid <- seq(0.1, 2, length.out = 5)  # Ejemplo con 5 valores para ilustrar
p_grid <- seq(0.1, 2, length.out = 5)
C_grid <- seq(0.01, 5, length.out = 5)

# Configuración para la paralelización
registerDoParallel(cores = detectCores() - 1)  # Registra el backend paralelo, reserva un núcleo

# Definir la cuadrícula de parámetros para cada modelo
tuneGrid_vanilla <- expand.grid(C = seq(0.01, 5, length.out = 30))
tuneGrid_sqexp <- expand.grid(C = seq(0.01, 5, length.out = 30),
                              sigma = seq(0.1, 2, length.out = 30))

# Configuración del control de entrenamiento
trainControl <- trainControl(method = "cv", number = 4, 
                             summaryFunction = defaultSummary, 
                             savePredictions = "final",
                             allowParallel = TRUE)

# Entrenamiento del modelo SVM con kernel lineal (vanilla)
svm_model_vanilla <- train(clase ~ ., data = df_scaled, 
                           method = "svmLinear",
                           trControl = trainControl,
                           tuneGrid = tuneGrid_vanilla)

# Entrenamiento del modelo SVM con kernel squared exponential
svm_model_sqexp <- train(clase ~ ., data = df_scaled, 
                         method = "svmRadial",
                         trControl = trainControl,
                         tuneGrid = tuneGrid_sqexp)

#SVM con kernel locally periodic
results_sml <- foreach(sigma = sigma_grid, .combine = rbind, .packages = 'kernlab') %:%
  foreach(p = p_grid, .combine = rbind) %:%
  foreach(C = C_grid, .combine = rbind) %dopar% {
    # Dividir los datos en conjuntos de entrenamiento y prueba según 'Kfold'
    x_train <- df_scaled[df_scaled$Kfold != 1, columnas_numericas, drop = FALSE]
    x_test <- df_scaled[df_scaled$Kfold == 1, columnas_numericas, drop = FALSE]
    y_train <- df_scaled$clase[df_scaled$Kfold != 1]
    
    # Preparar las matrices del kernel
    kernel_train <- locally_periodic_kernel(sigma, p, x_train, x_train)
    kernel_test <- locally_periodic_kernel(sigma, p, x_train, x_test)
    
    # Entrenar el modelo SVM
    model <- ksvm(y = y_train, x = as.kernelMatrix(kernel_train), kernel = "matrix", C = C)
    
    # Predecir y evaluar en el conjunto de prueba
    pred <- predict(model, newdata = as.kernelMatrix(kernel_test))
    cm <- confusionMatrix(pred, df_scaled$clase[df_scaled$Kfold == 1])
    
    # Devolver los resultados
    list(Accuracy = cm$overall['Accuracy'], Kappa = cm$overall['Kappa'], 
         sigma = sigma, p = p, C = C)
  }

# Detener el backend paralelo después del uso
stopImplicitCluster()

# Resultados
print(svm_model_vanilla)
print(svm_model_sqexp)
svm_model_locally <- as.data.frame(do.call("rbind", results_sml))
print(svm_model_locally)

# Graficamos los resultados
ggplot(svm_model_vanilla$results, aes(x = C, y = Accuracy)) + 
  geom_line() + 
  labs(title = "Accuracy vs C - Vanilla", y = "Accuracy", x = "C")

ggplot(svm_model_sqexp$results, aes(x = C, y = Accuracy)) +
  geom_line() +
  labs(title = "Accuracy vs C - Squared Exponential", y = "Accuracy", x = "C")
