# Instalar y cargar paquetes necesarios
install.packages("e1071")
library(e1071)

# Cargar los datos
train_data <- read.table(file.choose(), header = TRUE, sep=",")

# Estandarizar los datos
train_data_scaled <- scale(train_data[,c('X1', 'X2')])
test_data_scaled <- scale(test_data[,-ncol(test_data)])

# Asegúrate de que los datos están cargados y escalados correctamente
# train_data_scaled y train_data

# Verificar la estructura y los primeros registros de train_data
str(train_data)
head(train_data)

# Verificar la estructura y los primeros registros de train_data_scaled
str(train_data_scaled)
head(train_data_scaled)


# Crear un nuevo data frame combinando train_data_scaled y la columna Y
train_data_combined <- data.frame(train_data_scaled, Y = train_data$ytrain)


# Implementar clasificador SVM con una fórmula explícita
tune_result <- tune(svm, Y ~ ., data = train_data_combined,
                    kernel = "linear",
                    ranges = list(cost = 10^(-1:3)),
                    cross = 10)

best_model <- tune_result$best.model

# Cargar y preparar los datos de prueba
test_data <- read.table(file.choose(), header = TRUE, sep = ",")

# Verificar la estructura de test_data
str(test_data)
head(test_data)


test_data_scaled <- scale(test_data[,c('X1', 'X2')])
# Combinar la columna ytest con los datos escalados de prueba
test_data_combined <- data.frame(test_data_scaled, Y = test_data$ytest)

# Calcular métricas de desempeño en el conjunto de prueba
predictions <- predict(best_model, test_data_combined)
conf_matrix <- table(predictions, test_data$ytest)

accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
recall <- conf_matrix[2,2] / sum(conf_matrix[2,])
precision <- conf_matrix[2,2] / sum(conf_matrix[,2])

# Imprimir los resultados
print(list(Accuracy = accuracy, Recall = recall, Precision = precision))


# Instalar y cargar paquetes necesarios
install.packages("kernlab")
library(kernlab)

# Ejercicio 2 ---------------------------------------------------------------


# Cargar los datos
#train_data <- read.table("/mnt/data/simTrain.txt", header = TRUE)
#test_data <- read.table("/mnt/data/simTest.txt", header = TRUE)

# Función para calcular la matriz de kernel Gaussiano
gaussian_kernel <- function(X1, X2, sigma) {
  size1 <- nrow(X1)
  size2 <- nrow(X2)
  matrix <- matrix(0, nrow = size1, ncol = size2)
  for (i in 1:size1) {
    for (j in 1:size2) {
      diff <- X1[i,] - X2[j,]
      matrix[i,j] <- exp(-sum(diff^2) / (2 * sigma^2))
    }
  }
  return(matrix)
}

# Calcular la matriz de kernel para train_data
sigma <- 1 # Este valor puede ser ajustado
K <- gaussian_kernel(train_data[,-ncol(train_data)], train_data[,-ncol(train_data)], sigma)

# Kernel PCA
K_centered <- kcca(K, type = "kernelPCA")
K_transformed <- as.matrix(K_centered@xmatrix)

# Proyección de los datos de entrenamiento en el espacio de PCA
train_data_pca <- K_transformed[, 1:2] # Seleccionando las primeras 2 componentes


# Instalar y cargar paquete SVM
install.packages("e1071")
library(e1071)

# Implementar SVM con los datos de PCA
tune_result <- tune(svm, train.x = train_data_pca, train.y = train_data$Y,
                    kernel = "linear",
                    ranges = list(cost = 10^(-1:3)),
                    cross = 10)

best_model <- tune_result$best.model

# Transformar los datos de prueba usando el mismo Kernel PCA
test_data_pca <- predict(K_centered, test_data[,-ncol(test_data)])

# Calcular métricas de desempeño en el conjunto de prueba
predictions <- predict(best_model, test_data_pca)
conf_matrix <- table(predictions, test_data$Y)

accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
recall <- conf_matrix[2,2] / sum(conf_matrix[2,])
precision <- conf_matrix[2,2] / sum(conf_matrix[,2])

# Imprimir los resultados
print(list(Accuracy = accuracy, Recall = recall, Precision = precision))