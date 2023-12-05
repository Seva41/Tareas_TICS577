# Instalar y cargar paquetes necesarios
#install.packages("e1071")
library(e1071)

# Cargar los datos
train_data <- read.table(file.choose(), header = TRUE, sep=",")

# Estandarizar los datos
train_data_scaled <- scale(train_data[,c('X1', 'X2')])
#test_data_scaled <- scale(test_data[,-ncol(test_data)])

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

# Ejercicio 2 ---------------------------------------------------------------

# Cargar las bibliotecas necesarias
library(kernlab)
library(e1071)

# Definir un rango de valores para lambda (sigma)
sigma_values <- seq(0.1, 2, by = 0.1)

# Estructura para almacenar los resultados
results <- data.frame(sigma = sigma_values, accuracy = numeric(length(sigma_values)), 
                      recall = numeric(length(sigma_values)), precision = numeric(length(sigma_values)))

# Bucle sobre los valores de sigma
for (i in seq_along(sigma_values)) {
  sigma <- sigma_values[i]
  
  # Kernel PCA
  kpca_model <- kpca(~., data = train_data[,c('X1', 'X2')], kernel = "rbfdot", kpar = list(sigma = sigma))
  train_data_pca <- as.matrix(predict(kpca_model, train_data[,c('X1', 'X2')]))
  train_data_pca_combined <- data.frame(train_data_pca, ytrain = train_data$ytrain)
  
  # Entrenar el modelo SVM
  svm_model <- svm(ytrain ~ ., data = train_data_pca_combined)
  
  # Preparar los datos de prueba
  test_data_pca <- as.matrix(predict(kpca_model, test_data[,c('X1', 'X2')]))
  test_data_pca_combined <- data.frame(test_data_pca, ytest = test_data$ytest)
  
  # Evaluar el modelo
  predictions <- predict(svm_model, test_data_pca_combined)
  conf_matrix <- table(predictions, test_data$ytest)
  
  # Calcular métricas de desempeño
  accuracy2 <- sum(diag(conf_matrix)) / sum(conf_matrix)
  recall2 <- conf_matrix[2,2] / sum(conf_matrix[2,])
  precision2 <- conf_matrix[2,2] / sum(conf_matrix[,2])
  
  # Almacenar los resultados
  results[i, c("accuracy", "recall", "precision")] <- c(accuracy2, recall2, precision2)
}

# Encontrar el mejor sigma
best_result <- results[which.max(results$accuracy2), ]
print(best_result)

