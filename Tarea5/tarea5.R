# install.packages("e1071")
library(e1071)

# Ej 1 - Punto 1 ---------------------------------------------------------------

# Cargar los datos
train_data <- read.table(file.choose(), header = TRUE, sep = ",")

# Estandarizar los datos
train_data_scaled <- scale(train_data[, c("X1", "X2")])

str(train_data)
head(train_data)

str(train_data_scaled)
head(train_data_scaled)

# Crea un nuevo data frame combinando train_data_scaled y la columna Y
train_data_combined <- data.frame(train_data_scaled, Y = train_data$ytrain)

# Clasificador SVM
tune_result <- tune(svm, Y ~ .,
  data = train_data_combined,
  kernel = "linear",
  ranges = list(cost = 10^(-1:3)),
  cross = 10
)

best_model <- tune_result$best.model

# Datos de prueba
test_data <- read.table(file.choose(), header = TRUE, sep = ",")
str(test_data)
head(test_data)

test_data_scaled <- scale(test_data[, c("X1", "X2")])
# Combinar la columna ytest con los datos escalados de prueba
test_data_combined <- data.frame(test_data_scaled, Y = test_data$ytest)

# Métricas de desempeño en el conjunto de prueba
predictions <- predict(best_model, test_data_combined)
conf_matrix <- table(predictions, test_data$ytest)

accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])

print(list(Accuracy = accuracy, Recall = recall, Precision = precision))

# Ej 1 - Punto 2 ---------------------------------------------------------------

library(kernlab)
library(e1071)

# Rango de valores para lambda (sigma)
sigma_values <- seq(0.1, 2, by = 0.1)


results <- data.frame(
  sigma = sigma_values, accuracy = numeric(length(sigma_values)),
  recall = numeric(length(sigma_values)), precision = numeric(length(sigma_values))
)

# Bucle sobre los valores de sigma
for (i in seq_along(sigma_values)) {
  sigma <- sigma_values[i]

  # Kernel PCA
  kpca_model <- kpca(~., data = train_data[, c("X1", "X2")], kernel = "rbfdot", kpar = list(sigma = sigma))
  train_data_pca <- as.matrix(predict(kpca_model, train_data[, c("X1", "X2")]))
  train_data_pca_combined <- data.frame(train_data_pca, ytrain = train_data$ytrain)

  # Entrenar el modelo SVM
  svm_model <- svm(ytrain ~ ., data = train_data_pca_combined)

  test_data_pca <- as.matrix(predict(kpca_model, test_data[, c("X1", "X2")]))
  test_data_pca_combined <- data.frame(test_data_pca, ytest = test_data$ytest)

  # Evaluar
  predictions <- predict(svm_model, test_data_pca_combined)
  conf_matrix <- table(predictions, test_data$ytest)

  # Métricas de desempeño
  accuracy2 <- sum(diag(conf_matrix)) / sum(conf_matrix)
  recall2 <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
  precision2 <- conf_matrix[2, 2] / sum(conf_matrix[, 2])

  results[i, c("accuracy", "recall", "precision")] <- c(accuracy2, recall2, precision2)
}

# Encontrar el mejor sigma
best_result <- results[which.max(results$accuracy2), ]
print(best_result)

# Ej 2 - Punto 1 ---------------------------------------------------------------
# install.packages("mvtnorm")

library(openintro)
library(mvtnorm) # Para la estandarización multivariante

# Cargar datos
data(starbucks)
str(starbucks)

bakery_data <- subset(starbucks, type == "bakery", select = c("calories", "fat", "carb", "fiber", "protein"))
other_data <- subset(starbucks, type != "bakery", select = c("calories", "fat", "carb", "fiber", "protein"))

# Estandarizacion
bakery_data_scaled <- scale(bakery_data)
other_data_scaled <- scale(other_data)

# Función MMD
mmd_statistic <- function(x, y, sigma) {
  n <- nrow(x)
  m <- nrow(y)

  # Matrices de kernel
  Kxx <- exp(-as.matrix(dist(x))^2 / (2 * sigma^2))
  Kyy <- exp(-as.matrix(dist(y))^2 / (2 * sigma^2))

  # Matriz de distancia cruzada entre 'x' e 'y'
  xy <- rbind(x, y)
  dist_xy <- as.matrix(dist(xy))
  Kxy <- exp(-dist_xy[1:n, (n + 1):(n + m)]^2 / (2 * sigma^2))

  # Calcular el estadístico MMD
  mmd_square <- sum(Kxx) / (n^2) + sum(Kyy) / (m^2) - 2 * sum(Kxy) / (n * m)
  return(sqrt(mmd_square))
}

# Calcular el estadístico MMD observado
observed_mmd <- mmd_statistic(bakery_data_scaled, other_data_scaled, sigma = 0.5)

# Wild-Bootstrap
set.seed(123)
bootstrap_mmds <- replicate(5000, {
  x_bootstrap <- bakery_data_scaled * sample(c(-1, 1), nrow(bakery_data_scaled), replace = TRUE)
  mmd_statistic(x_bootstrap, other_data_scaled, sigma = 0.5)
})

# Histograma y P-value
hist(bootstrap_mmds, main = "Distribución de MMD Bootstrap", xlab = "Estadístico MMD", col = "blue", border = "black", xlim = c(min(bootstrap_mmds), max(bootstrap_mmds, observed_mmd)))
abline(v = observed_mmd, col = "red", lwd = 2)
legend("topright", legend = c("MMD Observado"), col = c("red"), lwd = 2)

# Calcular el P-value
p_value <- mean(bootstrap_mmds >= observed_mmd)

# Resultados
print(paste("MMD Observado:", observed_mmd))
print(paste("P-Value:", p_value))
