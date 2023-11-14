rm(list=ls())

install.packages("e1071")
install.packages("caret")
library(e1071)
library(caret)


# Cargar los datos
datos <- read.csv(file.choose(),header = TRUE)


# Convertir la columna 'Category' a factor
datos$Category <- as.factor(datos$Category)

# Seleccionar solo las columnas numéricas y añadir 'Category'
covariables <- datos[, sapply(datos, is.numeric)]
datos_final <- cbind(Category = datos$Category, covariables)

# Dividir los datos en conjuntos de entrenamiento y prueba
set.seed(123)
indices <- sample(1:nrow(datos_final), size = 0.8 * nrow(datos_final))
datos_train <- datos_final[indices, ]
datos_test <- datos_final[-indices, ]

## 1A. Crear modelo SVM con kernel lineal (Vanilla SVM)

# Crear modelo SVM con kernel lineal (Vanilla SVM)
modelo_vanilla <- svm(Category ~ ., data = datos_train, type = 'C-classification', kernel = 'linear')

# Predecir y evaluar
predicciones_vanilla <- predict(modelo_vanilla, datos_test)
confusionMatrix(predicciones_vanilla, datos_test$Category)

## 1B. Crear modelo SVM con kernel squared exponential (Gaussiano)

# Crear modelo SVM
modelo_gaussiano <- svm(Category ~ ., data = datos_train, type = 'C-classification', kernel = 'radial')

# Predecir y evaluar
predicciones_gaussiano <- predict(modelo_gaussiano, datos_test)
confusionMatrix(predicciones_gaussiano, datos_test$Category)

# Optimizar hiperparámetros para el modelo gaussiano
tune_result_gaussiano <- tune(svm, Category ~ ., data = datos_train, 
                              kernel = "radial", 
                              ranges = list(cost = 10^(-1:2), gamma = 10^(-2:2)))

# Mejores parámetros
mejores_parametros_gaussiano <- tune_result_gaussiano$best.parameters
mejores_parametros_gaussiano

## 1C. Crear modelo SVM con kernel locally periodic (Periodic)

library(kernlab)

# Función para calcular la matriz del kernel
generate_kernel_matrix <- function(x, y, sigma, p) {
  matrix(sapply(1:nrow(x), function(i) {
    sapply(1:nrow(y), function(j) {
      distance <- sqrt(sum((x[i, ] - y[j, ])^2))
      rbf_part <- exp(-1 / (sigma^2) * distance^2)
      periodic_part <- exp(-2 / p^2 * sin(pi * distance / p)^2)
      rbf_part * periodic_part
    })
  }), nrow = nrow(x), ncol = nrow(y))
}

# Parámetros del kernel
sigma <- 1  # Este valor debería ser optimizado
p <- 1      # Este valor también debería ser optimizado

# Datos de entrenamiento
xtrain <- as.matrix(datos_train[, -which(names(datos_train) == "Category")])
ytrain <- datos_train$Category

# Datos de prueba
xtest <- as.matrix(datos_test[, -which(names(datos_test) == "Category")])
ytest <- datos_test$Category

# Precalcular la matriz de kernel para el conjunto de entrenamiento
train_kernel_matrix <- generate_kernel_matrix(xtrain, xtrain, sigma, p)

# Entrenar el modelo SVM con la matriz de kernel
modelo_lp <- ksvm(as.kernelMatrix(train_kernel_matrix), ytrain, kernel = "matrix", C = 1)

# Precalcular la matriz de kernel para el conjunto de prueba (relacionando con los datos de entrenamiento)
test_kernel_matrix <- generate_kernel_matrix(xtest, xtrain, sigma, p)

# Realizar predicciones
predicciones_lp <- predict(modelo_lp, newdata = as.kernelMatrix(test_kernel_matrix))

# Evaluar el modelo
confusionMatrix(predicciones_lp, ytest)


# -----------------------------------------------------------------------------------------------------------

# Definir la cuadrícula de parámetros
# Estableceremos un espacio de parámetros que incluya diferentes valores de sigma y C.
# Sigma se relaciona con el parámetro gamma del kernel radial.
param_grid <- expand.grid(sigma = 2^(-5:5), C = 2^(-5:5))

# Limpiar los nombres de las clases en la variable 'Category'
datos_train$Category <- make.names(datos_train$Category, unique = TRUE)
datos_test$Category <- make.names(datos_test$Category, unique = TRUE)

# Ajustar el control de entrenamiento para clasificación multiclase
control <- trainControl(method = "repeatedcv",  # validación cruzada repetida
                        number = 10,            # número de pliegues en la validación cruzada
                        repeats = 3,            # repeticiones de CV
                        search = "grid",        # especifica búsqueda en cuadrícula
                        summaryFunction = multiClassSummary,  # para clasificación multiclase
                        classProbs = TRUE)  # calcular probabilidades de clase

# Entrenar el modelo SVM para clasificación multiclase (demora!!!)
svm_model <- train(Category ~ ., data = datos_train,
                   method = "svmRadial",
                   trControl = control,
                   tuneGrid = param_grid,
                   metric = "Accuracy",  # Cambiar a Accuracy ya que estamos en multiclase
                   preProcess = c("center", "scale"))  # centrar y escalar los datos

# Resultados de la búsqueda en cuadrícula
print(svm_model)

library(ggplot2)

# Graficar Accuracy vs. parámetros
ggplot(svm_model$results, aes(x = sigma, y = Accuracy, color = as.factor(C))) +
  geom_line() +
  labs(title = "Accuracy vs. Sigma para diferentes valores de C",
       color = "C")

# Graficar F1-score vs. parámetros
ggplot(svm_model$results, aes(x = sigma, y = F1, color = as.factor(C))) +
  geom_line() +
  labs(title = "F1-score vs. Sigma para diferentes valores de C",
       color = "C")

