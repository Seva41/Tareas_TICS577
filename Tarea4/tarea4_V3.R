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

# Crear modelo SVM con kernel lineal (Vanilla SVM)
modelo_vanilla <- svm(Category ~ ., data = datos_train, type = 'C-classification', kernel = 'linear')

# Predecir y evaluar
predicciones_vanilla <- predict(modelo_vanilla, datos_test)
confusionMatrix(predicciones_vanilla, datos_test$Category)





