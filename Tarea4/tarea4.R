install.packages("e1071")
install.packages("tidyverse")
install.packages("caret")
install.packages("kernlab")

# Cargar las bibliotecas
library(e1071)
library(tidyverse)
library(caret)
library(kernlab)

# Cargar los datos
menu_data <- read.csv("C:/Users/sebad/OneDrive - Universidad Adolfo Ibanez/Code/Métodos basados en Kernel/Tareas/Tarea4/menu.csv", stringsAsFactors = FALSE)

# Crear la variable objetivo binaria y convertirla a factor
menu_data$Is_Breakfast <- as.factor(as.integer(menu_data$Category == "Breakfast"))

# Seleccionar las covariables numéricas
numeric_features <- menu_data %>% select_if(is.numeric)

# Dividir los datos en conjuntos de entrenamiento y prueba
set.seed(42)
splitIndex <- createDataPartition(menu_data$Is_Breakfast, p = .75, list = FALSE, times = 1)
train_data <- menu_data[splitIndex, ]
test_data  <- menu_data[-splitIndex, ]

# Preparar los datos de entrenamiento
# Seleccionar solo las variables numéricas para el conjunto de entrenamiento
train_data_numeric <- train_data %>% select_if(is.numeric)
train_data_numeric$Is_Breakfast <- as.factor(train_data$Is_Breakfast)


# Entrenar el modelo SVM con kernel lineal (Vanilla SVM)
svm_model_vanilla <- svm(Is_Breakfast ~ ., data = train_data_numeric, type = 'C-classification', kernel = 'linear')

# Entrenar el modelo SVM con kernel radial (Squared Exponential)
svm_model_se <- svm(Is_Breakfast ~ ., data = train_data_numeric, type = 'C-classification', kernel = 'radial')

# Preparar los datos de prueba
# Seleccionar las características numéricas para los datos de prueba
test_data_numeric <- test_data %>% select_if(is.numeric)

# Realizar predicciones
predictions_vanilla <- predict(svm_model_vanilla, newdata = test_data_numeric)
predictions_se <- predict(svm_model_se, newdata = test_data_numeric)

# Convertir las predicciones y la variable objetivo a factores con los mismos niveles
predictions_vanilla_factor <- as.factor(predictions_vanilla)
predictions_se_factor <- as.factor(predictions_se)
Is_Breakfast_factor <- as.factor(test_data$Is_Breakfast)

levels(predictions_vanilla_factor) <- levels(Is_Breakfast_factor)
levels(predictions_se_factor) <- levels(Is_Breakfast_factor)

# Calcular y mostrar las matrices de confusión
confusionMatrix(predictions_vanilla_factor, Is_Breakfast_factor)
confusionMatrix(predictions_se_factor, Is_Breakfast_factor)

