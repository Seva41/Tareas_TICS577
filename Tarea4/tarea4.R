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
menu_data <- read.csv(file.choose(), header = TRUE)

# Crear la variable objetivo binaria y convertirla a factor
menu_data$Is_Breakfast <- as.factor(as.integer(menu_data$Category == "Breakfast"))

# Seleccionar las covariables numéricas
numeric_features <- menu_data %>% select_if(is.numeric)

# Dividir los datos en conjuntos de entrenamiento y prueba
set.seed(42)
splitIndex <- createDataPartition(menu_data$Is_Breakfast, p = .75, list = FALSE, times = 1)
train_data <- menu_data[splitIndex, ]
test_data <- menu_data[-splitIndex, ]

# Preparar los datos de entrenamiento
# Seleccionar solo las variables numéricas para el conjunto de entrenamiento
train_data_numeric <- train_data %>% select_if(is.numeric)
train_data_numeric$Is_Breakfast <- as.factor(train_data$Is_Breakfast)


# Entrenar el modelo SVM con kernel lineal (Vanilla SVM)
svm_model_vanilla <- svm(Is_Breakfast ~ ., data = train_data_numeric, type = "C-classification", kernel = "linear")

# Entrenar el modelo SVM con kernel radial (Squared Exponential)
svm_model_se <- svm(Is_Breakfast ~ ., data = train_data_numeric, type = "C-classification", kernel = "radial")

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

# ----------------------------
# Ejercicio 2

# 1. Obtener los índices de los vectores de soporte
support_vectors_indices <- svm_model_se$index

# Añadir Is_Breakfast a test_data_numeric
test_data_numeric$Is_Breakfast <- as.factor(test_data$Is_Breakfast)

# 2. Obtener los vectores de soporte
support_vectors <- test_data_numeric[support_vectors_indices, ]

# 3. Crear un Gráfico de Dispersión
ggplot(test_data_numeric, aes(x = Sodium, y = Protein, color = Is_Breakfast)) +
  geom_point() +
  geom_point(data = support_vectors, aes(x = Sodium, y = Protein), color = "red", size = 0.45) +
  labs(color = "Is Breakfast") +
  theme_minimal() +
  ggtitle("Support Vectors Highlighted in Red")

# ----------------------------
# Ejercicio 3

# 1. Calcular las distancias al hiperplano de decisión
distances <- as.numeric(predict(svm_model_se, test_data_numeric, decision.values = TRUE))

# 2. Identificar el umbral
threshold <- quantile(abs(distances), 0.15)

# 3. Aplicar el umbral para clasificar las observaciones
test_data_numeric$Prediction <- ifelse(abs(distances) > threshold, as.character(test_data_numeric$Is_Breakfast), "Intermediate")

# 4. Mostrar los resultados gráficamente
ggplot(test_data_numeric, aes(x = Sodium, y = Protein, color = Prediction)) +
  geom_point() +
  labs(color = "Prediction") +
  theme_minimal() +
  ggtitle("Classification with Intermediate Category")


# ----------------------------
# Ejercicio 4

# Crear la variable objetivo para "Beef & Pork and Chicken & Fish" vs "Otros"
menu_data$Is_Meat <- as.factor(as.integer(menu_data$Category %in% c("Beef & Pork", "Chicken & Fish")))

# Dividir los datos en conjuntos de entrenamiento y prueba para el nuevo problema
set.seed(42)
splitIndex_meat <- createDataPartition(menu_data$Is_Meat, p = .75, list = FALSE, times = 1)
train_data_meat <- menu_data[splitIndex_meat, ]
test_data_meat <- menu_data[-splitIndex_meat, ]

# Preparar los datos de entrenamiento
train_data_meat_numeric <- train_data_meat %>% select_if(is.numeric)
train_data_meat_numeric$Is_Meat <- as.factor(train_data_meat$Is_Meat)

# Entrenar el modelo SVM para "Beef & Pork and Chicken & Fish" vs "Otros"
svm_model_meat <- svm(Is_Meat ~ ., data = train_data_meat_numeric, type = "C-classification", kernel = "radial")

# Realizar predicciones con ambos modelos
predictions_breakfast <- predict(svm_model_se, newdata = test_data_numeric, decision.values = TRUE)
predictions_meat <- predict(svm_model_meat, newdata = test_data_numeric, decision.values = TRUE)

# Calcular los valores de decisión para ambos modelos
decision_values_breakfast <- attr(predictions_breakfast, "decision.values")
decision_values_meat <- attr(predictions_meat, "decision.values")

# Asegurarse de que los valores de decisión son vectores y tienen la misma longitud
decision_values_breakfast <- as.numeric(decision_values_breakfast)
decision_values_meat <- as.numeric(decision_values_meat)

# Clasificación final basada en los valores de decisión
final_predictions <- ifelse(decision_values_breakfast > 0, "Breakfast",
  ifelse(decision_values_meat > 0, "Meat", "Other")
)

# Verificar que final_predictions tenga la misma longitud que las otras columnas
length(final_predictions) == nrow(test_data)

# Crear un dataframe para graficar los resultados
result_data <- data.frame(Sodium = test_data$Sodium, Protein = test_data$Protein, Category = final_predictions)
result_data$Category <- as.factor(result_data$Category)

# Mostrar los resultados gráficamente
ggplot(result_data, aes(x = Sodium, y = Protein, color = Category)) +
  geom_point() +
  labs(color = "Category") +
  theme_minimal() +
  ggtitle("Classification into Three Categories")
