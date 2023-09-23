# Cargar las bibliotecas necesarias
library(caret)
library(ggplot2)

# Cargar los datos desde el archivo datos.txt
datos <- read.table("C:/Users/sebad/OneDrive - Universidad Adolfo Ibanez/Code/Métodos basados en Kernel/Tareas/Tarea2/datos.txt", header = TRUE)

# Seleccionar solo las covariables "x" y "z"
datos_subset <- datos[, c("pointX", "pointZ", "pointY")]

# Establecer el número de folds para cross-validation
num_folds <- 5

# Crear un contenedor para los resultados de cross-validation
cv_results <- data.frame(R2 = numeric(num_folds), RMSE = numeric(num_folds))

# Realizar k-fold cross-validation
set.seed(123) # Establecer una semilla para reproducibilidad
folds <- createFolds(datos$kfold, k = num_folds, returnTrain = FALSE)
for (i in 1:num_folds) {
    # Dividir los datos en conjunto de entrenamiento y prueba
    train_data <- datos_subset[-folds[[i]], ]
    test_data <- datos_subset[folds[[i]], ]

    # Ajustar el modelo de regresión lineal múltiple
    lm_model <- lm(pointY ~ pointX + pointZ, data = train_data)

    # Predecir en el conjunto de prueba
    predictions <- predict(lm_model, newdata = test_data)

    # Calcular R^2
    r_squared <- cor(predictions, test_data$pointY)^2

    # Calcular RMSE
    rmse <- sqrt(mean((predictions - test_data$pointY)^2))

    # Almacenar los resultados
    cv_results[i, "R2"] <- r_squared
    cv_results[i, "RMSE"] <- rmse
}

# a) Imprimir los estimadores de mínimos cuadrados
lm_model_summary <- summary(lm_model)
print(lm_model_summary$coefficients)

# b) Gráficos de dispersión
ggplot(datos_subset, aes(x = pointX, y = pointY)) +
    geom_point() +
    geom_smooth(method = "lm", formula = y ~ x, se = FALSE) +
    ggtitle("Relación entre x y la variable respuesta")

ggplot(datos_subset, aes(x = pointZ, y = pointY)) +
    geom_point() +
    geom_smooth(method = "lm", formula = y ~ x, se = FALSE) +
    ggtitle("Relación entre z y la variable respuesta")

# c) Gráfico boxplot para R^2 y RMSE
ggplot(cv_results, aes(x = "", y = R2)) +
    geom_boxplot() +
    ylab("R-squared") +
    ggtitle("Distribución de R-squared en Cross-Validation")

ggplot(cv_results, aes(x = "", y = RMSE)) +
    geom_boxplot() +
    ylab("RMSE") +
    ggtitle("Distribución de RMSE en Cross-Validation")

# 2
# a)
# Crear contenedores para resultados
ridge_results <- data.frame(Lambda = numeric(num_folds), R2 = numeric(num_folds), RMSE = numeric(num_folds))

# Definir función para estandarizar los datos
standardize <- function(x) {
    return((x - mean(x)) / sd(x))
}

# Realizar k-fold cross-validation
set.seed(123)
folds <- createFolds(datos$kfold, k = num_folds, returnTrain = FALSE)
for (i in 1:num_folds) {
    # Dividir los datos en conjunto de entrenamiento y prueba
    train_data <- datos_subset[-folds[[i]], ]
    test_data <- datos_subset[folds[[i]], ]

    # Estandarizar los predictores en ambos conjuntos
    train_data$pointX <- standardize(train_data$pointX) # Estandarizar x en el conjunto de entrenamiento
    train_data$pointZ <- standardize(train_data$pointZ) # Estandarizar z en el conjunto de entrenamiento
    test_data$pointX <- standardize(test_data$pointX) # Estandarizar x en el conjunto de prueba
    test_data$pointZ <- standardize(test_data$pointZ) # Estandarizar z en el conjunto de prueba

    # Realizar la regresión Ridge para diferentes valores de lambda
    lambda_values <- seq(0.1, 3, 0.05)
    rmse_values <- numeric(length(lambda_values))
    for (j in 1:length(lambda_values)) {
        lambda <- lambda_values[j]
        X <- cbind(1, train_data$pointX, train_data$pointZ) # Agregar intercepto
        Y <- train_data$pointY
        n <- nrow(X)
        p <- ncol(X)

        # Calcular coeficientes Ridge manualmente
        ridge_coef <- solve(t(X) %*% X + lambda * diag(p), t(X) %*% Y)

        # Predecir en el conjunto de prueba
        X_test <- cbind(1, test_data$pointX, test_data$pointZ)
        predictions <- X_test %*% ridge_coef

        # Calcular RMSE
        rmse_values[j] <- sqrt(mean((predictions - test_data$pointY)^2))
    }

    # Encontrar el lambda óptimo que minimiza el RMSE
    optimal_lambda <- lambda_values[which.min(rmse_values)]

    # Ajustar el modelo de Ridge con el lambda óptimo
    X <- cbind(1, train_data$pointX, train_data$pointZ) # Agregar intercepto
    Y <- train_data$pointY
    ridge_coef <- solve(t(X) %*% X + optimal_lambda * diag(p), t(X) %*% Y)

    # Predecir en el conjunto de prueba
    X_test <- cbind(1, test_data$pointX, test_data$pointZ)
    predictions <- X_test %*% ridge_coef

    # Calcular R^2
    r_squared <- cor(predictions, test_data$pointY)^2

    # Calcular RMSE
    rmse <- sqrt(mean((predictions - test_data$pointY)^2))

    # Almacenar los resultados
    ridge_results[i, "Lambda"] <- optimal_lambda
    ridge_results[i, "R2"] <- r_squared
    ridge_results[i, "RMSE"] <- rmse
}

# Imprimir la tabla de estimadores de Ridge y lambdas óptimos
print(ridge_results)

# b)
# Escoge una iteración de cross-validation para utilizar el lambda óptimo
iter_to_plot <- 1

# Ajustar el modelo de Ridge con el lambda óptimo de esa iteración
lambda_optimal <- ridge_results$Lambda[iter_to_plot]

# Estandarizar los predictores en el conjunto completo de datos
datos_subset$pointX <- standardize(datos_subset$pointX)
datos_subset$pointZ <- standardize(datos_subset$pointZ)

# Dividir los datos en conjunto de entrenamiento y prueba
train_data <- datos_subset[-folds[[iter_to_plot]], ]
test_data <- datos_subset[folds[[iter_to_plot]], ]

# Estandarizar los predictores en el conjunto de prueba
test_data$pointX <- standardize(test_data$pointX)
test_data$pointZ <- standardize(test_data$pointZ)

# Ajustar el modelo de Ridge con el lambda óptimo en el conjunto de entrenamiento
X_train <- cbind(1, train_data$pointX, train_data$pointZ) # Agregar intercepto
Y_train <- train_data$pointY
p <- ncol(X_train)

# Calcular coeficientes Ridge manualmente
ridge_coef <- solve(t(X_train) %*% X_train + lambda_optimal * diag(p), t(X_train) %*% Y_train)

# Predecir en el conjunto de prueba
X_test <- cbind(1, test_data$pointX, test_data$pointZ)
predictions <- X_test %*% ridge_coef

# Crear un gráfico de dispersión para x vs. y
plot(test_data$pointX, test_data$pointY, xlab = "x", ylab = "pointY", main = "Relación entre x y la variable respuesta (Ridge)")
points(test_data$pointX, predictions, col = "red")

# Crear un gráfico de dispersión para z vs. y
plot(test_data$pointZ, test_data$pointY, xlab = "z", ylab = "pointY", main = "Relación entre z y la variable respuesta (Ridge)")
points(test_data$pointZ, predictions, col = "red")

# c)
# Gráfico boxplot para R^2
boxplot(ridge_results$R2, main = "Distribución de R-squared en Cross-Validation (Ridge)", ylab = "R-squared")

# Gráfico boxplot para RMSE
boxplot(ridge_results$RMSE, main = "Distribución de RMSE en Cross-Validation (Ridge)", ylab = "RMSE")

# d)
# Escoge una iteración de cross-validation para mostrar el encogimiento de parámetros
iter_to_plot <- 1

# Ajustar el modelo de Ridge con el lambda óptimo de esa iteración
lambda_optimal <- ridge_results$Lambda[iter_to_plot]

# Estandarizar los predictores en el conjunto completo de datos
datos_subset$pointX <- standardize(datos_subset$pointX)
datos_subset$pointZ <- standardize(datos_subset$pointZ)

# Ajustar el modelo de Ridge con diferentes valores de lambda en el conjunto completo de datos
lambda_values <- seq(0.1, 3, 0.05)
coefficients <- matrix(NA, nrow = length(lambda_values), ncol = 3) # Para almacenar coeficientes

for (i in 1:length(lambda_values)) {
    lambda <- lambda_values[i]

    # Ajustar el modelo de Ridge en el conjunto completo de datos
    X <- cbind(1, datos_subset$pointX, datos_subset$pointZ) # Agregar intercepto
    Y <- datos_subset$pointY
    p <- ncol(X)

    # Calcular coeficientes Ridge manualmente
    ridge_coef <- solve(t(X) %*% X + lambda * diag(p), t(X) %*% Y)
    coefficients[i, ] <- ridge_coef
}

# Crear un gráfico de encogimiento de parámetros
plot(lambda_values, coefficients[, 2],
    type = "l", xlab = "Lambda", ylab = "Coeficiente para x",
    main = "Encogimiento de parámetros para x (Ridge)"
)

lines(lambda_values, coefficients[, 3], col = "red")
legend("topright", legend = c("x", "z"), col = 1:2, lty = 1, title = "Variables")

# 3
# a)
# Crear contenedores para resultados
kernel_ridge_results <- data.frame(L = numeric(num_folds), Lambda = numeric(num_folds))

# Crear una función de kernel
kernel <- function(u, v, l) {
    return(exp(-sum((u - v)^2) / (2 * l^2)))
}

# Realizar k-fold cross-validation
set.seed(123)
folds <- createFolds(datos$kfold, k = num_folds, returnTrain = FALSE)
for (i in 1:num_folds) {
    # Dividir los datos en conjunto de entrenamiento y prueba
    train_data <- datos_subset[-folds[[i]], ]
    test_data <- datos_subset[folds[[i]], ]

    # Estandarizar los predictores en ambos conjuntos
    train_data$pointX <- standardize(train_data$pointX)
    train_data$pointZ <- standardize(train_data$pointZ)
    test_data$pointX <- standardize(test_data$pointX)
    test_data$pointZ <- standardize(test_data$pointZ)

    # Parámetros de búsqueda
    grid <- expand.grid(L = seq(0.1, 2, length = 10), Lambda = seq(0.1, 3, length = 50))
    rmse_values <- numeric(nrow(grid))

    for (j in 1:nrow(grid)) {
        L <- grid$L[j]
        Lambda <- grid$Lambda[j]

        # Calcular matriz de kernel para el conjunto de entrenamiento
        n_train <- nrow(train_data)
        K_train <- matrix(0, n_train, n_train)
        for (k in 1:n_train) {
            for (l in 1:n_train) {
                K_train[k, l] <- kernel(
                    c(train_data$pointX[k], train_data$pointZ[k]),
                    c(train_data$pointX[l], train_data$pointZ[l]), L
                )
            }
        }

        # Ajustar el modelo de Kernel Ridge Regression
        alpha <- solve(K_train + Lambda * diag(n_train), train_data$pointY)

        # Calcular RMSE en el conjunto de prueba
        n_test <- nrow(test_data)
        K_test <- matrix(0, n_test, n_train)
        for (k in 1:n_test) {
            for (l in 1:n_train) {
                K_test[k, l] <- kernel(
                    c(test_data$pointX[k], test_data$pointZ[k]),
                    c(train_data$pointX[l], train_data$pointZ[l]), L
                )
            }
        }
        predictions <- K_test %*% alpha
        rmse_values[j] <- sqrt(mean((predictions - test_data$pointY)^2))
    }

    # Encontrar los parámetros (L, Lambda) óptimos que minimizan el RMSE
    optimal_params <- grid[which.min(rmse_values), ]

    # Almacenar los resultados
    kernel_ridge_results[i, "L"] <- optimal_params$L
    kernel_ridge_results[i, "Lambda"] <- optimal_params$Lambda
}

# Imprimir la tabla de parámetros óptimos
print(kernel_ridge_results)

# b)
# Elegir una iteración de cross-validation para utilizar los parámetros óptimos
iter_to_plot <- 1
L_optimal <- kernel_ridge_results$L[iter_to_plot]
Lambda_optimal <- kernel_ridge_results$Lambda[iter_to_plot]

# Estandarizar los predictores en el conjunto completo de datos
datos_subset$pointX <- standardize(datos_subset$pointX)
datos_subset$pointZ <- standardize(datos_subset$pointZ)

# Dividir los datos en conjunto de entrenamiento y prueba
train_data <- datos_subset[-folds[[iter_to_plot]], ]
test_data <- datos_subset[folds[[iter_to_plot]], ]

# Estandarizar los predictores en el conjunto de prueba
test_data$pointX <- standardize(test_data$pointX)
test_data$pointZ <- standardize(test_data$pointZ)

# Calcular matriz de kernel para el conjunto de entrenamiento
n_train <- nrow(train_data)
K_train <- matrix(0, n_train, n_train)
for (i in 1:n_train) {
    for (j in 1:n_train) {
        K_train[i, j] <- kernel(
            c(train_data$pointX[i], train_data$pointZ[i]),
            c(train_data$pointX[j], train_data$pointZ[j]), L_optimal
        )
    }
}

# Ajustar el modelo de Kernel Ridge Regression
alpha <- solve(K_train + Lambda_optimal * diag(n_train), train_data$pointY)

# Calcular matriz de kernel entre datos de prueba y entrenamiento
n_test <- nrow(test_data)
K_test <- matrix(0, n_test, n_train)
for (i in 1:n_test) {
    for (j in 1:n_train) {
        K_test[i, j] <- kernel(
            c(test_data$pointX[i], test_data$pointZ[i]),
            c(train_data$pointX[j], train_data$pointZ[j]), L_optimal
        )
    }
}

# Predecir en el conjunto de prueba
predictions <- K_test %*% alpha

# Crear un gráfico de dispersión para x vs. y
plot(test_data$pointX, test_data$pointY,
    xlab = "x", ylab = "pointY",
    main = "Relación entre x y la variable respuesta (Kernel Ridge)"
)
points(test_data$pointX, predictions, col = "red")

# Crear un gráfico de dispersión para z vs. y
plot(test_data$pointZ, test_data$pointY,
    xlab = "z", ylab = "pointY",
    main = "Relación entre z y la variable respuesta (Kernel Ridge)"
)
points(test_data$pointZ, predictions, col = "red")

# c)
# Crear contenedores para resultados de R^2 y RMSE
r_squared_values <- numeric(num_folds)
rmse_values <- numeric(num_folds)

# Realizar k-fold cross-validation nuevamente para calcular R^2 y RMSE
for (i in 1:num_folds) {
    # Elegir los parámetros óptimos
    L_optimal <- kernel_ridge_results$L[i]
    Lambda_optimal <- kernel_ridge_results$Lambda[i]

    # Estandarizar los predictores en el conjunto completo de datos
    datos_subset$pointX <- standardize(datos_subset$pointX)
    datos_subset$pointZ <- standardize(datos_subset$pointZ)

    # Dividir los datos en conjunto de entrenamiento y prueba
    train_data <- datos_subset[-folds[[i]], ]
    test_data <- datos_subset[folds[[i]], ]

    # Estandarizar los predictores en el conjunto de prueba
    test_data$pointX <- standardize(test_data$pointX)
    test_data$pointZ <- standardize(test_data$pointZ)

    # Calcular matriz de kernel para el conjunto de entrenamiento
    n_train <- nrow(train_data)
    K_train <- matrix(0, n_train, n_train)
    for (j in 1:n_train) {
        for (k in 1:n_train) {
            K_train[j, k] <- kernel(
                c(train_data$pointX[j], train_data$pointZ[j]),
                c(train_data$pointX[k], train_data$pointZ[k]), L_optimal
            )
        }
    }

    # Ajustar el modelo de Kernel Ridge Regression
    alpha <- solve(K_train + Lambda_optimal * diag(n_train), train_data$pointY)

    # Calcular matriz de kernel entre datos de prueba y entrenamiento
    n_test <- nrow(test_data)
    K_test <- matrix(0, n_test, n_train)
    for (j in 1:n_test) {
        for (k in 1:n_train) {
            K_test[j, k] <- kernel(
                c(test_data$pointX[j], test_data$pointZ[j]),
                c(train_data$pointX[k], train_data$pointZ[k]), L_optimal
            )
        }
    }

    # Predecir en el conjunto de prueba
    predictions <- K_test %*% alpha

    # Calcular R^2
    r_squared <- cor(predictions, test_data$pointY)^2

    # Calcular RMSE
    rmse <- sqrt(mean((predictions - test_data$pointY)^2))

    # Almacenar los resultados
    r_squared_values[i] <- r_squared
    rmse_values[i] <- rmse
}

# Crear un gráfico boxplot para R^2
boxplot(r_squared_values, main = "Distribución de R-squared en Cross-Validation (Kernel Ridge)", ylab = "R-squared")

# Crear un gráfico boxplot para RMSE
boxplot(rmse_values, main = "Distribución de RMSE en Cross-Validation (Kernel Ridge)", ylab = "RMSE")
