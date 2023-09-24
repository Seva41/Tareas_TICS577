# Se cargan las librerías a utilizar
library(caret)
library(ggplot2)


### Ejericio 1
## 1A)

# Se cargan los datos desde el archivo datos.txt
datos <- read.table("/Users/seva/Library/CloudStorage/OneDrive-UniversidadAdolfoIbanez/Code/Métodos basados en Kernel/Tareas/Tarea2/datos.txt", header = TRUE) # Cambiar la ruta según corresponda

datos_s <- datos[, c("pointX", "pointZ", "pointY")] # Se seleccionan las columnas de interés

num_folds <- 5 # Número de folds para cross-validation

# Se crea una nueva columna para el k-fold cross-validation
cv_results <- data.frame(R2 = numeric(num_folds), RMSE = numeric(num_folds))

# Se realiza el k-fold cross-validation
set.seed(123) # Set de semilla para reproducibilidad
folds <- createFolds(datos$kfold, k = num_folds, returnTrain = FALSE)

# Se itera sobre cada fold
for (i in 1:num_folds) {
    # Se dividen los datos en conjunto de entrenamiento y prueba
    train_data <- datos_s[-folds[[i]], ]
    test_data <- datos_s[folds[[i]], ]

    # Se ajusta el modelo de regresión lineal
    lm_model <- lm(pointY ~ pointX + pointZ, data = train_data)

    # Se predicen los valores de la variable respuesta en el conjunto de prueba
    prediccion <- predict(lm_model, newdata = test_data)

    # Se calcula R^2
    R_2 <- cor(prediccion, test_data$pointY)^2

    # Se calcula RMSE
    rmse <- sqrt(mean((prediccion - test_data$pointY)^2))

    # Se guardan los resultados
    cv_results[i, "R2"] <- R_2
    cv_results[i, "RMSE"] <- rmse
}

# Se imprimen los resultados
lm_model_summary <- summary(lm_model)
print(lm_model_summary$coefficients)

## 1B)
# Gráfico de dispersión para x vs y
ggplot(datos_s, aes(x = pointX, y = pointY)) +
    geom_point() +
    geom_smooth(method = "lm", formula = y ~ x, se = FALSE) +
    ggtitle("Relación entre x y la variable respuesta")

# Gráfico de dispersión para z vs y
ggplot(datos_s, aes(x = pointZ, y = pointY)) +
    geom_point() +
    geom_smooth(method = "lm", formula = y ~ x, se = FALSE) +
    ggtitle("Relación entre z y la variable respuesta")

## 1C)
# Gráfico boxplot para R^2
ggplot(cv_results, aes(x = "", y = R2)) +
    geom_boxplot() +
    ylab("R-squared") +
    ggtitle("Distribución de R-squared en Cross-Validation")

ggplot(cv_results, aes(x = "", y = RMSE)) +
    geom_boxplot() +
    ylab("RMSE") +
    ggtitle("Distribución de RMSE en Cross-Validation")



### Ejericio 2
## 2A)
# Se crea un contenedor para los resultados
ridge_resultado <- data.frame(
    Lambda = numeric(num_folds),
    R2 = numeric(num_folds),
    RMSE = numeric(num_folds)
)

# Se crea una función para estandarizar los datos
standardize <- function(x) {
    return((x - mean(x)) / sd(x))
}

# Se realiza el k-fold cross-validation
set.seed(123) # Set de semilla para reproducibilidad
folds <- createFolds(datos$kfold, k = num_folds, returnTrain = FALSE)
# Se itera sobre cada fold
for (i in 1:num_folds) {
    # Se dividen los datos en conjunto de entrenamiento y prueba
    train_data <- datos_s[-folds[[i]], ]
    test_data <- datos_s[folds[[i]], ]

    # Se estandarizan los predictores en ambos conjuntos
    train_data$pointX <- standardize(train_data$pointX) # Estandarizar x en el conjunto de entrenamiento
    train_data$pointZ <- standardize(train_data$pointZ) # Estandarizar z en el conjunto de entrenamiento
    test_data$pointX <- standardize(test_data$pointX) # Estandarizar x en el conjunto de prueba
    test_data$pointZ <- standardize(test_data$pointZ) # Estandarizar z en el conjunto de prueba

    # Se ajusta el modelo de Ridge con diferentes valores de lambda
    valores_l <- seq(0.1, 3, 0.05)
    valores_rmse <- numeric(length(valores_l))
    for (j in 1:length(valores_l)) {
        lambda <- valores_l[j]
        X <- cbind(1, train_data$pointX, train_data$pointZ) # Agregar intercepto
        Y <- train_data$pointY # Variable respuesta
        n <- nrow(X) # Número de observaciones
        p <- ncol(X) # Número de predictores

        # Se calculan los coeficientes Ridge manualmente
        coef_ridge <- solve(t(X) %*% X + lambda * diag(p), t(X) %*% Y)

        # Se predicen los valores de la variable respuesta en el conjunto de prueba
        X_test <- cbind(1, test_data$pointX, test_data$pointZ)
        prediccion <- X_test %*% coef_ridge

        # Se calcula RMSE
        valores_rmse[j] <- sqrt(mean((prediccion - test_data$pointY)^2))
    }

    # Se encuentra el lambda óptimo que minimiza el RMSE
    l_optimo <- valores_l[which.min(valores_rmse)]

    # Se ajusta el modelo de Ridge con el lambda óptimo en el conjunto de entrenamiento
    X <- cbind(1, train_data$pointX, train_data$pointZ) # Agregar intercepto
    Y <- train_data$pointY # Variable respuesta
    coef_ridge <- solve(t(X) %*% X + l_optimo * diag(p), t(X) %*% Y) # Calcular coeficientes Ridge manualmente

    # Se predicen los valores de la variable respuesta en el conjunto de prueba
    X_test <- cbind(1, test_data$pointX, test_data$pointZ)
    prediccion <- X_test %*% coef_ridge

    # Se calcula R^2
    R_2 <- cor(prediccion, test_data$pointY)^2

    # Se calcula RMSE
    rmse <- sqrt(mean((prediccion - test_data$pointY)^2))

    # Se guardan los resultados
    ridge_resultado[i, "Lambda"] <- l_optimo
    ridge_resultado[i, "R2"] <- R_2
    ridge_resultado[i, "RMSE"] <- rmse
}

print(ridge_resultado)

## 2B)
# Se escoge una iteración de cross-validation para mostrar el encogimiento de parámetros
iter <- 1

# Se ajusta el modelo de Ridge con el lambda óptimo de esa iteración
lambda_optimo <- ridge_resultado$Lambda[iter]

# Se estandarizan los predictores en el conjunto completo de datos
datos_s$pointX <- standardize(datos_s$pointX)
datos_s$pointZ <- standardize(datos_s$pointZ)

# Se ajusta el modelo de Ridge con diferentes valores de lambda en el conjunto completo de datos
train_data <- datos_s[-folds[[iter]], ]
test_data <- datos_s[folds[[iter]], ]

# Se estandarizan los predictores en ambos conjuntos
test_data$pointX <- standardize(test_data$pointX)
test_data$pointZ <- standardize(test_data$pointZ)

# Se ajusta el modelo de Ridge con diferentes valores de lambda
X_train <- cbind(1, train_data$pointX, train_data$pointZ) # Agregar intercepto
Y_train <- train_data$pointY # Variable respuesta
p <- ncol(X_train) # Número de predictores

# Se calculan los coeficientes Ridge manualmente
coef_ridge <- solve(t(X_train) %*% X_train + lambda_optimo * diag(p), t(X_train) %*% Y_train)

# Se predicen los valores de la variable respuesta en el conjunto de prueba
X_test <- cbind(1, test_data$pointX, test_data$pointZ)
prediccion <- X_test %*% coef_ridge

# Se crea un gráfico de dispersión para x vs. y
plot(test_data$pointX, test_data$pointY, xlab = "x", ylab = "pointY", main = "Relación entre x y la variable respuesta (Ridge)")
points(test_data$pointX, prediccion, col = "red")

# Se crea un gráfico de dispersión para z vs. y
plot(test_data$pointZ, test_data$pointY, xlab = "z", ylab = "pointY", main = "Relación entre z y la variable respuesta (Ridge)")
points(test_data$pointZ, prediccion, col = "red")

## 2C)
# Gráfico boxplot para R^2
boxplot(ridge_resultado$R2, main = "Distribución de R-squared en Cross-Validation (Ridge)", ylab = "R-squared")

# Gráfico boxplot para RMSE
boxplot(ridge_resultado$RMSE, main = "Distribución de RMSE en Cross-Validation (Ridge)", ylab = "RMSE")

## 2D)
# Se escoge una iteración de cross-validation para mostrar el encogimiento de parámetros
iter <- 1

# Se ajusta el modelo de Ridge con el lambda óptimo de esa iteración
lambda_optimo <- ridge_resultado$Lambda[iter]

# Se estandarizan los predictores en el conjunto completo de datos
datos_s$pointX <- standardize(datos_s$pointX)
datos_s$pointZ <- standardize(datos_s$pointZ)

# Se ajusta el modelo de Ridge con diferentes valores de lambda en el conjunto completo de datos
valores_l <- seq(0.1, 3, 0.05)
coefficients <- matrix(NA, nrow = length(valores_l), ncol = 3) # Se crea una matriz para almacenar los coeficientes

# Se itera sobre cada valor de lambda
for (i in 1:length(valores_l)) {
    lambda <- valores_l[i]

    # Se ajusta el modelo de Ridge con diferentes valores de lambda
    X <- cbind(1, datos_s$pointX, datos_s$pointZ) # Agregar intercepto
    Y <- datos_s$pointY # Variable respuesta
    p <- ncol(X) # Número de predictores

    # Se calculan los coeficientes Ridge manualmente
    coef_ridge <- solve(t(X) %*% X + lambda * diag(p), t(X) %*% Y)
    coefficients[i, ] <- coef_ridge
}

# Se crea un gráfico para mostrar el encogimiento de parámetros
plot(valores_l, coefficients[, 2],
    type = "l", xlab = "Lambda", ylab = "Coeficiente para x",
    main = "Encogimiento de parámetros para x (Ridge)"
)

# Se agrega una línea vertical para el lambda óptimo
lines(valores_l, coefficients[, 3], col = "red")
legend("topright", legend = c("x", "z"), col = 1:2, lty = 1, title = "Variables")

### Ejericio 3
## 3A)
# Se crea un contenedor para los resultados
kernel_ridge_resultado <- data.frame(L = numeric(num_folds), Lambda = numeric(num_folds))

# Se crea una función de kernel
kernel <- function(u, v, l) {
    return(exp(-sum((u - v)^2) / (2 * l^2)))
}

# Se realiza el k-fold cross-validation
set.seed(123) # Set de semilla para reproducibilidad
folds <- createFolds(datos$kfold, k = num_folds, returnTrain = FALSE)
# Se itera sobre cada fold
for (i in 1:num_folds) {
    # Se dividen los datos en conjunto de entrenamiento y prueba
    train_data <- datos_s[-folds[[i]], ]
    test_data <- datos_s[folds[[i]], ]

    # Se estandarizan los predictores en ambos conjuntos
    train_data$pointX <- standardize(train_data$pointX)
    train_data$pointZ <- standardize(train_data$pointZ)
    test_data$pointX <- standardize(test_data$pointX)
    test_data$pointZ <- standardize(test_data$pointZ)

    # Parámetros de búsqueda
    grid <- expand.grid(L = seq(0.1, 2, length = 10), Lambda = seq(0.1, 3, length = 50))
    valores_rmse <- numeric(nrow(grid)) # Contenedor para RMSE

    # Iterar sobre cada combinación de parámetros
    for (j in 1:nrow(grid)) {
        L <- grid$L[j]
        Lambda <- grid$Lambda[j]

        # Se calcula la matriz de kernel para el conjunto de entrenamiento
        n_train <- nrow(train_data)
        K_train <- matrix(0, n_train, n_train)
        # Se itera sobre cada par de observaciones
        for (k in 1:n_train) {
            for (l in 1:n_train) {
                K_train[k, l] <- kernel(
                    c(train_data$pointX[k], train_data$pointZ[k]), # Vector u
                    c(train_data$pointX[l], train_data$pointZ[l]), L # Vector v
                )
            }
        }

        # Se ajusta el modelo de Kernel Ridge Regression
        alpha <- solve(K_train + Lambda * diag(n_train), train_data$pointY)

        # Se calcula la matriz de kernel entre datos de prueba y entrenamiento
        n_test <- nrow(test_data)
        K_test <- matrix(0, n_test, n_train)
        # Se itera sobre cada par de observaciones
        for (k in 1:n_test) {
            for (l in 1:n_train) {
                K_test[k, l] <- kernel(
                    c(test_data$pointX[k], test_data$pointZ[k]), # Vector u
                    c(train_data$pointX[l], train_data$pointZ[l]), L # Vector v
                )
            }
        }
        # Se predicen los valores de la variable respuesta en el conjunto de prueba
        prediccion <- K_test %*% alpha
        valores_rmse[j] <- sqrt(mean((prediccion - test_data$pointY)^2))
    }

    # Se encuentra la combinación de parámetros que minimiza el RMSE
    optimal_params <- grid[which.min(valores_rmse), ]

    # Se guardan los resultados
    kernel_ridge_resultado[i, "L"] <- optimal_params$L
    kernel_ridge_resultado[i, "Lambda"] <- optimal_params$Lambda
}

print(kernel_ridge_resultado)

## 3B)
# Se escoge una iteración de cross-validation para mostrar el encogimiento de parámetros
iter <- 1
L_optimo <- kernel_ridge_resultado$L[iter]
lambda_optimo <- kernel_ridge_resultado$Lambda[iter]

# Se estandarizan los predictores en el conjunto completo de datos
datos_s$pointX <- standardize(datos_s$pointX)
datos_s$pointZ <- standardize(datos_s$pointZ)

# SSe divide el conjunto de datos en conjunto de entrenamiento y prueba
train_data <- datos_s[-folds[[iter]], ]
test_data <- datos_s[folds[[iter]], ]

# Se estandarizan los predictores en ambos conjuntos
test_data$pointX <- standardize(test_data$pointX)
test_data$pointZ <- standardize(test_data$pointZ)

# Se calcula la matriz de kernel para el conjunto de entrenamiento
n_train <- nrow(train_data)
K_train <- matrix(0, n_train, n_train)
# Se itera sobre cada par de observaciones
for (i in 1:n_train) {
    for (j in 1:n_train) {
        K_train[i, j] <- kernel(
            c(train_data$pointX[i], train_data$pointZ[i]), # Vector u
            c(train_data$pointX[j], train_data$pointZ[j]), L_optimo # Vector v
        )
    }
}

# Se ajusta el modelo de Kernel Ridge Regression
alpha <- solve(K_train + lambda_optimo * diag(n_train), train_data$pointY)

# Se calcula la matriz de kernel entre datos de prueba y entrenamiento
n_test <- nrow(test_data)
K_test <- matrix(0, n_test, n_train)
# Se itera sobre cada par de observaciones
for (i in 1:n_test) {
    for (j in 1:n_train) {
        K_test[i, j] <- kernel(
            c(test_data$pointX[i], test_data$pointZ[i]),
            c(train_data$pointX[j], train_data$pointZ[j]), L_optimo
        )
    }
}

# Se predicen los valores de la variable respuesta en el conjunto de prueba
prediccion <- K_test %*% alpha

# Se crea un gráfico de dispersión para x vs. y
plot(test_data$pointX, test_data$pointY,
    xlab = "x", ylab = "pointY",
    main = "Relación entre x y la variable respuesta (Kernel Ridge)"
)
points(test_data$pointX, prediccion, col = "red")

# Se crea un gráfico de dispersión para z vs. y
plot(test_data$pointZ, test_data$pointY,
    xlab = "z", ylab = "pointY",
    main = "Relación entre z y la variable respuesta (Kernel Ridge)"
)
points(test_data$pointZ, prediccion, col = "red")

## 3C)
# Se crea un contenedor para los resultados
valores_R_2 <- numeric(num_folds)
valores_rmse <- numeric(num_folds)

# Se realiza el k-fold cross-validation
for (i in 1:num_folds) {
    # Se escogen los parámetros óptimos de la iteración i
    L_optimo <- kernel_ridge_resultado$L[i]
    lambda_optimo <- kernel_ridge_resultado$Lambda[i]

    # Se estandarizan los predictores en el conjunto completo de datos
    datos_s$pointX <- standardize(datos_s$pointX)
    datos_s$pointZ <- standardize(datos_s$pointZ)

    # Se divide el conjunto de datos en conjunto de entrenamiento y prueba
    train_data <- datos_s[-folds[[i]], ]
    test_data <- datos_s[folds[[i]], ]

    # Se estandarizan los predictores en ambos conjuntos
    test_data$pointX <- standardize(test_data$pointX)
    test_data$pointZ <- standardize(test_data$pointZ)

    # Se calcula la matriz de kernel para el conjunto de entrenamiento
    n_train <- nrow(train_data)
    K_train <- matrix(0, n_train, n_train)
    # Se itera sobre cada par de observaciones
    for (j in 1:n_train) {
        for (k in 1:n_train) {
            K_train[j, k] <- kernel(
                c(train_data$pointX[j], train_data$pointZ[j]), # Vector u
                c(train_data$pointX[k], train_data$pointZ[k]), L_optimo # Vector v
            )
        }
    }

    # Se ajusta el modelo de Kernel Ridge Regression
    alpha <- solve(K_train + lambda_optimo * diag(n_train), train_data$pointY)

    # CS e calcula la matriz de kernel entre datos de prueba y entrenamiento
    n_test <- nrow(test_data)
    K_test <- matrix(0, n_test, n_train)
    # Se itera sobre cada par de observaciones
    for (j in 1:n_test) {
        for (k in 1:n_train) {
            K_test[j, k] <- kernel(
                c(test_data$pointX[j], test_data$pointZ[j]), # Vector u
                c(train_data$pointX[k], train_data$pointZ[k]), L_optimo # Vector v
            )
        }
    }

    # Se predicen los valores de la variable respuesta en el conjunto de prueba
    prediccion <- K_test %*% alpha

    # Se calcula R^2
    R_2 <- cor(prediccion, test_data$pointY)^2

    # Se calcula RMSE
    rmse <- sqrt(mean((prediccion - test_data$pointY)^2))

    # Se guardan los resultados
    valores_R_2[i] <- R_2
    valores_rmse[i] <- rmse
}

# Se crea un gráfico boxplot para R^2
boxplot(valores_R_2, main = "Distribución de R-squared en Cross-Validation (Kernel Ridge)", ylab = "R-squared")

# Se crea un gráfico boxplot para RMSE
boxplot(valores_rmse, main = "Distribución de RMSE en Cross-Validation (Kernel Ridge)", ylab = "RMSE")
