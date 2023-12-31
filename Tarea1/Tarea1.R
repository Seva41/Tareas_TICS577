# Métodos basados en Kernels para aprendizaje automático
# Tarea 1
# Integrantes: Sebastián Dinator, Cristóbal Quijanes, Mathias Reusser.

# Ejercicio 1

# 1)

vector <- matrix(c(seq(1, 20), seq(10, 100, 10), seq(31, 100)), 1, 100)
vector

# 2)
## 2.1)
A <- matrix(c(2, 5, -6, 1), 4, 1) # Matriz de 4x1
B <- matrix(c(1, 1, 1, 1), 4, 1) # Matriz de 4x1
A_t <- t(A) # Transpuesta de A

res <- A_t %*% B
res

## 2.2)

C <- matrix(c(-1, 3, -6, 1), 4, 1) # Matriz de 4x1
D <- matrix(c(1, -2, 3, -4), 4, 1) # Matriz de 4x1
E <- matrix(c(1, 0, 0, 2), 4, 1) # Matriz de 4x1

F <- C + D
F_t <- t(F)

res2 <- F_t %*% E # Calculo de la multiplicacion
res2

## 2.3)

G <- matrix(c(0, -1, 2, 0), 4, 1) # Matriz de 4x1
G_t <- t(G) # Transpuesta de G
H <- matrix(c(1, 8, -4, 1, 2, 7, -3, -1, 3, 6, -2, 1, 4, 5, -1, -1), 4, 4) # Matriz de 4x4
I <- matrix(c(1, 0, 0, 2), 4, 1) # Matriz de 4x1

J <- G_t %*% H # Calculo de la multiplicacion de G_t y H
res3 <- J %*% I # Calculo de la multiplicacion de J y I
res3

# 3)

v1 <- matrix(seq(1, 50), 1, 50) # Vector de 1x50
v2 <- matrix(seq(1, 500), 1, 500) # Vector de 1x500
v3 <- matrix(seq(1, 1000), 1, 1000) # Vector de 1x1000
v4 <- matrix(seq(1, 15000), 1, 15000) # Vector de 1x15000
v5 <- matrix(seq(1, 200000), 1, 200000) # Vector de 1x200000

norma <- function(v) { # Funcion para calcular la norma con for
  suma <- 0
  for (i in v) {
    suma <- suma + i^2
  }
  n <- sqrt(suma)
  return(n)
}

n1 <- norma(v1)
n2 <- norma(v2)
n3 <- norma(v3)
n4 <- norma(v4)
n5 <- norma(v5)

n1
n2
n3
n4
n5

# 4)

norma2 <- function(v) { # Funcion para calcular la norma sin for
  m <- sqrt(sum(v^2))
  return(m)
}

n1 <- norma2(v1)
n2 <- norma2(v2)
n3 <- norma2(v3)
n4 <- norma2(v4)
n5 <- norma2(v5)

n1
n2
n3
n4
n5

# 5)

varianza <- function(v) { # Funcion para calcular la varianza
  prom <- sum(v) / length(v)
  out <- sum((v - prom)^2) / (length(v) - 1) # Error en PDF, esta corregido en codigo
  return(out)
}

v6 <- matrix(c(3, 4, 3), 1, 3) # Vector de 1x3
output <- varianza(v6)
output

# 6) (sale como 5 en pdf)

factor <- function(e) { # Funcion para calcular el factorial
  new_e <- e
  while (e > 1) {
    new_e <- new_e * (e - 1)
    e <- e - 1
  }
  return(new_e)
}

fact_matr <- function(m) { # Funcion para calcular el factorial de una matriz
  elem <- NULL
  shape <- dim(m) # Dimensiones de la matriz

  for (i in m) {
    n_f <- factor(i)
    elem <- c(elem, n_f)
  }

  matriz <- matrix(elem, shape[1], shape[2])
  return(matriz)
}

mat_5 <- matrix(c(1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8), 4, 4)
res <- fact_matr(mat_5)
res

# 7) (sale como 6 en pdf)

func7 <- function(x, y, M) { # Funcion para calcular el resultado
  n <- length(x)
  m <- length(y)

  result <- 0

  for (l in 1:n) {
    suma_jk <- 0
    for (j in 1:m) {
      suma_k <- sum(y[j] * x[l] * x * M[l, j])
      suma_jk <- suma_jk + suma_k
    }

    result <- result + suma_jk
  }

  final <- (1 / n) * result
  return(final)
}

n <- 3
m <- 4
M <- matrix(0, n, m)

for (i in 1:n) { # Llenar la matriz M
  for (j in 1:m) {
    M[i, j] <- (i + j) * (i - j)
  }
}

x <- c(1, 2, 3)
y <- c(0.5, 0.7, 0.9)

res7 <- func7(x, y, M)
res7

###############################################
# Ejercicio 2

# 1)

kern_RQ <- function(x, y, sig, l, alpha) { # Funcion para calcular el kernel RQ
  dist <- sum((x - y)^2)
  par <- (dist / (2 * alpha * l)^2) + 1
  pot <- par^-alpha
  total <- sig^2 * pot

  return(total)
}

# 2)
alpha <- 1
sig <- 1
l <- 1
x <- seq(0, 10, length.out = 1000000) # Vector de 1x1000000
y <- numeric(length(x)) # Vector igual a x
for (i in seq_along(x)) { # Llenar el vector y
  y[i] <- kern_RQ(x[i], 0, sig, l, alpha)
}

plot(x, y, type = "l", main = paste("Alpha =", alpha), xlab = "||x-y||^2", ylab = "kern_RQ(x,y)") # Grafico

# 3)
alphas <- c(5, 10, 50)
for (alpha in alphas) { # Graficar para cada alpha
  y <- numeric(length(x))
  for (i in seq_along(x)) {
    y[i] <- kern_RQ(x[i], 0, sig, l, alpha)
  }
  plot(x, y, type = "l", main = paste("Alpha =", alpha), xlab = "||x-y||^2", ylab = "kern_RQ(x,y)") # Grafico
}

###############################################
# Ejercicio 3

# 1)

mult_trip <- function(x, y, A) { # Funcion para calcular la multiplicacion
  f <- length(x)
  x_t <- matrix(0.0, 1, f)

  for (i in seq_along(x)) { # Transpuesta de x
    x_t[1, i] <- x[i]
  }

  x_t_A <- matrix(0.0, 1, f)
  for (k in seq_along(f)) { # Multiplicacion de x_t y A
    sum_x <- 0
    for (j in seq_along(f)) {
      sum_x <- sum_x + (x_t[1, j] * A[j, k])
    }
    x_t_A[1, k] <- sum_x
  }

  res <- 0
  for (i in seq_along(f)) { # Multiplicacion de x_t_A y y
    res <- res + x_t_A[1, i] * y[i]
  }

  return(res)
}

# Datos de ejemplo
n <- 10000
x <- runif(n)
y <- runif(n)

A <- matrix(0.0, n, n)
for (i in 1:n) { # Llenar la matriz A
  for (j in 1:n) {
    A[i, j] <- runif(1)
  }
}

# Calculo con funcion mult_trip
start_time_MT <- system.time({
  mult_trip(x, y, A)
})

# Calculo con funcion %*%
start_time_T <- system.time({
  t(x) %*% A %*% y
})

# Resultados
print(paste("Tiempo de ejecucion MT:", start_time_MT[["elapsed"]], "seg"))
print(paste("Tiempo de ejecucion T:", start_time_T[["elapsed"]], "seg"))


print(paste("Tiempo de ejecucion MT:", start_time_MT[["elapsed"]], "seg"))
print(paste("Tiempo de ejecucion T:", start_time_T[["elapsed"]], "seg"))
dif <- start_time_MT[["elapsed"]] - start_time_T[["elapsed"]] # Diferencia de tiempos
format_diff <- sprintf("%.60f", dif)
print(paste("Diferencia:", format_diff))

n <- seq(100, 5000, by = 1)
b <- numeric(length(n))
c <- numeric(length(n))

for (i in seq_along(n)) { # Calcular el tiempo de ejecucion para cada n
  x <- runif(n[i])
  y <- runif(n[i])

  A <- matrix(runif(n[i] * n[i]), n[i], n[i])

  start_time_MT <- system.time({
    mult_trip(x, y, A)
  })
  b[i] <- start_time_MT[["elapsed"]]

  start_time_T <- system.time({
    t(x) %*% A %*% y
  })
  c[i] <- start_time_T[["elapsed"]]
}

par(mfrow = c(1, 2)) # Dividir la ventana gráfica en 1x2

plot(n, b, type = "l", main = "Tiempo de Ejecución MT", xlab = "n", ylab = "Tiempo de Ejecución")
plot(n, c, type = "l", main = "Tiempo de Ejecución T", xlab = "n", ylab = "Tiempo de Ejecución")


# Ejercicio 4

# a)

library(openintro)
starbucks2 <- starbucks[, c("calories", "fat", "carb", "fiber", "protein")] # Seleccionar columnas
starbucks3 <- as.matrix(starbucks2) # Convertir a matriz
head(starbucks3) # Ver los primeros 6 elementos

z <- starbucks3
c <- 1
G <- t(z) %*% z + c # Calcular la matriz
G

# b)

o <- 2
l <- 0.5

# Calcular la matriz de Gram
n <- nrow(z) # Filas en Z
G <- matrix(0, n, n)

for (i in 1:n) { # Llenar la matriz G
  for (j in 1:n) {
    dif <- z[i, ] - z[j, ] # diferencia entre los vectores x e y
    G[i, j] <- o^2 * exp(-1 / (l^2) * t(dif) %*% dif) # valor de K(x, y)
  }
}

G

###############################################
# Ejercicio 5

# 1)


# 2)
