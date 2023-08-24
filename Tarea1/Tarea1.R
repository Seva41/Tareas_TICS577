# Métodos basados en Kernels para aprendizaje automático
# Tarea 1

#Ejercicio 1

# 1)

vector <- matrix(c(seq(1,20),seq(10,100,10),seq(31,100)),1,100)
vector

# 2)
## 2.1)
A <- matrix(c(2,5,-6,1),4,1)
B <- matrix(c(1,1,1,1),4,1)
A_t <- t(A)

res <- A_t %*% B
res

## 2.2)

C <- matrix(c(-1,3,-6,1),4,1)
D <- matrix(c(1,-2,3,-4),4,1)
E <- matrix(c(1,0,0,2),4,1)

F <- C + D
F_t <- t(F)

res2 <- F_t %*% E
res2

## 2.3)

G <- matrix(c(0,-1,2,0),4,1)
G_t <- t(G)
H <- matrix(c(1,8,-4,1,2,7,-3,-1,3,6,-2,1,4,5,-1,-1),4,4)
I <- matrix(c(1,0,0,2),4,1)

J <- G_t %*% H
res3 <- J %*% I
res3

# 3)

v1 <- matrix(seq(1,50),1,50)
v2 <- matrix(seq(1,500),1,500)
v3 <- matrix(seq(1,1000),1,1000)
v4 <- matrix(seq(1,15000),1,15000)
v5 <- matrix(seq(1,200000),1,200000)

norma <- function(v) {
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

norma2 <- function(v) {
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

f5 <- function(v) {
  
  return()
}