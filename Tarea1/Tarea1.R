# Métodos basados en Kernels para aprendizaje automático
# Tarea 1

#Ejercicio 1

# 1)
vector <- c(seq(1,20),seq(10,100,10),seq(31,100))

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
