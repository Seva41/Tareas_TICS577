rm(list=ls())
df=read.csv(file.choose(),header=TRUE)

n=dim(df)[1]

#creamos etiquetas
unique(df$Category)
clase=df$Category=="Breakfast"
table(clase)/n*100
index_breakfast=which(df$Category=="Breakfast")
df_Breakfast=df[df$Category=="Breakfast",]
df_Otros=df[-index_breakfast,]
dim(df_Breakfast)
dim(df_Otros)
kfold_B=sample(1:4,dim(df_Breakfast)[1],replace=TRUE)
kfold_O=sample(1:4,dim(df_Otros)[1],replace=TRUE)
table(kfold_B)
table(kfold_O)
df[index_breakfast,"Kfold"]=kfold_B
df[-index_breakfast,"Kfold"]=kfold_O

fix(df)
df["class"]=2*clase-1
colnames(df)
columnas=c("Category","Item","Serving.Size")# agregar el resto
#para primera iteracion de kfold=2,3,4

# borrar columnas que no sirven

X=df[,-c(1,2,3)]

fix(X)
columnas=colnames(X)
columnas=columnas[-23]
columnas=columnas[-22]
target="class"
library("caret")
library("kernlab")

# pregunta 1 a

# Vanilla dot
# en este caso exploraremos la grilla de parametros para C
# haremos simulaciones con 1000 parametros y eligiremos el valor de C que se ajusta mejor a cada metrica
grilla<-seq(1,1000,1)
n1=length(grilla)

acg<-numeric(n1)
f1g<-numeric(n1)
for (j in grilla){
  
  print(j)
  ac<-numeric(4)
  f1<-numeric(4)
  for (i in 1:4){
    index_test=X$Kfold==i
    df_test=X[index_test,]
    row.names(df_test)=NULL
    df_train=X[-index_test,]
    row.names(df_train)=NULL
    #estandarizar
    scaler=preProcess(df_train[,columnas],method=c("center","scale"))
    df_train_scaled=predict(scaler,df_train)
    df_test_scaled=predict(scaler,df_test)
    # super vector machine
    
    Ksvm = ksvm(y=as.factor(df_train_scaled[,target]),x=as.matrix(df_train_scaled[,columnas]),kernel='vanilladot',scaled=FALSE,C=j)
    
    Y_pred=predict(Ksvm,df_test_scaled[,columnas])
    
    Table=table(df_test_scaled[,target],Y_pred)
    
    Accuracy=(Table[1,1]+Table[2,2])/sum(Table)
    Accuracy
    ac[i]=Accuracy
    #~~~~~~~~~~#
    #Precision #
    #~~~~~~~~~~#
    Precision=Table[2,2]/(Table[2,1]+Table[2,2])
    
    #~~~~~~~~~#
    #Recall   #
    #~~~~~~~~~#
    Recall=Table[2,2]/(Table[1,2]+Table[2,2])
    
    #~~~~~~~~~#
    #F1-score #
    #~~~~~~~~~#
    F1=2*(Recall*Precision)/(Recall+Precision)
    f1[i]=F1
  }
  acg[j]=mean(ac)
  f1g[j]=mean(f1)
  
}

answer<-data.frame("C"=grilla,
                   "acurracy"=unlist(acg),
                   "f1"=unlist(f1g))
index1=which(answer$f1==max(answer$f1))[1]
index1a=which(answer$acurracy==max(answer$acurracy))[1]
answer[index1a,]
answer[index1,]


# parte 1b
# kernel squared 
grillab<-seq(1,10,1)
lgrillab<-seq(1,100,1)
parm<-expand.grid(grillab,lgrillab)
n2=length(parm[,1])
# segunda aplicacion SVM
acg<-numeric(n2)
f1g<-numeric(n2)
Kfun = function(x,y){
  norm2 = sum((x-y)^2)
  exp(-norm2/l^2)
}
class(Kfun)='kernel'


for (k in 1:n2){
  c<-parm[k,1]
  l<-parm[k,2]
  ac<-numeric(4)
  f1 <- numeric(4)
  print(k)
  print(l)
  for (i in 1:4){
    index_test=X$Kfold==i
    df_test=X[index_test,]
    row.names(df_test)=NULL
    df_train=X[-index_test,]
    row.names(df_train)=NULL
    #estandarizar
    scaler=preProcess(df_train[,columnas],method=c("center","scale"))
    df_train_scaled=predict(scaler,df_train)
    df_test_scaled=predict(scaler,df_test)
    # super vector machine
    
    
    Ksvm = ksvm(y = as.factor(df_train_scaled[, target]), x = as.matrix(df_train_scaled[, columnas]), kernel = Kfun,l=l, scaled = FALSE, C = c)
    
    Y_pred=predict(Ksvm,df_test_scaled[,columnas])
    
    Table=table(df_test_scaled[,target],Y_pred)
    
    Accuracy=(Table[1,1]+Table[2,2])/sum(Table)
    Accuracy
    ac[i]=Accuracy
    #~~~~~~~~~~#
    #Precision #
    #~~~~~~~~~~#
    Precision=Table[2,2]/(Table[2,1]+Table[2,2])
    
    #~~~~~~~~~#
    #Recall   #
    #~~~~~~~~~#
    Recall=Table[2,2]/(Table[1,2]+Table[2,2])
    
    #~~~~~~~~~#
    #F1-score #
    #~~~~~~~~~#
    F1=2*(Recall*Precision)/(Recall+Precision)
    f1[i]=F1
  }
  acg[k]=mean(ac)
  f1g[k]=mean(f1)
}

answer2<-data.frame("C"=parm[,1],
                    "l"=parm[,2],
                    "acurracy"=acg,
                    "f1"=f1g)
answer2[is.na(answer2)] <- 0
index2=which(answer2$f1==max(answer2$f1))[1]
answer2[index2,]


index2a=which(answer2$acurracy==max(answer2$acurracy))[1]
answer2[index2a,]





grillac <- seq(1, 5, 1)
lgrillac <- seq(1, 5, 1)
ogrillcz<-seq(1, 5, 1)
pgrillcz<-seq(1, 5, 1)
parm2 <- expand.grid(grillab, lgrillab,ogrillcz,pgrillcz)
n3 <- length(parm2[, 1])


acg <- numeric(n3)
f1g <- numeric(n3)
#Kernel locally periodic:
Kfun2=function(x,y){
  exp(-2*(sin(pi*sum(abs(x-y))/p_val)^2)/ls_par^2)*exp(-sum(abs(x-y)/sigm^2))
}

class(Kfun2)='kernel'

for (k in 1:n3) {
  c <- parm2$Var1[k]
  ls_par <- parm2$Var2[k]
  sigm <- parm2$Var3[k]
  p_val <- parm2$Var4[k]
  ac <- numeric(4)
  f1 <- numeric(4)
  print(k)
  for (i in 1:4) {
    index_test <- X$Kfold == i
    df_test <- X[index_test, ]
    row.names(df_test) <- NULL
    df_train <- X[!index_test, ]
    row.names(df_train) <- NULL
    
    scaler <- preProcess(df_train[, columnas], method = c("center", "scale"))
    df_train_scaled <- predict(scaler, df_train)
    df_test_scaled <- predict(scaler, df_test)
    
    # Establecer el valor de l globalmente
    
    
    ls_par<<-ls_par
    sigm<<-sigm
    p_val<<-p_val
    
    Ksvm <- ksvm(y = as.factor(df_train_scaled[, target]), 
                 x = as.matrix(df_train_scaled[, columnas]), 
                 kernel = Kfun2,
                 scaled = FALSE, C = c)
    
    Y_pred <- predict(Ksvm, df_test_scaled[, columnas])
    
    Table <- table(df_test_scaled[, target], Y_pred)
    
    Accuracy <- (Table[1, 1] + Table[2, 2]) / sum(Table)
    ac[i] <- Accuracy
    
    Precision <- Table[2, 2] / (Table[2, 1] + Table[2, 2])
    Recall <- Table[2, 2] / (Table[1, 2] + Table[2, 2])
    F1 <- 2 * (Recall * Precision) / (Recall + Precision)
    f1[i] <- F1
  }
  
  acg[k] <- mean(ac)
  f1g[k] <- mean(f1)
}



answer3<-data.frame("C"=parm2[,1],
                    "ls"=parm2[,2],
                    "sigma"=parm2[,3],
                    "p"=parm2[,4],
                    "acurracy"=acg,
                    "f1"=f1g)
