#Leer datos
rm(list=ls())

df=read.csv(file.choose(),header = TRUE)
attach(df)
n=dim(df)[1]

#Creamos etiquetas para cross validation
unique(df$Category)

clase=df$Category=='Breakfast'
table(clase)/n*100

index_breakfast=which(df$Category=='Breakfast')
df_Breakfast=df[index_breakfast,]
df_Otras=df[-index_breakfast,]

dim(df_Breakfast)
dim(df_Otras)

kfold_B=sample(1:4,dim(df_Breakfast)[1],replace=TRUE)
kfold_O=sample(1:4,dim(df_Otras)[1],replace=TRUE)

table(kfold_B)
table(kfold_O)

df[index_breakfast,'Kfold']=kfold_B
df[-index_breakfast,'Kfold']=kfold_O

# Creamos la columna clase
df['class']=2*clase-1

colnames(df)

# Para la primera iteracion de kfold
# Set de testeo kfold=1; Set de entrenamiento kfold=2,3,4.
columnas=c("Calories","Total.Fat")
index_test=df$Kfold==1

df_test=df[index_test,]
row.names(df_test)=NULL
df_train=df[-index_test,]
row.names(df_train)=NULL

# Estandarizar
library(caret)

scaler=preProcess(df_train[,columnas],method=c("center","scale"))
df_train_scaled=predict(scaler,df_train)
df_test_scaled=predict(scaler,df_test)


# Vanilla SVM
library(kernlab)

y_train=df_train_scaled$class
x_train=as.matrix(df_train_scaled[,columnas])

y_test=df_test_scaled$class
x_test=as.matrix(df_test_scaled[,columnas])

KSVM1=ksvm(y=as.factor(y_train),x=x_train,kernel="vanilladot",C=1) #Hacer for para C
plot(KSVM1)

p1=predict(KSVM1,x_test)
table(y_test,p1)
table(y_test)

# Kernel SVM
Kfun=function(x,y,len=1){
  exp(-sum((x-y)^2)/len^2)
}
class(Kfun)="kernel"

KSVM2=ksvm(y=as.factor(y_train),x=x_train,kernel=Kfun,C=1) #Hacer for para C
plot(KSVM2)

p2=predict(KSVM2,x_test)
CM=table(y_test,p2)
table(y_test)

sum(diag(CM)/sum(CM))

alphay=rep(0,dim(x_train)[1])
alphay[KSVM2@alphaindex[[1]]]=KSVM2@coef[[1]]

KGram=matrix(0,nrow=dim(x_train)[1],ncol=dim(x_train)[1])
for(i in 1:dim(x_train)[1]){
  for(j in 1:dim(x_train)[1]){
    KGram[i,j]=Kfun(x_train[i,],x_train[j,])
  }
}

g=KGram%*%alphay-KSVM2@b #Encontrar 15% de datos con menores valores de g

quantile(sort(abs(g)),0.15)

Order_index=order(abs(g))[1:10]

calories_dificil=df_train_scaled$Calories[Order_index]
Fat_dificil=df_train_scaled$Total.Fat[Order_index]

ggplot()+
  geom_point(aes(x=df_train_scaled$Calories,y=df_train_scaled$Total.Fat,colour=df_train_scaled$class))+
  geom_point(aes(x=calories_dificil,y=Fat_dificil),colour='red',size=3)
