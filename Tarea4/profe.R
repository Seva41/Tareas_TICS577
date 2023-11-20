# Llamamos a las librerias que vamos a utilizar
library('kernlab')
library('caret')

## Leemos los datos
df=read.csv(file.choose(),header=TRUE)
attach(df)
#Creamos la variable case que asigna un 1 si el dato es de tipo breakfast
clase=(df$Category=='Breakfast')*1
df['clase']=clase

#Muestreo estratificado
df_Break=df[which(clase==1),]
df_Otros=df[-which(clase==1),]

n_Break=dim(df_Break)[1]
n_Otros=dim(df_Otros)[1]

index_train_Break=sample(1:4,n_Break,replace=TRUE)
index_train_Otros=sample(1:4,n_Otros,replace=TRUE)
table(index_train_Break)
table(index_train_Otros)

df[which(clase==1),'kfold']=index_train_Break
df[-which(clase==1),'kfold']=index_train_Otros

fix(df)

C.grid=seq(0.01,5,length.out=30)
sigma.grid=seq(0.1,2,length.out=30)
param.grid=as.matrix(expand.grid(C.grid,sigma.grid))
dim(param.grid)

#Resultados
Accuracy.matrix=matrix(0,nrow=dim(param.grid)[1],ncol=4)

#ciclo for asociado a cross-validation
for(k in 1:4)
{
  index_test=df$kfold==k
  df_test=df[index_test,]
  row.names(df_test)=NULL
  
  df_train=df[-index_test,]
  row.names(df_train)=NULL
  
  #Estandarización: Ojo en la tarea se deberian usar todas las varibales numericas
  columnas=c('Calories','Total.Fat')
  scaler=preProcess(df_train[,columnas],method=c('center','scale'))
  df_train_scaled=predict(scaler,df_train)
  df_test_scaled=predict(scaler,df_test)
  
  #Clasificación usando set de entrenamiento
  y_train=df_train_scaled$clase*2-1
  X_train=as.matrix(df_train_scaled[,columnas])
  
  y_test=df_test_scaled$clase*2-1
  X_test=as.matrix(df_test_scaled[,columnas])
  
  for(j in 1:dim(param.grid)[1])
  {
    if(j%%5==0)print(paste('Iter',j))
    C_param=param.grid[j,1]
    sigma_param=param.grid[j,2]
    
    #KSVM1=ksvm(y=as.factor(y_train),x=X_train,kernel='vanilladot',C=C_param)
    
    Kfun=function(x,y,len=sigma_param)
    {
      exp(-sum((x-y)^2)/len^2)
    }
    class(Kfun)='kernel'
    KSVM1=ksvm(y=as.factor(y_train),x=X_train,kernel=Kfun,C=C_param)
    
    #Predicción
    p1=predict(KSVM1,X_test)
    
    CM=table(y_test,p1)
    Accuracy.matrix[j,k]=sum(diag(CM))/sum(CM)
  }
}


plot(C.grid,Accuracy.matrix[,1])
