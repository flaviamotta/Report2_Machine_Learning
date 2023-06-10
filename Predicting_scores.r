library(readr)
library(dplyr)
library(tidyverse)
library(magrittr)
library(ggplot2)
library(glmnet)
library(xtable)
library(tm)
library(FNN)
library(xgboost)
library(keras)
library(KRMM)
library(ggpubr)

str(data)
data <-read_csv("D:/__Mestrado/2021-semestre2/
_Aprendizado demáquina/Lista2/TMDb_updated.CSV",
col_names =TRUE)
str(data)

# Disregarding films that do not have ratings
# We excluded films that did not receive any votes from our evaluation process. 
# Additionally, we disregarded terms that appeared in less than 50 evaluations 
# when constructing our term-document matrix.

# To split our data, we chose percentages of 60% for the training set, 20% for
# validation, and 20% for the test set. These choices were made considering the
# size of our dataset, which consisted of 9,757 observations after removing 
# observations without evaluations.

# To gain insights into the distribution of evaluations, we constructed a histogram,
# as shown in Figure 1. Furthermore, Table 1 presents several descriptive measures that
# provide an overview of the evaluation distribution.


data %<>%filter(!vote_count%in%0)
summary(data$vote_average)
ggplot(data, aes(x=vote_average))+
geom_histogram(color="white",fill="hotpink")+
theme_classic(base_size=18)+
scale_x_continuous(breaks=seq(from=0,to=10,by=1),limits=c(0,10))+
xlab("Avaliaçãomédia")+
ylab("Frequência")

#Letter a

# Creating the Term Matrix

corp <-data$overview%>%
VectorSource()%>%
VCorpus(readerControl=list(language="en"))

dtm <-DocumentTermMatrix(corp,
control =list(tolower=TRUE,
stemming =FALSE,
removeNumbers=TRUE,
removePunctuation=TRUE,
removeStripwhitespace=TRUE,
bounds=list(global=c(50,Inf)),
weighting =weightTf))

dtmMatrix <-sparseMatrix(i=dtm$i,j=dtm$j,x=dtm$v,
dimnames =list(NULL,dtm$dimnames[[2]]),
dims =c(dtm$nrow,dtm$ncol))

dim(dtmMatrix)#[1]9757  999/considerando 999 palavras
set.seed(123)

#Division 60%training,20%validatione20%test.

data_split <-sample(c("Treino","Validacao","Teste"),
size =nrow(data),
prob =c(0.6,0.2,0.2),
replace =TRUE)


# Letter b
# The FNN package's knn.reg function already incorporates cross-validation when
# the test argument is not provided. Thus, we utilize this function to determine
# the optimal value for k. Figure 9 displays the association between various
# nearest neighbors and the estimated risk.

k.grid <-round(seq(1,100,length.out=20))

erro <-rep(NA,length(k.grid))
for (iiinseq_along(k.grid)){
predito <-knn.reg(train=dtmMatrix[!data_split=="Teste",],
y=data$vote_average[!data_split=="Teste"],
k=k.grid[ii])$pred
erro[ii] <-mean((predito-data$vote_average[!data_split=="Teste"])^2)
}

plot(k.grid,erro)
d =data.frame(k.grid,erro)
ggplot(data=d)+geom_line(aes(x=d$k.grid,y=erro),color="hotpink")+xlab("K")+
ylab("Risco estimado")

best.k <-k.grid[which.min(erro)]
#[1] 84

predito <-knn.reg(train=dtmMatrix[!data_split=="Teste",],
test =dtmMatrix[data_split=="Teste",],
y=data$vote_average[!data_split=="Teste"],
k=best.k)$pred

risco <-list()
risco$knn <-(predito-data$vote_average[data_split=="Teste"])^2%>%mean
risco$knn
#[1] 0.8443181

erro_padrao <-list()
erro_padrao$knn<-sqrt(risco$knn/length(data$vote_average[data_split=="Teste"]))
erro_padrao$knn
#[1] 0.02077101

# Letter c

# In the context of linear regression using lasso, the optimal value of λ,
# determined through cross-validation, was found to be 0.01270997. The selection
# process for this tuning parameter is illustrated in Figure 3.

# Figure 4 depicts the most significant variables in the regression model. It is
# noteworthy that words commonly associated with horror movies, such as "blood",
# "horror", and "supernatural", tend to be linked to lower ratings. This observation
# aligns with the notion that horror films typically receive less favorable reception
# from both general audiences and specialized critics. Additionally, considering the
# relatively limited budgets often associated with this genre, it further supports the
# lower ratings.

# Conversely, words like "German" and "war" are associated with higher ratings. This
# association may stem from the fact that these words often appear in descriptions
# of feature films centered around World War II, a subject that carries significant
# prestige among the public and critics alike. Furthermore, words such as "director"
# and "friendship" are also linked to higher ratings.

y =data$vote_average[data_split=="Teste"]
#Treinando (Lasso já faz a validação cruzada internamente)
fitLinear.cv <-cv.glmnet(x=as.matrix(dtmMatrix[!data_split=="Teste",]),
y=data$vote_average[!data_split=="Teste"],
alpha =1) 

#optimal value of λ
lambda =fitLinear.cv$lambda.min
lambda 
#[1]0.01270997

plot(fitLinear.cv)

#Adjusting lasso with the optimal value of λ
lasso =glmnet(x=dtmMatrix[!data_split=="Teste",],
y =data$vote_average[!data_split=="Teste"],
alpha =1,lambda=lambda)
coefficients(lasso)
pred_lasso <-predict(lasso,newx=dtmMatrix[data_split=="Teste",])
risco$lasso <-(pred_lasso-data$vote_average[data_split=="Teste"])^2%>%mean
risco$lasso #0.7884528
erro_padrao$lasso<-sqrt(risco$lasso/length(data$vote_average[data_split=="Teste"]))
erro_padrao$lasso #0.02007208

table(coef(fitLinear.cv,s=lambda)[,1]!=0)

coefs_estimates<-coef(fitLinear.cv,s=lambda)
coefs <-coefs_estimates%>%as.matrix%>%as_tibble
names(coefs) <-"Estimativa"
coefs %>%mutate(variavel=rownames(coefs_estimates))%>%arrange(desc(abs(Estimativa)))

theme_set(theme_gray(base_size=18))
coefs_estimates<-coef(fitLinear.cv,s=lambda)

coefs_estimates=data.frame(Palavra=rownames(coefs_estimates),
                           Coeficientes =coefs_estimates[,1])

coef_pos =coefs_estimates%>%arrange(desc(Coeficientes))
coef_neg =coefs_estimates%>%arrange(Coeficientes)

graf_pos =ggplot(data=coef_pos[2:30,],
          aes(x =reorder(Palavra,Coeficientes), y =Coeficientes))+
          geom_bar(stat="identity",col="white",fill="dodgerblue1")+coord_flip()+
          xlab("") +ylab("Estimativa")+ ggtitle("CoefPositivos")

graf_neg =ggplot(data=coef_neg[1:30,],
         aes(x =reorder(Palavra,-Coeficientes), y =Coeficientes))+
         geom_bar(stat="identity",col="white",fill="firebrick4")+coord_flip()+
         xlab("Termo")+ylab("Estimativa")+ ggtitle("CoefNegativos")
         
ggarrange(graf_neg,graf_pos,ncol=2,nrow=1)

#Letter d

# Table 4 displays the values obtained for the Random Forest model, including the
# estimated risk and its corresponding standard error. Figure 5 presents a graph
# illustrating the importance of the covariates as estimated by the random forest.

# One notable observation is that the terms identified as the most important for 
# classification by the random forest, such as "the," "and," and "his," do not 
# appear among the most relevant coefficients estimated by the lasso. Conversely,
# words like "war," "evil," and "story" are identified as important terms by both models.

# It is interesting to note that the most influential words, as determined by the
# random forest classification, are auxiliary terms that are not necessarily associated
# with a specific genre of film or even adjectives with a positive or negative connotation,
# as opposed to the importance of each term as indicated by the lasso. Hence, based on 
# this observation, we consider the lasso model to be the best estimated model for 
# this particular database.

dtmMatriz <-data.frame(as.matrix(dtmMatrix))
dtmMatrixArvore<-cbind(Vote=data$vote_average,dtmMatriz)
names(dtmMatrixArvore)<-make.names(names(dtmMatrixArvore))

floresta <-ranger(Vote~.,data=dtmMatrixArvore[!data_split=="Teste",],
importance ="impurity")

predito_floresta<-predict(floresta, data =dtmMatrixArvore[data_split=="Teste",])

risco$floresta<-
     (predito_floresta$predictions-data$vote_average[data_split=="Teste"])^2%>%mean
risco$floresta#0.7895257

erro_padrao$floresta<-
      sqrt(risco$floresta/length(data$vote_average[data_split=="Teste"]))
erro_padrao$floresta#0.02008573

importances <-tibble(variable=names(importance(floresta)),
                     importance=importance(floresta))%>%
arrange(desc(importance))

ggplot(importances%>%top_n(n=30), aes(x=reorder(variable,importance), y=importance))+
  geom_bar(stat="identity",position="dodge")+coord_flip()+
  ylab("ImportânciadaVariável")+xlab("")+
  ggtitle("InformationValueSummary")+
  guides(fill=F)+
  scale_fill_gradient(low="red",high="blue")
  
  # Letter e
  
#In the context of boosting, the best value obtained for the tuning parameter, 
# using early-stopping set to 15, was 869. The selection process for determining
# the number of iterations is depicted in Figure 6. Figure 7 presents the graph 
# illustrating the importance of the covariates as estimated by boosting.

# It is noteworthy that the three most important terms for classification, 
# according to boosting, are the same as those identified by the random forest
# model, namely "the," "and," and "his." As previously discussed, these terms
# are auxiliary in nature and do not possess specific associations with any 
# particular film genre.

# While the most significant coefficients estimated by the lasso model did not
# demonstrate comparable importance in boosting, words such as "war" and "story"
# displayed some relevance across all three methodologies analyzed.


xgb_cv =xgb.cv(data=dtmMatrix[!data_split=="Teste",],
               label =data$vote_average[!data_split=="Teste"],
               nrounds =1000, nfold =5, eta =0.01, early_stopping_rounds=15, verbose =F)
xgb_cv
print(xgb_cv$best_iteration)
# Bestiteration:
# iter   train_rmse_mean   train_rmse_std   test_rmse_mean   test_rmse_std
# 869     0.7709064        0.003937863       0.9064264        0.02355329

d2 <-as.data.frame(xgb_cv$evaluation_log[c(1:884),c(1,4)])

ggplot(data=d2)+geom_line(aes(x=iter,y=test_rmse_mean),color="hotpink")+
   xlab("número deiterações")+ ylab("Risco estimado")
   
boosting <-xgb.train(data=xgb.DMatrix(dtmMatrix[!data_split=="Teste",],
           label =data$vote_average[!data_split=="Teste"]),
           eta =0.01,nrounds=869)
           
pred.boosting<-predict(boosting,dtmMatrix[data_split=="Teste",])

risco$boosting<-(pred.boosting-data$vote_average[data_split=="Teste"])^2%>%mean
risco$boosting#0.7958238

erro_padrao$boosting<-sqrt(risco$boosting/length(data$vote_average[data_split=="Teste"]))
erro_padrao$boosting#0.02016569

xgb.imp <-xgb.importance(model=boosting)
xgb.ggplot.importance(xgb.imp)

xgb.imp <-xgb.importance(model=boosting)

importance.boost<-tibble(variable=xgb.imp$Feature,
importance=xgb.imp$Importance)%>%
arrange(desc(importance))

ggplot(importance.boost%>%top_n(n=30), aes(x=reorder(variable,importance),
       y=importance))+
       geom_bar(stat="identity",position="dodge")+coord_flip()+
       ylab("ImportânciadaVariável")+xlab("")+
       ggtitle("InformationValueSummary")+
       guides(fill=F)
       
# Letter f

# The early-stopping neural network was fine-tuned, and the risk was estimated
# on the test set. In addition, drop-out was added to the network to examine if
# performance improved.

# Table 6 showcases the results obtained for risk and standard error from 
# implementing the models without drop-out on the test set. It is evident 
# that the performance indeed improved with the inclusion of drop-out in 
# the model.

# It is worth noting that the neural network was constructed with three 
# hidden layers: the first layer comprised eight neurons, the second layer
# had ten neurons, and the third layer had five neurons. Throughout these
# layers, the Rectified Linear Unit (ReLU) activation function was employed.
# ReLU is a linear function for positive inputs but returns zero for negative 
# inputs.
  
modelo_sem_dropout<-keras_model_sequential()%>%
  layer_dense(units=8,activation="relu",
              input_shape =dim(dtmMatrix[!data_split=="Teste",])[2])%>%
  layer_dense(units=10,activation="relu")%>%
  layer_dense(units=5,activation="relu")%>%
  layer_dense(units=1)

modelo_com_dropout<-keras_model_sequential()%>%
  layer_dense(units=8,activation="relu",
              input_shape =dim(dtmMatrix[!data_split=="Teste",])[2])%>%
  layer_dropout(0.2)%>%
  layer_dense(units=10,activation="relu")%>%
  layer_dropout(0.2)%>%
  layer_dense(units=5,activation="relu")%>%
  layer_dense(units=1)

modelo_sem_dropout%>%compile(
  loss ="mse",
  optimizer =optimizer_rmsprop(),
  metrics =list("mean_absolute_error")#avaliaçãodesempenho
)

modelo_com_dropout%>%compile(
  loss ="mse",
  optimizer =optimizer_rmsprop(),
  metrics =list("mean_absolute_error")#avaliaçãodesempenho
)

historico_sem_dropout<-modelo_sem_dropout%>%fit(
  dtmMatrix[!data_split=="Teste",],
  data$vote_average[!data_split=="Teste"],
  epochs =100,
  validation_split=0.2,
  callbacks=callback_early_stopping(monitor="val_loss",
                                    patience=30,mode="auto"),
  verbose =TRUE
)

historico_com_dropout<-modelo_com_dropout%>%fit(
  dtmMatrix[!data_split=="Teste",],
  data$vote_average[!data_split=="Teste"],
  epochs =100,validation_split=0.2,
  callbacks=callback_early_stopping(monitor="val_loss",patience=30,mode="auto"),
  verbose =TRUE
)

plot(historico_sem_dropout,metrics="loss")+
  theme_bw() +theme(legend.position="top")
plot(historico_com_dropout,metrics="loss")+
  theme_bw() +theme(legend.position="top")
set.seed(123)

predito_com <-modelo_com_dropout%>%predict(as.matrix(dtmMatrix[data_split=="Teste",]))
predito_sem <-modelo_sem_dropout%>%predict(as.matrix(dtmMatrix[data_split=="Teste",]))

risco$nnets_sem<-(predito_sem-data$vote_average[data_split=="Teste"])^2%>%mean()
risco$nnets_sem
erro_padrao$nnets_sem<-sqrt(risco$nnets_sem/length
                            (data$vote_average[data_split=="Teste"]))

erro_padrao$nnets_sem

risco$nnets_com<-(predito_com-data$vote_average[data_split=="Teste"])^2%>%mean()
risco$nnets_com
erro_padrao$nnets_com<-sqrt(risco$nnets_com/
                              length(data$vote_average[data_split=="Teste"]))

erro_padrao$nnets_com

# Letter g
# In the case of kernel ridge regression, the optimal value for λ was determined
# to be 0.14743. Figure 8 illustrates the process of selecting this value.

matriz <-as.matrix(cbind(1,dtmMatrix))
require(remotes)
#sigma=0.015

lambda.grid <-round(seq(0.016,.2,length.out=15),5)
erro.krr =rep(NA,length(lambda.grid))

for (iiinseq_along(lambda.grid)){
  ajuste_kernel=krr::krr(as.matrix(matriz[data_split=="Treino",]),
                         data$vote_average[data_split=="Treino"],
                         lambda.grid[ii],sigma=0.015)
  predito_kernel=predict(ajuste_kernel,
                         xnew =as.matrix(matriz[data_split=="Validacao",]))
  erro.krr[ii] =mean((predito_kernel-data$vote_average[data_split=="Validacao"])^2)
}

plot(lambda.grid,erro.krr)
best.lambda =lambda.grid[which.min(erro.krr)]
best.lambda #Best tuning paramenter 0.14743

d3 <-data.frame(lambda=lambda.grid,mse=erro.krr)

ggplot(data=d3)+geom_line(aes(x=lambda,y=mse),color="hotpink")+
  xlab("lambda")+
  ylab("Risco estimado")
ajuste_kernel_best=krr::krr(as.matrix(matriz[amostrinha=="am1",]),
                            data$vote_average[amostrinha=="am1"],
                            best.lambda, sigma=0.015)

predito_kernel_best=predict(ajuste_kernel_best,
                            xnew =as.matrix(matriz[data_split=="Teste",]))

risco$krr <-(predito_kernel_best-data$vote_average[data_split=="Teste"])^2%>%mean
risco$krr #1.129073

erro_padrao$krr<-sqrt(risco$krr/length(data$vote_average[data_split=="Teste"]))
erro_padrao$krr #0.02401959


# Letter h

# Next, we present the graphs comparing predicted values versus observed
# values in the test set. Visually, it is apparent that the KNN model and
# the random forests model achieved configurations that displayed a closer
# alignment between the points and the identity line. However, it is worth
# noting that even in these models, the observations did not exhibit the 
# desired level of concentration around the line.

#KNN
knn <-data.frame(predito,y)
ggplot(data=knn)+geom_point(aes(x=predito,y=y),color="hotpink")+
  xlab("ValoresPreditos")+
  ylab("ValoresObservados")+geom_abline(intercept=0,slope=1)+ggtitle("KNN")
plot(predito,data$vote_average[data_split=="Teste"],cex=0.4,
     ylab ="valoresobservados")
abline(a=0,b=1,lwd=3,col=2)

#LASSO
lasso <-data.frame(pred_lasso,y)
ggplot(data=lasso)+geom_point(aes(x=pred_lasso,y=y),color="hotpink")+
  xlab("ValoresPreditos")+
  ylab("ValoresObservados")+geom_abline(intercept=0,slope=1)+ggtitle("LASSO")
plot(pred_lasso,data$vote_average[data_split=="Teste"],cex=0.4,ylab=
       "valores observados")
abline(a=0,b=1,lwd=3,col=2)

#Random Forest
floresta <-data.frame(pred=predito_floresta$predictions,y)
ggplot(data=floresta)+geom_point(aes(x=pred,y=y),color="hotpink")+
  xlab("ValoresPreditos")+
  ylab("ValoresObservados")+geom_abline(intercept=0,slope=1)+
  ggtitle("FlorestaAleat?ria")
plot(floresta$pred,data$vote_average[data_split=="Teste"],cex=0.4,ylab=
       "valores observados")
abline(a=0,b=1,lwd=3,col=2)

#Boosting
dboosting <-data.frame(pred=pred.boosting,y)
ggplot(data=dboosting)+geom_point(aes(x=pred,y=y),color="hotpink")+
  xlab("ValoresPreditos")+
  ylab("ValoresObservados")+geom_abline(intercept=0,slope=1)+
  ggtitle("Boosting")
plot(dboosting$pred,data$vote_average[data_split=="Teste"],cex=0.4,ylab=
       "valores observados")
abline(a=0,b=1,lwd=3,col=2)

#Neural Network
neuralnet <-data.frame(predito_com,predito_sem,y)
ggplot(data=neuralnet)+geom_point(aes(x=predito_sem,y=y),color="hotpink")+
  xlab("ValoresPreditos")+
  ylab("ValoresObservados")+geom_abline(intercept=0,slope=1)+
  ggtitle("RedesNeuraissemdropout")
ggplot(data=neuralnet)+geom_point(aes(x=predito_com,y=y),color="hotpink")+
  xlab("ValoresPreditos")+
  ylab("ValoresObservados")+geom_abline(intercept=0,slope=1)+
  ggtitle("RedesNeuraiscomdropout")

#KRR
dkrr <-data.frame(pred=predito_kernel_best,y=y)
ggplot(data=dkrr)+geom_point(aes(x=pred,y=y),color="hotpink")+
  xlab("ValoresPreditos")+
  ylab("ValoresObservados")+geom_abline(intercept=0,slope=1)+ggtitle("KRR")
  
# CONCLUSION
# The lasso model exhibited the lowest estimated risk and the smallest associated
# standard error, as depicted in Table 8. Furthermore, when comparing the importance
# graphs of the covariates among the three models with the lowest risks (lasso, random
# forest, and boosting), the words identified by the lasso model as the most significant 
# for classification proved to be the most informative. These terms provided valuable 
# insights into the genres of movies associated with higher ratings and those associated
# with lower ratings, which could not be deduced from the importance graphs of the random
# forest or boosting models.

# Given that the objective is to utilize movie synopses as a proxy for their content 
# to predict the ratings they would receive, the lasso model not only yielded superior
# results but also presented coefficients that aligned with reality. Therefore, the
# lasso model emerged as the most suitable choice, providing accurate predictions while
# considering the importance of specific terms in the synopses.









