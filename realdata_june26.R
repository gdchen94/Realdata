library(ggplot2) # Data visualization
library(readr) # CSV file I/O, e.g. the read_csv function
library(tidyverse)
library(data.table)
library(dplyr)
library(stringr)
library(DT)
library(tidyr)
library(corrplot)
library(leaflet)
library(lubridate)
library(lme4)
library(caret)
library(knn)
source("Dcorr.R")
source("SubgraphScreen.R")
source("graph_generate.R")
source("bayes_plugin.R")
source("xval_data_split.R")
source("Cross-validation-function.R")
source("train-test-functions-cv.R")
library(glmnet)
library("ROCR")




#ARI
# ARI = covariate$log_transformed_ari
# Yorder[torder] = ARI
# Yorder = Yorder[-75]
# #my split as cv folds
# Array.adj.ICVF <- array(0, dim = c(200, 200, 122))
# Array.ICVF <- array(0, dim = c(200, 200, 122))
# for(i in 1:122 ){
#   Array.adj.ICVF[,,i] = Alist.ICVF[[i]]
#   Array.ICVF[,,i] <- ICVF[[i]]
# }
#10 folds
holdout <- list()
for(i in 1:9){
  holdout[[i]] <- seq(12)+12*(i-1)
}
holdout[[10]] <- seq(14)+12*9

#mycv for diff method
num_sig = seq(10,200,by=10)
#num_sig[1] =1
mse = rep(NA, 20)
mdcorrs = rep(0,20)
for(p in num_sig){
  #total 122 graphs
  #10 fold cv
  #sample size 122
  er <- rep(0,10)
  dcorrs <- rep(0,10)
  tictoc::tic()
  
  for(i in 1:10){
    sg.data <- sg.xval_split_data (Array.adj.ICVF, Yorder, holdout[[i]])
    train.set <- sg.data$train_set
    train.y <- sg.data$train_y
    test.set <- sg.data$test_set
    test.y <- sg.data$test_y
    train.g <- list()
    test.g <- list()
    train.m <- length(train.y)
    test.m <- length(test.y)
    for(k in 1:train.m){
      train.g[[k]] <- train.set[,,k]
    }
    for(k in 1:test.m){
      test.g[[k]] <- test.set[,,k]
    }
    
    train.y <- as.matrix(train.y)
    ##Shat- iter mgc and Phat
    #cors.ICVF <-subgraph_search_iter(train.g, (train.y), 'mcor', 'double',0.95)
    cors.ICVF <-subgraph_search(train.g, (train.y), 'mcor', 'double')
    #cors.ICVF <-subgraph_search(Alist.ICVF, factor(train.Y), 'mcor', 'double')
    Shat.ICVF[[i]] <- sort(cors.ICVF,decreasing = TRUE, index= TRUE)$ix[1:p]
    
    #Ai <- test.g
    
    
    
    #train.X <- matrix(NA,121,p*2)
    train.X <- matrix(NA,train.m,p*(p-1)/2)
    #train.X <- matrix(NA,121,200*(200-1)/2)
    for(j in 1:train.m){
      Aj <- train.g[[j]][Shat.ICVF[[i]],Shat.ICVF[[i]]]
      #Aj <- train.g[[j]]
      
      train.X[j,] <- Aj[upper.tri(Aj)]
      # train.X[j,] <- ase(Aj,2)
      
    }
    test.X <- matrix(NA,test.m,p*(p-1)/2)
    #train.X <- matrix(NA,121,200*(200-1)/2)
    for(j in 1:test.m){
      Ai <- test.g[[j]][Shat.ICVF[[i]],Shat.ICVF[[i]]]
      #Aj <- train.g[[j]]
      
      test.X[j,] <- Ai[upper.tri(Ai)]
      # train.X[j,] <- ase(Aj,2)
      
    }
    
    
    
    #random forest
    require(randomForest)
    fit <- randomForest(x = train.X, y = (train.y), mtry = 100, ntree = 500)
    #predict(fit, newdata = test.X)
    er[i] <- sum((test.y - predict(fit, newdata = test.X))^2)/test.m
   
    dcorrs[i] <- dcorr(train.X,train.y,'mcor','double')
    print(er[i])
  }
  sum(er)/10
  tictoc::toc()
  mse[p/10] = sum(er)/10
  print(mse[p/10])
  mdcorrs[p/10] <- mean(dcorrs)
}

save(dcorrs,mse,file ='ari_bi_nonit.RData')
#save(Yorder, file = 'Yorder.RData')

df <- rbind(cbind(seq(0,200,10), c(1,mse/var(Yorder)),rep(1,11),rep(1,11)),
            cbind(seq(0,200,10), c(0,mdcorrs),rep(2,11),rep(2,11)))

df <- data.frame(df)
colnames(df) <- c('nVertex', 'err', 'cg','dg')
df$cg <- factor(df$cg)
df$dg <- factor(df$dg)
ggplot(data = df, aes(x=nVertex, y=err,linetype = dg)) + geom_line( size=1.2) +
  geom_point(size=4)  +
  scale_linetype_discrete(name  ="Statistic",breaks=c("1", "2"),labels=c("Error","Corr")) +
  theme(text=element_text(size=10)) +
  theme(plot.title = element_text(hjust = 0.1)) +
  labs(title="Prediction Error and Distance Correlation \n Based on Signal Subgraph of ARI",  x = "Number of Vertices", y = "Statistic")+
  scale_x_continuous(breaks=seq(0,200,10),labels=seq(0,200,10))








