bestTuneRidge <- read.csv('extra/best_tune_ridge.csv')
bestTuneElasticNet <- read.csv('extra/best_tune_elasticNet.csv')


load("Data/fMRIdata.RData")

library(caret)
library(glmnet)


data <- as.data.frame(fit_feat)
response <- as.data.frame(resp_dat)
colnames(response) <- c(paste0("R", c(1:20)))
data_combine <- cbind(data, response)

#Data Partition
set.seed(7)
ind <- sample(2, nrow(data), replace = T, prob = c(0.7, 0.3))
train_combine <- data_combine[ind==1, ]    
test_combine <- data_combine[ind==2, ]
train <- train_combine[ ,c(1:10921)]

#Using the best hyperparameter tuned in the file bestTune.R
#To fit the best Elastic Net Models using Cross Validation to predict all the 20 voxel responses
en <- list()
response_names <- c(paste0("R", c(1:20)))
set.seed(7)
x <- as.matrix(train_combine[ c(1:10921)])
for(i in response_names){
  y <- as.matrix(train_combine[i])
  set.seed(7)
  en[[i]] <- glmnet(x, y, family = "gaussian", 
                    alpha = bestTuneElasticNet$best_alpha[which(bestTuneElasticNet$Voxel==i)],
                    lambda = bestTuneElasticNet$best_lambda[which(bestTuneElasticNet$Voxel==i)])}


#Using the best hyperparameter tuned in the file bestTune.R
#To fit the best Ridge Regression Models using Cross Validation to predict all the 20 voxel responses
rd <- list()
response_names <- c(paste0("R", c(1:20)))
set.seed(7)
x <- as.matrix(train_combine[ c(1:10921)])
for(i in response_names){
  y <- as.matrix(train_combine[i])
  set.seed(7)
  rd[[i]] <- glmnet(x, y, family = "gaussian", 
                    alpha = 0,
                    lambda = bestTuneRidge$best_lambda[which(bestTuneRidge$Voxel==i)])}


#Computing the prediction and Correlation score for these Ridge and Elastic Net models on validation set
Ridge_corr <- c()
for(i in response_names){
  p <- rd[[i]]$a0 + (as.matrix(test_combine[c(1:10921)]) %*% rd[[i]]$beta)
  Ridge_corr <- c(Ridge_corr, cor(as.vector(p), test_combine[[i]]))
}


EN_corr <- c()
for(i in response_names){
  p <- en[[i]]$a0 + (as.matrix(test_combine[c(1:10921)]) %*% en[[i]]$beta)
  EN_corr <- c(EN_corr, cor(as.vector(p), test_combine[[i]]))
}

#Saving the correlation score on the validation test for both ridge and EN
write.csv(cbind(Ridge_corr, EN_corr), "extra/RidgeENCorrComp.csv")


#Predicting the response of voxel1 on the separate validation set of 120 transformed images
predv1 <- en$R1$a0 + (as.matrix(val_feat) %*% en$R1$beta)

predv1 <- as.vector(predv1)

write.file(predv1, "output/predv1_Aummul.txt", row.names = F)
