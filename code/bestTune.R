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




#Hyperparameter tuning for Elastic Net Models using Cross Validation to predict 20 voxel responses
en <- list()
response_names <- c(paste0("R", c(1:20)))
best_tune_elasticNet <- data.frame()
set.seed(7)
#Custom Control Parameters
custom <- trainControl(method = "cv",
                       number = 10,
                       verboseIter = T)

for(i in response_names){
  train <- cbind(train_combine[ c(1:10921)], train_combine[i])
  names(train)[names(train) == i] <- 'RV'
  set.seed(7)
  en[[i]] <- train(RV ~ .,
                   train,
                   method = "glmnet",
                   tuneGrid = expand.grid(lambda = seq(0.025, 2, length = 80 ), alpha = seq(0.1, 1, length = 10)),
                   trControl = custom)
  best_tune_elasticNet <- rbind(best_tune_elasticNet, cbind(i, en[[i]]$bestTune))
}
colnames(best_tune_elasticNet) <- c("Voxel", "best_alpha", "best_lambda")


#Hyperparameter tuning for Ridge Regression Models using Cross Validation to predict 20 voxel responses
rd <- list()
best_tune_ridge <- data.frame()
set.seed(7)
for(i in response_names){
  train <- cbind(train_combine[ c(1:10921)], train_combine[i])
  names(train)[names(train) == i] <- 'RV'
  set.seed(7)
  rd[[i]] <- train(RV ~ .,
                   train,
                   method = "glmnet",
                   tuneGrid = expand.grid(lambda = seq(0.025, 2, length = 80 ), alpha = 0),
                   trControl = custom)
  best_tune_ridge <- rbind(best_tune_ridge, cbind(i, rd[[i]]$bestTune))
}
colnames(best_tune_ridge) <- c("Voxel", "best_alpha", "best_lambda")



#Saving the best Tume model to use for predictions.
write.csv(best_tune_ridge, "../extra/best_tune_ridge.csv")
write.csv(best_tune_elasticNet, "../extra/best_tune_elasticNet.csv")

