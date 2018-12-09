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




#1 Lasso Models using Cross Validation
cv <- list()
response_names <- c(paste0("R", c(1:20)))
#Custom Control Parameters
custom <- trainControl(method = "cv",
                       number = 10,
                       verboseIter = T)

for(i in response_names){
  train <- cbind(train_combine[ c(1:10921)], train_combine[i])
  names(train)[names(train) == i] <- 'RV'
  set.seed(7)
  cv[[i]] <- train(RV ~ .,
                             train,
                             method = "glmnet",
                             tuneGrid = expand.grid(alpha = 1,lambda = seq(0.04, 0.17, length = 14)),
                             trControl = custom)
}

# Method 2 for cv
cv2 <- list()
for(i in response_names){
  y <- train_combine[c(1:nrow(train_combine)), i]
  cv2[[i]] <- Lasso(x, y, fix.lambda = F, nfolds = 10, cv.method = "cv", parallel = F, intercept = T )
}


#Saving CV models
for(i in response_names){
  saveRDS(cv[[i]], paste0('cv_lasso_', i, ".rds"))
}

for(i in response_names){
  saveRDS(cv2[[i]], paste0('cv2_lasso_', i, ".rds"))
}



#2 LASSO Models using ESCV
library(HDCI)
set.seed(7)
escv <- list()
x <- as.matrix(train)
for(i in response_names){
  y <- train_combine[c(1:nrow(train_combine)), i]
  escv[[i]] <- Lasso(x, y, fix.lambda = F, nfolds = 10, cv.method = "escv", parallel = F, intercept = T )
}
#Saving ESCV model
for(i in response_names){
  saveRDS(escv[[i]], paste0('escv_lasso_', i, ".rds"))
}



#3 LASSO using AICc
source("IC_function.R")
set.seed(7)
aicc <- list()
x <- as.matrix(train)
for(i in response_names){
  y <- train_combine[c(1:nrow(train_combine)), i]
  aicc[[i]] <- ic.glmnet(x, y, crit = "aicc", alpha = 1 )
}
#Saving AICc model
for(i in response_names){
  saveRDS(aicc[[i]], paste0('aicc_lasso_', i, ".rds"))
}






#4 LASSO using AIC
source("IC_function.R")
set.seed(7)
aic <- list()
x <- as.matrix(train)
for(i in response_names){
  y <- train_combine[c(1:nrow(train_combine)), i]
  aic[[i]] <- ic.glmnet(x, y, crit = "aic", alpha = 1 )
}
#Saving AIC model
for(i in response_names){
  saveRDS(aic[[i]], paste0('aic_lasso_', i, ".rds"))
}




#5 LASSO using BIC
source("IC_function.R")
set.seed(7)
bic <- list()
x <- as.matrix(train)
for(i in response_names){
  y <- train_combine[c(1:nrow(train_combine)), i]
  bic[[i]] <- ic.glmnet(x, y, crit = "bic", alpha = 1 )
}
#Saving BIC model
for(i in response_names){
  saveRDS(bic[[i]], paste0('bic_lasso_', i, ".rds"))
}


## Comparing performances of all the 5 models
model_names <- c("cv", "escv", "aic", "aicc", "bic")


#1. Comparing by lambda
ESCV_lambda <- c()
for(i in response_names){
  ESCV_lambda <- c(ESCV_lambda, round(escv[[i]]$lambda, 3))
}
CV_lambda <- c()
for(i in response_names){
  CV_lambda <- c(CV_lambda, round(cv2[[i]]$lambda, 3))
}
AIC_lambda <- c()
for(i in response_names){
  AIC_lambda <- c(AIC_lambda, round(aic[[i]]$lambda, 3))
}
AICC_lambda <- c()
for(i in response_names){
  AICC_lambda <- c(AICC_lambda, round(aicc[[i]]$lambda, 3))
}
BIC_lambda <- c()
for(i in response_names){
  BIC_lambda <- c(BIC_lambda, round(bic[[i]]$lambda, 3))
}


#2 Comparing the correlation score
library(HDCI)
ESCV_corr <- c()
for(i in response_names){
  p <- mypredict(escv[[i]], test_combine[c(1:10921)])
  ESCV_corr <- c(ESCV_corr, cor(p, test_combine[[i]]))
}

CV_corr <- c()
for(i in response_names){
  p <- mypredict(cv2[[i]], test_combine[c(1:10921)])
  CV_corr <- c(CV_corr, cor(p, test_combine[[i]]))
}

AIC_corr <- c()
for(i in response_names){
  p <- aic[[i]]$coefficients[1] + (as.matrix(test_combine[c(1:10921)]) %*% aic[[i]]$coefficients[c(2:10922)])
  AIC_corr <- c(AIC_corr, cor(p, test_combine[[i]]))
}

BIC_corr <- c()
for(i in response_names){
  p <- bic[[i]]$coefficients[1] + (as.matrix(test_combine[c(1:10921)]) %*% bic[[i]]$coefficients[c(2:10922)])
  BIC_corr <- c(BIC_corr, cor(p, test_combine[[i]]))
}

AICC_corr <- c()
for(i in response_names){
  p <- aicc[[i]]$coefficients[1] + (as.matrix(test_combine[c(1:10921)]) %*% aicc[[i]]$coefficients[c(2:10922)])
  AICC_corr <- c(AICC_corr, cor(p, test_combine[[i]]))
}


#3. Comparing by model size
ESCV_ms <- c()
for(i in response_names){
  ESCV_ms <- c(ESCV_ms, length(which(escv[[i]]$beta > 0.001)))
}

CV_ms <- c()
for(i in response_names){
  CV_ms <- c(CV_ms, length(which(cv2[[i]]$beta > 0)))
}
AIC_ms <- c()
for(i in response_names){
  AIC_ms <- c(AIC_ms, length(which(aic[[i]]$coefficients > 0)))
}

AICC_ms <- c()
for(i in response_names){
  AICC_ms <- c(AICC_ms, length(which(aic[[i]]$coefficients > 0)))
}
BIC_ms <- c()
for(i in response_names){
  BIC_ms <- c(BIC_ms, length(which(aic[[i]]$coefficients > 0)))
}


model_size_comp <- cbind(ESCV_ms, CV_ms, AIC_ms, AICC_ms, BIC_ms)
corr_comp <- cbind(ESCV_corr, CV_corr, AIC_corr, AICC_corr, BIC_corr)
lam_comp <- cbind(round(ESCV_lambda, 3), round(CV2_lambda, 3), AIC_lambda, AICC_lambda, BIC_lambda)


#Writing all the comparision files
write.csv(model_size_comp, "extra/ms_comp.csv")
write.csv(corr_comp, "extra/correlationComp.csv")
write.csv(lam_comp, "extra/lambdaComp.csv")



##Checking for the model outliers for voxel 7 with the ESCV as well as CV models.
library(HDCI)
escv_pred7 <- mypredict(escv[["R7"]], test_combine[c(1:10921)])
escv_diff7 <- abs(escv_pred7 - test_combine[["R7"]])

cv_pred7 <- mypredict(cv2[["R7"]], test_combine[c(1:10921)])
cv_diff7 <- abs(cv_pred7 - test_combine[["R7"]])


escv_pred9 <- mypredict(escv[["R9"]], test_combine[c(1:10921)])
escv_diff9 <- abs(escv_pred9 - test_combine[["R9"]])

cv_pred9 <- mypredict(cv2[["R9"]], test_combine[c(1:10921)])
cv_diff9 <- abs(cv_pred9 - test_combine[["R9"]])

#Comparing the performance of CV model wrt ESCV model
perf_comp <- (sum(escv_diff7) - sum(cv_diff7)) * 100/sum(escv_diff7)


#Plotting the distribution of the errors to identify outliers
aum <- as.data.frame(cbind(escv_diff7, cv_diff7, escv_diff9, cv_diff9))
names(aum) <- c("ESCV_voxel7", "CV_voxel7", "ESCV_voxel9", "CV_voxel9")

library(reshape)
meltData <- melt(aum)
names(meltData) <- c("Model_Type", "Prediction_Error")
p <- ggplot(meltData, aes(factor(Model_Type), Prediction_Error)) 
png("extra/outliers.png")
p + geom_boxplot() + facet_grid(~Model_Type, scale="free") + xlab("Type of Model") + ylab("Prediction Error")+ theme_bw()
dev.off()


#Feature selection

png("extra/fs.png")
par(mfrow = c(1, 2))
plot(cv$R6$finalModel, xvar = "lambda", label = T)
plot(cv$R7$finalModel, xvar = "lambda", label = T)
dev.off()


#Bootstrapping
bl_escv <- bootLasso(as.matrix(train_combine[, c(1:10921)]), as.matrix(train_combine[,"R9"]), B = 500, type.boot = "residual", thres = 0.5, alpha = 0.05, 
          cv.method = "escv", nfolds = 10, foldid,  standardize = TRUE, intercept = TRUE)

bl_cv <- bootLasso(as.matrix(train_combine[, c(1:10921)]), as.matrix(train_combine[,"R9"]), B = 500, type.boot = "residual", thres = 0.5, alpha = 0.05, 
          cv.method = "cv", nfolds = 10, foldid,  standardize = TRUE, intercept = TRUE)



