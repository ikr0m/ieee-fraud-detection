library(readr)
library(tidyverse)
library(data.table)
library(MLmetrics)
library(lightgbm)
library(lubridate)
library(plyr)
library(moments)
library(dplyr)
library(rpart)
library(rattle)
library(irlba)
library(e1071)
library(caret)
library(h2o)
library(solitude)
library(imputeMissings)
library(fBasics)

options(warn=-1)
options(scipen = 99)

train <- read_csv("train.csv")

# %% [code]
y <- train$isFraud 
train$isFraud <- NULL

tr_idx <- which(train$TransactionDT < quantile(train$TransactionDT, 0.8))


tem <- fread("tem.csv")
temp <- fread("tem_unsupervised.csv")

temp <- temp[, c(201, 63, 209, 183, 214, 176, 166, 15, 115, 210,
                 216, 198, 203, 35, 24, 8, 18, 66, 257:262)]

tem <- data.frame(tem, temp)
rm(temp)

y_train <- y

train <- tem[1:nrow(train),]
test <- tem[-c(1:nrow(train)),]

rm(tem) ; invisible(gc())

# %% [code]
length(y_train)


############################################################################################################
# LGB model

cat("train_col :" , ncol(train), "test_col :", ncol(test) ,"\n" )


d0 <- lgb.Dataset(data.matrix(train[tr_idx,]), label = y[tr_idx] )
dval <- lgb.Dataset(data.matrix(train[-tr_idx,]), label = y[-tr_idx] ) 

lgb_param <- list(boosting_type = 'dart',
                  objective = "binary" ,
                  metric = "AUC",
                  boost_from_average = "false",
                  tree_learner  = "serial",
                  max_depth = -1,
                  learning_rate = 0.01,
                  num_leaves = 200,
                  feature_fraction = 0.3,          
                  bagging_freq = 1,
                  bagging_fraction = 0.6,
                  min_data_in_leaf = 100,
                  bagging_seed = 71,
                  max_bin = 255,
                  verbosity = -1)


valids <- list(valid = dval)
lgb <- lgb.train(params = lgb_param,  data = d0, nrounds = 15000, 
                 eval_freq = 200, valids = valids, early_stopping_rounds = 500, verbose = 1, seed = 71)


oof_pred <- predict(lgb, data.matrix(train[-tr_idx,]))
cat("best iter :" , lgb$best_iter, "best score :", AUC(oof_pred, y[-tr_idx]) ,"\n" )
iter <- lgb$best_iter


# full data
d0 <- lgb.Dataset(data.matrix(train), label = y )
lgb <- lgb.train(params = lgb_param, data = d0, nrounds = iter * 1.05, verbose = -1)
pred <- predict(lgb, data.matrix(test))

imp <- lgb.importance(lgb)
sub <- data.frame(read_csv("sample_submission.csv"))
sub[,2] <- pred

write.csv(sub,"sub_93584_80.csv",row.names = F)
write.csv(imp,"imp.csv",row.names = F)