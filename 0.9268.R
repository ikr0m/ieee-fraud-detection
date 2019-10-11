# Load libraries
library(data.table); library(tidyverse); library(tidymodels); library(uwot); library(DataExplorer)
library(irlba); library(fastknn); library(recipes); library(repr)
library(sessioninfo); library(visdat); library(correlationfunnel); library(lightgbm); library(MLmetrics)
options(scipen = 99)

# Load data
train_id <- fread("train_identity.csv") 
test_id <- fread("test_identity.csv") 

train_transaction <- fread("train_transaction.csv")
test_transaction <- fread("test_transaction.csv")

drop_col <- c('V300','V309','V111','C3','V124','V106','V125','V315','V134','V102','V123','V316','V113',
              'V136','V305','V110','V299','V289','V286','V318','V103','V304','V116','V29','V284','V293',
              'V137','V295','V301','V104','V311','V115','V109','V119','V321','V114','V133','V122','V319',
              'V105','V112','V118','V117','V121','V108','V135','V320','V303','V297','V120', 'M1')


is_blank <- function(x) {is.na(x) | x == ""}

train_id$DeviceType[is_blank(train_id$DeviceType)] <- gsub("", "unk", train_id$DeviceType[is_blank(train_id$DeviceType)])
train_id$DeviceInfo[is_blank(train_id$DeviceInfo)] <- gsub("", "unk", train_id$DeviceInfo[is_blank(train_id$DeviceInfo)])

test_id$DeviceType[is_blank(test_id$DeviceType)] <- gsub("", "unk", test_id$DeviceType[is_blank(test_id$DeviceType)])
test_id$DeviceInfo[is_blank(test_id$DeviceInfo)] <- gsub("", "unk", test_id$DeviceInfo[is_blank(test_id$DeviceInfo)])

train_id$DeviceInfo <- gsub("SAMSUNG ", "", train_id$DeviceInfo)
train_id$DeviceInfo <- gsub("Lenovo ", "", train_id$DeviceInfo)
train_id$DeviceInfo <- gsub("SAMSUNG-", "", train_id$DeviceInfo)
train_id$DeviceInfo <- gsub("HUAWEI ", "", train_id$DeviceInfo)
train_id$DeviceInfo <- gsub("ZTE ", "", train_id$DeviceInfo)
train_id$DeviceInfo <- gsub("Hisense ", "", train_id$DeviceInfo)
train_id$DeviceInfo <- gsub("Blade ", "", train_id$DeviceInfo)
train_id$DeviceInfo <- gsub("Moto ", "", train_id$DeviceInfo)
train_id$DeviceInfo <- gsub("HTC ", "", train_id$DeviceInfo)
train_id$DeviceInfo <- gsub("M4 ", "", train_id$DeviceInfo)
train_id$DeviceInfo <- gsub("iPhone", "iOS Device", train_id$DeviceInfo)
train_id$DeviceInfo <- gsub("Microsoft", "Windows", train_id$DeviceInfo)

test_id$DeviceInfo <- gsub("SAMSUNG ", "", test_id$DeviceInfo)
test_id$DeviceInfo <- gsub("Lenovo ", "", test_id$DeviceInfo)
test_id$DeviceInfo <- gsub("SAMSUNG-", "", test_id$DeviceInfo)
test_id$DeviceInfo <- gsub("HUAWEI ", "", test_id$DeviceInfo)
test_id$DeviceInfo <- gsub("ZTE ", "", test_id$DeviceInfo)
test_id$DeviceInfo <- gsub("Hisense ", "", test_id$DeviceInfo)
test_id$DeviceInfo <- gsub("Blade ", "", test_id$DeviceInfo)
test_id$DeviceInfo <- gsub("Moto ", "", test_id$DeviceInfo)
test_id$DeviceInfo <- gsub("HTC ", "", test_id$DeviceInfo)
test_id$DeviceInfo <- gsub("M4 ", "", test_id$DeviceInfo)
test_id$DeviceInfo <- gsub("iPhone", "iOS Device", test_id$DeviceInfo)
test_id$DeviceInfo <- gsub("Microsoft", "Windows", test_id$DeviceInfo)

temp_tr <- as.data.frame(str_split(train_id$DeviceInfo, "Build/", simplify = TRUE))
temp_te <- as.data.frame(str_split(test_id$DeviceInfo, "Build/", simplify = TRUE))


colnames(temp_tr) <- c("DeviceInfo","BuildInfo")
colnames(temp_te) <- c("DeviceInfo","BuildInfo")

train_id$DeviceInfo <- NULL
test_id$DeviceInfo <- NULL

train_id <- data.frame(train_id, temp_tr)
test_id <- data.frame(test_id, temp_te)

rm(temp_tr, temp_te); gc(); gc()

train_id$BuildInfo <- as.character(train_id$BuildInfo)
test_id$BuildInfo <- as.character(test_id$BuildInfo)

levels(train_id$DeviceInfo) <- c(levels(train_id$DeviceInfo), 'other')
temp <-as.data.frame(table(as.factor(train_id$DeviceInfo)))
rownames(temp) <- temp[, 1]
train_id$DeviceInfo <- as.factor(train_id$DeviceInfo)
train_id[train_id$DeviceInfo %in% as.factor(rownames(subset(temp, temp$Freq <= 3))), 'DeviceInfo'] <- 'other'
train_id$DeviceInfo <- droplevels(train_id$DeviceInfo)
train_id$DeviceInfo <- as.character(train_id$DeviceInfo)


levels(test_id$DeviceInfo) <- c(levels(test_id$DeviceInfo), 'other')
temp <-as.data.frame(table(as.factor(test_id$DeviceInfo)))
rownames(temp) <- temp[, 1]
test_id$DeviceInfo <- as.factor(test_id$DeviceInfo)
test_id[test_id$DeviceInfo %in% as.factor(rownames(subset(temp, temp$Freq <= 3))), 'DeviceInfo'] <- 'other'
test_id$DeviceInfo <- droplevels(test_id$DeviceInfo)
test_id$DeviceInfo <- as.character(test_id$DeviceInfo)



levels(train_id$BuildInfo) <- c(levels(train_id$BuildInfo), 'other')
temp <-as.data.frame(table(as.factor(train_id$BuildInfo)))
rownames(temp) <- temp[, 1]
train_id$BuildInfo <- as.factor(train_id$BuildInfo)
train_id[train_id$BuildInfo %in% as.factor(rownames(subset(temp, temp$Freq <= 3))), 'BuildInfo'] <- 'other'
train_id$BuildInfo <- droplevels(train_id$BuildInfo)
train_id$BuildInfo <- as.character(train_id$BuildInfo)


levels(test_id$BuildInfo) <- c(levels(test_id$BuildInfo), 'other')
temp <-as.data.frame(table(as.factor(test_id$BuildInfo)))
rownames(temp) <- temp[, 1]
test_id$BuildInfo <- as.factor(test_id$BuildInfo)
test_id[test_id$BuildInfo %in% as.factor(rownames(subset(temp, temp$Freq <= 3))), 'BuildInfo'] <- 'other'
test_id$BuildInfo <- droplevels(test_id$BuildInfo)
test_id$BuildInfo <- as.character(test_id$BuildInfo)

email_col <- c("P_emaildomain", "R_emaildomain")

train_transaction[,(email_col) := 
                    lapply(.SD, function(x) gsub(".*(yahoo|ymail|frontier|rocketmail).*", "yahoo", x)), .SDcols = email_col] 

train_transaction[,(email_col) := 
                    lapply(.SD, function(x) gsub(".*(hotmail|outlook|live).*|msn.com", "hotmail", x)), .SDcols = email_col] 

train_transaction[,(email_col) := 
                    lapply(.SD, function(x) gsub(".*(gmail).*", "gmail", x)), .SDcols = email_col]        

train_transaction[,(email_col) := 
                    lapply(.SD, function(x) gsub(".*(netzero).*", "netzero", x)), .SDcols = email_col]   

train_transaction[,(email_col) := 
                    lapply(.SD, function(x) gsub(".*(icloud).*|mac.com|me.com", "apple", x)), .SDcols = email_col] 

train_transaction[,(email_col) := 
                    lapply(.SD, function(x) gsub(".*(prodigy|sbcglobal).*|att.net", "att", x)), .SDcols = email_col] 

train_transaction[,(email_col) := 
                    lapply(.SD, function(x) gsub(".*(centurylink|embarqmail).*|q.com", "centurylink", x)), .SDcols = email_col]  

train_transaction[,(email_col) := 
                    lapply(.SD, function(x) gsub("aim.com|aol.com", "aol", x)), .SDcols = email_col]   

train_transaction[,(email_col) := 
                    lapply(.SD, function(x) gsub("twc.com|charter.net", "spectrum", x)), .SDcols = email_col]                                   


test_transaction[,(email_col) := 
                   lapply(.SD, function(x) gsub(".*(yahoo|ymail|frontier|rocketmail).*", "yahoo", x)), .SDcols = email_col] 

test_transaction[,(email_col) := 
                   lapply(.SD, function(x) gsub(".*(hotmail|outlook|live).*|msn.com", "hotmail", x)), .SDcols = email_col] 

test_transaction[,(email_col) := 
                   lapply(.SD, function(x) gsub(".*(gmail).*", "gmail", x)), .SDcols = email_col]        

test_transaction[,(email_col) := 
                   lapply(.SD, function(x) gsub(".*(netzero).*", "netzero", x)), .SDcols = email_col]   

test_transaction[,(email_col) := 
                   lapply(.SD, function(x) gsub(".*(icloud).*|mac.com|me.com", "apple", x)), .SDcols = email_col] 

test_transaction[,(email_col) := 
                   lapply(.SD, function(x) gsub(".*(prodigy|sbcglobal).*|att.net", "att", x)), .SDcols = email_col] 

test_transaction[,(email_col) := 
                   lapply(.SD, function(x) gsub(".*(centurylink|embarqmail).*|q.com", "centurylink", x)), .SDcols = email_col]  

test_transaction[,(email_col) := 
                   lapply(.SD, function(x) gsub("aim.com|aol.com", "aol", x)), .SDcols = email_col]   

test_transaction[,(email_col) := 
                   lapply(.SD, function(x) gsub("twc.com|charter.net", "spectrum", x)), .SDcols = email_col]    

train_transaction$trans_na <- rowSums(is.na(train_transaction))
train_id$id_na <- rowSums(is.na(train_id))
test_transaction$trans_na <- rowSums(is.na(test_transaction))
test_id$id_na <- rowSums(is.na(test_id))

train <- train_transaction %>% left_join(train_id) %>% as_tibble()
test <- test_transaction %>% left_join(test_id) %>% as_tibble()


rm(train_id,train_transaction,test_id, test_transaction); gc(); gc()

train[,drop_col] <- NULL
test[,drop_col] <- NULL

y <- train$isFraud 
train$isFraud <- NULL


# using single hold-out validation (70%)
tr_idx <- which(train$TransactionDT < quantile(train$TransactionDT,0.66))

tem <- train %>% bind_rows(test) %>%
  mutate(hr = floor( (TransactionDT / 3600) %% 24 ),
         weekday = floor( (TransactionDT / 3600 / 24) %% 7),
         TransactionAmt_decimal = (TransactionAmt - round(TransactionAmt)) * 1000
  ) %>%
  select(-TransactionID,-TransactionDT)

tem$acc = paste(tem$card1,tem$card2,tem$card3,tem$card4,tem$card5,tem$addr1,tem$addr2)
tem$addr1_addr2 = paste(tem$addr1,tem$addr2)

char_features = c("acc","card1","card2","card3","ProductCD")


char_features <- colnames(tem[, sapply(tem, class) %in% c('character', 'factor')])
for (f in char_features){
  levels <- unique(tem[[f]])
  tem[[f]] <- as.integer(factor(tem[[f]], levels=levels))
}

tem$decimal = nchar(tem$TransactionAmt - floor(tem$TransactionAmt))
fe_part2 <- tem[, c("acc","card1","card2","card3", "card4", "TransactionAmt", "id_02", "D15","addr1_addr2","decimal","C13","V258","C1","C14")]

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~card1, summarise, mean_card1_Trans= mean(TransactionAmt), sd_card1_Trans = sd(TransactionAmt),count_card1 = n()))
fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~addr1_addr2, summarise, mean_geo_Trans= mean(TransactionAmt), sd_geo_Trans = sd(TransactionAmt),count_geo = n()))
fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~card1, summarise, min_card1_Trans= min(TransactionAmt), max_card1_Trans = max(TransactionAmt)))
fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~card2, summarise, min_card2_Trans= min(TransactionAmt), max_card2_Trans = max(TransactionAmt),count_card2 = n()))
fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~card4, summarise, mean_card4_Trans = mean(TransactionAmt), sd_card4_Trans = sd(TransactionAmt),count_card4 = n()))
fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~acc, summarise, mean_acc_Trans = mean(TransactionAmt), sd_acc_Trans = sd(TransactionAmt),count_acc = n()))



fe_part2 = fe_part2 %>% left_join(plyr::ddply(fe_part2, ~acc, summarise, mean_acc_dec = mean(decimal,na.rm = T), sd_acc_dec = sd(decimal),acc_dec_q = quantile(decimal,.8,na.rm = T)))

# %% [code]
head(fe_part2)
head(unique(fe_part2$mean_card1_D15))
head(unique(fe_part2$mean_card1_id02))
fe_part2 <- fe_part2[, -c(1:14)]


# Add some feature interaction
tem <- tem %>%
  mutate(card4__M4 = unlist(map2(.x = card4, .y = M4, .f = function(x,y){
    y <- str_c(x, y, sep = "_")
    return(y)
  })),
  card1__dist1 = unlist(map2(.x = card1, .y = dist1, .f = function(x,y){
    y <- str_c(x, y, sep = "_")
    return(y)
  })),
  card1__P_emaildomain = unlist(map2(.x = card1, .y = P_emaildomain, .f = function(x,y){
    y <- str_c(x, y, sep = "_")
    return(y)
  })),
  addr1__card1 = unlist(map2(.x = addr1, .y = card1, .f = function(x,y){
    y <- str_c(x, y, sep = "_")
    return(y)
  })),
  card4__dist1 = unlist(map2(.x = card4, .y = dist1, .f = function(x,y){
    y <- str_c(x, y, sep = "_")
    return(y)
  })),
  addr1__card4 = unlist(map2(.x = addr1, .y = card4, .f = function(x,y){
    y <- str_c(x, y, sep = "_")
    return(y)
  })),
  P_emaildomain__C2 = unlist(map2(.x = P_emaildomain, .y = C2, .f = function(x,y){
    y <- str_c(x, y, sep = "_")
    return(y)
  })),
  card2__dist1 = unlist(map2(.x = card2, .y = dist1, .f = function(x,y){
    y <- str_c(x, y, sep = "_")
    return(y)
  })),
  card1__card5 = unlist(map2(.x = card1, .y = card5, .f = function(x,y){
    y <- str_c(x, y, sep = "_")
    return(y)
  })),
  card5__P_emaildomain = unlist(map2(.x = card5, .y = P_emaildomain, .f = function(x,y){
    y <- str_c(x, y, sep = "_")
    return(y)
  }))
  )

# FE part1 : Count encoding
char_features <- tem[,colnames(tem) %in% 
                       c("acc", "ProductCD","card1","card2","card3","card4","card5","card6","addr1","addr2","P_emaildomain",
                         "R_emaildomain","M2","M3","M4","M5","M6","M7","M8","M9",
                         "card4__M4", "card1__dist1", "card1__P_emaildomain", "addr1__card1",
                         "card4__dist1", "addr1__card4", "P_emaildomain__C2", "card2__dist1",
                         "card1__card5", "card5__P_emaildomain", "BuildInfo")]

fe_part1 <- data.frame(0)
for(a in colnames(char_features) ){
  tem1 <- char_features %>% group_by(.dots = a) %>% mutate(count = length(card4)) %>% ungroup() %>% select(count)
  colnames(tem1) <- paste(a,"__count_encoding",sep="")
  fe_part1 <- data.frame(fe_part1,tem1)
}

fe_part1 <- fe_part1[,-1]
rm(char_features,tem1) ; invisible(gc())
cat("fe_part1 ncol :" , ncol(fe_part1) ,"\n" )
#############################################################################################################
# label 
char_f <-  c("ProductCD","card1","card2","card3","card4","card5","card6","addr1","addr2","P_emaildomain",
             "R_emaildomain","M2","M3","M4","M5","M6","M7","M8","M9", "card4__M4", "card1__dist1", "card1__P_emaildomain",
             "addr1__card1", "card4__dist1", "addr1__card4", "P_emaildomain__C2", "card2__dist1",
             "card1__card5", "card5__P_emaildomain", "BuildInfo")
tem <- tem %>% mutate_at(char_f, function(x){
  y <- as.factor(x)
  y <- as.numeric(y)
  y <- y - 1
  return(y)
})

tem <- data.frame(tem,fe_part1,fe_part2)
rm(fe_part1,fe_part2); invisible(gc());

# Add some agreggations on card, addr and email variables
tem <- tem %>% group_by(card1) %>% mutate(mean_TransactionAmt_card1 = mean(TransactionAmt, na.rm = TRUE),
                                          sd_TransactionAmt_card1 = sd(TransactionAmt, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(TransactionAmt_to_mean_card1 = TransactionAmt/mean_TransactionAmt_card1,
         TransactionAmt_to_sd_card1 = TransactionAmt/sd_TransactionAmt_card1,
         TransactionAmt_subs_card1 = TransactionAmt - mean_TransactionAmt_card1)

tem <- tem %>% group_by(card2) %>% mutate(mean_TransactionAmt_card2 = mean(TransactionAmt, na.rm = TRUE),
                                          sd_TransactionAmt_card2 = sd(TransactionAmt, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(TransactionAmt_to_mean_card2 = TransactionAmt/mean_TransactionAmt_card2,
         TransactionAmt_to_sd_card2 = TransactionAmt/sd_TransactionAmt_card2,
         TransactionAmt_subs_card2 = TransactionAmt - mean_TransactionAmt_card2)

tem <- tem %>% group_by(addr1, card4) %>% mutate(mean_TransactionAmt_addr1_card4 = mean(TransactionAmt, na.rm = TRUE),
                                                 sd_TransactionAmt_addr1_card4 = sd(TransactionAmt, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(TransactionAmt_to_mean_addr1_card4 = TransactionAmt/mean_TransactionAmt_addr1_card4,
         TransactionAmt_to_sd_addr1_card4 = TransactionAmt/sd_TransactionAmt_addr1_card4,
         TransactionAmt_subs_addr1_card4 = TransactionAmt - mean_TransactionAmt_addr1_card4)

tem <- tem %>% group_by(card1) %>% mutate(mean_D15_card1 = mean(D15, na.rm = TRUE),
                                          sd_D15_card1 = sd(D15, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(D15_to_mean_card1 = D15/mean_D15_card1,
         D15_to_sd_card1 = D15/sd_D15_card1,
         D15_subs_card1 = D15 - mean_D15_card1)


tem <- tem %>% group_by(addr1, card1) %>% mutate(mean_TransactionAmt_addr1_card1 = mean(TransactionAmt, na.rm = TRUE),
                                                 sd_TransactioAmt_addr1_card1 = sd(TransactionAmt, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(TransactionAmt_to_mean_addr1_card1 = TransactionAmt/mean_TransactionAmt_addr1_card1,
         TransactionAmt_to_sd_addr1_card1 = TransactionAmt/sd_TransactioAmt_addr1_card1,
         TransactionAmt_subs_addr1_card1 = TransactionAmt - mean_TransactionAmt_addr1_card1)

tem <- tem %>% group_by(card5, card1) %>% mutate(mean_TransactionAmt_card5_card1 = mean(TransactionAmt, na.rm = TRUE),
                                                 sd_TransactioAmt_card5_card1 = sd(TransactionAmt, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(TransactionAmt_to_mean_card5_card1 = TransactionAmt/mean_TransactionAmt_card5_card1,
         TransactionAmt_to_sd_card5_card1 = TransactionAmt/sd_TransactioAmt_card5_card1,
         TransactionAmt_subs_card5_card1 = TransactionAmt - mean_TransactionAmt_card5_card1)

tem <- tem %>% group_by(card5, card1) %>% mutate(mean_D2_card5_card1 = mean(D2, na.rm = TRUE),
                                                 sd_D2_card5_card1 = sd(D2, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(D2_to_mean_card5_card1 = D2/mean_D2_card5_card1,
         D2_to_sd_card5_card1 = D2/sd_D2_card5_card1,
         D2_subs_card5_card1 = D2 - mean_D2_card5_card1)


tem <- tem %>% group_by(P_emaildomain) %>% mutate(mean_TransactionAmt_P_emaildomain = mean(TransactionAmt, na.rm = TRUE),
                                                  sd_TransactionAmt_P_emaildomain = sd(TransactionAmt, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(TransactionAmt_to_mean_P_emaildomain = TransactionAmt/mean_TransactionAmt_P_emaildomain,
         TransactionAmt_to_sd_P_emaildomain = TransactionAmt/sd_TransactionAmt_P_emaildomain,
         TransactionAmt_subs_P_emaildomain = TransactionAmt - mean_TransactionAmt_P_emaildomain)

tem <- tem %>% mutate(TransactionAmt = log(1 + TransactionAmt))


tem = tem[!duplicated(as.list(tem))]
tem$addr1_card1 = paste(tem$addr1,tem$card1) 
tem$card1_and_count = paste(tem$card1,tem$count_card1)
tem$card2_and_count = paste(tem$card2,tem$count_card2)
tem$new3 = tem$TransactionAmt * tem$C1
tem$new5 = tem$TransactionAmt * tem$C13
tem$new7 = tem$TransactionAmt * tem$C14

tem$NA_V1_V11 = apply(is.na(tem[,which(names(tem) %in% paste0("V",1:11))]), 1, sum)
tem$NA_V12_V34 = apply(is.na(tem[,which(names(tem) %in% paste0("V",12:34))]), 1, sum)
tem$NA_V35_V54 = apply(is.na(tem[,which(names(tem) %in% paste0("V",35:52))]), 1, sum)
tem$NA_V53_V74 = apply(is.na(tem[,which(names(tem) %in% paste0("V",53:74))]), 1, sum)
tem$NA_V75_V94 = apply(is.na(tem[,which(names(tem) %in% paste0("V",75:94))]), 1, sum)
tem$NA_V95_V137 = apply(is.na(tem[,which(names(tem) %in% paste0("V",95:137))]), 1, sum)
tem$NA_V138_V166 = apply(is.na(tem[,which(names(tem) %in% paste0("V",138:166))]), 1, sum)
tem$NA_V167_V216 = apply(is.na(tem[,which(names(tem) %in% paste0("V",167:216))]), 1, sum)
tem$NA_V217_V278 = apply(is.na(tem[,which(names(tem) %in% paste0("V",217:278))]), 1, sum)
tem$NA_V279_V321 = apply(is.na(tem[,which(names(tem) %in% paste0("V",279:321))]), 1, sum)
tem$NA_V322_V339 = apply(is.na(tem[,which(names(tem) %in% paste0("V",322:339))]), 1, sum)


tem$decimal_diff = tem$decimal - tem$mean_acc_dec
tem$new13 = tem$decimal_diff * tem$mean_acc_Trans

train <- tem[1:nrow(train),]
test <- tem[-c(1:nrow(train)),]
rm(tem) ; invisible(gc())

############################################################################################################
# model
cat("train_col :" , ncol(train), "test_col :", ncol(test) ,"\n" )


d0 <- lgb.Dataset(data.matrix( train[tr_idx,] ), label = y[tr_idx] )
dval <- lgb.Dataset(data.matrix( train[-tr_idx,] ), label = y[-tr_idx] ) 

lgb_param <- list(boosting_type = 'gbdt',
                  objective = "binary" ,
                  metric = "AUC",
                  boost_from_average = "false",
                  learning_rate = 0.01,
                  num_leaves = 197,
                  min_gain_to_split = 0,
                  feature_fraction = 0.3,
                  # feature_fraction_seed = 666666,
                  bagging_freq = 1,
                  bagging_fraction = 0.7,
                  # min_sum_hessian_in_leaf = 0,
                  min_data_in_leaf = 100,
                  lambda_l1 = 0,
                  lambda_l2 = 0
)

set.seed(71)
valids <- list(valid = dval)
lgb <- lgb.train(params = lgb_param,  data = d0, nrounds = 15000, 
                 eval_freq = 200, valids = valids, early_stopping_rounds = 400, verbose = 1, seed = 71)

iter <- lgb$best_iter

rm(d0, dval, lgb)
gc()
# full data
d0 <- lgb.Dataset( data.matrix( train ), label = y )
set.seed(71)
lgb <- lgb.train(params = lgb_param, data = d0, nrounds = iter*1.03, verbose = -1, seed = 71)
pred <- predict(lgb, data.matrix(test))


sub <- data.frame(read_csv("sample_submission.csv"))
sub[,2] <- pred

write.csv(sub,"submission_0.9265.csv",row.names=F)

imp <- lgb.importance(lgb)
write.csv(imp,"importances_0.9265.csv",row.names=F)

lgb.plot.importance(imp, top_n = 50)