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

train<- read_csv("train.csv")
test <- read_csv("test.csv")

is_blank <- function(x) {is.na(x) | x == ""}

# %% [code]
y <- train$isFraud 
train$isFraud <- NULL

train$n_na <- rowSums(is.na(train))
test$n_na <- rowSums(is.na(test))


train$n_na_D <- rowSums(is.na(train[, 31:45]))
test$n_na_D <- rowSums(is.na(test[, 31:45]))

train$rowSds_D <- rowSds(train[, 31:45] %>% mutate_all(funs(ifelse(is.na(.), -10,.))))
test$rowSds_D <- rowSds(test[, 31:45] %>% mutate_all(funs(ifelse(is.na(.), -10,.))))


train$rowSkewness_D <- rowSkewness(train[, 31:45] %>% mutate_all(funs(ifelse(is.na(.), -10,.))))
test$rowSkewness_D <- rowSkewness(test[, 31:45] %>% mutate_all(funs(ifelse(is.na(.), -10,.))))


train$rowKurtosis_D <- rowKurtosis(train[, 31:45] %>% mutate_all(funs(ifelse(is.na(.), -10,.))))
test$rowKurtosis_D <- rowKurtosis(test[, 31:45] %>% mutate_all(funs(ifelse(is.na(.), -10,.))))

drop_col <- c('V300','V309','V111','V124','V106','V125','V315','V134','V102','V123','V316','V113',
              'V136','V305','V110','V299','V289','V286','V318','V304','V116','V284','V293',
              'V137','V295','V301','V104','V311','V115','V109','V119','V321','V114','V133','V122','V319',
              'V105','V112','V118','V117','V121','V108','V135','V320','V303','V297','V120',
              'V1','V14','V41','V65','V88', 'V89', 'V107', 'V68', 'V28', 'V27', 'V29', 'V241','V269',
              'V240', 'V325', 'V138', 'V154', 'V153', 'V330', 'V142', 'V195', 'V302', 'V328', 'V327', 
              'V198', 'V196', 'V155')

train[,drop_col] <- NULL
test[,drop_col] <- NULL

# using single hold-out validation (20%)
tr_idx <- which(train$TransactionDT < quantile(train$TransactionDT, 0.8))

# create tem dataframe for both train and test data and engineer time of day feature
tem <- train %>% bind_rows(test) %>%
  mutate(
    card1__dist1 = unlist(map2(.x = card1, .y = dist1, .f = function(x,y){
      y <- str_c(x, y, sep = "_")
      return(y)
    })),
    card1__P_emaildomain = unlist(map2(.x = card1, .y = P_emaildomain, .f = function(x,y){
      y <- str_c(x, y, sep = "_")
      return(y)
    })),
    card1__R_emaildomain = unlist(map2(.x = card1, .y = R_emaildomain, .f = function(x,y){
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
    R_emaildomain__C2 = unlist(map2(.x = R_emaildomain, .y = C2, .f = function(x,y){
      y <- str_c(x, y, sep = "_")
      return(y)
    })),
    P_emaildomain__addr1 = unlist(map2(.x = P_emaildomain, .y = addr1, .f = function(x,y){
      y <- str_c(x, y, sep = "_")
      return(y)
    })),
    R_emaildomain__addr1 = unlist(map2(.x = R_emaildomain, .y = addr1, .f = function(x,y){
      y <- str_c(x, y, sep = "_")
      return(y)
    })),
    
    addr1__card2 = unlist(map2(.x = addr1, .y = card2, .f = function(x,y){
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
    })),
    card5__R_emaildomain = unlist(map2(.x = card5, .y = R_emaildomain, .f = function(x,y){
      y <- str_c(x, y, sep = "_")
      return(y)
    }))
  )

tem$acc = paste(tem$card1,tem$card2,tem$card3,tem$card4,tem$card5,tem$addr1,tem$addr2)
tem$addr1_addr2 = paste(tem$addr1,tem$addr2)
tem$decimal = nchar(tem$TransactionAmt - floor(tem$TransactionAmt))
tem$PRmailTF = as.factor(tem$P_emaildomain == tem$R_emaildomain)


# label 

char_features <- colnames(tem[, sapply(tem, class) %in% c('character', 'factor')])
for (f in char_features){
  levels <- unique(tem[[f]])
  tem[[f]] <- as.integer(factor(tem[[f]], levels=levels))
}


# reference: https://www.kaggle.com/jasonbian/2-fold-lgb-with-extensive-fe
# FE part1: # Latest browser

fe_part1 <- tem[, "id_31"]
fe_part1$latest_browser = 0

new_browsers <- c("samsung browser 7.0", "opera 53.0", "mobile safari 10.0", "google search application 49.0",
                  "firefox 60.0", "edge 17.0", "chourome 69.0", "chourome 67.0", "chourome 63.0", "chourome 63.0", 
                  "chourome 64.0", "chourome 64.0 for android", "chourome 64.0 for ios", "chourome 65.0", "chourome 65.0 for android",
                  "chourome 65.0 for ios", "chourome 66.0", "chourome 66.0 for android", "chourome 66.0 for ios")

fe_part1[fe_part1$id_31 %in% new_browsers,] <- 1

paste0(nrow(fe_part1[fe_part1$latest_browser == 1 ,])*100/nrow(fe_part1), " % total rows with latest browsers")
paste0(nrow(fe_part1[fe_part1$latest_browser == 0 ,])*100/nrow(fe_part1), " % total rows with old browsers")

fe_part1 <- fe_part1[, -c(1)]

#FE part2: mean and sd transaction amount to card for card1, card4, id_02, D15
fe_part2 <- tem[, c("acc","card1","card2","card3", "card4", "TransactionAmt", "id_02", "D15","addr1_addr2",
                    "decimal","C13","V258","C1","C14", "hour", "weekday", "P_emaildomain__addr1", "R_emaildomain__addr1",
                    "PRmailTF", "addr1__card1", "addr1__card2", "card1__P_emaildomain", "card5__P_emaildomain")]

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~acc, summarise, mean_acc_Trans= mean(TransactionAmt), sd_acc_Trans = sd(TransactionAmt),  min_acc_Trans= min(TransactionAmt), max_acc_Trans = max(TransactionAmt),
                                               median_acc_Trans= median(TransactionAmt), skewness_acc_Trans = skewness(TransactionAmt), kurtosis_acc_Trans = kurtosis(TransactionAmt)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~card1, summarise, mean_card1_Trans= mean(TransactionAmt), sd_card1_Trans = sd(TransactionAmt),  min_card1_Trans= min(TransactionAmt), max_card1_Trans = max(TransactionAmt),
                                               median_card1_Trans= median(TransactionAmt), skewness_card1_Trans = skewness(TransactionAmt), kurtosis_card1_Trans = kurtosis(TransactionAmt)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~card2, summarise, mean_card2_Trans= mean(TransactionAmt), sd_card2_Trans = sd(TransactionAmt),  min_card2_Trans= min(TransactionAmt), max_card2_Trans = max(TransactionAmt),
                                               median_card2_Trans= median(TransactionAmt), skewness_card2_Trans = skewness(TransactionAmt), kurtosis_card2_Trans = kurtosis(TransactionAmt)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~card3, summarise, mean_card3_Trans= mean(TransactionAmt), sd_card3_Trans = sd(TransactionAmt),  min_card3_Trans= min(TransactionAmt), max_card3_Trans = max(TransactionAmt),
                                               median_card3_Trans= median(TransactionAmt), skewness_card3_Trans = skewness(TransactionAmt), kurtosis_card3_Trans = kurtosis(TransactionAmt)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~card4, summarise, mean_card4_Trans= mean(TransactionAmt), sd_card4_Trans = sd(TransactionAmt),  min_card4_Trans= min(TransactionAmt), max_card4_Trans = max(TransactionAmt),
                                               median_card4_Trans= median(TransactionAmt), skewness_card4_Trans = skewness(TransactionAmt), kurtosis_card4_Trans = kurtosis(TransactionAmt)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~id_02, summarise, mean_id_02_Trans= mean(TransactionAmt), sd_id_02_Trans = sd(TransactionAmt),  min_id_02_Trans= min(TransactionAmt), max_id_02_Trans = max(TransactionAmt),
                                               median_id_02_Trans= median(TransactionAmt), skewness_id_02_Trans = skewness(TransactionAmt), kurtosis_id_02_Trans = kurtosis(TransactionAmt)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~D15, summarise, mean_D15_Trans= mean(TransactionAmt), sd_D15_Trans = sd(TransactionAmt),  min_D15_Trans= min(TransactionAmt), max_D15_Trans = max(TransactionAmt),
                                               median_D15_Trans= median(TransactionAmt), skewness_D15_Trans = skewness(TransactionAmt), kurtosis_D15_Trans = kurtosis(TransactionAmt)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~addr1_addr2, summarise, mean_addr1_addr2_Trans= mean(TransactionAmt), sd_addr1_addr2_Trans = sd(TransactionAmt),  min_addr1_addr2_Trans= min(TransactionAmt), max_addr1_addr2_Trans = max(TransactionAmt),
                                               median_addr1_addr2_Trans= median(TransactionAmt), skewness_addr1_addr2_Trans = skewness(TransactionAmt), kurtosis_addr1_addr2_Trans = kurtosis(TransactionAmt)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~C13, summarise, mean_C13_Trans= mean(TransactionAmt), sd_C13_Trans = sd(TransactionAmt),  min_C13_Trans= min(TransactionAmt), max_C13_Trans = max(TransactionAmt),
                                               median_C13_Trans= median(TransactionAmt), skewness_C13_Trans = skewness(TransactionAmt), kurtosis_C13_Trans = kurtosis(TransactionAmt)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~V258, summarise, mean_V258_Trans= mean(TransactionAmt), sd_V258_Trans = sd(TransactionAmt),  min_V258_Trans= min(TransactionAmt), max_V258_Trans = max(TransactionAmt),
                                               median_V258_Trans= median(TransactionAmt), skewness_V258_Trans = skewness(TransactionAmt), kurtosis_V258_Trans = kurtosis(TransactionAmt)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~C1, summarise, mean_C1_Trans= mean(TransactionAmt), sd_C1_Trans = sd(TransactionAmt),  min_C1_Trans= min(TransactionAmt), max_C1_Trans = max(TransactionAmt),
                                               median_C1_Trans= median(TransactionAmt), skewness_C1_Trans = skewness(TransactionAmt), kurtosis_C1_Trans = kurtosis(TransactionAmt)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~C14, summarise, mean_C14_Trans= mean(TransactionAmt), sd_C14_Trans = sd(TransactionAmt),  min_C14_Trans= min(TransactionAmt), max_C14_Trans = max(TransactionAmt),
                                               median_C14_Trans= median(TransactionAmt), skewness_C14_Trans = skewness(TransactionAmt), kurtosis_C14_Trans = kurtosis(TransactionAmt)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~hour, summarise, mean_hour_Trans= mean(TransactionAmt), sd_hour_Trans = sd(TransactionAmt),  min_hour_Trans= min(TransactionAmt), max_hour_Trans = max(TransactionAmt),
                                               median_hour_Trans= median(TransactionAmt), skewness_hour_Trans = skewness(TransactionAmt), kurtosis_hour_Trans = kurtosis(TransactionAmt)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~weekday, summarise, mean_weekday_Trans= mean(TransactionAmt), sd_weekday_Trans = sd(TransactionAmt),  min_weekday_Trans= min(TransactionAmt), max_weekday_Trans = max(TransactionAmt),
                                               median_weekday_Trans= median(TransactionAmt), skewness_weekday_Trans = skewness(TransactionAmt), kurtosis_weekday_Trans = kurtosis(TransactionAmt)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~P_emaildomain__addr1, summarise, mean_P_emaildomain__addr1_Trans= mean(TransactionAmt), sd_P_emaildomain__addr1_Trans = sd(TransactionAmt),  min_P_emaildomain__addr1_Trans= min(TransactionAmt), max_P_emaildomain__addr1_Trans = max(TransactionAmt),
                                               median_P_emaildomain__addr1_Trans= median(TransactionAmt), skewness_P_emaildomain__addr1_Trans = skewness(TransactionAmt), kurtosis_P_emaildomain__addr1_Trans = kurtosis(TransactionAmt)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~R_emaildomain__addr1, summarise, mean_R_emaildomain__addr1_Trans= mean(TransactionAmt), sd_R_emaildomain__addr1_Trans = sd(TransactionAmt),  min_R_emaildomain__addr1_Trans= min(TransactionAmt), max_R_emaildomain__addr1_Trans = max(TransactionAmt),
                                               median_R_emaildomain__addr1_Trans= median(TransactionAmt), skewness_R_emaildomain__addr1_Trans = skewness(TransactionAmt), kurtosis_R_emaildomain__addr1_Trans = kurtosis(TransactionAmt)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~PRmailTF, summarise, mean_PRmailTF_Trans= mean(TransactionAmt), sd_PRmailTF_Trans = sd(TransactionAmt),  min_PRmailTF_Trans= min(TransactionAmt), max_PRmailTF_Trans = max(TransactionAmt),
                                               median_PRmailTF_Trans= median(TransactionAmt), skewness_PRmailTF_Trans = skewness(TransactionAmt), kurtosis_PRmailTF_Trans = kurtosis(TransactionAmt)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~addr1__card1, summarise, mean_addr1__card1_Trans= mean(TransactionAmt), sd_addr1__card1_Trans = sd(TransactionAmt),  min_addr1__card1_Trans= min(TransactionAmt), max_addr1__card1_Trans = max(TransactionAmt),
                                               median_addr1__card1_Trans= median(TransactionAmt), skewness_addr1__card1_Trans = skewness(TransactionAmt), kurtosis_addr1__card1_Trans = kurtosis(TransactionAmt)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~card1__P_emaildomain, summarise, mean_card1__P_emaildomain_Trans= mean(TransactionAmt), sd_card1__P_emaildomain_Trans = sd(TransactionAmt),  min_card1__P_emaildomain_Trans= min(TransactionAmt), max_card1__P_emaildomain_Trans = max(TransactionAmt),
                                               median_card1__P_emaildomain_Trans= median(TransactionAmt), skewness_card1__P_emaildomain_Trans = skewness(TransactionAmt), kurtosis_card1__P_emaildomain_Trans = kurtosis(TransactionAmt)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~card5__P_emaildomain, summarise, mean_card5__P_emaildomain_Trans= mean(TransactionAmt), sd_card5__P_emaildomain_Trans = sd(TransactionAmt),  min_card5__P_emaildomain_Trans= min(TransactionAmt), max_card5__P_emaildomain_Trans = max(TransactionAmt),
                                               median_card5__P_emaildomain_Trans= median(TransactionAmt), skewness_card5__P_emaildomain_Trans = skewness(TransactionAmt), kurtosis_card5__P_emaildomain_Trans = kurtosis(TransactionAmt)))




fe_part2 = fe_part2 %>% left_join(plyr::ddply(fe_part2, ~acc, summarise, mean_acc_dec = mean(decimal,na.rm = T), sd_acc_dec = sd(decimal,na.rm = T), acc_dec_q = quantile(decimal,.8,na.rm = T),
                                              median_acc_dec= median(decimal,na.rm = T), skewness_acc_dec = skewness(decimal,na.rm = T), kurtosis_acc_dec = kurtosis(decimal,na.rm = T)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~card1, summarise, mean_card1_decimal= mean(decimal,na.rm = T), sd_card1_decimal = sd(decimal,na.rm = T),  min_card1_decimal= min(decimal,na.rm = T), max_card1_decimal = max(decimal,na.rm = T),
                                               median_card1_decimal= median(decimal,na.rm = T), skewness_card1_decimal = skewness(decimal,na.rm = T), kurtosis_card1_decimal = kurtosis(decimal,na.rm = T)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~card2, summarise, mean_card2_decimal= mean(decimal,na.rm = T), sd_card2_decimal = sd(decimal,na.rm = T),  min_card2_decimal= min(decimal,na.rm = T), max_card2_decimal = max(decimal,na.rm = T),
                                               median_card2_decimal= median(decimal,na.rm = T), skewness_card2_decimal = skewness(decimal,na.rm = T), kurtosis_card2_decimal = kurtosis(decimal,na.rm = T)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~card3, summarise, mean_card3_decimal= mean(decimal,na.rm = T), sd_card3_decimal = sd(decimal,na.rm = T),  min_card3_decimal= min(decimal,na.rm = T), max_card3_decimal = max(decimal,na.rm = T),
                                               median_card3_decimal= median(decimal,na.rm = T), skewness_card3_decimal = skewness(decimal,na.rm = T), kurtosis_card3_decimal = kurtosis(decimal,na.rm = T)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~card4, summarise, mean_card4_decimal= mean(decimal,na.rm = T), sd_card4_decimal = sd(decimal,na.rm = T),  min_card4_decimal= min(decimal,na.rm = T), max_card4_decimal = max(decimal,na.rm = T),
                                               median_card4_decimal= median(decimal,na.rm = T), skewness_card4_decimal = skewness(decimal,na.rm = T), kurtosis_card4_decimal = kurtosis(decimal,na.rm = T)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~id_02, summarise, mean_id_02_decimal= mean(decimal,na.rm = T), sd_id_02_decimal = sd(decimal,na.rm = T),  min_id_02_decimal= min(decimal,na.rm = T), max_id_02_decimal = max(decimal,na.rm = T),
                                               median_id_02_decimal= median(decimal,na.rm = T), skewness_id_02_decimal = skewness(decimal,na.rm = T), kurtosis_id_02_decimal = kurtosis(decimal,na.rm = T)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~D15, summarise, mean_D15_decimal= mean(decimal,na.rm = T), sd_D15_decimal = sd(decimal,na.rm = T),  min_D15_decimal= min(decimal,na.rm = T), max_D15_decimal = max(decimal,na.rm = T),
                                               median_D15_decimal= median(decimal,na.rm = T), skewness_D15_decimal = skewness(decimal,na.rm = T), kurtosis_D15_decimal = kurtosis(decimal,na.rm = T)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~addr1_addr2, summarise, mean_addr1_addr2_decimal= mean(decimal,na.rm = T), sd_addr1_addr2_decimal = sd(decimal,na.rm = T),  min_addr1_addr2_decimal= min(decimal,na.rm = T), max_addr1_addr2_decimal = max(decimal,na.rm = T),
                                               median_addr1_addr2_decimal= median(decimal,na.rm = T), skewness_addr1_addr2_decimal = skewness(decimal,na.rm = T), kurtosis_addr1_addr2_decimal = kurtosis(decimal,na.rm = T)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~C13, summarise, mean_C13_decimal= mean(decimal,na.rm = T), sd_C13_decimal = sd(decimal,na.rm = T),  min_C13_decimal= min(decimal,na.rm = T), max_C13_decimal = max(decimal,na.rm = T),
                                               median_C13_decimal= median(decimal,na.rm = T), skewness_C13_decimal = skewness(decimal,na.rm = T), kurtosis_C13_decimal = kurtosis(decimal,na.rm = T)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~V258, summarise, mean_V258_decimal= mean(decimal,na.rm = T), sd_V258_decimal = sd(decimal,na.rm = T),  min_V258_decimal= min(decimal,na.rm = T), max_V258_decimal = max(decimal,na.rm = T),
                                               median_V258_decimal= median(decimal,na.rm = T), skewness_V258_decimal = skewness(decimal,na.rm = T), kurtosis_V258_decimal = kurtosis(decimal,na.rm = T)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~C1, summarise, mean_C1_decimal= mean(decimal,na.rm = T), sd_C1_decimal = sd(decimal,na.rm = T),  min_C1_decimal= min(decimal,na.rm = T), max_C1_decimal = max(decimal,na.rm = T),
                                               median_C1_decimal= median(decimal,na.rm = T), skewness_C1_decimal = skewness(decimal,na.rm = T), kurtosis_C1_decimal = kurtosis(decimal,na.rm = T)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~C14, summarise, mean_C14_decimal= mean(decimal,na.rm = T), sd_C14_decimal = sd(decimal,na.rm = T),  min_C14_decimal= min(decimal,na.rm = T), max_C14_decimal = max(decimal,na.rm = T),
                                               median_C14_decimal= median(decimal,na.rm = T), skewness_C14_decimal = skewness(decimal,na.rm = T), kurtosis_C14_decimal = kurtosis(decimal,na.rm = T)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~hour, summarise, mean_hour_decimal= mean(decimal,na.rm = T), sd_hour_decimal = sd(decimal,na.rm = T),  min_hour_decimal= min(decimal,na.rm = T), max_hour_decimal = max(decimal,na.rm = T),
                                               median_hour_decimal= median(decimal,na.rm = T), skewness_hour_decimal = skewness(decimal,na.rm = T), kurtosis_hour_decimal = kurtosis(decimal,na.rm = T)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~weekday, summarise, mean_weekday_decimal= mean(decimal,na.rm = T), sd_weekday_decimal = sd(decimal,na.rm = T),  min_weekday_decimal= min(decimal,na.rm = T), max_weekday_decimal = max(decimal,na.rm = T),
                                               median_weekday_decimal= median(decimal,na.rm = T), skewness_weekday_decimal = skewness(decimal,na.rm = T), kurtosis_weekday_decimal = kurtosis(decimal,na.rm = T)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~P_emaildomain__addr1, summarise, mean_P_emaildomain__addr1_decimal= mean(decimal,na.rm = T), sd_P_emaildomain__addr1_decimal = sd(decimal,na.rm = T),  min_P_emaildomain__addr1_decimal= min(decimal,na.rm = T), max_P_emaildomain__addr1_decimal = max(decimal,na.rm = T),
                                               median_P_emaildomain__addr1_decimal= median(decimal,na.rm = T), skewness_P_emaildomain__addr1_decimal = skewness(decimal,na.rm = T), kurtosis_P_emaildomain__addr1_decimal = kurtosis(decimal,na.rm = T)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~R_emaildomain__addr1, summarise, mean_R_emaildomain__addr1_decimal= mean(decimal,na.rm = T), sd_R_emaildomain__addr1_decimal = sd(decimal,na.rm = T),  min_R_emaildomain__addr1_decimal= min(decimal,na.rm = T), max_R_emaildomain__addr1_decimal = max(decimal,na.rm = T),
                                               median_R_emaildomain__addr1_decimal= median(decimal,na.rm = T), skewness_R_emaildomain__addr1_decimal = skewness(decimal,na.rm = T), kurtosis_R_emaildomain__addr1_decimal = kurtosis(decimal,na.rm = T)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~PRmailTF, summarise, mean_PRmailTF_decimal= mean(decimal,na.rm = T), sd_PRmailTF_decimal = sd(decimal,na.rm = T),  min_PRmailTF_decimal= min(decimal,na.rm = T), max_PRmailTF_decimal = max(decimal,na.rm = T),
                                               median_PRmailTF_decimal= median(decimal,na.rm = T), skewness_PRmailTF_decimal = skewness(decimal,na.rm = T), kurtosis_PRmailTF_decimal = kurtosis(decimal,na.rm = T)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~addr1__card1, summarise, mean_addr1__card1_decimal= mean(decimal,na.rm = T), sd_addr1__card1_decimal = sd(decimal,na.rm = T),  min_addr1__card1_decimal= min(decimal,na.rm = T), max_addr1__card1_decimal = max(decimal,na.rm = T),
                                               median_addr1__card1_decimal= median(decimal,na.rm = T), skewness_addr1__card1_decimal = skewness(decimal,na.rm = T), kurtosis_addr1__card1_decimal = kurtosis(decimal,na.rm = T)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~card1__P_emaildomain, summarise, mean_card1__P_emaildomain_decimal= mean(decimal,na.rm = T), sd_card1__P_emaildomain_decimal = sd(decimal,na.rm = T),  min_card1__P_emaildomain_decimal= min(decimal,na.rm = T), max_card1__P_emaildomain_decimal = max(decimal,na.rm = T),
                                               median_card1__P_emaildomain_decimal= median(decimal,na.rm = T), skewness_card1__P_emaildomain_decimal = skewness(decimal,na.rm = T), kurtosis_card1__P_emaildomain_decimal = kurtosis(decimal,na.rm = T)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~card5__P_emaildomain, summarise, mean_card5__P_emaildomain_decimal= mean(decimal,na.rm = T), sd_card5__P_emaildomain_decimal = sd(decimal,na.rm = T),  min_card5__P_emaildomain_decimal= min(decimal,na.rm = T), max_card5__P_emaildomain_decimal = max(decimal,na.rm = T),
                                               median_card5__P_emaildomain_decimal= median(decimal,na.rm = T), skewness_card5__P_emaildomain_decimal = skewness(decimal,na.rm = T), kurtosis_card5__P_emaildomain_decimal = kurtosis(decimal,na.rm = T)))



# %% [code]
head(fe_part2)
head(unique(fe_part2$mean_card1_D15))
head(unique(fe_part2$mean_card1_id02))
fe_part2 <- fe_part2[, -c(1:24)]

# FE part3: email binning for purchaser and Recipiant

#proton mail is sketch
fe_part3 <- tem[, c("P_emaildomain", "R_emaildomain")]

#email bin function
bin_email <- function(df, grouped, P_colname, R_colname){
  
  typeof(df)
  df$P_placeholder <- 0
  df$R_placeholder <- 0
  
  names(df)[names(df) == "P_placeholder"] <- P_colname
  names(df)[names(df) == "R_placeholder"] <- R_colname
  
  df[df$P_emaildomain %in% grouped, P_colname] <- 1
  df[df$R_emaildomain %in% grouped, R_colname] <- 1
  
  print(paste0(nrow(df[df[, P_colname] == 1,])*100/nrow(df), " % total transactions are ", P_colname, " for Purchaser"))
  print(paste0(nrow(df[df[, R_colname] == 1,])*100/nrow(df), " % total transactions are ", R_colname, " for Recipiant"))
  
  return(df)
}

#is Yahoo
a<- c("yahoo.fr", "yahoo.de", "yahoo.es", "yahoo.co.uk", "yahoo.com", "yahoo.com.mx", "ymail.com", "rocketmail.com", "frontiernet.net")
fe_part3 <- bin_email(fe_part3,a, "P_isyahoo", "R_isyahoo")

#is Microsoft
b<- c("hotmail.com", "live.com.mx", "live.com", "msn.com", "hotmail.es", "outlook.es", "hotmail.fr", "hotmail.de", "hotmail.co.uk")
fe_part3 <- bin_email(fe_part3,b, "P_ismfst", "R_ismfst")

#is apple icloud / mac / me -> apple
c<- c("icloud.com", "mac.com", "me.com")
fe_part3 <- bin_email(fe_part3,c, "P_ismac", "R_ismac")

#is att
d <- c("prodigy.net.mx", "att.net", "sbxglobal.net")
fe_part3 <- bin_email(fe_part3,d, "P_isatt", "R_isatt")

#iscenturylink
e <- c("centurylink.net", "embarqmail.com", "q.com")
fe_part3 <- bin_email(fe_part3,e, "P_iscenturylink", "R_iscenturylink")

#isaol
f <- c("aim.com", "aol.com")
fe_part3 <- bin_email(fe_part3,f, "P_isaol", "R_isaol")

#isspectrum
g <- c("twc.com", "charter.com")
fe_part3 <- bin_email(fe_part3,g, "P_isspectrum", "R_isspectrum")

#isproton
h <- c("protonmail.com")
fe_part3 <- bin_email(fe_part3,h, "P_isproton", "R_isproton")

#iscomcast
i <- c("comcast.net")
fe_part3 <- bin_email(fe_part3,i, "P_iscomcast", "R_iscomcast")

#isgoogle
j <- c("gmail.com")
fe_part3 <- bin_email(fe_part3,j, "P_isgoogle", "R_isgoogle")

#isanonynous
k <- c("anonymous.com")
fe_part3 <- bin_email(fe_part3,k, "P_isanon", "R_isanon")

#isNA
l <- NA
fe_part3 <- bin_email(fe_part3,l, "P_isNA", "R_isNA")

#-c(a,b,c,d,e,f,g,h,i, j, k, l) remaining bins

fe_part3 <- fe_part3[, -c(1,2)]
#############################################################################################################

tem$DeviceType[is_blank(tem$DeviceType)] <- gsub("", "unk", tem$DeviceType[is_blank(tem$DeviceType)])
tem$DeviceInfo[is_blank(tem$DeviceInfo)] <- gsub("", "unk", tem$DeviceInfo[is_blank(tem$DeviceInfo)])

tem$DeviceInfo <- gsub("SAMSUNG ", "", tem$DeviceInfo)
tem$DeviceInfo <- gsub("Lenovo ", "", tem$DeviceInfo)
tem$DeviceInfo <- gsub("SAMSUNG-", "", tem$DeviceInfo)
tem$DeviceInfo <- gsub("HUAWEI ", "", tem$DeviceInfo)
tem$DeviceInfo <- gsub("ZTE ", "", tem$DeviceInfo)
tem$DeviceInfo <- gsub("Hisense ", "", tem$DeviceInfo)
tem$DeviceInfo <- gsub("Blade ", "", tem$DeviceInfo)
tem$DeviceInfo <- gsub("Moto ", "", tem$DeviceInfo)
tem$DeviceInfo <- gsub("HTC ", "", tem$DeviceInfo)
tem$DeviceInfo <- gsub("M4 ", "", tem$DeviceInfo)
tem$DeviceInfo <- gsub("iPhone", "iOS Device", tem$DeviceInfo)
tem$DeviceInfo <- gsub("Microsoft", "Windows", tem$DeviceInfo)



#FE part4: count encoding of base features
char_features <- tem[,colnames(tem) %in% 
                       c("ProductCD","card1","card2","card3","card4","card5","card6","addr1","addr2","P_emaildomain",
                         "R_emaildomain","M1","M2","M3","M4","M5","M6","M7","M8","M9","DeviceType","DeviceInfo","id_12",
                         "id_13","id_14","id_15","id_16","id_17","id_18","id_19","id_20","id_21","id_22","id_23","id_24",
                         "id_25","id_26","id_27","id_28","id_29","id_30","id_31","id_32","id_33","id_34","id_35","id_36",
                         "id_37","id_38", "card1__dist1", "card1__P_emaildomain", "card1__R_emaildomain", "addr1__card1",
                         "card4__dist1", "addr1__card4", "P_emaildomain__C2", "R_emaildomain__C2", "addr1__card2",  "P_emaildomain__addr1", "R_emaildomain__addr1",
                         "card1__card5", "card5__P_emaildomain", "card5__R_emaildomain", "acc", "DeviceInfo", "hour", "weekday", "PRmailTF")]

fe_part4 <- data.frame(0)
for(a in colnames(char_features) ){
  tem1 <- char_features %>% group_by(.dots = a) %>% mutate(count = length(card4)) %>% ungroup() %>% select(count)
  colnames(tem1) <- paste(a,"__count_encoding",sep="")
  fe_part4 <- data.frame(fe_part4,tem1)
}


fe_part4 <- fe_part4[,-1]
rm(char_features,tem1) ; invisible(gc())
cat("fe_part4 ncol :" , ncol(fe_part4) ,"\n" )

# %% [code]
#tem <- data.frame(tem,fe_part1, fe_part2, fe_part3, fe_part4)

#############################################################################################################

# label 
char_f <-  c("ProductCD","card1","card2","card3","card4","card5","card6","addr1","addr2","P_emaildomain",
             "R_emaildomain","M2","M3","M4","M5","M6","M7","M8","M9", "card1__dist1", "card1__P_emaildomain", "card1__R_emaildomain",
             "addr1__card1", "card4__dist1", "addr1__card4", "P_emaildomain__C2", "addr1__card2",  "P_emaildomain__addr1", "R_emaildomain__addr1",
             "card1__card5", "card5__P_emaildomain", "card5__R_emaildomain", "DeviceInfo", "hour", "weekday", "PRmailTF")

tem <- tem %>% mutate_at(char_f, function(x){
  y <- as.factor(x)
  y <- as.numeric(y)
  y <- y - 1
  return(y)
})

tem <- data.frame(tem,fe_part1, fe_part2, fe_part3, fe_part4)

rm(fe_part1, fe_part2, fe_part3, fe_part4); invisible(gc());

tem <- tem %>% group_by(addr1, card1) %>% mutate(mean_TransactionAmt_addr1_card1 = mean(TransactionAmt, na.rm = TRUE),
                                                 median_TransactionAmt_addr1_card1 = median(TransactionAmt, na.rm = TRUE),
                                                 skewness_TransactionAmt_addr1_card1 = skewness(TransactionAmt, na.rm = TRUE),
                                                 kurtosis_TransactionAmt_addr1_card1 = kurtosis(TransactionAmt, na.rm = TRUE),
                                                 sd_TransactionAmt_addr1_card1 = sd(TransactionAmt, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(TransactionAmt_to_mean_addr1_card1 = TransactionAmt/mean_TransactionAmt_addr1_card1,
         TransactionAmt_to_sd_addr1_card1 = TransactionAmt/sd_TransactionAmt_addr1_card1,
         TransactionAmt_to_skewness_addr1_card1 = TransactionAmt/skewness_TransactionAmt_addr1_card1,
         TransactionAmt_to_kurtosis_addr1_card1 = TransactionAmt/kurtosis_TransactionAmt_addr1_card1,
         TransactionAmt_subs_addr1_card1 = TransactionAmt - mean_TransactionAmt_addr1_card1,
         TransactionAmt_subs_addr1_card1_median = TransactionAmt - median_TransactionAmt_addr1_card1)

tem <- tem %>% group_by(addr1, card2) %>% mutate(mean_TransactionAmt_addr1_card2 = mean(TransactionAmt, na.rm = TRUE),
                                                 median_TransactionAmt_addr1_card2 = median(TransactionAmt, na.rm = TRUE),
                                                 skewness_TransactionAmt_addr1_card2 = skewness(TransactionAmt, na.rm = TRUE),
                                                 kurtosis_TransactionAmt_addr1_card2 = kurtosis(TransactionAmt, na.rm = TRUE),
                                                 sd_TransactionAmt_addr1_card2 = sd(TransactionAmt, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(TransactionAmt_to_mean_addr1_card2 = TransactionAmt/mean_TransactionAmt_addr1_card2,
         TransactionAmt_to_sd_addr1_card2 = TransactionAmt/sd_TransactionAmt_addr1_card2,
         TransactionAmt_to_skewness_addr1_card2 = TransactionAmt/skewness_TransactionAmt_addr1_card2,
         TransactionAmt_to_kurtosis_addr1_card2 = TransactionAmt/kurtosis_TransactionAmt_addr1_card2,
         TransactionAmt_subs_addr1_card2 = TransactionAmt - mean_TransactionAmt_addr1_card2,
         TransactionAmt_subs_addr1_card2_median = TransactionAmt - median_TransactionAmt_addr1_card2)

tem <- tem %>% group_by(card5, card1) %>% mutate(mean_TransactionAmt_card5_card1 = mean(TransactionAmt, na.rm = TRUE),
                                                 median_TransactionAmt_card5_card1 = median(TransactionAmt, na.rm = TRUE),
                                                 skewness_TransactionAmt_card5_card1 = skewness(TransactionAmt, na.rm = TRUE),
                                                 kurtosis_TransactionAmt_card5_card1 = kurtosis(TransactionAmt, na.rm = TRUE),
                                                 sd_TransactionAmt_card5_card1 = sd(TransactionAmt, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(TransactionAmt_to_mean_card5_card1 = TransactionAmt/mean_TransactionAmt_card5_card1,
         TransactionAmt_to_sd_card5_card1 = TransactionAmt/sd_TransactionAmt_card5_card1,
         TransactionAmt_to_skewness_card5_card1 = TransactionAmt/skewness_TransactionAmt_card5_card1,
         TransactionAmt_to_kurtosis_card5_card1 = TransactionAmt/kurtosis_TransactionAmt_card5_card1,
         TransactionAmt_subs_card5_card1 = TransactionAmt - mean_TransactionAmt_card5_card1,
         TransactionAmt_subs_card5_card1_median = TransactionAmt - median_TransactionAmt_card5_card1)

tem <- tem %>% group_by(hour, card1) %>% mutate(mean_TransactionAmt_hour_card1 = mean(TransactionAmt, na.rm = TRUE),
                                                median_TransactionAmt_hour_card1 = median(TransactionAmt, na.rm = TRUE),
                                                skewness_TransactionAmt_hour_card1 = skewness(TransactionAmt, na.rm = TRUE),
                                                kurtosis_TransactionAmt_hour_card1 = kurtosis(TransactionAmt, na.rm = TRUE),
                                                sd_TransactionAmt_hour_card1 = sd(TransactionAmt, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(TransactionAmt_to_mean_hour_card1 = TransactionAmt/mean_TransactionAmt_hour_card1,
         TransactionAmt_to_sd_hour_card1 = TransactionAmt/sd_TransactionAmt_hour_card1,
         TransactionAmt_to_skewness_hour_card1 = TransactionAmt/skewness_TransactionAmt_hour_card1,
         TransactionAmt_to_kurtosis_hour_card1 = TransactionAmt/kurtosis_TransactionAmt_hour_card1,
         TransactionAmt_subs_hour_card1 = TransactionAmt - mean_TransactionAmt_hour_card1,
         TransactionAmt_subs_hour_card1_median = TransactionAmt - median_TransactionAmt_hour_card1)


tem <- tem %>% group_by(weekday, card1) %>% mutate(mean_TransactionAmt_weekday_card1 = mean(TransactionAmt, na.rm = TRUE),
                                                   median_TransactionAmt_weekday_card1 = median(TransactionAmt, na.rm = TRUE),
                                                   skewness_TransactionAmt_weekday_card1 = skewness(TransactionAmt, na.rm = TRUE),
                                                   kurtosis_TransactionAmt_weekday_card1 = kurtosis(TransactionAmt, na.rm = TRUE),
                                                   sd_TransactionAmt_weekday_card1 = sd(TransactionAmt, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(TransactionAmt_to_mean_weekday_card1 = TransactionAmt/mean_TransactionAmt_weekday_card1,
         TransactionAmt_to_sd_weekday_card1 = TransactionAmt/sd_TransactionAmt_weekday_card1,
         TransactionAmt_to_skewness_weekday_card1 = TransactionAmt/skewness_TransactionAmt_weekday_card1,
         TransactionAmt_to_kurtosis_weekday_card1 = TransactionAmt/kurtosis_TransactionAmt_weekday_card1,
         TransactionAmt_subs_weekday_card1 = TransactionAmt - mean_TransactionAmt_weekday_card1,
         TransactionAmt_subs_weekday_card1_median = TransactionAmt - median_TransactionAmt_weekday_card1)

tem <- tem %>% group_by(PRmailTF, card1) %>% mutate(mean_TransactionAmt_PRmailTF_card1 = mean(TransactionAmt, na.rm = TRUE),
                                                    median_TransactionAmt_PRmailTF_card1 = median(TransactionAmt, na.rm = TRUE),
                                                    skewness_TransactionAmt_PRmailTF_card1 = skewness(TransactionAmt, na.rm = TRUE),
                                                    kurtosis_TransactionAmt_PRmailTF_card1 = kurtosis(TransactionAmt, na.rm = TRUE),
                                                    sd_TransactionAmt_PRmailTF_card1 = sd(TransactionAmt, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(TransactionAmt_to_mean_PRmailTF_card1 = TransactionAmt/mean_TransactionAmt_PRmailTF_card1,
         TransactionAmt_to_sd_PRmailTF_card1 = TransactionAmt/sd_TransactionAmt_PRmailTF_card1,
         TransactionAmt_to_skewness_PRmailTF_card1 = TransactionAmt/skewness_TransactionAmt_PRmailTF_card1,
         TransactionAmt_to_kurtosis_PRmailTF_card1 = TransactionAmt/kurtosis_TransactionAmt_PRmailTF_card1,
         TransactionAmt_subs_PRmailTF_card1 = TransactionAmt - mean_TransactionAmt_PRmailTF_card1,
         TransactionAmt_subs_PRmailTF_card1_median = TransactionAmt - median_TransactionAmt_PRmailTF_card1)


tem <- tem %>% group_by(card5, card1) %>% mutate(mean_D1_card5_card1 = mean(D1, na.rm = TRUE),
                                                 median_D1_card5_card1 = median(D1, na.rm = TRUE),
                                                 skewness_D1_card5_card1 = skewness(D1, na.rm = TRUE),
                                                 kurtosis_D1_card5_card1 = kurtosis(D1, na.rm = TRUE),
                                                 sd_D1_card5_card1 = sd(D1, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(D1_to_mean_card5_card1 = D1/mean_D1_card5_card1,
         D1_to_median_card5_card1 = D1/median_D1_card5_card1,
         D1_to_sd_card5_card1 = D1/sd_D1_card5_card1,
         D1_to_skewness_card5_card1 = D1/skewness_D1_card5_card1,
         D1_to_kurtosis_card5_card1 = D1/kurtosis_D1_card5_card1,
         D1_subs_card5_card1 = D1 - mean_D1_card5_card1,
         D1_subs_card5_card1_median = D1 - median_D1_card5_card1)

tem <- tem %>% group_by(card5, card1) %>% mutate(mean_D2_card5_card1 = mean(D2, na.rm = TRUE),
                                                 median_D2_card5_card1 = median(D2, na.rm = TRUE),
                                                 skewness_D2_card5_card1 = skewness(D2, na.rm = TRUE),
                                                 kurtosis_D2_card5_card1 = kurtosis(D2, na.rm = TRUE),
                                                 sd_D2_card5_card1 = sd(D2, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(D2_to_mean_card5_card1 = D2/mean_D2_card5_card1,
         D2_to_median_card5_card1 = D2/median_D2_card5_card1,
         D2_to_sd_card5_card1 = D2/sd_D2_card5_card1 ,
         D2_to_skewness_card5_card1 = D2/skewness_D2_card5_card1,
         D2_to_kurtosis_card5_card1 = D2/kurtosis_D2_card5_card1,
         D2_subs_card5_card1 = D2 - mean_D2_card5_card1,
         D2_subs_card5_card1_median = D2 - median_D2_card5_card1)

tem <- tem %>% group_by(card5, card1) %>% mutate(mean_D15_card5_card1 = mean(D15, na.rm = TRUE),
                                                 median_D15_card5_card1 = median(D15, na.rm = TRUE),
                                                 skewness_D15_card5_card1 = skewness(D15, na.rm = TRUE),
                                                 kurtosis_D15_card5_card1 = kurtosis(D15, na.rm = TRUE),
                                                 sd_D15_card5_card1 = sd(D15, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(D15_to_mean_card5_card1 = D15/mean_D15_card5_card1,
         D15_to_median_card5_card1 = D15/median_D15_card5_card1,
         D15_to_sd_card5_card1 = D15/sd_D15_card5_card1,
         D15_to_skewness_card5_card1 = D15/skewness_D15_card5_card1,
         D15_to_kurtosis_card5_card1 = D15/kurtosis_D15_card5_card1,
         D15_subs_card5_card1 = D15 - mean_D15_card5_card1,
         D15_subs_card5_card1_median = D15 - median_D15_card5_card1)

tem <- tem %>% group_by(P_emaildomain) %>% mutate(mean_TransactionAmt_P_emaildomain = mean(TransactionAmt, na.rm = TRUE),
                                                  median_TransactionAmt_P_emaildomain = median(TransactionAmt, na.rm = TRUE),
                                                  skewness_TransactionAmt_P_emaildomain = skewness(TransactionAmt, na.rm = TRUE),
                                                  kurtosis_TransactionAmt_P_emaildomain = kurtosis(TransactionAmt, na.rm = TRUE),
                                                  sd_TransactionAmt_P_emaildomain = sd(TransactionAmt, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(TransactionAmt_to_mean_P_emaildomain = TransactionAmt/mean_TransactionAmt_P_emaildomain,
         TransactionAmt_to_sd_P_emaildomain = TransactionAmt/sd_TransactionAmt_P_emaildomain,
         TransactionAmt_to_median_P_emaildomain = TransactionAmt/median_TransactionAmt_P_emaildomain,
         TransactionAmt_to_skewness_P_emaildomain = TransactionAmt/skewness_TransactionAmt_P_emaildomain,
         TransactionAmt_to_kurtosis_P_emaildomain = TransactionAmt/kurtosis_TransactionAmt_P_emaildomain,
         TransactionAmt_subs_P_emaildomain = TransactionAmt - mean_TransactionAmt_P_emaildomain)

tem <- tem %>% group_by(R_emaildomain) %>% mutate(mean_TransactionAmt_R_emaildomain = mean(TransactionAmt, na.rm = TRUE),
                                                  median_TransactionAmt_R_emaildomain = median(TransactionAmt, na.rm = TRUE),
                                                  skewness_TransactionAmt_R_emaildomain = skewness(TransactionAmt, na.rm = TRUE),
                                                  kurtosis_TransactionAmt_R_emaildomain = kurtosis(TransactionAmt, na.rm = TRUE),
                                                  sd_TransactionAmt_R_emaildomain = sd(TransactionAmt, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(TransactionAmt_to_mean_R_emaildomain = TransactionAmt/mean_TransactionAmt_R_emaildomain,
         TransactionAmt_to_sd_R_emaildomain = TransactionAmt/sd_TransactionAmt_R_emaildomain,
         TransactionAmt_to_skewness_R_emaildomain = TransactionAmt/skewness_TransactionAmt_R_emaildomain,
         TransactionAmt_to_kurtosis_R_emaildomain = TransactionAmt/kurtosis_TransactionAmt_R_emaildomain,
         TransactionAmt_subs_R_emaildomain = TransactionAmt - mean_TransactionAmt_R_emaildomain,
         TransactionAmt_subs_R_emaildomain_median = TransactionAmt - median_TransactionAmt_R_emaildomain)

tem <- tem %>% mutate(TransactionAmt = log(1 + TransactionAmt))

tem = tem[!duplicated(as.list(tem))]

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


tem$card1_and_count = paste(tem$card1,tem$count_card1)
tem$card2_and_count = paste(tem$card2,tem$count_card2)

tem$new3 = tem$TransactionAmt * tem$C1
tem$new5 = tem$TransactionAmt * tem$C13
tem$new7 = tem$TransactionAmt * tem$C14

tem$decimal_diff = tem$decimal - tem$mean_acc_dec

#fwrite(tem, "tem.csv")
#fread("tem.csv)

#set.seed(71)
#iso <- isolationForest$new()
#iso$fit(impute(x, method = "median/mode"))
#tem$iforest_score <- iso$scores$anomaly_score
#tem$iforest_depth <- iso$scores$average_depth

temp <- fread("tem_unsupervised.csv")

temp <- temp[, c(201, 63, 209, 183, 214, 176, 166, 15, 115, 210,
                 216, 198, 203, 35, 24, 8, 18, 66, 257:262)]

tem <- data.frame(tem, temp)
rm(temp)


# %% [code]
y_train <- y

train <- tem[1:nrow(train),]
test <- tem[-c(1:nrow(train)),]

#fwrite(tem, "tem.csv")
#fread("tem2.csv)

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
                  num_leaves = 197,
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

write.csv(sub,"sub_0.9377_80.csv",row.names = F)
write.csv(imp,"imp.csv",row.names = F)
