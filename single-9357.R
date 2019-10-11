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

options(warn=-1)
options(scipen = 99)

train_iden <- read_csv("train_identity.csv")
train_trans <- read_csv("train_transaction.csv")
test_iden <- read_csv("test_identity.csv")
test_trans <- read_csv("test_transaction.csv")

is_blank <- function(x) {is.na(x) | x == ""}

# %% [code]
y <- train_trans$isFraud 
train_trans$isFraud <- NULL
train <- train_trans %>% left_join(train_iden)
test <- test_trans %>% left_join(test_iden)

train$n_na <- rowSums(is.na(train))
test$n_na <- rowSums(is.na(test))

rm(train_iden,train_trans,test_iden,test_trans) ; invisible(gc())


# using single hold-out validation (20%)
tr_idx <- which(train$TransactionDT < quantile(train$TransactionDT, 0.8))


# create tem dataframe for both train and test data and engineer time of day feature
tem <- train %>% bind_rows(test) %>%
  mutate(hr = floor( (TransactionDT / 3600) %% 24 ),
         weekday = floor( (TransactionDT / 3600 / 24) %% 7)
  ) %>%
  select(-TransactionID,-TransactionDT)

tem <- tem %>%
  mutate(
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
    }))
  )

tem$acc = paste(tem$card1,tem$card2,tem$card3,tem$card4,tem$card5,tem$addr1,tem$addr2)
tem$addr1_addr2 = paste(tem$addr1,tem$addr2)
tem$decimal = nchar(tem$TransactionAmt - floor(tem$TransactionAmt))

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
                  "firefox 60.0", "edge 17.0", "chrome 69.0", "chrome 67.0", "chrome 63.0", "chrome 63.0", 
                  "chrome 64.0", "chrome 64.0 for android", "chrome 64.0 for ios", "chrome 65.0", "chrome 65.0 for android",
                  "chrome 65.0 for ios", "chrome 66.0", "chrome 66.0 for android", "chrome 66.0 for ios")

fe_part1[fe_part1$id_31 %in% new_browsers,] <- 1

paste0(nrow(fe_part1[fe_part1$latest_browser == 1 ,])*100/nrow(fe_part1), " % total rows with latest browsers")
paste0(nrow(fe_part1[fe_part1$latest_browser == 0 ,])*100/nrow(fe_part1), " % total rows with old browsers")

fe_part1 <- fe_part1[, -c(1)]

#FE part2: mean and sd transaction amount to card for card1, card4, id_02, D15
fe_part2 <- tem[, c("acc","card1","card2","card3", "card4", "TransactionAmt", "id_02", "D15","addr1_addr2","decimal","C13","V258","C1","C14")]

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~card1, summarise, mean_card1_Trans= mean(TransactionAmt), sd_card1_Trans = sd(TransactionAmt),
                                               median_card1_Trans= median(TransactionAmt), skewness_card1_Trans = skewness(TransactionAmt), kurtosis_card1_Trans = kurtosis(TransactionAmt)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~addr1_addr2, summarise, mean_geo_Trans= mean(TransactionAmt), sd_geo_Trans = sd(TransactionAmt),
                                               median_geo_Trans= median(TransactionAmt), skewness_geo_Trans = skewness(TransactionAmt), kurtosis_geo_Trans = kurtosis(TransactionAmt)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~card1, summarise, min_card1_Trans= min(TransactionAmt), max_card1_Trans = max(TransactionAmt),
                                               median_card1_Trans= median(TransactionAmt), skewness_card1_Trans = skewness(TransactionAmt), kurtosis_card1_Trans = kurtosis(TransactionAmt)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~card2, summarise, min_card2_Trans= min(TransactionAmt), max_card2_Trans = max(TransactionAmt),
                                               median_card2_Trans= median(TransactionAmt), skewness_card2_Trans = skewness(TransactionAmt), kurtosis_card2_Trans = kurtosis(TransactionAmt)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~card4, summarise, mean_card4_Trans = mean(TransactionAmt), sd_card4_Trans = sd(TransactionAmt),
                                               median_card4_Trans= median(TransactionAmt), skewness_card4_Trans = skewness(TransactionAmt), kurtosis_card4_Trans = kurtosis(TransactionAmt)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~acc, summarise, mean_acc_Trans = mean(TransactionAmt), sd_acc_Trans = sd(TransactionAmt),
                                               median_acc_Trans= median(TransactionAmt), skewness_acc_Trans = skewness(TransactionAmt), kurtosis_acc_Trans = kurtosis(TransactionAmt)))

fe_part2 = fe_part2 %>% left_join(plyr::ddply(fe_part2, ~acc, summarise, mean_acc_dec = mean(decimal,na.rm = T), sd_acc_dec = sd(decimal,na.rm = T), acc_dec_q = quantile(decimal,.8,na.rm = T),
                                              median_acc_dec= median(decimal,na.rm = T), skewness_acc_dec = skewness(decimal,na.rm = T), kurtosis_acc_dec = kurtosis(decimal,na.rm = T)))

# %% [code]
head(fe_part2)
head(unique(fe_part2$mean_card1_D15))
head(unique(fe_part2$mean_card1_id02))
fe_part2 <- fe_part2[, -c(1:14)]

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
                         "id_37","id_38", "card1__dist1", "card1__P_emaildomain", "addr1__card1",
                         "card4__dist1", "addr1__card4", "P_emaildomain__C2", "addr1__card2",
                         "card1__card5", "card5__P_emaildomain", "acc", "DeviceInfo")]

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
             "R_emaildomain","M2","M3","M4","M5","M6","M7","M8","M9", "card1__dist1", "card1__P_emaildomain",
             "addr1__card1", "card4__dist1", "addr1__card4", "P_emaildomain__C2", "addr1__card2",
             "card1__card5", "card5__P_emaildomain", "DeviceInfo")

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
         TransactionAmt_to_sd_addr1_card2 = TransactionAmt/skewness_TransactionAmt_addr1_card2,
         TransactionAmt_to_sd_addr1_card2 = TransactionAmt/kurtosis_TransactionAmt_addr1_card2,
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
         TransactionAmt_to_sd_card5_card1 = TransactionAmt/skewness_TransactionAmt_card5_card1,
         TransactionAmt_to_sd_card5_card1 = TransactionAmt/kurtosis_TransactionAmt_card5_card1,
         TransactionAmt_subs_card5_card1 = TransactionAmt - mean_TransactionAmt_card5_card1,
         TransactionAmt_subs_card5_card1_median = TransactionAmt - median_TransactionAmt_card5_card1)


tem <- tem %>% group_by(card5, card1) %>% mutate(mean_D2_card5_card1 = mean(D2, na.rm = TRUE),
                                                 median_D2_card5_card1 = median(D2, na.rm = TRUE),
                                                 skewness_D2_card5_card1 = skewness(D2, na.rm = TRUE),
                                                 kurtosis_D2_card5_card1 = kurtosis(D2, na.rm = TRUE),
                                                 sd_D2_card5_card5 = sd(D2, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(D2_to_mean_card5_card1 = D2/mean_D2_card5_card1,
         D2_to_sd_card5_card1 = D2/sd_TransactionAmt_card5_card1,
         D2_to_sd_card5_card1 = D2/skewness_TransactionAmt_card5_card1,
         D2_to_sd_card5_card1 = D2/kurtosis_TransactionAmt_card5_card1,
         D2_subs_card5_card1 = D2 - mean_D2_card5_card1,
         D2_subs_card5_card1_median = D2 - median_D2_card5_card1)

tem <- tem %>% group_by(P_emaildomain) %>% mutate(mean_TransactionAmt_P_emaildomain = mean(TransactionAmt, na.rm = TRUE),
                                                  sd_TransactionAmt_P_emaildomain = sd(TransactionAmt, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(TransactionAmt_to_mean_P_emaildomain = TransactionAmt/mean_TransactionAmt_P_emaildomain,
         TransactionAmt_to_sd_P_emaildomain = TransactionAmt/sd_TransactionAmt_P_emaildomain,
         TransactionAmt_subs_P_emaildomain = TransactionAmt - mean_TransactionAmt_P_emaildomain)

tem <- tem %>% group_by(P_emaildomain) %>% mutate(mean_TransactionAmt_P_emaildomain = mean(TransactionAmt, na.rm = TRUE),
                                                  median_TransactionAmt_P_emaildomain = median(TransactionAmt, na.rm = TRUE),
                                                  skewness_TransactionAmt_P_emaildomain = skewness(TransactionAmt, na.rm = TRUE),
                                                  kurtosis_TransactionAmt_P_emaildomain = kurtosis(TransactionAmt, na.rm = TRUE),
                                                  sd_TransactioAm_P_emaildomain = sd(TransactionAmt, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(TransactionAmt_to_mean_P_emaildomain = TransactionAmt/mean_TransactionAmt_P_emaildomain,
         TransactionAmt_to_sd_P_emaildomain = TransactionAmt/sd_TransactionAmt_P_emaildomain,
         TransactionAmt_to_sd_P_emaildomain = TransactionAmt/skewness_TransactionAmt_P_emaildomain,
         TransactionAmt_to_sd_P_emaildomain = TransactionAmt/kurtosis_TransactionAmt_P_emaildomain,
         TransactionAmt_subs_P_emaildomain = TransactionAmt - mean_TransactionAmt_P_emaildomain,
         TransactionAmt_subs_P_emaildomain_median = TransactionAmt - median_TransactionAmt_P_emaildomain)

tem <- tem %>% mutate(TransactionAmt = log(1 + TransactionAmt))

tem = tem[!duplicated(as.list(tem))]

tem$card1_and_count = paste(tem$card1,tem$count_card1)
tem$card2_and_count = paste(tem$card2,tem$count_card2)

tem$new3 = tem$TransactionAmt * tem$C1
tem$new5 = tem$TransactionAmt * tem$C13
tem$new7 = tem$TransactionAmt * tem$C14

tem$decimal_diff = tem$decimal - tem$mean_acc_dec
tem$new13 = tem$decimal_diff * tem$mean_acc_Trans

# Dim reduction of v-variables (groups of V variables - https://www.kaggle.com/abednadir/best-r-score)
part_1 <- names(tem)[which(names(tem) %in% paste0("V",1:11))]
part_2 <- names(tem)[which(names(tem) %in% paste0("V",12:34))]
part_3 <- names(tem)[which(names(tem) %in% paste0("V",35:52))]
part_4 <- names(tem)[which(names(tem) %in% paste0("V",53:74))]
part_5 <- names(tem)[which(names(tem) %in% paste0("V",75:94))]
part_6 <- names(tem)[which(names(tem) %in% paste0("V",95:137))]
part_7 <- names(tem)[which(names(tem) %in% paste0("V",138:166))]
part_8 <- names(tem)[which(names(tem) %in% paste0("V",167:216))]
part_9 <- names(tem)[which(names(tem) %in% paste0("V",217:278))]
part_10 <- names(tem)[which(names(tem) %in% paste0("V",279:321))]
part_11 <- names(tem)[which(names(tem) %in% paste0("V",322:339))]

for(j in c('part_1', 'part_2', 'part_3', 'part_4', 'part_5', 'part_6', 'part_7', 'part_8', 'part_9', 'part_10', 'part_11')){
  tem_pca <- prcomp_irlba(tem %>% select(!!!rlang::syms(j)) %>% replace(is.na(.), -1), .scale = TRUE)
  tem[[paste(j, '_V_1', sep = '_')]] <- tem_pca$x[,1]
  tem[[paste(j, '_V_2', sep = '_')]] <- tem_pca$x[,2]
  tem[[paste(j, '_V_3', sep = '_')]] <- tem_pca$x[,3]
  tem <- tem %>% select(-(!!!rlang::syms(j)))
}

x <- as.data.frame(tem)

missing_rate <- colSums(is.na(x))/nrow(x) 

del_vars <- missing_rate[missing_rate > 0.9] 
x <- x %>% select(-c("dist2", "D7", "id_07", "id_08", "id_18", "id_21", "id_22", "id_23", "id_24", "id_25", "id_26", "id_27"))


nums <- unlist(lapply(x, is.numeric))
x <- x[ , nums]
sample.df.cor <- cor(x)
sample.df.cor[is.na(sample.df.cor)] <- 0
sample.df.highly.cor <- findCorrelation(sample.df.cor, cutoff=.7)
x <- x[,-sample.df.highly.cor]


x <- data.matrix(x[1:133])

n_pca <- 3
tem_pca <- prcomp_irlba(x, n = n_pca, scale. = TRUE)
tem_pca <- as.data.frame(tem_pca$x)
tem <- data.frame(tem, tem_pca)

rm(tem_pca, x, sample.df.cor); gc(); gc()

# %% [code]
y_train <- y

train <- tem[1:nrow(train),]
test <- tem[-c(1:nrow(train)),]

h2o.no_progress()
h2o.init(nthreads = 4, max_mem_size = "10G")

tr_h2o <- as.h2o(train)
te_h2o <- as.h2o(test)

n_ae <- 4

m_ae <- h2o.deeplearning(training_frame = tr_h2o,
                         x = 1:ncol(tr_h2o),
                         autoencoder = T,
                         activation="Rectifier",
                         reproducible = TRUE,
                         seed = 0,
                         sparse = T,
                         standardize = TRUE,
                         hidden = c(32, n_ae, 32),
                         max_w2 = 5,
                         epochs = 25)

tr_ae <- h2o.deepfeatures(m_ae, tr_h2o, layer = 2) %>% as_tibble
te_ae <- h2o.deepfeatures(m_ae, te_h2o, layer = 2) %>% as_tibble

rm(tr_h2o, te_h2o, m_ae); invisible(gc())
h2o.shutdown(prompt = FALSE)

train <- data.frame(train, tr_ae)
test <- data.frame(test, te_ae)


rm(tem, tr_ae, te_ae) ; invisible(gc())

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
                  num_leaves = 240,
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

write.csv(sub,"sub_0.9357_80.csv",row.names = F)
write.csv(imp,"imp.csv",row.names = F)
