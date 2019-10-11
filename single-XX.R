library(readr)
library(tidyverse)
library(data.table)
library(MLmetrics)
library(lightgbm)
library(lubridate)
library(plyr)
library(moments)
library(dplyr)
library(rattle)
library(rpart)
options(warn=-1)
options(scipen = 99)

train <- read_csv("train.csv")
test <- read_csv("test.csv")


# %% [code]
y <- train$isFraud 
train$isFraud <- NULL


rm(train_iden,train_trans,test_iden,test_trans) ; invisible(gc())

#EDA, remove coorelated features (I think this came from PCA in a python module, forgot which one)

drop_col <- c('V300','V309','V111','V124','V106','V125','V315','V134','V102','V123','V316','V113',
              'V136','V305','V110','V299','V289','V286','V318','V304','V116','V284','V293',
              'V137','V295','V301','V104','V311','V115','V109','V119','V321','V114','V133','V122','V319',
              'V105','V112','V118','V117','V121','V108','V135','V320','V303','V297','V120',
              'V1','V14','V41','V65','V88', 'V89', 'V107', 'V68', 'V28', 'V27', 'V29', 'V241','V269',
              'V240', 'V325', 'V138', 'V154', 'V153', 'V330', 'V142', 'V195', 'V302', 'V328', 'V327', 
              'V198', 'V196', 'V155')

# using single hold-out validation (20%)
tr_idx <- which(train$TransactionDT < quantile(train$TransactionDT, 0.8))

# %% [code]
train[,drop_col] <- NULL
test[,drop_col] <- NULL

train$n_na <- rowSums(is.na(train))
test$n_na <- rowSums(is.na(test))

train$n_na_D <- rowSums(is.na(train[, 31:45]))
test$n_na_D <- rowSums(is.na(test[, 31:45]))


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

fe_part2 <- tem[, c("acc","card1","card2","card3", "card4", "TransactionAmt", "id_02", "D15","addr1_addr2","decimal","C13","V258","C1","C14",
                    "addr1__card1", "addr1__card2", "card1__P_emaildomain", "card5__P_emaildomain")]

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~card1, summarise, mean_card1_Trans= mean(TransactionAmt), sd_card1_Trans = sd(TransactionAmt)))
fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~addr1_addr2, summarise, mean_geo_Trans= mean(TransactionAmt), sd_geo_Trans = sd(TransactionAmt)))
fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~addr1_addr2, summarise, min_geo_Trans= min(TransactionAmt), max_geo_Trans = max(TransactionAmt)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~addr1__card1, summarise, mean_addr1__card1_Trans= mean(TransactionAmt), sd_addr1__card1_Trans = sd(TransactionAmt)))
fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~addr1__card2, summarise, mean_addr1__card2_Trans= mean(TransactionAmt), sd_addr1__card2_Trans = sd(TransactionAmt)))
fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~card1__P_emaildomain, summarise, mean_card1__P_emaildomain_Trans= mean(TransactionAmt), sd_card1__P_emaildomain_Trans = sd(TransactionAmt)))
fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~card5__P_emaildomain, summarise, mean_card5__P_emaildomain_Trans= mean(TransactionAmt), sd_card5__P_emaildomain_Trans = sd(TransactionAmt)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~addr1__card1, summarise, min_addr1__card1_Trans= min(TransactionAmt), max_addr1__card1_Trans = max(TransactionAmt)))
fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~addr1__card2, summarise, min_addr1__card2_Trans= min(TransactionAmt), max_addr1__card2_Trans = max(TransactionAmt)))
fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~card1__P_emaildomain, summarise, min_card1__P_emaildomain_Trans= min(TransactionAmt), max_card1__P_emaildomain_Trans = max(TransactionAmt)))
fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~card5__P_emaildomain, summarise, min_card5__P_emaildomain_Trans= min(TransactionAmt), max_card5__P_emaildomain_Trans = max(TransactionAmt)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~card1, summarise, min_card1_Trans= min(TransactionAmt), max_card1_Trans = max(TransactionAmt)))
fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~card2, summarise, min_card2_Trans= min(TransactionAmt), max_card2_Trans = max(TransactionAmt)))
fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~card4, summarise, mean_card4_Trans = mean(TransactionAmt), sd_card4_Trans = sd(TransactionAmt)))

fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~acc, summarise, mean_acc_Trans = mean(TransactionAmt), sd_acc_Trans = sd(TransactionAmt)))
fe_part2 <- fe_part2 %>% left_join(plyr::ddply(fe_part2, ~acc, summarise, min_acc_Trans = min(TransactionAmt), max_acc_Trans = max(TransactionAmt)))
fe_part2 = fe_part2 %>% left_join(plyr::ddply(fe_part2, ~acc, summarise, mean_acc_dec = mean(decimal,na.rm = T), sd_acc_dec = sd(decimal),acc_dec_q = quantile(decimal,.8,na.rm = T)))
fe_part2 = fe_part2 %>% left_join(plyr::ddply(fe_part2, ~acc, summarise, min_acc_dec = min(decimal,na.rm = T), max_acc_dec = max(decimal)))

fe_part2 = fe_part2 %>% left_join(plyr::ddply(fe_part2, ~addr1__card1, summarise, mean_addr1__card1_dec = mean(decimal,na.rm = T), sd_addr1__card1_dec = sd(decimal), addr1__card1_dec_q = quantile(decimal,.8,na.rm = T)))
fe_part2 = fe_part2 %>% left_join(plyr::ddply(fe_part2, ~addr1__card1, summarise, min_addr1__card1_dec = min(decimal,na.rm = T), max_addr1__card1_dec = max(decimal)))

fe_part2 = fe_part2 %>% left_join(plyr::ddply(fe_part2, ~addr1__card2, summarise, mean_addr1__card2_dec = mean(decimal,na.rm = T), sd_addr1__card2_dec = sd(decimal), addr1__card2_dec_q = quantile(decimal,.8,na.rm = T)))
fe_part2 = fe_part2 %>% left_join(plyr::ddply(fe_part2, ~addr1__card2, summarise, min_addr1__card2_dec = min(decimal,na.rm = T), max_addr1__card2_dec = max(decimal)))

fe_part2 = fe_part2 %>% left_join(plyr::ddply(fe_part2, ~card1__P_emaildomain, summarise, mean_card1__P_emaildomain_dec = mean(decimal,na.rm = T), sd_card1__P_emaildomain_dec = sd(decimal), card1__P_emaildomain_dec_q = quantile(decimal,.8,na.rm = T)))
fe_part2 = fe_part2 %>% left_join(plyr::ddply(fe_part2, ~card1__P_emaildomain, summarise, min_card1__P_emaildomain_dec = min(decimal,na.rm = T), max_card1__P_emaildomain_dec = max(decimal)))

fe_part2 = fe_part2 %>% left_join(plyr::ddply(fe_part2, ~card5__P_emaildomain, summarise, mean_card5__P_emaildomain_dec = mean(decimal,na.rm = T), sd_card5__P_emaildomain_dec = sd(decimal), card5__P_emaildomain_dec_q = quantile(decimal,.8,na.rm = T)))
fe_part2 = fe_part2 %>% left_join(plyr::ddply(fe_part2, ~card5__P_emaildomain, summarise, min_card5__P_emaildomain_dec = min(decimal,na.rm = T), max_card5__P_emaildomain_dec = max(decimal)))


# %% [code]
head(fe_part2)
head(unique(fe_part2$mean_card1_D15))
head(unique(fe_part2$mean_card1_id02))
fe_part2 <- fe_part2[, -c(1:18)]


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

#FE part4: count encoding of base features
char_features <- tem[,colnames(tem) %in% 
                       c("ProductCD","card1","card2","card3","card4","card5","card6","addr1","addr2","P_emaildomain",
                         "R_emaildomain","M1","M2","M3","M4","M5","M6","M7","M8","M9","DeviceType","DeviceInfo","id_12",
                         "id_13","id_14","id_15","id_16","id_17","id_18","id_19","id_20","id_21","id_22","id_23","id_24",
                         "id_25","id_26","id_27","id_28","id_29","id_30","id_31","id_32","id_33","id_34","id_35","id_36",
                         "id_37","id_38", "card1__dist1", "card1__P_emaildomain", "addr1__card1",
                         "card4__dist1", "addr1__card4", "P_emaildomain__C2", "addr1__card2",
                         "card1__card5", "card5__P_emaildomain", "acc")]

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
             "card1__card5", "card5__P_emaildomain")

tem <- tem %>% mutate_at(char_f, function(x){
  y <- as.factor(x)
  y <- as.numeric(y)
  y <- y - 1
  return(y)
})


tem <- data.frame(tem,fe_part1, fe_part2, fe_part3, fe_part4)

rm(fe_part1, fe_part2, fe_part3, fe_part4); invisible(gc());

tem <- tem %>% group_by(addr1, card1) %>% mutate(mean_TransactionAmt_addr1_card1 = mean(TransactionAmt, na.rm = TRUE),
                                                 sd_TransactioAmt_addr1_card1 = sd(TransactionAmt, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(TransactionAmt_to_mean_addr1_card1 = TransactionAmt/mean_TransactionAmt_addr1_card1,
         TransactionAmt_to_sd_addr1_card1 = TransactionAmt/sd_TransactioAmt_addr1_card1,
         TransactionAmt_subs_addr1_card1 = TransactionAmt - mean_TransactionAmt_addr1_card1)

tem <- tem %>% group_by(addr1, card2) %>% mutate(mean_TransactionAmt_addr1_card2 = mean(TransactionAmt, na.rm = TRUE),
                                                 sd_TransactioAmt_addr1_card2 = sd(TransactionAmt, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(TransactionAmt_to_mean_addr1_card2 = TransactionAmt/mean_TransactionAmt_addr1_card2,
         TransactionAmt_to_sd_addr1_card2 = TransactionAmt/sd_TransactioAmt_addr1_card2,
         TransactionAmt_subs_addr1_card2 = TransactionAmt - mean_TransactionAmt_addr1_card2)

tem <- tem %>% group_by(card5, card1) %>% mutate(mean_TransactionAmt_card5_card1 = mean(TransactionAmt, na.rm = TRUE),
                                                 sd_TransactioAmt_card5_card1 = sd(TransactionAmt, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(TransactionAmt_to_mean_card5_card1 = TransactionAmt/mean_TransactionAmt_card5_card1,
         TransactionAmt_to_sd_card5_card1 = TransactionAmt/sd_TransactioAmt_card5_card1,
         TransactionAmt_subs_card5_card1 = TransactionAmt - mean_TransactionAmt_card5_card1)

tem <- tem %>% group_by(card1, P_emaildomain) %>% mutate(mean_TransactionAmt_card1_P_emaildomain = mean(TransactionAmt, na.rm = TRUE),
                                                         sd_TransactionAmt_card1_P_emaildomain = sd(TransactionAmt, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(TransactionAmt_to_mean_card1_P_emaildomain = TransactionAmt/mean_TransactionAmt_card1_P_emaildomain,
         TransactionAmt_to_sd_card1_P_emaildomain = TransactionAmt/sd_TransactionAmt_card1_P_emaildomain,
         TransactionAmt_subs_card1_P_emaildomain = TransactionAmt - mean_TransactionAmt_card1_P_emaildomain)

tem <- tem %>% group_by(P_emaildomain) %>% mutate(mean_TransactionAmt_P_emaildomain = mean(TransactionAmt, na.rm = TRUE),
                                                  sd_TransactionAmt_P_emaildomain = sd(TransactionAmt, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(TransactionAmt_to_mean_P_emaildomain = TransactionAmt/mean_TransactionAmt_P_emaildomain,
         TransactionAmt_to_sd_P_emaildomain = TransactionAmt/sd_TransactionAmt_P_emaildomain,
         TransactionAmt_subs_P_emaildomain = TransactionAmt - mean_TransactionAmt_P_emaildomain)

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

tem$new3 = tem$TransactionAmt * tem$C1
tem$new5 = tem$TransactionAmt * tem$C13
tem$new7 = tem$TransactionAmt * tem$C14

tem$decimal_diff = tem$decimal - tem$mean_acc_dec
tem$new13 = tem$decimal_diff * tem$mean_acc_Trans

# %% [code]
train <- tem[1:nrow(train),]
test <- tem[-c(1:nrow(train)),]

#missing_rate <- colSums(is.na(train))/nrow(train) 
#missing_rate %>% sort(decreasing=T)

#del_vars <- missing_rate[missing_rate > 0.9] 
#train <- train %>% select(-del_vars)
#test <- test %>% select(-del_vars)

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

PCA_D = tem[,which(names(tem) %in% paste0("D", 1:15))] %>%
  scale() %>%
  tbl_df() %>%
  mutate_all(funs(ifelse(is.na(.),0,.))) %>%
  data.matrix() %>%
  fast.prcomp()

PCA_C = tem[,which(names(tem) %in% paste0("C", 1:14))] %>%
  scale() %>%
  tbl_df() %>%
  mutate_all(funs(ifelse(is.na(.),0,.))) %>%
  data.matrix() %>%
  fast.prcomp()

PCA_id = data.frame(tem[,which(names(tem) %in% paste0("id_", 10:38))], tem[,which(names(tem) %in% paste0("id_0", 1:9))]) %>%
  scale() %>%
  tbl_df() %>%
  mutate_all(funs(ifelse(is.na(.),0,.))) %>%
  data.matrix() %>%
  fast.prcomp()

PCA_card = tem[,which(names(tem) %in% paste0("card", 1:6))] %>%
  scale() %>%
  tbl_df() %>%
  mutate_all(funs(ifelse(is.na(.),0,.))) %>%
  data.matrix() %>%
  fast.prcomp()

PCA_M = tem[,which(names(tem) %in% paste0("M", 1:9))] %>%
  scale() %>%
  tbl_df() %>%
  mutate_all(funs(ifelse(is.na(.),0,.))) %>%
  data.matrix() %>%
  fast.prcomp()

tem <- data.frame(tem, PCA_C$x, PCA_card$x, PCA_D$x, PCA_id$x, PCA_M$x)
rm(PCA_C, PCA_card, PCA_D, PCA_id, PCA_M); gc(); gc()

y_train <- y
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
                  bagging_fraction = 0.7,
                  min_data_in_leaf = 100,
                  bagging_seed = 11,
                  max_bin = 255,
                  verbosity = -1)

valids <- list(valid = dval)
lgb <- lgb.train(params = lgb_param,  data = d0, nrounds = 15000, 
                 eval_freq = 200, valids = valids, early_stopping_rounds = 400, verbose = 1, seed = 123)


oof_pred <- predict(lgb, data.matrix(train[-tr_idx,]))
cat("best iter :" , lgb$best_iter, "best score :", AUC(oof_pred, y[-tr_idx]) ,"\n" )
iter <- lgb$best_iter

rm(lgb,d0,dval) ; invisible(gc())

# full data
d0 <- lgb.Dataset(data.matrix(train), label = y )
lgb <- lgb.train(params = lgb_param, data = d0, nrounds = iter * 1.05, verbose = -1, seed = 123)
pred <- predict(lgb, data.matrix(test))

imp <- lgb.importance(lgb)
sub <- data.frame(read_csv("sample_submission.csv"))
sub[,2] <- pred

write.csv(sub,"submission_93795.csv",row.names = F)
write.csv(imp,"features_importance.csv",row.names = F)