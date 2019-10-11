library(readr)
library(tidyverse)
library(data.table)
library(irlba)
library(gmodels)
library(h2o)
library(uwot)
library(caret)


set.seed(71)

tem <- fread("tem.csv")

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


x$SV258 <- x$V258

# Dim reduction of v-variables (groups of V variables - https://www.kaggle.com/abednadir/best-r-score)
part_1 <- names(x)[which(names(x) %in% paste0("V",1:11))]
part_2 <- names(x)[which(names(x) %in% paste0("V",12:34))]
part_3 <- names(x)[which(names(x) %in% paste0("V",35:52))]
part_4 <- names(x)[which(names(x) %in% paste0("V",53:74))]
part_5 <- names(x)[which(names(x) %in% paste0("V",75:94))]
part_6 <- names(x)[which(names(x) %in% paste0("V",95:137))]
part_7 <- names(x)[which(names(x) %in% paste0("V",138:166))]
part_8 <- names(x)[which(names(x) %in% paste0("V",167:216))]
part_9 <- names(x)[which(names(x) %in% paste0("V",217:278))]
part_10 <- names(x)[which(names(x) %in% paste0("V",279:321))]
part_11 <- names(x)[which(names(x) %in% paste0("V",322:339))]

for(j in c('part_1', 'part_2', 'part_3', 'part_4', 'part_5', 'part_6', 'part_7', 'part_8', 'part_9', 'part_10', 'part_11')){
  x_pca <- prcomp_irlba(x %>% select(!!!rlang::syms(j)) %>% replace(is.na(.), -1), .scale = TRUE)
  x[[paste(j, '_V_1', sep = '_')]] <- x_pca$x[,1]
  x[[paste(j, '_V_2', sep = '_')]] <- x_pca$x[,2]
  x[[paste(j, '_V_3', sep = '_')]] <- x_pca$x[,3]
  x <- x %>% select(-(!!!rlang::syms(j)))
}

tem_pca <- x %>%
  scale() %>%
  data.matrix()

tem_pca[is.na(tem_pca)] <- 0

tem_pca <- fast.prcomp(tem_pca)

tem_h2o <- x %>%
  scale() %>%
  data.matrix()

tem_h2o[is.na(tem_h2o)] <- 0

h2o.no_progress()
h2o.init(nthreads = 4, max_mem_size = "10G")

tem_h2o <- as.h2o(tem_h2o)

n_ae <- 4

m_ae <- h2o.deeplearning(training_frame = tem_h2o,
                         x = 1:ncol(tem_h2o),
                         autoencoder = T,
                         activation="Rectifier",
                         reproducible = TRUE,
                         seed = 71,
                         sparse = T,
                         standardize = TRUE,
                         hidden = c(32, n_ae, 32),
                         max_w2 = 5,
                         epochs = 25)

tem_ae <- h2o.deepfeatures(m_ae, tem_h2o, layer = 2) %>% as_tibble

rm(tem_h2o, m_ae)
h2o.shutdown(prompt = FALSE); invisible(gc())


tem_pca <- x %>%
  scale() %>%
  data.matrix()

tem_pca[is.na(tem_pca)] <- 0

tem_pca <- fast.prcomp(tem_pca)

tem_umap <- umap(tem_pca$x, n_neighbors = 15,
                 min_dist = 0.001, verbose = TRUE, n_threads = 6)

tem_umap <- as.data.frame(tem_umap)


tem_unsupervised <- data.frame(tem_pca$x, tem_ae, tem_umap)

fwrite(tem_unsupervised, "tem_unsupervised.csv")
