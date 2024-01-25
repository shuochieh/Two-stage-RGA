### Last update: 2023-01-04

library(glmnet)
library(rrpack)

dev.off()
rm(list=ls())
diff_unit_root = TRUE      # whether or not to first-difference volume series
plot_or_not = F         # whether to plot EDA plots
catwise_scale = TRUE       # whether to scale each category to have roughly same magnitude
include_volume = TRUE      # whether to include volume as response variable (should set to TRUE)

num_topics = 200
dta = read.csv(paste0("./final_data_lsi_", num_topics, ".csv"), header = FALSE)

source("./tsrga.r")
source("./iRRR.R")
source("./predo_outliers.r")

### category extraction
categories = c("volatility", "volume", "beta", "return")
res = c()
if ("volatility" %in% categories) {
  res = c(res, 1:12)
} 
if ("volume" %in% categories) {
  res = c(res, 13:24)
}
if ("beta" %in% categories) {
  res = c(res, 25:36)
}
if ("return" %in% categories) {
  res = c(res, 37:48)
}

train_Y = train_Y[,res]
train_Ylag = train_Ylag[,res]
test_Y = test_Y[,res]
test_Ylag = test_Ylag[,res]
sds = sds[res]

response_d = length(res)
cat_d = length(categories)
###

source("./benchmarks_do.r")

save_train_Y = train_Y
save_test_Y = test_Y
save_sds = sds

### iRRR

for (zz in 1:4) {
  cat_id = zz
  train_Y = save_train_Y[c(1:(256 * 6)), c(1:12) + (cat_id - 1) * 12]
  val_Y = save_train_Y[-c(1:(256 * 6)), c(1:12) + (cat_id - 1) * 12]
  test_Y = save_test_Y[,c(1:12) + (cat_id - 1) * 12]
  sds = save_sds[c(1:12) + (cat_id - 1) * 12]
  
  train_features = cbind(train_Ylag[c(1:(256 * 6)),], 
                         train_X[c(1:(256 * 6)),])
  val_features = cbind(train_Ylag[-c(1:(256 * 6)),],
                       train_X[-c(1:(256 * 6)),])
  test_features = cbind(test_Ylag, test_X)
  
  ptt = vector(mode = "list", length = 5)
  ptt[[1]] = c(1:12)
  ptt[[2]] = c(13:24)
  ptt[[3]] = c(25:36)
  ptt[[4]] = c(37:48)
  ptt[[5]] = c(49:(48 + ncol(train_X)))
  
  y_means = colMeans(train_Y)
  x_means = colMeans(train_features)
  
  train_Y = t(t(train_Y) - y_means) 
  val_Y = t(t(val_Y) - y_means)
  test_Y = t(t(test_Y) - y_means)

  train_features = t(t(train_features) - x_means)
  val_features = t(t(val_features) - x_means)
  test_features = t(t(test_features) - x_means)

  x_2norms = rep(0, 5)
  x_2norms[1] = norm(train_features[,1:12], type = "2")
  x_2norms[2] = norm(train_features[,13:24], type = "2")
  x_2norms[3] = norm(train_features[,25:36], type = "2")
  x_2norms[4] = norm(train_features[,37:48], type = "2")
  x_2norms[5] = norm(train_features[,49:(48 + num_topics)], type = "2")
  
  train_features[,1:12] = train_features[,1:12] / x_2norms[1]
  train_features[,13:24] = train_features[,13:24] / x_2norms[2]
  train_features[,25:36] = train_features[,25:36] / x_2norms[3]
  train_features[,37:48] = train_features[,37:48] / x_2norms[4]
  train_features[,49:(48 + num_topics)] = train_features[,49:(48 + num_topics)] / x_2norms[5]
  
  val_features[,1:12] = val_features[,1:12] / x_2norms[1]
  val_features[,13:24] = val_features[,13:24] / x_2norms[2]
  val_features[,25:36] = val_features[,25:36] / x_2norms[3]
  val_features[,37:48] = val_features[,37:48] / x_2norms[4]
  val_features[,49:(48 + num_topics)] = val_features[,49:(48 + num_topics)] / x_2norms[5]

  test_features[,1:12] = test_features[,1:12] / x_2norms[1]
  test_features[,13:24] = test_features[,13:24] / x_2norms[2]
  test_features[,25:36] = test_features[,25:36] / x_2norms[3]
  test_features[,37:48] = test_features[,37:48] / x_2norms[4]
  test_features[,49:(48 + num_topics)] = test_features[,49:(48 + num_topics)] / x_2norms[5]

  cat("iRRR calculating category", cat_id, "\n")
  model = iRRR_validation(train_Y, train_features, ptt, 
                          lambdas = exp(seq(from = -7, to = -6, length.out = 10)),
                          y_val = val_Y, X_val = val_features, 
                          expedite_factor = 1.005, tol = 1e-3,
                          verbose = F)
  
  pred = test_features %*% model$B
  temp = pred - test_Y
  temp = t(t(temp) * sds)

  cat("Category", cat_id, "\n")
  cat("  iRRR RMSE:", round(sqrt(mean((temp)^2)), 3), "\n")
}

### TSRGA 

for (zz in 1:4) {
  cat_d = 1
  cat_id = zz
  
  train_Y = save_train_Y[c(1:(256 * 6)), c(1:12) + (cat_id - 1) * 12]
  val_Y = save_train_Y[-c(1:(256 * 6)), c(1:12) + (cat_id - 1) * 12]
  test_Y = save_test_Y[,c(1:12) + (cat_id - 1) * 12]
  sds = save_sds[c(1:12) + (cat_id - 1) * 12]
  
  feature_train = vector(mode = "list", length = 5)
  feature_val = vector(mode = "list", length = 5)
  feature_test = vector(mode = "list", length = 5)
  
  for (i in 1:4) {
    feature_train[[i]] = train_Ylag[c(1:(256 * 6)), c(1:12) + (i - 1) * 12]
    feature_val[[i]] = train_Ylag[-c(1:(256 * 6)),c(1:12) + (i - 1) * 12]
    feature_test[[i]] = test_Ylag[,c(1:12) + (i - 1) * 12]
  }
  feature_train[[5]] = train_X[c(1:(256 * 6)),]
  feature_val[[5]] = train_X[-c(1:(256 * 6)),]
  feature_test[[5]] = test_X

  t_n_grid = round(seq(0.1, 1, length.out = 10) / log(nrow(save_train_Y)), 3)
  dims = c(rep(12, 5), ncol(train_X))
  
  t_n = tsrga_pilot(train_Y, feature_train, val_Y, feature_val,
                    dims, 10^6, 12, 500, t_n_grid)
  
  train_Y = save_train_Y[, c(1:12) + (cat_id - 1) * 12]
  feature_train = vector(mode = "list", length = 5)
  for (i in 1:4) {
    feature_train[[i]] = train_Ylag[, c(1:12) + (i - 1) * 12]
  }
  feature_train[[5]] = train_X

  model = tsrga(train_Y, feature_train, dims, 10^6, t_n, 12, 500)
  pred = tsrga_fit(model, feature_test, model$x_means, model$y_means, 
                   model$x_2norms)
  
  temp = pred - test_Y
  temp = t(t(temp) * sds)
  
  cat("Category", cat_id, "; TSRGA RMSE:", round(sqrt(mean(temp^2)), 3), "\n")
}


### old code

for (zz in 1:4) {
  cat_d = 1
  cat_id = zz
  
  train_Y = save_train_Y[,c(1:12) + (cat_id - 1) * 12]
  test_Y = save_test_Y[,c(1:12) + (cat_id - 1) * 12]
  sds = save_sds[c(1:12) + (cat_id - 1) * 12]
  
  feature_train = vector(mode = "list", length = 5)
  feature_test = vector(mode = "list", length = 5)
  
  for (i in 1:4) {
    feature_train[[i]] = train_Ylag[, c(1:12) + (i - 1) * 12]
    feature_test[[i]] = test_Ylag[,c(1:12) + (i - 1) * 12]
  }
  feature_train[[5]] = train_X
  feature_test[[5]] = test_X
  
  K = 12
  
  MSPEs = rep(0, K)
  tsrga_cat_MSPEs = matrix(0, nrow = cat_d, ncol = K)
  tsrga_cat_MAEs = matrix(0, nrow = cat_d, ncol = K)
  
  par(mfrow = c(4, 2))
  for (i in 1:K) {
    dims = c(12, rep(12, length(feature_train) - 1), ncol(train_X))
    model = tsrga(train_Y, feature_train, L = 10^6, t_n = 0,
                  dims = dims,
                  Kn1 = i,
                  Kn2 = 500, verbose = F)
    pred = tsrga_fit(model, feature_test, model$x_means, model$y_means, 
                     model$x_2norms)
    cat("TSRGA step", i, "MSPE:", mean((test_Y - pred)^2), "\n")
    MSPEs[i] = mean((test_Y - pred)^2)
    
    temp = test_Y - pred
    temp = t(t(temp) * sds)
    
    for (j in 1:cat_d) {
      tsrga_cat_MSPEs[j,i] = mean((temp^2)[,((j - 1) * 12 + 1):(j * 12)])
      tsrga_cat_MAEs[j,i] = median(abs(temp)[,((j - 1) * 12 + 1):(j * 12)])
    }
    
    # check convergence
    plot(model$loss2, xlab = "step", ylab = "",
         main = paste("Second stage for", i))
  }
  
  par(mfrow = c(3 ,1))
  plot(model$path1, xlab = "step", ylab = "index")
  plot(model$loss1, xlab = "step", ylab = "loss")
  jits = model$loss1[-1] / model$loss1[-K]
  thres1 = 1 - 1 / (10 * log(nrow(train_Y)))
  thres2 = 1 - 1 / log(nrow(train_Y))
  plot(x = c(2:K), y = jits, xlab = "step", ylab = "jits", 
       ylim = c(min(jits, thres1, thres2), max(jits, thres1, thres2)))
  points(x = 1 + which(jits > thres2), y = jits[which(jits > thres2)], pch = 19,
         col = "green")
  points(x = 1 + which(jits > thres1), y = jits[which(jits > thres1)], pch = 19,
         col = "red")
  abline(h = thres1, col = "red", lty = 2)
  abline(h = thres2, col = "green", lty = 2)
  
  o_step = min(which(jits > thres1)) + 1
  cat("Optimal number of step:", o_step, "\n")
  
  cat(categories[zz], "RMSE:", round(sqrt(tsrga_cat_MSPEs[1,o_step]), 4), "\n")
  cat(categories[zz], "MAE:", round(tsrga_cat_MAEs[1,o_step], 4), "\n")
  
  par(mfrow = c(1, 1))
  plot(sqrt(tsrga_cat_MSPEs[1,]), ylab = "", xlab = "", type = "b",
       main = categories[cat_id])
}

