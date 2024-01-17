### Last update: 2023-01-05
### Benchmark training

### Lasso Benchmark

Y_lasso = matrix(NA, nrow = nrow(test_Y), ncol = ncol(test_Y))
nzs = rep(NA, response_d)

start_time = proc.time()
for (d in 1:response_d) {
  cat("Lasso: fitting the", d, "th model...")
  model = cv.glmnet(x = cbind(train_Ylag, train_X), y = as.vector(train_Y[,d]),
                    nfolds = 5, nlambda = 25, maxit = 5000)
  Y_lasso[,d] = predict(model, newx = cbind(test_Ylag, test_X), 
                        s = "lambda.min")
  nzs[d] = length(which(which(coefficients(model, s = "lambda.min")[-1] != 0) > 48))
  cat("done. Time elapsed:", round((proc.time() - start_time)[3] / 60, 2), "minutes", 
      "\n")
}

cat("Lasso MSPE:", mean((Y_lasso - test_Y)^2), "\n")

if (catwise_scale) {
  Lasso_error_temp = Y_lasso - test_Y
  Lasso_error_temp = t(t(Lasso_error_temp) * sds)
  idx = 1:12
  for (j in 1:cat_d) {
    cat(paste0("  Lasso RMSE (category ", j, "):"), 
        round(sqrt(mean((Lasso_error_temp[,idx])^2)), 4), "\n")
    cat(paste0("  Lasso MAE (category ", j, "):"), 
        round(median(abs(Lasso_error_temp[,idx])), 4), "\n")
    cat(paste0("  Avg # of nonzero topics (Cat ", j, "):"),
        round(mean(nzs[idx]), 2), "\n")
    idx = idx + 12
  }
} else {
  Lasso_error_temp = Y_lasso - test_Y
  idx = 1:12
  for (j in 1:cat_d) {
    cat(paste0("  Lasso RMSE (category ", j, "):"), 
        round(sqrt(mean((Lasso_error_temp[,idx])^2)), 4), "\n")
    cat(paste0("  Lasso MAE (category ", j, "):"), 
        round(median(abs(Lasso_error_temp[,idx])), 4), "\n")
    cat(paste0("  Avg # of nonzero topics (Cat ", j, "):"),
        round(mean(nzs[idx]), 2), "\n")
    idx = idx + 12
  }
}

### VAR Benchmark

Y_VAR = matrix(NA, nrow = nrow(test_Y), ncol = ncol(test_Y))

for (d in 1:response_d) {
  #cat("VAR: fitting the", d, "th model...")
  model = lm(as.vector(train_Y[,d])~train_Ylag)
  Y_VAR[,d] = cbind(1, test_Ylag) %*% model$coefficients
  #cat("done.", "\n")
}

cat("VAR MSPE:", mean((Y_VAR - test_Y)^2), "\n")

if (catwise_scale) {
  VAR_error_temp = Y_VAR - test_Y
  VAR_error_temp = t(t(VAR_error_temp) * sds)
  
  idx = 1:12
  for (j in 1:cat_d) {
    cat(paste0("  VAR RMSE (category ", j, "):"), 
        round(sqrt(mean((VAR_error_temp[,idx])^2)), 4), "\n")
    cat(paste0("  VAR MAE (category ", j, "):"), 
        round(median(abs(VAR_error_temp[,idx])), 4), "\n")
    idx = idx + 12
  }
} else {
  VAR_error_temp = Y_VAR - test_Y

  idx = 1:12
  for (j in 1:cat_d) {
    cat(paste0("  VAR RMSE (category ", j, "):"), 
        round(sqrt(mean((VAR_error_temp[,idx])^2)), 4), "\n")
    cat(paste0("  VAR MAE (category ", j, "):"), 
        round(median(abs(VAR_error_temp[,idx])), 4), "\n")
    idx = idx + 12
  }
}

### category-wise VAR Benchmark

Y_gVAR = matrix(NA, nrow = nrow(test_Y), ncol = ncol(test_Y))

for (j in 1:cat_d) {
  #cat("gVAR: fitting the", j, "th category...")
  for (d in 1:12) {
    model = lm(as.vector(train_Y[,d + (j - 1) * 12])~train_Ylag[,((j - 1) * 12 + 1):(j * 12)])
    Y_gVAR[,d + (j - 1) * 12] = cbind(1, test_Ylag[,((j - 1) * 12 + 1):(j * 12)]) %*% model$coefficients
  }
  #cat("done.", "\n")
}

cat("gVAR MSPE:", mean((Y_gVAR - test_Y)^2), "\n")

if (catwise_scale) {
  gVAR_error_temp = Y_gVAR - test_Y
  gVAR_error_temp = t(t(gVAR_error_temp) * sds)

  idx = 1:12
  for (j in 1:cat_d) {
    cat(paste0("  gVAR RMSE (category ", j, "):"), 
        round(sqrt(mean((gVAR_error_temp[,idx])^2)), 4), "\n")
    cat(paste0("  gVAR MAE (category ", j, "):"), 
        round(median(abs(gVAR_error_temp[,idx])), 4), "\n")
    idx = idx + 12
  }
} else {
  cat("  gVAR RMSE (category 1):", round(sqrt(mean((gVAR_error_temp[,1:12])^2)), 4), "\n")
  cat("  gVAR MAE (category 1):", round(median(abs(gVAR_error_temp[,1:12])), 4), "\n")
  cat("  gVAR RMSE (category 2):", round(sqrt(mean((gVAR_error_temp[,13:24])^2)), 4), "\n")
  cat("  gVAR MAE (category 2):", round(median(abs(gVAR_error_temp[,13:24])), 4), "\n")
  cat("  gVAR RMSE (category 3):", round(sqrt(mean((gVAR_error_temp[,25:36])^2)), 4), "\n")
  cat("  gVAR MAE (category 3):", round(median(abs(gVAR_error_temp[,25:36])), 4), "\n")
  cat("  gVAR RMSE (category 4):", round(sqrt(mean((gVAR_error_temp[,37:48])^2)), 4), "\n")
  cat("  gVAR MAE (category 4):", round(median(abs(gVAR_error_temp[,37:48])),4), "\n")
}

# for future use
write.csv(Y_lasso, "Y_lasso.csv", row.names = FALSE)
write.csv(Y_VAR, "Y_VAR.csv", row.names = FALSE)
write.csv(Y_gVAR, "Y_gVAR.csv", row.names = FALSE)

### Additional benchmark: low-rank category-wise regression

Y_LR_VAR = matrix(NA, nrow = nrow(test_Y), ncol = ncol(test_Y))
cat_d = 4
ranks_LR_VAR = rep(0, cat_d)

for (j in 1:cat_d) {
  means = colMeans(train_Y)[c(1:12) + (j - 1) * 12]
  model = rrr(t(t(train_Y[,c(1:12) + (j - 1) * 12]) - means), 
              t(t(train_Ylag[,((j - 1) * 12 + 1):(j * 12)]) - means), penaltySVD = "rank",
              ic.type = "GIC")
  Y_LR_VAR[,c(1:12) + (j - 1) * 12] = t(t(test_Ylag[,((j - 1) * 12 + 1):(j * 12)]) - means) %*% model$coef
  Y_LR_VAR[,c(1:12) + (j - 1) * 12] = t(t(Y_LR_VAR[,c(1:12) + (j - 1) * 12]) + means)
  ranks_LR_VAR[j] = model$rank
}

if (catwise_scale) {
  LR_VAR_error_temp = Y_LR_VAR - test_Y
  LR_VAR_error_temp = t(t(LR_VAR_error_temp) * sds)
  
  idx = 1:12
  for (j in 1:cat_d) {
    cat(paste0("(Selected rank: ", ranks_LR_VAR[j], ")"), "\n")
    cat(paste0("  Low-rank VAR RMSE (category ", j, "):"), 
        round(sqrt(mean((LR_VAR_error_temp[,idx])^2)), 4), "\n")
    cat(paste0("  Low-rank VAR MAE (category ", j, "):"), 
        round(median(abs(LR_VAR_error_temp[,idx])), 4), "\n")
    idx = idx + 12
  }
} else {
  cat("  Low-rank VAR RMSE (category 1):", round(sqrt(mean((LR_VAR_error_temp[,1:12])^2)), 4), "\n")
  cat("  Low-rank VAR MAE (category 1):", round(median(abs(LR_VAR_error_temp[,1:12])), 4), "\n")
  cat("  Low-rank VAR RMSE (category 2):", round(sqrt(mean((LR_VAR_error_temp[,13:24])^2)), 4), "\n")
  cat("  Low-rank VAR MAE (category 2):", round(median(abs(LR_VAR_error_temp[,13:24])), 4), "\n")
  cat("  Low-rank VAR RMSE (category 3):", round(sqrt(mean((LR_VAR_error_temp[,25:36])^2)), 4), "\n")
  cat("  Low-rank VAR MAE (category 3):", round(median(abs(LR_VAR_error_temp[,25:36])), 4), "\n")
  cat("  Low-rank VAR RMSE (category 4):", round(sqrt(mean((LR_VAR_error_temp[,37:48])^2)), 4), "\n")
  cat("  Low-rank VAR MAE (category 4):", round(median(abs(LR_VAR_error_temp[,37:48])),4), "\n")
}

# for future use
write.csv(Y_LR_VAR, "Y_LR_VAR.csv", row.names = FALSE)
