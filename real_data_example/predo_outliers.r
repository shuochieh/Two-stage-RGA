### Last update: 2023-02-19

y = as.matrix(dta[,c(2:49)])
X = as.matrix(dta[,-c(1:49)])

### take lags and train-test split
# 2010~2017 as training set
n_companies = nrow(dta) / 10

if (include_volume) {
  if (diff_unit_root) {
    diff_y =  cbind(y[(n_companies + 1):(10 * n_companies), 1:12],
                    y[(n_companies + 1):(10 * n_companies), 13:24] - y[1:(9 * n_companies), 13:24],
                    y[(n_companies + 1):(10 * n_companies),25:48])
    train_Y = diff_y[(n_companies + 1):(n_companies * 8),]                      # 2012~2018
    test_Y = diff_y[(n_companies * 8 + 1):(n_companies * 9),]                   # 2019
  
    train_X = X[1:(n_companies * 7),]                                           # 2011~2017
    test_X = X[(n_companies * 7 + 1):(n_companies * 8),]                        # 2018
  
    train_Ylag = diff_y[1:(n_companies * 7),]                                   # 2011~2017
    test_Ylag = diff_y[(n_companies * 7 + 1):(n_companies * 8),]                # 2018
  } else {
    diff_y = y
  
    train_Y = diff_y[(n_companies + 1):(n_companies * 8),]                      # 2011~2017
    test_Y = diff_y[(n_companies * 8 + 1):(n_companies * 10),]                  # 2018~2019
  
    train_X = X[1:(n_companies * 7),]                                           # 2010~2016
    test_X = X[(n_companies * 7 + 1):(n_companies * 9),]                        # 2017~2018
  
    train_Ylag = diff_y[1:(n_companies * 7),]                                   # 2010~2016
    test_Ylag = diff_y[(n_companies * 7 + 1):(n_companies * 9),]                # 2017~2018
  }
} else {
  diff_y = y[,c(1:12, 25:48)]
  
  train_Y = diff_y[(n_companies + 1):(n_companies * 8),]                        # 2011~2017
  test_Y = diff_y[(n_companies * 8 + 1):(n_companies * 10),]                    # 2018~2019
  
  train_X = X[1:(n_companies * 7),]                                             # 2010~2016
  test_X = X[(n_companies * 7 + 1):(n_companies * 9),]                          # 2017~2018
  
  train_Ylag = diff_y[1:(n_companies * 7),]                                     # 2010~2016
  test_Ylag = diff_y[(n_companies * 7 + 1):(n_companies * 9),]                  # 2017~2018
}

### preliminary plots

if (plot_or_not) {
  par(mfrow = c(2, 2))
  hist(c(diff_y[,1:12]), breaks = 50, main = "volatility", xlab = "", 
       probability = TRUE)
  # abline(v = quantile(c(diff_y[,1:12]), 0.001), col = "red", lty = 2, lwd = 0.8)
  # abline(v = quantile(c(diff_y[,1:12]), 0.999), col = "red", lty = 2, lwd = 0.8)
  
  hist(c(diff_y[,13:24]), breaks = 50, main = "volume", xlab = "", 
       probability = TRUE)
  # abline(v = quantile(c(diff_y[,13:24]), 0.001), col = "red", lty = 2, lwd = 0.8)
  # abline(v = quantile(c(diff_y[,13:24]), 0.999), col = "red", lty = 2, lwd = 0.8)
  
  hist(c(diff_y[,25:36]), breaks = 100, main = "beta", xlab = "", 
       probability = TRUE)
  # abline(v = quantile(c(diff_y[,25:36]), 0.001), col = "red", lty = 2, lwd = 0.8)
  # abline(v = quantile(c(diff_y[,25:36]), 0.999), col = "red", lty = 2, lwd = 0.8)
  
  hist(c(diff_y[,37:48]), breaks = 100, main = "return", xlab = "", 
       probability = TRUE)
  # abline(v = quantile(c(diff_y[,37:48]), 0.001), col = "red", lty = 2, lwd = 0.8)
  # abline(v = quantile(c(diff_y[,37:48]), 0.999), col = "red", lty = 2, lwd = 0.8)
}

### remove outliers

if (include_volume) {
  ind = unique(which(diff_y[,25:36] > quantile(diff_y[,25:36], 0.999), arr.ind = TRUE)[,1] %% n_companies)
  temp = unique(which(diff_y[,25:36] < quantile(diff_y[,25:36], 0.001), arr.ind = TRUE)[,1] %% n_companies)
  ind = unique(c(ind, temp))
  temp = which(diff_y[,37:48] > 10, arr.ind = TRUE)[,1] %% n_companies
  ind = unique(c(ind, temp))
  ind[which(ind == 0)] = n_companies
  ind = sort(ind)
} else {
  ind = unique(which(diff_y[,13:24] > quantile(diff_y[,13:24], 0.999), arr.ind = TRUE)[,1] %% n_companies)
  temp = unique(which(diff_y[,13:24] < quantile(diff_y[,13:24], 0.001), arr.ind = TRUE)[,1] %% n_companies)
  ind = unique(c(ind, temp))
  temp = which(diff_y[,25:36] > 10, arr.ind = TRUE)[,1] %% n_companies
  ind = unique(c(ind, temp))
  ind[which(ind == 0)] = n_companies
  ind = sort(ind)
}

for (i in 1:10) {
  if (i == 1) {
    temp = ind
  } else {
    temp = c(temp, ind + (n_companies * (i - 1)))
  }
}

train_years = nrow(train_Y) / n_companies
test_years = nrow(test_Y) / n_companies

train_Y = train_Y[-temp[1:(length(ind) * train_years)],]
test_Y = test_Y[-temp[1:(length(ind) * test_years)],]
train_X = train_X[-temp[1:(length(ind) * train_years)],]
test_X = test_X[-temp[1:(length(ind) * test_years)],]
train_Ylag = train_Ylag[-temp[1:(length(ind) * train_years)],]
test_Ylag = test_Ylag[-temp[1:(length(ind) * test_years)],]

### plot new distributions
if (plot_or_not) {
  temp = rbind(train_Y, test_Y)
  
  par(mfrow = c(2, 2))
  hist(c(temp[,1:12]), breaks = 50, main = "volatility", xlab = "", 
       probability = TRUE)
  # abline(v = quantile(c(temp[,1:12]), 0.001), col = "red", lty = 2, lwd = 0.8)
  # abline(v = quantile(c(temp[,1:12]), 0.999), col = "red", lty = 2, lwd = 0.8)
  
  hist(c(temp[,13:24]), breaks = 50, main = "volume", xlab = "", 
       probability = TRUE)
  # abline(v = quantile(c(temp[,13:24]), 0.001), col = "red", lty = 2, lwd = 0.8)
  # abline(v = quantile(c(temp[,13:24]), 0.999), col = "red", lty = 2, lwd = 0.8)
  
  hist(c(temp[,25:36]), breaks = 50, main = "beta", xlab = "", 
       probability = TRUE)
  # abline(v = quantile(c(temp[,25:36]), 0.001), col = "red", lty = 2, lwd = 0.8)
  # abline(v = quantile(c(temp[,25:36]), 0.999), col = "red", lty = 2, lwd = 0.8)
  
  hist(c(temp[,37:48]), breaks = 50, main = "return", xlab = "", 
       probability = TRUE)
  # abline(v = quantile(c(temp[,37:48]), 0.001), col = "red", lty = 2, lwd = 0.8)
  # abline(v = quantile(c(temp[,37:48]), 0.999), col = "red", lty = 2, lwd = 0.8)
}


### normalize

if (catwise_scale) {
  sds = apply(train_Y, MARGIN = 2, FUN = sd)
  train_Y = t(t(train_Y) / sds)
  test_Y = t(t(test_Y) / sds)
  train_Ylag = t(t(train_Ylag) / sds)
  test_Ylag = t(t(test_Ylag) / sds)
}
