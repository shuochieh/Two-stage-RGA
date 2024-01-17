# Laso update: 2023-01-05
## check missing data and NAs

final_data = read.csv("./stock_prices_extracted.csv", header = TRUE)

u_symbols = unique(final_data$symbol)
n_symbols = length(u_symbols)
flawed_symbols = NULL

for (i in 1:n_symbols) {
  test_company = u_symbols[i]
  sub_data = filter(final_data, symbol == test_company)
  n_obs = dim(sub_data)[1]
  
  if (n_obs != 120) {
    cat(test_company, "has only", n_obs, "observations.", "\n")
    flawed_symbols = c(flawed_symbols, test_company)
  }
  if (sum(is.na(sub_data[,-c(1:3)])) != 0) {
    cat(test_company, "has NAs.", "\n")
    flawed_symbols = c(flawed_symbols, test_company)
  } 
  if (sum(is.nan(Reduce(c, sub_data[,-c(1:3)]))) != 0) {
    cat(test_company, "has NaNs.", "\n")
    flawed_symbols = c(flawed_symbols, test_company)
  }
  if (sum(is.infinite(Reduce(c, sub_data[,-c(1:3)]))) != 0) {
    cat(test_company, "has Inf values.", "\n")
    flawed_symbols = c(flawed_symbols, test_company)
  }
}

flawed_symbols = unique(flawed_symbols)
final_data = filter(final_data, !(symbol %in% flawed_symbols)) # only retains tickers with full 120 month observations.
final_data = filter(final_data, symbol != "SECI") # remove tickers with NAs

## Reshaping

flat = function (dta, sb, yr) {
  # dta, a dataframe containing symbol and year.
  sub_data = filter(dta, symbol == sb, year == yr)[,-c(1:3)]
  
  return(Reduce(c, sub_data))
}

tickers = unique(final_data$symbol)
n_tickers = length(tickers)

row_calc = function (i, t, n) {
  (t - 2010) * n + i
}

Y = matrix(NA, nrow = n_tickers * 10, ncol = 49)
for (t in 2010:2019) {
  for (i in 1:n_tickers) {
    Y[row_calc(i, t, n_tickers),] = c(as.numeric(t),
                                      as.numeric(flat(final_data, sb = tickers[i], yr = t)))
  }
}

## Merge with LSI features

num_topics = c(150, 200, 250, 300, 350, 400)

for (k in num_topics) {
  lsi_train = read.csv(paste0("./train_", k, ".csv"), header = FALSE)
  
  new_dta = matrix(NA, nrow = n_tickers * 10, ncol = 49 + ncol(lsi_train) - 3)
  for (i in 1:n_tickers) {
    for (t in 2010:2017) {
      ind = intersect(which(lsi_train[,3] == tickers[i]), 
                      which(lsi_train[,2] == t))
      if (length(ind) != 1) {
        stop(paste("index not uniquely defined. ticker id =", i, "year =", t))
      }
      if (lsi_train[ind, 3] != tickers[i]) {
        stop("ticker error")
      }
      features = Reduce(c, lsi_train[ind, -c(1:3)])
      new_dta[row_calc(i, t, n_tickers),] = c(Y[row_calc(i, t, n_tickers),], 
                                              features)
    }
  }
  
  lsi_test = read.csv(paste0("./test_", k, ".csv"), header = FALSE)
  
  for (i in 1:n_tickers) {
    for (t in 2018:2019) {
      ind = intersect(which(lsi_test[,3] == tickers[i]), 
                      which(lsi_test[,2] == t))
      if (length(ind) != 1) {
        stop(paste("index not uniquely defined. ticker id =", i, "year =", t))
      }
      if (lsi_test[ind, 3] != tickers[i]) {
        stop("ticker error")
      }
      features = Reduce(c, lsi_test[ind, -c(1:3)])
      new_dta[row_calc(i, t, n_tickers),] = c(Y[row_calc(i, t, n_tickers),], 
                                              features)
    }
  }
  
  write.table(new_dta, paste0("./final_data_lsi_", k, ".csv"), sep = ",", 
              row.names = FALSE, col.names = FALSE)
}


