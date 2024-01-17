library(tidyquant)
library(tidyverse)
library(ggplot2)
library(attempt)

get_returns = function (stock_symbol, from, to, period = "monthly") {
  stock_symbol %>%
    tq_get(get = "stock.prices",
           from = from, 
           to = to) %>%
      group_by(symbol) %>%
        tq_transmute(select = adjusted,
                     mutate_fun = periodReturn,
                     type = "arithmetic",
                     period = period)
}

get_beta = function (daily_returns, from, to) {
  # daily_returns contains three columns. One is date; one is symbol; 
  # the other daily.returns. 
  
  sp500 = get_returns("^GSPC", from = from, to = to, period = "daily")

  tickers = unique(daily_returns$symbol)
  betas = tibble("symbol" = character(), "year" = numeric(), "month" = numeric(), 
                 "beta" = numeric())
  
  for (ticker in tickers) {
    adj_returns = filter(daily_returns, symbol == ticker)
    merged_data = merge(adj_returns, sp500, by = "date")
    merged_data$year = year(merged_data$date)
    merged_data$month = month(merged_data$date)
    
    years = unique(merged_data$year)
    for (yr in years) {
      sub_data = filter(merged_data, year == yr)
      for (m in sort(unique(sub_data$month))) {
        b = lm(daily.returns.x~daily.returns.y, data = filter(sub_data, month == m))$coefficients[2]
        betas = add_row(betas, symbol = ticker, year = yr, month = m, beta = b)
      }
    }
  }
  
  return(betas)
}

get_volatility = function (daily_returns) {
  tickers = unique(daily_returns$symbol)
  vol = tibble("symbol" = character(), "year" = numeric(), "month" = numeric(), 
               "volatility" = numeric())
  
  for (ticker in tickers) {
    adj_returns = filter(daily_returns, symbol == ticker)
    years = unique(year(adj_returns$date))
    
    for (yr in years) {
      sub_data = filter(adj_returns, year(date) == yr)
      for (m in sort(unique(month(sub_data$date)))) {
        v = as.numeric(log(var(filter(sub_data, month(date) == m)["daily.returns"])))
        vol = add_row(vol, symbol = ticker, year = yr, month = m, volatility = v)
      }
    }
  }
  
  return(vol)
}

get_volume = function (raw_data) {
  v = tibble("symbol" = character(), "year" = numeric(), "month" = numeric(),
             "volume" = numeric())
  tickers = unique(raw_data$symbol)
  
  for (ticker in tickers) {
    company = filter(raw_data, symbol == ticker)
    years = unique(year(company$date))
    
    for (yr in years) {
      sub_data = filter(company, year(date) == yr)
      for (m in sort(unique(month(sub_data$date)))) {
        avg_v = as.numeric(mean(filter(sub_data, month(date) == m)$volume)) # mean is defunct for data frames
        v = add_row(v, symbol = ticker, year = yr, month = m, volume = log(avg_v))
      }
    }
  }
  
  return(v)
}

get_raw_data = function (stock_symbol, from, to) {
  n_symbol = length(stock_symbol)
  failed_tickers = NULL
  raw_data = NULL
  
  for (i in 1:n_symbol) {
    cat("catching", stock_symbol[i])
    temp = try_catch(tq_get(stock_symbol[i], get = "stock.prices", from = from,
                            to = to), .e = ~NULL, .w = ~NULL)
    if (is.null(temp)) {
      failed_tickers = c(failed_tickers, stock_symbol[i])
      cat(" .... extraction failed", "\n")
    } else {
      raw_data = rbind(raw_data, temp)
      cat(" ....", i, "/", n_symbol, "tickers completed", "\n")
    }
  }
  
  return(list("raw_data" = raw_data, "failed_tickers" = failed_tickers))
}

new_order = function (d) {
  # d is a data frame containing symbol, year, and month.

  i = 0
  res = NULL
  tickers = unique(d$symbol)
  for (ticker in tickers) {
    sub_data = filter(d, symbol == ticker)
    years = sort(unique(sub_data$year))
    
    for (yr in years) {
      months = filter(sub_data, year == yr)$month
      res = c(res, i + order(months))
      i = length(res)
    }
  }
  
  return(res)
}

get_RVB = function (stock_symbol, from, to) {
  
  # raw_data = tq_get(stock_symbol, get = "stock.prices", from = from, to = to) %>%
  #   group_by(symbol)
  raw_data_ = get_raw_data(stock_symbol, from, to)
  raw_data = raw_data_$raw_data
  failed_tickers = raw_data_$failed_tickers
  
  cat("computing daily returns...")
  raw_data = group_by(raw_data, symbol)
  daily_returns = tq_transmute(raw_data, select = adjusted, mutate_fun = periodReturn,
                               type = "arithmetic", period = "daily")
  cat(" done", "\n")
    
  # get betas
  cat("computing market betas...")
  betas = get_beta(daily_returns, from = from, to = to)
  cat(" done.", "\n")
  
  # get monthly adjusted returns
  cat("computing monthly returns...")
  returns = tq_transmute(raw_data, select = adjusted, mutate_fun = periodReturn,
                         type = "arithmetic", period = "monthly")
  returns$month = month(returns$date)
  returns$year = year(returns$date)
  returns = returns[,!(names(returns) == "date")]
  names(returns)[names(returns) == "monthly.returns"] = "returns"
  cat(" done.", "\n")
  
  # get monthly variance of adjusted returns
  cat("computing log volatility...")
  vlty = get_volatility(daily_returns)
  cat(" done.", "\n")
  
  # get volume
  cat("computing log trading volume...")
  vlm = get_volume(raw_data)
  cat(" done.", "\n")
  
  cat("wrapping up...")
  res = merge(vlty, vlm, by = c("symbol", "year", "month"))
  res = merge(res, betas, by = c("symbol", "year", "month"))
  res = merge(res, returns, by = c("symbol", "year", "month"))
  res = res[new_order(res),]
  cat(" done.", "\n")
  
  return(list("data" = res, "failed_tickers" = failed_tickers))
}

get_yearly_beta = function (daily_returns, from, to) {
  
  sp500 = get_returns("^GSPC", from = from, to = to, period = "daily")
  
  tickers = unique(daily_returns$symbol)
  betas = tibble("symbol" = character(), "year" = numeric(), "beta" = numeric())
  
  for (ticker in tickers) {
    adj_returns = filter(daily_returns, symbol == ticker)
    merged_data = merge(adj_returns, sp500, by = "date")
    merged_data$year = year(merged_data$date)
    
    years = unique(merged_data$year)
    for (yr in years) {
      sub_data = filter(merged_data, year == yr)
      b = lm(daily.returns.x~daily.returns.y, data = sub_data)$coefficients[2]
      betas = add_row(betas, symbol = ticker, year = yr, beta = b)
    }
  }
  
  return(betas)
}

get_yearly_volatility = function (daily_returns) {
  tickers = unique(daily_returns$symbol)
  vol = tibble("symbol" = character(), "year" = numeric(), "volatility" = numeric())
  
  for (ticker in tickers) {
    adj_returns = filter(daily_returns, symbol == ticker)
    years = unique(year(adj_returns$date))
    
    for (yr in years) {
      sub_data = filter(adj_returns, year(date) == yr)
      v = as.numeric(log(var(sub_data["daily.returns"])))
      vol = add_row(vol, symbol = ticker, year = yr, volatility = v)
    }
  }
  
  return(vol)
}

get_yearly_volume = function (raw_data) {
  v = tibble("symbol" = character(), "year" = numeric(), "volume" = numeric())
  tickers = unique(raw_data$symbol)
  
  for (ticker in tickers) {
    company = filter(raw_data, symbol == ticker)
    years = unique(year(company$date))
    
    for (yr in years) {
      sub_data = filter(company, year(date) == yr)
      avg_v = as.numeric(mean(sub_data$volume)) # mean is defunct for data frames
      v = add_row(v, symbol = ticker, year = yr, volume = log(avg_v))
    }
  }
  
  return(v)
}

get_yearly_RVB = function (stock_symbol, from, to) {
  
  raw_data_ = get_raw_data(stock_symbol, from, to)
  raw_data = raw_data_$raw_data
  failed_tickers = raw_data_$failed_tickers
  
  cat("computing daily returns...")
  raw_data = group_by(raw_data, symbol)
  daily_returns = tq_transmute(raw_data, select = adjusted, mutate_fun = periodReturn,
                               type = "arithmetic", period = "daily")
  cat(" done", "\n")
  
  # get betas
  cat("computing market betas...")
  betas = get_yearly_beta(daily_returns, from = from, to = to)
  cat(" done.", "\n")
  
  # get yearly adjusted returns
  cat("computing yearly returns...")
  returns = tq_transmute(raw_data, select = adjusted, mutate_fun = periodReturn,
                         type = "arithmetic", period = "yearly")
  returns$year = year(returns$date)
  returns = returns[,!(names(returns) == "date")]
  names(returns)[names(returns) == "yearly.returns"] = "returns"
  cat(" done.", "\n")
  
  # get monthly variance of adjusted returns
  cat("computing log volatility...")
  vlty = get_yearly_volatility(daily_returns)
  cat(" done.", "\n")
  
  # get volume
  cat("computing log trading volume...")
  vlm = get_yearly_volume(raw_data)
  cat(" done.", "\n")
  
  cat("wrapping up...")
  res = merge(vlty, vlm, by = c("symbol", "year"))
  res = merge(res, betas, by = c("symbol", "year"))
  res = merge(res, returns, by = c("symbol", "year"))
  cat(" done.", "\n")
  
  return(list("data" = res, "failed_tickers" = failed_tickers))
}
