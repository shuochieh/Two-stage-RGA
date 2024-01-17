### Get stock data via Tidyquant package

tickers = unique(read.csv("./train_50.csv", header = FALSE)[,3])

final_dta = NULL

start_date = "2010-01-01"
end_date = "2019-12-31"

final_data = get_RVB(tickers, from = start_date, to = end_date)
failed_tickers = final_data$failed_tickers
final_data = final_data$data

write.csv(final_data, "./stock_prices_extracted.csv", row.names = FALSE)

