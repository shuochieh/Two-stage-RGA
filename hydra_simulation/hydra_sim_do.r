### Testing the performance of Hydra, Hydra2, RGA, TSRGA

### Update: 2023-01-30
### Last update: 2024-01-18
setwd("./hydra_simulation")

library(glmnet)
library(ggplot2)
source("hydra.r")

array.id = as.numeric(Sys.getenv("SLURM_ARRAY_TASK_ID"))

# if (array.id %/% 3 == 0 || array.id %/% 3 == 3) {
#   spec = 1
# } else if (array.id %/% 3 == 1) {
#   spec = 2
# } else if (array.id %/% 3 == 2) {
#   spec = 3
# } else {
#   stop(paste("array.id categorizing error. array.id", array.id))
# }

# if (array.id %% 3 == 0) {
#   n = 800
#   p = 1200
# } else if (array.id %% 3 == 1) {
#   n = 1200
#   p = 2000
# } else if (array.id %% 3 == 2) {
#   n = 1500
#   p = 3000
# } else {
#   stop(paste("array.id categorizing error. array.id", array.id))
# }

spec = ifelse(array.id == 1, 1, 2)
n = 1500
p = 3000

cat("Starting simulation: n", n, "p", p, "spec", spec, "\n")

n_sim = 100
n_iter = 600
c = 10                        # number of nodes

if (spec == 1) {
  step_size = 4               # Hydra step-size
  step_size_factor = 4        # Hydra2 step-size factor
} else if (spec == 2) {
  step_size = 500
  step_size_factor = 500
} else if (spec == 3) {
  step_size = 20
  step_size_factor = 20
}

b_error_lasso = rep(NA, n_sim)
y_error_lasso = rep(NA, n_sim)
os_rmse_lasso = rep(NA, n_sim)

b_error_hydra = matrix(NA, ncol = n_sim, nrow = n_iter)
y_error_hydra = matrix(NA, ncol = n_sim, nrow = n_iter)
elapsed_time_hydra = matrix(NA, ncol = n_sim, nrow = n_iter)
os_rmse_hydra = matrix(NA, ncol = n_sim, nrow = n_iter)

b_error_hydra_5 = matrix(NA, ncol = n_sim, nrow = n_iter)
y_error_hydra_5 = matrix(NA, ncol = n_sim, nrow = n_iter)
elapsed_time_hydra_5 = matrix(NA, ncol = n_sim, nrow = n_iter)
os_rmse_hydra_5 = matrix(NA, ncol = n_sim, nrow = n_iter)

b_error_hydra2 = matrix(NA, ncol = n_sim, nrow = n_iter)
y_error_hydra2 = matrix(NA, ncol = n_sim, nrow = n_iter)
elapsed_time_hydra2 = matrix(NA, ncol = n_sim, nrow = n_iter)
os_rmse_hydra2 = matrix(NA, ncol = n_sim, nrow = n_iter)

b_error_hydra2_5 = matrix(NA, ncol = n_sim, nrow = n_iter)
y_error_hydra2_5 = matrix(NA, ncol = n_sim, nrow = n_iter)
elapsed_time_hydra2_5 = matrix(NA, ncol = n_sim, nrow = n_iter)
os_rmse_hydra2_5 = matrix(NA, ncol = n_sim, nrow = n_iter)

b_error_rga = matrix(NA, ncol = n_sim, nrow = n_iter)
y_error_rga = matrix(NA, ncol = n_sim, nrow = n_iter)
elapsed_time_rga = matrix(NA, ncol = n_sim, nrow = n_iter)
os_rmse_rga = matrix(NA, ncol = n_sim, nrow = n_iter)

b_error_rga_reg = matrix(NA, ncol = n_sim, nrow = n_iter)
y_error_rga_reg = matrix(NA, ncol = n_sim, nrow = n_iter)
elapsed_time_rga_reg = matrix(NA, ncol = n_sim, nrow = n_iter)
os_rmse_rga_reg = matrix(NA, ncol = n_sim, nrow = n_iter)

b_error_tsrga = matrix(NA, ncol = n_sim, nrow = n_iter)
y_error_tsrga = matrix(NA, ncol = n_sim, nrow = n_iter)
elapsed_time_tsrga = matrix(NA, ncol = n_sim, nrow = n_iter)
os_rmse_tsrga = matrix(NA, ncol = n_sim, nrow = n_iter)

loss_hydra = matrix(0, ncol = n_sim, nrow = n_iter)
loss_hydra2 = matrix(0, ncol = n_sim, nrow = n_iter)
loss_hydra_5 = matrix(0, ncol = n_sim, nrow = n_iter)
loss_hydra2_5 = matrix(0, ncol = n_sim, nrow = n_iter)
loss_lasso = rep(NA, ncol = n_sim)

s = floor(p^(1/3))                  # number of relevant variables

for (ii in 1:n_sim){
  if (spec != 2) {
    b = sample(c(-1, 1), size = s, replace = TRUE) * runif(s, min = 2.5, max = 5.5)
  } else if (spec == 2) {
    b = seq(from = 2.5, by = 1.2, length.out = s)
  }
  beta = rep(0, p)
  ptt = vector(mode = "list", length = c)
  for (i in 1:c) {
    ptt[[i]] = ((i - 1) * p / c + 1):(i * (p / c))
  }
  beta[sample(1:p, size = s)] = b
  
  
  if (spec == 1) {
    X = matrix(rt(n * p, df = 5), nrow = n, ncol = p)
    X_new = matrix(rt(500 * p, df = 5), nrow = 500, ncol = p)
  } else if (spec == 2) {
    X = matrix(rnorm(n * p), nrow = n, ncol = p)
    X_new = matrix(rnorm(500 * p), nrow = 500, ncol = p)
    X = X + rnorm(n, sd = 1)
    X_new = X_new + rnorm(500, sd = 1)
  } else if (spec == 3) {
    ### Specification 3 is no longer used
    X = matrix(0, nrow = n, ncol = p)
    block_size = n / c
    for (i in 1:c) {
      X[((i - 1) * block_size + 1):(i * block_size),ptt[[i]]] = rnorm(block_size * length(ptt[[i]]))
      if (i != c) {
        X[((c - 1) * block_size + 1):(c * block_size),ptt[[i]]] = rnorm(block_size * length(ptt[[i]]))
      }
    }
  }
  
  y = X %*% beta + rt(n, df = 5)
  y_new = X_new %*% beta + rt(500, df = 5)
  
  time = proc.time()[3]
  pilot = cv.glmnet(x = X, y = y, nfolds = 5)
  lambda = pilot$lambda.min * n

  b_error_lasso[ii] = sum((coef.glmnet(pilot, s = "lambda.min")[-1] - beta)^2)
  y_error_lasso[ii] = mean((y - predict(pilot, newx = X, s = "lambda.min"))^2)
  loss_lasso[ii] = 0.5 * sum((y - predict(pilot, newx = X, s = "lambda.min"))^2) + lambda * sum(abs(coef(pilot, s = "lambda.min")[-1]))
  os_rmse_lasso[ii] = sqrt(mean((y_new - predict(pilot, newx = X_new, s = "lambda.min"))^2))
    
  tau = floor(0.25 * (p / c))         # number of coordinates to update for Hydra
  
  
  res_hydra = hydra(y = y, X = X, ptt = ptt, lambda = lambda,
                    step_size = step_size, tau = tau, n_iter = n_iter,
                    beta_truth = beta, x_new = X_new, y_new = y_new)
  res_hydra2 = hydra2(y = y, X = X, ptt = ptt, lambda = lambda, tau = tau,
                      step_size_factor = step_size_factor, n_iter = n_iter,
                      beta_truth = beta, x_new = X_new, y_new = y_new)
  
  b_error_hydra[,ii] = res_hydra$b_error
  y_error_hydra[,ii] = res_hydra$y_error
  elapsed_time_hydra[,ii] = res_hydra$elapsed_time
  os_rmse_hydra[,ii] = res_hydra$RMSE
  
  b_error_hydra2[,ii] = res_hydra2$b_error
  y_error_hydra2[,ii] = res_hydra2$y_error
  elapsed_time_hydra2[,ii] = res_hydra2$elapsed_time
  os_rmse_hydra2[,ii] = res_hydra2$RMSE
  
  loss_hydra[,ii] = res_hydra$loss
  loss_hydra2[,ii] = res_hydra2$loss

  tau = floor(0.5 * (p / c))         # number of coordinates to update for Hydra
  res_hydra = hydra(y = y, X = X, ptt = ptt, lambda = lambda,
                    step_size = step_size, tau = tau, n_iter = n_iter,
                    beta_truth = beta, x_new = X_new, y_new = y_new)
  res_hydra2 = hydra2(y = y, X = X, ptt = ptt, lambda = lambda, tau = tau,
                      step_size_factor = step_size_factor, n_iter = n_iter,
                      beta_truth = beta, x_new = X_new, y_new = y_new)

  b_error_hydra_5[,ii] = res_hydra$b_error
  y_error_hydra_5[,ii] = res_hydra$y_error
  elapsed_time_hydra_5[,ii] = res_hydra$elapsed_time
  os_rmse_hydra_5[,ii] = res_hydra$RMSE

  b_error_hydra2_5[,ii] = res_hydra2$b_error
  y_error_hydra2_5[,ii] = res_hydra2$y_error
  elapsed_time_hydra2_5[,ii] = res_hydra2$elapsed_time
  os_rmse_hydra2_5[,ii] = res_hydra2$RMSE

  loss_hydra_5[,ii] = res_hydra$loss
  loss_hydra2_5[,ii] = res_hydra2$loss
  
  res_rga = rga_core(y = y, X = X, L = 500, Kn = n_iter, beta_truth = beta,
                     x_new = X_new, y_new = y_new)
  res_rga_reg = rga_core(y = y, X = X, L = sum(abs(beta)), Kn = n_iter, beta_truth = beta,
                         x_new = X_new, y_new = y_new)
  res_tsrga = tsrga(y = y, X = X, L = 500, Kn1 = n_iter / 2, Kn2 = n_iter / 2,
                    t_n = 1 / (10 * log(n)), beta_truth = beta,
                    x_new = X_new, y_new = y_new)
  b_error_rga[,ii] = res_rga$b_error
  y_error_rga[,ii] = res_rga$y_error
  elapsed_time_rga[,ii] = res_rga$elapsed_time
  os_rmse_rga[,ii] = res_rga$RMSE
  
  b_error_rga_reg[,ii] = res_rga_reg$b_error
  y_error_rga_reg[,ii] = res_rga_reg$y_error
  elapsed_time_rga_reg[,ii] = res_rga_reg$elapsed_time
  os_rmse_rga_reg[,ii] = res_rga_reg$RMSE
  
  b_error_tsrga[,ii] = res_tsrga$b_error[1:n_iter]
  y_error_tsrga[,ii] = res_tsrga$y_error[1:n_iter]
  elapsed_time_tsrga[,ii] = res_tsrga$elapsed_time
  os_rmse_tsrga[,ii] = res_tsrga$RMSE

  cat("simulation", ii, ":", round(proc.time()[3] - time, 2), "sec", "\n")
}

plot_dta = data.frame(
  x_axis = 1:n_iter,
  y1 = log(rowMeans(b_error_tsrga))[1:n_iter],
  y2 = log(rowMeans(b_error_rga))[1:n_iter],
  y3 = log(rowMeans(b_error_rga_reg))[1:n_iter],
  y4 = log(rowMeans(b_error_hydra))[1:n_iter],
  y5 = log(rowMeans(b_error_hydra_5))[1:n_iter],
  y6 = log(rowMeans(b_error_hydra2))[1:n_iter],
  y7 = log(rowMeans(b_error_hydra2_5))[1:n_iter],
  y8 = log(mean(b_error_lasso))
)

pdf(paste0("n", n, "_p", p, "_spec", spec, ".pdf"), width = 12, height = 10)
par(mfrow = c(1, 1))
dta_long = tidyr::gather(plot_dta, key = "Methods", value = "Value", -x_axis)
I = ggplot(dta_long, aes(x = x_axis, y = Value, color = Methods)) + 
  geom_line(aes(linetype = Methods), size = 0.8, show.legend = T) +
  scale_linetype_manual(label = c("TSRGA", "RGA (L=500)", "RGA (oracle L)", "Hydra (25%)", "Hydra (50%)",
                                  "Hydra2 (25%)", "Hydra2 (50%)", "Lasso"),
                        values = c("solid", "dashed", "solid", "solid", "dashed", "solid", "dashed", "dashed")) +
  scale_color_manual(label = c("TSRGA", "RGA (L=500)", "RGA (oracle L)", "Hydra (25%)", "Hydra (50%)",
                               "Hydra2 (25%)", "Hydra2 (50%)", "Lasso"),
                     values = c("black", "red", "red", "blue", "blue", "green", "green", "magenta")) +
  labs(title = "", x = "iteration", y = "log estimation error")
if (n == 1500) {
  I + theme(legend.text = element_text(size = 10), legend.key.width = unit(0.8, 'cm'))
} else {
  I + theme(legend.position = "none")
}
dev.off()


plot_dta = data.frame(
  x_axis = 1:n_iter,
  y1 = log(rowMeans(os_rmse_tsrga))[1:n_iter],
  y2 = log(rowMeans(os_rmse_rga))[1:n_iter],
  y3 = log(rowMeans(os_rmse_rga_reg))[1:n_iter],
  y4 = log(rowMeans(os_rmse_hydra))[1:n_iter],
  y5 = log(rowMeans(os_rmse_hydra_5))[1:n_iter],
  y6 = log(rowMeans(os_rmse_hydra2))[1:n_iter],
  y7 = log(rowMeans(os_rmse_hydra2_5))[1:n_iter],
  y8 = log(mean(os_rmse_lasso))
)

pdf(paste0("n", n, "_p", p, "_spec", spec, "_RMSE.pdf"), width = 12, height = 10)
par(mfrow = c(1, 1))
dta_long = tidyr::gather(plot_dta, key = "Methods", value = "Value", -x_axis)
I = ggplot(dta_long, aes(x = x_axis, y = Value, color = Methods)) + 
  geom_line(aes(linetype = Methods), size = 0.8, show.legend = T) +
  scale_linetype_manual(label = c("TSRGA", "RGA (L=500)", "RGA (oracle L)", "Hydra (25%)", "Hydra (50%)",
                                  "Hydra2 (25%)", "Hydra2 (50%)", "Lasso"),
                        values = c("solid", "dashed", "solid", "solid", "dashed", "solid", "dashed", "dashed")) +
  scale_color_manual(label = c("TSRGA", "RGA (L=500)", "RGA (oracle L)", "Hydra (25%)", "Hydra (50%)",
                               "Hydra2 (25%)", "Hydra2 (50%)", "Lasso"),
                     values = c("black", "red", "red", "blue", "blue", "green", "green", "magenta")) +
  labs(title = "", x = "iteration", y = "log RMSE")
if (n == 1500) {
  I + theme(legend.text = element_text(size = 10), legend.key.width = unit(0.8, 'cm'))
} else {
  I + theme(legend.position = "none")
}
dev.off()


plot_dta = data.frame(
  elapsed_time = c(rowMeans(elapsed_time_tsrga), rowMeans(elapsed_time_rga),
                   rowMeans(elapsed_time_rga_reg), rowMeans(elapsed_time_hydra),
                   rowMeans(elapsed_time_hydra_5), rowMeans(elapsed_time_hydra2),
                   rowMeans(elapsed_time_hydra2_5)),
  y1 = c(log(rowMeans(b_error_tsrga)), log(rowMeans(b_error_rga)), log(rowMeans(b_error_rga_reg)),
         log(rowMeans(b_error_hydra)), log(rowMeans(b_error_hydra_5)), 
         log(rowMeans(b_error_hydra2)), log(rowMeans(b_error_hydra2_5))),
  group = rep(c("TSRGA", "RGA (L=500)", "RGA (oracle L)", "Hydra (25%)", "Hydra (50%)", 
                "Hydra2 (25%)", "Hydra2 (50%)"), each = n_iter)
)

pdf(paste0("n", n, "_p", p, "_spec", spec, "_time.pdf"), width = 12, height = 10)
par(mfrow = c(1, 1))
I = ggplot(plot_dta, aes(x = elapsed_time, y = y1, color = group, group = group)) +
  geom_line(aes(linetype = group), size = 0.8) +
  scale_linetype_manual(values = c("TSRGA" = "solid", "RGA (L=500)" = "dashed", 
                                   "RGA (oracle L)" = "solid", "Hydra (25%)" = "solid", 
                                   "Hydra (50%)" = "dashed", "Hydra2 (25%)" = "solid", 
                                   "Hydra2 (50%)" = "dashed")) +
  scale_color_manual(values = c("TSRGA" = "black", "RGA (L=500)" = "red", 
                                "RGA (oracle L)" = "red", "Hydra (25%)" = "blue", 
                                "Hydra (50%)" = "blue", "Hydra2 (25%)" = "green", 
                                "Hydra2 (50%)" = "green")) +
  labs(title = "", x = "elapsed time (sec)", y = "log estimation error")
if (n == 1500) {
  I + theme(legend.text = element_text(size = 10), legend.key.width = unit(0.8, 'cm'))
} else {
  I + theme(legend.position = "none")
}
dev.off()

plot_dta = data.frame(
  elapsed_time = c(rowMeans(elapsed_time_tsrga), rowMeans(elapsed_time_rga),
                   rowMeans(elapsed_time_rga_reg), rowMeans(elapsed_time_hydra),
                   rowMeans(elapsed_time_hydra_5), rowMeans(elapsed_time_hydra2),
                   rowMeans(elapsed_time_hydra2_5)),
  y1 = c(log(rowMeans(os_rmse_tsrga)), log(rowMeans(os_rmse_rga)), log(rowMeans(os_rmse_rga_reg)),
         log(rowMeans(os_rmse_hydra)), log(rowMeans(os_rmse_hydra_5)), 
         log(rowMeans(os_rmse_hydra2)), log(rowMeans(os_rmse_hydra2_5))),
  group = rep(c("TSRGA", "RGA (L=500)", "RGA (oracle L)", "Hydra (25%)", "Hydra (50%)", 
                "Hydra2 (25%)", "Hydra2 (50%)"), each = n_iter)
)

pdf(paste0("n", n, "_p", p, "_spec", spec, "_RMSEtime.pdf"), width = 12, height = 10)
par(mfrow = c(1, 1))
I = ggplot(plot_dta, aes(x = elapsed_time, y = y1, color = group, group = group)) +
  geom_line(aes(linetype = group), size = 0.8) +
  scale_linetype_manual(values = c("TSRGA" = "solid", "RGA (L=500)" = "dashed", 
                                   "RGA (oracle L)" = "solid", "Hydra (25%)" = "solid", 
                                   "Hydra (50%)" = "dashed", "Hydra2 (25%)" = "solid", 
                                   "Hydra2 (50%)" = "dashed")) +
  scale_color_manual(values = c("TSRGA" = "black", "RGA (L=500)" = "red", 
                                "RGA (oracle L)" = "red", "Hydra (25%)" = "blue", 
                                "Hydra (50%)" = "blue", "Hydra2 (25%)" = "green", 
                                "Hydra2 (50%)" = "green")) +
  labs(title = "", x = "elapsed time (sec)", y = "log RMSE")
if (n == 1500) {
  I + theme(legend.text = element_text(size = 10), legend.key.width = unit(0.8, 'cm'))
} else {
  I + theme(legend.position = "none")
}
dev.off()

# pdf(paste0("n", n, "_p", p, "_spec", spec,"_training_error.pdf"), width = 12, height = 10)
# y_l = log(min(rowMeans(y_error_tsrga), rowMeans(y_error_rga), rowMeans(y_error_hydra),
#               rowMeans(y_error_hydra2), rowMeans(y_error_hydra_5), rowMeans(y_error_hydra2_5),
#               mean(y_error_lasso),
#               na.rm = TRUE))
# y_u = log(max(rowMeans(y_error_tsrga), rowMeans(y_error_rga), rowMeans(y_error_hydra),
#               rowMeans(y_error_hydra2), rowMeans(y_error_hydra_5), rowMeans(y_error_hydra2_5),
#               mean(y_error_lasso),
#               na.rm = TRUE))
# plot(x = 1:n_iter, y = log(rowMeans(y_error_tsrga))[1:n_iter], xlab = "iterations", 
#      ylab = "training error (log)", type = "l", lwd = 2, ylim = c(y_l, y_u))
#      #,main = paste0("n=", n, "; p=", p, "; spec: ", spec))
# lines(x = 1:n_iter, y = log(rowMeans(y_error_rga))[1:n_iter], col = "red", 
#       type = "l", lwd = 2)
# lines(x = 1:n_iter, y = log(rowMeans(y_error_rga_reg))[1:n_iter], col = "red", 
#       type = "l", lwd = 2, lty = 2)
# lines(x = 1:n_iter, y = log(rowMeans(y_error_hydra))[1:n_iter], col = "blue", 
#       type = "l", lwd = 2)
# lines(x = 1:n_iter, y = log(rowMeans(y_error_hydra_5))[1:n_iter], col = "blue", 
#       type = "l", lwd = 2, lty = 2)
# lines(x = 1:n_iter, y = log(rowMeans(y_error_hydra2))[1:n_iter], col = "green", 
#       type = "l", lwd = 2)
# lines(x = 1:n_iter, y = log(rowMeans(y_error_hydra2_5))[1:n_iter], col = "green", 
#       type = "l", lwd = 2, lty = 2)
# abline(h = mean(y_error_lasso), col = "gray", lty = 2, lwd = 1.5)
# # txt = paste0("n=", n, "; p=", p, "; spec: ", spec)
# # mtext(txt, line = -2, outer = TRUE, cex = 2)
# dev.off()
# 
# pdf(paste0("n", n, "_p", p, "_spec", spec,"_loss.pdf"), width = 8, height = 8)
# par(mfrow = c(1, 1))
# plot(x = 1:nrow(loss_hydra), y = log(rowMeans(loss_hydra)), xlab = "iterations",
#      ylab = "loss (log)", type = "l", col = "blue", lwd = 2,
#      ylim = c(min(log(rowMeans(loss_hydra)), log(rowMeans(loss_hydra2)), log(mean(loss_lasso))),
#               max(log(rowMeans(loss_hydra)), log(rowMeans(loss_hydra2)), log(mean(loss_lasso)))))
#     #,main = paste0("n=", n, "; p=", p, "; spec: ", spec))
# lines(x = 1:nrow(loss_hydra_5), y = log(rowMeans(loss_hydra_5)), col = "blue", lwd = 2, lty = 2)
# lines(x = 1:nrow(loss_hydra2), y = log(rowMeans(loss_hydra2)), col = "green", lwd = 2)
# lines(x = 1:nrow(loss_hydra2_5), y = log(rowMeans(loss_hydra2_5)), col = "green", lwd = 2, lty = 2)
# abline(h = log(mean(loss_lasso)), col = "gray", lty = 2, lwd = 1.5)
# # txt = paste0("n=", n, "; p=", p, "; spec: ", spec)
# # mtext(txt, line = -2, outer = TRUE, cex = 2)
# dev.off()
