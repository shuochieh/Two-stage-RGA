### Last update: 2023-01-30
setwd("./hydra_simulation")

library(glmnet)
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
### Testing the performance of Hydra, Hydra2, TSRGA

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
b_error_hydra = matrix(NA, ncol = n_sim, nrow = n_iter)
y_error_hydra = matrix(NA, ncol = n_sim, nrow = n_iter)
b_error_hydra_5 = matrix(NA, ncol = n_sim, nrow = n_iter)
y_error_hydra_5 = matrix(NA, ncol = n_sim, nrow = n_iter)
b_error_hydra2 = matrix(NA, ncol = n_sim, nrow = n_iter)
y_error_hydra2 = matrix(NA, ncol = n_sim, nrow = n_iter)
b_error_hydra2_5 = matrix(NA, ncol = n_sim, nrow = n_iter)
y_error_hydra2_5 = matrix(NA, ncol = n_sim, nrow = n_iter)
b_error_rga = matrix(NA, ncol = n_sim, nrow = n_iter)
y_error_rga = matrix(NA, ncol = n_sim, nrow = n_iter)
b_error_rga_reg = matrix(NA, ncol = n_sim, nrow = n_iter)
y_error_rga_reg = matrix(NA, ncol = n_sim, nrow = n_iter)
b_error_tsrga = matrix(NA, ncol = n_sim, nrow = n_iter)
y_error_tsrga = matrix(NA, ncol = n_sim, nrow = n_iter)
  
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
  } else if (spec == 2) {
    X = matrix(rnorm(n * p), nrow = n, ncol = p)
    X = X + rnorm(n, sd = 1)
  } else if (spec == 3) {
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
  
  time = proc.time()[3]
  pilot = cv.glmnet(x = X, y = y, nfolds = 5)
  lambda = pilot$lambda.min * n

  b_error_lasso[ii] = sum((coef.glmnet(pilot, s = "lambda.min")[-1] - beta)^2)
  y_error_lasso[ii] = mean((y - predict(pilot, newx = X, s = "lambda.min"))^2)
  loss_lasso[ii] = 0.5 * sum((y - predict(pilot, newx = X, s = "lambda.min"))^2) + lambda * sum(abs(coef(pilot, s = "lambda.min")[-1]))
    
  tau = floor(0.25 * (p / c))         # number of coordinates to update for Hydra
  res_hydra = hydra(y = y, X = X, ptt = ptt, lambda = lambda,
                    step_size = step_size, tau = tau, n_iter = n_iter,
                    beta_truth = beta)
  res_hydra2 = hydra2(y = y, X = X, ptt = ptt, lambda = lambda, tau = tau,
                      step_size_factor = step_size_factor, n_iter = n_iter,
                      beta_truth = beta)
  b_error_hydra[,ii] = res_hydra$b_error
  y_error_hydra[,ii] = res_hydra$y_error
  b_error_hydra2[,ii] = res_hydra2$b_error
  y_error_hydra2[,ii] = res_hydra2$y_error
  loss_hydra[,ii] = res_hydra$loss
  loss_hydra2[,ii] = res_hydra2$loss

  tau = floor(0.5 * (p / c))         # number of coordinates to update for Hydra
  res_hydra = hydra(y = y, X = X, ptt = ptt, lambda = lambda,
                    step_size = step_size, tau = tau, n_iter = n_iter,
                    beta_truth = beta)
  res_hydra2 = hydra2(y = y, X = X, ptt = ptt, lambda = lambda, tau = tau,
                      step_size_factor = step_size_factor, n_iter = n_iter,
                      beta_truth = beta)
  b_error_hydra_5[,ii] = res_hydra$b_error
  y_error_hydra_5[,ii] = res_hydra$y_error
  b_error_hydra2_5[,ii] = res_hydra2$b_error
  y_error_hydra2_5[,ii] = res_hydra2$y_error
  loss_hydra_5[,ii] = res_hydra$loss
  loss_hydra2_5[,ii] = res_hydra2$loss
  
  res_rga = rga_core(y = y, X = X, L = 500, Kn = n_iter, beta_truth = beta)
  res_rga_reg = rga_core(y = y, X = X, L = sum(abs(beta)), Kn = n_iter, beta_truth = beta)
  res_tsrga = tsrga(y = y, X = X, L = 500, Kn1 = n_iter / 2, Kn2 = n_iter / 2,
                    t_n = 1 / (10 * log(n)), beta_truth = beta)
  b_error_rga[,ii] = res_rga$b_error
  y_error_rga[,ii] = res_rga$y_error
  b_error_rga_reg[,ii] = res_rga_reg$b_error
  y_error_rga_reg[,ii] = res_rga_reg$y_error
  b_error_tsrga[,ii] = res_tsrga$b_error[1:n_iter]
  y_error_tsrga[,ii] = res_tsrga$y_error[1:n_iter]

  cat("simulation", ii, ":", round(proc.time()[3] - time, 2), "sec", "\n")
}

# n_iter = n_iter * 1.2
pdf(paste0("n", n, "_p", p, "_spec", spec,".pdf"), width = 12, height = 10)
par(mfrow = c(1, 1))
y_l = log(min(rowMeans(b_error_tsrga), rowMeans(b_error_rga), rowMeans(b_error_hydra),
              rowMeans(b_error_hydra2), rowMeans(b_error_hydra_5), rowMeans(b_error_hydra2_5),
              mean(b_error_lasso),
              na.rm = TRUE))
y_u = log(max(rowMeans(b_error_tsrga), rowMeans(b_error_rga), rowMeans(b_error_hydra),
              rowMeans(b_error_hydra2), rowMeans(b_error_hydra_5), rowMeans(b_error_hydra2_5),
              mean(b_error_lasso),
              na.rm = TRUE))
plot(x = 1:n_iter, y = log(rowMeans(b_error_tsrga))[1:n_iter], xlab = "iterations", 
     ylab = "beta error (log)", type = "l", lwd = 2,
     ylim = c(y_l, y_u))
     #, main = paste0("n=", n, "; p=", p, "; spec: ", spec))
lines(x = 1:n_iter, y = log(rowMeans(b_error_rga))[1:n_iter], col = "red", 
      type = "l", lwd = 2)
lines(x = 1:n_iter, y = log(rowMeans(b_error_rga_reg))[1:n_iter], col = "red", 
      type = "l", lwd = 2, lty = 2)
lines(x = 1:n_iter, y = log(rowMeans(b_error_hydra))[1:n_iter], col = "blue", 
      type = "l", lwd = 2)
lines(x = 1:n_iter, y = log(rowMeans(b_error_hydra_5))[1:n_iter], col = "blue", 
      type = "l", lwd = 2, lty = 2)
lines(x = 1:n_iter, y = log(rowMeans(b_error_hydra2))[1:n_iter], col = "green", 
      type = "l", lwd = 2)
lines(x = 1:n_iter, y = log(rowMeans(b_error_hydra2_5))[1:n_iter], col = "green", 
      type = "l", lwd = 2, lty = 2)
abline(h = mean(b_error_lasso), col = "gray", lty = 2, lwd = 1.5)
if (n == 1500) {
  legend("topright",
         legend = c("Hydra (25%)", "Hydra2 (25%)", "Hydra (50%)", "Hydra2 (50%)", 
                    "Lasso", "RGA (L=500)", "RGA (oracle L)", "TSRGA"),
         col = c(rep(c("blue", "green"), 2), "gray", "red", "red", "black"),
         lty = c(1, 1, 2, 2, 2, 1, 2, 1), lwd = c(rep(2, 4), 1.5, 2, 2, 2),
         bg = "white")
}
dev.off()

pdf(paste0("n", n, "_p", p, "_spec", spec,"_training_error.pdf"), width = 12, height = 10)
y_l = log(min(rowMeans(y_error_tsrga), rowMeans(y_error_rga), rowMeans(y_error_hydra),
              rowMeans(y_error_hydra2), rowMeans(y_error_hydra_5), rowMeans(y_error_hydra2_5),
              mean(y_error_lasso),
              na.rm = TRUE))
y_u = log(max(rowMeans(y_error_tsrga), rowMeans(y_error_rga), rowMeans(y_error_hydra),
              rowMeans(y_error_hydra2), rowMeans(y_error_hydra_5), rowMeans(y_error_hydra2_5),
              mean(y_error_lasso),
              na.rm = TRUE))
plot(x = 1:n_iter, y = log(rowMeans(y_error_tsrga))[1:n_iter], xlab = "iterations", 
     ylab = "training error (log)", type = "l", lwd = 2, ylim = c(y_l, y_u))
     #,main = paste0("n=", n, "; p=", p, "; spec: ", spec))
lines(x = 1:n_iter, y = log(rowMeans(y_error_rga))[1:n_iter], col = "red", 
      type = "l", lwd = 2)
lines(x = 1:n_iter, y = log(rowMeans(y_error_rga_reg))[1:n_iter], col = "red", 
      type = "l", lwd = 2, lty = 2)
lines(x = 1:n_iter, y = log(rowMeans(y_error_hydra))[1:n_iter], col = "blue", 
      type = "l", lwd = 2)
lines(x = 1:n_iter, y = log(rowMeans(y_error_hydra_5))[1:n_iter], col = "blue", 
      type = "l", lwd = 2, lty = 2)
lines(x = 1:n_iter, y = log(rowMeans(y_error_hydra2))[1:n_iter], col = "green", 
      type = "l", lwd = 2)
lines(x = 1:n_iter, y = log(rowMeans(y_error_hydra2_5))[1:n_iter], col = "green", 
      type = "l", lwd = 2, lty = 2)
abline(h = mean(y_error_lasso), col = "gray", lty = 2, lwd = 1.5)
# txt = paste0("n=", n, "; p=", p, "; spec: ", spec)
# mtext(txt, line = -2, outer = TRUE, cex = 2)
dev.off()

pdf(paste0("n", n, "_p", p, "_spec", spec,"_loss.pdf"), width = 8, height = 8)
par(mfrow = c(1, 1))
plot(x = 1:nrow(loss_hydra), y = log(rowMeans(loss_hydra)), xlab = "iterations",
     ylab = "loss (log)", type = "l", col = "blue", lwd = 2,
     ylim = c(min(log(rowMeans(loss_hydra)), log(rowMeans(loss_hydra2)), log(mean(loss_lasso))),
              max(log(rowMeans(loss_hydra)), log(rowMeans(loss_hydra2)), log(mean(loss_lasso)))))
    #,main = paste0("n=", n, "; p=", p, "; spec: ", spec))
lines(x = 1:nrow(loss_hydra_5), y = log(rowMeans(loss_hydra_5)), col = "blue", lwd = 2, lty = 2)
lines(x = 1:nrow(loss_hydra2), y = log(rowMeans(loss_hydra2)), col = "green", lwd = 2)
lines(x = 1:nrow(loss_hydra2_5), y = log(rowMeans(loss_hydra2_5)), col = "green", lwd = 2, lty = 2)
abline(h = log(mean(loss_lasso)), col = "gray", lty = 2, lwd = 1.5)
# txt = paste0("n=", n, "; p=", p, "; spec: ", spec)
# mtext(txt, line = -2, outer = TRUE, cex = 2)
dev.off()
