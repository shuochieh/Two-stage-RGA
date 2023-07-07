### Last update: 2023-02-16
### 
setwd("./iRRR_simulation")
library(MASS)
library(Matrix)
library(doParallel)
library(foreach)
source("tsrga.r")

unit_vec_sampler = function (num, d, family = "gaussian") {
  if (family == "gaussian") {
    temp = matrix(rnorm(num * d), ncol = num)
    Q = qr.Q(qr(temp))
  }
  
  return (Q)
}

B_step = function (y, X, rho, A, Lambda) {
  n = nrow(X)
  temp1 = t(X) %*% X / n 
  diag(temp1) = diag(temp1) + max(rho, 1e-3)
  
  ### Old codes using QR decomposition
  # qr_base = base::qr
  # qrQ_base = base::qr.Q
  # qrR_base = base::qr.R
  # qr_x = qr_base(temp1)
  # Q = qrQ_base(qr_x)
  # R = qrR_base(qr_x)
  # temp2 = t(X) %*% y / n + rho * A + Lambda
  # return(solve(R) %*% t(Q) %*% temp2)
  
  ### Using chol2inv
  temp3 = chol(temp1)
  inv = chol2inv(temp3)
  temp2 = t(X) %*% y / n + rho * A + Lambda
  
  return(inv %*% temp2)
}

SVST_core = function (index, B, Lambda, rho, lambda, ptt) {
  coords = ptt[[index]]
  temp = B[coords,] - (Lambda[coords,] / rho)
  model = svd(temp)
  d = pmax(model$d - lambda / rho, 0)
  return(model$u %*% diag(d) %*% t(model$v))
}

B_nuc = function (B, ptt) {
  res = 0
  for (i in 1:length(ptt)) {
    res = res + sum(svd(B, nu = 1, nv = 1)$d)
  }
  return(res)
}

iRRR = function (y, X, ptt, lambda, rho, A_init = NULL, Lambda_init = NULL, 
                 B_truth = NULL, max_iter = 1000, expedite_factor = 1.1, 
                 tol = 1e-2, verbose = FALSE, prototype = FALSE) {
  loss = rep(NA, max_iter)
  B_loss = rep(NA, max_iter)
  B_tilde = matrix(rnorm(ncol(X) * ncol(y)), nrow = ncol(X), ncol = ncol(y))
  if (is.null(A_init)) {
    A = matrix(rnorm(ncol(X) * ncol(y)), nrow = ncol(X), ncol = ncol(y))
  } else {
    A = A_init
  }
  if (is.null(Lambda_init)) {
    Lambda = matrix(rnorm(ncol(X) * ncol(y)), nrow = ncol(X), ncol = ncol(y))
  } else {
    Lambda = Lambda_init
  }
  
  SVST_step = function (j) {
    return(SVST_core(j, B = B, Lambda = Lambda, rho = rho, lambda = lambda, ptt = ptt))
  }
  
  early_stop = FALSE
  for (i in 1:max_iter) {
    
    if (prototype) {
      start_time = Sys.time()
    }
    B = B_step(y = y, X = X, rho = rho, A = A, Lambda = Lambda)
    if (prototype) {
      cat("      B step:", round(difftime(Sys.time(), start_time, units = "mins"), 4), "minutes", "\n")
    }
    
    if (prototype) {
      start_time = Sys.time()
    }
    temp = lapply(1:length(ptt), SVST_step)
    A = Reduce(rbind, temp)
    if (prototype) {
      cat("      SVST:", round(difftime(Sys.time(), start_time, units = "mins"), 4), "minutes", "\n")
    }
    # A = SVST_step(B = B, Lambda = Lambda, rho = rho, lambda = lambda, ptt = ptt)
    Lambda = Lambda + rho * (A - B)
    loss[i] = 0.5 * ncol(y) * mean((y - X %*% B)^2) + lambda * B_nuc(B = B, ptt = ptt)
    rho = rho * expedite_factor
    
    primal_residual = sqrt(sum((A - B)^2))
    dual_residual = rho * sqrt(sum((B - B_tilde)^2))
    if (verbose) {
      cat("    iRRR iteration", i, ": primal res", primal_residual, 
          "dual res", dual_residual, "rho", rho, "\n")
    }
    
    B_tilde = B
    
    if (!is.null(B_truth)) {
      B_loss[i] = sum(abs(B_truth - B))
    }
    
    if (primal_residual < tol && dual_residual < tol) {
      if (verbose) {
        cat("    Stopping criterion triggered", "\n")
      }
      loss = loss[1:i]
      B_loss = B_loss[1:i]
      early_stop = TRUE
      break
    }
  }
  
  return(list("B" = B, "loss" = loss, "B_loss" = B_loss, "A" = A, 
              "Lambda" = Lambda, "rho" = rho, "early_stop" = early_stop))
}

iRRR_validation = function (y, X, ptt, rho = NULL, y_val, X_val, 
                            lambdas = exp(seq(from = -5, to = 5, length.out = 20)),
                            max_iter = 1000,
                            expedite_factor = 1, 
                            tol = 1e-3,
                            B_truth = NULL, verbose = FALSE) {
  val_error = rep(NA, length(lambdas))
  models = vector(mode = "list", length = length(lambdas))
  for (i in 1:length(lambdas)) {
    if (verbose) {
      cat("  ---------", i, "-th lambda", "----------\n")
    }
    if (i > 1) {
      A_init = model$A
      Lambda_init = model$Lambda
      rho = model$rho * 0.5
    } else {
      A_init = NULL
      Lambda_init = NULL
      rho = 2 * lambdas[1]
    }
    model = iRRR(y = y, X = X, ptt = ptt, lambda = lambdas[i], rho = rho,
                 A_init = A_init, Lambda_init = Lambda_init, B_truth = B_truth, 
                 max_iter = max_iter, expedite_factor = expedite_factor, tol = tol,
                 verbose = verbose)
    val_error[i] = mean((y_val - X_val %*% model$B)^2)
    models[[i]] = model
  }
  opt_m = which.min(val_error)
  if (verbose) {
    cat("  Best lambda:", lambdas[opt_m], "\n")
  }
  return(models[[opt_m]])
}

### ---- #######################################################################

sim_call = function (n, d, q, p, s, r, spec, 
                     lambdas = seq(from = exp(-5), to = exp(5), length.out = 200),
                     expedite_factor = 1.01, n_val = 500, run.iRRR = TRUE,
                     verbose = FALSE) {
  X = matrix(rt(n * p * q, df = 6), ncol = p * q, nrow = n)
  X_val = matrix(rt(n_val * q * p, df = 6), ncol = p * q, nrow = n_val)
  if (spec == 2) {
    X = matrix(0, ncol = p * q, nrow = n)
    X_val = matrix(0, ncol = p * q, nrow = n_val)
    S = matrix(rnorm(n * q), ncol = q, nrow = n)
    S_val = matrix(rnorm(n_val * q), ncol = q, nrow = n_val)
    
    for (i in 1:p) {
      X[,((i - 1) * q + 1):(i * q)] = 2 * S + matrix(rnorm(n * q), ncol = q, nrow = n)
      X_val[,((i - 1) * q + 1):(i * q)] = 2 * S_val + matrix(rnorm(n_val * q), ncol = q, nrow = n_val)
    }
  }
  y = matrix(0, nrow = n, ncol = d)
  y_val = matrix(0, nrow = n_val, ncol = d)
  B_star = matrix(0, nrow = p * q, ncol = d)
  
  ptt = vector(mode = "list", length = p)
  for (i in 1:p) {
    ptt[[i]] = c(((i - 1) * q + 1):(i * q))
    if (i <= s) {
      if (spec == 1) {
        B_star[ptt[[i]],] = unit_vec_sampler(r, q) %*% diag(runif(r, min = 7, max = 15), nrow = r, ncol = r) %*% 
          t(unit_vec_sampler(r, d))
      } else if (spec == 2) {
        B_star[ptt[[i]],] = unit_vec_sampler(r, q) %*% diag(runif(r, min = 7, max = 15), nrow = r, ncol = r) %*%
          t(unit_vec_sampler(r, d))
      }
    }
    # for (j in 1:r) {
    #   if (i <= s) {
    #     if (spec == 1) {
    #       B_star[ptt[[i]],] = B_star[ptt[[i]],] +
    #         sample(c(1, -1), 1) * runif(1, min = 7, max = 15) * (unit_vec_sampler(q) %o% unit_vec_sampler(d))
    #     } else if (spec == 2) {
    #       B_star[ptt[[i]],] = B_star[ptt[[i]],] +
    #         runif(1, min = 3, max = 10) * (unit_vec_sampler(q) %o% unit_vec_sampler(d))
    #     }
    #   }
    # }
    y = y + X[,ptt[[i]]] %*% B_star[ptt[[i]],]
    y_val = y_val + X_val[,ptt[[i]]] %*% B_star[ptt[[i]],]
  }
  if (spec == 2) {
    y = y + matrix(rt(n * d, df = 5), ncol = d)
    y_val = y_val + matrix(rt(n_val * d, df = 5), ncol = d)
  } else {
    y = y + matrix(rt(n * d, df = 5), ncol = d) # matrix(rnorm(n * d), ncol = d)
    y_val = y_val + matrix(rt(n_val * d, df = 5), ncol = d) # matrix(rnorm(n_val * d), ncol = d)
  }
  
  t_n = 1 / (log(n)) #2 / (1 * d) # ifelse(spec == 1, 1 / (2 * log(n)), 1 / (n^(1/3)))
  tsrga_X = vector(mode = "list", length = p)
  dims = c(d, rep(NA, p))
  for (i in 1:p) {
    tsrga_X[[i]] = X[,c(((i - 1) * q + 1):(i * q))]
    dims[i + 1] = q
  }
  tsrga_res = tsrga(y = y, X = tsrga_X, dims = dims, L = 1e+5, Kn1 = min(d, q) - 2, 
                    Kn2 = 300, t_n = t_n)
  
  tsrga_loss = 0
  counter = 1
  for (i in 1:p) {
    if (i %in% tsrga_res$J_hat) {
      if (is.null(tsrga_res$B[[counter]])) {
        tsrga_loss = tsrga_loss + sum((B_star[c(((i - 1) * q + 1):(i * q)),])^2)
      } else {
        Bhat = tsrga_res$B[[counter]] / tsrga_res$x_2norms[i]
        tsrga_loss = tsrga_loss + sum((B_star[c(((i - 1) * q + 1):(i * q)),] - Bhat)^2)
      }
      counter = counter + 1
    } else {
      tsrga_loss = tsrga_loss + sum((abs(B_star[c(((i - 1) * q + 1):(i * q)),])^2))
    }
  }
  tsrga_loss = sqrt(tsrga_loss)
  
  if (verbose){
    cat("TSRGA J hat:", tsrga_res$J_hat, "\n")
    cat("TSRGA ranks:", tsrga_res$ranks, "\n")
  }
  
  ## Oracle Least squares
  J_star = c(1:s)
  B_hat = lm(y~0+X[,1:(s * q)])$coefficients
  LS_loss = sqrt(sum((B_hat - B_star[1:(s * q),])^2))
  
  ### iRRR
  if (run.iRRR) {
    time = proc.time()[3]
    iRRR_res = iRRR_validation(y = y, X = X, ptt = ptt, y_val = y_val, X_val = X_val,
                               max_iter = 400, lambdas = lambdas,
                               expedite_factor = expedite_factor, 
                               B_truth = B_star, tol = 1e-1,
                               verbose = verbose)
    if (verbose) {
      cat("  iRRR takes", round((proc.time()[3] - time) / 60, 2), "mins", "\n")
    }
    iRRR_ranks = rep(0, p)
    for (i in 1:p) {
      iRRR_ranks[i] = rankMatrix(iRRR_res$A[ptt[[i]],])
    }
    if (verbose) {
      cat("iRRR J hat:", which(iRRR_ranks != 0), "\n")
      cat("iRRR ranks:", iRRR_ranks[which(iRRR_ranks != 0)], "\n")
    }    
    return(list("tsrga_loss" = tsrga_loss, 
                "iRRR_loss" = sqrt(sum((B_star - iRRR_res$B)^2)),
                "tsrga_Jhat" = tsrga_res$J_hat, 
                "iRRR_Jhat" = which(iRRR_ranks != 0),
                "LS_loss" = LS_loss,
                "tsrga_rank" = sum(tsrga_res$ranks),
                "iRRR_early_stop" = iRRR_res$early_stop))
  }
  
  return(list("tsrga_loss" = tsrga_loss, "tsrga_Jhat" = tsrga_res$J_hat, "LS_loss" = LS_loss,
              "tsrga_rank" = sum(tsrga_res$ranks)))
}

#### Simulation
proto = TRUE
iter = 500
n = 200
d = 10
q = 12
p = 20
s = 1
r = 2
spec = 1
run.iRRR = TRUE

Nslots = 15
print(sprintf("%d processes were allocated", Nslots))
cl = makeCluster(Nslots)
registerDoParallel(cl)

start_time = Sys.time()
Result = foreach (i = 1:iter, .combine = rbind, .inorder = FALSE,
                  .packages = c("MASS", "Matrix")) %dopar% {
                    model = sim_call(n = n, d = d, q = q, p = p, s = s, r = r,
                                     spec = spec,
                                     expedite_factor = 1.03,
                                     run.iRRR = run.iRRR) 
                    sink()
                    cat("iteration", i, ":", round(difftime(Sys.time(), start_time, units = "hours"), 3), "hours", "\n")
                    cat("    Result:", round(c(model$tsrga_loss, model$iRRR_loss, model$LS_loss), 3), "\n")
                    cat("    TSRGA ranks:", model$tsrga_rank, " TSRGA Jhat:", model$tsrga_Jhat, "\n")
                    if (run.iRRR) {
                      cat("    iRRR early stop:", model$iRRR_early_stop, "\n")
                    }
                    c(model$tsrga_loss, model$iRRR_loss, model$LS_loss)
                  }

warnings()
print(warnings())
cat("total time:", round(difftime(Sys.time(), start_time, units = "hours"), 3), "hours", "\n")

print(dim(Result))
est_error = colMeans(Result)
cat("Average parameter estimation errors:\n")
print(round(est_error, 4))

cat("Median parameter estimation errors:\n")
print(round(apply(Result, 2, median), 4))

if (proto) {
  pdf(file = paste0("iRRR_", spec, "_n_", n, ".pdf"), width = 8)
  boxplot(Result, main=paste0("n=",n," d=",d," q=", q, " p=",p, " a=", s, " r=", r))
  dev.off()
}

write.csv(Result, file = paste0("iRRR_sim_", spec, "_n_", n, ".csv"))
stopCluster(cl)

quit(save = "no")
