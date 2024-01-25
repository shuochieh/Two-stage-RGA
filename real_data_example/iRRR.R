### Last update: 2024-01-20

library(MASS)
library(Matrix)

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
