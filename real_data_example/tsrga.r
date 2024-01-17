### Last update: 2022-12-08

library(parallel)
library(MASS)
library(Matrix)

tr = function(X) {
  if (!is.matrix(X)) {
    stop("tr: X should be a matrix.")
  }
  if (dim(X)[1] != dim(X)[2]) {
    stop("tr: X should be a square matrix.")
  }
  return(sum(diag(X)))
}

# matlist computes sum of product of matrices from the two lists of matrices, mlst1 and mlst2.

matlist_prod = function(indices, mlst1, mlst2) {
  mat_prod = function(idx, mlst1, mlst2) {
    if (is.null(mlst2[[idx]]) || is.null(mlst1[[idx]])) {
      return(0)
    } else {
      return(mlst1[[idx]] %*% mlst2[[idx]])
    }
  }
  
  res = lapply(indices, mat_prod, mlst1, mlst2)
  
  return(Reduce('+', res))
}

comp_ip = function(u, X, u_dim, x_dim, L) {
  if (u_dim == 1) {
    u = as.vector(u)
  }
  if (x_dim == 1) {
    X = as.vector(X)
  }
  u_x = t(u) %*% X
  
  if (u_dim == 1 && x_dim == 1) {
    if (length(u_x) > 1) {
      stop(paste0("comp_ip: u_x should be scalar; length(u_x) = ", length(u_x)))
    }
  } else if (u_dim == 1 && x_dim > 1) {
    if (length(u_x) != x_dim) {
      stop(paste0("comp_ip: u_x should be of length ", x_dim, "; length(u_x) = ", 
                  length(u_x)))
    }
  } else if (u_dim > 1 && x_dim == 1) {
    if (length(u_x) != u_dim) {
      stop(paste0("comp_ip: u_x should be of length ", u_dim, "; length(u_x) = ",
                  length(u_x)))
    }
  } else {
    if (!is.matrix(u_x)) {
      stop("comp_ip: u_x shuould be a matrix.")
    }
    if (dim(u_x)[1] != u_dim || dim(u_x)[2] != x_dim) {
      stop(paste0("comp_ip: u_x incorrect dimensions: ", dim(u_x)[1], ", ", 
                  dim(u_x)[2]))
    }
  }
  
  if (u_dim == 1 && x_dim == 1) {
    return(list("IP" = L * abs(u_x), "B" = L * sign(u_x)))
  } else if (u_dim == 1 && x_dim > 1) {
    u_x_norm = sqrt(sum(u_x^2))
    return(list("IP" = L * u_x_norm, "B" = L * t(u_x) / u_x_norm))
  } else if (u_dim > 1 && x_dim == 1) {
    u_x_norm = sqrt(sum(u_x^2))
    return(list("IP" = L * u_x_norm, "B" = L * t(u_x) / u_x_norm))
  } else {
    dc = svd(u_x, nu = 1, nv = 1)
    return(list("IP" = L * dc$d[1], "B" = L * outer(as.vector(dc$v), 
                                                    as.vector(dc$u))))
  }
}

comp_lambda = function(u, X, B, G) {
  XB = X %*% B
  C = XB - G
  
  num = tr(t(u) %*% C)
  denum = sum(C^2)
  lambda_uc = num / denum
  
  return(max(min(lambda_uc, 1), 0))
}

data_mean = function(data_dim, dta) {
  if (data_dim == 1) {
    return(mean(dta))
  } else {
    return(colMeans(dta))
  }
}

data_2norm = function(data_dim, dta) {
  if (data_dim == 1) {
    return(sqrt(sum(dta^2)))
  } else {
    dc = svd(dta, nu = 1, nv = 1)
    return(dc$d[1])
  }
}

rga_core = function(y, X, dims, L, Kn, B_init = NULL, mc_cores = 1, 
                    parallel = FALSE, verbose = FALSE) {
  ###
  # rga_core: relaxed greedy algorithm core function
  # Inputs:
  # y: an n by d matrix of the response variables, where d = number of tasks.
  # X: a list of covariate observation matrices. X should have p elements, each 
  #    of which is a matrix or a vector representing the measurements.
  # dims: a (p + 1)-vector of integers indicating the number of tasks and the dimensions
  #       of each of the predictors
  # L: user-prescribed parameter
  # Kn: maximal number of iterations.
  # w_init: initial values for the coefficient matrices; default is zero.
  # mc_cores: number of cores available for parallel computing
  # parallel: whether to perform parallel computing for the p variables.
  #
  # Outputs:
  # B: a list of coefficient matrices. Zero matrix is represented by NULL.
  # J_hat: a vector of indices corresponding to the selected variables.
  # path: a vector of indices representing the selected variables in each step.
  # G: matrix of fitted values, of the same dimensions as y.
  # loss: a sequence of training losses.
  # lambda_seq: lambda's along the algorithm's path.
  ###
  
  p = length(dims) - 1
  if (dims[1] == 1) {
    n = length(y)
  } else {
    n = dim(y)[1]
  }
  
  if (is.null(B_init)) {
    B = vector(mode = "list", length = p)
    G = 0
  } else {
    B = B_init
    G = matlist_prod(1:p, X, B_init)
  }
  
  u = y - G
  loss = rep(NA, Kn)
  lambda_seq = rep(NA, Kn)
  J_hat = NULL
  
  rga_search = function(j) {
    return(comp_ip(u, X[[j]], u_dim = dims[1], x_dim = dims[j + 1], L = L)$IP)
  }
  
  for (i in 1:Kn) {
    if (parallel) {
      ips = unlist(mclapply(1:p, rga_search, mc.cores = mc_cores))
    } else {
      ips = unlist(lapply(1:p, rga_search))
    }
    j_star = which.max(ips)
    J_hat = c(J_hat, j_star)
    B_tilde = comp_ip(u, X[[j_star]], u_dim = dims[1], x_dim = dims[j_star + 1],
                      L = L)$B
    lambda = comp_lambda(u, X[[j_star]], B_tilde, G)
    lambda_seq[i] = lambda
    
    for (j in 1:p) {
      if (j == j_star) {
        if (is.null(B[[j]])) {
          B[[j]] = lambda * B_tilde
        } else {
          B[[j]] = (1 - lambda) * B[[j]] + lambda * B_tilde
        }
      } else {
        if (!is.null(B[[j]])) {
          B[[j]] = (1 - lambda) * B[[j]]
        }
      }
    }
    
    G = (1 - lambda) * G + lambda * (X[[j_star]] %*% B_tilde)
    u = y - G
    loss[i] = mean(u^2)
    
    if (verbose) {
      cat("iteration", i, "\n")
    }
  }
  
  return(list("B" = B, "J_hat" = unique(J_hat), "path" = J_hat,
              "G" = G, "loss" = loss, "lambda" = lambda_seq))
}

rga = function(y, X, dims, L, Kn, B_init = NULL, mc_cores = 1, 
               parallel = FALSE, scale = TRUE, verbose = FALSE) {
  ###
  # rga: wrapper to implement RGA
  ###
  
  # Pre-processing
  p = length(dims) - 1
  y_means = data_mean(data_dim = dims[1], dta = y)
  x_means = vector(mode = "list", length = p)
  x_2norms = rep(NA, p)
  
  if (dims[1] == 1) {
    y = y - y_means
  } else {
    y = t(t(y) - y_means)
  }
  
  for (i in 1:p) {
    x_means[[i]] = data_mean(data_dim = dims[i + 1], dta = X[[i]])
    if (dims[i + 1] > 1) {
      X[[i]] = t(t(X[[i]]) - x_means[[i]])
    } else {
      X[[i]] = X[[i]] - x_means[[i]]
    }
    if (scale) {
      x_2norms[i] = data_2norm(data_dim = dims[i + 1], dta = X[[i]])
      X[[i]] = X[[i]] / x_2norms[i]
    } else {
      x_2norms[i] = 1
    }
  }
  
  # Run RGA using core function
  model = rga_core(y = y, X = X, dims = dims, L = L, Kn = Kn, B_init = B_init,
                   mc_cores = mc_cores, parallel = parallel, verbose = verbose)
  
  # Calculate intercept term
  if (dims[1] > 1) {
    fitted_val = t(t(model$G) + y_means)
  } else {
    fitted_val = model$G + y_means
  }
  
  return(list("B" = model$B, "J_hat" = model$J_hat, 
              "path" = model$path, "fitted_values" = fitted_val, 
              "loss" = model$loss, "y_means" = y_means, "x_means" = x_means,
              "x_2norms" = x_2norms))
}

rga_jit_core = function(y, X, dims, L, Kn, t_n, B_init = NULL, mc_cores = 1,
                        parallel = FALSE, verbose = FALSE) {
  p = length(dims) - 1
  if (dims[1] == 1) {
    n = length(y)
  } else {
    n = dim(y)[1]
  }
  
  if (is.null(B_init)) {
    B = vector(mode = "list", length = p)
    G = 0
  } else {
    B = B_init
    G = matlist_prod(1:p, X, B)
  }
  
  u = y - G
  loss = rep(NA, Kn)
  lambda_seq = rep(NA, Kn)
  J_hat = NULL
  ranks = rep(0, p)
  
  rga_search = function(j) {
    return(comp_ip(u, X[[j]], u_dim = dims[1], x_dim = dims[j + 1], L = L)$IP)
  }
  
  for (i in 1:Kn) {
    if (parallel) {
      ips = unlist(mclapply(1:p, rga_search, mc.cores = mc_cores))
    } else {
      ips = unlist(lapply(1:p, rga_search))
    }
    j_star = which.max(ips)
    J_hat = c(J_hat, j_star)
    B_tilde = comp_ip(u, X[[j_star]], u_dim = dims[1], x_dim = dims[j_star + 1],
                      L = L)$B
    lambda = comp_lambda(u, X[[j_star]], B_tilde, G)
    lambda_seq[i] = lambda
    
    for (j in 1:p) {
      if (j == j_star) {
        if (is.null(B[[j]])) {
          B[[j]] = lambda * B_tilde
        } else {
          B[[j]] = (1 - lambda) * B[[j]] + lambda * B_tilde
        }
      } else {
        if (!is.null(B[[j]])) {
          B[[j]] = (1 - lambda) * B[[j]]
        }
      }
    }
    
    G = (1 - lambda) * G + lambda * (X[[j_star]] %*% B_tilde)
    u = y - G
    loss[i] = mean(u^2)
    
    if (verbose) {
      cat("iteration", i)
    }
    
    if ((i > 1) && ((loss[i] / loss[i - 1]) > 1 - t_n)) {
      if (verbose) {
        cat("... early stopping criterion met; abort first-stage RGA.", "\n")
      }
      break
    } else {
      if (verbose) {
        cat("\n")
      }
    }
  }
  
  for (j in 1:p) {
    if (is.null(B[[j]])) {
      ranks[j] = 0
    } else if (dims[1] == 1) {
      ranks[j] = 1
    } else if (dims[j + 1] == 1) {
      ranks[j] = 1
    } else {
      ranks[j] = rankMatrix(B[[j]])[1]
    }
  }
  
  return(list("B" = B, "J_hat" = unique(J_hat), "path" = J_hat, "ranks" = ranks,
              "G" = G, "loss" = loss, "lambda" = lambda_seq))
}

rga_jit = function(y, X, dims, L, Kn, t_n, B_init = NULL, mc_cores = 1, 
                   parallel = FALSE, scale = TRUE, demean = TRUE, 
                   verbose = FALSE) {
  ###
  # rga_jit : wrapper to implement RGA with the just-in-time stopping criterion.
  ###
  
  p = length(dims) - 1
  if (demean) {
    y_means = data_mean(data_dim = dims[1], dta = y)
  } else {
    y_means = 0
  }
  
  if (dims[1] == 1) {
    y = y - y_means
  } else {
    y = t(t(y) - y_means)
  }
  
  x_means = vector(mode = "list", length = p)
  x_2norms = rep(NA, p)
  for (i in 1:p) {
    if (demean) {
      x_means[[i]] = data_mean(data_dim = dims[i + 1], dta = X[[i]])
      if (dims[i + 1] > 1) {
        X[[i]] = t(t(X[[i]]) - x_means[[i]])
      } else {
        X[[i]] = X[[i]] - x_means[[i]]
      }
    } else {
      x_means[[i]] = 0
    }
    if (scale) {
      x_2norms[i] = data_2norm(data_dim = dims[i + 1], dta = X[[i]])
      X[[i]] = X[[i]] / x_2norms[i]
    } else {
      x_2norms[i] = 1
    }
  }
  
  model = rga_jit_core(y = y, X = X, dims = dims, L = L, Kn = Kn, t_n = t_n, 
                       B_init = B_init, mc_cores = mc_cores, 
                       parallel = parallel, verbose = verbose)
  
  if (dims[1] > 1) {
    fitted_val = t(t(model$G) + y_means)
  } else {
    fitted_val = model$G + y_means
  }
  
  return(list("B" = model$B, "J_hat" = model$J_hat, "path" = model$path, 
              "ranks" = model$ranks, "fitted_values" = fitted_val, 
              "loss" = model$loss, "lambda" = model$lambda, "y_means" = y_means,
              "x_means" = x_means, "x_2norms" = x_2norms))
}

second_stage_RGA_core = function(y, X, ranks, dims, L, Kn, mc_cores = 1, 
                                 parallel = FALSE, verbose = FALSE) {
  ###
  # Inputs:
  # ranks: a p-vector with the prescribed ranks for each B_j.
  ###
  p = length(dims) - 1
  d = dims[1]
  n = ifelse(d == 1, length(y), dim(y)[1])
  r_hat = sum(ranks)
  
  B = vector(mode = "list", length = p)
  G = 0
  
  u = y - G
  loss = rep(NA, Kn)
  lambda_seq = rep(NA, Kn)
  J_hat = NULL
  
  Us = vector(mode = "list", length = p)
  Vs = vector(mode = "list", length = p)
  sigma_inv = vector(mode = "list", length = p)
  new_xs = vector(mode = "list", length = p)
  
  for (i in 1:p) {
    sigma_inv[[i]] = ginv(t(X[[i]]) %*% X[[i]] / n)
    if (r_hat < min(d, dims[i + 1])) {
      dcmp = svd(t(X[[i]]) %*% y, nu = r_hat, nv = r_hat)
      Us[[i]] = dcmp$u
      Vs[[i]] = dcmp$v
      new_xs[[i]] = X[[i]] %*% sigma_inv[[i]] %*% Us[[i]]
    }
  }
  
  rga_search = function(j) {
    if (is.null(Us[[j]])) {
      return(comp_ip(u, X[[j]], u_dim = dims[1], x_dim = dims[j + 1], L = L)$IP)
    } else {
      new_u = u %*% Vs[[j]]
      return(comp_ip(new_u, new_xs[[j]], u_dim = r_hat, x_dim = r_hat, L = L)$IP)
    }
  }
  
  for (i in 1:Kn) {
    if (parallel) {
      ips = unlist(mclapply(1:p, rga_search, mc.cores = mc_cores))
    } else {
      ips = unlist(lapply(1:p, rga_search))
    }
    
    j_star = which.max(ips)
    J_hat = c(J_hat, j_star)
    if (is.null(Us[[j_star]])) {
      B_tilde = sigma_inv[[j_star]] %*% 
        comp_ip(u, X[[j_star]] %*% sigma_inv[[j_star]], u_dim = dims[1], 
                x_dim = dims[j_star + 1], L = L)$B
      lambda = comp_lambda(u, X[[j_star]], B_tilde, G)
    } else {
      new_u = u %*% Vs[[j_star]]
      new_x = X[[j_star]] %*% sigma_inv[[j_star]] %*% Us[[j_star]]
      B_tilde = sigma_inv[[j_star]] %*% Us[[j_star]] %*%
        comp_ip(new_u, new_x, u_dim = r_hat, x_dim = r_hat, L = L)$B %*%
        t(Vs[[j_star]])
      if ((dim(B_tilde)[1] != dims[j_star + 1]) || (dim(B_tilde)[2] != d)) {
        stop(paste("second_stage_RGA_core: iteration", i, 
                   ": incorrect B_tilde dimensions.", "Should be", 
                   dims[j_star + 1], "by",  d, "but get",
                   dim(B_tilde)[1], "by", dim(B_tilde)[2]))
      }
      lambda = comp_lambda(u, X[[j_star]], B_tilde, G)
    }
    lambda_seq[i] = lambda
    
    for (j in 1:p) {
      if (j == j_star) {
        if (is.null(B[[j]])) {
          B[[j]] = lambda * B_tilde
        } else {
          B[[j]] = (1 - lambda) * B[[j]] + lambda * B_tilde
        }
      } else {
        if (!is.null(B[[j]])) {
          B[[j]] = (1 - lambda) * B[[j]]
        }
      }
    }
    
    G = (1 - lambda) * G + lambda * (X[[j_star]] %*% B_tilde)
    u = y - G
    loss[i] = mean(u^2)
    
    if (verbose) {
      cat("iteration", i, "selected:", j_star, "\n")
    }
  }
  
  return(list("B" = B, "J_hat" = unique(J_hat), "path" = J_hat,
              "G" = G, "loss" = loss, "lambda" = lambda_seq,
              "Us" = Us, "Vs" = Vs))
}

second_stage_RGA = function(y, X, ranks, dims, L, Kn, mc_cores = 1, 
                            parallel = FALSE, scale = TRUE, demean = TRUE,
                            verbose = FALSE) {
  ###
  # second_stage_RGA: wrapper for second_stage RGA
  ###
  
  p = length(dims) - 1
  if (demean) {
    y_means = data_mean(data_dim = dims[1], dta = y)
  } else {
    y_means = 0
  }
  
  if (dims[1] == 1) {
    y = y - y_means
  } else {
    y = t(t(y) - y_means)
  }
  
  x_means = vector(mode = "list", length = p)
  x_2norms = rep(NA, p)
  for (i in 1:p) {
    if (demean) {
      x_means[[i]] = data_mean(data_dim = dims[i + 1], dta = X[[i]])
      if (dims[i + 1] > 1) {
        X[[i]] = t(t(X[[i]]) - x_means[[i]])
      } else {
        X[[i]] = X[[i]] - x_means[[i]]
      }
    } else {
      x_means[[i]] = 0
    }
    if (scale) {
      x_2norms[i] = data_2norm(data_dim = dims[i + 1], dta = X[[i]])
      X[[i]] = X[[i]] / x_2norms[i]
    } else {
      x_2norms[i] = 1
    }
  }
  
  model = second_stage_RGA_core(y = y, X = X, ranks = ranks, dims = dims, L = L,
                                Kn = Kn, mc_cores = mc_cores, 
                                parallel = parallel, verbose = verbose)
  
  if (dims[1] > 1) {
    fitted_val = t(t(model$G) + y_means)
  } else {
    fitted_val = model$G + y_means
  }
  
  return(list("B" = model$B, "J_hat" = model$J_hat, "path" = model$path, 
              "fitted_values" = fitted_val, "loss" = model$loss, 
              "lambda" = model$lambda, "y_means" = y_means, "x_means" = x_means,
              "x_2norms" = x_2norms, "Us" = model$Us, "Vs" = model$Vs))
}

tsrga = function(y, X, dims, L, t_n, Kn1, Kn2, mc_cores = 1, parallel = FALSE,
                 scale = TRUE, demean = TRUE, verbose = FALSE) {
  res1 = rga_jit(y = y, X = X, dims = dims, L = L, Kn = Kn1, t_n = t_n,
                 mc_cores = mc_cores, parallel = parallel, scale = scale,
                 demean = demean, verbose = verbose)
  
  chosen_subset = sort(res1$J_hat)
  chosen_X = X[chosen_subset]
  chosen_dims = dims[c(1, 1 + chosen_subset)]
  chosen_ranks = res1$ranks[chosen_subset]
  
  res2 = second_stage_RGA(y = y, X = chosen_X, ranks = chosen_ranks, 
                          dims = chosen_dims, L = L, Kn = Kn2, 
                          mc_cores = mc_cores, parallel = parallel, 
                          scale = scale, demean = demean, verbose = verbose)
  
  return(list("J_hat" = chosen_subset, "B" = res2$B, 
              "fitted_values" = res2$fitted_values, "loss1" = res1$loss,
              "loss2" = res2$loss, "x_means" = res1$x_means,
              "x_2norms" = res1$x_2norms, "y_means" = res1$y_means,
              "ranks" = chosen_ranks, 
              # "Us" = res2$Us, "Vs" = res2$Vs,
              "path1" = res1$path, "path2" = res2$path,
              "lambda1" = res1$lambda, "lambda2" = res2$lambda, "dims" = dims))
}

tsrga_fit = function(model, X, x_means, y_means, x_2norms) {
  J_hat = model$J_hat
  dims = model$dims
  for (i in 1:length(X)) {
    X[[i]] = t(t(X[[i]]) - x_means[[i]]) / x_2norms[i]
  }
  X = X[J_hat]
  pred = matlist_prod(1:length(J_hat), X, model$B)
  
  if (dims[1] > 1) {
    pred = t(t(pred) + y_means)
  } else {
    pred = pred + y_means
  }
  return(pred)
}

var_tsrga = function(y, features, L, t_n, Kn1, Kn2, dims, partial_out_dims, 
                     mc_cores = 1, parallel = FALSE, scale = TRUE, 
                     verbose = FALSE) {
  # partial out
  new_features = vector(mode = "list", length = length(features))
  Z = features[[1]][,partial_out_dims]
  
  y_means = data_mean(dims[1], y)
  z_means = data_mean(length(partial_out_dims), Z)
  if (dims[1] > 1) {
    y = t(t(y) - y_means)
  } else {
    y = y - y_means
  }
  
  if (length(partial_out_dims) > 1) {
    Z = t(t(Z) - z_means)
  } else {
    Z = Z - z_means
  }
  
  Qz = ginv(t(Z) %*% Z) %*% t(Z)
  
  x_means = vector(mode = "list", length = length(features))
  x_2norms = rep(1, length(features))
  for (i in 1:length(features)) {
    features[[i]] = features[[i]][,-partial_out_dims]
    x_means[[i]] = data_mean(data_dim = dims[i + 1], features[[i]])
    if (dims[i + 1] > 1) {
      features[[i]] = t(t(features[[i]]) - x_means[[i]])
    } else {
      features[[i]] = features[[i]] - x_means[[i]]
    }
    if (scale) {
      x_2norms[i] = svd(features[[i]], 1, 1)$d[1]
      features[[i]] = features[[i]] / x_2norms[i]
    } else {
      x_2norms[i] = 1
    }
    new_features[[i]] = features[[i]] - Z %*% Qz %*% features[[i]] 
  }
  
  # feed into tsrga
  Y = y - Z %*% Qz %*% y
  
  model = tsrga(y = Y, X = new_features, L = L, t_n = t_n, Kn1 = Kn1, Kn2 = Kn2,
                dims = dims, mc_cores = mc_cores, parallel = parallel, 
                demean = FALSE, scale = FALSE, verbose = verbose)
  
  features = features[model$J_hat]
  XB = matlist_prod(1:length(model$J_hat), features, model$B)
  R = y - XB
  C = Qz %*% R
  
  fitted_vals = Z %*% C + XB
  fitted_vals = t(t(fitted_vals) + y_means)
  
  return(list("J_hat" = model$J_hat, "B_ylag" = C, "B" = model$B,
              "fitted_values" = fitted_vals, "loss1" = model$loss1, 
              "loss2" = model$loss2, "x_means" = x_means,
              "y_means" = y_means, "x_2norms" = x_2norms,
              "z_means" = z_means, "ranks" = model$ranks,
              "path1" = model$path1, "path2" = model$path2))
}

tsrga_eval = function(model, new_X, new_Y = NULL, y_dim, 
                      partial_out_dims = c(1:48), scale = TRUE) {
  Z = new_X[[1]][,partial_out_dims]
  for (i in 1:length(new_X)) {
    new_X[[i]] = new_X[[i]][,-partial_out_dims]
    new_X[[i]] = t(t(new_X[[i]]) - model$x_means[[i]])
    if (scale) {
      new_X[[i]] = new_X[[i]] / model$x_2norms[i]
    }
  }
  
  new_X = new_X[model$J_hat]
  pred = matlist_prod(1:length(model$J_hat), new_X, model$B)
  pred = pred + t(t(Z) - model$z_means) %*% model$B_ylag
  if (y_dim == 1) {
    pred = pred + model$y_means
  } else {
    pred = t(t(pred) + model$y_means)
  }
  
  if (is.null(new_Y)) {
    return(list("pred" = pred))
  } else {
    return(list("pred" = pred, "loss" = mean((new_Y - pred)^2))) 
  }
}

# the below function only works for this example; use with caution
constrained_var_tsrga = function(y, features, L, t_n, Kn1, Kn2, dims, partial_out_dims, 
                                 mc_cores = 1, parallel = FALSE, scale = TRUE, 
                                 verbose = FALSE) {
  # partial out
  new_features = vector(mode = "list", length = length(features))
  Z = features[[1]][,partial_out_dims]
  
  y_means = data_mean(dims[1], y)
  z_means = data_mean(length(partial_out_dims), Z)
  if (dims[1] > 1) {
    y = t(t(y) - y_means)
  } else {
    y = y - y_means
  }
  
  if (length(partial_out_dims) > 1) {
    Z = t(t(Z) - z_means)
  } else {
    Z = Z - z_means
  }
  
  Qz = ginv(t(Z) %*% Z) %*% t(Z)
  
  x_means = vector(mode = "list", length = length(features))
  x_2norms = rep(1, length(features))
  for (i in 1:length(features)) {
    features[[i]] = features[[i]][,-partial_out_dims]
    x_means[[i]] = data_mean(data_dim = dims[i + 1], features[[i]])
    if (dims[i + 1] > 1) {
      features[[i]] = t(t(features[[i]]) - x_means[[i]])
    } else {
      features[[i]] = features[[i]] - x_means[[i]]
    }
    if (scale) {
      x_2norms[i] = svd(features[[i]], 1, 1)$d[1]
      features[[i]] = features[[i]] / x_2norms[i]
    } else {
      x_2norms[i] = 1
    }
    new_features[[i]] = features[[i]] - Z %*% Qz %*% features[[i]] 
  }
  
  # feed into tsrga
  Y = y - Z %*% Qz %*% y
  
  model = tsrga(y = Y, X = new_features, L = L, t_n = t_n, Kn1 = Kn1, Kn2 = Kn2,
                dims = dims, mc_cores = mc_cores, parallel = parallel, 
                demean = FALSE, scale = FALSE, verbose = verbose)
  
  features = features[model$J_hat]
  XB = matlist_prod(1:length(model$J_hat), features, model$B)
  R = y - XB
  C = matrix(0 , nrow = 24, ncol = 24)
  for (d in 1:2) {
    for (j in 1:12) {
      temp_model = lm(R[,(d - 1) * 12 + j] ~ 0 + Z[,((d - 1) * 12 + 1):(d * 12)])
      C[(d - 1) * 12 + j,((d - 1) * 12 + 1):(d * 12)] = temp_model$coefficients
    }
  }
  
  fitted_vals = Z %*% C + XB
  fitted_vals = t(t(fitted_vals) + y_means)
  
  return(list("J_hat" = model$J_hat, "B_ylag" = C, "B" = model$B,
              "fitted_values" = fitted_vals, "loss1" = model$loss1, 
              "loss2" = model$loss2, "x_means" = x_means,
              "y_means" = y_means, "x_2norms" = x_2norms,
              "z_means" = z_means, "ranks" = model$ranks,
              "path1" = model$path1, "path2" = model$path2))
}

