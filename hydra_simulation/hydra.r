### Update: 2023-01-26
### Last update: 2024-01-18

hydra_core = function (y, X, ptt, lambda, step_size, tau, beta_current, M) {
  ###
  # lambda: tuning parameter for Lasso
  # tau: number of coordinates to update in each node
  # M: X'X matrix, computed outside this function
  ###
  U = y - X %*% beta_current
  
  c = length(ptt)
  coord = NULL
  for (i in 1:c) {
    coord_update = sample(1:length(ptt[[i]]), size = tau)
    temp = ptt[[i]][coord_update]
    coord = c(coord, temp)
  }
  f_prime_x = - t(U) %*% X[,coord]
  upp_bd = as.vector((lambda - f_prime_x) / (M[coord] * step_size))
  lwr_bd = as.vector((-lambda - f_prime_x) / (M[coord] * step_size))
  h = pmin(pmax(-beta_current[coord], lwr_bd), upp_bd)
  beta_current[coord] = beta_current[coord] + h

  return(beta_current)
}

hydra = function (y, X, ptt, lambda, step_size, tau, n_iter = 500,
                  beta_truth, x_new, y_new) {
  
  start_time = proc.time()[3]
  beta = rep(0, ncol(X))
  M = colSums(X^2) # t(X) %*% X
  loss = rep(0, n_iter)
  y_error = rep(0, n_iter)
  b_error = rep(0, n_iter)
  elapsed_time = rep(0, n_iter)
  rmse = rep(0, n_iter)
  
  for (i in 1:n_iter) {
    beta = hydra_core(y = y, X = X, ptt = ptt, lambda = lambda, 
                      step_size = step_size, tau = tau, beta_current = beta,
                      M = M)
    
    b_error[i] = sum((beta - beta_truth)^2)
    y_error[i] = mean((y - X %*% beta)^2)
    loss[i] = 0.5 * sum((y - X %*% beta)^2) + lambda * sum(abs(beta))
    
    rmse[i] = sqrt(mean( (y_new - (x_new %*% beta))^2 ))
    elapsed_time[i] = proc.time()[3] - start_time
  }
  
  return(list("beta" = beta, "b_error" = b_error, "y_error" = y_error, 
              "loss" = loss, "RMSE" = rmse, "elapsed_time" = elapsed_time))
}

hydra2_core = function (y, X, ptt, lambda, D_para, tau, beta_current, theta,
                        u, z) {
  s = length(ptt[[1]]) # assuming equal partition
  c = length(ptt)
  U = y - X %*% beta_current
  
  coord = NULL
  for (i in 1:c) {
    coord_update = sample(1:length(ptt[[i]]), size = tau)
    temp = ptt[[i]][coord_update]
    coord = c(coord, temp)
  }
  # coord = ptt[[i]][coord_update]
  f_prime = -t(U) %*% X[,coord]
  D = D_para[coord]
  zi = z[coord]
  t = -zi + sign(zi - ((f_prime * tau) / (s * theta * D))) * 
    pmax(0, abs(((f_prime * tau) / (s * theta * D))) - (tau * lambda) / (s * theta * D))
  z[coord] = z[coord] + t
  u[coord] = u[coord] - ((1 / theta^2) - (s / (tau * theta))) * t
    
  theta = 0.5 * (sqrt(theta^4 + 4 * theta^2) - theta^2)
  beta_current = (theta^2) * u + z

  return(list("theta" = theta, "beta" = beta_current, "u" = u, "z" = z))
}

hydra2 = function (y, X, ptt, lambda, tau, step_size_factor = 10, 
                   n_iter = 500, beta_truth, x_new, y_new) {
  
  start_time = proc.time()[3]
  D = colSums(X^2) * step_size_factor
  theta = tau / length(ptt[[1]])
  u = rep(0, ncol(X))
  z = rep(0, ncol(X))
  b_error = rep(0, n_iter)
  y_error = rep(0, n_iter)
  loss = rep(0, n_iter)
  beta = rep(0, ncol(X))
  elapsed_time = rep(0, n_iter)
  rmse = rep(0, n_iter)
  
  for (i in 1:n_iter) {
    temp = hydra2_core(y = y, X = X, ptt = ptt, lambda = lambda, D_para = D,
                       tau = tau, beta_current = beta, theta = theta, u = u,
                       z = z)
    theta = temp$theta
    beta = temp$beta
    u = temp$u
    z = temp$z
    
    b_error[i] = sum((beta - beta_truth)^2)
    y_error[i] = mean((y - X %*% beta)^2)
    loss[i] = 0.5 * sum((y - X %*% beta)^2) + lambda * sum(abs(beta))
    
    rmse[i] = sqrt(mean( (y_new - (x_new %*% beta))^2 ))
    elapsed_time[i] = proc.time()[3] - start_time
  }
  
  return(list("y_error" = y_error, "b_error" = b_error, "beta" = beta,
              "loss" = loss, "RMSE" = rmse, "elapsed_time" = elapsed_time))
} 

rga_core = function (y, X, L, Kn, B_init = NULL, beta_truth, x_new, y_new) {
  start_time = proc.time()[3]
  p = ncol(X)
  if (is.null(B_init)) {
    B = rep(0, p)
    G = 0
  } else {
    B = B_init
    G = X %*% B
  }
  
  y_error = rep(NA, Kn)
  b_error = rep(NA, Kn)
  lambda_seq = rep(NA, Kn)
  J_hat = NULL
  elapsed_time = rep(NA, Kn)
  rmse = rep(NA, Kn)
  
  u = y - G
  for (i in 1:Kn) {
    ips = c(t(u) %*% X)
    j_star = which.max(abs(ips))
    J_hat = c(J_hat, j_star)
    B_tilde = L * sign(ips[j_star])
    lambda = t(u) %*% (X[,j_star] * B_tilde - G) / sum((X[,j_star] * B_tilde - G)^2)
    lambda = min(max(lambda, 0), 1)
    lambda_seq[i] = lambda
    
    B = (1 - lambda) * B
    B[j_star] = B[j_star] + lambda * B_tilde
    
    G = (1 - lambda) * G + lambda * (X[,j_star] * B_tilde)
    u = y - G
    
    y_error[i] = mean(u^2)
    b_error[i] = sum((B - beta_truth)^2)
    rmse[i] = sqrt(mean((y_new - x_new %*% B)^2))
    elapsed_time[i] = proc.time()[3] - start_time
  }
  
  return(list("y_error" = y_error, "b_error" = b_error, "J_hat" = J_hat,
              "B" = B, "RMSE" = rmse, "elapsed_time" = elapsed_time))
}

tsrga = function (y, X, L, Kn1, Kn2, t_n, beta_truth, x_new, y_new) {
  
  start_time = proc.time()[3]
  p = ncol(X)
  B = rep(0, p)
  
  y_error = rep(NA, Kn1 + Kn2)
  b_error = rep(NA, Kn1 + Kn2)
  J_hat = NULL
  elapsed_time = rep(NA, Kn1 + Kn2)
  rmse = rep(NA, Kn1 + Kn2)
  
  G = 0
  u = y - G
  for (i in 1:Kn1) {
    ips = c(t(u) %*% X)
    j_star = which.max(abs(ips))
    J_hat = c(J_hat, j_star)
    B_tilde = L * sign(ips[j_star])
    lambda = t(u) %*% (X[,j_star] * B_tilde - G) / sum((X[,j_star] * B_tilde - G)^2)
    lambda = min(max(lambda, 0), 1)

    B = (1 - lambda) * B
    B[j_star] = B[j_star] + lambda * B_tilde
    
    G = (1 - lambda) * G + lambda * (X[,j_star] * B_tilde)
    u = y - G
    
    y_error[i] = mean(u^2)
    b_error[i] = sum((B - beta_truth)^2)
    rmse[i] = sqrt(mean((y_new - x_new %*% B)^2))
    elapsed_time[i] = proc.time()[3] - start_time
    
    if (i > 1 && y_error[i] / y_error[i - 1] > 1 - t_n) {
      detach_i = i
      J_hat = sort(unique(J_hat))
      break
    } else if (i == Kn1) {
      detach_i = i
      J_hat = sort(unique(J_hat))
    }
  }
  
  X = X[,J_hat]
  B = B[J_hat]
  G = X %*% B
  u = y - G
  J_hat2 = NULL
  for (i in 1:Kn2) {
    ips = t(t(u) %*% X)
    j_star = which.max(abs(ips))
    J_hat2 = c(J_hat2, j_star)
    B_tilde = L * sign(ips[j_star])
    lambda = t(u) %*% (X[,j_star] * B_tilde - G) / sum((X[,j_star] * B_tilde - G)^2)
    lambda = min(max(lambda, 0), 1)
    
    B = (1 - lambda) * B
    B[j_star] = B[j_star] + lambda * B_tilde
    
    G = (1 - lambda) * G + lambda * (X[,j_star] * B_tilde)
    u = y - G
    
    y_error[i + detach_i] = mean(u^2)
    b_error[i + detach_i] = sum((B - beta_truth[J_hat])^2) + sum((beta_truth[-J_hat])^2)
    rmse[i + detach_i] = sqrt(mean((y_new - x_new[,J_hat] %*% B)^2))
    elapsed_time[i + detach_i] = proc.time()[3] - start_time
  }
  
  B_res = rep(0, p)
  B_res[J_hat] = B
  
  return(list("y_error" = y_error, "b_error" = b_error, "J_hat" = J_hat, 
              "B" = B_res, "detach" = detach_i, "RMSE" = rmse, "elapsed_time" = elapsed_time))
}

