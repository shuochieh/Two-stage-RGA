from mpi4py import MPI
import numpy as np
from sklearn.utils.extmath import randomized_svd
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Simulation parameters
#spec = 0

#if spec == 0:
    # used to prototype
#    n = 500
#    d = 100
#    q = 10
#    p = 100
#    a = 3
#    r = 2
#if spec == 1:
#    n = 20000
#    q = 80
#    p = 400
#    d = 100
#    r = 5
#    a = 10

# num_pred_per_node = int(p / size) # should be an integer
# if rank == 0:
#     print(f"Estimated predictor size: {num_pred_per_node * q * n * 8 / 10**9} GB\n")

# nonzero = np.empty(a)
# if rank == 0:
#     nonzero = np.random.choice(p, size = a, replace = False).astype("float64")
# comm.Bcast(nonzero)

# #### Generate data
# nu = np.empty(n)
# if rank == 0:
#     nu = np.random.normal(size = n)
# comm.Bcast(nu, root = 0)

# nu = nu.reshape(n, 1)
# w = np.random.normal(size = n * q * num_pred_per_node).reshape(num_pred_per_node, n, q)
# x = w + nu

# def unit_vector_sampler(d):
#     res = np.random.randn(d)
#     res /= np.linalg.norm(res)
#     return res

# B = np.zeros((num_pred_per_node, q, d))
# for i in range(num_pred_per_node):
#     if (rank * num_pred_per_node + i) in nonzero:
#         for j in range(r):
#             B[i,:,:] += np.random.uniform(low = 7, high = 15) * np.outer(unit_vector_sampler(q), unit_vector_sampler(d))
# output = np.sum(np.matmul(x, B), axis = 0)
# output = output.flatten()

# print(f"Pod {rank}: B star norms are {np.linalg.norm(B, axis = (1, 2)) ** 2}\n")

# recv_output = None
# if rank == 0:
#     recv_output = np.empty((size, n * d))
# comm.Gather(output, recv_output, root = 0)
# if rank == 0:
#     recv_output = recv_output.reshape(size, n, d)
#     Y_tosend = np.sum(recv_output, axis = 0)
#     Y_tosend += np.random.standard_t(5, size = n * d).reshape(n, d)

# Y = np.empty(n * d)
# if rank == 0:
#     Y = Y_tosend.flatten()
# comm.Bcast(Y, root = 0)
# Y = Y.reshape(n , d)
####

### DGP
def dgp(n, d, q, p, a, r, spec = 0):
    """
    Used to generate synthetic data
    Output:
        x, Y, B, num_pred_per_node
    """
    assert p % size == 0, "dgp: p / size should be an integer"
    num_pred_per_node = int(p / size)

    if rank == 0:
        print(f"n = {n}, p = {p}, num_pred_per_node = {num_pred_per_node}, q = {q}, d = {d}")
        print(f"Estimated predictor size: {num_pred_per_node * q * n * 8 / 10**9} GB\n")

    # generate the index of relevant predictors
    nonzero = np.empty(a)
    if rank == 0:
        nonzero = np.random.choice(p, size = a, replace = False).astype("float64")
        print(f"Relevant predictors: {nonzero}")
    comm.Bcast(nonzero, root = 0)

    # generate predictor data
    if spec == 1:
        nu = np.empty(n * q)
        if rank == 0:
            nu = np.random.normal(size = n * q)
        comm.Bcast(nu, root = 0)

        nu = nu.reshape(n, q)
        w = np.random.normal(size = n * q * num_pred_per_node).reshape(num_pred_per_node, n, q)
        x = w + 2 * nu
    else:
        x = np.random.standard_t(df = 6, size = n * q * num_pred_per_node).reshape(num_pred_per_node, n, q)

    # helper function
    def unit_vector_sampler(num, d):
        #res = np.random.randn(d)
        #res /= np.linalg.norm(res)
        temp = np.random.randn(d, num)
        Q, _ = np.linalg.qr(temp)

        return Q

    # generate coefficient matrices
    B = np.zeros((num_pred_per_node, q, d))
    for i in range(num_pred_per_node):
        if (rank * num_pred_per_node + i) in nonzero:
            B[i] = np.linalg.multi_dot([unit_vector_sampler(r, q), np.diag(np.random.uniform(low = 7, high = 15, size = r)), unit_vector_sampler(r, d).transpose()])

            #for j in range(r):
            #    B[i] += np.random.uniform(low = 7, high = 15) * np.outer(unit_vector_sampler(q), unit_vector_sampler(d))

    # generate XB
    output = np.sum(np.matmul(x, B), axis = 0)
    output = output.flatten()

    recv_output = None
    if rank == 0:
        recv_output = np.empty((size, n * d))
    comm.Gather(output, recv_output, root = 0)
    
    # generate Y
    if rank == 0:
        recv_output = recv_output.reshape(size, n, d)
        Y_tosend = np.sum(recv_output, axis = 0)
        Y_tosend += np.random.standard_t(df = 5, size = n * d).reshape(n, d)

    Y = np.empty(n * d)
    if rank == 0:
        Y = Y_tosend.flatten()
    comm.Bcast(Y, root = 0)
    Y = Y.reshape(n ,d)

    return x, Y, B, num_pred_per_node, nonzero

#### Learning algorithm
def comp_ip(u, x, L):
    """
    u: a numpy array of residual matrix (n * d)
    x: a (numpy) array of predictor matrices (numpy arrays) (num_pred_per_node * n * q)
    output:
    a tuple (res, B_tilde, U2, V2)
        res: maximum value of inner product (scalar)
        which_max: which predictor (in the node) achieves maximum inner product (scalar)
        B_tilde: B_tilde candidate (q * d)
        U2: leading left singular vector of XB (q * 1)
        V2: leading right singular vector of XB (d * 1)
    """
    ip = np.einsum('jk,ikl->ijl', np.transpose(u), x) # an array of matrix products u'x (num_pred_per_node * d * q)
    which_max = None
    res = 0
    for i in range(ip.shape[0]):
        _, S, _ = randomized_svd(ip[i], n_components = 1, random_state = None, n_oversamples = 30)
        if S.item() > res:
            res = S.item()
            which_max = i
    U, _, Vt = randomized_svd(ip[which_max], n_components = 1, random_state = None, n_oversamples = 30)
    B_tilde = L * np.outer(np.transpose(Vt), U)
    X_star = x[which_max]
    XB = np.matmul(X_star, B_tilde)
    U2, S2, V2t = randomized_svd(XB, n_components = 1, random_state = None, n_oversamples = 30)
    U2 = S2 * U2

    return L * res, which_max, B_tilde, U2, np.transpose(V2t)    

def comp_lambda(u, x, B_tilde, G):
    """
    u: a numpy array of residual matrix (n * d)
    x: a numpy array of the selected predictor matrix (n * q)
    B_tilde: selected coefficient vertex matrix (q * d)
    G: current approximators (n * d)
    output:
    a scalar lambda:
        lambda in the RGA algorithm (scalar)
    """
    xB = np.matmul(x, B_tilde)
    C = xB - G
    num1 = np.trace(np.matmul(np.transpose(u), C))
    num2 = np.linalg.norm(C) ** 2
    lambda_uc = num1 / num2
    
    return max(min(lambda_uc, 1),0)

def rga_core(Y, x, Kn, L, B, num_pred_per_node, scale_factor = 1, verbose = False):

    """
    core RGA functionality
    output:
        B_hat: final coefficient matrix estimate
        total_loss: the trajectory of total estimation loss
        training_loss: the trajectory of training loss
        rank0_times: time elapsed at the end of each iterations
    """

    # each node initializes
    B_hat = np.zeros((num_pred_per_node, q, d))
    
    G = np.zeros((n, d))
    u = Y - G

    loss = np.empty(Kn)
    training_loss = np.empty(Kn)
    J_hat = np.zeros(Kn) - 1
    rank0_times = np.empty(Kn)

    # Iterations
    start_time = MPI.Wtime()
    for i in range(Kn):
        ip, which_max, B_tilde, U2, V2 = comp_ip(u, x, L = L)
        lambda_hat = comp_lambda(u, x[which_max], B_tilde, G)

        msg_to_master = np.concatenate(([ip], [lambda_hat], U2.flatten(), V2.flatten())) # msg to be sent (2 + n + d)

        # Master gathers ip and B_tilde from all nodes
        recv_search = None
        if rank == 0:
            recv_search = np.empty((size, 2 + n + d))
        comm.Gather(msg_to_master, recv_search, root = 0)

        
        # Master send winner information (rank, lambda, U and V) to workers
        winner_info = np.empty(2 + n + d)
        if rank == 0:
            winner_node = np.argmax(recv_search[:,0])
            winner_info = np.concatenate(([winner_node], recv_search[winner_node,1:]))
        comm.Bcast(winner_info, root = 0)

        # workers reconstruct G
        winner_node = winner_info[0]
        lambda_hat = winner_info[1]
        U_jk = winner_info[2:(n + 2)].reshape(n, 1)
        V_jk = winner_info[(n + 2):].reshape(d, 1)
        suv = np.outer(U_jk, V_jk)
        G = (1 - lambda_hat) * G + lambda_hat * suv

        # update coefficient matrices
        if rank == winner_node:
            B_hat = (1 - lambda_hat) * B_hat
            B_hat[which_max] += lambda_hat * B_tilde
            J_hat[i] = which_max + rank * num_pred_per_node
        else:
            B_hat = (1 - lambda_hat) * B_hat

        # update residuals and loss
        u = Y - G
        loss_temp = np.linalg.norm(B - B_hat / scale_factor, axis = (1, 2)) ** 2
        training_loss[i] = np.linalg.norm(u) ** 2
        #if verbose:
        #    print(f"Pod {rank}: loss length expected is {num_pred_per_node}, get {len(loss_temp)}")
        loss[i] = np.sum(loss_temp)
        # loss being sum of Frobenius norms squared
        
        if rank == 0:
            rank0_times[i] = MPI.Wtime() - start_time
        if verbose:
            print(f"Pod{rank}: RGA iteration {i}, {np.around(MPI.Wtime() - start_time, decimals = 4)} seconds")


    # End of training, report elapsed time
    elapsed_time = MPI.Wtime() - start_time
    elapsed_times = comm.gather(elapsed_time, root = 0)

    # Total loss
    total_loss = None
    if rank == 0:
        total_loss = np.empty(shape = (size, Kn))
        avg_time = sum(elapsed_times) / size
        print(f"Pod {rank}: average elapsed time is {avg_time} seconds")
    comm.Gather(loss, total_loss, root = 0)
    if verbose:
        print(f"Pod {rank}: selected indices are {J_hat}")
    if rank == 0:
        total_loss = np.sqrt(np.sum(total_loss, axis = 0))
        if verbose:
            print(f"Pod {rank}: loss trajectory is {np.around(total_loss, decimals = 2)}")
            print(f"Pod {rank}: training_loss trajectory is {np.around(training_loss, decimals = 4)}")
        return B_hat / scale_factor, total_loss, training_loss, rank0_times
    else: 
        return B_hat / scale_factor, None, None, None

def rga(Y, x, Kn, L, B, num_pred_per_node, verbose = False):
    """
    A wrapper for calling rga_core
    """
    y_means = np.mean(Y, axis = 0) # a vector of sample means of each target (d,)
    x_means = np.mean(x, axis = 1).reshape(num_pred_per_node, 1, q) # an array of sample means of each predictor (num_pred_per_node, q)

    yy = Y - y_means # still of shape (n, d)
    xx = x - x_means
    x_2norms = np.linalg.norm(xx, axis = (1, 2), ord = 2).reshape(num_pred_per_node, 1, 1) # an array of operator norms for each predictor (num_pred_per_node, 1, 1)
    xx = xx / x_2norms # still of shape (num_pred_per_node, n, q)

    B_hat, total_loss, training_loss, rank0_times = rga_core(yy, xx, Kn, L, B, num_pred_per_node, scale_factor = x_2norms, verbose = verbose)

    return B_hat, total_loss, training_loss, rank0_times, y_means, x_means, x_2norms

def rga_jit_core(Y, x, Kn, L, t_n, scale_factor, B, num_pred_per_node, verbose = False):
    """
    core RGA with the just-in-time stopping criterion functionality
    """
    # each node initializes
    B_hat = np.zeros((num_pred_per_node, q, d))
    
    G = np.zeros((n, d))
    u = Y - G

    loss = np.zeros(Kn) - 1
    training_loss = np.zeros(Kn) - 1
    selected_pred = None
    rank0_times = np.zeros(Kn) - 1

    # Iterations
    start_time = MPI.Wtime()
    for i in range(Kn):
        ip, which_max, B_tilde, U2, V2 = comp_ip(u, x, L = L)
        lambda_hat = comp_lambda(u, x[which_max], B_tilde, G)

        msg_to_master = np.concatenate(([ip], [lambda_hat], U2.flatten(), V2.flatten())) # msg to be sent (2 + n + d)

        # Master gathers ip and B_tilde from all nodes
        recv_search = None
        if rank == 0:
            recv_search = np.empty((size, 2 + n + d))
        comm.Gather(msg_to_master, recv_search, root = 0)

        
        # Master send winner information (rank, lambda, U and V) to workers
        winner_info = np.empty(2 + n + d)
        if rank == 0:
            winner_node = np.argmax(recv_search[:,0])
            winner_info = np.concatenate(([winner_node], recv_search[winner_node,1:]))
        comm.Bcast(winner_info, root = 0)

        # workers reconstruct G
        winner_node = winner_info[0]
        lambda_hat = winner_info[1]
        U_jk = winner_info[2:(n + 2)].reshape(n, 1)
        V_jk = winner_info[(n + 2):].reshape(d, 1)
        suv = np.outer(U_jk, V_jk)
        G = (1 - lambda_hat) * G + lambda_hat * suv

        # update coefficient matrices
        if rank == winner_node:
            B_hat = (1 - lambda_hat) * B_hat
            B_hat[which_max] += lambda_hat * B_tilde
            if selected_pred is None:
                selected_pred = []
            selected_pred.append(which_max)
        else:
            B_hat = (1 - lambda_hat) * B_hat

        # update residuals and loss
        u = Y - G
        loss_temp = np.linalg.norm(B - B_hat / scale_factor, axis = (1, 2)) ** 2
        training_loss[i] = np.mean(np.square(u)) # (np.linalg.norm(u) ** 2) / (n * d)
        #if verbose:
        #    print(f"Pod {rank}: loss length expected is {num_pred_per_node}, get {len(loss_temp)}")
        loss[i] = np.sum(loss_temp)
        # loss being sum of Frobenius norms squared
        
        if rank == 0:
            rank0_times[i] = MPI.Wtime() - start_time
            if verbose:
                print(f"RGA-jit iteration {i}: {np.around(rank0_times[i], decimals = 4)} seconds")
        if i > 0 and (training_loss[i] / training_loss[i - 1]) > 1 - t_n:
            if verbose and rank == 0:
                print(f"Pod {rank}: early-stopping triggered at iteration {i}")
            rank0_times = rank0_times[:(i + 1)]
            j_stop = i
            training_loss = training_loss[:(i + 1)]
            loss = loss[:(i + 1)]
            break
        if i == Kn - 1:
            j_stop = i

    # End of training, report elapsed time
    elapsed_time = MPI.Wtime() - start_time
    #if verbose:
    #    print(f"Pod {rank}: elapsed time is {elapsed_time} seconds")
    elapsed_times = comm.gather(elapsed_time, root = 0)

    # Total loss
    total_loss = None
    if rank == 0:
        total_loss = np.empty(shape = (size, j_stop + 1))
        avg_time = sum(elapsed_times) / size
        #if verbose:
        #    print(f"Pod {rank}: average elapsed time is {avg_time} seconds")
    comm.Gather(loss, total_loss, root = 0)
    if rank == 0:
        total_loss = np.sqrt(np.sum(total_loss, axis = 0))
        if verbose:
            print(f"Pod {rank}: loss trajectory is {np.around(total_loss, decimals = 2)}")
            print(f"Pod {rank}: training_loss trajectory is {np.around(training_loss, decimals = 4)}")
        if selected_pred is not None:
            return B_hat / scale_factor, list(set(selected_pred)), total_loss, training_loss, rank0_times
        else:
            return B_hat / scale_factor, None, total_loss, training_loss, rank0_times
    elif selected_pred is not None: 
        return B_hat / scale_factor, list(set(selected_pred)), None, None, None
    else:
        return B_hat / scale_factor, None, None, None, None

def rga_jit(Y, x, Kn, L, t_n, B, num_pred_per_node, verbose = False):
    """
    A wrapper to call RGA with the just-in-time stopping criterion
    """
    y_means = np.mean(Y, axis = 0) # a vector of sample means of each target (d,)
    x_means = np.mean(x, axis = 1).reshape(num_pred_per_node, 1, q) # an array of sample means of each predictor (num_pred_per_node, q)

    yy = Y - y_means # still of shape (n, d)
    xx = x - x_means
    x_2norms = np.linalg.norm(xx, axis = (1, 2), ord = 2).reshape(num_pred_per_node, 1, 1) # an array of operator norms for each predictor (num_pred_per_node, 1, 1)
    xx = xx / x_2norms # still of shape (num_pred_per_node, n, q)

    B_hat, selected_pred, total_loss, training_loss, rank0_times = rga_jit_core(yy, xx, Kn, L, t_n, x_2norms, B, num_pred_per_node, verbose)

    return B_hat, selected_pred, total_loss, training_loss, rank0_times

def second_stage_RGA_core(Y, x, L, Kn, rbar, selected_pred, scale_factor, B, num_pred_per_node, verbose = False):
    """
    core functionality of the second-stage RGA
    rbar: the maxium rank for each predictor
    selected_pred: a list of indices of selected predictors for each node; if no predictors in the node is relevant, it's None.
    """
    start_time = MPI.Wtime()

    # each node initializes
    B_hat = np.zeros((num_pred_per_node, q, d))
    
    G = np.zeros((n, d))
    u = Y - G

    loss = np.zeros(Kn) - 1
    training_loss = np.zeros(Kn)
    rank0_times = np.zeros(Kn) - 1

    # compute sigma_inv, Us_temp (len(selected_pred), rbar, n), 
    # Us (len(selected_pred), q, rbar), Vs ((len(selected_pred), d, rbar))
    if selected_pred is not None:
        sigma_inv = np.zeros((len(selected_pred), q, q))
        Us_temp = np.zeros((len(selected_pred), rbar, n))
        Us = np.zeros((len(selected_pred), q, rbar))
        Vs = np.zeros((len(selected_pred), d, rbar))
        for i in range(len(selected_pred)):
            idx = selected_pred[i]
            sigma_inv[i] = np.linalg.inv(np.matmul(np.transpose(x[idx]), x[idx]) / n)
            if rbar < np.amin([d, q]):
                A, _, Bt = randomized_svd(np.matmul(np.transpose(x[idx]), Y), n_components = rbar, random_state = None, n_oversamples = 30)
                Us_temp[i] = np.linalg.multi_dot([np.transpose(A), sigma_inv[i], np.transpose(x[idx])])
                Us[i] = A
                Vs[i] = np.transpose(Bt)


    # New search criterion
    def rga_search(selected_pred, rbar):
        if selected_pred is None:
            return -1, None, None
        res = 0
        which_max = None
        reduced_rank = False
        for i in range(len(selected_pred)):
            if rbar < np.amin([d, q]):
                Uhat, S, Vhatt = randomized_svd(np.linalg.multi_dot([Us_temp[i], u, Vs[i]]),
                                                n_components = 1,
                                                random_state = None,
                                                n_oversamples = 30)
                if S.item() > res:
                    reduced_rank = True
                    res = S.item()
                    Shat = L * np.outer(Uhat, Vhatt)
                    which_max = i
            else:
                idx = selected_pred[i]
                x_temp = np.matmul(x[idx], sigma_inv[i])
                ip = np.matmul(np.transpose(x_temp), u)
                Uhat, S, Vhatt = randomized_svd(ip, n_components = 1, random_state = None, n_oversamples = 30)
                if S.item() > res:
                    reduced_rank = False
                    res = S.item()
                    B_tilde = L * np.linalg.multi_dot([sigma_inv[i], np.outer(Uhat, Vhatt)])
                    which_max = i
        if reduced_rank:
            B_tilde = np.linalg.multi_dot([sigma_inv[which_max], Us[which_max], Shat, np.transpose(Vs[which_max])])

        return res, which_max, B_tilde            

    # iterations
    for i in range(Kn):
        if selected_pred is not None:
            ip, which_max, B_tilde = rga_search(selected_pred, rbar)
            XB = np.matmul(x[selected_pred[which_max]], B_tilde)
            uu, ss, vv = randomized_svd(XB, n_components = 1, random_state = None, n_oversamples = 30)
            uu = uu * ss.item()
            lambda_hat = comp_lambda(u, x[selected_pred[which_max]], B_tilde, G)
            msg_to_master = np.concatenate(([ip, lambda_hat], uu.flatten(), vv.flatten())) # (2 + n + d)
        else:
            msg_to_master = np.zeros(2 + n + d)
            msg_to_master[0] = -1
            
        # Master gathers messages
        recv_search = None
        if rank == 0:
            recv_search = np.empty((size, 2 + n + d))
        comm.Gather(msg_to_master, recv_search, root = 0)

        # Master compose winner information and broadcast
        winner_info = np.empty(2 + n + d)
        if rank == 0:
            winner_node = np.argmax(recv_search[:,0])
            #if verbose:
            #    print(f"Pod {rank}: received ip's are {recv_search[:,0]}")
            winner_info = np.concatenate(([winner_node], recv_search[winner_node,1:]))
        comm.Bcast(winner_info, root = 0)

        # Workers reconstruct G
        winner_node = winner_info[0]
        lambda_hat = winner_info[1]
        U_jk = winner_info[2:(n + 2)].reshape(n, 1)
        V_jk = winner_info[(n + 2):].reshape(d, 1)
        suv = np.outer(U_jk, V_jk)
        G = (1 - lambda_hat) * G + lambda_hat * suv

        # update coefficient matrices
        B_hat = (1 - lambda_hat) * B_hat
        if rank == winner_node:
            B_hat[selected_pred[which_max]] += lambda_hat * B_tilde

        # update residuals and loss
        u = Y - G
        loss_temp = np.linalg.norm(B - (B_hat / scale_factor), axis = (1, 2)) ** 2
        training_loss[i] = (np.linalg.norm(u) ** 2) / (n * d)
        loss[i] = np.sum(loss_temp)

        #if rank == 0 and verbose:
            #print(f"Pod {rank}: current training loss is {np.around(training_loss[i], decimals = 2)}")
        if rank == 0:
            rank0_times[i] = MPI.Wtime() - start_time
            #if verbose:
            #    print(f"2S-RGA iteration {i}: {np.around(rank0_times[i], decimals = 4)} seconds")

    # End of training; report elapsed time
    elapsed_time = MPI.Wtime() - start_time

    # Total loss
    total_loss = None
    if rank == 0:
        total_loss = np.empty(shape = (size, Kn))
    comm.Gather(loss, total_loss, root = 0)
    if rank == 0:
        total_loss = np.sqrt(np.sum(total_loss, axis = 0))
        if verbose:
            print(f"Pod {rank}: loss trajectory is {np.around(total_loss, decimals = 4)}")
            print(f"Pod {rank}: training loss trajectory is {np.around(training_loss, decimals = 4)}")
        return B_hat / scale_factor, total_loss, training_loss, rank0_times
    else:
        return B_hat / scale_factor, None, None, None

def second_stage_RGA(Y, x, Kn, L, rbar, selected_pred, B, num_pred_per_node, verbose = False):
    """
    A wrapper to call second stage RGA
    """
    y_means = np.mean(Y, axis = 0)
    x_means = np.mean(x, axis = 1).reshape(num_pred_per_node, 1, q)

    yy = Y - y_means
    xx = x - x_means
    x_2norms = np.linalg.norm(xx, axis = (1, 2), ord = 2).reshape(num_pred_per_node, 1, 1)
    xx = xx / x_2norms

    B_hat, total_loss, training_loss, rank0_times = second_stage_RGA_core(Y = yy, x = xx, L = L, Kn = Kn, 
                                                                          rbar = rbar, 
                                                                          selected_pred = selected_pred, 
                                                                          scale_factor = x_2norms,
                                                                          B = B,
                                                                          num_pred_per_node = num_pred_per_node,
                                                                          verbose = verbose)
    return B_hat, total_loss, training_loss, rank0_times, y_means, x_means, x_2norms

def tsrga(Y, x, t_n, L1, L2, Kn1, Kn2, B, num_pred_per_node, verbose = False):

    B_hat, selected_pred, total_loss, training_loss, rank0_times = rga_jit(Y, x, Kn1, L1, t_n, B, num_pred_per_node, verbose)

    if verbose:
        if selected_pred is None:
            print(f"Pod {rank}: selected predictors = None")
        else:
            print(f"Pod {rank}: selected predictors = {np.array(selected_pred) + num_pred_per_node * rank}")

    r = np.array([np.sum(np.linalg.matrix_rank(B_hat))], dtype = np.int32)
    
    rbar_temp = np.empty(size, dtype = np.int32)
    comm.Gather(r, rbar_temp, root = 0)

    rbar = np.zeros(1, dtype = np.int32)
    if rank == 0:
        rbar = np.sum(rbar_temp, dtype = np.int32)
    comm.Bcast(rbar, root = 0)
    rbar = int(rbar)

    B_hat, total_loss2, training_loss2, rank0_times2, y_means2, x_means2, x_2norms2 = second_stage_RGA(Y, x, Kn2, L2, rbar, selected_pred, B, num_pred_per_node, verbose)

    if rank == 0:
        return B_hat, total_loss, total_loss2, training_loss, training_loss2, rank0_times, rank0_times2
    else:
        return B_hat, None, None, None, None, None, None

### Main program

def main_core(n, d, q, p, a, r, L1, L2, Kn1, Kn2, Kn3, spec = 0, verbose = False):
    x, Y, B, num_pred_per_node, nonzero = dgp(n, d, q, p, a, r, spec)

    ### TSRGA training
    t_n = 1 / (2 * np.log(n))
    _, loss, loss2, t_loss, t_loss2, r0_times, r0_times2 = tsrga(Y, x, t_n, L1, L2, Kn1, Kn2, B, num_pred_per_node, verbose)

    if rank == 0:
        tsrga_loss = np.concatenate((loss, loss2))
        tsrga_time = np.concatenate((r0_times, r0_times[-1] + r0_times2))

    ### Oracle RGA training
    oracle = None
    for i in range(num_pred_per_node):
        if (rank * num_pred_per_node + i) in nonzero:
            if oracle is None:
                oracle = [i]
            else:
                oracle.append(i)

    _, rga_loss, _, rga_times, _, _, _ = second_stage_RGA(Y = Y, x = x, Kn = Kn3, L = np.amax([L1, L2]), 
                                                          rbar = np.amin([d,q]), selected_pred = oracle, 
                                                          B = B, num_pred_per_node = num_pred_per_node, 
                                                          verbose = verbose)

    #_, rga_loss, _, rga_times, _, _, _ = rga(Y, x, Kn3, np.amax([L1, L2]), B, num_pred_per_node, verbose)

    if rank == 0:
        return tsrga_loss, tsrga_time, rga_loss, rga_times
    else:
        return tuple([None] * 4)

def main(n_iter, n, d, q, p, a, r, L1, L2, Kn1, Kn2, Kn3, spec = 0, verbose = False):
    tsrga_res = np.zeros((n_iter, Kn1 + Kn2)) - 1
    rga_res = np.zeros((n_iter, Kn3)) 
    tsrga_times = np.zeros((n_iter, Kn1 + Kn2)) - 1
    rga_times = np.zeros((n_iter, Kn3))

    for zz in range(n_iter):
        tsrga_loss, tsrga_time, rga_loss, rga_time = main_core(n, d, q, p, a, r, L1, L2, Kn1, Kn2, Kn3, spec, verbose)
        if rank == 0:
            tsrga_res[zz,:len(tsrga_loss)] = tsrga_loss
            tsrga_times[zz,:len(tsrga_time)] = tsrga_time
            rga_res[zz,:] = rga_loss
            rga_times[zz,:] = rga_time
            print(f"\nPod {rank}: iteration {zz}\n")

    if rank == 0:
        return tsrga_res, tsrga_times, rga_res, rga_times
    else:
        return None, None, None, None

### Simulation
n = 20000
d = 100
q = 100
p = 1024
a = 4
r = 4

Kn1 = np.floor((np.amin([d, q]) * 0.8)).astype(np.int32)
Kn2 = np.floor(20 * np.log(n)).astype(np.int32)
Kn3 = 90

tsrga_res, tsrga_times, rga_res, rga_times = main(10, n, d, q, p, a, r, 1e5, 1e5, Kn1, Kn2, Kn3, spec = 1, verbose = True)

if rank == 0:
    np.savetxt("tsrga_res_" + str(size) + ".csv", tsrga_res, delimiter = ",")
    np.savetxt("tsrga_times_" + str(size) + ".csv", tsrga_times, delimiter = ",")
    np.savetxt("rga_res_" + str(size) + ".csv", rga_res, delimiter = ",")
    np.savetxt("rga_times_" + str(size) + ".csv", rga_times, delimiter = ",")

    # find the largest number that tsrga_res are nonnegative for all n_iter
    nonneg = np.prod(tsrga_res >= 0, axis = 0)
    for i in range(len(nonneg) - 1, -1, -1):
        if nonneg[i] > 0:
            ind = i + 1
            break

    temp1 = np.mean(tsrga_res[:,:ind], axis = 0)
    temp2 = np.mean(tsrga_times[:,:ind], axis = 0)

    fig, ax = plt.subplots()
    ax.plot(np.arange(Kn1 + Kn2)[:len(temp1)], np.log(temp1), '-o', label = "TSRGA")
    ax.axhline(y = np.log(np.mean(rga_res, axis = 0))[-1], color = "grey", linestyle = '--', label = "Oracle LS")
    #ax.plot(np.arange(Kn3), np.log(np.mean(rga_res, axis = 0)), label = "RGA")
    ax.legend()
    ax.set_xlabel("iterations")
    ax.set_ylabel("log estimation errors")
    ax.set_title("")
    fig.savefig("iteration_" + str(size) + ".pdf")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(temp2, np.log(temp1), '-o', label = "TSRGA")
    ax.axhline(y = np.log(np.mean(rga_res, axis = 0))[-1], color = "grey", linestyle = '--', label = "Oracle LS")
    #ax.plot(np.mean(rga_times, axis = 0), np.log(np.mean(rga_res, axis = 0)), label = "RGA")
    ax.legend()
    ax.set_xlabel("elapsed time (sec)")
    ax.set_ylabel("log estimation errors")
    ax.set_title("")
    fig.savefig("time_" + str(size) + ".pdf")
    plt.close(fig)


