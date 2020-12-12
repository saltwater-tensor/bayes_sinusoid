import config as cfg
import illustrative_example as ill
import functions_RJMCMC_Segment as rjmcmc
import numpy as np
import math
import copy
from scipy import signal as scsig
import scipy.optimize as optim
import scipy.special as special
import scipy.stats as scstats
global a, data, b, time_star, y, y_star, n, omega, sigma


data = ill.data
# Function: sample the mode z's i.e. number of times a state occurs, given the observations, transition distributions
#           and emission parameters/observation model


def sample_z(pi_z, pi_init, m , beta, omega, sigma):

    N = np.zeros([ill.Kz + 1, ill.Kz], dtype=np.int64)
    z = np.zeros([ill.T], dtype=np.int64)
    totSeq = np.zeros([ill.Kz], dtype=np.int64)
    indSeq = np.zeros([ill.T, ill.Kz], dtype=np.int64)
    classProb = np.zeros([ill.T, ill.Kz])

    # Compute the likelihood of each observation under each parameter theta
    likelihood = compute_likelihood(m, beta, omega, sigma)

    # Compute backward messages
    # given the likelihood of the observation under different states at different time points
    # to compute backward messages we would need pi_z distribution

    partial_marg = backwards_message_vec(likelihood, pi_z)

    for t in range(0, ill.T):

        # --- Sample z(t)
        if t == 0:
            Pz = np.multiply(pi_init, partial_marg[:, 0])
        else:
            Pz = np.multiply(pi_z[z[t-1], :], partial_marg[:, t])

        classProb[t, :] = Pz/sum(Pz)
        Pz = np.cumsum(Pz)

        z[t] = np.sum((Pz[len(Pz)-1] * np.random.rand()) > Pz)

        # -- Add state to counts matrix
        if t > 0:
            N[z[t-1], z[t]] = N[z[t-1], z[t]] + 1
        else:
            N[ill.Kz, z[t]] = N[ill.Kz, z[t]] + 1

        totSeq[z[t]] = totSeq[z[t]] + 1
        indSeq[totSeq[z[t]], z[t]] = copy.copy(t)

    output = {"stateSeq": z, "N": N, "totSeq": totSeq, "indSeq": indSeq, "classProb":classProb}

    return output


def sample_tables(N, alpha_plus_k, rho, beta_vec):

    # - This is the Dirichlet process sampling process
    # - Define alpha and k in terms of alpha+k and rho
    alpha = alpha_plus_k*(1-rho)
    k = alpha_plus_k*rho

    # Sample M, where M(i,j) = no. of tables in restaurant i served dish j:
    # α*repeat(β_vec', outer = [Kz, 1]) +  κ*eye(Kz); α*β_vec'
    beta_vec_transpose = np.transpose(beta_vec)
    beta_vec_transpose_matrix = np.array([beta_vec,]*ill.Kz)
    beta_vec_transpose_matrix = alpha * beta_vec_transpose_matrix
    beta_vec_transpose_matrix = beta_vec_transpose_matrix + k*np.eye(ill.Kz)
    beta_vec_transpose_matrix = np.vstack([beta_vec_transpose_matrix, alpha*beta_vec_transpose])
    M = randnumtable(beta_vec_transpose_matrix, N)
    M = np.reshape(M, [8, 7])
    # Sample barM (the table counts for the underlying restaurant), where
    # barM(i,j) = # tables in restaurant i that considered dish j:
    temp = sample_barM(M, beta_vec, rho)
    barM = temp["barM"]
    sum_w = temp["sum_w"]

    output = {"M": M, "barM": barM, "sum_w": sum_w}

    return output


# Function: sample number of tables that served dish k in restaurant j
#           for each k and j. It returns a (Kz+1)×(Kz) matrix


def randnumtable(alpha, numdata):

    numtable = np.zeros(np.size(numdata), dtype=np.int64)
    numdata = np.matrix.flatten(numdata)
    alpha = np.matrix.flatten(alpha)
    for ii in range(np.prod(numdata.shape)):
        if numdata[ii] > 0:
            comp_one = alpha[ii] + (np.arange(0, numdata[ii]-1))
            comp_one = np.ones([int(numdata[ii]-1)], dtype=np.float64) * alpha[ii]/comp_one
            comp_two = (np.random.rand(int(numdata[ii]-1)))
            numtable[ii] = 1 + np.sum(comp_two < comp_one)
        else:
            numtable[ii] = 1

    numtable[np.where(numdata == 0)] = 0

    return numtable


# Function sample barM and sum_w


def sample_barM(M, beta_vec, rho):

    barM = copy.copy(M)
    sum_w = np.zeros([ill.Kz], dtype=np.int64)

    for j in range(ill.Kz):
        p_rho = rho/(beta_vec[j] * (1-rho) + rho)
        sum_w[j] = np.random.binomial(n=M[j, j], p=p_rho)
        barM[j, j] = M[j, j] - sum_w[j]

    output = {"sum_w": sum_w, "barM": barM}
    return output


def compute_likelihood(m, beta, omega, sigma):

    log_likelihood = np.zeros([ill.Kz, ill.T])
    mu = np.zeros([ill.Kz, ill.T], dtype=np.float64)

    for j in range(0, ill.Kz):  # for all possible states
        # get the basis matrix for the given omegas
        X = rjmcmc.get_X(omega[j, range(0, int(m[j]))], 0, ill.T-1)

        # calculate the mean of the gaussian distirbution
        mu[j, :] = np.matmul(X, beta[j, 0:int(2*m[j])])

    for kz in range(0, ill.Kz):
        data_rmvmean = ill.data - mu[kz, :]
        u = (1/sigma[kz]) * data_rmvmean
        log_likelihood[kz, :] = -0.5*np.power(u, 2) - np.log(sigma[kz])
        nan_places = np.isnan(log_likelihood[kz, :])
        nan_indices = np.where(nan_places == True)
        if not np.size(nan_indices) == 0:
            log_likelihood[kz, nan_indices[0][0:len(nan_indices[0])]] = -1e100


    normalizer = np.array([np.max([log_likelihood[:, ii]]) for ii in range(0, ill.T)])
    log_likelihood = log_likelihood - normalizer
    likelihood = np.exp(log_likelihood)

    return likelihood

# Function: compute backward messages


def backwards_message_vec(likelihood, pi_z):

    bwds_mssgs = np.ones([ill.Kz, ill.T], dtype=np.float64)
    partial_marg = np.zeros([ill.Kz, ill.T])

    # Compute messages backward in time
    back_time = np.arange((ill.T-2), -1, -1)
    for t in range(len(back_time)):
        time_point = back_time[t]
        # Multiplying likelihood by incoming message
        # At the current time point basically weighing the likelihood by the weight of the message
        partial_marg[:, time_point+1] = np.multiply(likelihood[:, time_point+1], bwds_mssgs[:, time_point+1])

        # Integrate out z_t
        bwds_mssgs[:, time_point] = np.matmul(pi_z, bwds_mssgs[:, time_point+1])
        bwds_mssgs[:, time_point] = bwds_mssgs[:, time_point]/np.sum(bwds_mssgs[:, time_point])

    # Compute marginal for the first time point
    partial_marg[:, 0] = np.multiply(likelihood[:, 0], bwds_mssgs[:, 0])

    return partial_marg


def sample_hyperparams(N, M, barM, sum_w, alpha_plus_k, gamma, hyperHMMhyperparams):

    a_alpha = hyperHMMhyperparams["a_alpha"]
    b_alpha = hyperHMMhyperparams["b_alpha"]
    a_gamma = hyperHMMhyperparams["a_gamma"]
    b_gamma = hyperHMMhyperparams["b_gamma"]
    c = hyperHMMhyperparams["c"]
    d = hyperHMMhyperparams["d"]

    Nkdot = np.sum(N, 1)
    Mkdot = np.sum(M, 1)
    temp_var = np.sum(barM, 0)
    temp_where = np.where(temp_var > 0)
    barK = np.size(temp_where, 1)
    validindices = np.where(Nkdot > 0)

    # Resample concentration parameters -----
    if np.size(validindices, 1) == 0:
        alpha_plus_k = scstats.gamma.rvs(a=a_alpha, loc=(1/b_alpha))
        gamma = scstats.gamma.rvs(a=a_gamma, loc=(1/b_gamma))
    else:
        alpha_plus_k = gibbs_conc_param(alpha_plus_k, Nkdot[validindices], Mkdot[validindices], a_alpha, b_alpha, 50)

        gamma = gibbs_conc_param(gamma, np.sum(barM), barK, a_gamma, b_gamma, 50)

    # Resample self-transition proportion parameter
    # rand(Beta(c + sum(sum_w), d+(sum(sum(M))-sum(sum_w))))
    rho = np.random.beta((c + np.sum(sum_w)), d + (np.sum(np.sum(M))-np.sum(sum_w)))

    output = {"gamma": gamma, "alpha_plus_k": alpha_plus_k, "rho": rho}

    return output


def gibbs_conc_param(alpha, numdata, numclass, aa, bb, numiter):

    numgroup = np.size(numdata)
    totalclass = np.sum(numclass)
    if numgroup == 1:
        numdata = [numdata]

    for ii in range(0, numiter):

        # beta auxiliary variables
        xx = [np.random.beta((alpha+1), numdata[j]) for j in range(0, np.size(numdata))]
        xx = np.array(xx)
        # binomial auxiliary variable
        tA = alpha + numdata
        t_random = np.random.rand(numgroup)
        t_mply = np.multiply(t_random, tA)
        t_compare = t_mply < numdata

        zz = np.multiply(t_compare, 1)

        # gamma resampling of concentration parameter
        gammaa = aa + totalclass - sum(zz)
        gammab = bb - np.sum(np.log(xx))
        alpha = scstats.gamma.rvs(a=gammaa, loc=(1/gammab))
        alpha_final = copy.copy(alpha)

    return alpha_final


def sample_hyperparams_init(hyperHMMhyperparams):

    # Hyperparams for Gamma dist over α + κ, where transition distributions
    # π_j ∼ DP (α+κ, (α⋅β + κ δ(j)/(α + κ))), which is the same as
    # π_j ∼ DP (α+κ, (1-ρ)⋅β + ρ⋅δ(j))
    # The transition distribution is governed by a Dirichlet process

    a_alpha = hyperHMMhyperparams["a_alpha"]
    b_alpha = hyperHMMhyperparams["b_alpha"]

    # Hyperparams for gamma dist over γ, where average transition distribution
    # β ∼ stick(γ) # this is the beta of the DP and not the model

    a_gamma = hyperHMMhyperparams["a_gamma"]
    b_gamma = hyperHMMhyperparams["b_gamma"]

    # Hyperparams for Beta dist over ρ, where ρ relates α+κ
    # to α and κ individually.
    c = hyperHMMhyperparams["c"]
    d = hyperHMMhyperparams["d"]

    # Resample concentration parameters, from the prior

    alpha_plus_k = a_alpha/b_alpha
    gamma = a_gamma/ b_gamma
    rho = c/(c+d)

    HMMhyperparams = {"gamma": gamma, "alpha_plus_k": alpha_plus_k, "rho": rho}

    return HMMhyperparams


def sample_dist(alpha_plus_k, gamma, rho, N, barM, Kz):


    # -- Define alpha and k in terms of (alpha + k) and rho -- #
    alpha = alpha_plus_k * (1-rho)
    k = alpha_plus_k*rho

    # -- Sample beta, the global menu, given new barM -- #
    barM = np.array(barM)
    temp1 = np.sum([barM], axis=1) + gamma/Kz
    temp1 = np.reshape([temp1], [Kz])
    beta_vec = np.random.dirichlet(temp1)

    if any(beta_vec == 0):

        loc_zeros = np.where(beta_vec == 0)
        beta_vec[loc_zeros] = np.power(1, -100)

    pi_z = np.zeros([Kz, Kz])

    for j in range(Kz):
        k_vec = np.zeros(Kz)
        # Add an amount κ to a Dirichlet parameter corresponding to a
        # self transition.
        k_vec = copy.copy(k_vec)

        # Sample π_j's given sampled β_vec and counts N, where
        # DP(α + κ, (α⋅β)/(α+κ)) is  Dirichelet distributed over the finite partition defined by β_vec
        pi_z[j, :] = np.random.dirichlet(alpha*beta_vec + k_vec + N[j, :])

    pi_init = np.random.dirichlet(alpha*beta_vec + N[Kz, :])

    output = {"pi_z": pi_z, "pi_init": pi_init, "beta_vec": beta_vec}

    return output


def get_starting_value_theta(z_start, n_freq_max_start):

    Seq_start = get_Seq(z_start, cfg.T, cfg.Kz)
    totSeq_start = Seq_start["totSeq"]
    indSeq_start = Seq_start["indSeq"]
    global a, b, time_star, y, y_star, n, omega, sigma
    for j in range(ill.Kz):
        if totSeq_start[j] > ill.n_min_obs:

            temp_ind = indSeq_start[0:totSeq_start[j], j]
            temp_ind_seg = get_time_indexes(temp_ind)
            # In the begining we will sample the long
            info_longest_ts = get_segment_ts(ill.data, temp_ind_seg, False)
            a = info_longest_ts["time"][0]
            b = info_longest_ts["time"][len(info_longest_ts["time"])-1]
            time_star = np.arange(a, b+1)
            y = info_longest_ts["data"]
            y_star = copy.copy(y)
            n = len(y)

            omega = find_significant_omega(y, n_freq_max_start, cfg.sigma_smoothing)
            ill.m_sample[j, 0] = len(omega)
            ill.omega_sample[j, np.arange(0, len(omega)), 0] = copy.copy(omega)
            sigma = 2

            beta = optim.minimize(rjmcmc.neg_f_posterior_beta_stationary,
                                  args=(omega, sigma, ill.sigma_beta, time_star, y_star),
                                  x0=np.zeros([2*int(ill.m_sample[j, 0])], dtype=np.int64), method='Newton-CG',
                                  jac=rjmcmc.neg_g_posterior_beta_stationary,
                                  hess=rjmcmc.neg_h_posterior_beta_stationary)

            ill.beta_sample[j, 0:int(2*ill.m_sample[j, 0]), 0] = copy.copy(beta.x)

            ill.sigma_sample[j, 0] = copy.copy(sigma)

        else:
            ill.m_sample[j, 0] = 1
            ill.beta_sample[j, 0:int(2*ill.m_sample[j, 0]), 0] = np.random.multivariate_normal(
                                                                                    np.zeros(int(2*ill.m_sample[j, 0])),
                                                                                    np.eye(int(2*ill.m_sample[j, 0])))
            ill.omega_sample[j, 0:int(ill.m_sample[j, 0]), 0] = np.random.uniform(0, ill.phi_omega)
            ill.sigma_sample[j, 0] = scstats.invgamma.rvs(a=1, loc=2)


def get_Seq(z, T, Kz):


    totSeq = np.zeros([Kz],dtype=np.int64)
    indSeq = np.zeros([T, Kz], dtype=np.int64)

    for t in range(T-1):
        # --- adding to the quantity of number of states detected
        totSeq[int(z[t])] = totSeq[int(z[t])] + 1
        indSeq[totSeq[int(z[t])], int(z[t])] = copy.copy(t)


    output = {"totSeq": totSeq, "indSeq": indSeq}

    return output


def get_time_indexes(indexes):

    # indexes are sent for the current state under evaluation
    #A = np.zeros([2], dtype=np.int64)
    A = list()
    global a
    for t in range(len(indexes)-1):
        if t == 0:
            a = indexes[0]
        else:
            if (indexes[t+1] != (indexes[t] + 1)):
                b = indexes[t]
                tb = [a, b]
                A.append(tb)
                a = indexes[t+1]

    tbep = [a, indexes[len(indexes)-1]]
    A.append(tbep)
    #A = A[:, 2:len(A)]

    return A

# Function: return time points and data w.p
# proportional to length of the time series (if ts_sample = true)
# otherwise it returns the longest time series.
# Does not sample a continous length but randomly samples from different time points


def get_segment_ts(data, A, ts_sample):

    #temp = [(len(A[0, i]:A[1, i]) for i in range(np.size([A], axis=1))]
    temp = np.zeros([np.size([A], axis=1)])

    for i in range(np.size([A], axis=1)):
        tup = A[i]
        low = tup[0]
        up = tup[1]
        if len(range(low, up)) == 0:
            temp[i] = 1
        else:
            temp[i] = len(range(low, up))

    temp_prob = temp/np.sum(temp)



    # - Sample time series in proportion to its length.

    if ts_sample:
        ind = np.random.choice(range(len(temp)), p=temp_prob)

    # - Sample longest time series
    else:
        ind = np.where(temp == np.max(temp))
        if len(ind) > 1:
            ind = np.random.choice(ind[0][:])
        if len(ind) == 1:
            ind = ind[0][0]

    #y_ind = np.reshape([A[:, ind], 2])
    y_ind = A[ind]

    output = {"data": data[y_ind[0]:y_ind[1]+1], "time": (np.arange(y_ind[0], y_ind[1]+1))}

    return output


def find_significant_omega(y,n_freq_max, sigma_smoothing ):

    omega_local = []

    y_demean = y - np.mean([y])
    n_local = len(y_demean)

    #f_period, power = scsig.periodogram([y_demean],
    #                                   window=scsig.windows.gaussian(n, sigma_smoothing), scaling="spectrum")

    #win_dow = scsig.gaussian(n_local, sigma_smoothing)
    f_period, power = scsig.periodogram([y_demean],
                                        window="boxcar", scaling="spectrum")

    I_smooth = power[0][1:np.size(power)]
    I_smooth = np.transpose(I_smooth)
    freq = f_period[range(1, (len(f_period)))]

    test = findpeaks_sorted(I_smooth, freq, ill.n_freq_max)
    I_test = test["power"]
    freq_test = test["freq"]

    a_local = math.floor((n_local-1)/2)

    for i in range(len(I_test)):

        g_test = I_test[i]/np.sum(I_smooth)
        b_local = math.floor(1/g_test)
        b_max = get_max_b(a_local, b_local)
        x = copy.copy(g_test)
        p_val = get_p_value_freq(x, a_local, b_max)

        if p_val <= 0.05 and p_val >=0:
            omega_local.append(freq_test[i])

    return omega_local[0:n_freq_max+1]


# Function: findpeaks of a 1D array, sorted in decreasing order (of power) both the power list and frequency list


def findpeaks_sorted(A, freq, n_freq_max):

    peaks = scsig.find_peaks(A)
    peaks_indices = peaks[0]
    peak_values = A[peaks_indices]
    freq_values = freq[peaks_indices]
    peaks_list = np.ndarray.tolist(peak_values)
    freq_list = np.ndarray.tolist(freq_values)
    peaks_list_sorted, freq_list_sorted = zip(*sorted(zip(peaks_list, freq_list), reverse=True))

    output = {"freq": freq_list_sorted, "power": peaks_list_sorted}

    return output


def get_max_b(a, b_local):

    b_max = 1
    try:
        for i in range((b_local)):
            comb_num = special.comb(a, i)

            b_max = b_max + 1

        return_val = (b_max - 1)

    except:

        return_val = b_max - 1

    return return_val


def get_p_value_freq(x_local, a_local, b_local):

    out = 0.0
    for i in range(b_local):
        out = out + math.pow(-1, i)*special.comb(a_local, i)*math.pow((1-i*x_local), (a_local-1))

    return out






