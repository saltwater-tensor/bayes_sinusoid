import config as cfg
import os
import numpy as np
import functions_RJMCMC_Segment as rjmcmc
import scipy as sc
import threading
import illustrative_example as ill
import scipy.stats as scstats
import math
# global a, data, b, time_star, y, y_star, n, omega, sigma


def generate_labels(pi_z, pi_init, T):

    n_states = np.size(pi_z, 1) #number of states
    a = np.array(range(n_states))
    labels = np.zeros((T, ))

    for t in range(T):
        print(T)
        if t == 0:
            labels[t] = int(np.random.choice(a, p=pi_init)) # choose between elements of a weighed by pi_init
        else:
            p_conditioned = pi_z[int(labels[t-1]), :]
            labels[t] = int(np.random.choice(a, p=p_conditioned))

    return labels


def generate_data(z, beta, omega, sigma):

    T = len(z)
    change_point_info = get_info_changepoints(z, T)
    locations_CP = change_point_info["locations"]
    n_CP = change_point_info["n_CP"]
    start_point = 0
    locations_CP.append(T)
    locations_CP.insert(0, start_point)
    temp = np.array(locations_CP)

    signal = []
    noise = []
    count = 0

    for j in range(len(temp)-1):
        if (j==0):
            a = temp[j]
            b = temp[j+1]
        else:
            a = temp[j] + 1
            b = temp[j+1]

        # State or regime
        regime = int(z[temp[j] + 1])

        X = rjmcmc.get_X(omega[regime], a, b)       #
        f = np.matmul(X, np.transpose(beta[regime]))
        num_samples = np.size(X, axis=0)
        epsilon = np.random.normal(0, sigma[regime], num_samples)

        f = f.tolist()
        epsilon = epsilon.tolist()
        signal = signal + f
        noise = noise + epsilon
    signal = np.array(signal[0:len(signal)-1])
    noise = np.array(noise[0:len(noise)-1])
    data = signal + noise
    out = {"data": data, "signal": signal}

    return out


def get_info_changepoints(z, T):

    n_CP = 0
    locations = []

    for t in range(T-1):
        if (z[t] != z[t+1]):
            n_CP = n_CP + 1
            locations.append(t)

    output = {"n_CP": n_CP, "locations": locations}

    return output


def get_likelik_system(z, m, omega, beta, sigma, pi_z, pi_init):

    tot = 0.0
    threads = []
    results = []
    for ii in range(0, ill.T):

        # Arguments to be sent
        omega_send = omega[int(z[ii]), 0:int(m[int(z[ii])])]
        beta_aux_send = beta[int(z[ii]), 0:int(2*m[int(z[ii])])]
        sigma_send = sigma[int(z[ii])]
        data_send = ill.data[ii]

        if ii == 0:
            pi_init_send = pi_init[int(z[ii])]
            pi_z_send = []
        else:
            pi_init_send = []
            pi_z_send = pi_z[int(z[ii - 1]), int(z[ii])]

        z_send = int(z[ii])

        tot = tot + threaded_get_likelik_system(data_send, z_send, omega_send, beta_aux_send, sigma_send, pi_z_send,
                                         pi_init_send, ii, results)

    return tot
"""""
        process = threading.Thread(target=threaded_get_likelik_system,
                                   args=[data_send, z_send, omega_send, beta_aux_send, sigma_send, pi_z_send,
                                         pi_init_send, ii, results])
        process.start()
        threads.append(process)

    total_value = np.array(results)
    total_value = np.sum(total_value)

    return total_value
"""""

def threaded_get_likelik_system(data, z, omega, beta, sigma, pi_z, pi_init, ii, result):

    X = rjmcmc.get_X_star(omega, ii)
    mu = np.matmul(X, beta)[0]

    likelik = scstats.norm.pdf(data, loc=mu, scale=sigma)

    if ii == 0:
        transition_prob = pi_init
    else:
        transition_prob = pi_z

    #result[ii] = math.log(transition_prob) + math.log(likelik)
    print(math.log(transition_prob) + math.log(likelik))
    tot = (math.log(transition_prob) + math.log(likelik))

    return tot