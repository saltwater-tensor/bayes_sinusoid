import config as cfg
import illustrative_example as ill
import numpy as np
import math
import scipy as sc
import copy
from scipy.stats import poisson
from scipy.stats import multivariate_normal as mvnormal
from scipy.stats import uniform
from scipy import stats
from scipy import signal as scsig
import types

global a, data, b, time_star, y, y_star, n, omega, sigma, sigma_beta, omega_star, omega_prop


class CustomError(Exception):
    message = "DIMENSION MISMATCH OMEGA AND BETA"


def globalInit(y_s, time_s):
    global y_star, time_star, n
    y_star = y_s
    time_star = time_s
    n = len(y_star)


def RJMCMC_SegmentModelSearch(y_star,info_segment_ts, m_current, beta_current,
                              omega_current, sigma_current, time_star, lambda_var, c, phi_omega,
                              chi_omega, n_freq_max):

    globalInit(y_star, time_star)
    if(len(beta_current)) != (2*m_current) or (len(omega_current) != m_current):
        raise CustomError

    # If m == 1 then either birth or death move
    if m_current == 1:
        poisson_eval =  poisson.pmf(2, lambda_var)/poisson.pmf(1, lambda_var)
        birth_prob = c*min(1, poisson_eval)

        U = np.random.rand()

        if U <= birth_prob:

            MCMC = birth_move_stationary(info_segment_ts, m_current, beta_current, omega_current, sigma_current, time_star,
                                         lambda_var, c, phi_omega, chi_omega)
            m_out = m_current + int(MCMC["accepted"])
            beta_out = MCMC["beta"]
            omega_out = MCMC["omega"]
            sigma_out = MCMC["sigma"]

        else:
            MCMC = within_model_move_stationary(info_segment_ts, m_current, beta_current, omega_current, sigma_current,time_star
                                                , lambda_var, c, phi_omega, chi_omega)
            m_out = m_current
            beta_out = MCMC["beta"]
            omega_out = MCMC["omega"]
            sigma_out = MCMC["sigma"]

    elif m_current == ill.n_freq_max:
        poisson_eval = poisson.pmf((ill.n_freq_max-1), lambda_var)/poisson.pmf(ill.n_freq_max, lambda_var)
        death_prob = c*min(1, poisson_eval)

        U = np.random.rand()

        if U <= death_prob:

            MCMC = death_move_stationary(info_segment_ts, m_current, beta_current, omega_current,
                                sigma_current, time_star, lambda_var, c, phi_omega, chi_omega)

            m_out = m_current - int(MCMC["accepted"])
            beta_out = MCMC["beta"]
            omega_out = MCMC["omega"]
            sigma_out = MCMC["sigma"]
        else:
            MCMC = within_model_move_stationary(info_segment_ts, m_current, beta_current, omega_current, sigma_current,time_star
                                                , lambda_var, c, phi_omega, chi_omega)
            m_out = m_current
            beta_out = MCMC["beta"]
            omega_out = MCMC["omega"]
            sigma_out = MCMC["sigma"]

    else:
        poisson_eval = poisson.pmf(m_current + 1, lambda_var) / poisson.pmf(m_current, lambda_var)
        birth_prob = (c * min(1, poisson_eval))

        poisson_eval = poisson.pmf(m_current - 1, lambda_var) / poisson.pmf(m_current, lambda_var)
        death_prob = (c * min(1, poisson_eval))

        U = np.random.rand()

        # --- Birth
        if (U <= birth_prob):

            MCMC = birth_move_stationary(info_segment_ts, m_current, beta_current, omega_current, sigma_current, time_star,
                                         lambda_var, c, phi_omega, chi_omega)
            m_out = m_current + int(MCMC["accepted"])
            beta_out = MCMC["beta"]
            omega_out = MCMC["omega"]
            sigma_out = MCMC["sigma"]

        # --- Death
        elif ((U > birth_prob) and (U <= (birth_prob + death_prob))):

            MCMC = death_move_stationary(info_segment_ts, m_current, beta_current, omega_current,
                                sigma_current, time_star, lambda_var, c, phi_omega, chi_omega)

            m_out = m_current - int(MCMC["accepted"])
            beta_out = MCMC["beta"]
            omega_out = MCMC["omega"]
            sigma_out = MCMC["sigma"]

        # --- Within model
        else:

            MCMC = within_model_move_stationary(info_segment_ts, m_current, beta_current, omega_current, sigma_current,time_star
                                                , lambda_var, c, phi_omega, chi_omega)
            m_out = m_current
            beta_out = MCMC["beta"]
            omega_out = MCMC["omega"]
            sigma_out = MCMC["sigma"]

    output = {"m": m_out, "beta": beta_out, "omega": omega_out, "sigma": sigma_out}

    return output


def birth_move_stationary(info_segment_ts, m_current, beta_current, omega_current,
                                sigma_current, time_star, lambda_var, c, phi_omega, chi_omega):

    if 2*len(omega_current) != len(beta_current):
        raise CustomError

    global sigma
    sigma = sigma_current

    time_indexes = info_segment_ts["time"]

    m_proposed = m_current + 1

    # - Proposing a new omega (normalised frequency)
    omega_current_aux = np.hstack([0, omega_current, phi_omega])
    omega_current_aux = np.sort(omega_current_aux)
    support_omega = np.zeros([2*len(omega_current)+2])
    spom = 0
    for k in range(0, len(omega_current_aux)-1):
        support_omega[spom] = omega_current_aux[k] + chi_omega
        support_omega[spom+1] = omega_current_aux[k+1] - chi_omega
        spom = spom + 2

    length_support_omega = phi_omega - (2*(m_current + 1)*chi_omega)

    #omega_star = sample_uniform_continous_interval([1, support_omega])[0]
    omega_star = uniform.rvs(loc=support_omega[0], scale=support_omega[-1]-support_omega[0])

    omega_proposed = np.sort(np.hstack([omega_current, omega_star]))

    # - Proposing beta ~ Normal(beta_prop, sigma_prop)

    X_prop = get_X_star(omega_proposed, time_star)
    beta_var_prop = np.eye(int(2*m_proposed))/math.pow(ill.sigma_beta, 2) + \
                    np.matmul(np.transpose(X_prop), X_prop)/math.pow(sigma, 2)
    beta_var_prop = np.linalg.inv(beta_var_prop)
    beta_mean_prop = np.matmul(beta_var_prop, ((np.matmul(np.transpose(X_prop), y_star)) / math.pow(sigma, 2)))
    beta_proposed = np.random.multivariate_normal(mean=beta_mean_prop,
                                                  cov=0.5*(beta_var_prop + np.transpose(beta_var_prop)))

    # Obtain beta_current and sigma_current to evaluate the proposal ratio

    X_curr = get_X_star(omega_current, time_star)
    beta_var_curr = np.eye(int(2*m_current))/math.pow(ill.sigma_beta, 2) + \
                    np.matmul(np.transpose(X_curr), X_curr)/math.pow(sigma, 2)
    beta_var_curr = np.linalg.inv(beta_var_curr)
    beta_mean_curr = np.matmul(beta_var_curr, ((np.matmul(np.transpose(X_curr), y_star)) / math.pow(sigma, 2)))


    # Acceptance probability

    # log likelihood ratios
    log_likelik_prop = log_likelik(beta_proposed, omega_proposed, time_star)
    log_likelik_curr = log_likelik(beta_current, omega_current, time_star)
    log_likelik_ratio = log_likelik_prop - log_likelik_curr

    # Log prior ratio
    log_m_prior_ratio =  math.log(poisson.pmf(m_proposed, lambda_var)) - math.log(poisson.pmf(m_current, lambda_var))
    log_beta_prior_ratio = np.log(mvnormal.pdf(x=beta_proposed, mean=np.zeros(2*m_proposed),
                                        cov=math.pow(ill.sigma_beta, 2)*np.eye(2*m_proposed))) - np.log(
                                                            mvnormal.pdf(x=beta_current, mean=np.zeros(2*m_current),
                                                            cov=math.pow(ill.sigma_beta,2)*np.eye(2*m_current)))
    log_omega_prior_ratio = np.log(2)

    log_prior_ratio = log_m_prior_ratio + log_beta_prior_ratio + log_omega_prior_ratio

    # -- Log proposal ratio
    # (-0.5*(β_proposed - β_mean_prop)'*inv(β_var_prop)*(β_proposed - β_mean_prop) -
    #                             log(sqrt(det(2π*β_var_prop))))


    beta_intermediate1 = beta_proposed - beta_mean_prop
    beta_intermediate2 = np.transpose(beta_intermediate1)
    beta_intermediate3 = np.linalg.inv(beta_var_prop)
    beta_lsd = np.log(np.sqrt(np.linalg.det(2*np.pi*beta_var_prop)))
    beta_matmul = np.matmul(beta_intermediate2, beta_intermediate3, beta_intermediate1)
    beta_matmul = -0.5*beta_matmul
    log_proposal_beta_prop = beta_matmul - beta_lsd
    log_proposal_beta_prop = log_proposal_beta_prop[0]

    beta_intermediate1 = beta_current - beta_mean_curr
    beta_intermediate2 = np.transpose(beta_intermediate1)
    beta_intermediate3 = np.linalg.inv(beta_var_curr)
    beta_lsd = np.log(np.sqrt(np.linalg.det(2*np.pi*beta_var_curr)))
    beta_matmul = np.matmul(beta_intermediate2, beta_intermediate3, beta_intermediate1)
    beta_matmul = -0.5*beta_matmul
    log_proposal_beta_current = beta_matmul - beta_lsd
    log_proposal_beta_current = log_proposal_beta_current[0]

    log_proposal_omega_proposed = np.log((1/(length_support_omega)))
    log_proposal_omega_current = np.log((1/m_proposed))

    poisson_eval = poisson.pmf(m_proposed, lambda_var) / poisson.pmf(m_current, lambda_var)
    log_proposal_birth_move = np.log(c * min(1, poisson_eval))

    poisson_eval = poisson.pmf(m_current, lambda_var) / poisson.pmf(m_proposed, lambda_var)
    log_proposal_death_move = np.log(c * min(1, poisson_eval))

    log_proposal_ratio = log_proposal_death_move - log_proposal_birth_move + log_proposal_omega_current - \
                         log_proposal_omega_proposed + log_proposal_beta_current - log_proposal_beta_prop


    # - Metropolis Hastings acceptance step
    MH_ratio_birth = log_likelik_ratio + log_prior_ratio + log_proposal_ratio
    epsilon_birth = min(1, np.exp(MH_ratio_birth))

    U = np.random.rand()

    if U <= epsilon_birth:
        beta_out = beta_proposed
        omega_out = omega_proposed
        accepted = True
    else:
        beta_out = beta_current
        omega_out = omega_current
        accepted = False

    # - Updating sigma in a Gibbs step

    X_post = get_X_star(omega_out, time_star)
    residual_variance = y_star - np.matmul(X_post, beta_out)
    residual_variance = np.power(residual_variance, 2)
    res_var = np.sum(residual_variance)
    nu_post = (n + ill.nu0)/2
    gamma_post = (ill.gamm0 + res_var)/2

    invgamma = stats.invgamma.rvs(a=nu_post , loc=gamma_post)
    sigma_out = np.sqrt(invgamma)

    # -------- output
    omega_out = np.matrix.flatten(omega_out)
    output = {"beta": beta_out, "omega": omega_out, "sigma": sigma_out, "accepted": accepted, "omega_star": omega_star}

    return output

# Function: within move. Proposed frequencies via mixture of FFT sampling and RW
#           (one-at-time frequency updating)


def within_model_move_stationary(info_segment_ts, m_current, beta_current,
                                      omega_current, sigma_current, time_star, lambda_Var, c, phi_omega, chi_omega):
    if 2*len(omega_current) != len(beta_current):
        raise CustomError

    global sigma
    sigma = sigma_current

    time_indexes = copy.copy(info_segment_ts["time"])

    #  ------------- Sampling frequencies ------------------
    y = info_segment_ts["data"]
    y_detrended = scsig.detrend(y)
    f_period, power = scsig.periodogram([y_detrended],
                                        window="boxcar", scaling="spectrum")
    power = np.matrix.flatten(power)

    power_norm = power/np.sum(power)

    U = np.random.rand()

    # ----------- Gibbs step (FFT) ---------

    if U <= ill.alpha_mixing:

        omega_current_aux = copy.copy(omega_current)

        for j in range(0, m_current):

            omega_curr = copy.copy(omega_current)

            # Avoiding a vector with two same frequencies (D column would be linear dependent)
            aux_temp = False

            while(aux_temp == False):

                # Proposing frequencies
                global omega_star
                omega_star = np.random.choice(f_period, p=np.matrix.flatten(power_norm))

                global omega_prop
                omega_prop = copy.copy(omega_curr)

                # Updating jth component
                omega_prop[j] = omega_star

                stack = np.hstack([omega_prop[np.arange(0, j-1)], omega_prop[np.arange(j+1, (len(omega_prop)-1))]])

                if not np.any(stack == omega_star):
                    aux_temp = True

            log_likelik_ratio = log_posterior_omega_stationary(omega_prop, beta_current, time_star) - \
                                    log_posterior_omega_stationary(omega_curr, beta_current, time_star)

            power_norm_idx = np.searchsorted(f_period, omega_curr[j], side='right')

            if power_norm_idx == len(power_norm):
                power_norm_idx = power_norm_idx - 1

            log_proposal_ratio = np.log(power_norm[power_norm_idx]) - \
                                                np.log(power_norm[np.searchsorted(f_period, omega_star)])

            MH_ratio = np.exp(log_likelik_ratio + log_proposal_ratio)

            U = np.random.rand()

            if U <= min(1,MH_ratio):
                omega_current_aux = omega_prop
            else:
                omega_current_aux = omega_curr

            omega_out = np.sort(omega_current_aux)

        omega_out = np.sort(omega_current_aux)
    else:

        omega_current_aux = copy.copy(omega_current)

        for j in range(0, m_current):

            omega_curr = copy.copy(omega_current)
            aux_temp = False

            # We can control the frequency range between 0 and 1 1 being the sampling frequency

            # Currently coded to lie within [0, 0.5]

            while not aux_temp:         # That is when aux_temp == False
                # global omega_star
                omega_star = np.random.normal(loc=omega_current[j], scale=ill.sigma_RW)
                if not(omega_star <=0 or omega_star >=0.5):
                    aux_temp = True

            # global omega_prop, omega_star
            omega_prop = copy.copy(omega_curr)

            # Updating the jth component
            omega_prop[j] = omega_star

            log_likelik_ratio = log_posterior_omega_stationary(omega_prop, beta_current, time_star) - \
                                    log_posterior_omega_stationary(omega_curr, beta_current, time_star)

            MH_ratio = np.exp(log_likelik_ratio)

            U = np.random.rand()

            if U <= min(1, MH_ratio):
                omega_current_aux = omega_prop
            else:
                omega_current_aux = omega_curr

        omega_out = np.sort(omega_current_aux)

    # --- Sampling basis function coefficients from the posterior-----

    X_post = get_X_star(omega_out, time_star)

    beta_var_post = np.eye(int(2 * m_current)) / math.pow(ill.sigma_beta, 2) + \
                    np.matmul(np.transpose(X_post), X_post) / math.pow(sigma, 2)
    beta_var_post = np.linalg.inv(beta_var_post)
    beta_var_post = (np.transpose(beta_var_post) + beta_var_post)/2
    beta_mean_post = np.matmul(beta_var_post, ((np.matmul(np.transpose(X_post), y_star)) / math.pow(sigma, 2)))
    beta_out = np.random.multivariate_normal(mean=beta_mean_post,
                                                  cov=0.5 * (beta_var_post + np.transpose(beta_var_post)))

    # --- Sampling sigma from the posterior distribution -----

    X_post = get_X_star(omega_out, time_star)
    residual_variance = y_star - np.matmul(X_post, beta_out)
    residual_variance = np.power(residual_variance, 2)
    res_var = np.sum(residual_variance)
    nu_post = (n + ill.nu0)/2
    gamma_post = (ill.gamm0 + res_var)/2

    invgamma = stats.invgamma.rvs(a=nu_post, loc=gamma_post)
    sigma_out = np.sqrt(invgamma)

    # ---- output ----
    output = {"beta": beta_out, "omega": omega_out, "sigma":sigma_out}

    return output


def death_move_stationary(info_segment_ts, m_current, beta_current,
                          omega_current, sigma_current, time_star, lambda_var, c, phi_omega,chi_omega):

    if 2*len(omega_current) != len(beta_current):
        raise CustomError

    global sigma
    sigma = sigma_current

    time_indexes = info_segment_ts["time"]

    m_proposed = m_current - 1

    index = np.random.choice(np.arange(0, m_current))
    m_current_set = np.arange(0, m_current)
    # omega_proposed = np.vstack([omega_current[np.arange(0, index-1)],
    #                           omega_current[np.arange(index+1, (len(omega_current)-1))]])
    m_proposed_set = np.setdiff1d(m_current_set, index)
    omega_proposed = omega_current[m_proposed_set]

    # - Proposing beta ~ Normal(beta_prop, sigma_prop)
    X_prop = get_X_star(omega_proposed, time_star)
    beta_var_prop = np.eye(int(2*m_proposed))/math.pow(ill.sigma_beta, 2) + \
                    np.matmul(np.transpose(X_prop), X_prop)/math.pow(sigma, 2)
    beta_var_prop = np.linalg.inv(beta_var_prop)
    beta_mean_prop = np.matmul(beta_var_prop, ((np.matmul(np.transpose(X_prop), y_star)) / math.pow(sigma, 2)))
    beta_proposed = np.random.multivariate_normal(mean=beta_mean_prop,
                                                  cov=0.5*(beta_var_prop + np.transpose(beta_var_prop)))

    # Obtain beta_current and sigma_current to evaluate the proposal ratio

    X_curr = get_X_star(omega_current, time_star)
    beta_var_curr = np.eye(int(2*m_current))/math.pow(ill.sigma_beta, 2) + \
                    np.matmul(np.transpose(X_curr), X_curr)/math.pow(sigma, 2)
    beta_var_curr = np.linalg.inv(beta_var_curr)
    beta_mean_curr = np.matmul(beta_var_curr, ((np.matmul(np.transpose(X_curr), y_star)) / math.pow(sigma, 2)))
    length_support_omega = phi_omega - (2 * (m_current + 1) * chi_omega)
    # ---- Evaluating acceptance probability

    # ---- Log likelihood ratio
    log_likelik_prop = log_likelik(beta_proposed, omega_proposed, time_star)
    log_likelik_curr = log_likelik(beta_current, omega_current, time_star)
    log_likelik_ratio = log_likelik_prop - log_likelik_curr

    # Log prior ratio
    log_m_prior_ratio =  math.log(poisson.pmf(m_proposed, lambda_var)) - math.log(poisson.pmf(m_current, lambda_var))
    log_beta_prior_ratio = np.log(mvnormal.pdf(x=beta_proposed, mean=np.zeros(2*m_proposed),
                                        cov=math.pow(ill.sigma_beta,2)*np.eye(2*m_proposed))) - np.log(
                                                            mvnormal.pdf(x=beta_current, mean=np.zeros(2*m_current),
                                                            cov=math.pow(ill.sigma_beta,2)*np.eye(2*m_current)))

    log_omega_prior_ratio = np.log(0.5)

    log_prior_ratio = log_m_prior_ratio + log_beta_prior_ratio + log_omega_prior_ratio

    beta_intermediate1 = beta_proposed - beta_mean_prop
    beta_intermediate2 = np.transpose(beta_intermediate1)
    beta_intermediate3 = np.linalg.inv(beta_var_prop)
    beta_lsd = np.log(np.sqrt(np.linalg.det(2*np.pi*beta_var_prop)))
    beta_matmul = np.matmul(beta_intermediate2, beta_intermediate3, beta_intermediate1)
    beta_matmul = -0.5*beta_matmul
    log_proposal_beta_prop = beta_matmul - beta_lsd
    log_proposal_beta_prop = log_proposal_beta_prop[0]

    beta_intermediate1 = beta_current - beta_mean_curr
    beta_intermediate2 = np.transpose(beta_intermediate1)
    beta_intermediate3 = np.linalg.inv(beta_var_curr)
    beta_lsd = np.log(np.sqrt(np.linalg.det(2*np.pi*beta_var_curr)))
    beta_matmul = np.matmul(beta_intermediate2, beta_intermediate3, beta_intermediate1)
    beta_matmul = -0.5*beta_matmul
    log_proposal_beta_current = beta_matmul - beta_lsd
    log_proposal_beta_current = log_proposal_beta_current[0]

    log_proposal_omega_current = np.log((1/(length_support_omega)))
    log_proposal_omega_proposed = np.log((1/m_current))

    poisson_eval = poisson.pmf(m_current, lambda_var) / poisson.pmf(m_proposed, lambda_var)
    log_proposal_birth_move = np.log(c * min(1, poisson_eval))

    poisson_eval = poisson.pmf(m_proposed, lambda_var) / poisson.pmf(m_current, lambda_var)
    log_proposal_death_move = np.log(c * min(1, poisson_eval))

    log_proposal_ratio = log_proposal_birth_move - log_proposal_death_move + log_proposal_omega_current - \
                        log_proposal_omega_proposed + log_proposal_beta_current - log_proposal_beta_prop

    # - Metropolis Hastings acceptance step
    MH_ratio_death = log_likelik_ratio + log_prior_ratio + log_proposal_ratio
    epsilon_death = min(1, np.exp(MH_ratio_death))

    U = np.random.rand()

    if U <= epsilon_death:
        beta_out = beta_proposed
        omega_out = omega_proposed
        accepted = True
    else:
        beta_out = beta_current
        omega_out = omega_current
        accepted = False

    # - Updating sigma in a Gibbs step

    X_post = get_X_star(omega_out, time_star)
    residual_variance = y_star - np.matmul(X_post, beta_out)
    residual_variance = np.power(residual_variance, 2)
    res_var = np.sum(residual_variance)
    nu_post = (n + ill.nu0)/2
    gamma_post = (ill.gamm0 + res_var)/2

    invgamma = stats.invgamma.rvs(a=nu_post , loc=gamma_post)
    sigma_out = np.sqrt(invgamma)

    # -------- output
    omega_out = np.matrix.flatten(omega_out)
    output = {"beta": beta_out, "omega": omega_out, "sigma": sigma_out, "accepted": accepted}

    return output


def sample_uniform_continous_interval(n_sample, intervals):

    out = np.zeros(n_sample)
    n_intervals = len(intervals)

    # - Getting length of each interval
    len_intervals = np.zeros(n_intervals)

    for k in range(0, n_intervals):
        aux = intervals[k, 1] - intervals[k, 0]
        if aux < 0:
            aux = 0
        len_intervals[k] = aux

    # Getting proportion of each interval
    weights = np.zeros(n_intervals)

    for k in range(0, n_intervals):
        weights[k] = len_intervals[k]/np.sum(len_intervals)

    # Getting samples
    for j in range(0, n_sample):
        indicator = np.random.choice(np.arange(0, n_intervals),p=weights,replace=False)
        out[j] = np.random.uniform(intervals[indicator[0]], intervals[indicator[1]])

    return out


# ------ Basis function matrix -------
# ------ Returning length(time_points) X 2length(omega)
# Function: function return design matrix with basis function,
#           that refers to all data points belonging to a certain
#           regime. Namely, time_star correspond to time points
#           that are related to the same regime.


def get_X_star(omega, time_star):


    if not type(omega) is list and np.size(omega) < 2:
        omega = [omega]

    if not type(time_star) is list and np.size(time_star) < 2:
        time_star = [time_star]

    omega = np.array(omega)
    time_star = np.array(time_star)
    #M = len(omega)
    M = np.size(omega)
    X = np.ones([len(time_star), 2*M])

    for tt in range(len(time_star)):
        t = time_star[tt]
        M_cos = np.arange(0, 2*M, 2)
        M_sin = np.arange(1, 2*M+1, 2)
        for j in range(M):
            X_temp_cos = math.cos(2 * math.pi * t * omega[j])
            X_temp_sin = math.sin(2 * math.pi * t * omega[j])
            #X[t][M_cos[j]] = X_temp_cos
            #X[t][M_sin[j]] = X_temp_sin
            X[tt, M_cos[j]] = X_temp_cos
            X[tt, M_sin[j]] = X_temp_sin


    return X



def get_X(omega, a , b):

    #M = len(omega)
    M = np.size(omega)

    if not type(omega) is list and np.size(omega) < 2:
        omega = [omega]


    time = np.array(range(a, (b+1)))
    X = np.ones([len(time), 2*M])

    for tt in range(len(time)):
        t = time[tt]
        M_cos = np.arange(0, 2*M, 2)
        M_sin = np.arange(1, 2*M+1, 2)
        for j in range(M):
            X_temp_cos = math.cos(2 * math.pi * t * omega[j])
            X_temp_sin = math.sin(2 * math.pi * t * omega[j])
            X[tt, M_cos[j]] = X_temp_cos
            X[tt, M_sin[j]] = X_temp_sin

    return X


def neg_f_posterior_beta_stationary(beta, omega, sigma, sigma_beta, time_star, y_star):
   val = neg_log_posterior_beta_stationary(beta, omega, sigma, sigma_beta, time_star, y_star)
   return val

def neg_log_posterior_beta_stationary(beta, omega, sigma, sigma_beta, time_star, y_star ):

    X = get_X_star(omega, time_star)
    fa = y_star - np.matmul(X, beta)
    fa = np.power(fa, 2)
    fa = -np.sum(fa)
    fa = fa/(2*math.pow(sigma, 2))
    fb = np.matmul(beta, np.transpose(beta))
    fb = fb/(2*math.pow(sigma_beta, 2))
    f = fa - fb
    f = -f

    return f

def neg_g_posterior_beta_stationary(beta, omega, sigma, sigma_beta, time_star, y_star):
    val = neg_grad_log_posterior_beta_stationary(beta, omega, sigma, sigma_beta, time_star, y_star)
    return val


def neg_grad_log_posterior_beta_stationary(beta, omega, sigma, sigma_beta, time_star, y_star):

    p = len(beta)
    g = np.zeros(p)
    X = get_X_star(omega, time_star)

    for i in range(p):
        fa = y_star - np.matmul(X, beta)
        fa = np.multiply(fa, X[:, i])
        fa = fa/(math.pow(sigma, 2))
        fa = fa - (beta[i]/(math.pow(sigma_beta, 2)))
        fa = np.sum(fa)
        g[i] = fa
    return -g


def neg_h_posterior_beta_stationary(beta, omega, sigma, sigma_beta, time_star, y_star):
    val = neg_hess_log_posterior_beta_stationary(beta, omega, sigma, sigma_beta, time_star, y_star)
    return val


def neg_hess_log_posterior_beta_stationary(beta, omega, sigma, sigma_beta, time_star, y_star):

    p = len(beta)
    h = np.zeros([p, p])
    X = get_X_star(omega, time_star)

    for i in range(p):
        fa = np.power(X[:, i], 2)
        fa = fa/(math.pow(sigma, 2))
        fa = fa - (1/(math.pow(sigma_beta, 2)))
        fa = -np.sum(fa)
        h[i, i] = fa
    return -h

# Log posterior omega


def log_posterior_omega_stationary(omega, beta, time_star):

    X = get_X_star(omega, time_star)
    fa = y_star - np.matmul(X, beta)
    fa = fa * np.power(fa, 2)
    fa = fa / (2 * math.pow(sigma, 2))
    fa = -np.sum(fa)

    return fa


def log_likelik(beta, omega, time_star):
    X = get_X_star(omega, time_star)
    fa = y_star - np.matmul(X, beta)
    fa = fa * np.power(fa, 2)
    fa = fa / 2*math.pow(sigma, 2)
    fa = np.sum(fa)

    fb = n*np.log(math.pow(sigma, 2))/2
    fb = n*np.log(2*np.pi)/2 - fb

    out = -fb -fa

    return out

