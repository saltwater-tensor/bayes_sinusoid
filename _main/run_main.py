import illustrative_example as ill
import functions_Auxiliary as aux
import functions_RJMCMC_Segment as rjmcmc
import functions_StickyHDPHMM as hdphmm
import config as cfg
import os
import numpy as np
import copy
from scipy import signal as scsig
import scipy.stats as scstats
import showprogress
import matplotlib.pyplot as plt
import psutil
import time
import matplotlib.pyplot as plt

process = psutil.Process(os.getpid())
print(process.memory_info().rss/1000000000)

os.chdir("D:\\Abhinav_Sharma\\Software\\SpectralHMM\\SpectralHMM-master")

global y, time_star, y_star, n

# ------- Generating data -------
start_main = time.time()
np.random.seed([212])

z_true = aux.generate_labels(ill.pi_z_true, ill.pi_init_true, ill.T)

simulation = aux.generate_data(z_true, ill.beta_true, ill.omega_true, ill.sigma_true)
ill.data = simulation["data"]
ill.signal = simulation["signal"]

"""""
fig, axs = plt.subplots(2)
fig.suptitle('data and signal')
axs[0].plot(data)
axs[1].plot(signal)
axs[1].plot(z_true)

# plt.show()
"""""

HMMhyperparams = hdphmm.sample_hyperparams_init(ill.hyperHMMhyperparams)
ill.n_plus_k_sample[0] = HMMhyperparams["alpha_plus_k"]
ill.gamma_sample[0] = HMMhyperparams["gamma"]
ill.rho_sample[0] = HMMhyperparams["rho"]

dist_struct = hdphmm.sample_dist(ill.n_plus_k_sample[0], ill.gamma_sample[0], ill.rho_sample[0], ill.N_sample[:, :, 0],
                                 ill.barM_sample[:, :, 0], ill.Kz)
ill.pi_z_sample[:, :, 0] = dist_struct["pi_z"]

# ---- start positions ----- #
z_start = 2*np.ones([ill.T], dtype=np.int64)
pi_init_start = dist_struct["pi_init"]
ill.pi_init_sample[:, 0] = copy.copy(pi_init_start)
ill.alpha_vec_sample[:, 0] = dist_struct["beta_vec"]
sigma_smoothing = 0.1
hdphmm.get_starting_value_theta(z_start, 3)
print(ill.m_sample[:, 0])
print(ill.beta_sample[:, :, 0])
print(ill.omega_sample[:, :, 0])
print(ill.sigma_sample[:, 0])
# -------------- NO BUGS TILL HERE --------------------------
detrended_data = scsig.detrend(ill.data[0:ill.T])
f_period, power = scsig.periodogram([detrended_data],
                                        window="boxcar", scaling="spectrum")

I_smooth = power[0][1:np.size(power)]

freq = f_period[range(1, (len(f_period)))]

plt.plot(freq, I_smooth)
plt.xlabel("Normalised frequency i.e. F/samplingfreq")
plt.ylabel("Power spectrum value")
plt.close()
# ------------------ NO BUGS --------------------------------
#np.random.seed(10)

for tt in range(1, ill.n_iter_MCMC + 1):

    #  --- Block sample z_{1:T} | y_{1:T}
    start_mcmc_cycle = time.time()
    print(process.memory_info().rss/1000000000)
    modes_struct = hdphmm.sample_z(ill.pi_z_sample[:, :, tt-1], ill.pi_init_sample[:, tt-1], ill.m_sample[:, tt-1],
                                   ill.beta_sample[:, :, tt-1], ill.omega_sample[:, :, tt-1], ill.sigma_sample[:, tt-1])
    ill.N_sample[:, :, tt] = modes_struct["N"]
    ill.stateSeq_sample[:, tt] = modes_struct["stateSeq"]
    ill.classProb_sample[tt, :, :] = modes_struct["classProb"]  # need for relabelling algorithm

    totSeq = copy.copy(modes_struct["totSeq"])
    indSeq = copy.copy(modes_struct["indSeq"])

    ill.indSeq_sample[:, :, tt] = copy.copy(indSeq)
    ill.totSeq_sample[:, tt] = copy.copy(totSeq)
    ## ------------------------------ BUG FREE, both logical and runtime -----------------------------------
    if np.mod(tt, 1000) == 0:

        print(tt, "- totSeq - : ", totSeq)

    # --- Based on mode sequence assignment, sample how many tables in each
    #     restaurant are serving each of the selected dishes. Also sample the
    #     dish override variables

    tables = hdphmm.sample_tables(ill.N_sample[:, :, tt], ill.n_plus_k_sample[tt-1], ill.rho_sample[tt-1],
                                  ill.alpha_vec_sample[:, tt-1])
    ill.M_sample[:, :, tt] = tables["M"]
    ill.barM_sample[:, :, tt] = tables["barM"]
    ill.sum_w_sample[:, tt] = tables["sum_w"]
    ## ------------------------------ BUG FREE runtime CHECK TABLE SAMPLING WHEN OUTPUT COMES -------------------------
    # - Now sample pi init pi z and the average transition distribution beta
    dist_struct = hdphmm.sample_dist(ill.n_plus_k_sample[tt-1], ill.gamma_sample[tt-1], ill.rho_sample[tt-1],
                                     ill.N_sample[:, :, tt],
                                     ill.barM_sample[:, :, tt], ill.Kz)

    ill.pi_z_sample[:, :, tt] = dist_struct["pi_z"]
    ill.pi_init_sample[:, tt] = dist_struct["pi_init"]
    ill.alpha_vec_sample[:, tt] = dist_struct["beta_vec"]

    ## ------------------------------ BUG FREE runtime CHECK TABLE SAMPLING WHEN OUTPUT COMES -------------------------
    # ---- Updating emission parameters for different states
    for j in range(ill.Kz):

        # -- Updates for modes with at leat n_min observations
        if totSeq[j] > ill.n_min_obs:
            temp_ind = copy.copy(indSeq[0:totSeq[j], j])
            temp_ind_seg = hdphmm.get_time_indexes(temp_ind)
            #len_seg = [len(np.arange(temp_ind_seg[0, ii], temp_ind_seg[1, ii])) for ii in range(len(temp_ind_seg))]
            len_seg = np.zeros(len(temp_ind_seg))
            for ii in range(len(temp_ind_seg)):
                length_seg = len(np.arange(temp_ind_seg[ii][0], temp_ind_seg[ii][1]))
                if length_seg == 0:
                    len_seg[ii] = 1
                else:
                    len_seg[ii] = length_seg


            info_segment_ts = []
            if np.any(len_seg >= ill.n_min_obs):

                while True:
                    info_segment_ts = hdphmm.get_segment_ts(ill.data, temp_ind_seg, ill.sample_time_series)
                    if len(info_segment_ts["time"]) >= ill.n_min_obs:
                        break

                y = info_segment_ts["data"]
                time_star = indSeq[0:totSeq[j], j]
                y_star = ill.data[time_star]
                n = len(y_star)

                m_current = ill.m_sample[j, tt-1]
                beta_current = ill.beta_sample[j, 0: int(2*m_current), tt-1]
                omega_current = ill.omega_sample[j, 0: int(m_current), tt-1]
                sigma_current = ill.sigma_sample[j, tt-1]

                m_inner = np.zeros([ill.n_inner_MCMC+1], dtype=np.int64)
                beta_inner = np.zeros([2*ill.n_freq_max, ill.n_inner_MCMC+1])
                omega_inner = np.zeros([ill.n_freq_max, ill.n_inner_MCMC+1])
                sigma_inner = np.zeros([ill.n_inner_MCMC+1])

                print(j)
                print(tt)
                m_inner[0] = copy.copy(m_current)
                beta_inner[0: int(2*m_inner[0]), 0] = copy.copy(beta_current)
                omega_inner[0: int(m_inner[0]), 0] = copy.copy(omega_current)
                sigma_inner[0] = copy.copy(sigma_current)

                # Inner loop for updating emission parameters RJMCMC
                for ii in range(1, ill.n_inner_MCMC+1):

                    m_temp = copy.copy(m_inner[ii - 1])
                    beta_temp = copy.copy(beta_inner[0: int(2 * m_temp), ii - 1])
                    omega_temp = copy.copy(omega_inner[0: int(m_temp), ii - 1])
                    sigma_temp = copy.copy(sigma_inner[ii - 1])

                    # -- RJMCMC move
                    # MCMC = RJMCMC_SegmentModelSearch(info_segment_ts, m_temp, β_temp,
                    #                               ω_temp, σ_temp, time_star, λ_S, c_S, ϕ_ω, ψ_ω, n_freq_max)

                    MCMC = rjmcmc.RJMCMC_SegmentModelSearch(y_star, info_segment_ts, m_temp, beta_temp,
                                                            omega_temp, sigma_temp, time_star, ill.lambda_S, ill.c_S,
                                                            ill.phi_omega, ill.chi_omega, ill.n_freq_max)
                    m_inner[ii] = MCMC["m"]
                    beta_inner[0: int(2 * m_inner[ii]), ii] = MCMC["beta"]
                    omega_inner[0: int(m_inner[ii]), ii] = MCMC["omega"]
                    sigma_inner[ii] = MCMC["sigma"]

                ill.m_sample[j, tt] = copy.copy(m_inner[len(m_inner)-1])
                ill.beta_sample[j, 0: int(2 * ill.m_sample[j, tt]), tt] = copy.copy\
                                                            (beta_inner[0: int(2 * m_inner[len(m_inner)-1]), -1])

                ill.omega_sample[j, 0: int(ill.m_sample[j, tt]), tt] = copy.copy\
                                                            (omega_inner[0: int(m_inner[len(m_inner)-1]), -1])

                ill.sigma_sample[j, tt] = copy.copy(sigma_inner[-1])

            else:
                ill.m_sample[j, tt] = 1
                temp_dist = np.random.multivariate_normal(mean=np.zeros(int(2*ill.m_sample[j, tt])),
                                                          cov=np.eye(int(2*ill.m_sample[j, tt])))
                ill.beta_sample[j, 0: int(2 * ill.m_sample[j, tt]), tt] = temp_dist
                ill.omega_sample[j, 0: int(ill.m_sample[j, tt]), tt] = np.random.uniform(0, ill.phi_omega)
                ill.sigma_sample[j, tt] = scstats.invgamma.rvs(a=1, loc=4)

        else:
            ill.m_sample[j, tt] = 1
            temp_dist = np.random.multivariate_normal(mean=np.zeros(int(2 * ill.m_sample[j, tt])),
                                                      cov=np.eye(int(2 * ill.m_sample[j, tt])))
            ill.beta_sample[j, 0: int(2 * ill.m_sample[j, tt]), tt] = temp_dist
            ill.omega_sample[j, 0: int(ill.m_sample[j, tt]), tt] = np.random.uniform(0, ill.phi_omega)
            ill.sigma_sample[j, tt] = scstats.invgamma.rvs(a=1, loc=4)

    # --- Resample concentration parameters
    print(process.memory_info().rss/1000000000)

    HMMhyperparams = hdphmm.sample_hyperparams(ill.N_sample[:, :, tt], ill.M_sample[:, :, tt],
                                               ill.barM_sample[:, :, tt], ill.sum_w_sample[:, tt],
                                               ill.n_plus_k_sample[tt-1], ill.gamma_sample[tt-1],
                                               ill.hyperHMMhyperparams)
    ill.n_plus_k_sample[tt] = HMMhyperparams["alpha_plus_k"]
    ill.gamma_sample[tt] = HMMhyperparams["gamma"]
    ill.rho_sample[tt] = HMMhyperparams["rho"]

    # --- Log likelihood
    z = copy.copy(ill.stateSeq_sample[:, tt])
    m = copy.copy(ill.m_sample[:, tt])
    omega = copy.copy(ill.omega_sample[:, :, tt])
    beta = copy.copy(ill.beta_sample[:, :, tt])
    sigma = copy.copy(ill.sigma_sample[:, tt])
    pi_z = copy.copy(ill.pi_z_sample[:, :, tt])
    pi_init = copy.copy(ill.pi_init_sample[:, tt])

    ill.log_likelik_sample[tt] = aux.get_likelik_system(z, m, omega, beta, sigma, pi_z, pi_init)

    end_mcmc_cycle = time.time()

    print("ONE MCMC CYCLE in seconds is " + str(end_mcmc_cycle - start_mcmc_cycle))
# --- THE MAIN MCMC LOOP ENDS HERE
end_main = time.time()
print("ALL THE ITERATIONS in hours = " + str((end_main - start_main)/(60*60)))




















