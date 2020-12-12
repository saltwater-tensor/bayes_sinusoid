
import numpy as np
import copy


#global a, data, b, time_star, y, y_star, n, omega, sigma

# ---------------------------SIMULATED TIME SERIES--------------------------- #

T = 1450
pi_z_true = np.zeros((3, 3))
pi_z_true[0][0] = 0.99
pi_z_true[0][1] = 0.0097
pi_z_true[0][2] = 0.0003
pi_z_true[1][0] = 0.0001
pi_z_true[1][1] = 0.99
pi_z_true[1][2] = 0.0099
pi_z_true[2][0] = 0.0097
pi_z_true[2][1] = 0.0003
pi_z_true[2][2] = 0.99

pi_init_true = np.zeros([3, ])
pi_init_true[0] = 0.99
pi_init_true[1] = 0.005
pi_init_true[2] = 0.005

beta_true = ([[0.8, 0.8],       # basis coefficients
          [0.2, 0.2],
          [1.0, 1.0, 1.0, 1.0]])

omega_true = ([[1/25],      # frequencies
          [1/19],
          [1/12, 1/8]])

sigma_true = np.zeros([3, ])       # Gaussian innovations
sigma_true[0] = 0.4
sigma_true[1] = 0.08
sigma_true[2] = 0.3

data = []
signal = []
# ------------------------ Parameters of the SpectralHMM ---------------------- #
n_iter_MCMC = 15000

Kz = 7      # maximum number of states
n_freq_max = 5      # maximum number of frequencies per state

# -- Hyperparametrs RJMCMC_SegmentModelSearch --

c_S = 0.4   # c for RJCMC - SegmentModelSearch
lambda_S = 1  # λ for RJMCMC - SegmentModelSearch
phi_omega = 0.25    # New frequency is sampled from Unif (0, ψ_ω) - SegmentModelSearch
chi_omega = (1/T)    # Minimum distance between frequency - SegmentModelSearch
sigma_beta = 10     # prior β ∼ N(0, σ_β I )
alpha_mixing = 0.2  # mixing proportion for sampling ω (within-step)
sigma_RW = (1/(50*T))   # variance RW for sampling ω (within-step)
nu0 = 1/100     # prior σ ~ InverseGamma(ν0/2, γ0)
gamm0 = 1/100

# --- Hyperparameters sticky HDP-HMM --

a_alpha = 1; b_alpha = 0.01 # - (γ+κ) ∼ Gamma(a_α,b_α)
a_gamma = 1; b_gamma = 0.01 # - γ ∼ Gamma(a_γ,b_γ)
c = 100
d = 1   # ρ ∼ Beta(c,d)
hyperHMMhyperparams = {"a_alpha" : a_alpha, "b_alpha" : b_alpha,
                       "a_gamma" : a_gamma,
                       "b_gamma" : b_gamma,
                       "c" : c, "d" : d}

# --- Other parameteres --

n_inner_MCMC = 2        # n of inner RJMCMC for updating θ_j
n_min_obs = 10        # if a state is assigned less than n_min_obs, the
                      # parameters corresponding to that state are drawn
                      # from their priors.
sample_time_series = True # if true: when updating frequencies (sampling periodogram)
                          # for state j, select segment of time series with probability
                          # proportional to n of observation in that segment.
                          # if false: select MAP estimate.


# ------- MCMC objects --------------------- #

# ------ Model Parameters --------- #
# ----- These are the model parameters
m_sample = np.ones([Kz, n_iter_MCMC+1])
beta_sample = np.zeros([Kz, 2*n_freq_max, n_iter_MCMC+1])
omega_sample = np.zeros([Kz, n_freq_max, n_iter_MCMC+1])
sigma_sample = np.zeros([Kz, n_iter_MCMC+1])

# ------ Mode sequence ---------- #
stateSeq_sample = np.zeros([T, n_iter_MCMC+1])
indSeq_sample = np.zeros([T, Kz, n_iter_MCMC+1])
totSeq_sample = np.zeros([Kz, n_iter_MCMC+1])
classProb_sample = np.zeros([n_iter_MCMC+1, T, Kz])         # necessary for postprocessing
#                                                           the sample with relabelling
#                                                           algorithm.

# ------ HMM transitions distributions (pi) ------ #
pi_z_sample = np.zeros([Kz, Kz, n_iter_MCMC + 1])
pi_init_sample = np.zeros([Kz, n_iter_MCMC+1])
alpha_vec_sample = np.zeros([Kz, n_iter_MCMC+1])

# ----- HMM Hyperparameters ------ #
n_plus_k_sample = np.zeros([n_iter_MCMC + 1])
gamma_sample = np.zeros ([n_iter_MCMC + 1])
rho_sample = np.zeros([n_iter_MCMC + 1])

# ---- State counts: ---- #
N_sample = np.zeros([Kz + 1, Kz, n_iter_MCMC + 1])  # ---- to accumulate transition #
#                                                          from state i to j at every MCMC iter
M_sample = np.zeros([Kz + 1, Kz, n_iter_MCMC + 1])
barM_sample = np.zeros([Kz + 1, Kz, n_iter_MCMC + 1])
sum_w_sample = np.zeros([Kz, n_iter_MCMC + 1])

# ------ Log likelihood ------ #
log_likelik_sample = np.zeros(n_iter_MCMC+ 1)





# -- Sample transition distributions π_z, initial distribution π_init based on the hyperparameters above
# -- Sample global distribution beta that governs the dirichlet over pi_z and pi_z_init

"""""

# ---- start positions ----- #
z_start = 2*np.ones([T])
pi_init_start = dist_struct["pi_init"]
pi_init_sample[:, 0] = copy.copy(pi_init_start)
alpha_vec_sample[:, 0] = dist_struct["beta_vec"]

# -- The observation parameters also called emission parameters
sigma_smoothing = 0.1
hdphmm.get_starting_value_theta(z_start, 3)

"""








