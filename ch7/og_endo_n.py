import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os

# Household Parameters
yrs_live = 80
S = 80
beta_annual = .96
beta = beta_annual ** (yrs_live / S)
sigma = 2.5
l_ub = 1.0
b = 0.501
upsilon = 1.554

# Firms Parameters
alpha = 0.35
A = 1.0
delta_annual = 0.05
delta = 1.0 - ((1.0 - delta_annual) ** (yrs_live / S))

# Household functions
def get_c(c1, params): # function for implied steady-state consumption, given initial guess c1
    r, beta, sigma, S = params
    cvec = np.zeros(S)
    cvec[0] = c1
    cs = c1
    s = 0
    while s < S - 1:
        cvec[s + 1] = cs * (beta * (1 + r)) ** (1 / sigma)
        cs = cvec[s + 1]
        s += 1
    return cvec

def MU_c_stitch(cvec, sigma):
    epsilon = 1e-4
    muc = cvec ** (-sigma)
    m1 = (-sigma) * epsilon ** (-sigma - 1)
    m2 = epsilon ** (-sigma) - m1 * epsilon
    c_cnstr = cvec < epsilon
    muc[c_cnstr] = m1 * cvec[c_cnstr] + m2
    return muc

def MU_n_stitch(nvec, params):
    l_ub, b, upsilon = params
    epsilon_lb = 1e-6
    epsilon_ub = l_ub - epsilon_lb

    mun = ((b / l_ub) * ((nvec / l_ub) ** (upsilon - 1)) * (1 - ((nvec / l_ub) ** upsilon)) **\
           ((1 - upsilon) / upsilon))

    m1 = (b * (l_ub ** (-upsilon)) * (upsilon - 1) * (epsilon_lb ** (upsilon - 2)) * \
         ((1 - ((epsilon_lb / l_ub) ** upsilon)) ** ((1 - upsilon) / upsilon)) * \
         (1 + ((epsilon_lb / l_ub) ** upsilon) * ((1 - ((epsilon_lb / l_ub) ** upsilon)) ** (-1))))
    m2 = ((b / l_ub) * ((epsilon_lb / l_ub) ** (upsilon - 1)) * \
         ((1 - ((epsilon_lb / l_ub) ** upsilon)) ** ((1 - upsilon) / upsilon)) - (m1 * epsilon_lb))

    q1 = (b * (l_ub ** (-upsilon)) * (upsilon - 1) * (epsilon_ub ** (upsilon - 2)) * \
         ((1 - ((epsilon_ub / l_ub) ** upsilon)) ** ((1 - upsilon) / upsilon)) * \
         (1 + ((epsilon_ub / l_ub) ** upsilon) * ((1 - ((epsilon_ub / l_ub) ** upsilon)) ** (-1))))

    q2 = ((b / l_ub) * ((epsilon_ub / l_ub) ** (upsilon - 1)) * \
         ((1 - ((epsilon_ub / l_ub) ** upsilon)) ** ((1 - upsilon) / upsilon)) - (q1 * epsilon_ub))

    nl_cstr = nvec < epsilon_lb
    nu_cstr = nvec > epsilon_ub

    mun[nl_cstr] = m1 * nvec[nl_cstr] + m2
    mun[nu_cstr] = q1 * nvec[nu_cstr] + q2
    return mun

def get_b(cvec, nvec, params):
    r, w = params
    bvec = np.zeros_like(cvec)
    bs = 0.0
    s = 0
    while s < S :
        bvec[s] = (1 + r) * bs + w * nvec[s] - cvec[s]
        bs = bvec[s]
        s += 1
    return bvec

def get_n_errors(nvec, *args):
    cvec, sigma, l_ub, b, upsilon, w = args
    muc = MU_c_stitch(cvec, sigma)
    mun = MU_n_stitch(nvec, (l_ub, b, upsilon))
    n_errors = w * muc - mun
    return n_errors

def get_n(params): # params = cvec, sigma, l_ub, b, upsilon, w
    n_args = params
    n_guess = 0.5 * l_ub * np.ones(S)
    result = opt.root(get_n_errors, n_guess, args = (n_args), method = 'lm')
    if result.success:
        nvec = result.x
    else:
        raise ValueError("failed to find an appropriate labor decision")
    return nvec

def get_b_last(c1, *args):
    r, w, beta, sigma, l_ub, b, upsilon, S = args
    c_params = (r, beta, sigma, S)
    cvec = get_c(c1, c_params)
    n_params = (cvec, sigma, l_ub, b, upsilon, w)
    b_params = (r, w)
    nvec = get_n(n_params)
    bvec = get_b(cvec, nvec, b_params)
    b_last = bvec[-1]
    return b_last

# Firm Functions
def get_w(K, L, params): # function for wage
    A, alpha = params
    w = (1 - alpha) * A * ((K / L) ** alpha)
    return w

def get_r(K, L, params): # function for interest rate
    A, alpha, delta = params
    r = alpha * A * ((L / K) ** (1 - alpha)) - delta
    return r

# Aggregate supply functions
def get_L(nvec): # function for aggregate labor supply
    L = nvec.sum()
    return L

def get_K(bvec): # function for aggregate capital supply
    K = bvec.sum()
    return K

# Find Steady State
max_iter = 400
tol = 1e-9
xi = 0.2
abs_ss = 1
ss_iter = 0
K_old = 500.0
L_old = 60.0
while abs_ss > tol and ss_iter < max_iter:
    ss_iter += 1
    w = get_w(K_old, L_old, (A, alpha))
    r = get_r(K_old, L_old, (A, alpha, delta))
    # Calculate household decisions that make last-period savings zero
    c1_guess = 1.4
    c1_args = (r, w, beta, sigma, l_ub, b, upsilon, S)
    result_c1 = opt.root(get_b_last, c1_guess, args = (c1_args))
    if result_c1.success:
        c1 = result_c1.x
    else:
        raise ValueError("failed to find an appropriate initial consumption")
    # Calculate aggregate supplies for capital and labor
    cvec = get_c(c1, (r, beta, sigma, S))
    nvec = get_n((cvec, sigma, l_ub, b, upsilon, w))
    bvec = get_b(cvec, nvec, (r, w))
    K_new = get_K(bvec)
    L_new = get_L(nvec)
    # Check market clearing
    abs_ss = (K_old- K_new) ** 2 + (L_old- L_new) ** 2
    # Update guess
    K_old = xi * K_new + (1 - xi) * K_old
    L_old = xi * L_new + (1 - xi) * L_old
    print('iteration:', ss_iter, ' squared distance: ', abs_ss)

plot = True
if plot:
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = "images_1"
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)

    plt.plot (21 + np.arange(80), bvec, 'go--', color = 'green', label = 'savings')
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.plot (21 + np.arange(80), cvec, 'go--', color = 'blue', label = 'consumption')
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Steady-State distribution of Consumption and Savings', fontsize=10)
    plt.xlabel('age')
    plt.ylabel('units of consumption')
    plt.legend()
    output_path1 = os.path.join(output_dir, 'ss_bc')
    plt.savefig(output_path1)
    plt.close()

    plt.plot (1 + np.arange(80), nvec, 'go--', label = 'labor supply')
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Steady-State distribution of Labor Supply', fontsize=20)
    plt.xlabel('age')
    plt.ylabel('labor supply')
    plt.legend()
    output_path2 = os.path.join(output_dir, 'ss_n')
    plt.savefig(output_path2)
    plt.close()

print("Steady State Capital: {} \n Labor : {}".format(K_old, L_old))

# TPI
T1 = 250
T2 = 300
TPI_tol = 1e-12
TPI_xi = 0.4
