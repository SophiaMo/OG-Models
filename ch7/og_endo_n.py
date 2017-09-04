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
    r, beta, sigma, p= params
    cvec = np.zeros(p)
    cvec[0] = c1
    cs = c1
    s = 0
    while s < p - 1:
        cvec[s + 1] = cs * (beta * (1 + r[s + 1])) ** (1 / sigma)
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

def get_b(cvec, nvec, r, w, p, bs = 0.0): # function for calculating lifetime savings, given consumption and labor decisions
    bvec = np.zeros(p)
    s = 0
    bvec[0] = bs
    while s < p - 1:
        bvec[s + 1] = (1 + r[s]) * bs + w[s] * nvec[s] - cvec[s]
        bs = bvec[s + 1]
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

def get_b_last(c1, *args): # function for last-period savings, given intial guess c1
    r, w, beta, sigma, l_ub, b, upsilon, p = args
    cvec = get_c(c1, (r, beta, sigma, p))
    nvec = get_n((cvec, sigma, l_ub, b, upsilon, w))
    bvec = get_b(cvec, nvec, r, w, p, bs = 0.0)
    b_last = (1 + r[-1]) * bvec[-1] + w[-1] * nvec[-1] - cvec[-1]
    return b_last

# Firm Functions
def get_w(r, params): # function for wage given interest rate
    A, alpha, delta = params
    w = (1 - alpha) * A * (((alpha * A) / (r + delta)) ** (alpha / (1 - alpha)))
    return w

def get_r(K, L, params): # function for interest rate given aggregate capital and labor
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
xi = 0.1
abs_ss = 1
ss_iter = 0
r_old = 0.06
while abs_ss > tol and ss_iter < max_iter:
    ss_iter += 1
    r_old = r_old * np.ones(S)
    w_old = get_w(r_old, (A, alpha, delta)) * np.ones(S)
    # Calculate household decisions that make last-period savings zero
    c1_guess = 0.1
    c1_args = (r_old, w_old, beta, sigma, l_ub, b, upsilon, S)
    result_c1 = opt.root(get_b_last, c1_guess, args = (c1_args))
    if result_c1.success:
        c1 = result_c1.x
    else:
        raise ValueError("failed to find an appropriate initial consumption")
    # Calculate aggregate supplies for capital and labor
    cvec = get_c(c1, (r_old, beta, sigma, S))
    nvec = get_n((cvec, sigma, l_ub, b, upsilon, w_old))
    bvec = get_b(cvec, nvec, r_old, w_old, S)
    K = get_K(bvec)
    L = get_L(nvec)
    r_new = get_r(K, L, (A, alpha, delta))
    # Check market clearing
    abs_ss = ((r_new - r_old) ** 2).max()
    # Update guess
    r_old = xi * r_new + (1 - xi) * r_old
    print('iteration:', ss_iter, ' squared distance: ', abs_ss)
r_ss = r_old * np.ones(S)
w_ss = get_w(r_ss, (A, alpha, delta)) * np.ones(S)


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


# TPI

T1 = 250
T2 = 300
TPI_tol = 1e-12
TPI_xi = 0.4
b1vec = 1.08 * bvec
K1 = get_K(b1vec)
Kpath_old = np.zeros(T2 + S - 1)
Kpath_old[:T1] = np.linspace(K1, K_old, T1) # Until reaching steady state aggregate capital
Kpath_old[T1:] = K_old
Lpath_old = L_old / K_old * Kpath_old # guess for aggregate labor path

abs_tpi = 1
tpi_iter = 0
while abs_tpi > tol and tpi_iter < max_iter:
    tpi_iter += 1
    w_path = get_w(Kpath_old, Lpath_old, (A, alpha))
    r_path = get_r(Kpath_old, Lpath_old, (A, alpha, delta))
    bmat = np.zeros((S, T2 + S - 1))
    bmat[:, 0] = b1vec
    nmat = np.zeros((S, T2 + S - 1))
    # Solve the incomplete remaining lifetime decisions of agents alive
    # in period t=1 but not born in period t=1
    for p in range(1, S): # p is remaining periods of life
        c1_guess = 1.4
        c1_args = (r[:p], w[:p], beta, sigma, l_ub, b, upsilon, p)
        result_c1 = opt.root(get_b_last, c1_guess, args = (c1_args))
        if result_c1.success:
            c1 = result_c1.x
        else:
            raise ValueError("failed to find an appropriate initial consumption")
        # Calculate aggregate supplies for capital and labor
        cp = get_c(c1, (r[:p], beta, sigma, p))
        np = get_n((cvec, sigma, l_ub, b, upsilon, w[:p]))
        bp = get_b(cvec, nvec, (r[:p], w[:p]), bs = b1vec[S - p],)[:-1]
    # Insert the vector lifetime solutions diagonally (twist donut)
        DiagMaskbp = np.eye(p - 1)
        bp_path = DiagMaskbp * bp
        bmat[S - p + 1:, 1:p] += bp_path

        DiagMasknp = np.eye(p)
        np_path = DiagMasknp * np
        nmat[S - p:, :p] += np_path

    # Solve for complete lifetime decisions of agents born in periods
    # 1 to T and insert the vector lifetime solutions diagonally (twist
    # donut) into the cpath, bpath, and EulErrPath matrices
    for t in range(1, T2):
        c1_guess = 1.4
        c1_args = (r[t - 1 : S + t - 1], w[t - 1 : S + t - 1], beta, sigma, l_ub, b, upsilon, p)
        result_c1 = opt.root(get_b_last, c1_guess, args = (c1_args))
        if result_c1.success:
            c1 = result_c1.x
        else:
            raise ValueError("failed to find an appropriate initial consumption")
        # Calculate aggregate supplies for capital and labor
        cvec = get_c(c1, (r[t - 1 : S + t - 1], beta, sigma, S))
        nvec = get_n((cvec, sigma, l_ub, b, upsilon, w[t - 1 : S + t - 1]))
        bp = get_b(cvec, nvec, (r[t - 1 : S + t - 1], w[t - 1 : S + t - 1]))
        DiagMaskbt = np.eye(S - 1)
        bt_path = DiagMaskbt * bt
        bmat[1:, t + 1: t + S] += bt_path

        DiagMasknp = np.eye(S)
        np_path = DiagMasknp * np
        nmat[:, t: t + S] += np_path
    # Calculate the implied capital stock from conjecture and the error
    Kpath_new = bmat.sum(axis = 0)
    Lpath_new = nmat.sum(axis = 0)
    abs_tpi = ((Kpath_old[:T1] - Kpath_new[:T1]) ** 2).sum() + ((Lpath_old[:T1] - Lpath_new[:T1]) ** 2).sum()

    # Update guess
    Kpath_old[:T1] = TPI_xi * Kpath_new[:T1] + (1 - TPI_xi) * Kpath_old[:T1]
    Lpath_old[:T1] = TPI_xi * Lpath_new[:T1] + (1 - TPI_xi) * Lpath_old[:T1]
    print('iteration:', tpi_iter, ' squared distance: ', abs_tpi)

w_path = get_w(Kpath_old, Lpath_old, (A, alpha))
r_path = get_r(Kpath_old, Lpath_old, (A, alpha, delta))

plot = True
if plot:
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = "images_1"
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)

    plt.plot (np.arange(T2), w_path, 'go--', color = 'green', label = 'savings')
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Steady-State distribution of Consumption and Savings', fontsize=10)
    plt.xlabel('age')
    plt.ylabel('units of consumption')
    plt.legend()
    output_path1 = os.path.join(output_dir, 'ss_bc')
    plt.savefig(output_path1)
    plt.close()
