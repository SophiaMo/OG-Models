import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os

# Household Parameters
yrs_live = 80
S = 80
beta_annual = .96
beta = beta_annual ** (yrs_live / S)
sigma = 3.0
nvec = np.ones(S)
cut = round(2 * S / 3)
nvec[cut: ] = 0.2

# Firm Parameters
alpha = 0.35
A = 1.0
delta_annual = 0.05
delta = 1- ((1 - delta_annual) ** (yrs_live / S))

# Define functions
def get_L(nvec): # function for aggregate labor
    L = nvec.sum()
    return L

def get_K(bvec): # function for aggregate capital
    K = bvec.sum()
    return K

def get_w(K, L, params): # function for wage
    A, alpha = params
    w = (1 - alpha) * A * ((K / L) ** alpha)
    return w

def get_r(K, L, params): #function for interest rate
    A, alpha, delta = params
    r = alpha * A * ((L / K) ** (1 - alpha)) - delta
    return r

# Euler function error
def errors(bvec, *args):
    A, alpha, delta, nvec, beta, sigma = args
    L = get_L(nvec)
    K = get_K(bvec)
    w = get_w(K, L, (A, alpha))
    r = get_r(K, L, (A, alpha, delta))
    b1 = np.append(0, bvec)
    b2 = np.append(bvec, 0)
    c = (1+r) * b1 + w * nvec - b2
    muc = c ** (-sigma)
    errors = muc[:-1] - beta * (1+r) * muc[1:]
    return errors

# Solve for steady state values
bvec_guess = 0.1 * np.ones(S - 1)
b = opt.root(errors, bvec_guess, args = (A, alpha, delta, nvec, beta, sigma))
print(b)
b_ss = b.x
K_ss = get_K(b_ss)
L_ss = get_L(nvec)
w_ss = get_w(K_ss, L_ss, (A, alpha))
r_ss = get_r(K_ss, L_ss, (A, alpha, delta))
b1_ss = np.append(0, b_ss)
b2_ss = np.append(b_ss, 0)
c_ss = (1+r_ss) * b1_ss + w_ss * nvec - b2_ss
print('\n Savings: \t\t\t {} \n Capital and Labor: \t\t {} \n Wage and Interest rate: \t {} \n Consumption: \t\t\t {}'.format(b_ss, np.array([K_ss, L_ss]), np.array([w_ss, r_ss]), c_ss))


# TPI params
T = 200
max_iter = 300
tol = 1e-9
xi = 0.2

# Initial guess for capital stock
b1vec = 0.93 * b_ss # Starting distribution of wealth
K1 = get_K(b1vec)
Kpath_old = np.zeros(T + S - 1)
Kpath_old[:T] = np.linspace(K1, K_ss, T) # Until reaching steady state
Kpath_old[T:] = K_ss

# Euler function error
'''
Calculate lifetime uler function error. Remaining lifetime can be of varying length p.
bvec is of length p-1.

'''

def get_errors(bvec, *args):
    beg_wealth, nvec, beta, sigma, w_path, r_path = args
    b1 = np.append(beg_wealth, bvec)
    b2 = np.append(bvec, 0)
    c = (1 + r_path) * b1 + w_path * nvec - b2
    muc = c ** (-sigma)
    errors = muc[:-1] - beta * (1 + r_path[1:]) * muc[1:]
    return errors


# Begin TPI

abs2 = 1
tpi_iter = 0
while abs2 > tol and tpi_iter < max_iter:
    tpi_iter = tpi_iter + 1
    w_path = get_w(Kpath_old, L_ss, (A, alpha))
    r_path = get_r(Kpath_old, L_ss, (A, alpha, delta))
    # Initialize savings matrix
    b = np.zeros((S - 1, T + S - 1))
    b[:, 0] = b1vec
    # Solve the incomplete remaining lifetime decisions of agents alive
    # in period t=1 but not born in period t=1
    for p in range(2, S):
        bvec_guess = np.diagonal(b[S - p:, :p - 1]) # Initial guess of the lifetime saving path who has p periods to live
        beg_wealth = b[S - p - 1, 0]
        args_bp = (beg_wealth, nvec[-p:], beta, sigma, w_path[:p], r_path[:p])
        bp = opt.root(get_errors, bvec_guess, args = (args_bp)).x
    # Insert the vector lifetime solutions diagonally (twist donut)
        DiagMaskbp = np.eye(p - 1)
        bp_path = DiagMaskbp * bp
        b[S - p:, 1:p] += bp_path

    # Solve for complete lifetime decisions of agents born in periods
    # 1 to T and insert the vector lifetime solutions diagonally (twist
    # donut) into the cpath, bpath, and EulErrPath matrices
    for t in range(1, T + 1):
        bvec_guess = np.diagonal(b[:, t - 1:S + t - 2])
        args_bt = (0, nvec, beta, sigma, w_path[t - 1 : S + t - 1], r_path[t - 1 : S + t - 1])
        bt = opt.root(get_errors, bvec_guess, args = (args_bt)).x
        DiagMaskbt = np.eye(S - 1)
        bt_path = DiagMaskbt * bt
        b[:, t: S + t - 1] += bt_path

    # Calculate the implied capital stock from conjecture and the error
    Kpath_new = b.sum(axis = 0)
    abs2 = ((Kpath_old[:T] - Kpath_new[:T]) ** 2).sum()
    # Update guess
    Kpath_old[:T] = xi * Kpath_new[:T] + (1 - xi) * Kpath_old[:T]
    print('iteration:', tpi_iter, ' squared distance: ', abs2)


plot = True
if plot:
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = "images_1"
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)

    plt.plot (1 + np.arange(T), Kpath_old[: T], 'go--')
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Time path of aggregate capital', fontsize=20)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$K$')
    output_path = os.path.join(output_dir, 'kplot1')
    plt.savefig(output_path)
    plt.close()
