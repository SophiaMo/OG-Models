import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os

# Household Parameters
nvec = np.array([1.0,1.0,0.2])
yrs_live = 60
s = 3
beta = 0.442
sigma = 3

# Firm Parameters
alpha = 0.35
A = 1.0
delta_annual = 0.05
delta = 1- ((1 - delta_annual) ** (yrs_live/s))

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
    b2, b3 = bvec
    A, alpha, delta, nvec, beta, sigma = args
    L = get_L(nvec)
    K = get_K(bvec)
    w = get_w(K, L, (A, alpha))
    r = get_r(K, L, (A, alpha, delta))
    c1 = w * nvec[0] - b2
    c2 = (1 + r) * b2 + w * nvec[1] - b3
    c3 = (1 + r) * b3 + w * nvec[2]
    muc1 = c1 ** (-sigma)
    muc2 = c2 ** (-sigma)
    muc3 = c3 ** (-sigma)
    error1 = muc1 - beta * (1+r) * muc2
    error2= muc2 - beta * (1+r) * muc3
    errors = np.array([error1, error2])
    return errors

# Solve for steady state values
bvec_guess = np.array([0.1, 0.1])
b = opt.root(errors, bvec_guess, args = (A, alpha, delta, nvec, beta, sigma))
print(b)
b_ss = b.x
b2_ss, b3_ss = b_ss
K_ss = get_K(b_ss)
L_ss = get_L(nvec)
w_ss = get_w(K_ss, L_ss, (A, alpha))
r_ss = get_r(K_ss, L_ss, (A, alpha, delta))
c1_ss = w_ss * nvec[0] - b2_ss
c2_ss = (1 + r_ss) * b2_ss + w_ss * nvec[1] - b3_ss
c3_ss = (1 + r_ss) * b3_ss + w_ss * nvec[2]
print('\n Savings: \t\t\t {} \n Capital and Labor: \t\t {} \n Wage and Interest rate: \t {} \n Consumption: \t\t\t {}'.format(b_ss, np.array([K_ss, L_ss]), np.array([w_ss, r_ss]), np.array([c1_ss, c2_ss, c3_ss])))


# TPI params
T = 30
max_iter = 300
tol = 1e-9
xi = 0.2

# Initial guess for capital stock
b1vec = np.array([0.8 * b2_ss, 1.1 * b3_ss])
K1 = get_K(b1vec)
Kpath_old = np.zeros(T + 1)
Kpath_old[:-1] = np.linspace(K1, K_ss, T)
Kpath_old[-1] = K_ss

# Time Path Iteration
# Euler function error assuming b2 is known
def get_error(b3, *args):
    nvec, beta, sigma, b2, w_path, r_path = args
    n2, n3 = nvec
    w1, w2 = w_path
    r1, r2 = r_path
    c2 = (1 + r1) * b2 + w1 * n2 - b3
    c3 = (1 + r2) * b3 + w2 * n3
    muc2 = c2 ** (-sigma)
    muc3 = c3 ** (-sigma)
    error = muc2 - beta * (1 + r2) * muc3
    return error

# Euler function error
def get_errors(bvec, *args):
    b2, b3 = bvec
    nvec, beta, sigma, w_path, r_path = args
    w1, w2, w3 = w_path
    r1, r2 = r_path
    c1 = w1 * nvec[0] - b2
    c2 = (1 + r1) * b2 + w2 * nvec[1] - b3
    c3 = (1 + r2) * b3 + w3 * nvec[2]
    muc1 = c1 ** (-sigma)
    muc2 = c2 ** (-sigma)
    muc3 = c3 ** (-sigma)
    error1 = muc1 - beta * (1 + r1) * muc2
    error2 = muc2 - beta * (1 + r2) * muc3
    errors = np.array([error1, error2])
    return errors

abs2 = 1
tpi_iter = 1
while abs2 > tol and tpi_iter < max_iter:
    tpi_iter = tpi_iter + 1
    w_path = get_w(Kpath_old, L_ss, (A, alpha))
    r_path = get_r(Kpath_old, L_ss, (A, alpha, delta))
    # Initialize savings matrix
    b = np.zeros((2, T + 1))
    b[:, 0] = b1vec
    # solve for b32
    b32 = opt.root(get_error, b[1, 0], args=(nvec[1:], beta, sigma, b[0, 0], w_path[:2], r_path[:2]))
    b[1, 1] = b32.x
    for t in range(T - 1):
        bvec_guess = np.array([b[0, t], b[1, t + 1]])
        bt = opt.root(get_errors, bvec_guess, (nvec, beta, sigma, w_path[t : t + 3], r_path[t + 1: t + 3]))
        b[0, t + 1] = bt.x[0]
        b[1, t + 2] = bt.x[1]
    # Calculate the implied capital stock from conjecture and the error
    Kpath_new = b.sum(axis = 0)
    abs2 = ((Kpath_old[:] - Kpath_new[:]) ** 2).sum()
    # Update guess
    Kpath_old = xi * Kpath_new + (1 - xi) * Kpath_old
    print('iteration:', tpi_iter, ' squared distance: ', abs2)

'''
[ 0.07970233  0.07537298  0.07773098  0.07716654  0.0776182   0.07757443
  0.07767366  0.07768355  0.07770935  0.07771693  0.07772447  0.07772758
  0.07772972  0.0777306   0.077731    0.07773101  0.07773085  0.07773058
  0.07773026  0.07772991  0.07772954  0.07772916  0.07772878  0.0777284
  0.07772802  0.07772764  0.07772718  0.07772477  0.07767434]'''

# Calculate T
print(Kpath_old[0:-2])
k_first = [k for k in Kpath_old if abs(k - K_ss) < 1e-4][0]
print(k_first)
T1 = np.where(Kpath_old == k_first)[0][0]
print(T1)
k_second = [k for k in Kpath_old if abs(k - K_ss) < 1e-4][1]
print(k_second)
T2 = np.where(Kpath_old == k_second)[0][0]
print(T2)
plot = True
if plot:
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = "images_1"
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)

    plt.plot (1 + np.arange(T1 + 5), Kpath_old[0:T1 + 5], 'go--')
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Time path of aggregate capital', fontsize=20)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$K$')
    output_path1 = os.path.join(output_dir, 'kplot1')
    plt.savefig(output_path1)
    plt.close()

    plt.plot (1 + np.arange(T2 + 5), Kpath_old[0:T2 + 5], 'go--')
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Time path of aggregate capital', fontsize=20)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$K$')
    output_path2 = os.path.join(output_dir, 'kplot2')
    plt.savefig(output_path2)
    plt.close()
