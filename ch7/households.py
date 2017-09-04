import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import scipy.optimize as opt

def get_c(c1, r, beta, sigma, p): # function for consumption path given initial guess c1
    cvec = np.zeros(p)
    cvec[0] = c1
    cs = c1
    s = 0
    while s < p - 1:
        cvec[s + 1] = cs * (beta * (1 + r[s + 1])) ** (1 / sigma)
        cs = cvec[s + 1]
        s += 1
    return cvec

def MU_c_stitch(cvec, sigma, graph = False):
    epsilon = 1e-4
    muc = cvec ** (-sigma)
    m1 = (-sigma) * epsilon ** (-sigma - 1)
    m2 = epsilon ** (-sigma) - m1 * epsilon
    c_cnstr = cvec < epsilon
    muc[c_cnstr] = m1 * cvec[c_cnstr] + m2

    if graph:
        # Create directory if images directory does not already exist
        cur_path = os.path.split(os.path.abspath(__file__))[0]
        output_fldr = "images"
        output_dir = os.path.join(cur_path, output_fldr)
        if not os.access(output_dir, os.F_OK):
            os.makedirs(output_dir)

        # Plot steady-state consumption and savings distributions
        cvec_CRRA = np.linspace(epsilon / 2, epsilon * 3, 1000)
        MU_CRRA = cvec_CRRA ** (-sigma)
        cvec_stitch = np.linspace(-0.00005, epsilon, 500)
        MU_stitch = m1 * cvec_stitch + m2
        fig, ax = plt.subplots()
        plt.plot(cvec_CRRA, MU_CRRA, ls='solid', label='$u\'(c)$: CRRA')
        plt.plot(cvec_stitch, MU_stitch, ls='dashed', color='red',
                 label='$g\'(c)$: stitched')
        # for the minor ticks, use no labels; default NullFormatter
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Marginal utility of consumption with stitched function', fontsize=15)
        plt.xlabel(r'Consumption $c$')
        plt.ylabel(r'Marginal utility $u\'(c)$')
        plt.xlim((-0.00005, epsilon * 3))
        plt.legend(loc='upper right')
        output_path = os.path.join(output_dir, "MU_c_stitched")
        plt.savefig(output_path)

    return muc

def MU_n_stitch(nvec, params, graph=False):
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

    if graph:
        cur_path = os.path.split(os.path.abspath(__file__))[0]
        output_fldr = "images"
        output_dir = os.path.join(cur_path, output_fldr)
        if not os.access(output_dir, os.F_OK):
            os.makedirs(output_dir)

        # Plot steady-state consumption and savings distributions
        nvec_ellip = np.linspace(epsilon_lb / 2, epsilon_ub +
                                 ((l_ub - epsilon_ub) / 5), 1000)
        MU_ellip = ((b / l_ub) * ((nvec_ellip / l_ub) ** (upsilon - 1)) * \
                    ((1 - ((nvec_ellip / l_ub) ** upsilon)) ** ((1 - upsilon) / upsilon)))
        n_stitch_low = np.linspace(-0.05, epsilon_lb, 500)
        MU_stitch_low = m1 * n_stitch_low + m2
        n_stitch_high = np.linspace(epsilon_ub, l_ub + 0.000005, 500)
        MU_stitch_high = q1 * n_stitch_high + q2

        fig, ax = plt.subplots()
        plt.plot(nvec_ellip, MU_ellip, ls='solid', color='black', label='$v\'(n)$: Elliptical')
        plt.plot(n_stitch_low, MU_stitch_low, ls='dashed', color='red', label='$g\'(n)$: low stitched')
        plt.plot(n_stitch_high, MU_stitch_high, ls='dotted', color='blue', label='$g\'(n)$: high stitched')
        # for the minor ticks, use no labels; default NullFormatter
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Marginal disutility of labor with stitched function', fontsize=15)
        plt.xlabel(r'Labor $n$')
        plt.ylabel(r'Marginal disutility $v\'(n)$')
        plt.xlim((-0.05, l_tilde + 0.01))
        # plt.ylim((-1.0, 1.15 * (b_ss.max())))
        plt.legend(loc='upper left')
        output_path = os.path.join(output_dir, "MU_n_stitched")
        plt.savefig(output_path)

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

def get_n_errors(nvec, *args): # function for calculating intratemporal eulaer error
    cvec, sigma, l_ub, b, upsilon, w = args
    muc = MU_c_stitch(cvec, sigma)
    mun = MU_n_stitch(nvec, (l_ub, b, upsilon))
    n_errors = w * muc - mun
    return n_errors

def get_n(cvec, sigma, l_ub, b, upsilon, w, p): # function for labor supply, calculated from intratemporal euler, given path of lifetime consumption
    n_args = (cvec, sigma, l_ub, b, upsilon, w)
    n_guess = 0.5 * l_ub * np.ones(p)
    result = opt.root(get_n_errors, n_guess, args = (n_args), method = 'lm')
    if result.success:
        nvec = result.x
    else:
        raise ValueError("failed to find an appropriate labor decision")
    return nvec

def get_b_last(c1, *args): # function for last-period savings, given intial guess c1
    r, w, beta, sigma, l_ub, b, upsilon, p, bs = args
    cvec = get_c(c1, r, beta, sigma, p)
    nvec = get_n(cvec, sigma, l_ub, b, upsilon, w, p)
    bvec = get_b(cvec, nvec, r, w, p, bs)
    b_last = (1 + r[-1]) * bvec[-1] + w[-1] * nvec[-1] - cvec[-1]
    return b_last
