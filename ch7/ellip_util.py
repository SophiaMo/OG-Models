import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Ellipse
import os

def get_sumsq(ellip_params, *args):
    b, upsilon = params
    theta, chi, l_ub, l = args
    mu_cfe= theta * (l ** (1 / chi))
    mu_ellip = ((b / l_tilde) * ((l / l_ub) ** (upsilon - 1)) * ((1 - ((l / l_ub) ** upsilon)) ** \
               ((1 - upsilon) / upsilon)))
    sumsq = ((mu_cfe - mu_ellip) ** 2).sum()

    return sumsq

def fit_ellip(ellip_init, cfe_params, l_ub, graph = False):
    theta, chi = cfe_params
    l_min = 0.02
    l_max = 0.98 * l_ub
    labor_sup = np.linspace(l_min, l_max, 1000)
    args = (theta, chi, l_ub, l)
    bnds = ((1e-10, None), (1 + 1e-10, None))
    ellip_results = opt.minimize(get_sumsq, ellip_init, args=(args), bounds=bnds)
    if ellip_result.success:
        b, upsilon = ellip_params.x
        sumsq = ellip_params.fun
    else:
        raise ValueError("Failed to fit an ellipse for the given Frisch elasticity")
    if graph:
        cur_path = os.path.split(os.path.abspath(__file__))[0]
        output_fldr = "images"
        output_dir = os.path.join(cur_path, output_fldr)
        if not os.access(output_dir, os.F_OK):
            os.makedirs(output_dir)

        # Plot steady-state consumption and savings distributions
        mu_ellip = ((b / l_ub) * ((l / l_ub) ** (upsilon - 1)) * ((1 - ((l / l_ub) ** upsilon)) ** \
                   ((1 - upsilon) / upsilon)))
        mu_cfe = chi * (l ** (1 / theta))
        fig, ax = plt.subplots()
        plt.plot(l, mu_ellip, label='Elliptical MU')
        plt.plot(l, mu_cfe, label='CFE MU')
        # for the minor ticks, use no labels; default NullFormatter
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('CFE marginal utility and fitted elliptical utility', fontsize=20)
        plt.xlabel(r'Labor supply $n_{s,t}$')
        plt.ylabel(r'Marginal disutility')
        plt.xlim((0, l_ub))
        # plt.ylim((-1.0, 1.15 * (b_ss.max())))
        plt.legend(loc='upper left')
        output_path = os.path.join(output_dir, 'Elliptical Utility')
        plt.savefig(output_path)
        # plt.show()
        plt.close()
    return b, upsilon
