from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
from lmfit import minimize, Parameters, report_fit, fit_report
from scipy.optimize import fsolve
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('text.latex', preamble=r"\usepackage{amsmath}\usepackage{siunitx}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.), gridspec_kw={'width_ratios': [1, 1], "wspace": 0.1})

fontsize = 12
xleft = 0
xright = 45

data_avg = np.array([[0.5, 105, 200, 385, 750, 1627, 2380, 2680]]).T

raw_data = np.loadtxt("LM2-4LUC.txt", delimiter="\t")
data_var_lookup = np.zeros((raw_data.shape[0], 6))

t_arr = []
tmice_arr = []
mean_arr = []
std_arr = []
var_arr = []

j = 0
for i in range(65):
    index = np.array(raw_data[np.where(raw_data[:, 0] == i), 1].ravel(), dtype=np.int32)
    val = raw_data[np.where(raw_data[:, 0] == i), 2].ravel() * 1e6
    data_plot_cloud, = ax1.plot(index, val, 'o', color="gray", alpha=0.2, zorder=5)

    search_filter = np.where(raw_data[:, 1] == i)
    data_stack = raw_data[search_filter, 2].reshape(-1) * 1e6

    if data_stack.size > 0:
        t_arr.append(i)
        mean_arr.append(np.mean(data_stack))
        stack_size = data_stack.size
        data_var_lookup[j:j + stack_size, 0] = np.full(stack_size, i)  # insert t_i of datapoint
        data_var_lookup[j:j + stack_size, 1] = data_stack  # insert x_i of datapoint
        data_var_lookup[j:j + stack_size, 2] = np.full(stack_size,
                                                       np.std(data_stack, axis=0))  # insert (sigma_i)^2 of datapoint
        data_var_lookup[j:j + stack_size, 3] = np.full(stack_size,
                                                       np.mean(data_stack))

        data_var_lookup[j:j + stack_size, 4] = np.full(stack_size,
                                                       np.min(data_stack, axis=0))

        data_var_lookup[j:j + stack_size, 5] = np.full(stack_size,
                                                       np.max(data_stack, axis=0))

        j = j + stack_size

avg, = ax1.plot(t_arr, mean_arr, '^', color="black", lw=2, alpha=1.0, zorder=5)
ax1.plot(0, 1, '^', color="black", lw=2, alpha=1.0, zorder=5, clip_on=False)



def visual_cloud(ax):
    """
    bloat code for nice plots
    :return:
    """
    xleft = 0
    xright = 45  # 39
    ax.set_xlim([xleft, xright])
    ax.set_ylim([0.0, 3.5e9])

    ax.set_xlabel(r"$t\,[days]$", fontsize=fontsize + 1)
    ax.set_ylabel(r"$\times10^{9}\,x(t)\,[cells]$", fontsize=fontsize + 1, rotation=90)
    ax.yaxis.set_label_coords(-0.06, 0.5)
    ax.xaxis.set_label_coords(0.939, -0.025)

    xtick_loc = np.array([0, 7, 14, 21, 28, 35, 42])
    xtick_label = np.array([0, 7, 14, 21, 28, 35, r""])
    ax.set_xticks(xtick_loc)
    ax.set_xticklabels(xtick_label, fontsize=fontsize + 1)

    ytick_loc = [0, 1e9, 2e9, 3e9]
    ytick_label = [0, r"$1$", r"$2$", r"$3$"]
    ax.set_yticks(ytick_loc)
    ax.set_yticklabels(ytick_label, fontsize=fontsize + 1)


def visual_avg(ax):
    """
    bloat code for nice plots
    :return:
    """
    t = np.arange(15, 34, 3)
    t = np.insert(t, 0, 0.0).flatten()
    data = data_avg
    ax.set_clip_on(False)
    global data_plot
    data_plot, = ax.plot(t, data * 1e6, '^', color="black", lw=2, alpha=1.0, zorder=3, clip_on=False, )

    ax.set_xlim([0, 45])
    ax.set_ylim([0.0, 3.5e9])

    ax.set_xlabel(r"$t\,[days]$", fontsize=fontsize + 1)
    ax.xaxis.set_label_coords(0.94, -0.025)

    xtick_loc = np.array([0, 7, 14, 21, 28, 35, 42])
    xtick_label = np.array([0, 7, 14, 21, 28, 35, r""])
    ax.set_xticks(xtick_loc)
    ax.set_xticklabels(xtick_label, fontsize=fontsize + 1)

    ytick_loc = [0, 1e9, 2e9, 3e9]
    ytick_label = [0, r"$1$", r"$2$", r"$3$"]
    ax.set_yticks(ytick_loc)
    ax.set_yticklabels(ytick_label, fontsize=fontsize + 1)


def richard_nice(t, xs, ps):
    """
    Model to be fitted (Richard with harvesting rate h)
    """
    try:
        r = ps['r'].value
        A = ps['A'].value
        K = ps['K'].value
        n = ps['n'].value
    except:
        r, A, K, n = ps
    x = xs
    return n * r * (x - A) * (1 - np.power(x / K, 1 / n))


def richard_harvest(t, xs, ps):
    """
    Model to be fitted (Richard with harvesting rate h)
    """
    try:
        r = ps['r'].value
        mu = ps['mu'].value
        h = ps['h'].value
        n = ps['n'].value
    except:
        r, mu, q, n = ps
    x = xs
    return n * r * x - n * mu * x * np.power(x, 1 / n) - h


def volterra_allee(t, xs, ps):
    """
    Model to be fitted (Volterra)
    """
    try:
        r = ps['r'].value
        A = ps['A'].value
        K = ps['K'].value
    except:
        r = ps
    x = xs
    return r * x * (1 - x / K) * (1 - x / A)


def volterra_allee_gen(t, xs, ps):
    """
    Model to be fitted (Volterra)
    """
    try:
        r = ps['r'].value
        A = ps['A'].value
        K = ps['K'].value
        n = ps["n"].value
    except:
        r = ps
    x = xs
    return r * x * np.square(n) * (1 - np.power(x / K, 1. / n)) * (1 - np.power(x / A, 1. / n))


def gompertz(t, xs, ps):
    """
    Model to be fitted (gompertz)
    """
    try:
        r = ps['r'].value
        K = ps['K'].value
    except:
        r = ps
    x = xs
    if xs < 0:
        return 0
    else:
        return r * x * np.log(K / x)


def g(t, x0, ps, f):
    """
    Solution to the ODE x'(t) = f(t,x) with initial condition x(0) = x0
    """
    x_temp = solve_ivp(f, t_span=(t[0], t[-1]), t_eval=t, y0=(x0,), args=(ps,), method="BDF").y
    return x_temp.ravel()


def residual_multi(ps, ts, data, model):
    x0 = ps['x0'].value
    model = g(ts, x0, ps, model)
    res = []

    for i in range(data.shape[0]):
        model_val = np.take(model, np.array(data[i][0], dtype=np.int32))

        diff = (data[i, 1] - model_val)  # / data[i,3]
        res.append(diff)

    return np.array(res).ravel()


def residual_avg(ps, ts, data, model):
    """
    Residual function
    """
    x0 = ps['x0'].value
    model = g(ts, x0, ps, model).ravel()
    data = data.ravel()
    return (model - data).ravel()

models = [volterra_allee,
          volterra_allee_gen,
          richard_nice,
          richard_harvest,
          ]

for model in models:
    params = Parameters()
    params.add('x0', value=1e6, vary=False)

    if model.__name__ == "volterra_allee":
        params.add('r', value=-0.005, min=-0.1, max=0)
        params.add('A', value=1e4, min=1e0, max=5e5)
        params.add('K', value=2e9, min=0.5e9, max=3e9)
    elif model.__name__ == "volterra_allee_gen":
        params.add('r', value=-0.00001, min=-0.1, max=0, vary=True)
        params.add('A', value=1e3, min=1e1, max=1e5, vary=True)
        params.add('K', value=3e9, min=0.5e9, max=4e9, vary=True)
        params.add('n', value=2760, min=1000, max=100000, vary=True)
    elif model.__name__ == "richard_nice":
        params.add('r', value=0.07, min=.001, max=1, vary=True)
        params.add('A', value=2.9e4, min=1e1, max=5e5, vary=True)
        params.add('K', value=4.1e9, min=2.0e9, max=6e9, vary=True)
        params.add('n', value=1e5, min=1000, max=1e5, vary=True)
    elif model.__name__ == "richard_harvest":
        params.add("r", value=rr, min=rr - 0.1, max=rr + 0.1, vary=True)
        params.add('mu', value=muu, min=muu - 0.1, max=muu + 0.1, vary=True)
        params.add('h', value=hh + 100, min=hh - 100, max=hh + 1000, vary=True)
        params.add('n', value=nn, min=1000, max=1e6, vary=True)

    t = np.arange(0, 39, 1)
    result = minimize(residual_multi, params, args=(t, data_var_lookup, model), method='leastsq')

    tt = np.linspace(0, 45, 1000000)
    if model.__name__ == "volterra_allee":
        volt_traj, = ax1.plot(tt, g(tt, result.params["x0"].value, result.params, model),
                              color="red", lw=2, zorder=4, label=r"Model $I$")
        K_va_line = ax1.hlines(result.params["K"], xleft, 45, color="red", lw=1, ls="dashed", label="$K_{VGM}$")
        ax1.text(2.5, 0.8e9, r"$K$ for $\nu=1$", fontsize=fontsize, color="red", va="center")

    if model.__name__ == "volterra_allee_gen":
        volt_traj_gen, = ax1.plot(tt, g(tt, result.params["x0"].value, result.params, model),
                                  color="red", lw=2, zorder=6, label=r"Model $I$")
        K_va_line_gen = ax1.hlines(result.params["K"], xleft, 45, color="red", zorder=5, lw=1, ls="dashed",
                                   label="$K_{VGM}$")
        ax1.text(2.5, 2.05e9, r"$K$ for $\nu={:,}".format(int(np.round(result.params["n"]))) + "$", fontsize=fontsize,
                 color="red",
                 va="center")

    if model.__name__ == "richard_nice":
        rr = result.params["r"].value
        muu = result.params["r"].value / np.power(result.params["K"].value, 1 / result.params["n"].value)
        hh = (result.params["A"].value * result.params["n"].value * result.params["r"].value) / (
                muu * result.params["n"].value + 1)
        nn = result.params["n"].value


    if model.__name__ == "richard_harvest":
        rich_traj, = ax1.plot(tt, g(tt, result.params["x0"].value, result.params, model),
                              color="green", lw=2, zorder=10, label=r"$RGM$")
        ax1.text(2.5, 2.7e9, r"$K$ for $\nu={:,}".format(int(np.round(result.params["n"]))) + "$", fontsize=fontsize,
                 color="green", va="center")


        def ff(x):
            return result.params["n"] * result.params["r"] * x \
                - result.params["n"] * result.params["mu"] * x * np.power(x, 1 / result.params["n"]) \
                - result.params["h"]


        K_rh = fsolve(ff, 3e9)
        K_rh_line = ax1.hlines(K_rh, xleft, 45, color="green", lw=1, ls="dashed", label="$K_{RGM}$")

        A_rh = fsolve(ff, 3e1)


    if model.__name__ == "gompertz":
        report_fit(result)
        ax1.plot(tt, g(tt, result.params["x0"].value, result.params, model),
                 color="orange", lw=2, zorder=6, label=r"Model $I$")

        ax1.hlines(result.params["K"], 0, 40, color="orange", lw=1, zorder=6, ls="dotted",
                   label="$K_{VGM}$")

        ax1.hlines(2.6e9, 0, 40, color="black", lw=1, zorder=6, ls="dotted",
                   label="$K_{VGM}$")


models = [volterra_allee,
          volterra_allee_gen,
          richard_nice,
          richard_harvest
          ]

for model in models:
    params = Parameters()
    params.add('x0', value=0.5e6, vary=False)

    if model.__name__ == "volterra_allee":
        params.add('r', value=-0.00257168, min=-1.1, max=0)
        params.add('A', value=33492.5614, min=1e2, max=1e5, vary=True)
        params.add('K', value=2.9e9, min=2.0e9, max=3.7e9, vary=True)
    elif model.__name__ == "volterra_allee_gen":
        params.add('r', value=-0.0002, min=-1.0, max=0, vary=True)
        params.add('A', value=1e4, min=1e0, max=1e5, vary=True)
        params.add('K', value=3.3e9, min=2.5e9, max=3.8e9, vary=True)
        params.add('n', value=3930.516, min=1, max=1e6, vary=True)
    elif model.__name__ == "richard_nice":
        params.add('r', value=0.2, min=.001, max=1, vary=True)
        params.add('A', value=1.e2, min=1e1, max=3e5, vary=True)
        params.add('K', value=3.1e9, min=2e9, max=4e9, vary=True)
        params.add('n', value=2.0, min=1.0, max=1e5, vary=True)
    elif model.__name__ == "richard_harvest":
        params.add("r", value=rr, min=rr - 0.001, max=rr + 0.001, vary=True)
        params.add('mu', value=muu, min=muu - 0.001, max=muu + 0.001, vary=True)
        params.add('h', value=hh, min=hh - 100, max=hh + 100, vary=True)
        params.add('n', value=nn, min=1.0, max=2e1, vary=True)

    t = np.arange(15, 34, 3)
    t = np.insert(t, 0, 0.0).flatten()

    result = minimize(residual_avg, params, args=(t, data_avg * 1e6, model), method='leastsq')

    tt = np.linspace(0, 45, 1000000)
    if model.__name__ == "volterra_allee":
        volt_traj, = ax2.plot(tt, g(tt, result.params["x0"].value, result.params, model),
                              color="red", lw=2, zorder=4, label=r"Model $I$")
        K_va_line = ax2.hlines(result.params["K"], xleft, 45, color="red", lw=1, ls="dashed", label="$K_{VGM}$")
        ax2.text(2.5, 2.36e9, r"$K$ for $\nu=1$", fontsize=fontsize, color="red",
                 va="center")

    if model.__name__ == "volterra_allee_gen":
        volt_traj_gen, = ax2.plot(tt, g(tt, result.params["x0"].value, result.params, model),
                                  color="red", lw=2, zorder=6, label=r"Model $I$")
        K_va_line_gen = ax2.hlines(result.params["K"], xleft, 45, color="red", zorder=5, lw=1, ls="dashed",
                                   label="$K_{VGM}$")
        ax2.text(2.5, 3.15e9, r"$K$ for $\nu={:,}".format(int(np.round(result.params["n"]))) + "$", fontsize=fontsize,
                 color="red",
                 va="center")

    if model.__name__ == "richard_nice":
        rr = result.params["r"].value
        muu = result.params["r"].value / np.power(result.params["K"].value, 1 / result.params["n"].value)
        hh = (result.params["A"].value * result.params["n"].value * result.params["r"].value) / (
                muu * result.params["n"].value + 1)
        nn = result.params["n"].value


    if model.__name__ == "richard_harvest":
        rich_traj, = ax2.plot(tt, g(tt, result.params["x0"].value, result.params, model),
                              color="green", lw=2, zorder=10, label=r"$RGM$")

        ax2.text(2.4, 2.85e9, r"$K$ for $\nu={:,}".format(int(np.round(result.params["n"]))) + "$", fontsize=fontsize,
                 color="green",
                 va="center")


        def ff(x):
            return result.params["n"] * result.params["r"] * x \
                - result.params["n"] * result.params["mu"] * x * np.power(x, 1 / result.params["n"]) \
                - result.params["h"]


        K_rh = fsolve(ff, 3e9)
        A_rh = fsolve(ff, 3e1)
        ax2.hlines(K_rh, xleft, 45, color="green", lw=1, ls="dashed")


visual_cloud(ax1)
visual_avg(ax2)

ax1.text(-5.0, 3.7e9, r"$(a)$", fontsize=fontsize)
ax2.text(-2.6, 3.7e9, r"$(b)$", fontsize=fontsize)

filename = r"figure_5"
plt.savefig(filename, bbox_inches='tight', dpi=300)
plt.savefig(filename + ".pdf", bbox_inches='tight')