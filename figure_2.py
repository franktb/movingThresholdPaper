import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib import rc
from scipy.optimize import fsolve
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('text.latex', preamble=r"\usepackage{amsmath}")

fontsize = 17

xstart = 6e-2
xend = 300
yend = 4.1
ystart = 0.0


r_gen = 1.
a = 1.
k = 100.
r_nice = 1.

C = k/a


def richards(x, n):
    return n * r_nice * x * (1 - np.power(x / k, 1 / n))

def richards_Allee(x, n):
    return n * r_nice * (x - a) * (1 - np.power(x / k, 1 / n))

def richards_harvest(x, par, n):
    mu, h = par
    return n * r_nice * x - n * mu * x * np.power(x, 1 / n) - h

def df_harvest(x, par, n):
    mu, h = par
    return r_nice *n - x**(-1 + (1 + n)/n) *mu* (1 + n)

def est_richards_harvest(par, n):
    return[ richards_harvest(a,par,n), richards_harvest(k, par,n) ]


def df_genVolt(n, x):
    return (r_gen * x * (x / k) ** (-1 + 1 / n) * (1 - (x / a) ** (1 / n)) * n) / k + (
            r_gen * x * (x / a) ** (-1 + 1 / n) * (1 - (x / k) ** (1 / n)) * n) / a - r_gen * (
                   1 - (x / a) ** (1 / n)) * (1 - (x / k) ** (1 / n)) * n ** 2


def df_nice(n, x):
    return r_nice * (n - (np.power(x / k, 1. / n) * (-a + x + x * n)) / x)


def ratio_nice(n):
    return (-1 + C) / ((-1 + (1 / C) ** (1 / n)) * C * n)


n = 10
par = fsolve(est_richards_harvest, [0.1,0.1]  ,args=(n,))
n_gen = np.linspace(xstart, xend, 100000)


ratio_harvest = []
for n in n_gen:
    par = fsolve(est_richards_harvest, [0.1, 0.1], args=(n,))
    ratio_harvest.append(np.abs(df_harvest(a, par, n) / df_harvest(k, par, n)))


limit = -(C * np.log(1 / C) / (-1 + C))
ax = plt.gca()

rich_line, = ax.plot(n_gen, (n_gen * r_nice) / (r_nice), lw=2, color="black")
volt_line, = ax.plot(n_gen, -df_genVolt(n_gen, a) / df_genVolt(n_gen, k), lw=2, color="red", zorder=3,clip_on=False)
hack_line, = ax.plot(n_gen, ratio_harvest, color="green", lw=2)


ax.hlines(1, xstart, xend, lw=1, ls="dashed", color="black")
ax.hlines(a / k, xstart, 1, lw=1, ls="dashed", color="red")
ax.vlines(1, ystart, 1, lw=1, ls="dashed", color="black")

ax.add_patch(Rectangle((xstart, 1), 500, 5000, edgecolor='gray', facecolor='blue', alpha=0.1, fill=True, ))

ax.set_xscale("log")
ax.set_ylim([ystart, yend])
ax.set_xlim([xstart, xend])



ytick_loc = [0,  1,2,3,4]
ytick_label = [0,  1,2,3,r""]
ax.set_yticks(ytick_loc)
ax.set_yticklabels(ytick_label, fontsize=fontsize)

ax.set_xlabel(r"$\nu$", fontsize=fontsize)
ax.set_ylabel(r"$\left|\frac{\lambda_{grow}}{\lambda_{sat}}\right|$", rotation=0, fontsize=fontsize)

ax.yaxis.set_label_coords(-0.065, 0.82)
ax.xaxis.set_label_coords(0.98, -0.02)

xtick_loc = [0.1,1, 10 ,100]
xtick_label = [r"$10^{-1}$",r"$10^{0}$",r"$10^{1}$" ,r"$10^{2}$"]
ax.set_xticks(xtick_loc)
ax.set_xticklabels(xtick_label, fontsize=fontsize)



ax.text(18,2.0+0.5, r"{\bf typical}", fontsize=fontsize, color="blue", ha="left",alpha=0.4)#ax.text(100,limit+0.1, r"$\lambda_C$", fontsize=fontsize, color="black", ha="center",)
ax.text(9,1.6+0.5, r"{\bf cancer growth}", fontsize=fontsize, color="blue", ha="left",alpha=0.4)#ax.text(100,limit+0.1, r"$\lambda_C$", fontsize=fontsize, color="black", ha="center",)



ax.legend(
    [
        rich_line,
        volt_line,
        hack_line
    ],
    [
        "Richards",
        "Model 1",
        "Model 2",
    ],
    loc='upper left', fontsize=fontsize-3)



filename = "figure_2"
plt.savefig(filename, bbox_inches='tight', dpi=300)
plt.savefig(filename+".pdf", bbox_inches='tight')

