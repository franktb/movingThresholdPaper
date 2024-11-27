import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import binom
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('text.latex', preamble=r"\usepackage{siunitx}")


fontsize = 21

n = 10e9


p_low=  2.73930502e-05
p_up =  2.74086039e-05




fig, ax = plt.subplots(1, 2, figsize=(20, 7), gridspec_kw={'width_ratios': [1, 2], "wspace": 0.1})
x_low = 271900
x_up = 276100

year = 365
age = 80
T = age * year
p_inc = (p_up - p_low) / T

c_t = []
e_t = []

TT = np.arange(0, T, 1)

for i in range(0, T):
    pp = p_low + i * p_inc

    c_t.append(np.random.poisson(n * pp))
    e_t.append(n * pp)



colors = [(0, 0, 0), (1, 0, 0)]
cm = LinearSegmentedColormap.from_list("Custom", colors)
ax[1].scatter(TT,c_t,c=TT, cmap=cm, marker=',',lw=1,s=2, alpha=0.5)


ax[1].plot(TT, e_t, color="blue", zorder=5, lw=3)
ax[1].set_xlim([0, T])

xtick_loc = [0, 10 * year, 20 * year, 30 * year, 40 * year, 50 * year, 60 * year, 70 * year, 80 * year]
xtick_label = [r"$0$", r"$10$", r"$20$", r"30", r"$40$", r"$50$", r"$60$", r"$70$",""]
ax[1].set_xticks(xtick_loc)
ax[1].set_xticklabels(xtick_label, fontsize=fontsize)

ax[1].set_ylabel(r"$\times 10^{3}\,m(t)\, [cells/day]$", fontsize=fontsize)
ax[1].set_xlabel(r"$t\,[years]$", fontsize=fontsize)

ax[1].xaxis.set_label_coords(0.953, -0.014)
ax[1].yaxis.set_label_coords(-0.015, 0.76)


ytick_loc = [2.72e5, 2.73e5, 2.74e5, 2.75e5,  2.76e5    ]
ytick_label = [r"$272$", r"$273$", r"", r"", r""     ]
ax[1].set_yticks(ytick_loc)
ax[1].set_yticklabels(ytick_label, fontsize=fontsize)
ax[1].set_ylim([x_low, x_up])



x = np.arange(x_low, x_up)

p_low_pmf, = ax[0].plot(x, binom.pmf(x, n, p_low), 'black', lw=2)
p_up_pmf, = ax[0].plot(x, binom.pmf(x, n, p_up), 'red', lw=2)



xtick_loc = [2.72e5, 2.73e5, 2.74e5, 2.75e5,2.76e5  ]
xtick_label = [r"$272$", r"$273$", r"", r"",r""]
ax[0].set_xticks(xtick_loc)
ax[0].set_xticklabels(xtick_label, fontsize=fontsize)

ytick_loc = [0, 0.0002, 0.0004,0.0006,0.0008]
ytick_label = [r"$0$", r"$2$", r"$4$", r"",r""]
ax[0].set_yticks(ytick_loc)
ax[0].set_yticklabels(ytick_label, fontsize=fontsize)

ax[0].set_xlabel(r"$\times 10^{3}\,m\, [cells/day]$", fontsize=fontsize)
ax[0].set_ylabel(r"$\times 10^{-4}\,P(m;n,p)$", fontsize=fontsize)


ax[0].xaxis.set_label_coords(0.79, -0.014)
ax[0].yaxis.set_label_coords(-0.02, 0.8)


ax[0].vlines(n*p_low,0,binom.pmf(int((n+1)*p_low), n, p_low), ls="dotted", color="black")
ax[0].vlines(n*p_up,0,binom.pmf(int((n+1)*p_up), n, p_up), ls="dotted", color="red")

ax[0].set_xlim([x_low, x_up])
ax[0].set_ylim([0, 0.0008])


ax[0].text(271700,0.000825, r"$(a)$", fontsize=fontsize, ha="center")
ax[0].text(276500,0.000825, r"$(b)$", fontsize=fontsize, ha="center")

filename = "sup_figure_1"
plt.savefig(filename, bbox_inches='tight', dpi=300)
plt.savefig(filename + ".pdf", bbox_inches='tight')