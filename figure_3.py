from scipy import integrate
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
fontsize = 17

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)



A = 1
r = -0.12
x0=2.0


def f(x):
    return r * x * (1 - x / K) * (1 - x / A)


def x_t_v2(t):
    return (A* (-1 + np.exp(t/r)) *x0)/(A + (-1 + np.exp(t/r)) *x0)


def x_t_v3(t):
   return (A* np.exp(r* t)* x0) / (A - x0 + np.exp(r* t) *x0)


def cancer(t, x):
    # x, u = xu
    return [f(x)]


def x_t(t):
    return (A*x0)/(A*np.exp(-r*t)- x0)



yup = 2e9
xleft= 0
xright = 12
ylow= 0.7



fig, ax2 = plt.subplots(tight_layout=True)
tt = np.linspace(0,5.77,10000)


K = 10000
sol1 = integrate.solve_ivp(cancer, t_span=(0, 20), y0=np.array([x0]), max_step=0.01,method='LSODA')
ax2.plot(sol1.t, sol1.y.T[:, 0], color="blue", lw=2, zorder=5)
ax2.hlines(K,xright,xleft, color="blue", ls="dashed", lw=1.)


ax2.set_xlim([xleft,xright])
ax2.set_ylim([ylow,yup])

ax2.set_yscale('log')
ax2.hlines(A,xright,xleft, color="black", ls="dashed", lw=1.,clip_on=False)




xtick_loc = np.arange(0,11,2)
xtick_label = xtick_loc   # , r"$t[d]$"]
ax2.set_xticks(xtick_loc)
ax2.set_xticklabels(xtick_label, fontsize=fontsize)

ytick_loc = [1e1,1e3,1e5,1e7]
ytick_label = [r"$10^1$",r"$10^3$",r"$10^5$",r"$10^7$",]  # , r"$t[d]$"]

ax2.set_yticks(ytick_loc)
ax2.set_yticklabels(ytick_label, fontsize=fontsize )

ax2.set_xlabel("$t$", fontsize=fontsize )
ax2.set_ylabel("$x(t)$", fontsize=fontsize , rotation=0)
ax2.yaxis.set_label_coords(-0.04, 0.9)
ax2.xaxis.set_label_coords(0.964, -0.015)



ax2.text(11.,20,r"$K=10^1$",fontsize=fontsize, ha="center", color="green")
ax2.text(11.,20000,r"$K=10^4$",fontsize=fontsize, ha="center", color="blue")
ax2.text(11.,2.0e8,r"$K=10^8$",fontsize=fontsize, ha="center", color="red")
ax2.text(11.,1.4,r"$A=10^0$",fontsize=fontsize, ha="center")


K = 10
sol1 = integrate.solve_ivp(cancer, t_span=(0, 20), y0=np.array([x0]), max_step=0.01,method='LSODA')
ax2.plot(sol1.t, sol1.y.T[:, 0], color="green", lw=2)
ax2.hlines(K,xright,xleft, color="green", ls="dashed", lw=1.)




K = 1e8
sol1 = integrate.solve_ivp(cancer, t_span=(0, 20), y0=np.array([x0]), max_step=0.01,method='LSODA')
ax2.plot(sol1.t, sol1.y.T[:, 0], color="red", lw=2)
ax2.hlines(K,xright,xleft, color="red", ls="dashed", lw=1.)




filename = r"figure_3"
plt.savefig(filename, bbox_inches='tight', dpi =300)
plt.savefig(filename+".pdf")

