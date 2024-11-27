import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy import integrate
from scipy.optimize import fsolve

from matplotlib.patches import Rectangle
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('text.latex', preamble=r"\usepackage{siunitx}")


x0 = (4. / 3.) * np.pi * np.power(0.5, 3) * 1e9
print("x0 cells", np.format_float_scientific(x0))
x1 = (4. / 3.) * np.pi * np.power(1.2, 3) * 1e9
print("x1 cells", np.format_float_scientific(x1))

K = 3.1e12
A = 10000.
t_diff = 182.
year = 365.

fontsize = 13
ymax = 5e12
ymin =3e8

xright = 3.05 * year

step_size = 0.01


r_volt = -9.73939594e-08#-1.00145201e-07

def visual(ax):
    """
    bloat code for nice plots
    :return:
    """
    ax.set_yscale('log')
    ax.set_ylim([ymin,ymax])

    ax.set_xlim([0, xright])
    #ax.hlines(x0, 0, xright, color="gray", ls="dashed")
    ax.hlines(1e12, 0, xright, color="gray", ls="dotted", lw = 1)

    #ax.text(1050, 0.55e12, r"Lethal threshold", fontsize=fontsize, ha="left", c="black")
    #ax.text(950, 0.6e9, r"Initial tumour size", fontsize=fontsize, ha="left", c="black")
    #ax.text(950, 4e12, r"$K=\SI{3.1e12}{}$", fontsize=fontsize, ha="left", c="black")

    xtick_loc = [0, 0.5*year,1.5* year,1 * year,2 * year, 2.5*year]  # ,3.5*year]#,5*year,6*year,7*year,8*year,9*year,10*year]
    xtick_label = [0,0.5, 1.5,1, 2,r""]  # ,r"$t[yr]$"]#,5,6,7,8,r"$t[a]$",10]
    ax.set_xticks(xtick_loc)
    ax.set_xticklabels( xtick_label, fontsize=fontsize + 1)

    ytick_loc = [ 1e9, 1e10, 1e11, 1e12, ]
    ytick_label = [
                   "$10^9$",
                   "$10^{10}$",
                   "$10^{11}$",
                   r"",
                   ]
    ax.set_yticks(ytick_loc)
    ax.set_yticklabels(ytick_label, fontsize=fontsize + 1)

    ax.set_xlabel(r"$t\,[years]$", fontsize=fontsize + 1)
    ax.set_ylabel(r"$x(t)\,[cells]$", fontsize=fontsize + 1)

    #ax.yaxis.set_label_coords(-0.03, 0.7)
    #ax.xaxis.set_label_coords(0.95, -0.015)

    ax.yaxis.set_label_coords(-0.03, 0.87)
    ax.xaxis.set_label_coords(0.93, -0.016)



def volterra(t, x):
    return r_volt * x * (1 - x / K) * (1 - x / A)


def est_gompertz(r):
    return t_diff - quad(lambda x: 1 / (r * x * np.log(x / K ) * np.log(x / A)), x0, x1)[0]




def gompertz(t,x,r):
    return r * x * np.log(x / K ) * np.log(x / A)


def richard_harvest(t, x, n, r, mu, h):
    return n * r * x - n * mu * x * np.power(x, 1 / n) - h

def est_richard_harvest(par, n):
    r, mu, h = par
    return [t_diff - quad(lambda x: 1 / (n * r * x - n * mu * x * np.power(x, 1 / n) - h), x0, x1)[0],
            n * r * A - n * mu * A * np.power(A, 1 / n) - h,
            n * r * K - n * mu * K * np.power(K, 1 / n) - h]



def est_par_survival(par, t_death):
    r, mu, h, n = par

    return [t_diff - quad(lambda x: 1 / (n * r * x - n * mu * x * np.power(x, 1 / n) - h), x0, x1)[0],
            t_death*year - quad(lambda x: 1 / (n * r * x - n * mu * x * np.power(x, 1 / n) - h), x0, 1e12)[0],
            n * r * A - n * mu * A * np.power(A, 1 / n) - h,
            n * r * K - n * mu * K * np.power(K, 1 / n) - h]



def est_nu_richard(n, t_death):
    global r, mu, h
    h = A * 0.002867
    mu = (0.002867) / np.power(K, 1. / n)
    est = fsolve(est_richard_harvest, np.array([0.002867, mu, h],dtype=float), args=(np.array(n,dtype=float),))
    r, mu, h = est[0], est[1], est[2]
    #print("")
    #print(f"r {r:6f} mu {mu:6f} h {h:6f} n {n[0]:6f}")
    #print(r,mu,h)
    sol = integrate.solve_ivp(richard_harvest, t_span=(0, 5 * year), y0=np.array([x0]), args=(n, r, mu, h), max_step=0.1, method='LSODA')
    #np.savetxt("debug.txt", sol.y.T)

    for index, xx in np.ndenumerate(sol.y.T[:, 0]):
         if xx>1e12:
             t_lethal = sol.t[index]
             break

    #print(f"r {r:6f} mu {mu:6f} h {h:6f} n {n[0]:6f}")
    #print(t_lethal/year- t_death)
    return t_lethal/year- t_death






ax = plt.gca()



r_gompertz = fsolve(est_gompertz, -0.005)
print("r Gompertz", r_gompertz)

sol_gom = integrate.solve_ivp(gompertz, t_span=(0, 4 * year), y0=np.array([x0]),  args=(r_gompertz[0],), max_step=step_size,
                          method='LSODA')


t_lethal = -1
for index, xx in np.ndenumerate(sol_gom.y.T[:, 0]):
     if xx>1e12:
         t_lethal = sol_gom.t[index]
         break
print("lethal gompertz", t_lethal/year, t_lethal)
t_lethal_gom = ax.vlines(t_lethal, ymin, ymax, ls="dashed", lw=1, color="red")

gom_traj, = ax.plot(sol_gom.t, sol_gom.y.T[:, 0], color="red", lw=2)

t_lethal_volt_right = t_lethal



n_large = 1.

h_large = A * 0.0002867
mu_large = (0.0002867) / np.power(K, 1. / n_large)
est = fsolve(est_richard_harvest, np.array([0.0002867, mu_large, h_large],dtype=float), args=(np.array(n_large,dtype=float),))

print(est)

sol_large = integrate.solve_ivp(richard_harvest, t_span=(0, 4 * year), y0=np.array([x0]),  args=(n_large, est[0], est[1], est[2]), max_step=step_size,
                          method='LSODA')

for index, xx in np.ndenumerate(sol_large.y.T[:, 0]):
     if xx>1e12:
         print("huhu")
         t_lethal = sol_large.t[index]
         break
t_large = ax.vlines(t_lethal, ymin, ymax, ls="dashed", lw=1, color="green")
print("t_large death 1", t_lethal/year,  t_lethal)

t_lethal_harvest_left = t_lethal


ax.vlines(t_lethal, ymin, ymax, ls="dashed", lw=1, color="green")
ax.plot(sol_large.t, sol_large.y.T[:, 0], color="green", lw=2)




n_large = 10000000.

h_large = A * 0.0002867
mu_large = (0.0002867) / np.power(K, 1. / n_large)
est = fsolve(est_richard_harvest, np.array([0.0002867, mu_large, h_large],dtype=float), args=(np.array(n_large,dtype=float),))

print("est100000 ", est)

sol_large = integrate.solve_ivp(richard_harvest, t_span=(0, 4 * year), y0=np.array([x0]),  args=(n_large, est[0], est[1], est[2]), max_step=step_size,
                          method='LSODA')

ax.plot(sol_large.t, sol_large.y.T[:, 0], color="green", lw=2)


for index, xx in np.ndenumerate(sol_large.y.T[:, 0]):
     if xx>1e12:
         print("huhu")
         t_lethal = sol_large.t[index]
         break
t_large = ax.vlines(t_lethal, ymin, ymax, ls="dashed", lw=1, color="green")
print("t_large death", t_lethal/year, t_lethal)

t_lethal_harvest_right = t_lethal

print(t_lethal_harvest_left/year)
print(t_lethal_harvest_right/year)

#richard_harvest(t, x, n, r, mu, h):



ax.text(2.77*year, 4.e8, r"$\nu\to\infty$", fontsize=fontsize,rotation=90, ha="center",va="bottom", c="green")

ax.text(1.4*year, 4.e8, r"$\nu="+ str(1) +r"$", fontsize=fontsize,rotation=90, ha="center",va="bottom", c="green")




visual(ax)
#ax.plot(sol.t, sol.y.T[:, 0], color="green", lw=2)
#ax.plot(sol1.t, sol1.y.T[:, 0], color="green", lw=2)

data_plot  =ax.scatter([0,t_diff], [x0,x1], color="gray", zorder=5, clip_on=False)











sol_volt= integrate.solve_ivp(volterra, t_span=(0, 4 * year), y0=np.array([x0]), max_step=step_size, method='LSODA')
t_lethal = -1
for index, xx in np.ndenumerate(sol_volt.y.T[:, 0]):
     if xx>1e12:
         t_lethal = sol_volt.t[index]
         break
print("Volterra lethal", t_lethal/year)
print("Volterra lethal", t_lethal)
t_lethal_volt = ax.vlines(t_lethal, ymin, ymax, ls="dashed", lw=1, color="red")
volt_traj, = ax.plot(sol_volt.t, sol_volt.y.T[:, 0], color="red", lw=2)
t_lethal_volt_left = t_lethal


ax.text(0.475*year, 4.e8, r"$\nu=1$", fontsize=fontsize,rotation=90, ha="center",va="bottom", c="red")
ax.text(2.15*year, 4.e8, r"$\nu\to\infty$", fontsize=fontsize,rotation=90, ha="center",va="bottom", c="red")



K_line = ax.hlines(K, 0, xright, color="gray", ls="dashed")
#ax.text(((2.3+2.72)/2)*year, 5e10, r"mean survival time", color="gray",fontsize=fontsize+1, rotation=90, alpha =0.8, ha="center", va="center")

ax.add_patch(Rectangle((t_lethal_harvest_left,0), t_lethal_harvest_right-t_lethal_harvest_left, ymax, edgecolor = 'none', facecolor = 'green',alpha=0.15,fill=True,zorder=1))
ax.add_patch(Rectangle((t_lethal_volt_left,0), t_lethal_volt_right-t_lethal_volt_left, ymax, edgecolor = 'none', facecolor = 'red',alpha=0.15,fill=True,zorder=1))

print("volt", t_lethal_volt_left)
print(t_lethal_volt_left-t_lethal_volt_right)

filename = r"sup_figure_3"
plt.savefig(filename, bbox_inches='tight', dpi =300)
plt.savefig(filename+".pdf", bbox_inches='tight')

