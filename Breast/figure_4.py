import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.optimize import fsolve
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from matplotlib.legend_handler import HandlerTuple, HandlerBase

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('text.latex', preamble=r"\usepackage{amsmath}\usepackage{siunitx}")

lw_A = 1.2


class AnyObjectHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        l1 = plt.Line2D([x0, y0 + width], [0.7 * height, 0.7 * height], color='r', lw=lw_A)
        l2 = plt.Line2D([x0, y0 + width], [0.3 * height, 0.3 * height], color='blue', lw=lw_A)
        return [l1, l2]


nu = 10
r = 2.78777026e-03
mu = 1.57079970e-04
h = 2.39320322e+02
n = 10e9

fontsize = 13
year = 365
age = 80
T = age * year

p_low = 2.73940502e-05
p_up = 2.74080039e-05
p_inc = (p_up - p_low) / T


def findA(x, rr, H_eff):
    return nu * r * x - nu * mu * x * np.power(x, 1 / nu) - H_eff + rr


def approx_A(rr, H_eff):
    return (H_eff - rr) / (-mu * (nu + np.log(10000)) + nu * r)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 4.), gridspec_kw={'height_ratios': [1, 1], "hspace": 0.25})

data = np.loadtxt("traj_save.txt", delimiter=' ')
less = data[-1, 0]
data = data[:-1, :]
ax1.plot(data[:, 0], data[:, 1], color="black")

ax1.set_yscale('log')
ax1.set_xlim([0, 80 * year])
ax1.set_ylim([1, 1e13])

xtick_loc = np.arange(0, 82 * year, step=10 * year)
xtick_label = ["0", "10", "20", "30", "40", "50", "60", "", ""]
ax1.set_xticks(xtick_loc)
ax1.set_xticklabels(xtick_label, fontsize=fontsize)

ax1.set_xlabel(r"$t\,[years]$", fontsize=fontsize)
ax1.set_ylabel(r"$x_t\,[cells]$", fontsize=fontsize)

ytick_loc = [1e0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12]
ytick_label = [r"$10^0$", r"", r"$10^4$", r"", r"$10^8$", r"", r"$10^{12}$"]
ax1.set_yticks(ytick_loc)
ax1.set_yticklabels(ytick_label, fontsize=fontsize)

ax1.xaxis.set_label_coords(0.93, -0.04)
ax1.yaxis.set_label_coords(-0.05, 0.5)

zoom_left = 50.75 * year
zoom_right = 51.75 * year
zoom_up = 50000

ax1.add_patch(
    Rectangle((zoom_left, 1), zoom_right - zoom_left, zoom_up, edgecolor='forestgreen', alpha=1., fill=False, zorder=5,
              lw=2))

neg_off=-700
ax2.add_patch(
    Rectangle((zoom_left, neg_off), zoom_right - zoom_left, zoom_up-neg_off, edgecolor='forestgreen', alpha=1., fill=False, zorder=3,
              lw=1.6, clip_on=False))


avg_noise = []


noise_raw = data[:, 2]
noise_avg = np.zeros(noise_raw.shape[0])
ll = int(np.round(less))

i = 12 * year + 1
for i in range(1, T):
    phase = np.remainder(i, np.round(28 - less))
    if int(phase) == 0:
        noise_avg[i:i + 14 - ll] = np.full(14 - ll, np.sum(noise_raw[i:i + 14 - ll], dtype=np.float64) / float(14 - ll))
        noise_avg[i + 14 - ll: i + 28 - ll] = np.full(14, np.sum(noise_raw[i + 14 - ll: i + 28 - ll],
                                                                 dtype=np.float64) / float(14))

A = []
A_approx = []

feq, _, flag, mes = fsolve(findA, 10000, args=(noise_avg[51 * year], data[51 * year, 3]), full_output=1)

for i in range(1, T):
    A_approx.append(approx_A(noise_avg[i], data[i, 3]))
    feq, _, flag, mes = fsolve(findA, 10000, args=(noise_avg[i], data[i, 3]), full_output=1)
    if flag == 1:
        A.append(feq[0])
    else:
        feq, _, flag2, mes2 = fsolve(findA, 1000, args=(noise_avg[i], data[i, 3]), full_output=1)
        if flag2 == 1:
            print(flag, mes)
            A.append(feq[0])
        else:
            A.append(0)

red_line = ax2.stairs(A, data[:, 0], fill=False, alpha=1.0, color="red", zorder=3, lw=lw_A)

blue_line = ax2.stairs(A_approx, data[:, 0], fill=False, alpha=1.0, color="blue", zorder=4, lw=lw_A)

A = np.array(A)

H = []
for i in range(1, T):
    if (i < 12 * year):
        H.append(0000)
        # H.append(40000)
    else:
        phase = np.remainder(i, np.round(28 - less))
        if phase < 14 - less:
            H.append(0000)
            # H.append(40000)
        else:
            H.append(70000)
            # H.append(20000)

ax2.stairs(H, data[:, 0], fill=True, alpha=0.3, color="gray", zorder=1)


ax2.set_xlim([zoom_left, zoom_right])
ax2.set_ylim([1e0, zoom_up])

xtick_loc = [50.75 * year, 51. * year, 51.25 * year, 51.5 * year, 51.75 * year]
xtick_label = ["50.75", "51.0", "51.25", "51.5", "", ]
ax2.set_xticks(xtick_loc)
ax2.set_xticklabels(xtick_label, fontsize=fontsize)

ax2.set_xlabel(r"$t\,[years]$", fontsize=fontsize)


ytick_loc = [10000, 20000, 30000, 40000, 50000]
ytick_label = ["1", "2", "3", "4", "5", ]
ax2.set_yticks(ytick_loc)
ax2.set_yticklabels(ytick_label, fontsize=fontsize)

ax2.set_ylabel(r"$\times 10^{4}\,x_t\, [cells]$", fontsize=fontsize)

ax2.xaxis.set_label_coords(0.93, -0.04)
ax2.yaxis.set_label_coords(-0.05, 0.5)

black_line, = ax2.plot(data[:, 0], data[:, 1], color="black", lw=2, zorder=5)


xyA = [zoom_right, 1]
xyB = [51.435*year, 50000]
arrow = patches.ConnectionPatch(
    xyA,
    xyB,
    coordsA=ax1.transData,
    coordsB=ax2.transData,
    # Default shrink parameter is 0 so can be omitted
    color="forestgreen",
    arrowstyle="-|>",  # "<|-",  # "normal" arrow
    mutation_scale=13,  # controls arrow head size
    linewidth=1.8,
)
fig.patches.append(arrow)

legend = ax2.legend(
    [
        black_line,
        [object]
    ],
    [
        r"$x_t$",
        r"$\bar{A}(t)$",
    ],
    handler_map={object: AnyObjectHandler()},  # {tuple: HandlerTuple(ndivide=None)},
    loc='upper center', ncol=2, fontsize=fontsize, framealpha=1.0)

ax1.text(-8.1 * year, 6.0e11, r"$(a)$", fontsize=fontsize)
ax2.text(50.649 * year, 49000, r"$(b)$", fontsize=fontsize)

filename = r"moving_threshold_traj"
plt.savefig(filename, bbox_inches='tight', dpi=300)
plt.savefig(filename + ".jpg", bbox_inches='tight', dpi=300)
plt.savefig(filename + ".pdf", bbox_inches='tight')