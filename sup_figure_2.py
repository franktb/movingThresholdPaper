import numpy as np
import matplotlib.pyplot as plt
from sympy.physics.units import planck_time
from matplotlib.legend_handler import HandlerTuple, HandlerBase
from matplotlib.patches import  ConnectionStyle


from matplotlib import rc
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


data = np.loadtxt("realisationRescueEvents.txt", delimiter=" ")
year = 365
age = 50

fontsize = 13


leftT = int(37.085 * year)
rightT= int(37.5* year)


black_line, = plt.plot(data[leftT:rightT,0], data[leftT:rightT,1],  color="black")
plt.step(data[leftT:rightT,0], data[leftT:rightT,2],  color="blue", zorder=3)
plt.step(data[leftT:rightT,0], data[leftT:rightT,3],  color="red")

plt.ylim([0,22000])
plt.xlim([leftT,rightT])



ax = plt.gca()

xtick_loc = [int(37.1 * year), int(37.2 * year), int(37.3 * year), int(37.4 * year)]
xtick_label = [37.1, 37.2 , 37.3 , 37.4 ]
ax.set_xticks(xtick_loc)
ax.set_xticklabels(xtick_label, fontsize=fontsize)



ytick_loc = [5000,10000,15000,20000]
ytick_label = [5000,10000,15000,""]
ax.set_yticks(ytick_loc)
ax.set_yticklabels(ytick_label, fontsize=fontsize)



ax.set_xlabel(r"$t\,[years]$", fontsize=fontsize)
ax.set_ylabel(r"$x_t\,[cells]$", fontsize=fontsize)


ax.xaxis.set_label_coords(0.93, -0.02)
ax.yaxis.set_label_coords(-0.05, 0.87)



legend = ax.legend(
    [
        black_line,
        [object]
    ],
    [
        r"$x_t$",
        r"$\bar{A}(t)$",
    ],
    handler_map={object: AnyObjectHandler()},  # {tuple: HandlerTuple(ndivide=None)},
    loc='upper right', ncol=1, fontsize=fontsize, framealpha=1.0)




less = 14-int(11)
T = age * year
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

ax.stairs(H, data[:, 0], fill=True, alpha=0.3, color="gray", zorder=1)



ax.annotate("Rescue event",xy=(data[13574,0], data[13574,1]), xytext=(13610,20000), ha="center",va="center",
            arrowprops=dict(arrowstyle="->",
                            color="black",
                            connectionstyle=ConnectionStyle("arc,angleA=180,angleB=100, armA=30, armB=110, rad=30")), color="green", fontsize=fontsize,
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))


ax.annotate("Rescue event",xy=(data[13624,0], data[13624,1]), xytext=(13610,20000), ha="center",va="center",
            arrowprops=dict(arrowstyle="->"), color="green", fontsize=fontsize,
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))


ax.annotate("Rescue event",xy=(data[13649,0], data[13649,1]), xytext=(13610,20000), ha="center",va="center",
            arrowprops=dict(arrowstyle="->",
                            #connectionstyle=ConnectionStyle("angle3,angleA=30,angleB=100")),
                            connectionstyle=ConnectionStyle("arc,angleA=0,angleB=100, armA=65, armB=90, rad=15")),
            color="green", fontsize=fontsize,
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))


ax.annotate("Escape event",xy=(data[13536,0], data[13536,1]), xytext=(13590,1500), ha="center",va="center",
            arrowprops=dict(arrowstyle="->"), color="green", fontsize=fontsize,
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))


ax.annotate("Escape event",xy=(data[13610,0], data[13610,1]), xytext=(13590,1500), ha="center",va="center",
            arrowprops=dict(arrowstyle="->"), color="green", fontsize=fontsize,
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

ax.annotate("Escape event",xy=(data[13636,0], data[13636,1]), xytext=(13590,1500), ha="center",va="center",
            arrowprops=dict(arrowstyle="->",
                            connectionstyle=ConnectionStyle("arc3,rad=0.1")),
            color="green", fontsize=fontsize,
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))



filename = "sup_figure_2"
plt.savefig(filename, bbox_inches='tight', dpi=300)
plt.savefig(filename + ".pdf", bbox_inches='tight')