import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('text.latex', preamble=r"\usepackage{amsmath}\usepackage{siunitx}")

fig, ax2 = plt.subplots(1, 1, figsize=(20,8.))

left, bottom, width, height = [0.22, 0.5, 0.5, 0.3]
ax1 = fig.add_axes([left, bottom, width, height])

fontsize = 21
year = 365

left = 0
right =28*2

A_max = 9
A_min = 3

A_avg = (A_min+A_max)/2


print_menarch = 12
lw =3


def parabola(x):
    return -0.2*(x-14)*(x-28)


axProg = ax1.twinx()
data = np.loadtxt("progesterone.txt", delimiter=" ")
progesterone = data[:,1]
axProg.plot(np.concatenate((progesterone[1:-1],progesterone[1:-1],np.atleast_1d(progesterone[1])),dtype=float).ravel(),zorder=1,color = "blue", lw=lw)
axProg.set_ylim([0,51])
ytick_loc_prog = [0,5,10,15,20,25,30,35,40]
ytick_labels_prog = [0,"",10,"",20,"",30,"",40]
axProg.set_yticks(ytick_loc_prog)
axProg.set_yticklabels(ytick_labels_prog, fontsize=fontsize + 1)

axProg.set_ylabel(r"$Progesterone$"+"\n"+r"$[nmol/Liter]$", fontsize=fontsize, color="blue")
axProg.xaxis.set_label_coords(0.97, -0.022)
axProg.yaxis.set_label_coords(1.05, 0.5)

xx = np.linspace(left,right,1000)
xx_shift = np.linspace(left+28,right+28,1000)

ax1.hlines(A_max,0,14,color = "red", lw=lw, zorder=5)
ax1.hlines(A_min,14,28,color = "red", lw=lw, zorder=5)
ax1.vlines(14,A_min,A_max,color = "red", lw=lw, zorder=5)


ax1.vlines(28,A_min,A_max,color = "red", lw=lw, zorder=5)
ax1.hlines(A_max,0+28,14+28,color = "red", lw=lw, zorder=5)
ax1.hlines(A_min,14+28,28+28,color = "red", lw=lw, zorder=5)
ax1.vlines(14+28,A_min,A_max,color = "red", lw=lw, zorder=5)


ax1.hlines(A_avg,0,2*28,ls=(0, (1., 1.)), lw=1.5, color="red")


xtick_loc=[0,14,28,42]
xtick_label=["0",r"$t_f$","$t_f + 14$",r"$2t_f+14$"]


ax1.set_xlabel("$t [days]$", fontsize=fontsize)
ax1.xaxis.set_label_coords(0.95, -0.04)


ax1.set_xticks(xtick_loc)
ax1.set_xticklabels(xtick_label, fontsize=fontsize + 1)

ax1.set_yticks([A_min,A_avg,A_max])
ax1.set_yticklabels([r"$s_{min}$",r"$s_{avg}$",r"$s_{max}$"], fontsize=fontsize)
ax1.set_xlim([left,right])
ax1.set_ylim([0,13])


ax1.set_ylabel(r"$s_m(t)$"+"\n"+r"$[cells/day]$", rotation=90,fontsize=fontsize, color="red")
ax1.yaxis.set_label_coords(-0.08, 0.5)



ax1.vlines(14,0,13, ls="dashed", lw=1, color="black")
ax1.vlines(28,0,13, ls="dashed", lw=1, color="black")
ax1.vlines(42,0,13, ls="dashed", lw=1, color="black")


ax1.text(7,11.,"follicular",ha="center", fontsize=fontsize)
ax1.text(21,11.,"luteal",ha="center", fontsize=fontsize)
ax1.text(7+28,11.,"follicular",ha="center", fontsize=fontsize)
ax1.text(21+28,11.,"luteal",ha="center", fontsize=fontsize)


ax1.add_patch(Rectangle((14,0), 14, 130, edgecolor = 'gray',facecolor="gray",alpha=0.3,fill=True,zorder=4, lw=2))
ax1.add_patch(Rectangle((42,0), 14, 130, edgecolor = 'gray',facecolor="gray",alpha=0.3,fill=True,zorder=4, lw=2))


ax2.set_xlim([0,80])
ax2.set_ylim([0,15])

A_max = 5
A_min = 2
A_avg = (A_min+A_max)/2

ax2.set_yticks([A_min,A_avg,A_max])
ax2.set_yticklabels([r"$s_{min}$",r"$s_{avg}$",r"$s_{max}$"], fontsize=fontsize)

m_star= 53.5


xtick_loc=[0,10,12,20,30,40,50,m_star,60,70,80]
xtick_label=["0","10","12","20","30","40","50","M","60","70",""]


ax2.set_xlabel(r"$t\,[years]$", fontsize=fontsize)
ax2.xaxis.set_label_coords(0.96, -0.01)


ax2.hlines(A_max,0,print_menarch, lw=lw, color="red")
ax2.hlines(A_max,m_star,80, lw=lw, color="red")

ax2.vlines(print_menarch, 0,A_max, ls="dashed", lw=1, color="black")
ax2.vlines(m_star,0,A_max, ls="dashed", lw=1, color="black")

ax2.hlines(A_avg,print_menarch, m_star, ls=(0, (1., 1.)), lw=1.5, color="red")
ax2.set_xticks(xtick_loc)
ax2.set_xticklabels(xtick_label, fontsize=fontsize + 1)


ax2.text(11.,2.7,r"Menarche",rotation=90, va="center", ha="center", fontsize=fontsize)
ax2.text(55.7,2.7,r"Effective"+"\n"+"menopause",rotation=90, va="center", ha="center", fontsize=fontsize)
d = 0.1

xyB = [32, A_avg]
xyA = [0., -3]

arrow = patches.ConnectionPatch(
    xyA,
    xyB,
    coordsA=ax1.transData,
    coordsB=ax2.transData,
    # Default shrink parameter is 0 so can be omitted
    color="black",
    arrowstyle="-",#"<|-",  # "normal" arrow
    mutation_scale=30,  # controls arrow head size
    linewidth=1,
)
fig.patches.append(arrow)

xyB = [32., A_avg]
xyA = [56.,-3]
arrow2 = patches.ConnectionPatch(
    xyA,
    xyB,
    coordsA=ax1.transData,
    coordsB=ax2.transData,
    # Default shrink parameter is 0 so can be omitted
    color="black",
    arrowstyle="-",#"<|-",  # "normal" arrow
    mutation_scale=30,  # controls arrow head size
    linewidth=1,
)
fig.patches.append(arrow2)


A_max = 7
A_min = 2
A_avg = (A_min+A_max)/2



xtick_loc=[0,10,20,30,40,50,60,70,80]
xtick_label=["0","10","20","30","40","50","60","70",""]

ax2.set_ylabel(r"$s_m(t)$"+"\n"+r"$[cells/day]$", rotation=90,fontsize=fontsize, )
ax2.yaxis.set_label_coords(-0.02, 0.8)
ax1.set_zorder(10)
ax1.patch.set_visible(False)


filename = "figure_8"
plt.savefig(filename, bbox_inches='tight', dpi=300)
plt.savefig(filename+".pdf", bbox_inches='tight')