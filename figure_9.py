import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
import scipy as sc


rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

fontsize = 14

data = np.loadtxt("cycle_length_case4.txt" , delimiter=" ")


fig, ax1 = plt.subplots(1, 1, )

def formatAxis(ax):
    ax.set_xlim([8,20])
    xtick_loc = [8,9, 10, 11,12, 13,14, 15,16, 17,18, 19,20, ]
    #xtick_label = ["",8, "",10, "",12, "",14, "",16, "",18, "","","",]
    xtick_label = [8,9, 10, 11,12, 13,14, 15,16, 17, 18, 19, 20, ]
    ax.set_xticks(xtick_loc)
    ax.set_xticklabels(xtick_label, fontsize=fontsize)
    ax.set_xlabel(r"Length of the follicular phase $t_f$ [days]", fontsize=fontsize)



formatAxis(ax1)


black_line, = ax1.plot(data[2:-2,0], data[2:-2,1]/data[2:-2,3], markersize=6,color="black", marker='o',clip_on = False,zorder=3)
red_line, = ax1.plot(data[2:-2,0], data[2:-2,4]/data[2:-2,3], markersize=6, color="blue", marker='o',clip_on = False,zorder=3)
blue_line,  = ax1.plot(data[2:-2,0], data[2:-2,5]/data[2:-2,3], markersize=6, color ="red", marker='o',clip_on = False,zorder=3)


avg_fol =np.sum(data[:,4]/data[:,1])/data.shape[0]
avg_lut =np.sum(data[:,5]/data[:,1])/data.shape[0]

ax1.set_ylim([0,0.07])
ax1.set_ylabel(r"$\hat{\mathcal{R}}(51, t_f)\,[\%]$", fontsize=fontsize)
ax1.yaxis.set_label_coords(-0.02, 0.86)
ax1.vlines(11,0,0.1,color="gray", lw=1, ls="dashed")
ax1.vlines(17,0,0.1,color="gray", lw=1, ls="dashed")

axTotal = ax1.twiny()
axTotal.set_xlim([8,20])
axTotal.set_xlabel(r"Length of the menstrual cycle $t_f$ + 14 [days]", fontsize=fontsize)
axTotal.xaxis.set_label_coords(0.5, 1.11)


xtick_loc = [8,9, 10, 11,12, 13,14, 15,16, 17,18, 19,20, ]
xtick_label = np.array(["",23,24,25,26,27,28,29,30,31,32,33,"" ])
axTotal.set_xticks(xtick_loc)
axTotal.set_xticklabels(xtick_label, fontsize=fontsize + 1)



xtick_loc = [0.,0.02,0.04,0.06,]
xtick_label = [0,2,4,"",]
ax1.set_yticks(xtick_loc)
ax1.set_yticklabels(xtick_label, fontsize=fontsize)


xtick_loc = [0.,0.2,0.4,0.6,0.8,1.0]
xtick_label = [0,20,40,60,80,100]


risk22_Include25 = data[2:6,1]/data[2:6,3]
trap25 = sc.integrate.trapezoid(risk22_Include25, x=data[2:6,0])/3.
print("int25", trap25)


risk25_31 = data[5:12,1]/data[5:12,3]
trap2531 = sc.integrate.trapezoid(risk25_31,x=data[5:12,0].ravel())/6.
print("int2531", trap2531)



legend = ax1.legend(
    [
        black_line,
        red_line,
        blue_line
    ],
    [
        r"All cancers",
        r"Follicular phase cancers",
        r"Luteal phase cancers",
    ],
    loc='upper right', ncol=1, fontsize=fontsize, framealpha=1.0)


filename = r"figure_9"
plt.savefig(filename, bbox_inches='tight', dpi =300)
plt.savefig(filename+".pdf", bbox_inches='tight')




