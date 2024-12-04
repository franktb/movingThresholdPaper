import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import norm, gamma



rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('text.latex', preamble=r"\usepackage{amsmath}\usepackage{color}")



fig, axs = plt.subplots(2, 2, figsize=(15, 9.), gridspec_kw={"wspace": 0.1,"hspace": 0.20})#, gridspec_kw={'width_ratios': [1, 1,1], "wspace": 0.1})

print(axs.shape)



fontsize = 15
age = 80
rawT = np.arange(0, age, 5)




data_files = ["out_exp.txt","out_strict_51.txt","out_strict_meno.txt","out_hrt.txt"]
for id, df in enumerate(data_files):
    data = np.loadtxt(df, delimiter=" ")
    traj = data.shape[0]
    unique, counts = np.unique(np.array(data).ravel(), return_counts=True)

    HIST_BINS = np.linspace(0, age, 17)
    HIST_BINS = np.insert(HIST_BINS, 0, -1, axis=0)
    hist, bin_edges = np.histogram(data, HIST_BINS)
    pp = np.zeros(rawT.shape[0])
    for i in range(1, hist.shape[0]):
        pp[i - 1] = hist[i] / (traj - np.sum(hist[1:i]))

    cum_prob = 1. - np.cumprod(1 - pp)
    (u, v) = np.unravel_index(id, shape=(2,2))
    print(u,v)
    red_bar = axs[u,v].bar(rawT + 2.5, cum_prob, width=1.75, color="red", clip_on=True, fill=True, zorder=1,
                        align="edge")



incidences_data = np.loadtxt("Breast_C50_females_in_Ireland_20162016_20230701.txt", delimiter=",")
incidences_data = np.delete(incidences_data, 2, 1)

age_adjusted_rates = incidences_data[:-2,2]/100000
cum_rate = np.cumsum(5 * age_adjusted_rates)
cum_risk = 1 - np.exp(-cum_rate)
gradient_cum_risk = np.gradient(cum_risk)



xtick_loc = np.arange(0, age, 5)+2.5
xtick_label = [0,r"",10,r"",20,r"",30,r"",40,r"",50,r"",60,r"",r"",r""]
xtick_label = ["0-4",r"","10-14",r"","20-24",r"","30-34",r"","40-44",r"","50-54",r"","60-64",r"",r"",r""]


ytick_loc = [0,0.02,0.04,0.06,0.08,0.1,0.12]
ytick_label = [0,2,4,6,8,10,""]

for index, ax in np.ndenumerate(axs):

    left, bottom, width, height = [0.11, 0.53, 0.55, 0.45]
    axins = inset_axes(ax, width="100%", height="100%",
                       bbox_to_anchor=(left, bottom, width, height),
                       bbox_transform=ax.transAxes)
    ins_left = 30
    ins_right = 80
    ins_up = 0.09
    ins_low = 0
    axins.set_xlim([ins_left, ins_right])
    axins.set_ylim([ins_low, ins_up])
    ravId = np.ravel_multi_index(index, (2, 2))

    xtickIn_loc = [30, 40, 50, 60, 70, 80]
    xtickIn_label = ["30", "40", "50", "60", "", ""]
    axins.set_xticks(xtickIn_loc)
    axins.set_xticklabels(xtickIn_label, fontsize=fontsize - 2)

    ytickIn_loc = [0, 0.025, 0.05, 0.075]
    ytickIn_label = ["0.0", "", "0.05", "", ]
    axins.set_yticks(ytickIn_loc)
    axins.set_yticklabels(ytickIn_label, fontsize=fontsize - 2)

    x = np.linspace(ins_left, ins_right, 100)

    if (ravId == 0):
        axins.text((ins_left + ins_right) / 2., (ins_up + ins_low) / 2., "Constant immune response", ha="center", va="center",
                   color="green", fontsize=fontsize)

    if (ravId == 1):
        axins.vlines(51, ins_low, ins_up, lw=2, color="green")
        axins.hlines(0, ins_left, ins_right, lw=2, color="green", clip_on=False)
        axins.text((51+80)/2., 0.053, "average\n menopause", ha="center", va="center",
                   color="green", fontsize=fontsize)

    if (ravId == 2):
        sd_meno = 4.86
        axins.plot(x, norm.pdf(x, loc=51.0, scale=sd_meno), lw=2, color="green")
        axins.text(68.5, 0.057, "distributed\nmenopause", ha="center", va="center",
                   color="green", fontsize=fontsize)

    if (ravId == 3):
        traj = 10000000
        hrt_fraction = 0.25
        hrt_index = int(round(hrt_fraction * traj, 0))
        sd_meno = 4.86
        # mg = 1.75
        # sg = 5 / (mg - 1.)
        mg = 3.2553
        sg = 2.66

        mean, var, skew, kurt = gamma.stats(mg, loc=0, scale=sg, moments='mvsk')
        mode = (mg - 1.) * sg

        age_dist_natural = norm.rvs(loc=51.0, scale=sd_meno, size=traj - hrt_index)
        age_dist_hrt = norm.rvs(loc=51.0, scale=sd_meno, size=hrt_index)
        hrt = age_dist_hrt + gamma.rvs(mg, loc=0, scale=sg, size=hrt_index)
        eff_meno = np.concatenate((age_dist_natural, hrt))
        HIST_BINS = np.arange(29, 92, 1)
        hist, bin_edges = np.histogram(eff_meno, HIST_BINS)
        axins.plot(bin_edges[:-1] + 0.5, (hist / traj), lw=2, color="green")

        hist_nat, bin_edges = np.histogram(age_dist_natural, HIST_BINS)
        axins.plot(bin_edges[:-1] + 0.5, (hist_nat / traj), lw=1, color="green", ls="dashed")

        hist_hrt, bin_edges = np.histogram(hrt, HIST_BINS)
        axins.plot(bin_edges[:-1] + 0.5, (hist_hrt / traj), lw=1, color="green", ls="dashed")

        #axins.text(68.5, 0.057, "natural\n menopause\n\phantom{ + HRT}", ha="center", va="center",
         #          color="green", fontsize=fontsize)
        d_menoText = axins.text(68.5, 0.0642, "distributed\nmenopause", ha="center", va="center",
                   color="green", fontsize=fontsize)
        axins.annotate(r"+ HRT", xycoords=d_menoText, xy=(0.1, -0.65), fontsize=fontsize,
                           color="green")

    axins.set_xlabel(r"$M\,[years]$", fontsize=fontsize - 2)
    axins.yaxis.set_label_coords(-0.14, 0.8)
    axins.xaxis.set_label_coords(0.887, -0.06)

    black_bar = ax.bar(rawT+.75,cum_risk, width=1.75, color = "black", clip_on = True, fill=True, zorder=1, align="edge")
    ax.set_xticks(xtick_loc)
    ax.set_xticklabels(xtick_label, fontsize=fontsize)
    ax.set_yticks(ytick_loc)
    ax.set_yticklabels(ytick_label, fontsize=fontsize)
    ax.set_xlabel(r"$age\,[years]$", fontsize=fontsize)
    ax.set_ylabel(r"$\mathcal{R}(t)\,[\%]$", fontsize=fontsize)
    ax.yaxis.set_label_coords(-0.0175, 0.895)
    ax.xaxis.set_label_coords(0.92, -0.024)
    ax.set_xlim([0, 80.5])
    ax.set_ylim([0, 0.13765])



axs[0,1].text(40.0, 0.08, r"average menopause",va="center", ha="center",fontsize=fontsize, zorder=5)
axs[0,0].text(-4.5,0.144,"(a)", fontsize=fontsize-1)
axs[0,1].text(-4.5,0.144,"(b)", fontsize=fontsize-1)
axs[1,0].text(-4.5,0.144,"(c)", fontsize=fontsize-1)
axs[1,1].text(-4.5,0.144,"(d)", fontsize=fontsize-1)


fig.legend([black_bar,red_bar], [r"$\mathcal{R}(t)$ obtained from real data",r"$\hat{\mathcal{R}}(t)$ obtained from the complete model"], loc='lower center', fontsize=fontsize, ncol=2)


filename = "figure_7"
plt.savefig(filename, bbox_inches='tight', dpi=300)
plt.savefig(filename+".pdf", bbox_inches='tight')
