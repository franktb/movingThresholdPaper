import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('text.latex', preamble=r"\usepackage{amsmath}\usepackage{wasysym}")

fig, axs = plt.subplots(3, 2, figsize=(15, 14.),gridspec_kw={"wspace": 0.12,
                                                             "hspace": 0.2})
age = 80
rawT = np.arange(0, age, 5)

fontsize = 18



def common_top_visuals(ax):
    ax.set_xlim([0,75])
    xtick_loc = np.arange(0, age, 5)
    xtick_label = [0, r"", 10, r"", 20, r"", 30, r"", 40, r"", 50, r"", r"", r"", r"", r""]
    ax.set_xticks(xtick_loc)
    ax.set_xticklabels(xtick_label, fontsize=fontsize)
    ax.set_xlabel(r"$age\,[year]$", fontsize=fontsize)
    ax.xaxis.set_label_coords(0.91, -0.035)


def common_mid_visuals(ax):
    ax.set_xlim([5,75])
    xtick_loc = np.arange(5, age, 5)
    xtick_label = [r"5", 10,r"", 20, r"", 30, r"", 40, r"", 50, r"", r"", r"", r"", r""]
    ax.set_xticks(xtick_loc)
    ax.set_xticklabels(xtick_label, fontsize=fontsize)
    ax.set_xlabel(r"$age\,[year]$", fontsize=fontsize)
    ax.xaxis.set_label_coords(0.91, -0.035)


def common_logx_visuals(ax):
    ax.set_xlim([np.log10(5), np.log10(75)])

    xtick_loc = np.log10(np.arange(5, age, 5))
    xtick_label = [ r"$\log(5)$", r"$\log(10)$", r"", r"$\log(20)$", r"", r"$\log(30)$", r"", r"", r"", r"", r"", r"", r"", r"", r""]
    ax.set_xticks(xtick_loc)
    ax.set_xticklabels(xtick_label, fontsize=fontsize)
    ax.set_xlabel(r"$\log(age)$", fontsize=fontsize)
    ax.xaxis.set_label_coords(0.93, -0.035)



incidences_data_leu = np.loadtxt("Leukaemia_tal_C91C95_females_in_Ireland_19942021_20240927.csv", delimiter=",")
incidences_data_leu = np.delete(incidences_data_leu, 2, 1)
age_adjusted_rates_leu = incidences_data_leu[:-2,2]/100000
cum_rate_leu = np.cumsum(5 * age_adjusted_rates_leu)
cum_risk_leu = 1 - np.exp(-cum_rate_leu)

incidences_data_brain = np.loadtxt("Brain_C71_females_in_Ireland_19942021_20240927.csv", delimiter=",")
incidences_data_brain = np.delete(incidences_data_brain, 2, 1)
age_adjusted_rates_brain = incidences_data_brain[:-2,2]/100000
cum_rate_brain = np.cumsum(5 * age_adjusted_rates_brain)
cum_risk_brain = 1 - np.exp(-cum_rate_brain)


axs[0,0].plot(rawT, cum_risk_leu, '-o', markersize=4,color="red",clip_on = False)
axs[0,0].plot(rawT, cum_risk_brain, '-o', markersize=4,color="blue",clip_on = False)


ytick_loc = np.array([0.00,0.0025,0.005,0.0075,0.01,])
ytick_label = [ r"0",r"25",r"50",r"",r""]
axs[0,0].set_ylim([0,0.01])
axs[0,0].set_yticks(ytick_loc)
axs[0,0].set_yticklabels(ytick_label, fontsize=fontsize)
axs[0,0].set_ylabel(r"$\mathcal{R}(t)\,[\permil]$", fontsize=fontsize)
axs[0,0].yaxis.set_label_coords(-0.0175, 0.85)


axs[0,1].plot(rawT, np.gradient(cum_risk_leu,rawT, edge_order=2),'-o', markersize=4,color="red",clip_on = False)
axs[0,1].plot(rawT, np.gradient(cum_risk_brain,rawT, edge_order=2),'-o', markersize=4,color="blue",clip_on = False)


ytick_loc = np.array([0,0.0001,0.0002,0.0003,0.0004,0.0005])
ytick_label = [r"0",r"1",r"2",r"",r"",r""]
axs[0,1].set_ylim([0,0.0005])
axs[0,1].set_yticks(ytick_loc)
axs[0,1].set_yticklabels(ytick_label, fontsize=fontsize)
axs[0,1].set_ylabel(r"$\times 10^{-4}\,\frac{\Delta\,\mathcal{R}(t)}{\Delta\,t}$", fontsize=fontsize)
axs[0,1].yaxis.set_label_coords(-0.0175, 0.75)



axs[1,0].plot(rawT[1:],np.log10(cum_risk_leu[1:]),'-o', markersize=4,color="red",clip_on = False)
axs[1,0].plot(rawT[1:],np.log10(cum_risk_brain[1:]),'-o', markersize=4,color="blue",clip_on = False)



ytick_loc = np.array([-4,-3,-2,])
ytick_label = [-4,-3,r"",]
axs[1,0].set_yticks(ytick_loc)
axs[1,0].set_yticklabels(ytick_label, fontsize=fontsize)
axs[1,0].set_ylabel(r"$\log\mathcal{R}(t)$", fontsize=fontsize)
axs[1,0].yaxis.set_label_coords(-0.0175, 0.78)

axs[1,1].plot(rawT[1:], np.gradient(np.log10(cum_risk_leu[1:]),rawT[1:]), '-o', markersize=4,color="red",clip_on = False)
axs[1,1].plot(rawT[1:], np.gradient(np.log10(cum_risk_brain[1:]),rawT[1:]), '-o', markersize=4,color="blue",clip_on = False)



ytick_loc = np.array([0,0.025,0.05,0.075,0.1,])
ytick_label =        [0,r"", r"0.05",r"", r""]
axs[1,1].set_ylim([0,0.105])
axs[1,1].set_yticks(ytick_loc)
axs[1,1].set_yticklabels(ytick_label, fontsize=fontsize)
axs[1,1].set_ylabel(r"$\frac{\Delta\,\log\mathcal{R}(t)}{\Delta\,t}$", fontsize=fontsize)
axs[1,1].yaxis.set_label_coords(-0.0175, 0.75)




red_bar, = axs[2,0].plot(np.log10(rawT[1:]),np.log10(cum_risk_leu[1:]),'-o', markersize=4,color="red",clip_on = False)
blue_bar, = axs[2,0].plot(np.log10(rawT[1:]),np.log10(cum_risk_brain[1:]),'-o', markersize=4,color="blue",clip_on = False)

axs[2,1].plot(np.log10(rawT[1:]), np.gradient(np.log10(cum_risk_leu[1:]),np.log10(rawT[1:])),'-o', markersize=4,color="red",clip_on = False)
axs[2,1].plot(np.log10(rawT[1:]), np.gradient(np.log10(cum_risk_brain[1:]),np.log10(rawT[1:])),'-o', markersize=4,color="blue",clip_on = False)



ytick_loc = np.array([-4,-3,-2,])
ytick_label = [-4,-3,r"",]
axs[2,0].set_yticks(ytick_loc)
axs[2,0].set_yticklabels(ytick_label, fontsize=fontsize)
axs[2,0].set_ylabel(r"$\log\mathcal{R}(t)$", fontsize=fontsize)
axs[2,0].yaxis.set_label_coords(-0.0175, 0.78)






ytick_loc = np.array([1,2,3,4,5])
ytick_label =        [r"1", r"2",r"3", r"",r""]
axs[2,1].set_yticks(ytick_loc)
axs[2,1].set_yticklabels(ytick_label, fontsize=fontsize)
axs[2,1].set_ylabel(r"$\frac{\Delta\,\log\mathcal{R}(t)}{\Delta\,\log\, t}$", fontsize=fontsize)
axs[2,1].yaxis.set_label_coords(-0.0175, 0.75)







for index, ax in np.ndenumerate(axs):
    if index[0]==0:
        common_top_visuals(ax)
    elif index[0]==1:
        common_mid_visuals(ax)
    elif index[0]==2:
        common_logx_visuals(ax)



relX = -0.038
relY = 1.04
axs[0,0].text(relX , relY, '(a)', ha='center', va='center', transform=axs[0,0].transAxes, fontsize=fontsize-1)
axs[0,1].text(relX , relY, '(b)', ha='center', va='center', transform=axs[0,1].transAxes, fontsize=fontsize-1)
axs[1,0].text(relX , relY, '(c)', ha='center', va='center', transform=axs[1,0].transAxes, fontsize=fontsize-1)
axs[1,1].text(relX , relY, '(d)', ha='center', va='center', transform=axs[1,1].transAxes, fontsize=fontsize-1)
axs[2,0].text(relX , relY, '(e)', ha='center', va='center', transform=axs[2,0].transAxes, fontsize=fontsize-1)
axs[2,1].text(relX , relY, '(f)', ha='center', va='center', transform=axs[2,1].transAxes, fontsize=fontsize-1)



fig.legend([ red_bar, blue_bar],
           [
            r"Blood cancer: $\mathcal{R}(t)$ obtained from data from years 1994-2021",
            r"Brain cancer: $\mathcal{R}(t)$ obtained from data from years 1994-2021"],
           loc='lower center', fontsize=fontsize, ncol=1)



filename = r"sup_figure_6"
plt.savefig(filename, bbox_inches='tight', dpi =300)
plt.savefig(filename+".pdf", bbox_inches='tight')