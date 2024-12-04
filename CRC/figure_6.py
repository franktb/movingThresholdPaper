import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from lmfit import minimize, Parameters, Parameter, report_fit, fit_report




rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


fontsize = 15
age = 80
rawT = np.arange(0, age, 5)


fig, ax = plt.subplots(1, 1, figsize=(15./2, 9./2), gridspec_kw={"wspace": 0.1,"hspace": 0.20})


def common_visuals(ax):
    xtick_loc = np.arange(0, age, 5) + 2.5
    xtick_label = ["0-4", r"", "10-14", r"", "20-24", r"", "30-34", r"", "40-44", r"", "50-54", r"", "60-64", r"", r"",
                   r""]

    ax.set_xticks(xtick_loc)
    ax.set_xticklabels(xtick_label, fontsize=fontsize)
    ax.set_xlabel(r"$age\,[years]$", fontsize=fontsize)
    ax.yaxis.set_label_coords(-0.0175, 0.895)
    ax.xaxis.set_label_coords(0.92, -0.024)
    ax.set_xlim([0, 80.5])


#################################################################
# Top left
#################################################################


data = np.loadtxt("out_strict_CRC.txt", delimiter=" ")
traj = data.shape[0]
unique, counts = np.unique(np.array(data).ravel(), return_counts=True)

HIST_BINS = np.linspace(0, age, 17)
HIST_BINS = np.insert(HIST_BINS, 0, -1, axis=0)
hist, bin_edges = np.histogram(data, HIST_BINS)

pp = np.zeros(rawT.shape[0])
for i in range(1, hist.shape[0]):
    pp[i - 1] = hist[i] / (traj - np.sum(hist[1:i]))

cum_prob = 1. - np.cumprod(1 - pp)

red_bar = ax.bar(rawT + 2.5, cum_prob, width=1.75, color="red", clip_on=True, fill=True, zorder=1, align="edge")

incidences_data_crc = np.loadtxt("Colorectal_C18C21_females_in_Ireland_19942021_20240528.txt", delimiter=",")

incidences_data = np.delete(incidences_data_crc, 2, 1)
age_adjusted_rates = incidences_data[:-2, 2] / 100000
cum_rate = np.cumsum(5 * age_adjusted_rates)
cum_risk = 1 - np.exp(-cum_rate)
gradient_cum_risk = np.gradient(cum_risk)

black_bar = ax.bar(rawT + .75, cum_risk, width=1.75, color="black", clip_on=True, fill=True, zorder=1,
                          align="edge")

ytick_loc = [0, 0.01, 0.02, 0.03, 0.04]
ytick_label = ["0", "1", "2", "3", ""]
ylabel = r"$\mathcal{R}(t)\,[\%]$"
ax.legend([black_bar,red_bar], [r"$\mathcal{R}(t)$ obtained from real data",r"$\hat{\mathcal{R}}(t)$ obtained from the complete model"], loc='upper left', fontsize=fontsize, ncol=1)



ax.set_yticks(ytick_loc)
ax.set_yticklabels(ytick_label, fontsize=fontsize)
ax.set_ylabel(ylabel, fontsize=fontsize)


common_visuals(ax)
filename = r"figure_6"
plt.savefig(filename, bbox_inches='tight', dpi=300)
plt.savefig(filename + ".pdf", bbox_inches='tight')