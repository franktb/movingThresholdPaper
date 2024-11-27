import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from lmfit import minimize, Parameters, Parameter, report_fit, fit_report


rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('text.latex', preamble=r"\usepackage{amsmath}\usepackage{siunitx}")

fig, axs = plt.subplots(3, 2, figsize=(15, 14.),gridspec_kw={"wspace": 0.12,
                                                             "hspace": 0.2})

age = 80
rawT = np.arange(0, age, 5)

fontsize = 18




def residual_exp(ps, cum_risk):
    a = ps["a"].value
    b = ps["b"].value
    logrisk = np.log10(cum_risk[3:])
    lin_log = a * rawT[3:] + b
    return logrisk - lin_log




def common_top_visuals(ax):
    ax.set_xlim([0,75])
    xtick_loc = np.arange(0, age, 5)
    xtick_label = [0, r"", 10, r"", 20, r"", 30, r"", 40, r"", 50, r"", r"", r"", r"", r""]
    ax.set_xticks(xtick_loc)
    ax.set_xticklabels(xtick_label, fontsize=fontsize)
    ax.set_xlabel(r"$age\,[year]$", fontsize=fontsize)
    ax.xaxis.set_label_coords(0.91, -0.035)


def common_mid_visuals(ax):
    ax.set_xlim([15,75])
    xtick_loc = np.arange(15, age, 5)
    xtick_label = [r"", 20, r"", 30, r"", 40, r"", 50, r"", 60, r"", r"", r""]
    ax.set_xticks(xtick_loc)
    ax.set_xticklabels(xtick_label, fontsize=fontsize)
    ax.set_xlabel(r"$age\,[year]$", fontsize=fontsize)
    ax.xaxis.set_label_coords(0.91, -0.035)


def common_logx_visuals(ax):
    ax.set_xlim([np.log10(15), np.log10(75)])

    xtick_loc = np.log10(np.arange(15, age, 5))
    xtick_label = [r"$\log(15)$", r"$\log(20)$", r"", r"$\log(30)$", r"", r"$\log(40)$", r"", r"$\log(50)$", r"", r"", r"", r"", r""]
    ax.set_xticks(xtick_loc)
    ax.set_xticklabels(xtick_label, fontsize=fontsize)
    ax.set_xlabel(r"$\log(age)$", fontsize=fontsize)
    ax.xaxis.set_label_coords(0.93, -0.035)


incidences_data = np.loadtxt("Colorectal_C18C21_females_in_Ireland_19942021_20240528.txt", delimiter=",")
incidences_data = np.delete(incidences_data, 2, 1)
age_adjusted_rates = incidences_data[:-2,2]/100000
cum_rate = np.cumsum(5 * age_adjusted_rates)
cum_risk = 1 - np.exp(-cum_rate)




params = Parameters()

params.add('a', value=0.05, min=0.00001, max=2., vary=True)
params.add('b', value=-4., min=-8., max=1., vary=True)
result = minimize(residual_exp, params, args=((cum_risk,)))
report_fit(result)

mya = result.params["a"].value
myb = result.params["b"].value


with open('CRCFit_Exclude.txt', 'w') as fh:
    fh.write(fit_report(result))




axs[1,0].plot(rawT[3:], mya * rawT[3:] + myb,'-o', markersize=4,color="green")

green_bar, = axs[1,1].plot(rawT[3:], np.gradient(mya * rawT[3:] + myb,rawT[3:]), '-o', markersize=4,color="green",clip_on = False)









data = np.loadtxt("out_CRC.txt", delimiter=" ")
traj = data.shape[0]
unique, counts = np.unique(np.array(data).ravel(), return_counts=True)

HIST_BINS = np.linspace(0, age, 17)
HIST_BINS = np.insert(HIST_BINS, 0, -1, axis=0)
hist, bin_edges = np.histogram(data, HIST_BINS)

pp = np.zeros(rawT.shape[0])
for i in range(1, hist.shape[0]):
    pp[i - 1] = hist[i] / (traj - np.sum(hist[1:i]))

cum_prob = 1. - np.cumprod(1 - pp)







axs[0,0].plot(rawT, cum_risk, '-o', markersize=4,color="black",clip_on = False)
axs[0,0].plot(rawT, cum_prob, '-o', markersize=4,color="red",clip_on = False)

ytick_loc = np.array([0.00,0.01,0.02,0.03,0.04,])
ytick_label = [ r"0",r"1",r"2",r"",r""]
axs[0,0].set_yticks(ytick_loc)
axs[0,0].set_yticklabels(ytick_label, fontsize=fontsize)
axs[0,0].set_ylabel(r"$\mathcal{R}(t)\,[\%]$", fontsize=fontsize)
axs[0,0].yaxis.set_label_coords(-0.0175, 0.85)


axs[0,1].plot(rawT, np.gradient(cum_risk,rawT, edge_order=2),'-o', markersize=4,color="black",clip_on = False)
axs[0,1].plot(rawT, np.gradient(cum_prob,rawT, edge_order=2),'-o', markersize=4,color="red",clip_on = False)

ytick_loc = np.array([0,0.0005,0.001,0.0015,0.002,0.0025,0.003])
ytick_label = [r"0",r"",r"1",r"",r"",r"",r""]
axs[0,1].set_yticks(ytick_loc)
axs[0,1].set_yticklabels(ytick_label, fontsize=fontsize)
axs[0,1].set_ylabel(r"$\times 10^{-3}\,\frac{\Delta\,\mathcal{R}(t)}{\Delta\,t}$", fontsize=fontsize)
axs[0,1].yaxis.set_label_coords(-0.0175, 0.75)



axs[1,0].plot(rawT[3:],np.log10(cum_risk[3:]),'-o', markersize=4,color="black")
axs[1,0].plot(rawT[3:],np.log10(cum_prob[3:]),'-o', markersize=4,color="red")

ytick_loc = np.array([-4,-3,-2,-1,])
ytick_label = [-4,-3,r"",r"",]
axs[1,0].set_yticks(ytick_loc)
axs[1,0].set_yticklabels(ytick_label, fontsize=fontsize)
axs[1,0].set_ylabel(r"$\log\mathcal{R}(t)$", fontsize=fontsize)
axs[1,0].yaxis.set_label_coords(-0.0175, 0.85)

axs[1,1].plot(rawT[3:], np.gradient(np.log10(cum_risk[3:]),rawT[3:]), '-o', markersize=4,color="black",clip_on = False)
axs[1,1].plot(rawT[3:], np.gradient(np.log10(cum_prob[3:]),rawT[3:]), '-o', markersize=4,color="red",clip_on = False)


ytick_loc = np.array([0,0.025,0.05,0.075,0.1,])
ytick_label =        [0,r"", r"0.05",r"", r"0.1"]
axs[1,1].set_ylim([0,0.105])
axs[1,1].set_yticks(ytick_loc)
axs[1,1].set_yticklabels(ytick_label, fontsize=fontsize)
axs[1,1].set_ylabel(r"$\frac{\Delta\,\log\mathcal{R}(t)}{\Delta\,t}$", fontsize=fontsize)
axs[1,1].yaxis.set_label_coords(-0.0175, 0.75)




black_bar, = axs[2,0].plot(np.log10(rawT[3:]),np.log10(cum_risk[3:]),'-o', markersize=4,color="black",clip_on = False)
red_bar, =axs[2,0].plot(np.log10(rawT[3:]),np.log10(cum_prob[3:]),'-o', markersize=4,color="red",clip_on = False)
axs[2,1].plot(np.log10(rawT[3:]), np.gradient(np.log10(cum_risk[3:]),np.log10(rawT[3:])),'-o', markersize=4,color="black",clip_on = False)
axs[2,1].plot(np.log10(rawT[3:]), np.gradient(np.log10(cum_prob[3:]),np.log10(rawT[3:])),'-o', markersize=4,color="red",clip_on = False)



ytick_loc = np.array([-4,-3,-2,-1,])
ytick_label = [-4,-3,r"",r"",]
axs[2,0].set_yticks(ytick_loc)
axs[2,0].set_yticklabels(ytick_label, fontsize=fontsize)
axs[2,0].set_ylabel(r"$\log\mathcal{R}(t)$", fontsize=fontsize)
axs[2,0].yaxis.set_label_coords(-0.0175, 0.85)






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





fig.legend([black_bar,red_bar,green_bar],
           [r"$\mathcal{R}(t)$ obtained from data from years 1994-2021",
            r"$\hat{\mathcal{R}}(t)$ obtained from the complete model",
            r"linear fit of $\mathcal{R}(t)$ in the log-lin scale" ], loc='lower center', fontsize=fontsize, ncol=2)



filename = r"sup_figure_4"
plt.savefig(filename, bbox_inches='tight', dpi =300)
plt.savefig(filename+".pdf", bbox_inches='tight')