import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from lmfit import minimize, Parameters, Parameter, report_fit, fit_report



rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('text.latex', preamble=r"\usepackage{amsmath}\usepackage{siunitx}")

fig, axs = plt.subplots(3, 2, figsize=(15, 17.5),gridspec_kw={"wspace": 0.15,
                                                             "hspace": 0.15})

age = 80
rawT = np.arange(0, age, 5)

fontsize = 18

xlabelX = 0.91



def residual_exp(ps, cum_risk):
    a = ps["a"].value
    b = ps["b"].value

    logrisk = np.log10(cum_risk[4:9])
    lin_log = a * rawT[4:9] + b
    return logrisk - lin_log



def residual_exp_postmeno(ps, cum_risk):
    a = ps["a"].value
    b = ps["b"].value

    logrisk = np.log10(cum_risk[10:])
    lin_log = a * rawT[10:] + b
    return logrisk - lin_log



def residual_power_premeno(ps, cum_risk):
    a = ps["a"].value
    b = ps["b"].value

    logrisk = np.log10(cum_risk[4:10])
    lin_log = a * np.log10(rawT[4:10]) + b
    return logrisk - lin_log


def residual_power_postmeno(ps, cum_risk):
    a = ps["a"].value
    b = ps["b"].value

    logrisk = np.log10(cum_risk[10:])
    lin_log = a * np.log10(rawT[10:]) + b
    return logrisk - lin_log



def residual_postmeno_poly(ps, cum_risk):
    a = ps["a"].value
    b = ps["b"].value
    c = ps["c"].value
    d = ps["d"].value


    cum_prob = a * rawT[10:]**3 + b * rawT[10:]**2 + c*rawT[10:] + d

    return cum_risk[10:] - cum_prob




def common_top_visuals(ax):
    ax.set_xlim([0,75])
    xtick_loc = np.arange(0, age, 5)
    xtick_label = [0, r"", 10, r"", 20, r"", 30, r"", 40, r"", 50, r"", r"", r"", r"", r""]
    ax.set_xticks(xtick_loc)
    ax.set_xticklabels(xtick_label, fontsize=fontsize)
    ax.set_xlabel(r"$age\,[year]$", fontsize=fontsize)
    ax.xaxis.set_label_coords(xlabelX, -0.03)


def common_mid_visuals(ax):
    ax.set_xlim([50,75])
    xtick_loc = np.arange(50, age, 5)
    xtick_label = [50, 55, 60, 65, r"", r""]
    ax.set_xticks(xtick_loc)
    ax.set_xticklabels(xtick_label, fontsize=fontsize)
    ax.set_xlabel(r"$age\,[year]$", fontsize=fontsize)
    ax.xaxis.set_label_coords(xlabelX, -0.03)




def common_log_visuals(ax):
    ax.set_ylim([np.log10(0.00003), np.log10(0.2)])

    ytick_loc = np.log10([0.0001,.001,0.01,0.1])
    ytick_label = [-4, -3, -2, r"", ]
    ax.set_yticks(ytick_loc)
    ax.set_yticklabels(ytick_label, fontsize=fontsize)






incidences_data = np.loadtxt("Breast_C50_females_in_Ireland_20162016_20230701.txt", delimiter=",")
incidences_data = np.delete(incidences_data, 2, 1)
age_adjusted_rates = incidences_data[:-2,2]/100000
cum_rate = np.cumsum(5 * age_adjusted_rates)
cum_risk = 1 - np.exp(-cum_rate)




params = Parameters()

params.add('a', value=0.0, vary=False)
params.add('b', value=0.0, vary=False)
params.add('c', value=0.00295560, min=-8., max=1., vary=True)
params.add('d', value=-0.10971344, min=-1, max=2., vary=True)
result = minimize(residual_postmeno_poly, params, args=((cum_risk,)))
report_fit(result)


mycPost = result.params["c"].value
mydPost = result.params["d"].value
axs[1,0].plot(rawT[10:], mycPost*rawT[10:] + mydPost,'-o', markersize=4,color="green",clip_on = False)
green_bar, = axs[1,1].plot(rawT[10:], cum_risk[10:] -  (mycPost*rawT[10:] + mydPost),'-o', markersize=4,color="green",clip_on = False)





params = Parameters()

params.add('a', value=0.0, vary=False)
params.add('b', value=1.1, min=-8., max=10., vary=True)
params.add('c', value=0.00295560, min=-8., max=1., vary=True)
params.add('d', value=-0.10971344, min=-1, max=2., vary=True)
result = minimize(residual_postmeno_poly, params, args=((cum_risk,)))
report_fit(result)




mybPost = result.params["b"].value
mycPost = result.params["c"].value
mydPost = result.params["d"].value


params = Parameters()

params.add('a', value=1.1, min=-8., max=10., vary=True)
params.add('b', value=1.1, min=-8., max=10., vary=True)
params.add('c', value=0.00295560, min=-8., max=1., vary=True)
params.add('d', value=-0.10971344, min=-1, max=2., vary=True)
result = minimize(residual_postmeno_poly, params, args=((cum_risk,)))
report_fit(result)



myaPost = result.params["a"].value
mybPost = result.params["b"].value
mycPost = result.params["c"].value
mydPost = result.params["d"].value




params = Parameters()
params.add('a', value=7.1, min=0.01, max=20., vary=True)
params.add('b', value=-2., min=-20., max=1., vary=True)
result = minimize(residual_power_premeno, params, args=((cum_risk,)))
report_fit(result)
myaPre = result.params["a"].value
mybPre = result.params["b"].value




orange_bar, = axs[2,1].plot(np.log10(rawT[4:10]), myaPre * np.log10(rawT[4:10]) + mybPre,'-o', markersize=4,color="orange",clip_on = False)



params = Parameters()
params.add('a', value=7.1, min=0.01, max=20., vary=True)
params.add('b', value=-2., min=-20., max=1., vary=True)
result = minimize(residual_power_postmeno, params, args=((cum_risk,)))
report_fit(result)

myaPost = result.params["a"].value
mybPost = result.params["b"].value




incidences_data_pool = np.loadtxt("Breast_C50_females_in_Ireland_19942021_20240923.csv", delimiter=",")
incidences_data_pool = np.delete(incidences_data_pool, 2, 1)
age_adjusted_rates_pool = incidences_data_pool[:-2,2]/100000
cum_rate_pool = np.cumsum(5 * age_adjusted_rates_pool)
cum_risk_pool = 1 - np.exp(-cum_rate_pool)



data = np.loadtxt("out_hrt.txt", delimiter=" ")
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
axs[1,0].plot(rawT[10:], cum_risk[10:], '-o', markersize=4,color="black",clip_on = False)
axs[1,1].plot(rawT[10:], cum_risk[10:] - cum_risk[10:], '-o', markersize=4,color="black",clip_on = False)
axs[0,0].plot(rawT, cum_risk_pool, '-o', markersize=4,color="blue",clip_on = False)
axs[0,0].set_ylim([0.0,0.13])

ytick_loc = np.array([0,0.02,0.04,0.06,0.08,0.1,0.12])
ytick_label = [r"0", r"2",r"4",r"6",r"8",r"",r""]
axs[0,0].set_yticks(ytick_loc)
axs[0,0].set_yticklabels(ytick_label, fontsize=fontsize)
axs[0,0].set_ylabel(r"$\mathcal{R}(t)\,[\%]$", fontsize=fontsize)
axs[0,0].yaxis.set_label_coords(-0.0175, 0.85)

axs[1,0].set_yticks(ytick_loc)
axs[1,0].set_yticklabels(ytick_label, fontsize=fontsize)
axs[1,0].set_ylabel(r"$\mathcal{R}(t)\,[\%]$", fontsize=fontsize)
axs[1,0].yaxis.set_label_coords(-0.0175, 0.85)

axs[1,0].set_xlim([50,75])
axs[1,0].set_ylim([0.03,0.12])




ytick_loc = np.linspace(-0.003,0.003,7)
ytick_label = [ -3,-2,-1,0,1,2,r""]
axs[1,1].set_yticks(ytick_loc)
axs[1,1].set_yticklabels(ytick_label, fontsize=fontsize)


axs[1,1].set_xlim([50,75])
axs[1,1].set_ylim([-0.003,0.003])

axs[1,1].set_ylabel(r"$\times 10^{-3}\,\left(\mathcal{R}(t) - \mathcal{R}_{fit}(t)\right)$", fontsize=fontsize)
axs[1,1].yaxis.set_label_coords(-0.05, 0.5)


axs[0,1].plot(rawT, np.gradient(cum_risk,rawT, edge_order=2),'-o', markersize=4,color="black",clip_on = False)
axs[0,1].plot(rawT, np.gradient(cum_prob,rawT, edge_order=2),'-o', markersize=4,color="red",clip_on = False)
axs[0,1].plot(rawT, np.gradient(cum_risk_pool,rawT, edge_order=2),'-o', markersize=4,color="blue",clip_on = False)


axs[0,1].set_ylim([0,0.007])
ytick_loc = np.array([0,0.001,0.002,0.003,0.004,0.005,0.006])
ytick_label = [r"0", r"1",r"2",r"3",r"",r"",r""]
axs[0,1].set_yticks(ytick_loc)
axs[0,1].set_yticklabels(ytick_label, fontsize=fontsize)
axs[0,1].set_ylabel(r"$\times 10^{-3}\,\frac{\Delta\,\mathcal{R}(t)}{\Delta\,t}$", fontsize=fontsize)
axs[0,1].yaxis.set_label_coords(-0.0175, 0.75)



axs[2,0].plot(rawT[4:10],np.log10(cum_risk[4:10]),'-o', markersize=4,color="black",clip_on = False)
axs[2,0].plot(rawT[4:10],np.log10(cum_prob[4:10]),'-o', markersize=4,color="red",clip_on = False)
axs[2,0].plot(rawT[4:10],np.log10(cum_risk_pool[4:10]),'-o', markersize=4,color="blue",clip_on = False)

xtick_loc = np.arange(20,50,5)
xtick_label = [20,25,30,35,r"",r""]
axs[2,0].set_xticks(xtick_loc)
axs[2,0].set_xticklabels(xtick_label, fontsize=fontsize)
axs[2,0].set_xlim([20,45])
axs[2,0].set_ylabel(r"$\log\mathcal{R}(t)$", fontsize=fontsize)
axs[2,0].yaxis.set_label_coords(-0.0175, 0.85)

ytick_loc = np.array([0.025,0.05,0.075,0.1,0.125,0.15,0.175])
ytick_label =        [r"", r"0.05",r"", r"0.1",r"",r"",r""]

axs[2,0].set_xlabel(r"$age\,[year]$", fontsize=fontsize)
axs[2,0].xaxis.set_label_coords(xlabelX, -0.03)



black_bar, = axs[2,1].plot(np.log10(rawT[4:10]),np.log10(cum_risk[4:10]),'-o', markersize=4,color="black",clip_on = False)
blue_bar, = axs[2,1].plot(np.log10(rawT[4:10]),np.log10(cum_risk_pool[4:10]),'-o', markersize=4,color="blue",clip_on = False)
red_bar, =axs[2,1].plot(np.log10(rawT[4:10]),np.log10(cum_prob[4:10]),'-o', markersize=4,color="red",clip_on = False)


xtick_loc = np.log10(np.arange(20,50,5))
xtick_label = [r"$\log(20)$",r"$\log(25)$",r"$\log(30)$",r"$\log(35)$",r"",r""]
axs[2,1].set_xticks(xtick_loc)
axs[2,1].set_xticklabels(xtick_label, fontsize=fontsize)
axs[2,1].set_xlim(np.log10([20,45]))
axs[2,1].set_ylabel(r"$\log\mathcal{R}(t)$", fontsize=fontsize)
axs[2,1].yaxis.set_label_coords(-0.0175, 0.85)



axs[2,1].set_xlabel(r"$\log(age)$", fontsize=fontsize)
axs[2,1].xaxis.set_label_coords(xlabelX, -0.03)



ytick_loc = np.array([1,2,3,4,5,6,7,8,9])
ytick_label =        [r"1", r"2",r"3", r"4",r"5",r"",r"",r"",r""]


for index, ax in np.ndenumerate(axs):
    if index[0]==0:
        common_top_visuals(ax)
    elif index[0]==1:
        common_mid_visuals(ax)
    elif index[0]==2:
        common_log_visuals(ax)


relX = -0.038
relY = 1.03
axs[0,0].text(relX , relY, '(a)', ha='center', va='center', transform=axs[0,0].transAxes, fontsize=fontsize-1)
axs[0,1].text(relX , relY, '(b)', ha='center', va='center', transform=axs[0,1].transAxes, fontsize=fontsize-1)
axs[1,0].text(relX , relY, '(c)', ha='center', va='center', transform=axs[1,0].transAxes, fontsize=fontsize-1)
axs[1,1].text(relX , relY, '(d)', ha='center', va='center', transform=axs[1,1].transAxes, fontsize=fontsize-1)
axs[2,0].text(relX , relY, '(e)', ha='center', va='center', transform=axs[2,0].transAxes, fontsize=fontsize-1)
axs[2,1].text(relX , relY, '(f)', ha='center', va='center', transform=axs[2,1].transAxes, fontsize=fontsize-1)

fig.legend([black_bar, blue_bar, red_bar, green_bar, orange_bar],
           [r"$\mathcal{R}(t)$ obtained from data for year 2016",
            r"$\mathcal{R}(t)$ obtained from data from years 1994-2021",
            r"$\hat{\mathcal{R}}(t)$ obtained from the complete model",
            r"$\mathcal{R}_{fit}(t)$ post-menopause linear fit in the lin-lin scale",
            r"$\mathcal{R}_{fit}(t)$ post-menopause cubic fit in the lin-lin scale",
            r"linear fit of pre-menopause $\mathcal{R}(t)$ in the log-log scale",
            ], loc='lower center', fontsize=fontsize, ncol=2)

filename = r"sup_figure_5"
plt.savefig(filename, bbox_inches='tight', dpi =300)
plt.savefig(filename+".pdf", bbox_inches='tight')

