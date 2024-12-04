import time
from lmfit import minimize, Parameters, Parameter, report_fit, fit_report
import numpy as np
import torch
import euler_cuda

year = 365
age = 80
T = age * year

nu = 10.
r =  2.78777026e-03
mu =  1.57079970e-04

def residual(ps, cum_risk):
    p_low = ps['p_low'].value * 1e-5
    p_up = ps['p_up'].value * 1e-5
    H_max = ps['H_max'].value


    allArrays = np.array([])
    for gpu_i in range(1):
        mytensor = euler_cuda.euler_51(p_low, p_up, H_max, H_max, r, nu, mu)
        out = mytensor.cpu().numpy()
        allArrays = np.concatenate((allArrays, out))


    data = allArrays

    HIST_BINS = np.linspace(0, age, 17)
    HIST_BINS = np.insert(HIST_BINS, 0, -1, axis=0)
    hist, bin_edges = np.histogram(data, HIST_BINS)
    rawT = np.arange(0, age, 5)
    pp = np.zeros(rawT.shape[0])

    for i in range(1, hist.shape[0]):
        pp[i - 1] = hist[i] / (data.shape - np.sum(hist[1:i]))

    if np.isnan(pp).any():
        return np.ones(13)

    cum_prob = 1 - np.cumprod(1 - pp)
    diff = cum_prob[3:] - cum_risk[3:]
    return diff.ravel()


if __name__ == '__main__':
    start = time.time()
    print("hello")
    print(" ")
    incidences_data = np.loadtxt("Colorectal_C18C21_females_in_Ireland_19942021_20240528.txt", delimiter=",")
    incidences_data = np.delete(incidences_data, 2, 1)
    age_adjusted_rates = incidences_data[:-2, 2] / 100000
    cum_rate1 = np.cumsum(5 * age_adjusted_rates)
    cum_risk1 = 1 - np.exp(-cum_rate1)

    params = Parameters()

    p_low_init = 2.73943844
    p_up_init =  2.7402675
    H_max_init = 274250



    H_min_init = H_max_init
    params.add('p_low', value=p_low_init, min=p_low_init - 0.001, max=p_low_init + 0.001, vary=True)
    params.add('p_up', value=p_up_init, min=p_up_init - 0.001, max=p_up_init + 0.001, vary=True)
    params.add('H_max', value=H_max_init, min=H_max_init - 10, max=H_max_init + 10, vary=True)


    mytensor = euler_cuda.init_cuda(p_low_init * 1e-5, p_up_init* 1e-5, H_max_init, H_max_init, r, nu, mu)
    print(mytensor)
    dummy = mytensor.cpu().numpy()

    result = minimize(residual, params, args=((cum_risk1,)), ftol=1.e-12, xtol=1.e-12 )

    report_fit(result)

    # save fit report to a file:
    with open('fit_crc_EXCLUDE_5_10.txt', 'w') as fh:
        fh.write(fit_report(result))
        fh.write("\n")
        fh.write("p_low " + str(np.format_float_positional(result.params["p_low"], unique=False, precision=15)))
        fh.write("\n")
        fh.write("p_up " + str(np.format_float_positional(result.params["p_up"], unique=False, precision=15)))

    print("p_low", np.format_float_positional(result.params["p_low"], unique=False, precision=15))
    print("p_up", np.format_float_positional(result.params["p_up"], unique=False, precision=15))

    end = time.time()
    print("duration:", end - start)