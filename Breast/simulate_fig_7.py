import torch
import euler_cuda
import numpy as np

nu = 10
r = 2.78777026e-03
mu = 1.57079970e-04
h = 2.39320322e+02


mytensor =euler_cuda.init_cuda(p_low,p_up, H_min, H_max,r ,nu,mu)



mytensor = euler_cuda.euler_dist_meno(p_low, p_up, H_min, H_max,r ,nu,mu)
mytensor = mytensor.cpu().numpy()
np.savetxt("out_strict_meno.txt", mytensor .T)


mytensor = euler_cuda.euler_hrt(p_low, p_up, H_min, H_max,r ,nu,mu, 0.26)
mytensor = mytensor.cpu().numpy()
np.savetxt("out_hrt.txt", mytensor .T)