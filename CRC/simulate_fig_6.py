import torch
import euler_cuda
import numpy as np

nu = 10.
r =  2.78777026e-03
mu =  1.57079970e-04


mytensor =euler_cuda.init_cuda(2.739905e-05,2.74106e-05, 274300., 274300.,r ,nu,mu)

p_low = 2.73943844e-05
p_up = 2.74027764e-05
H_max =  274250.001

mytensor = euler_cuda.euler_51(p_low, p_up, H_max, H_max,r ,nu,mu)
mytensor = mytensor.cpu().numpy()
np.savetxt("out_strict_CRC.txt", mytensor .T)