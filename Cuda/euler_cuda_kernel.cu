#include <torch/extension.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <assert.h>
#include <random>
#include <chrono>
#define MIN 2
#define MAX 7
#define ITER 10000000
#define year 365
#define T 80 * year
#define BLOCKSIZE 512
#define THREADSIZE 1024
#define sd_meno 4.86

namespace{
__global__ void setup_kernel(curandState *state)
{
  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  curand_init(1234, idx, 0, &state[idx]);

}


__global__ void euler_hrt_kernel(curandState *state,
                                double *y,
                                double p_low,
                                double p_inc,
                                double H_min,
                                double H_max,
                                double r,
                                double nu,
                                double mu)
{
    bool no_cancer = true;
    double p;
    double n = 10e9;
    double X = 0.0;
    double rr;
    int phase;
    double H_eff;
    double f;
    double diameter = 1.8;
    double radius = diameter / 2;
    double x_detect = (4./3.)* M_PI* pow(radius,3)* 1e9;

    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Copy state to local memory for efficiency */
    curandState localState = state[id];
    int less = lroundf(curand_normal(&localState)*2.4);
    double meno = 51.+curand_normal(&localState)*sd_meno; //= 51;

    if(y[id]!=0.0f){
      meno = meno +y[id];
    }

    for(int i = 1; i < T; i++) {
      p = p_low + i * p_inc;
      if ( i <12* year || i > meno * year){
        H_eff = H_max;
      }else{
        phase = i % 28-less;
        if (phase <14- less){
          H_eff = H_max;
        }else{
          H_eff = H_min;
        }
      }
      f = nu * r * X - nu * mu * X *pow(X, 1. / nu) - H_eff;
      rr = (double)curand_poisson(&localState, n * p);
      X = X + f + rr;
      if (X> x_detect && no_cancer){
        y[id]=int(i/year);
        no_cancer =false;
      }
      if (X>0){
        X = X;
      }else{
        X = 0;
      }
    }
    if (no_cancer){
      y[id]=-1;
    }
    /* Copy state back to global memory */
    state[id] = localState;
}

__global__ void euler_dist_meno_kernel(curandState *state,
                                double *y,
                                double p_low,
                                double p_inc,
                                double H_min,
                                double H_max,
                                double r,
                                double nu,
                                double mu)
{
    bool no_cancer = true;
    double p;
    double n = 10e9;
    double X = 0.0;
    double rr;
    int phase;
    double H_eff;
    double f;
    double diameter = 1.8;
    double radius = diameter / 2;
    double x_detect = (4./3.)* M_PI* pow(radius,3)* 1e9;

    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curandState localState = state[id];
    int less = lroundf(curand_normal(&localState)*2.4);
    double meno = 51.+curand_normal(&localState)*sd_meno; //= 51;
    for(int i = 1; i < T; i++) {
      p = p_low + i * p_inc;
      if ( i <12* year || i > meno * year){
        H_eff = H_max;
      }else{
        phase = i % 28-less;
        if (phase <14- less){
          H_eff = H_max;
        }else{
          H_eff = H_min;
        }
      }
      f = nu * r * X - nu * mu * X *pow(X, 1. / nu) - H_eff;
      rr = (double)curand_poisson(&localState, n * p);
      X = X + f + rr;
      if (X> x_detect && no_cancer){
        y[id]=int(i/year);
        no_cancer =false;
      }
      if (X>0){
        X = X;
      }else{
        X = 0;
      }
    }
    if (no_cancer){
      y[id]=-1;
    }
    /* Copy state back to global memory */
    state[id] = localState;
}


__global__ void euler_51_kernel(curandState *state,
                                double *y,
                                double p_low,
                                double p_inc,
                                double H_min,
                                double H_max,
                                double r,
                                double nu,
                                double mu)
{
    bool no_cancer = true;
    double p;
    double n = 10e9;
    double X = 0.0;
    double rr;
    int phase;
    double H_eff;
    double f;
    double diameter = 1.8;
    double radius = diameter / 2;
    double x_detect = (4./3.)* M_PI* pow(radius,3)* 1e9;

    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Copy state to local memory for efficiency */
    curandState localState = state[id];
    int less = lroundf(curand_normal(&localState)*2.4);
    int meno = 51;
    for(int i = 1; i < T; i++) {
      p = p_low + i * p_inc;
      if ( i <12* year || i > meno * year){
        H_eff = H_max;
      }else{
        phase = i % 28-less;
        if (phase <14- less){
          H_eff = H_max;
        }else{
          H_eff = H_min;
        }
      }
      f = nu * r * X - nu * mu * X *pow(X, 1. / nu) - H_eff;
      rr = (double)curand_poisson(&localState, n * p);
      X = X + f + rr;
      if (X> x_detect && no_cancer){
        y[id]=int(i/year);
        no_cancer =false;
      }
      if (X>0){
        X = X;
      }else{
        X = 0;
      }
    }
    if (no_cancer){
      y[id]=-1;
    }
    /* Copy state back to global memory */
    state[id] = localState;
}
}

curandState *devStates;
double *x;

torch::Tensor euler_dist_meno(double p_low,
                      double p_up,
                      double H_min,
                      double H_max,
                      double r,
                      double nu,
                      double mu){
  int blockSize = BLOCKSIZE;
  int threadSize =THREADSIZE;
  double p_inc;
  p_inc = (p_up-p_low)/(365.*80.);
  int N = blockSize*threadSize;
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
  }
  euler_dist_meno_kernel<<<blockSize, threadSize>>>(devStates, x, p_low, p_inc, H_min, H_max, r, nu, mu);
  cudaDeviceSynchronize();
  auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA, 0).pinned_memory(true);
  torch::Tensor myt = torch::from_blob(x, {N,}, options);
  return myt;
}

std::random_device rd;
std::mt19937 gen(rd());

torch::Tensor euler_hrt(double p_low,
                      double p_up,
                      double H_min,
                      double H_max,
                      double r,
                      double nu,
                      double mu,
                      double hrt_frac){
  int blockSize = BLOCKSIZE;
  int threadSize =THREADSIZE;
  double mg = 3.2553;
  double sg = 2.66;
  std::gamma_distribution<> d(mg, sg);
  double p_inc;
  p_inc = (p_up-p_low)/(365.*80.);
  int N = blockSize*threadSize;
  for (int i = 0; i < N; i++) {
    if (i < hrt_frac*N){
      x[i] = d(gen);
    }
    else{
      x[i] = 0.0f;
    }
    //y[i] = 2.0f;
  }
  euler_hrt_kernel<<<blockSize, threadSize>>>(devStates, x, p_low, p_inc, H_min, H_max, r, nu, mu);
  cudaDeviceSynchronize();
  auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA, 0).pinned_memory(true);
  torch::Tensor myt = torch::from_blob(x, {N,}, options);
  return myt;
}



torch::Tensor init_cuda(double p_low,
                      double p_up,
                      double H_min,
                      double H_max,
                      double r,
                      double nu,
                      double mu)
{
  int blockSize = BLOCKSIZE;
  int threadSize =THREADSIZE;

  double p_inc;
  p_inc = (p_up-p_low)/(365.*80.);
  printf("P inc %.16f", p_inc);
  /* Allocate space for prng states on device */
  cudaMalloc((void **)&devStates, blockSize * threadSize * sizeof(curandState));

  int N = blockSize*threadSize;
  cudaMallocManaged(&x, N*sizeof(double));


   // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
  }

  setup_kernel<<<blockSize, threadSize>>>(devStates);
  euler_51_kernel<<<blockSize, threadSize>>>(devStates, x, p_low, p_inc, H_min, H_max, r, nu, mu);
  cudaDeviceSynchronize();
  auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA, 0).pinned_memory(true);
  torch::Tensor myt = torch::from_blob(x, {N,}, options);
  return myt;
}