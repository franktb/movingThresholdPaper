#include <torch/extension.h>


torch::Tensor init_cuda (double, double, double,double, double, double, double);
torch::Tensor euler_51 (double, double, double,double, double, double, double);
torch::Tensor euler_dist_meno (double, double, double,double, double, double, double);
torch::Tensor euler_hrt (double, double, double,double, double, double, double, double);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("euler_51", &euler_51, "euler_51");
  m.def("euler_hrt", &euler_hrt, "euler_hrt");
  m.def("euler_dist_meno", &euler_dist_meno, "euler_dist_meno");
  m.def("init_cuda", &init_cuda, "Initialise Cuda");
}