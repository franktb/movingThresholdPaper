from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='euler_cuda',
    ext_modules=[
        CUDAExtension('euler_cuda', [
            'euler.cpp',
            'euler_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
