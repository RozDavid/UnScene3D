from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='raycast_cuda',
    version='2.2',
    ext_modules=[
        CUDAExtension('raycast_cuda', [
            'raycast_cuda.cpp',
            'raycast_cuda_kernel.cu',
        ]),
        CUDAExtension('project_features_cuda', [
            'project_image_cuda.cpp',
            'project_image_cuda_kernel.cu',
        ]),
        CUDAExtension('custom_cuda_utils', [
            'cuda_utils.cpp',
            'cuda_utils_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
