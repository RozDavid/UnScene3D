import sys
from glob import glob

from pybind11 import get_cmake_dir
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

__version__ = "1.2.0"


ext_modules = [
    Pybind11Extension("felzenszwalb_cpp",
                      ["segmentator.cpp"],
                      # Example: passing in the version to the compiled code
                      define_macros = [('VERSION_INFO', __version__)],
                      ),
]

setup(
    name='felzenszwalb_cpp',
    version=__version__,
    author='David Rozenberszki',
    author_email='david.rozenberszki@tum.de',
    description='Wrapping efficient oversegmentation on python module',
    long_description="",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)