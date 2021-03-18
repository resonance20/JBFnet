from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='nlmeans_cpp',
      ext_modules=[cpp_extension.CppExtension('nlmeans_cpp', ['nlmeans.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})