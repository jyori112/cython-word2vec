from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [
        Extension("word2vec.data", ["word2vec/data.pyx"], include_dirs=[np.get_include()]),
        Extension("word2vec.train", ["word2vec/train.pyx"], include_dirs=[np.get_include()])]
)
