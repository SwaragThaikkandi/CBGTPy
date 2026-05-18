from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="cbgtpy.nchoice.agent_timestep_plasticity",
        sources=["cbgtpy/nchoice/agent_timestep_plasticity.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        name="cbgtpy.stopsignal.agent_timestep_stop_signal",
        sources=["cbgtpy/stopsignal/agent_timestep_stop_signal.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    ext_modules=cythonize(extensions),
)
