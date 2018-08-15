from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext


setup(
	maintainer='Maria Tsantaki and Daniel Andreasen',
    name='specML',
	version=0.1,
    packages=find_packages(),
    url='https://github.com/MariaTsantaki/spectroscopy-ML',
    cmdclass={'build_ext': build_ext},
	package_data={'specML': ['data/combined_spec.hdf',
	                         'FASMA_large_ML.pkl']}
)
