from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='Hybrid Grammar Supertagging',
    version="0.1",
    ext_modules=cythonize("sdcp/grammar/parser/activeparser.pyx"),
    include_dirs=[numpy.get_include()],
    packages=["sdcp"],
    package_dir={"sdcp": "sdcp"},
    entry_points={"console_scripts": ["sdcp=sdcp.cli:main"]},
    install_requires=[
        "sortedcontainers>=2.4.0",
        "bitarray>=2.7",
        "datasets>=2.10",
        "flair>=0.12.2",
        "torch>=2.0",
        "cython>=3.0"
    ],
    setup_requires=[
        "setuptools",
        "wheel",
        "cython>=3.0"
    ]
)