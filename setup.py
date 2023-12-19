from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='Discontinuous Supertagging and Parsing',
    version="3.2",
    ext_modules=cythonize(["disutapa/grammar/parser/*.pyx", "disutapa/grammar/composition.pyx", "disutapa/tagging/parser_adapter.pyx"]),
    include_dirs=[numpy.get_include()],
    packages=["disutapa"],
    package_dir={"disutapa": "disutapa"},
    entry_points={"console_scripts": ["disutapa=disutapa.cli:main"]},
    install_requires=[
        "sortedcontainers>=2.4.0",
        "bitarray>=2.7",
        "datasets>=2.10",
        "flair>=0.13",
        "torch>=2.0",
        "cython>=3.0",
        "disco-dop"
    ],
    setup_requires=[
        "setuptools",
        "wheel",
        "cython>=3.0"
    ]
)