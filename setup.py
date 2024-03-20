from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy

setup(
    name='Disutapa',
    description="Discontinuous supertag-based parsing",
    long_description="This is a prototype implementation for the extraction of supertags from discontinuous constituent treebanks, as well as the training, prediction, and parsing with such supertags.",
    version="3.2",
    ext_modules=cythonize(["disutapa/grammar/parser/*.pyx", "disutapa/grammar/composition.pyx", "disutapa/tagging/parser_adapter.pyx"]),
    include_dirs=[numpy.get_include()],
    packages=find_packages(),
    package_dir={"disutapa": "disutapa"},
    entry_points={"console_scripts": ["disutapa=disutapa.cli:main"]},
    install_requires=[
        "sortedcontainers>=2.4.0",
        "bitarray>=2.7",
        "datasets>=2.10",
        "flair>=0.13",
        "torch>=2.0",
        "cython>=3.0",
        "disco-dop @ git+https://github.com/andreasvc/disco-dop.git",
        "numpy",
        "grapheme" # disco-dop misses this dependency
    ],
    setup_requires=[
        "setuptools",
        "wheel",
        "cython>=3.0",
        "numpy"
    ]
)
