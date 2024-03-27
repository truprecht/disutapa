from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy
from sys import argv

ext_modules = cythonize(["src/grammar/parser/*.pyx", "src/grammar/composition.pyx", "src/tagging/parser_adapter.pyx"])
if "-e" in argv or "--editable" in argv:
    for mod in ext_modules:
        mod.name = mod.name.replace("src", "disutapa", 1)
packages = [pak.replace("src", "disutapa", 1) for pak in find_packages()]

setup(
    name='disutapa',
    description="Discontinuous supertag-based parsing",
    long_description="This is a prototype implementation for the extraction of supertags from discontinuous constituent treebanks, as well as the training, prediction, and parsing with such supertags.",
    version="3.2",
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include()],
    packages=packages,
    package_dir={"disutapa": "src"},
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
