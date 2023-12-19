# Disutapa

A discontinuous constituent parser based on supertagging.


## How does this parser work?

The parser uses a discriminative classifier to predict lexical grammar rules (supertags) for each token in each input sentence. These rules are interpreted as a weighted grammar with the softmax prediction scores as pseudo-probabilistic weight values. A statistical parser for the grammar finds a derivation of grammar rules. Lastly, this derivation is converted into a constituent tree.

Before parsing, an extraction/training procedure converts a provided constituent treebank into a corpus of lexical grammar rules and trains a specified neural model for the prediction. This repository does not provide any treebanks. [This paper](https://aclanthology.org/2022.findings-emnlp.105/) describes the process in more detail.

## Build

The project was developed and tested using python 3.9.
We strongly recommend using a conda (or virtualenv) environment when running it:

    conda create -n supertags python && conda activate supertags

Build and install all dependencies, build the source files and install the package and executable

    pip install .

## Usage

The above instructions install an executable called `disutapa`. There are several subcommands for the extraction, training, evaluation, and so on. Each subcommand has its own set of parameters, which can be printed using the `--help` flag. The following three commands use example files to extract a tiny corpus of grammar rules, train a classifier and evaluate the prediction/parsing process on a test portion of the data:

    disutapa extract resources/disco-dop/alpinosample.export data/sample
    disutapa train data/sample model/sample --lr 0.01
    disutapa eval --dev model/sample/best-model.pt data/sample

These three commands should work out-of-the box. For a real-world example, the filename `resources/disco-dop/alpinosample.export` is swapped in favor of a real treebank.