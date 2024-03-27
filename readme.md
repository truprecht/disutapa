# Disutapa

A discontinuous constituent parser based on supertagging.


## How does this parser work?

The parser uses a discriminative classifier to predict lexical grammar rules (supertags) for each token in each input sentence. These rules are interpreted as a weighted grammar with the softmax prediction scores as pseudo-probabilistic weight values. A statistical parser for the grammar finds a derivation of grammar rules. Lastly, this derivation is converted into a constituent tree.

Before parsing, an extraction/training procedure converts a provided constituent treebank into a corpus of lexical grammar rules and trains a specified neural network model for the prediction. [This paper](https://aclanthology.org/2022.findings-emnlp.105/) describes the process in more detail.

## Build

The project was developed and tested using python 3.10 and pip>=22.
We strongly recommend using a conda (or venv) environment when running it:

    conda create -n supertags python && conda activate supertags
    # or: python -m venv ./virtualenv && . ./virtualenv/bin/activate

Build and install all dependencies, build the source files and install the package and executable.
If you are running a machine without a discrete gpu, then you probably want to install the (much smaller) cpu version of pytorch; skip this step if you want to run the gpu implementation of torch.

    # (optional) install cpu version of pytorch: pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install "cython>=3.0" numpy setuptools wheel && pip install --no-build-isolation .

## Usage

The above instructions install an executable called `disutapa`. There are several subcommands for the extraction, training, evaluation, and so on. Each subcommand has its own set of parameters, which can be printed using the `--help` flag. The following three commands use example files to extract a tiny corpus of grammar rules, train a classifier and evaluate the prediction/parsing process on a test portion of the data:

    disutapa extract resources/disco-dop/alpinosample.export data/sample
    disutapa train data/sample model/sample --lr 0.01 --ktags 2
    disutapa eval model/sample/best-model.pt data/sample

These three commands should work out-of-the box. For a real-world example, the filename `resources/disco-dop/alpinosample.export` is swapped in favor of a real treebank.

### Extraction Parameters

The extraction process is parametrized by a grammar formalism (`--composition`), rank transformation (eg. binarization) strategy (`--factor`) and Markovization (`--vmarkov` and `--hmarkov`) as well as a guide (`--guide`) and nonterminal constructor (`--nts`); certain rank transformations and guide constructors require head assignments that are supplied in a file passed with `--headrules <file>`). The extraction procedure creates three subcorpora for the training/development/test split, respectively; the split is specified with `--split`.
The full list of parameters and valid values are as follows:
  * `--hmarkov <h>` where `<h>` is an integer ≥ 0; in the case of `--guide head --hmarkov 999` binarization is disabled
  * `--vmarkov <v>` where `<v>` is an integer ≥ 1
  * `--factor <strategy>` where `<strategy>` is one of `right`, `left` or `headoutward`; `headoutward` requires a head assignment with `--headrules`; if either the head or dependent guide constructor is used with `--guide head` or `--guide dependent`, then this option has no effect
  * `--guide <constructor>` where `<constructor>` is one of `strict`, `vanilla`, `dependent`, `head`, `least`, `near`; in the case of `--guide head` or `--guide dependent`, a head assignment must be passed with with `--headrules`
  * `--headrules <file>` specifies rules for head assignments, there are default files in `resources/disco-dop/` for NEGRA and DPTB, we use the file `resources/disco-dop/negra.headrules` for NEGRA as well as TIGER
  * `--nts <constructor>` where `<constructor>` is one of `vanilla`, `classic`, `coarse`; if the coarse nonterminal constructor is used with `--nts coarse`, then an optional table for nonterminal clusters may be provided with `--coarsents <file>` where `<file>` is a filename, some examples are proveded in `resources/coarse-constituents.clusters`, `resources/xtag.clusters` and `resources/stts.clusters`
  * `--composition <comp>` where `<comp>` is one of `lcfrs` or `dcp`, the option `lcfrs` creates hybrid grammar rules (with lcfrs and dcp compositions) and the `dcp` option drops the lcfrs compositions to extract `dcp` rules
  * `--split dict(train=range(t₁,t₂), dev=range(d₁, d₂), test=range(f₁, f₂))` where `t₁, t₂, d₁, d₂, f₁, f₂` specify the trees in each split by index, e.g. in the case of NEGRA `--split dict(train=range(0,18602), test=range(18602, 19602), dev=range(19602, 20602))`, there are default values for NEGRA, TIGER and DPTB such that this parameter does not need to be passed
  * the first positional parameter is the file for the treebank supplied in the usual format (NEGRA in export and iso-8859-1 encoding, TIGER in xml format, and DPTB in export format)
  * the second and optional positional parameter is the output location where the extraction script creates a directory

Using the defaults values, a call of

    disutapa extract negra.export

is equivalent to

    disutapa extract --hmarkov 999 --vmarkov 1 --factor right --guide strict --nts classic --composition lcfrs --split dict(train=range(0,18602), test=range(18602, 19602), dev=range(19602, 20602)) negra.export /tmp/negra

### Extracted Supertags

The result of the `extract` subcommand is already split into a training, dev and test portion and can be read using the `datasets` python library as follows:

    disutapa extract resources/disco-dop/alpinosample.export data/sample
    python
    >>> from datasets import DatasetDict
    >>> corpus = DatasetDict.load_from_disk("data/sample")
    
The `corpus` object is a python dictionary object containing the three portions, each portion is a dataset where each data point consists of a sentence, a sequence of supertag blueprints, a sequence of pos tags and a constituent tree:

    >>> corpus
    DatasetDict({
        train: Dataset({
            features: ['sentence', 'supertag', 'pos', 'tree'],
            num_rows: 2
        })
        dev: Dataset({
            features: ['sentence', 'supertag', 'pos', 'tree'],
            num_rows: 1
        })
        test: Dataset({
            features: ['sentence', 'supertag', 'pos', 'tree'],
            num_rows: 2
        })
    })

The list of supertag blueprints can be accessed for each portion as follows:

    >>> corpus["train"].features["supertag"].feature.names
    ["rule('arg(PP)')", "rule('PP/1', ('arg(PP)', -1), dcp=sdcp_clause('(PP 0 2)', args=(1,)))", ...
    >>> len(corpus["train"].features["supertag"].feature.names)
    39
    
Each element in the list is a string representation that can be deserialized into a python object after loading the necessary libraries:

    from disutapa.grammar.sdcp import rule, sdcp_clause, grammar
    from disutapa.grammar.composition import Composition
    blueprints = [eval(bstr) for bstr in corpus["train"].features["supertag"].feature.names]

### Training Parameters

The training subcommand provides a set of parameters for the neural network architecture, its training and the parameters for parsing with the predicted supertags during the evaluation passes after each batch and at the end of the training. The parameters are as follows:
  * `--epochs <n>` where `<n>` is the number of training epochs (default: 32)
  * `--lr <l>` where `<l>` is the base learning rate  (default: 5e-5)
  * `--batch <b>` where `<b>` is the batch size (number of sentences per training iteration, default: 32)
  * `--micro_batch <b>` where `<b>` is the micro batch size (evaluates the neural network predictions with even smaller samples than `--batch`, default: None)
  * `--weight_decay <w>` where `<w>` is the weight decay factor (default: 1e-2)
  * `--optimizer <o>` where `<o>` is an optimizer class in `torch.optim` (https://pytorch.org/docs/stable/optim.html, default: `AdamW`)
  * `--random_seed <r>` where `<r>` is an integer, determines the initializations for the neural network modules (default: 0)
  * `--patience <p>` where `<p>` is an integer ≥ 0; after `<p>`+1 epochs without an increase in parsing accuracy, the learning rate is multiplied with 0.2 (default: 2)
  * `--embedding <e>` where `<e>` is either a model supplied by huggingface (https://huggingface.co/models), `flair`, `fasttext` or `Supervised` for supervised word and character embeddings (default: `Supervised`)
  * `--dropout <d>` determines the dropout probability `<p>` (default: 0.1)
  * `--lstm_layers <l>` determines the number `<lp>` of bidirectional lstm layers above the embedding (default: 0)
  * `--lstm_size <s>` determines the output dimension `<s>` of all lstm layers (default: 512)
  * `--ktags <n>` where `<n>` is a positive integer that determines how many best supertags per position are used for parsing (default: 1)
  * `--step <beta>` where `<beta>` is a positive number that defines a step size for the incremental parsing process (default: 2)
  * a boolean flag `--parameter-search` prevents that neural network models are saved and just writes the results of evaluation passes and the training log to the output location  (default: off)
  * the first positional parameter is the output directory for an extracted corpus and serves as the input for this training procedure
  * the second and optional positional parameter is the output location (default: /tmp/disutapa-training)
  
The subcommand `eval` uses the same parameters `--ktags` and `--step` as above, and evaluates the prediction and parsing using a test portion in a provided corpus.

### Grid search

The `grid` subcommand implements grid searches using parameter sets specified in configuration files.
There are examples provided in resources/gridfiles; each such file defines variables that are assigned to lists of all possible values that they assume during the search, and constants that must be specified in the call of the command in the form `disutapa grid <file> CONSTANT1=VALUE1 CONSTANT2=VALUE2 ...`.

### Reranking with DOP

The `dop` subcommand extracts a discontinuous data-oriented parsing model for reweighting constituent trees.
Providing the Markovization parameters `--bin`(for the strategy `right`, `left` or `headoutward`), `--hmarkov` and `--vmarkov` as above will binarize the trees before extracting fragments.
The output file given in the second positional argument of the command contains all extracted fragments and probability assignments.
It can be passed to the `eval` subcommand using the `--reranking` option and by specifying the number of predicted constituent tree to rerank using `--ktrees`.

## Data

This repository does not provide any treebanks.
The extraction does include preconfigured parameters for [Negra](http://www.coli.uni-saarland.de/projects/sfb378/negra-corpus/corpus-license.html), [Tiger](https://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/tiger/) and the discontinuous Penn Treebank (distributed via [Kilian Evang](https://kilian.evang.name/)).
The Tiger treebank is patched by removing overfluous links (cf. https://github.com/mcoavoux/multilingual_disco_data/blob/master/generate_tiger_data.sh) as follows:

    sed -e "3097937d;3097954d;3097986d;3376993d;3376994d;3377000d;3377001d;3377002d;3377008d;3377048d;3377055d" tiger.xml > tiger-fixed.xml
