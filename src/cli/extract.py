from dataclasses import dataclass, fields, MISSING
from argparse import ArgumentParser
from pathlib import Path
from datasets import Dataset, DatasetDict, Features, Value, ClassLabel, Sequence
from tqdm import tqdm
from disutapa.grammar.extraction.corpus import corpus_extractor, ExtractionParameter
from disutapa.grammar.extraction.nonterminal import read_clusters

from discodop.treebank import READERS, CorpusReader  # type: ignore
from os.path import exists


def splitstr(s: str) -> dict:
    return eval(s)


preset_splits = {
    "negra": { "train": range(18602), "test": range(18602, 19602), "dev": range(19602, 20602) },
    "dptb": { "train": range(3914, 43746), "dev": range(43746, 45446), "test": range(45446, 47862) },
    "tiger": { "train": range(40472), "dev": range(40472, 45472), "test": range(45472, 50472) },
    "alpinosample": { "train": range(3), "test": range(2, 3) },
}


@dataclass
class CliParams:
    input: str
    output: str = None
    split: splitstr = None # e.g. "dict(train=range(18602), dev=range(18602, 19602), test=range(19602, 20602))"
    headrules: str = None
    override: bool = False


def main(config):
    if config.output is None:
        config.output = f"/tmp/{Path(config.input).stem}"
    elif exists(config.output) and not config.override:
        print(f"Specified output destination '{config.output}' already exists. Change it, remove it or start the app with '--override'.")
        exit(1)

    filetype = config.input.split(".")[-1]
    if filetype == "xml":
        filetype = "tiger"
    encoding = "iso-8859-1" if filetype == "export" else "utf8"
    trees: CorpusReader = READERS[filetype](config.input, encoding=encoding, punct="move", headrules=config.headrules)

    if not config.coarsents is None:
        config.coarsents = read_clusters(config.coarsents)
    ex = corpus_extractor(config)
    
    splitdict = config.split or next(preset_splits[k] for k in preset_splits if k in config.input.lower())
    datasets = {}
    for split, portion in splitdict.items():
        dataset = { "sentence": [], "supertag": [], "pos": [], "tree": [] }
        total = portion.stop-portion.start
        desc = f"extracting {split} portion"
        for _, corpusobj in tqdm(trees.itertrees(portion.start, portion.stop), total=total, desc=desc):
            rules, pos = ex.read_tree(corpusobj.tree)
            sentence = corpusobj.sent if not "ptb" in config.input else \
                corpus_extractor.ptb_sentence(corpusobj.sent)
            dataset["sentence"].append(list(sentence))
            dataset["supertag"].append(list(rules))
            dataset["pos"].append(list(pos))
            dataset["tree"].append(str(corpusobj.tree))
        datasets[split] = Dataset.from_dict(dataset, features=Features({
            "sentence": Sequence(Value("string")),
            "supertag": Sequence(ClassLabel(names = list(ex.rules.keys()))),
            "pos": Sequence(ClassLabel(names = list(ex.postags.keys()))),
            "tree": Value("string"),
        }))

    if not "dev" in datasets:
        datasets["dev"] = datasets["train"]
    DatasetDict(**datasets).save_to_disk(config.output)


def subcommand(sub: ArgumentParser):
    for f in fields(ExtractionParameter) + fields(CliParams):
        required = f.default is MISSING and f.default_factory is MISSING
        name = f.name if required or f.name == "output" else f"--{f.name}"
        default = None if required else (f.default if not f.default is MISSING else f.default_factory())
        if f.type is bool:
            sub.add_argument(name, action="store_true", default=False)
        else:
            sub.add_argument(name, type=f.type, default=default, nargs="?" if name=="output" else None)
    sub.set_defaults(func=lambda args: main(args))
