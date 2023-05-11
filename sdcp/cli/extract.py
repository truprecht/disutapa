from dataclasses import dataclass, field, fields, MISSING
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable
from datasets import Dataset, DatasetDict, Features, Value, ClassLabel, Sequence
from tqdm import tqdm
from sdcp.corpus import corpus_extractor, Split

def splitstr(s: str) -> dict:
    return eval(s)

@dataclass
class ExtractionParameter:
    corpus: str
    output: str = None
    split: splitstr = None # e.g. "dict(train=range(18602), dev=range(18602, 19602), test=range(19602, 20602))"
    hmarkov: int = 999
    vmarkov: int = 1
    rightmostunary: bool = False
    headrules: str = None
    coarsents: str = None
    bindirection: bool = False
    composition: str = "lcfrs"


preset_splits = {
    "negra": { "train": range(18602), "dev": range(18602, 19602), "test": range(19602, 20602) },
    "dptb": { "train": range(3914, 43746), "dev": range(43746, 45446), "test": range(45446, 47862) },
    "tiger": { "train": range(40472), "dev": range(40472, 45472), "test": range(45472, 50472) },
    "alpinosample": { "train": range(2), "dev": range(2), "test": range(2, 3) },
}
        

def main(config: ExtractionParameter):
    ex = corpus_extractor(config.corpus,
            horzmarkov=config.hmarkov, vertmarkov=config.vmarkov, headrules=config.headrules,
            rightmostunary=config.rightmostunary, coarselabels=config.coarsents, bindirection=config.bindirection, cmode=config.composition)
    splitdict = config.split or next(preset_splits[k] for k in preset_splits if k in config.corpus.lower())
    for r in Split(**splitdict).nonoverlapping():
        ex.read(r)
    rules = list(ex.rules)
    datasets = {}
    for split, portion in splitdict.items():
        dataset = { "sentence": [], "supertag": [], "pos": [], "tree": [], "derivation": [] }
        total = portion.stop-portion.start
        desc = f"extracting {split} portion"
        for idx in tqdm(range(portion.start, portion.stop), total=total, desc=desc):
            gtree, sentence, gpos, grules, gderiv = ex[idx]
            dataset["sentence"].append(list(sentence))
            dataset["supertag"].append([repr(rules[r]) for r in grules])
            dataset["pos"].append(list(gpos))
            dataset["tree"].append(str(gtree))
            dataset["derivation"].append(str(gderiv))
        datasets[split] = dataset
    tagsets = {
        "train": {
            "supertag": set(tag for tagseq in datasets["train"]["supertag"] for tag in tagseq),
            "pos": set(tag for tagseq in datasets["train"]["pos"] for tag in tagseq)
        }
    }
    for split in splitdict:
        if split == "train": continue
        tagsets[split] = {}
        for field in ("supertag", "pos"):
            newtags = set(tag for tagseq in datasets[split][field] for tag in tagseq) - tagsets["train"][field]
            tagsets[split][field] = list(tagsets["train"][field]) + list(newtags)
    
    datasets = {
        split: Dataset.from_dict(dataset, features=Features({
            "sentence": Sequence(Value("string")),
            "supertag": Sequence(ClassLabel(names = list(tagsets[split]["supertag"]))),
            "pos": Sequence(ClassLabel(names = list(tagsets[split]["pos"]))),
            "tree": Value("string"),
            "derivation": Value("string")
        }))
        for split, dataset in datasets.items()
    }
    DatasetDict(**datasets).save_to_disk(config.output or f"/tmp/{Path(config.corpus).stem}")


def subcommand(sub: ArgumentParser):
    for f in fields(ExtractionParameter):
        required = f.default is MISSING and f.default_factory is MISSING
        name = f.name if required else f"--{f.name}"
        default = None if required else (f.default if not f.default is MISSING else f.default_factory())
        if f.type is bool:
            sub.add_argument(name, action="store_true", default=False)
        else:
            sub.add_argument(name, type=f.type, default=default)
    sub.set_defaults(func=lambda args: main(args))