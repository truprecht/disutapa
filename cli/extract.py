from dataclasses import dataclass, field
from argparse import ArgumentParser
from typing import Iterable
from datasets import Dataset, DatasetDict, Features, Value, ClassLabel, Sequence
from tqdm import tqdm
from sdcp.corpus import corpus_extractor, Split

@dataclass
class ExtractionParameter:
    corpus: str
    output: str
    split: dict = field(default_factory=lambda: { "train": range(1), "dev": range(1), "test": range(1) })
    hmarkov: int = 0
    vmarkov: int = 1
        

def main(config: ExtractionParameter):
    ex = corpus_extractor(config.corpus, horzmarkov=config.hmarkov, vertmarkov=config.vmarkov)
    for r in Split(**config.split).nonoverlapping():
        ex.read(r)
    rules = list(ex.rules)
    datasets = {}
    for split, portion in config.split.items():
        dataset = { "sentence": [], "supertag": [], "pos": [], "tree": [] }
        total = portion.stop-portion.start
        desc = f"extracting {split} portion"
        for idx in tqdm(range(portion.start, portion.stop), total=total, desc=desc):
            gtree, sentence, gpos, grules = ex[idx]
            dataset["sentence"].append(list(sentence))
            dataset["supertag"].append([repr(rules[r]) for r in grules])
            dataset["pos"].append(list(gpos))
            dataset["tree"].append(str(gtree))
        datasets[split] = dataset
    tagsets = {
        "train": {
            "supertag": set(tag for tagseq in datasets["train"]["supertag"] for tag in tagseq),
            "pos": set(tag for tagseq in datasets["train"]["pos"] for tag in tagseq)
        }
    }
    for split in config.split:
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
            "tree": Value("string")
        }))
        for split, dataset in datasets.items()
    }
    DatasetDict(**datasets).save_to_disk(config.output)

    
def subcommand(sub: ArgumentParser):
    sub.add_argument("config", type=open)
    sub.set_defaults(func=lambda args: main(eval(args.config.read())))