from datasets import Dataset, DatasetDict  # type: ignore
from flair.data import Sentence, Dictionary, Token  # type: ignore
from collections import Counter
from typing import cast

from ..grammar.derivation import Derivation

class SentenceWrapper(Sentence):
    def __init__(self, dataobj: dict[str, list[int]]):
        super().__init__(dataobj["sentence"], use_tokenizer=False)
        self.__gold_label_ids = dataobj
        self.__predicted_label_ids: dict[str, list[int]] = {}
        self.__cache: dict[str, object] = {}

    def get_derivation(self) -> Derivation:
        if not "derivation" in self.__cache:
            deriv = self.get_raw_labels("derivation")
            st = self.get_raw_labels("supertag")
            self.__cache["derivation"] = Derivation.from_str(deriv, st)
        return cast(Derivation, self.__cache["derivation"])
    
    def cache(self, key, value=None):
        if not value is None:
            self.__cache[key] = value
        return self.__cache.get(key)

    def get_raw_labels(self, field: str):
        return self.__gold_label_ids[field]

    def store_raw_prediction(self, field: str, labels):
        self.__predicted_label_ids[field] = labels

    def get_raw_prediction(self, field: str):
        return self.__predicted_label_ids[field]


class DatasetWrapper:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.sentences = [SentenceWrapper(s) for s in self.dataset]

    def __getitem__(self, idx: int = 0) -> SentenceWrapper:
        return self.sentences[idx]

    def __len__(self):
        return len(self.dataset)

    def labels(self, field: str = "supertag"):
        if field in ("supertag", "pos"):
            return self.dataset.features[field].feature.names
        elif field == "text":
            return (tok for row in self.dataset for tok in row["sentence"])
        raise NotImplementedError()

    def build_dictionary(self, field: str, add_unk: bool = True, minfreq: int = 1):
        count: Counter = Counter()
        vocab = Dictionary(add_unk=add_unk)
        for token in self.labels(field):
            count[token] += 1
            if count[token] == minfreq:
                vocab.add_item(token)
        return vocab


class CorpusWrapper:
    def __init__(self, location: str):
        corpus = DatasetDict.load_from_disk(location)
        self.train = DatasetWrapper(corpus["train"])
        self.dev = DatasetWrapper(corpus["dev"])
        self.test = DatasetWrapper(corpus["test"])
