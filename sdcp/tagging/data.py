from datasets import Dataset, DatasetDict
from flair.data import Sentence, Dictionary, Token
from collections import Counter
from ..grammar.derivation import Derivation

class SentenceWrapper(Sentence):
    def __init__(self, dataobj: dict):
        super().__init__(dataobj["sentence"], use_tokenizer=False)
        self.__gold_label_ids = dataobj
        self.__predicted_label_ids = {}

    def get_derivation(self):
        deriv = self.get_raw_labels("derivation")
        st = self.get_raw_labels("supertag")
        return Derivation.from_str(deriv, st)

    def get_raw_labels(self, field: str):
        return self.__gold_label_ids[field]

    def store_raw_prediction(self, field: str, labels):
        self.__predicted_label_ids[field] = labels

    def get_raw_prediction(self, field: str):
        return self.__predicted_label_ids[field]


class DatasetWrapper:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __getitem__(self, idx: int = 0) -> SentenceWrapper:
        return SentenceWrapper(self.dataset[idx])

    def __len__(self):
        return len(self.dataset)

    def labels(self, field: str = "supertag"):
        if field in ("supertag", "pos"):
            return self.dataset.features[field].feature.names
        elif field == "text":
            return (tok for row in self.dataset for tok in row["sentence"])
        raise NotImplementedError()

    def build_dictionary(self, field: str, add_unk: bool = True, minfreq: int = 1):
        count = Counter()
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
