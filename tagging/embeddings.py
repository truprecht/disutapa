import flair
from flair.embeddings import TokenEmbeddings, FlairEmbeddings, StackedEmbeddings, \
        WordEmbeddings, OneHotEmbeddings, CharacterEmbeddings, TransformerWordEmbeddings

from abc import ABC, abstractmethod
from enum import Enum

from.data import DatasetWrapper


class TokenEmbeddingBuilder(ABC):
    def build_vocab(self, corpus: DatasetWrapper):
        return self

    def fine_tune(self):
        return False

    @abstractmethod
    def produce(self) -> TokenEmbeddings:
        raise NotImplementedError()


class PretrainedBuilder(TokenEmbeddingBuilder):
    def __init__(self, name: str, language: str = "en", tune: bool = False):
        if any((spec in name) for spec in ("bert", "gpt", "xlnet")):
            self.embedding_t = TransformerWordEmbeddings
            self.model_str = name
        elif name == "flair":
            self.embedding_t = FlairEmbeddings
            self.model_str = language
        elif name == "fasttext":
            self.embedding_t = WordEmbeddings
            self.model_str = language
        else:
            raise NotImplementedError(f"Cound not recognize embedding {name}")
        self.tune = tune

    def fine_tune(self):
        return self.tune

    def produce(self) -> TokenEmbeddings:
        if self.embedding_t is TransformerWordEmbeddings:
            return TransformerWordEmbeddings(model=self.model_str, fine_tune=self.tune, layers="-1,-2,-3,-4", layer_mean=False)
        if self.embedding_t is FlairEmbeddings:
            return StackedEmbeddings([
                FlairEmbeddings(f"{self.model_str}-forward", fine_tune=self.tune),
                FlairEmbeddings(f"{self.model_str}-backward", fine_tune=self.tune)])
        if self.embedding_t is WordEmbeddings:
            return WordEmbeddings(self.model_str)


class CharacterEmbeddingBuilder(TokenEmbeddingBuilder):
    def __init__(self, embedding_dim: int = 32, lstm_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.hidden_size = lstm_dim

    def produce(self) -> TokenEmbeddings:
        return CharacterEmbeddings(
            char_embedding_dim=self.embedding_dim,
            hidden_size_char=self.hidden_size)


class OneHotEmbeddingBuilder(TokenEmbeddingBuilder):
    def __init__(self, field: str, minfreq: int = 1, embedding_dim: int = 64):
        self.field = field
        self.min_freq = minfreq
        self.length = embedding_dim
        self.vocab = None

    def build_vocab(self, corpus: DatasetWrapper):
        self.vocab = corpus.build_dictionary(self.field, add_unk=True, minfreq=self.min_freq)
        return self

    def produce(self) -> TokenEmbeddings:
        return OneHotEmbeddings(self.vocab, self.field, self.length)


class EmbeddingPresets(Enum):
    Supervised = [OneHotEmbeddingBuilder("text")]

    Bert = [PretrainedBuilder("bert-base-cased", tune=True)]
    GBert = [PretrainedBuilder("bert-base-german-cased", tune=True)]
    BertLarge = [PretrainedBuilder("bert-large-cased", tune=True)]
    GBertLarge = [PretrainedBuilder("deepset/gbert-large", tune=True)]

    Flair = [PretrainedBuilder("fasttext", "en"), PretrainedBuilder("flair", "en")]
    GFlair = [PretrainedBuilder("fasttext", "de"), PretrainedBuilder("flair", "de")]
