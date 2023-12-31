from flair import embeddings # type: ignore

from abc import ABC, abstractmethod
from enum import Enum

from.data import DatasetWrapper


class TokenEmbeddingBuilder(ABC):
    def build_vocab(self, _corpus: DatasetWrapper):
        return self

    def fine_tune(self):
        return False

    @abstractmethod
    def produce(self) -> embeddings.TokenEmbeddings:
        raise NotImplementedError()


class PretrainedBuilder(TokenEmbeddingBuilder):
    def __init__(self, name: str, **kwargs):
        if any((spec in name.lower()) for spec in ("bert", "gpt", "xlnet")):
            self.embedding_t = embeddings.TransformerWordEmbeddings
            self.model_str = name
            # by default, flair activates fine tuning
            kwargs.setdefault("fine_tune", True)
        elif name == "flair":
            self.embedding_t = embeddings.FlairEmbeddings
            self.model_str = kwargs.pop("language") if "language" in kwargs else "en"
        elif name == "fasttext":
            self.embedding_t = embeddings.WordEmbeddings
            self.model_str = kwargs.pop("language") if "language" in kwargs else "en"
        else:
            raise NotImplementedError(f"Cound not recognize embedding {name}")
        self.options = kwargs

    def fine_tune(self):
        return self.options.get("fine_tune", False)

    def produce(self) -> embeddings.TokenEmbeddings:
        if self.embedding_t is embeddings.TransformerWordEmbeddings:
            return embeddings.TransformerWordEmbeddings(model=self.model_str, **self.options)
        if self.embedding_t is embeddings.FlairEmbeddings:
            return embeddings.StackedEmbeddings([
                embeddings.FlairEmbeddings(f"{self.model_str}-forward", **self.options),
                embeddings.FlairEmbeddings(f"{self.model_str}-backward", **self.options)])
        if self.embedding_t is embeddings.WordEmbeddings:
            return embeddings.WordEmbeddings(self.model_str, **self.options)


class CharacterEmbeddingBuilder(TokenEmbeddingBuilder):
    def __init__(self, embedding_dim: int = 32, lstm_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.hidden_size = lstm_dim

    def produce(self) -> embeddings.TokenEmbeddings:
        return embeddings.CharacterEmbeddings(
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

    def produce(self) -> embeddings.TokenEmbeddings:
        return embeddings.OneHotEmbeddings(self.vocab, self.field, self.length)

EmbeddingPresets = dict(
    Supervised = [OneHotEmbeddingBuilder("text", minfreq=3, embedding_dim=256), CharacterEmbeddingBuilder(lstm_dim=128, embedding_dim=64)],
    
    BERT = [PretrainedBuilder("bert-base-cased", fine_tune=True)],
    GBert = [PretrainedBuilder("bert-base-german-cased", fine_tune=True)],
    BertLarge = [PretrainedBuilder("bert-large-cased", fine_tune=True)],
    GBertLarge = [PretrainedBuilder("deepset/gbert-large", fine_tune=True)],
    
    Flair = [PretrainedBuilder("fasttext", language="en"), PretrainedBuilder("flair", language="en")],
    GFlair = [PretrainedBuilder("fasttext", language="de"), PretrainedBuilder("flair", language="de")],
)
