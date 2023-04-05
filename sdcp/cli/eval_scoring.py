from argparse import ArgumentParser, Namespace
from sdcp.grammar.sdcp import rule
from sdcp.tagging.data import CorpusWrapper
from sdcp.tagging.parsing_scorer import ScoringBuilder
from sdcp.tagging.embeddings import EmbeddingPresets, PretrainedBuilder

from datasets import DatasetDict
from collections import Counter, defaultdict
import flair
import torch

def combinatorial_stats(corpus, config):
    combinations: dict[tuple[int], int] = defaultdict(lambda: 0)
    denominator: dict[tuple[int], int] = defaultdict(lambda: 0)
    cnt_by_rhs = defaultdict(lambda: 0)

    for supertag in corpus.labels():
        sobj: rule = eval(supertag)
        if sobj.rhs:
            if config.separate and len(sobj.rhs) == 2:
                cnt_by_rhs[(sobj.rhs[0], None)] += 1
                cnt_by_rhs[(None, sobj.rhs[1])] += 1
            else:
                cnt_by_rhs[sobj.rhs] += 1

    for sentence in corpus:
        deriv = sentence.get_derivation()
        for node in deriv.subtrees():
            if not node.children:
                continue
            if config.separate and len(node) == 2:
                combinations[(node.label[0], node[0].label[0], None)] += 1
                combinations[(node.label[0], None, node[1].label[0])] += 1
                for node_ in deriv.subtrees():
                    if not len(node_) == 2: continue
                    if node[0].label[0] == node_[0].label[0]:
                        denominator[(node.label[0], node[0].label[0], None)] += 1
                    if node[1].label[0] == node_[1].label[0]:
                        denominator[(node.label[0], None, node[1].label[0])] += 1
            else:
                combinations[(node.label[0], *(c.label[0] for c in node))] += 1
                rhs = tuple(c.label[0] for c in node)
                for node_ in deriv.subtrees():
                    if rhs == tuple(c.label[0] for c in node_):
                        denominator[(node.label[0], *rhs)] += 1

    tot = 0
    coms = Counter()
    for _, v in combinations.items():
        tot += v
        coms[v] += 1
    print("total:", tot, "occurrences of", len(combinations), "combinations")
    print(coms[1], "combinations occur only once")


class ScoringModel(flair.nn.Model):
    @classmethod
    def instantiate(cls, config, corpus):
        builder = ScoringBuilder(config.type, corpus, *config.options)
        if not config.type in ("span",):
            embed_builder = []
        elif config.embedding in EmbeddingPresets:
            embed_builder = EmbeddingPresets[config.embedding]
        else:
            options = eval(f"dict({','.join(config.embedding_options)})")
            embed_builder = [PretrainedBuilder(config.embedding, **options)]
        embed_builder = [embedding.build_vocab(corpus) for embedding in embed_builder]
        return cls(len(corpus.labels()), builder, embed_builder)

    def __init__(self, ntags, scorebuilder, embedbuilder):
        super().__init__()
        self.scorebuilder = scorebuilder
        self.embedbuilder = embedbuilder
        self.embedding = flair.embeddings.StackedEmbeddings([
            builder.produce() for builder in embedbuilder
        ]) if embedbuilder else None
        emlen = self.embedding.embedding_length if not self.embedding is None else None
        self.scoring = self.scorebuilder.produce(emlen)
        self.ntags = ntags
        self.to(flair.device)

    def forward_loss(self, batch):
        loss = torch.tensor(0.0, device=flair.device)
        if not self.scoring.requires_training:
            return loss
        if not type(batch) is list:
            batch = [batch]
        for i, sentence in enumerate(batch):
            if not self.embedding is None:
                self.embedding.embed([sentence])
                embedding_name = self.embedding.get_names()
                input = torch.stack([word.get_embedding(embedding_name) for word in sentence]).to(flair.device)
            else:
                input = None
            deriv = sentence.get_derivation()
            for node in deriv.subtrees():
                leaves = list(n.label[1] for n in node.subtrees())
                loss += self.scoring.forward_loss(
                    node.label[0],
                    tuple(c.label[0] for c in node),
                    node.label[1],
                    leaves,
                    input)
        return loss
    
    def evaluate(self, dataset, **kwargs):
        hits = 0
        total = 0
        for i, sentence in enumerate(sent for sent in dataset):
            if not self.embedding is None:
                self.embedding.embed([sentence])
                embedding_name = self.embedding.get_names()
                input = torch.stack([word.get_embedding(embedding_name) for word in sentence]).to(flair.device)
            else:
                input = None
            deriv = sentence.get_derivation()
            for node in deriv.subtrees():
                if not node: continue
                if self.scoring.snd_order and any(c.label[0] >= self.ntags for c in node):
                    total += 1
                    continue
                leaves = list(n.label[1] for n in node.subtrees())
                distro = self.scoring.get_probab_distribution(
                    tuple(c.label[0] for c in node),
                    node.label[1],
                    leaves,
                    input)
                total += 1
                hits += distro.argmin(dim=0).item() == node.label[0]
        result_args = dict(
            main_score=hits/total,
            log_header="accuracy",
            log_line=hits/total,
            detailed_results='')
        return flair.training_utils.Result(**result_args, loss=None, classification_report=None)
    
    def _get_state_dict(self):
        return {
            "state_dict": self.state_dict(),
            "embedding_builder": self.embedbuilder,
            "scoring_builder": self.scorebuilder,
            "ntags": self.ntags
        }

    @classmethod
    def _init_model_with_state_dict(cls, state):
        model = cls(
            state["ntags"],
            state["scoring_builder"],
            state["embedding_builder"])
        model.load_state_dict(state["state_dict"])
        return model
    
    def label_type(self):
        return "supertag"

        
def main(config: Namespace):
    corpus = CorpusWrapper(config.corpus)
    # override test, s.t. the evaluation is not executed
    corpus.test = None
    if config.stats:
        combinatorial_stats(corpus.train, config)
    model = ScoringModel.instantiate(config, corpus.train)
    if model.scoring.requires_training:
        trainer = flair.trainers.ModelTrainer(model, corpus)
        train = trainer.fine_tune if any(em.fine_tune() for em in model.embedbuilder) else \
                    trainer.train
        train(
            "/tmp/sdcp-scoring",
            learning_rate=config.lr,
            mini_batch_size=32,
            mini_batch_chunk_size=None,
            max_epochs=config.epochs,
            weight_decay=config.weight_decay,
            optimizer=torch.optim.__dict__[config.optimizer],
            #scheduler=torch.optim.lr_scheduler.OneCycleLR
        )
    else:
        print("accuracy:", model.evaluate(corpus.dev).main_score)
    


def subcommand(sub: ArgumentParser):
    sub.add_argument("corpus", help="file containing gold tags", type=str)
    sub.add_argument("type", help="scoring type constructor", type=str, choices=["snd", "neu", "span"])
    sub.add_argument("--options", type=str, nargs="+", default=list())
    sub.add_argument("--embedding", type=str, default="Supervised")
    sub.add_argument("--embedding_options", type=str, nargs="+")
    sub.add_argument("--separate", action="store_true", default=False)
    sub.add_argument("--stats", action="store_true", default=False)
    sub.add_argument("--epochs", type=int, default=32)
    sub.add_argument("--lr", type=float, default=1e-5)
    sub.add_argument("--weight_decay", type=float, default=1e-2)
    sub.add_argument("--optimizer", type=str, default="AdamW")
    sub.set_defaults(func=lambda args: main(args))