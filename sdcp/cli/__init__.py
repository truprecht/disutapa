from . import eval_scoring, add_scoring, extract, test, train, eval, add_reranker
from argparse import ArgumentParser

def main():
    args = ArgumentParser("hybrid-supertags", description="Supertags for constituent parsing using hybrid grammars.")
    args.set_defaults(func=lambda x: args.print_help())
    subcommands = args.add_subparsers()
    extract.subcommand(
        subcommands.add_parser(name="extract", description="Extract supertags from corpora"))
    test.subcommand(
        subcommands.add_parser(name="test", description="Test extracted gold supertags and report parsing score"))
    train.subcommand(
        subcommands.add_parser(name="train", description="Train discriminative model for supertagging"))
    eval_scoring.subcommand(
        subcommands.add_parser(name="scoring", description="Show statistics about second-order rule scoring"))
    add_scoring.subcommand(
        subcommands.add_parser(name="add-scoring", description="Add a scoring module to the parser"))
    eval.subcommand(
        subcommands.add_parser(name="eval", description="Evaluate a trained classifier"))
    add_reranker.subcommand(
        subcommands.add_parser(name="reranking", description="Train a reranking model and add it to a parser"))
    parsed_args = args.parse_args()
    parsed_args.func(parsed_args)