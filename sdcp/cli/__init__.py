from . import extract, test, train, eval, train_dop
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
    eval.subcommand(
        subcommands.add_parser(name="eval", description="Evaluate a trained classifier"))
    train_dop.subcommand(
        subcommands.add_parser(name="dop", description="Train a reranking dop model"))
    parsed_args = args.parse_args()
    parsed_args.func(parsed_args)