from . import extract, test, train, eval, train_dop, grid, prediction_statistic, parsing_statistic, dop_statistic
from argparse import ArgumentParser

def main():
    args = ArgumentParser("disutapa", description="Discontinuous Supertagging and Parsing")
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
    grid.subcommand(
        subcommands.add_parser(name="grid", description="Execute commands for each combination of variables specified in a configuration file. (Grid Search)"))
    prediction_statistic.subcommand(
        subcommands.add_parser(name="stats", description="Print statistics for the supertag prediction using a trained classifier"))
    parsing_statistic.subcommand(
        subcommands.add_parser(name="p_stats", description="Print statistics for parsing using a trained classifier"))
    dop_statistic.subcommand(
        subcommands.add_parser(name="d_stats", description="Print statistics for parsing using a trained classifier"))
    parsed_args = args.parse_args()
    parsed_args.func(parsed_args)