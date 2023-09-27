from argparse import ArgumentParser
from itertools import product
from subprocess import run
import yaml


def main(args):
    yamlfile = yaml.safe_load(args.gridfile)
    variables_and_values = yamlfile.get("variables", {})
    subcommands = yamlfile.get("commands", [])    
    variables = variables_and_values.keys()
    grid = product(*variables_and_values.values())
    already_executed = set()
    for point in grid:
        for command in subcommands:
            for (var,value) in zip(variables,point):
                command = command.replace(f"[{var}]", value)
            if not command in already_executed:
                already_executed.add(command)
                print("executing:", command)
                run(command.split())


def subcommand(sub: ArgumentParser):
    sub.add_argument("gridfile", type=open)
    sub.set_defaults(func=lambda args: main(args))