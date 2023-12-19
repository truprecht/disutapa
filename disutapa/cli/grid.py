from argparse import ArgumentParser
from itertools import product
from subprocess import run
import yaml


def main(args):
    yamlfile = yaml.safe_load(args.gridfile)
    constants = {}
    inputs = dict(a.split("=", 1) for a in args.constants)
    for constant in yamlfile.get("constants", []):
        if not constant in inputs:
            print(f"missing value for {constant} add it as parameter of the form {constant}=VALUE")
            exit(1)
        constants[constant] = inputs[constant]
    variables_and_values = yamlfile.get("variables", {})
    subcommands = yamlfile.get("commands", [])    
    variables = variables_and_values.keys()
    grid = product(*variables_and_values.values())
    already_executed = set()
    for point in grid:
        for command in subcommands:
            for (var,value) in zip(variables,point):
                command = command.replace(f"[{var}]", value)
            for cname, cvalue in constants.items():
                command = command.replace(f"[{cname}]", cvalue)
            if not command in already_executed:
                already_executed.add(command)
                print("executing:", command)
                run(command.split())


def subcommand(sub: ArgumentParser):
    sub.add_argument("gridfile", type=open)
    sub.add_argument("constants", type=str, nargs="*")
    sub.set_defaults(func=lambda args: main(args))