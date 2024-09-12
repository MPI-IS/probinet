import argparse
import logging

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("-dictionary", default="spanish", help="Dictionary file")

subparsers = parser.add_subparsers(dest="algorithm", help="Choose the algorithm to run")

parser_a = subparsers.add_parser("numbers", help="Algorithm A")
parser_a.add_argument("N", help="Actual number")

parser_b = subparsers.add_parser("letters", help="Algorithm B")
parser_b.add_argument("L", help="Actual letter")

args_a = parser.parse_args(["numbers", "10"])
print(args_a)
args_b = parser.parse_args(["letters", "A"])
print(args_b)
args = parser.parse_args(["-dictionary", "english", "numbers", "10"])
print(args)

# Print all the args used in alphabetical order
logging.basicConfig(level=logging.INFO)
logging.info("Arguments used:")
for arg in sorted(vars(args)):
    logging.info("%s: %s", arg, getattr(args, arg))
