import argparse

from src.config import Config
from src.utils import set_seed

set_seed(42)
config = Config()

parser = argparse.ArgumentParser()

parser.add_argument("-w", action="store_true", help="run the word level model")
parser.add_argument("-c", action="store_true", help="run the character level model")

args = parser.parse_args()

if __name__ == "__main__":
    if args.w:
        pass
    if args.c:
        pass
