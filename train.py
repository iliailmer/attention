import argparse
from pprint import pprint

from src.config import Config
from src.training import train_c, train_w
from src.utils import set_seed


set_seed(42)

parser = argparse.ArgumentParser()

parser.add_argument("-w", action="store_true", help="run the word level model")
parser.add_argument("-c", action="store_true", help="run the character level model")
parser.add_argument("--config", "-cfg", type=str, help="config file path")
parser.add_argument("--embedding_size", "-es", type=int, help="embedding size", default=384)
parser.add_argument("--block_size", "-blk", type=int, help="block size (context length)", default=128)
parser.add_argument("--num_heads", "-nh", type=int, help="number of heads", default=6)
parser.add_argument("--num_blocks", "-nb", type=int, help="number of transformer blocks", default=6)
parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
parser.add_argument("--wd", type=float, help="weight decay", default=1e-2)
parser.add_argument("--batch_size", "-bs", type=int, help="batch size", default=16)
parser.add_argument("--n_epochs", "-ne", type=int, help="number of epochs", default=5000)
parser.add_argument("--device", "-d", type=str, help="device", default="mps")
parser.add_argument("--num_eval_steps", "-nes", type=int, help="number of evaluation steps", default=200)
parser.add_argument("--accumulate_grad", "-ag", type=int, help="accumulate gradient", default=4)
parser.add_argument("--use_flash", "-uf", action="store_true", help="use flash attention")

args = parser.parse_args()

config = Config(**args.__dict__)

if args.w:
    config.load("config_w.json")
elif args.c:
    config.load("config_c.json")
elif args.config:
    config.load(args.config)

pprint(config.__dict__)
if __name__ == "__main__":
    if args.w:
        train_w(config)
    if args.c:
        train_c(config)
    if not args.c and not args.w:
        print("Both arguments.c and arguments.w are False!")
