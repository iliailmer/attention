import torch
import argparse

from src.config import Config
from src.tokenization import Tokenizer, TokenizerByWord
from src.transformer import GPTModel
from src.utils import read_data, set_seed

set_seed(42)
config = Config()
parser = argparse.ArgumentParser()
parser.add_argument("-w", action="store_true", help="run the word level model")
parser.add_argument("-c", action="store_true", help="run the character level model")
args = parser.parse_args()
text = read_data()
tokenizer = None
if args.w:
    config.load("config_w.json")
    tokenizer = TokenizerByWord(from_text=text)
elif args.c:
    config.load("config_c.json")
    tokenizer = Tokenizer(from_text=text)

if tokenizer is not None:
    model = GPTModel(
        vocab_size=tokenizer.vocab_size,
        embedding_size=config.embedding_size,
        head_size=config.embedding_size // config.num_heads,
        block_size=config.block_size,
        num_heads=config.num_heads,
        num_blocks=config.num_blocks,
    )
    model = model.to(config.device)
    model.load_state_dict(torch.load("model_c.pt"))
    context = tokenizer.encode("T'was a ni").unsqueeze(0).long().to(config.device)
    print(tokenizer.decode(model.generate(context, max_new_tokens=500, block_size=config.block_size)[0].tolist()))
