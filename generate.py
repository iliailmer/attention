import argparse

import torch

from src.config import Config
from src.flash_attn import GPTModel as GPTModelFA
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
model_name = ""
if args.w:
    config.load("config_w.json")
    model_name = "model_w.pt"
    tokenizer = TokenizerByWord(from_text=text)
elif args.c:
    config.load("config_c.json")
    model_name = "model_c.pt"
    tokenizer = Tokenizer(from_text=text)

if tokenizer is not None:
    ffn_hidden = config.ffn_multiplier * config.embedding_size
    if config.use_flash:
        model = GPTModelFA(
            vocab_size=tokenizer.vocab_size,
            embedding_size=config.embedding_size,
            block_size=config.block_size,
            num_heads=config.num_heads,
            num_blocks=config.num_blocks,
            ffn_hidden_size=ffn_hidden,
        )
    else:
        model = GPTModel(
            vocab_size=tokenizer.vocab_size,
            embedding_size=config.embedding_size,
            block_size=config.block_size,
            num_heads=config.num_heads,
            num_blocks=config.num_blocks,
            ffn_hidden_size=ffn_hidden,
        )
    model = model.to(config.device)
    model.load_state_dict(torch.load(model_name, weights_only=True, map_location=config.device))
    model.eval()
    model = torch.compile(model)
    context = tokenizer.encode("as a night").unsqueeze(0).to(config.device)

    with torch.no_grad():
        output = model.generate(context, max_new_tokens=500, block_size=config.block_size)

    print(tokenizer.decode(output[0].tolist()))
