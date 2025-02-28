import json

import torch

from src.config import Config
from src.tokenization import Tokenizer
from src.transformer import GPTModel
from src.utils import read_data, set_seed

set_seed(42)
config = Config()
config.load("config_c.json")
text = read_data()
tokenizer = Tokenizer(from_text=text)


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
