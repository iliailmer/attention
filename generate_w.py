import json

import torch

from src.tokenization import TokenizerByWord as Tokenizer
from src.transformer import GPTModel
from src.utils import read_data


with open("config.json") as f:
    Config = json.load(f)
text = read_data()
tokenizer = Tokenizer(from_text=text)


model = GPTModel(
    vocab_size=tokenizer.vocab_size,
    embedding_size=Config.get("embedding_size"),
    head_size=Config.get("embedding_size") // Config.get("num_heads"),
    block_size=Config.get("block_size"),
    num_heads=Config.get("num_heads"),
    num_blocks=Config.get("num_blocks"),
)
model = model.to(Config.get("device"))
model.load_state_dict(torch.load("model_w.pt"))
context = tokenizer.encode("It was a night").unsqueeze(0).long().to(Config.get("device"))
print(tokenizer.decode(model.generate(context, max_new_tokens=100, block_size=Config.get("block_size"))[0].tolist()))
