import json
import random

import numpy as np
import torch
from tqdm.auto import tqdm

from src.tokenization import TokenizerByWord as Tokenizer
from src.transformer import GPTModel
from src.utils import read_data, tinyshakespeare_batch_words as tinyshakespeare_batch


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


set_seed(42)


class Config:
    embedding_size = 384
    block_size = 128
    num_heads = 6
    num_blocks = 6
    lr = 1e-5
    wd = 1e-2
    batch_size = 16
    n_epochs = 5000
    device = "mps"
    eval_every = n_epochs // 10
    num_eval_steps = 200
    accumulate_grad = 4


text = read_data()
tokenizer = Tokenizer(from_text=text)


model = GPTModel(
    vocab_size=tokenizer.vocab_size,
    embedding_size=Config.embedding_size,
    head_size=Config.embedding_size // Config.num_heads,
    block_size=Config.block_size,
    num_heads=Config.num_heads,
    num_blocks=Config.num_blocks,
)
model = model.to(Config.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=Config.wd)

data_train, data_val = text[: int(0.9 * len(text))], text[int(0.9 * len(text)) :]

pbar = tqdm(range(Config.n_epochs), leave=None)

model.train()
try:
    for epoch in pbar:
        x, y = tinyshakespeare_batch(data_train, Config.block_size, Config.batch_size, tokenizer)
        x = x.to(Config.device)
        y = y.to(Config.device)
        out, loss = model(x, y)
        loss.backward()
        if epoch % Config.accumulate_grad == 0:
            optimizer.step()
            optimizer.zero_grad()

        if epoch % Config.eval_every == 0:
            model.eval()
            with torch.no_grad():
                loss_dict = dict(train=0.0, val=0.0)
                for step, data in [("train", data_train), ("val", data_val)]:
                    pbar_val = tqdm(range(Config.num_eval_steps), leave=None)
                    losses = torch.zeros(Config.num_eval_steps)
                    for eval_step in pbar_val:
                        x, y = tinyshakespeare_batch(data, Config.block_size, Config.batch_size, tokenizer)
                        x = x.to(Config.device)
                        y = y.to(Config.device)
                        out, loss = model(x, y)
                        losses[eval_step] = loss.item()
                    loss_dict[step] = torch.mean(losses).item()
                pbar.set_description(
                    f"Epoch {epoch} - Training Loss: {loss_dict['train']:.4f} - Val Loss: {loss_dict['val']:.4f}"
                )
            model.train()
except KeyboardInterrupt:
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device="mps")
    print(tokenizer.decode(model.generate(context, max_new_tokens=500, block_size=Config.block_size)[0].tolist()))
torch.save(model.state_dict(), "model.pt")
with open("config.json", "w") as f:
    f.write(json.dumps(Config, default=lambda x: x.__dict__))
