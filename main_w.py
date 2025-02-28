# TODO: Merge different generation codes and training codes
# TODO: add flash attention full GPT model
import random

import numpy as np
import torch
from tqdm.auto import tqdm

from src.config import Config, save_config
from src.tokenization import TokenizerByWord as Tokenizer
from src.transformer import GPTModel
from src.utils import read_data
from src.utils import tinyshakespeare_batch_words as tinyshakespeare_batch


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


set_seed(42)
config = Config()

if __name__ == "__main__":
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.wd)

    data_train, data_val = text[: int(0.9 * len(text))], text[int(0.9 * len(text)) :]

    pbar = tqdm(range(config.n_epochs), leave=None)

    model.train()
    try:
        for epoch in pbar:
            x, y = tinyshakespeare_batch(data_train, config.block_size, config.batch_size, tokenizer)
            x = x.to(config.device)
            y = y.to(config.device)
            out, loss = model(x, y)
            loss.backward()
            if epoch % config.accumulate_grad == 0:
                optimizer.step()
                optimizer.zero_grad()

            if epoch % config.eval_every == 0:
                model.eval()
                with torch.no_grad():
                    loss_dict = dict(train=0.0, val=0.0)
                    for step, data in [("train", data_train), ("val", data_val)]:
                        pbar_val = tqdm(range(config.num_eval_steps), leave=None)
                        losses = torch.zeros(config.num_eval_steps)
                        for eval_step in pbar_val:
                            x, y = tinyshakespeare_batch(data, config.block_size, config.batch_size, tokenizer)
                            x = x.to(config.device)
                            y = y.to(config.device)
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
        print(tokenizer.decode(model.generate(context, max_new_tokens=500, block_size=config.block_size)[0].tolist()))
    torch.save(model.state_dict(), "model_w.pt")
    save_config(config, "config_w.json")
