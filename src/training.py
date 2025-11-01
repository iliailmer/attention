import torch
from tqdm.auto import tqdm

from src.config import Config
from src.flash_attn import GPTModel as GPTModelFA
from src.optimizer import Muon
from src.tokenization import Tokenizer, TokenizerByWord
from src.transformer import GPTModel
from src.utils import read_data, tinyshakespeare_batch, tinyshakespeare_batch_words


def train_w(config: Config):
    text = read_data()
    tokenizer = TokenizerByWord(from_text=text)
    print("Vocab Size:", tokenizer.vocab_size)

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
    optimizer = Muon(model.parameters(), lr=config.lr, weight_decay=config.wd)
    # torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.wd)

    data_train, data_val = text[: int(0.9 * len(text))], text[int(0.9 * len(text)) :]

    pbar = tqdm(range(config.n_epochs), leave=None)

    model.train()
    optimizer.zero_grad()
    try:
        for epoch in pbar:
            x, y = tinyshakespeare_batch_words(data_train, config.block_size, config.batch_size, tokenizer)
            x = x.to(config.device)
            y = y.to(config.device)
            out, loss = model(x, y)
            (loss / config.accumulate_grad).backward()
            if (epoch + 1) % config.accumulate_grad == 0:
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
                            x, y = tinyshakespeare_batch_words(data, config.block_size, config.batch_size, tokenizer)
                            x = x.to(config.device)
                            y = y.to(config.device)
                            out, loss = model(x, y)
                            losses[eval_step] = loss.item()
                        loss_dict[step] = torch.mean(losses).item()
                    pbar.set_description(
                        f"Epoch {epoch} - Training Loss: {loss_dict['train']:.4f} - Val Loss: {loss_dict['val']:.4f}"
                    )
                model.train()
        torch.save(model.state_dict(), "model_w.pt")
        config.save("config_w.json")
    except KeyboardInterrupt:
        torch.save(model.state_dict(), "model_w.pt")
        config.save("config_w.json")


def train_c(config: Config):
    text = read_data()
    tokenizer = Tokenizer(from_text=text)

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
    optimizer = Muon(model.parameters(), lr=config.lr, weight_decay=config.wd)

    # torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.wd)

    data_train, data_val = text[: int(0.9 * len(text))], text[int(0.9 * len(text)) :]

    pbar = tqdm(range(config.n_epochs), leave=None)

    model.train()
    optimizer.zero_grad()
    try:
        for epoch in pbar:
            x, y = tinyshakespeare_batch(data_train, config.block_size, config.batch_size, tokenizer)
            x = x.to(config.device)
            y = y.to(config.device)
            out, loss = model(x, y)
            (loss / config.accumulate_grad).backward()
            if (epoch + 1) % config.accumulate_grad == 0:
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
        torch.save(model.state_dict(), "model_c.pt")
        config.save("config_c.json")
    except KeyboardInterrupt:
        model.eval()
        torch.save(model.state_dict(), "model_c.pt")
        config.save("config_c.json")
