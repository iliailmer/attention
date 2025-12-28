# TODO: Add learning rate scheduler
import torch
from loguru import logger
from tqdm.auto import tqdm

from src.config import Config
from src.flash_attn import GPTModel as GPTModelFA
from src.tokenization import Tokenizer, TokenizerByWord
from src.transformer import GPTModel
from src.utils import read_data, tinyshakespeare_batch, tinyshakespeare_batch_words


def train_w(config: Config):
    text = read_data()
    tokenizer = TokenizerByWord(from_text=text)
    logger.info(f"Vocab Size: {tokenizer.vocab_size}")

    tokens = tokenizer.encode(text)
    split_idx = int(0.9 * len(tokens))
    tokens_train, tokens_val = tokens[:split_idx], tokens[split_idx:]
    logger.info(f"Total tokens: {len(tokens)}, Train: {len(tokens_train)}, Val: {len(tokens_val)}")

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
    # model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.wd)

    pbar = tqdm(range(config.n_epochs), leave=None)

    model.train()
    optimizer.zero_grad()
    try:
        for epoch in pbar:
            x, y = tinyshakespeare_batch_words(tokens_train, config.block_size, config.batch_size)
            x = x.to(config.device)
            y = y.to(config.device)
            out, loss = model(x, y)
            (loss / config.accumulate_grad).backward()
            if (epoch + 1) % config.accumulate_grad == 0:
                optimizer.step()
                optimizer.zero_grad()

            if epoch > 0 and epoch % config.eval_every == 0:
                model.eval()
                with torch.no_grad():
                    val_losses = torch.zeros(config.num_eval_steps)
                    for eval_step in range(config.num_eval_steps):
                        x, y = tinyshakespeare_batch_words(tokens_val, config.block_size, config.batch_size)
                        x = x.to(config.device)
                        y = y.to(config.device)
                        _, loss = model(x, y)
                        val_losses[eval_step] = loss.item()
                    val_loss = torch.mean(val_losses).item()
                    pbar.set_description(f"Epoch {epoch} - Train Loss: {loss.item():.4f} - Val Loss: {val_loss:.4f}")
                model.train()
        torch.save(model.state_dict(), "model_w.pt")
        config.save("config_w.json")
    except KeyboardInterrupt:
        torch.save(model.state_dict(), "model_w.pt")
        config.save("config_w.json")


def train_c(config: Config):
    text = read_data()
    tokenizer = Tokenizer(from_text=text)
    logger.info(f"Vocab Size: {tokenizer.vocab_size}")

    tokens = tokenizer.encode(text)
    split_idx = int(0.9 * len(tokens))
    tokens_train, tokens_val = tokens[:split_idx], tokens[split_idx:]
    logger.info(f"Total tokens: {len(tokens)}, Train: {len(tokens_train)}, Val: {len(tokens_val)}")

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
    # model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.wd)

    pbar = tqdm(range(config.n_epochs), leave=None)

    model.train()
    optimizer.zero_grad()
    try:
        for epoch in pbar:
            x, y = tinyshakespeare_batch(tokens_train, config.block_size, config.batch_size)
            x = x.to(config.device)
            y = y.to(config.device)
            out, loss = model(x, y)
            (loss / config.accumulate_grad).backward()
            if (epoch + 1) % config.accumulate_grad == 0:
                optimizer.step()
                optimizer.zero_grad()

            if epoch > 0 and epoch % config.eval_every == 0:
                model.eval()
                with torch.no_grad():
                    val_losses = torch.zeros(config.num_eval_steps)
                    for eval_step in range(config.num_eval_steps):
                        x, y = tinyshakespeare_batch(tokens_val, config.block_size, config.batch_size)
                        x = x.to(config.device)
                        y = y.to(config.device)
                        _, loss = model(x, y)
                        val_losses[eval_step] = loss.item()
                    val_loss = torch.mean(val_losses).item()
                    pbar.set_description(f"Epoch {epoch} - Train Loss: {loss.item():.4f} - Val Loss: {val_loss:.4f}")
                model.train()
        torch.save(model.state_dict(), "model_c.pt")
        config.save("config_c.json")
    except KeyboardInterrupt:
        model.eval()
        torch.save(model.state_dict(), "model_c.pt")
        config.save("config_c.json")
