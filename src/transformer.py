import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MultiHeadAttentionEfficient(nn.Module):
    def __init__(self, embedding_size: int, num_heads: int, block_size: int, is_decoder=False, dropout=0.2) -> None:
        super().__init__()
        assert embedding_size % num_heads == 0, "embedding_size must be divisible by num_heads"

        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_size = embedding_size // num_heads
        self.is_decoder = is_decoder

        self.qkv = nn.Linear(embedding_size, 3 * embedding_size, bias=False)
        self.proj = nn.Linear(embedding_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

        if is_decoder:
            self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.split(self.embedding_size, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float("-inf"))
        elif self.is_decoder:
            scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # pyright: ignore

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.proj(out)
        out = self.dropout(out)

        return out


class EncoderBlock(nn.Module):
    def __init__(self, embedding_size: int, block_size: int, num_heads: int, ffn_hidden_size: int = 512):
        super().__init__()
        self.attn = MultiHeadAttentionEfficient(embedding_size, num_heads, block_size, is_decoder=False)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_size, ffn_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(ffn_hidden_size, embedding_size),
            nn.Dropout(0.2),
        )
        self.layer_norm_1 = nn.LayerNorm(embedding_size)
        self.layer_norm_2 = nn.LayerNorm(embedding_size)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.layer_norm_1(x), mask)
        x = x + self.ffn(self.layer_norm_2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, embedding_size: int, block_size: int, num_heads: int, ffn_hidden_size: int = 512):
        super().__init__()
        self.self_attn = MultiHeadAttentionEfficient(embedding_size, num_heads, block_size, is_decoder=True)
        self.cross_attn = MultiHeadAttentionEfficient(embedding_size, num_heads, block_size, is_decoder=False)
        self.layer_norm_1 = nn.LayerNorm(embedding_size)
        self.layer_norm_2 = nn.LayerNorm(embedding_size)
        self.layer_norm_3 = nn.LayerNorm(embedding_size)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_size, ffn_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(ffn_hidden_size, embedding_size),
            nn.Dropout(0.2),
        )

    def forward(self, x: Tensor, mask: Tensor | None = None, enc_x: Tensor | None = None) -> Tensor:
        x = x + self.self_attn(self.layer_norm_1(x), mask)
        if enc_x is not None:
            x_cat = torch.cat([x, enc_x], dim=1)
            x = x + self.cross_attn(self.layer_norm_2(x_cat))
        x = x + self.ffn(self.layer_norm_3(x))
        return x


class GPTModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        block_size: int,
        num_heads: int,
        num_blocks: int,
        ffn_hidden_size: int | None = None,
    ) -> None:
        super().__init__()
        if ffn_hidden_size is None:
            ffn_hidden_size = 4 * embedding_size
        self.t_embedding = nn.Embedding(vocab_size, embedding_size)
        self.p_embedding = nn.Embedding(block_size, embedding_size)
        self.decoder_blocks = nn.Sequential(
            *[DecoderBlock(embedding_size, block_size, num_heads, ffn_hidden_size) for _ in range(num_blocks)]
        )
        self.layer_norm = nn.LayerNorm(embedding_size)
        self.final = nn.Linear(embedding_size, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        _, T = x.shape
        x = self.t_embedding(x) + self.p_embedding(torch.arange(T, device=x.device))
        x = self.decoder_blocks(x)
        x = self.layer_norm(x)
        logits = self.final(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens=256, block_size=128):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # last block
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # NOTE: this is the last token in the sequence
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1).to("mps")
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
