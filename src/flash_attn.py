import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm.auto import tqdm


def flash_attention(q: Tensor, k: Tensor, v: Tensor, mask=None, is_decoder=False, tril=None, block_size=128):
    """
    Flash Attention: Computes attention using memory-efficient tiling.
    """
    batch_size, seq_len, head_dim = q.shape
    output = torch.zeros_like(q)  # Stores final results

    for i in range(0, seq_len, block_size):
        q_block = q[:, i : i + block_size]
        k_block = k[:, i : i + block_size]
        v_block = v[:, i : i + block_size]

        # Compute QK^T only for this block
        scores = torch.matmul(q_block, k_block.transpose(-2, -1)) / (head_dim**0.5)
        if mask is not None:
            scores = scores.masked_fill(mask[:, i : i + block_size, i : i + block_size] == 0, float("-inf"))
        elif is_decoder and tril is not None:
            if block_size > scores.shape[-1]:
                scores = scores.masked_fill(
                    tril[i : i + scores.shape[-1], i : i + scores.shape[-1]] == 0, float("-inf")
                )
            else:
                scores = scores.masked_fill(tril[i : i + block_size, i : i + block_size] == 0, float("-inf"))
        # Compute online softmax (normalize per tile)
        scores = scores - scores.max(dim=-1, keepdim=True).values  # Stabilize softmax
        scores = scores.exp()
        scores_sum = scores.sum(dim=-1, keepdim=True)
        scores = scores / (scores_sum + 1e-6)  # Avoid division by zero

        # Compute weighted sum
        output[:, i : i + block_size] = torch.matmul(scores, v_block)

    return output


class MultiHeadFlashAttention(nn.Module):
    def __init__(self, embedding_size: int, num_heads: int, context_size: int, is_decoder=False, flash_block_size=128, dropout=0.2):
        super().__init__()
        assert embedding_size % num_heads == 0

        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_size = embedding_size // num_heads
        self.is_decoder = is_decoder
        self.flash_block_size = flash_block_size

        self.qkv = nn.Linear(embedding_size, 3 * embedding_size, bias=False)
        self.proj = nn.Linear(embedding_size, embedding_size)
        self.dropout = nn.Dropout(dropout)

        if is_decoder:
            self.register_buffer("tril", torch.tril(torch.ones(context_size, context_size)))

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.split(self.embedding_size, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)

        output = torch.zeros_like(q)

        for i in range(0, T, self.flash_block_size):
            end = min(i + self.flash_block_size, T)
            q_block = q[:, :, i:end, :]
            k_block = k[:, :, i:end, :]
            v_block = v[:, :, i:end, :]

            scores = q_block @ k_block.transpose(-2, -1) / (self.head_size ** 0.5)

            if mask is not None:
                scores = scores.masked_fill(mask[:, None, i:end, i:end] == 0, float("-inf"))
            elif self.is_decoder:
                scores = scores.masked_fill(self.tril[i:end, i:end] == 0, float("-inf"))  # pyright: ignore

            scores = scores - scores.max(dim=-1, keepdim=True).values
            scores = scores.exp()
            scores_sum = scores.sum(dim=-1, keepdim=True)
            scores = scores / (scores_sum + 1e-6)

            output[:, :, i:end, :] = scores @ v_block

        out = output.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        out = self.dropout(out)

        return out




class EncoderBlock(nn.Module):
    def __init__(self, embedding_size: int, block_size: int, num_heads: int, ffn_hidden_size: int = 512):
        super().__init__()
        self.attn = MultiHeadFlashAttention(embedding_size, num_heads, block_size, is_decoder=False)
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
        self.self_attn = MultiHeadFlashAttention(embedding_size, num_heads, block_size, is_decoder=True)
        self.cross_attn = MultiHeadFlashAttention(embedding_size, num_heads, block_size, is_decoder=False)
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
        ffn_hidden_size: int = None,
    ) -> None:
        super().__init__()
        if ffn_hidden_size is None:
            ffn_hidden_size = 4 * embedding_size
        self.t_emembedding = nn.Embedding(vocab_size, embedding_size)
        self.p_embedding = nn.Embedding(block_size, embedding_size)
        self.decoder_blocks = nn.Sequential(
            *[
                DecoderBlock(embedding_size, block_size, num_heads, ffn_hidden_size)
                for _ in range(num_blocks)
            ]
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
        x = self.t_emembedding(x) + self.p_embedding(torch.arange(T, device=x.device))
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


# benchmarking routines
def regular_attention(q, k, v):
    # simplified attention step, no masking
    scores = q @ k.transpose(-2, -1) / torch.sqrt(torch.tensor(k.shape[-1]))  # B x SL x SL
    scores = F.softmax(scores, dim=-1)
    return scores @ v


def benchmark_fa(num_iter=1000):
    # Simulate input tensors
    batch_size, seq_len, head_dim = 1, 2**15, 64
    device = "mps"

    fa_times = []

    # Warm-up runs (5 iterations to trigger JIT optimization)
    for _ in range(5):
        q = torch.randn(batch_size, seq_len, head_dim, device=device)
        k = torch.randn(batch_size, seq_len, head_dim, device=device)
        v = torch.randn(batch_size, seq_len, head_dim, device=device)
        _ = flash_attention(q, k, v, block_size=256)

    torch.mps.synchronize()  # Ensure GPU is ready

    for _ in tqdm(range(num_iter), desc="Benchmarking"):
        q = torch.randn(batch_size, seq_len, head_dim, device=device)
        k = torch.randn(batch_size, seq_len, head_dim, device=device)
        v = torch.randn(batch_size, seq_len, head_dim, device=device)

        # Measure Flash Attention
        torch.mps.synchronize()
        start_time = time.perf_counter()
        _ = flash_attention(q, k, v, block_size=256)
        torch.mps.synchronize()
        end_time = time.perf_counter()
        fa_times.append(end_time - start_time)

    print(f"Average Flash Attention time: {np.mean(fa_times):.6f} ± {np.std(fa_times):.6f} s")


def benchmark_a(num_iter=1000):
    # Simulate input tensors
    batch_size, seq_len, head_dim = 1, 2**15, 64
    device = "mps"

    a_times = []

    # Warm-up runs (5 iterations to trigger JIT optimization)
    for _ in range(5):
        q = torch.randn(batch_size, seq_len, head_dim, device=device)
        k = torch.randn(batch_size, seq_len, head_dim, device=device)
        v = torch.randn(batch_size, seq_len, head_dim, device=device)
        _ = regular_attention(q, k, v)

    torch.mps.synchronize()  # Ensure GPU is ready
    for _ in tqdm(range(num_iter), desc="Benchmarking"):
        q = torch.randn(batch_size, seq_len, head_dim, device=device)
        k = torch.randn(batch_size, seq_len, head_dim, device=device)
        v = torch.randn(batch_size, seq_len, head_dim, device=device)

        # Measure regular attention
        torch.mps.synchronize()  # Sync before timing
        start_time = time.perf_counter()
        _ = regular_attention(q, k, v)
        torch.mps.synchronize()  # Sync after timing
        end_time = time.perf_counter()
        a_times.append(end_time - start_time)

    print(f"Average Regular Attention time: {np.mean(a_times):.6f} ± {np.std(a_times):.6f} s")
