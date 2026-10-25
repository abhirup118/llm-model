"""
GPT-style decoder-only transformer.

Mirrors the architecture and parameter naming of create-llm's
`models/architectures/gpt.py` (token_embedding, position_embedding,
blocks.N.{ln1,attn.qkv,attn.proj,ln2,ff.fc1,ff.fc2}, ln_f, lm_head) so that
checkpoints are interchangeable, with these fixes/improvements:

  * Next-token loss is computed with a proper shift (logits[:, :-1] vs labels[:, 1:]).
    The upstream template compared position t with label t, which teaches the
    model to copy its input instead of predicting the next token.
  * Uses fused scaled_dot_product_attention (Flash/MemEfficient kernels when available).
  * GPT-2 style scaled init for residual projections.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = -100


@dataclass
class GPTConfig:
    vocab_size: int = 32000
    max_length: int = 512
    layers: int = 6
    heads: int = 6
    dim: int = 384
    dropout: float = 0.1

    def __post_init__(self):
        assert self.dim % self.heads == 0, f"dim ({self.dim}) must be divisible by heads ({self.heads})"

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GPTConfig":
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: d[k] for k in keys if k in d})


class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.heads = config.heads
        self.dim = config.dim
        self.qkv = nn.Linear(config.dim, 3 * config.dim)
        self.proj = nn.Linear(config.dim, config.dim)
        self.attn_dropout_p = config.dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.dim, dim=2)
        q = q.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        k = k.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        v = v.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.dim, 4 * config.dim)
        self.fc2 = nn.Linear(4 * config.dim, config.dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.dropout(F.gelu(self.fc1(x)))))


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.dim)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.dim)
        self.ff = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPTModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.position_embedding = nn.Embedding(config.max_length, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.layers)])
        self.ln_f = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)
        # Scaled init for residual output projections (GPT-2 paper).
        for name, p in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("ff.fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.layers))

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self, exclude_embeddings: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n -= self.position_embedding.weight.numel()
        return n

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,  # accepted for API compat; causal mask suffices for right-padding
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        B, T = input_ids.shape
        if T > self.config.max_length:
            raise ValueError(f"Sequence length {T} exceeds max_length {self.config.max_length}")

        pos = torch.arange(T, dtype=torch.long, device=input_ids.device)
        x = self.dropout(self.token_embedding(input_ids) + self.position_embedding(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        if labels is None:
            # Inference: only the last position is needed for sampling.
            return {"logits": self.lm_head(x[:, -1:, :]), "loss": None}

        logits = self.lm_head(x)
        # Shift so that position t predicts token t+1.
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
        )
        return {"logits": logits, "loss": loss}


def clean_state_dict(state_dict: dict) -> dict:
    """Normalize state dicts from torch.compile / DDP / upstream create-llm checkpoints."""
    out = {}
    for k, v in state_dict.items():
        for prefix in ("_orig_mod.", "module."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        if k.endswith(".attn.mask"):  # upstream registers a causal-mask buffer; we use is_causal
            continue
        out[k] = v
    return out


def infer_config_from_state_dict(state_dict: dict, heads: Optional[int] = None, dropout: float = 0.0) -> GPTConfig:
    """Recover architecture hyper-params from weights (heads cannot be inferred and must be given)."""
    vocab_size, dim = state_dict["token_embedding.weight"].shape
    max_length = state_dict["position_embedding.weight"].shape[0]
    layers = len({k.split(".")[1] for k in state_dict if k.startswith("blocks.")})
    if heads is None:
        raise ValueError("Number of attention heads not found in checkpoint; pass --heads")
    return GPTConfig(vocab_size=vocab_size, max_length=max_length, layers=layers,
                     heads=heads, dim=dim, dropout=dropout)
