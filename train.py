#!/usr/bin/env python3
"""
Train a GPT-style language model from scratch.

Compatible with the project layout produced by create-llm
(https://github.com/theaniketgiri/create-llm):
  * reads `llm.config.js` (via Node, or a built-in fallback parser) or a template JSON
  * uses / produces a Hugging Face `tokenizers` file (tokenizer/tokenizer.json)
  * accepts raw text (file or directory) OR pre-tokenized `data/processed/{train,val}.pt`
  * writes checkpoints with the same core keys (`model_state_dict`, `optimizer_state_dict`, `step`, ...)
    plus the model config and the full tokenizer, so generate.py needs only the checkpoint.

Examples:
  python train.py --data data/raw/input.txt --template nano --max_steps 2000
  python train.py --config llm.config.js --data data/processed
  python train.py --resume checkpoints/checkpoint-1000.pt
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from model import IGNORE_INDEX, GPTConfig, GPTModel, clean_state_dict

SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>"]  # same order/ids as create-llm (pad=0, unk=1, bos=2, eos=3)
TEXT_SUFFIXES = {".txt", ".md", ".text", ".csv", ".json", ".jsonl", ".py"}

# Hyper-parameters of create-llm's templates/*.json
TEMPLATES: Dict[str, dict] = {
    "nano": {
        "model": dict(layers=3, heads=4, dim=128, vocab_size=5000, max_length=256, dropout=0.1),
        "training": dict(batch_size=8, learning_rate=5e-4, warmup_steps=100, max_steps=1000, eval_interval=200,
                         save_interval=500, weight_decay=0.01, gradient_clip=1.0, mixed_precision=False,
                         gradient_accumulation_steps=1),
        "data": dict(max_length=256, stride=128, val_split=0.1, shuffle=True),
    },
    "tiny": {
        "model": dict(layers=4, heads=4, dim=256, vocab_size=10000, max_length=512, dropout=0.2),
        "training": dict(batch_size=16, learning_rate=6e-4, warmup_steps=500, max_steps=10000, eval_interval=500,
                         save_interval=2000, weight_decay=0.01, gradient_clip=1.0, mixed_precision=False,
                         gradient_accumulation_steps=1),
        "data": dict(max_length=512, stride=256, val_split=0.1, shuffle=True),
    },
    "medium": {
        "model": dict(layers=8, heads=8, dim=512, vocab_size=8000, max_length=512, dropout=0.1),
        "training": dict(batch_size=16, learning_rate=4e-4, warmup_steps=1000, max_steps=20000, eval_interval=500,
                         save_interval=2000, weight_decay=0.01, gradient_clip=1.0, mixed_precision=True,
                         gradient_accumulation_steps=1),
        "data": dict(max_length=512, stride=256, val_split=0.05, shuffle=True),
    },
    "small": {
        "model": dict(layers=12, heads=12, dim=768, vocab_size=32000, max_length=1024, dropout=0.1),
        "training": dict(batch_size=32, learning_rate=3e-4, warmup_steps=2000, max_steps=100000, eval_interval=1000,
                         save_interval=5000, weight_decay=0.01, gradient_clip=1.0, mixed_precision=True,
                         gradient_accumulation_steps=2),
        "data": dict(max_length=1024, stride=512, val_split=0.05, shuffle=True),
    },
    "base": {
        "model": dict(layers=24, heads=16, dim=1536, vocab_size=50000, max_length=2048, dropout=0.1),
        "training": dict(batch_size=64, learning_rate=2e-4, warmup_steps=5000, max_steps=500000, eval_interval=2000,
                         save_interval=10000, weight_decay=0.01, gradient_clip=1.0, mixed_precision=True,
                         gradient_accumulation_steps=4),
        "data": dict(max_length=2048, stride=1024, val_split=0.05, shuffle=True),
    },
}


# --------------------------------------------------------------------------------------------
# Config loading (Pure Python / JSON)
# --------------------------------------------------------------------------------------------

def load_config_file(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    if p.suffix == ".py":
        import importlib.util
        spec = importlib.util.spec_from_file_location("user_config", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out = {}
        for section in ("model", "training", "data", "tokenizer"):
            if hasattr(mod, section):
                val = getattr(mod, section)
                out[section] = val if isinstance(val, dict) else val.__dict__
        return out
    # Otherwise treat as JSON
    with open(p, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg.get("config", cfg)


def build_config(args: argparse.Namespace) -> dict:
    """Merge: template defaults <- config file (JSON or Python) <- CLI overrides."""
    cfg = {k: dict(v) for k, v in TEMPLATES[args.template].items()}
    if args.config:
        file_cfg = load_config_file(args.config)
        for section in ("model", "training", "data", "tokenizer", "checkpoints", "logging"):
            cfg.setdefault(section, {}).update(file_cfg.get(section) or {})

    m, t, d = cfg["model"], cfg["training"], cfg["data"]
    cli_map = {
        ("model", "layers"): args.layers, ("model", "heads"): args.heads, ("model", "dim"): args.dim,
        ("model", "max_length"): args.max_length, ("model", "dropout"): args.dropout,
        ("model", "vocab_size"): args.vocab_size,
        ("training", "batch_size"): args.batch_size, ("training", "learning_rate"): args.lr,
        ("training", "warmup_steps"): args.warmup_steps, ("training", "max_steps"): args.max_steps,
        ("training", "eval_interval"): args.eval_interval, ("training", "save_interval"): args.save_interval,
        ("training", "weight_decay"): args.weight_decay, ("training", "gradient_clip"): args.grad_clip,
        ("training", "gradient_accumulation_steps"): args.grad_accum,
        ("data", "stride"): args.stride, ("data", "val_split"): args.val_split,
    }
    for (section, key), val in cli_map.items():
        if val is not None:
            cfg[section][key] = val
    # The training window can never exceed the model's context; --max_length sets both.
    if args.max_length is not None:
        d["max_length"] = args.max_length
    d["max_length"] = min(d.get("max_length", m["max_length"]), m["max_length"])
    d["stride"] = min(d.get("stride", d["max_length"]), d["max_length"])
    t.setdefault("min_lr_ratio", 0.1)
    t.setdefault("log_interval", (cfg.get("logging") or {}).get("log_interval", 10))
    if args.log_interval is not None:
        t["log_interval"] = args.log_interval
    return cfg


# --------------------------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------------------------

def list_text_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"No text files ({', '.join(sorted(TEXT_SUFFIXES))}) under {path}")
    return files


def train_tokenizer(files: List[Path], vocab_size: int, min_frequency: int, out_path: Path):
    """Byte-level BPE: lossless round-trip (preserves whitespace/newlines) and never emits <unk>."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True,
    )
    tok.train([str(f) for f in files], trainer)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))
    log(f"✓ Trained byte-level BPE tokenizer ({tok.get_vocab_size():,} tokens) -> {out_path}")
    return tok


def load_or_train_tokenizer(args, cfg, raw_files: Optional[List[Path]]):
    from tokenizers import Tokenizer

    tok_path = Path(args.tokenizer)
    if tok_path.exists():
        tok = Tokenizer.from_file(str(tok_path))
        log(f"✓ Loaded tokenizer {tok_path} ({tok.get_vocab_size():,} tokens)")
        return tok, tok_path
    if not raw_files:
        raise FileNotFoundError(f"Tokenizer {tok_path} not found and no raw text to train one from.")
    tcfg = cfg.get("tokenizer", {})
    vocab = tcfg.get("vocab_size", cfg["model"]["vocab_size"]) if args.vocab_size is None else args.vocab_size
    return train_tokenizer(raw_files, vocab, tcfg.get("min_frequency", 2), tok_path), tok_path


def special_ids(tok) -> Dict[str, Optional[int]]:
    return {name: tok.token_to_id(t) for name, t in
            zip(("pad", "unk", "bos", "eos"), SPECIAL_TOKENS)}


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------

class TokenWindowDataset(Dataset):
    """Sliding windows of `block_size` tokens over a 1-D token stream (labels == inputs; model shifts)."""

    def __init__(self, tokens: torch.Tensor, block_size: int, stride: int):
        assert tokens.dim() == 1
        self.tokens, self.block, self.stride = tokens, block_size, max(1, stride)
        self.n = 1 if len(tokens) <= block_size else (len(tokens) - block_size) // self.stride + 1

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> torch.Tensor:
        s = i * self.stride
        return self.tokens[s:s + self.block]


class SequenceDataset(Dataset):
    """Pre-tokenized sequences, e.g. create-llm's data/processed/train.pt ([N, L] tensor or list of lists)."""

    def __init__(self, seqs, block_size: int):
        self.seqs, self.block = seqs, block_size

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, i: int) -> torch.Tensor:
        s = self.seqs[i]
        s = s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.long)
        return s[: self.block]


def make_collate(pad_id: int):
    """Right-pad to the longest sequence; padded positions get IGNORE_INDEX labels."""

    def collate(batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        L = max(len(x) for x in batch)
        input_ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), L), dtype=torch.long)
        for i, x in enumerate(batch):
            input_ids[i, : len(x)] = x
            attention_mask[i, : len(x)] = 1
        labels = input_ids.masked_fill(attention_mask == 0, IGNORE_INDEX)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return collate


def encode_files(tok, files: List[Path], eos_id: Optional[int]) -> torch.Tensor:
    """Encode each file as a document, separated by </s>. Large files are encoded in line-aligned chunks."""
    ids: List[int] = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        chunks, chunk_chars, buf, size = [], 1 << 20, [], 0
        for line in text.splitlines(keepends=True):
            buf.append(line); size += len(line)
            if size >= chunk_chars:
                chunks.append("".join(buf)); buf, size = [], 0
        if buf:
            chunks.append("".join(buf))
        for enc in tok.encode_batch(chunks, add_special_tokens=False):
            ids.extend(enc.ids)
        if eos_id is not None:
            ids.append(eos_id)
    return torch.tensor(ids, dtype=torch.long)


def build_datasets(args, cfg, tok) -> Tuple[Dataset, Optional[Dataset]]:
    data_path = Path(args.data)
    block, stride, val_split = cfg["data"]["max_length"], cfg["data"]["stride"], cfg["data"]["val_split"]

    # Pre-tokenized create-llm data (directory with train.pt[/val.pt] or a single .pt file).
    pt_train = data_path / "train.pt" if data_path.is_dir() else (data_path if data_path.suffix == ".pt" else None)
    if pt_train is not None and pt_train.exists():
        def load_pt(p: Path):
            obj = torch.load(p, map_location="cpu")
            return obj.long() if isinstance(obj, torch.Tensor) else obj

        train = load_pt(pt_train)
        val_file = pt_train.with_name("val.pt")
        val = load_pt(val_file) if val_file.exists() and val_file != pt_train else None
        if isinstance(train, torch.Tensor) and train.dim() == 1:  # flat token stream
            if val is None:
                cut = int(len(train) * (1 - val_split)); train, val = train[:cut], train[cut:]
            return (TokenWindowDataset(train, block, stride),
                    TokenWindowDataset(val, block, block) if val is not None and len(val) > 1 else None)
        if val is None and val_split > 0 and len(train) > 1:
            cut = max(1, int(len(train) * (1 - val_split))); train, val = train[:cut], train[cut:]
        log(f"✓ Loaded pre-tokenized data from {pt_train.parent}")
        return SequenceDataset(train, block), (SequenceDataset(val, block) if val is not None and len(val) else None)

    # Raw text.
    files = list_text_files(data_path)
    t0 = time.time()
    tokens = encode_files(tok, files, special_ids(tok)["eos"])
    log(f"✓ Tokenized {len(files)} file(s): {len(tokens):,} tokens in {time.time() - t0:.1f}s")
    if len(tokens) < 2 * block:
        log(f"⚠️  Only {len(tokens):,} tokens for a context of {block}; consider more data or a smaller --max_length")
    cut = int(len(tokens) * (1 - val_split)) if val_split > 0 else len(tokens)
    train_ds = TokenWindowDataset(tokens[:cut], block, stride)
    val_ds = TokenWindowDataset(tokens[cut:], block, block) if len(tokens) - cut > 1 else None
    return train_ds, val_ds


def infinite_loader(loader: DataLoader):
    epoch = 0
    while True:
        for batch in loader:
            yield epoch, batch
        epoch += 1


# --------------------------------------------------------------------------------------------
# Training utilities
# --------------------------------------------------------------------------------------------

def log(msg: str):
    print(msg, flush=True)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def pick_precision(requested: str, device: torch.device) -> Optional[torch.dtype]:
    """Return autocast dtype or None for fp32. 'auto' = bf16 (or fp16+GradScaler) on CUDA, fp32 elsewhere."""
    if requested == "fp32":
        return None
    if requested == "auto":
        if device.type != "cuda":
            return None  # AMP on CPU/MPS is rarely a win for small models
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[requested]


def configure_optimizer(model: torch.nn.Module, lr: float, weight_decay: float, device: torch.device):
    """AdamW; decay only matrices/embeddings (dim >= 2), not biases or LayerNorm params."""
    decay, no_decay = [], []
    for _, p in model.named_parameters():  # named_parameters de-duplicates the tied embedding
        if p.requires_grad:
            (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    extra = {"fused": True} if device.type == "cuda" else {}
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, **extra)


def lr_lambda_factory(warmup: int, max_steps: int, min_ratio: float):
    """Linear warmup, then cosine decay to `min_ratio * lr`."""

    def fn(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, max_steps - warmup))
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return fn


@torch.no_grad()
def evaluate(model, loader, device, amp_ctx, max_batches: int) -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with amp_ctx():
            loss = model(**batch)["loss"]
        n = int((batch["labels"][:, 1:] != IGNORE_INDEX).sum())
        total_loss += loss.item() * n
        total_tokens += n
    model.train()
    return total_loss / max(1, total_tokens)


def save_checkpoint(path: Path, *, model, optimizer, scheduler, scaler, step, epoch, loss, best_val_loss,
                    model_cfg: GPTConfig, cfg: dict, tok, tok_path: Path, args):
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    ckpt = {
        "step": step,
        "epoch": epoch,
        "model_state_dict": raw.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        "loss": loss,
        "best_val_loss": best_val_loss,
        "model_config": model_cfg.to_dict(),
        "config": cfg,  # full merged config (same key create-llm uses)
        "tokenizer": tok.to_str(),  # embedded so generation needs only the checkpoint
        "tokenizer_path": str(tok_path),
        "special_tokens": special_ids(tok),
        "args": vars(args),
        "timestamp": datetime.now().isoformat(),
        "rng": {"torch": torch.get_rng_state(), "python": random.getstate(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)  # atomic: never leaves a half-written checkpoint
    log(f"💾 Saved checkpoint -> {path}")


def prune_checkpoints(out_dir: Path, keep: int):
    if keep <= 0:
        return
    step_ckpts = sorted(out_dir.glob("checkpoint-[0-9]*.pt"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
    for p in step_ckpts[:-keep]:
        p.unlink(missing_ok=True)


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a GPT language model from scratch",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("data / config")
    g.add_argument("--data", default="data/raw", help="Raw text file/dir, or dir/file with pre-tokenized train.pt")
    g.add_argument("--tokenizer", default="tokenizer/tokenizer.json", help="Tokenizer path (trained if missing)")
    g.add_argument("--config", default=None, help="llm.config.js or template .json (overrides --template)")
    g.add_argument("--template", default="nano", choices=list(TEMPLATES), help="Built-in size preset")
    g.add_argument("--out_dir", default="checkpoints")
    g.add_argument("--resume", default=None, help="Checkpoint to resume from")
    g.add_argument("--stride", type=int, default=None, help="Sliding-window stride over the token stream")
    g.add_argument("--val_split", type=float, default=None)

    g = p.add_argument_group("model (override config)")
    for name, typ in [("layers", int), ("heads", int), ("dim", int), ("max_length", int),
                      ("dropout", float), ("vocab_size", int)]:
        g.add_argument(f"--{name}", type=typ, default=None)

    g = p.add_argument_group("optimization (override config)")
    for name, typ in [("batch_size", int), ("lr", float), ("warmup_steps", int), ("max_steps", int),
                      ("weight_decay", float), ("grad_clip", float), ("grad_accum", int),
                      ("eval_interval", int), ("save_interval", int), ("log_interval", int)]:
        g.add_argument(f"--{name}", type=typ, default=None)
    g.add_argument("--eval_iters", type=int, default=50, help="Max validation batches per evaluation")
    g.add_argument("--keep_checkpoints", type=int, default=3, help="How many step checkpoints to keep")

    g = p.add_argument_group("runtime")
    g.add_argument("--device", default="auto", help="auto | cuda | cuda:N | mps | cpu")
    g.add_argument("--precision", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    g.add_argument("--compile", action="store_true", help="torch.compile the model (PyTorch 2.x)")
    g.add_argument("--num_workers", type=int, default=0)
    g.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main():
    args = parse_args()
    resume_ckpt = None
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        # Re-use the original run's config so architecture/data match; a few loop settings may be overridden
        # (e.g. `--max_steps 5000` to continue training past the original budget).
        cfg = resume_ckpt.get("config") or build_config(args)
        for key, val in (("max_steps", args.max_steps), ("eval_interval", args.eval_interval),
                         ("save_interval", args.save_interval), ("log_interval", args.log_interval)):
            if val is not None:
                cfg["training"][key] = val
    else:
        cfg = build_config(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    tcfg = cfg["training"]
    amp_dtype = pick_precision(args.precision, device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    log("=" * 64)
    log(f"🚀 GPT training | device={device} | precision={amp_dtype or torch.float32}")
    log("=" * 64)

    # ---- tokenizer & data -------------------------------------------------------------------
    data_path = Path(args.data)
    is_pretokenized = (data_path.is_dir() and (data_path / "train.pt").exists()) or data_path.suffix == ".pt"
    if resume_ckpt is not None and resume_ckpt.get("tokenizer"):
        from tokenizers import Tokenizer
        tok = Tokenizer.from_str(resume_ckpt["tokenizer"])
        tok_path = Path(resume_ckpt.get("tokenizer_path", args.tokenizer))
    else:
        raw_files = None if is_pretokenized else list_text_files(data_path)
        tok, tok_path = load_or_train_tokenizer(args, cfg, raw_files)
    ids = special_ids(tok)
    pad_id = ids["pad"] if ids["pad"] is not None else 0

    train_ds, val_ds = build_datasets(args, cfg, tok)
    collate = make_collate(pad_id)
    bs = tcfg["batch_size"]
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=cfg["data"].get("shuffle", True),
                              drop_last=len(train_ds) >= bs, collate_fn=collate,
                              num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate,
                            num_workers=args.num_workers, pin_memory=pin) if val_ds is not None else None
    log(f"✓ Train sequences: {len(train_ds):,} | Val sequences: {len(val_ds) if val_ds else 0:,} "
        f"| context={cfg['data']['max_length']}")

    # ---- model ------------------------------------------------------------------------------
    mcfg = dict(cfg["model"])
    if resume_ckpt is not None and resume_ckpt.get("model_config"):
        model_cfg = GPTConfig.from_dict(resume_ckpt["model_config"])
    else:
        if mcfg.get("vocab_size") != tok.get_vocab_size():
            log(f"ℹ️  Using tokenizer vocab size {tok.get_vocab_size():,} (config said {mcfg.get('vocab_size')})")
        mcfg["vocab_size"] = tok.get_vocab_size()
        model_cfg = GPTConfig.from_dict(mcfg)
    model = GPTModel(model_cfg).to(device)
    log(f"✓ Model: {model.count_parameters():,} parameters | {model_cfg}")

    # ---- optimizer / scheduler / AMP --------------------------------------------------------
    max_steps, warmup = tcfg["max_steps"], tcfg.get("warmup_steps", 0)
    optimizer = configure_optimizer(model, tcfg["learning_rate"], tcfg.get("weight_decay", 0.01), device)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda_factory(warmup, max_steps, tcfg.get("min_lr_ratio", 0.1)))
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16 and device.type == "cuda"))
    amp_ctx = (lambda: torch.autocast(device_type=device.type, dtype=amp_dtype)) if amp_dtype else contextlib.nullcontext

    step, epoch, best_val = 0, 0, float("inf")
    if resume_ckpt is not None:
        model.load_state_dict(clean_state_dict(resume_ckpt["model_state_dict"]))
        if resume_ckpt.get("optimizer_state_dict"):
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if resume_ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        if resume_ckpt.get("scaler_state_dict") and scaler.is_enabled():
            scaler.load_state_dict(resume_ckpt["scaler_state_dict"])
        step, epoch = resume_ckpt.get("step", 0), resume_ckpt.get("epoch", 0)
        best_val = resume_ckpt.get("best_val_loss", float("inf")) or float("inf")
        log(f"↻ Resumed from {args.resume} at step {step}")
        del resume_ckpt

    if args.compile:
        model = torch.compile(model)

    accum = max(1, tcfg.get("gradient_accumulation_steps", 1))
    grad_clip = tcfg.get("gradient_clip", 1.0)
    log_every, eval_every = tcfg.get("log_interval", 10), tcfg.get("eval_interval", 200)
    save_every = tcfg.get("save_interval", 500)
    out_dir = Path(args.out_dir)
    ckpt_kwargs = dict(model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                       model_cfg=model_cfg, cfg=cfg, tok=tok, tok_path=tok_path, args=args)
    (out_dir).mkdir(parents=True, exist_ok=True)
    metrics_file = open(out_dir / "train_log.jsonl", "a", encoding="utf-8")

    log(f"Training for {max_steps:,} steps | batch={bs} x accum={accum} "
        f"| tokens/step≈{bs * accum * cfg['data']['max_length']:,}")
    log("-" * 64)

    batches = infinite_loader(train_loader)
    model.train()
    running_loss, running_steps, running_tokens, t_last, loss_val = 0.0, 0, 0, time.time(), float("nan")

    try:
        while step < max_steps:
            # ---- one optimizer step (with gradient accumulation) ----
            step_loss = 0.0
            for _ in range(accum):
                epoch_i, batch = next(batches)
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                with amp_ctx():
                    loss = model(**batch)["loss"] / accum
                scaler.scale(loss).backward()
                step_loss += loss.item()
                running_tokens += int(batch["attention_mask"].sum())
            epoch = max(epoch, epoch_i)

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip) if grad_clip else None
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            running_loss += step_loss
            running_steps += 1
            loss_val = step_loss

            # ---- logging ----
            if step % log_every == 0 or step == 1:
                n = running_steps
                dt = time.time() - t_last
                avg = running_loss / n
                rec = {"step": step, "loss": avg, "ppl": math.exp(min(avg, 20)),
                       "lr": scheduler.get_last_lr()[0], "ms_per_step": 1000 * dt / n,
                       "tok_per_s": running_tokens / max(dt, 1e-9),
                       "grad_norm": float(grad_norm) if grad_norm is not None else None, "epoch": epoch}
                log(f"step {step:>6}/{max_steps} | loss {avg:.4f} | ppl {rec['ppl']:9.2f} | lr {rec['lr']:.2e} "
                    f"| {rec['ms_per_step']:7.1f} ms/step | {rec['tok_per_s']:9.0f} tok/s | epoch {epoch}")
                metrics_file.write(json.dumps(rec) + "\n"); metrics_file.flush()
                running_loss, running_steps, running_tokens, t_last = 0.0, 0, 0, time.time()

            # ---- evaluation ----
            if val_loader is not None and (step % eval_every == 0 or step == max_steps):
                t_eval = time.time()
                val_loss = evaluate(model, val_loader, device, amp_ctx, args.eval_iters)
                log(f"  ↳ eval | val_loss {val_loss:.4f} | val_ppl {math.exp(min(val_loss, 20)):.2f}")
                metrics_file.write(json.dumps({"step": step, "val_loss": val_loss}) + "\n"); metrics_file.flush()
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(out_dir / "checkpoint-best.pt", step=step, epoch=epoch, loss=loss_val,
                                    best_val_loss=best_val, **ckpt_kwargs)
                t_last += time.time() - t_eval  # exclude eval time from step timing

            # ---- periodic checkpoint ----
            if step % save_every == 0 and step < max_steps:
                t_save = time.time()
                save_checkpoint(out_dir / f"checkpoint-{step}.pt", step=step, epoch=epoch, loss=loss_val,
                                best_val_loss=best_val, **ckpt_kwargs)
                prune_checkpoints(out_dir, args.keep_checkpoints)
                t_last += time.time() - t_save

        save_checkpoint(out_dir / "checkpoint-final.pt", step=step, epoch=epoch, loss=loss_val,
                        best_val_loss=best_val, **ckpt_kwargs)
        log("-" * 64)
        log(f"✅ Done. {step} steps | best val loss {best_val:.4f}")
    except KeyboardInterrupt:
        log("\n⏹  Interrupted — saving checkpoint...")
        save_checkpoint(out_dir / f"checkpoint-interrupt-{step}.pt", step=step, epoch=epoch, loss=loss_val,
                        best_val_loss=best_val, **ckpt_kwargs)
        sys.exit(130)
    finally:
        metrics_file.close()


if __name__ == "__main__":
    main()
