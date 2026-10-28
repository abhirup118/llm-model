#!/usr/bin/env python3
"""
Knowledge Distillation for GPT Language Models (Teacher-Student paradigm).

Transfers knowledge from a large Hugging Face pre-trained teacher (e.g., 'gpt2')
to a compact, efficient student GPT model (implemented in model.py).

Loss:
    Total Loss = alpha * Loss_KD + (1 - alpha) * Loss_CE
    where Loss_KD = T^2 * KL(LogSoftmax(student_logits / T) || Softmax(teacher_logits / T))

Examples:
    # Quick distillation on TinyStories using GPT-2 as teacher
    python distill.py --data data/raw/tinystories.txt --teacher gpt2 --student_template nano \
        --max_steps 1000 --batch_size 8 --out_dir checkpoints_distill

    # Distill with custom temperature and alpha
    python distill.py --data data/raw --teacher gpt2 --student_template tiny \
        --temperature 2.0 --alpha 0.6 --lr 5e-4 --max_steps 3000
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import IGNORE_INDEX, GPTConfig, GPTModel, clean_state_dict

TEXT_SUFFIXES = {".txt", ".md", ".text", ".csv", ".json", ".jsonl", ".py"}

# Presets for the student model architecture
STUDENT_TEMPLATES: Dict[str, dict] = {
    "nano": dict(layers=3, heads=4, dim=128, dropout=0.1),    # ~1.5M params with GPT-2 vocab
    "tiny": dict(layers=4, heads=4, dim=256, dropout=0.1),    # ~5.5M params with GPT-2 vocab
    "medium": dict(layers=6, heads=6, dim=384, dropout=0.1),  # ~15M params with GPT-2 vocab
}


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
    if requested == "fp32":
        return None
    if requested == "auto":
        if device.type != "cuda":
            return None
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[requested]


# --------------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------------

def list_text_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"No text files under {path}")
    return files


def encode_files(hf_tokenizer, files: List[Path], max_chars_per_file: Optional[int] = None) -> torch.Tensor:
    """Encode text files into a single continuous 1D token stream using the teacher's tokenizer.
    
    Files are encoded in line-aligned chunks using the underlying Rust fast tokenizer to
    prevent Hugging Face sequence length warnings and maximize tokenization speed.
    """
    all_tokens: List[int] = []
    eos_id = hf_tokenizer.eos_token_id
    fast_tok = getattr(hf_tokenizer, "_tokenizer", None)

    for f in files:
        with open(f, "r", encoding="utf-8", errors="replace") as fp:
            text = fp.read(max_chars_per_file) if max_chars_per_file else fp.read()

        # Chunk into ~1MB text segments to keep memory bounded and enable parallel encoding
        chunks, chunk_chars, buf, size = [], 1 << 20, [], 0
        for line in text.splitlines(keepends=True):
            buf.append(line)
            size += len(line)
            if size >= chunk_chars:
                chunks.append("".join(buf))
                buf, size = [], 0
        if buf:
            chunks.append("".join(buf))

        if fast_tok is not None:
            # Multi-threaded Rust tokenization (no sequence length warning)
            for enc in fast_tok.encode_batch(chunks, add_special_tokens=False):
                all_tokens.extend(enc.ids)
        else:
            for c in chunks:
                all_tokens.extend(hf_tokenizer.encode(c, add_special_tokens=False, truncation=False))

        if eos_id is not None:
            all_tokens.append(eos_id)

    return torch.tensor(all_tokens, dtype=torch.long)


class TokenWindowDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, block_size: int, stride: int):
        assert tokens.dim() == 1
        self.tokens, self.block, self.stride = tokens, block_size, max(1, stride)
        self.n = 1 if len(tokens) <= block_size else (len(tokens) - block_size) // self.stride + 1

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> torch.Tensor:
        s = i * self.stride
        return self.tokens[s:s + self.block]


def make_collate(pad_id: int):
    def collate_fn(batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = torch.stack(batch)
        return {"input_ids": x, "labels": x.clone()}
    return collate_fn


def infinite_loader(loader: DataLoader):
    epoch = 0
    while True:
        for batch in loader:
            yield epoch, batch
        epoch += 1


# --------------------------------------------------------------------------------------------
# Optimizer & Scheduler
# --------------------------------------------------------------------------------------------

def configure_optimizer(model: torch.nn.Module, lr: float, weight_decay: float, device: torch.device):
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if p.requires_grad:
            (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    extra = {"fused": True} if device.type == "cuda" else {}
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, **extra)


def lr_lambda_factory(warmup: int, max_steps: int, min_ratio: float = 0.1):
    def fn(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, max_steps - warmup))
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return fn


# --------------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------------

@torch.no_grad()
def evaluate_distillation(student: GPTModel, teacher: nn.Module, loader: DataLoader,
                          device: torch.device, amp_ctx, max_batches: int,
                          temperature: float, alpha: float):
    student.eval()
    teacher.eval()
    total_loss, total_kd, total_ce = 0.0, 0.0, 0.0
    total_tokens = 0
    T2 = temperature * temperature

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        shift_labels = labels[:, 1:].contiguous()
        n_tokens = shift_labels.numel()

        with amp_ctx():
            s_out = student(input_ids, labels=labels)
            s_logits = s_out["logits"][:, :-1, :].contiguous()
            loss_ce = s_out["loss"]

            t_out = teacher(input_ids)
            t_logits = t_out.logits[:, :-1, :].contiguous()

            # Soft targets KL divergence (flattened across tokens for per-token normalization)
            p_teacher = F.softmax(t_logits / temperature, dim=-1)
            log_p_student = F.log_softmax(s_logits / temperature, dim=-1)
            loss_kd = F.kl_div(
                log_p_student.view(-1, log_p_student.size(-1)),
                p_teacher.view(-1, p_teacher.size(-1)),
                reduction="batchmean",
            ) * T2

            loss = alpha * loss_kd + (1.0 - alpha) * loss_ce

        total_loss += loss.item() * n_tokens
        total_kd += loss_kd.item() * n_tokens
        total_ce += loss_ce.item() * n_tokens
        total_tokens += n_tokens

    student.train()
    denom = max(1, total_tokens)
    return {
        "val_loss": total_loss / denom,
        "val_kd": total_kd / denom,
        "val_ce": total_ce / denom,
        "val_ppl": math.exp(min(20.0, total_ce / denom)),
    }


# --------------------------------------------------------------------------------------------
# Main Distillation Procedure
# --------------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Knowledge Distillation for GPT (Teacher-Student)",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Teacher & Student configs
    p.add_argument("--teacher", default="gpt2", help="Hugging Face teacher model identifier")
    p.add_argument("--student_template", default="nano", choices=list(STUDENT_TEMPLATES),
                   help="Size template for the student model")
    p.add_argument("--student_layers", type=int, default=None, help="Override student layers")
    p.add_argument("--student_heads", type=int, default=None, help="Override student heads")
    p.add_argument("--student_dim", type=int, default=None, help="Override student embedding dim")
    p.add_argument("--max_length", type=int, default=256, help="Sequence block size")

    # Distillation hyperparameters
    p.add_argument("--temperature", type=float, default=2.0, help="Softmax temperature (T)")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Loss weight for soft KD target (1 - alpha for hard target)")

    # Data
    p.add_argument("--data", default="data/raw", help="Path to text file or directory of .txt files")
    p.add_argument("--val_split", type=float, default=0.05, help="Validation fraction")
    p.add_argument("--stride", type=int, default=None, help="Sliding window stride (default = max_length)")

    # Optimization
    p.add_argument("--batch_size", type=int, default=8, help="Batch size per step")
    p.add_argument("--lr", type=float, default=5e-4, help="Peak learning rate for student")
    p.add_argument("--warmup_steps", type=int, default=100, help="Linear warmup steps")
    p.add_argument("--max_steps", type=int, default=1000, help="Total training steps")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")

    # Intervals & Output
    p.add_argument("--out_dir", default="checkpoints_distill", help="Directory to save checkpoints")
    p.add_argument("--eval_interval", type=int, default=100, help="Steps between validation evaluations")
    p.add_argument("--save_interval", type=int, default=500, help="Steps between checkpoint saves")
    p.add_argument("--log_interval", type=int, default=10, help="Steps between stdout logs")
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    p.add_argument("--precision", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    amp_dtype = pick_precision(args.precision, device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log(f"🧠 Knowledge Distillation | Teacher={args.teacher} | Student={args.student_template}")
    log(f"⚡ Device={device} | Precision={amp_dtype or torch.float32} | Temp={args.temperature} | Alpha={args.alpha}")
    log("=" * 70)

    # 1. Load Hugging Face Teacher & Tokenizer
    log(f"⏳ Loading Teacher model & tokenizer: '{args.teacher}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(args.teacher).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    teacher_params = sum(p.numel() for p in teacher.parameters())
    log(f"✓ Teacher loaded: {teacher_params:,} parameters (FROZEN)")

    # 2. Build Student Model
    vocab_size = tokenizer.vocab_size
    template = STUDENT_TEMPLATES[args.student_template]
    student_cfg = GPTConfig(
        vocab_size=vocab_size,
        max_length=args.max_length,
        layers=args.student_layers or template["layers"],
        heads=args.student_heads or template["heads"],
        dim=args.student_dim or template["dim"],
        dropout=template["dropout"],
    )
    student = GPTModel(student_cfg).to(device)
    student_params = student.count_parameters()
    compression_ratio = teacher_params / max(1, student_params)
    log(f"✓ Student initialized: {student_params:,} parameters | Compression: {compression_ratio:.1f}x smaller")
    log(f"  Architecture: {student_cfg.layers} layers, {student_cfg.heads} heads, dim={student_cfg.dim}")

    # 3. Tokenize & Prepare Data
    data_path = Path(args.data)
    files = list_text_files(data_path)
    log(f"⏳ Tokenizing {len(files)} file(s) with teacher tokenizer...")
    tokens = encode_files(tokenizer, files)
    log(f"✓ Total tokens in corpus: {len(tokens):,}")

    block_size = args.max_length
    stride = args.stride or block_size
    cut = int(len(tokens) * (1 - args.val_split)) if args.val_split > 0 else len(tokens)

    train_ds = TokenWindowDataset(tokens[:cut], block_size, stride)
    val_ds = TokenWindowDataset(tokens[cut:], block_size, block_size) if len(tokens) - cut > block_size else None
    collate = make_collate(tokenizer.pad_token_id)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=len(train_ds) >= args.batch_size, collate_fn=collate, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, pin_memory=pin) if val_ds is not None else None
    log(f"✓ Train sequences: {len(train_ds):,} | Val sequences: {len(val_ds) if val_ds else 0:,}")

    # 4. Optimization Setup
    optimizer = configure_optimizer(student, args.lr, args.weight_decay, device)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda_factory(args.warmup_steps, args.max_steps, min_ratio=0.1)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16 and device.type == "cuda"))
    amp_ctx = (lambda: torch.autocast(device_type=device.type, dtype=amp_dtype)) if amp_dtype else contextlib.nullcontext

    # 5. Training Loop
    log("\n🏁 Starting Knowledge Distillation Training...")
    data_iter = infinite_loader(train_loader)
    train_log_path = out_dir / "distill_log.jsonl"
    T2 = args.temperature * args.temperature
    alpha = args.alpha

    t0 = time.time()
    best_val_loss = float("inf")
    student.train()

    for step in range(1, args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accum_loss, accum_kd, accum_ce = 0.0, 0.0, 0.0

        for _ in range(args.grad_accum):
            epoch, batch = next(data_iter)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with amp_ctx():
                # Student forward
                s_out = student(input_ids, labels=labels)
                s_logits = s_out["logits"][:, :-1, :].contiguous()
                loss_ce = s_out["loss"]

                # Teacher forward (no gradients needed)
                with torch.no_grad():
                    t_out = teacher(input_ids)
                    t_logits = t_out.logits[:, :-1, :].contiguous()

                # Soft KD loss (flattened across tokens for per-token normalization)
                p_teacher = F.softmax(t_logits / args.temperature, dim=-1)
                log_p_student = F.log_softmax(s_logits / args.temperature, dim=-1)
                loss_kd = F.kl_div(
                    log_p_student.view(-1, log_p_student.size(-1)),
                    p_teacher.view(-1, p_teacher.size(-1)),
                    reduction="batchmean",
                ) * T2

                # Combined objective
                loss = alpha * loss_kd + (1.0 - alpha) * loss_ce
                scaled_loss = loss / args.grad_accum

            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            accum_loss += loss.item() / args.grad_accum
            accum_kd += loss_kd.item() / args.grad_accum
            accum_ce += loss_ce.item() / args.grad_accum

        # Step optimizer
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            optimizer.step()

        scheduler.step()

        # Logging
        if step % args.log_interval == 0 or step == 1:
            dt = time.time() - t0
            tok_per_s = (args.batch_size * args.grad_accum * block_size * args.log_interval) / max(1e-5, dt) if step > 1 else 0
            cur_lr = scheduler.get_last_lr()[0]
            ppl = math.exp(min(20.0, accum_ce))

            log(f"Step {step:5d}/{args.max_steps} | Loss: {accum_loss:.4f} "
                f"(KD: {accum_kd:.4f}, CE: {accum_ce:.4f}) | PPL: {ppl:6.2f} | "
                f"LR: {cur_lr:.2e} | {tok_per_s:6.0f} tok/s")

            with open(train_log_path, "a") as f:
                f.write(json.dumps({
                    "step": step, "loss": accum_loss, "loss_kd": accum_kd,
                    "loss_ce": accum_ce, "ppl": ppl, "lr": cur_lr,
                }) + "\n")
            t0 = time.time()

        # Periodic Validation
        if val_loader is not None and (step % args.eval_interval == 0 or step == args.max_steps):
            metrics = evaluate_distillation(student, teacher, val_loader, device, amp_ctx,
                                            max_batches=20, temperature=args.temperature, alpha=alpha)
            log(f"  ↳ 🔍 Eval: Val Loss={metrics['val_loss']:.4f} (KD={metrics['val_kd']:.4f}, "
                f"CE={metrics['val_ce']:.4f}) | Val PPL={metrics['val_ppl']:.2f}")

            tok_str = tokenizer._tokenizer.to_str()
            if metrics["val_loss"] < best_val_loss:
                best_val_loss = metrics["val_loss"]
                torch.save({
                    "step": step,
                    "model_state_dict": student.state_dict(),
                    "model_config": student_cfg.to_dict(),
                    "tokenizer": tok_str,
                    "teacher": args.teacher,
                    "best_val_loss": best_val_loss,
                }, out_dir / "checkpoint-best.pt")
                log(f"  ↳ ⭐ Saved new best student checkpoint (val_loss={best_val_loss:.4f})")

        # Periodic Saving
        if step % args.save_interval == 0 or step == args.max_steps:
            save_path = out_dir / f"checkpoint-{step}.pt"
            torch.save({
                "step": step,
                "model_state_dict": student.state_dict(),
                "model_config": student_cfg.to_dict(),
                "tokenizer": tokenizer._tokenizer.to_str(),
                "teacher": args.teacher,
            }, save_path)
            log(f"  ↳ 💾 Checkpoint saved: {save_path.name}")

    # Final Save
    torch.save({
        "step": args.max_steps,
        "model_state_dict": student.state_dict(),
        "model_config": student_cfg.to_dict(),
        "tokenizer": tokenizer._tokenizer.to_str(),
        "teacher": args.teacher,
    }, out_dir / "checkpoint-final.pt")
    log(f"\n🎉 Distillation complete! Checkpoints saved in '{out_dir}'.")


if __name__ == "__main__":
    main()
