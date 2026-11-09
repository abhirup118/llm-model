#!/usr/bin/env python3
"""
Automated End-to-End Runner: Pre-train a 100M Parameter LLM from Scratch on Kaggle.

This script manages the complete pipeline:
  1. Detects GPU hardware and environment (Kaggle / Colab / Local).
  2. Downloads ~150M tokens of clean educational text from Hugging Face FineWeb-Edu.
  3. Trains a 32,000-vocabulary Byte-Level BPE tokenizer.
  4. Launches 100M model training (12 layers, 12 heads, dim 768, ctx 1024) for 4,500 steps.
  5. Performs post-training generation validation on the best checkpoint.

Usage on Kaggle:
  !python kaggle_train_100m.py
"""

import os
import subprocess
import sys
from pathlib import Path

import torch


def log(msg: str):
    print(f"\n{msg}", flush=True)


def check_gpu():
    log("=" * 65)
    log("🚀 100M MODEL PRE-TRAINING SETUP")
    log("=" * 65)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        count = torch.cuda.device_count()
        print(f"✓ GPU Detected: {gpu_name} (Count: {count}, VRAM: {vram_gb:.1f} GB per GPU)")
        if vram_gb < 14:
            print("⚠️ Warning: Less than 14GB VRAM detected. Consider reducing batch_size if OOM occurs.")
    else:
        print("⚠️ No CUDA GPU detected! Training a 100M model on CPU will be extremely slow.")
        print("   If on Kaggle, ensure 'Accelerator: GPU T4 x 2' is selected in the right sidebar.")


def prepare_dataset(corpus_path: Path, target_docs: int = 250_000) -> Path:
    if corpus_path.exists() and corpus_path.stat().st_size > 10 * 1024 * 1024:
        size_mb = corpus_path.stat().st_size / (1024 * 1024)
        print(f"✓ Dataset already exists at {corpus_path} ({size_mb:.2f} MB). Skipping download.")
        return corpus_path

    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"📥 Step 1/4: Streaming {target_docs:,} educational articles from Hugging Face FineWeb-Edu...")

    try:
        from datasets import load_dataset
        from tqdm.auto import tqdm
    except ImportError:
        print("Installing datasets and tqdm...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets", "tqdm"])
        from datasets import load_dataset
        from tqdm.auto import tqdm

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True
    )

    with open(corpus_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(tqdm(dataset, total=target_docs, desc="Downloading text")):
            if i >= target_docs:
                break
            text = item.get("text", "").strip()
            if len(text) > 200:
                f.write(text + "\n\n")

    size_mb = corpus_path.stat().st_size / (1024 * 1024)
    print(f"✓ Corpus successfully saved: {corpus_path} ({size_mb:.2f} MB)")
    return corpus_path


def train_tokenizer(corpus_path: Path, tok_dir: Path, vocab_size: int = 32000) -> Path:
    tok_path = tok_dir / "tokenizer.json"
    if tok_path.exists():
        print(f"✓ Tokenizer already exists at {tok_path}. Skipping training.")
        return tok_path

    tok_dir.mkdir(parents=True, exist_ok=True)
    log(f"⚙️ Step 2/4: Training Byte-Level BPE Tokenizer (Vocab: {vocab_size:,})...")

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
        min_frequency=2,
        show_progress=True,
    )

    tok.train(files=[str(corpus_path)], trainer=trainer)
    tok.save(str(tok_path))
    print(f"✓ Tokenizer trained and saved to {tok_path} (Vocab size: {tok.get_vocab_size()})")
    return tok_path


def run_training(corpus_path: Path, tok_path: Path, out_dir: Path, max_steps: int = 4500):
    log(f"🏋️ Step 3/4: Starting 100M LLM Pre-Training ({max_steps:,} steps)...")
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "train.py",
        "--data", str(corpus_path),
        "--tokenizer", str(tok_path),
        "--template", "small",
        "--vocab_size", "32000",
        "--batch_size", "8",
        "--grad_accum", "2",
        "--max_steps", str(max_steps),
        "--eval_interval", "500",
        "--save_interval", "1500",
        "--out_dir", str(out_dir),
    ]

    print("Running command:")
    print(" ".join(cmd))
    print("-" * 65)
    subprocess.check_call(cmd)
    print("-" * 65)
    print(f"✓ Training finished! Checkpoints saved in {out_dir}")


def validate_generation(out_dir: Path):
    best_ckpt = out_dir / "checkpoint-best.pt"
    if not best_ckpt.exists():
        best_ckpt = out_dir / "checkpoint-final.pt"

    if not best_ckpt.exists():
        print("⚠️ No checkpoint found to test generation.")
        return

    log("✨ Step 4/4: Testing Post-Training Generation...")
    cmd = [
        sys.executable, "generate.py",
        "--checkpoint", str(best_ckpt),
        "--prompt", "The discovery of gravity began when",
        "--temperature", "0.7",
        "--max_tokens", "100"
    ]
    subprocess.run(cmd)


def main():
    check_gpu()

    # Determine working directory (detects if running on Kaggle)
    is_kaggle = Path("/kaggle/working").exists()
    base_dir = Path("/kaggle/working") if is_kaggle else Path(".")

    corpus_path = base_dir / "data" / "large_corpus" / "fineweb_edu_subset.txt"
    tok_dir = base_dir / "tokenizer"
    out_dir = base_dir / "checkpoints_100m"

    # Execute Pipeline
    prepare_dataset(corpus_path, target_docs=250_000)
    tok_path = train_tokenizer(corpus_path, tok_dir, vocab_size=32000)
    run_training(corpus_path, tok_path, out_dir, max_steps=4500)
    validate_generation(out_dir)

    log("=" * 65)
    log("🎉 ALL STEPS COMPLETE!")
    if is_kaggle:
        print("Download your checkpoint from the Kaggle Output tab:")
        print("   /kaggle/working/checkpoints_100m/checkpoint-best.pt")
    log("=" * 65)


if __name__ == "__main__":
    main()

