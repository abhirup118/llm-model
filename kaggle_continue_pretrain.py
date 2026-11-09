#!/usr/bin/env python3
"""
Automated Resume Runner: Continue Pre-Training the 100M LLM on Kaggle.

Implements Training Plan V2:
  - Phase 1: 15,870 steps (~260M tokens seen, 1 full epoch of FineWeb-Edu)
  - Phase 2: 31,700 steps (~520M tokens seen, 2 epochs)

Usage on Kaggle:
  !python kaggle_continue_pretrain.py --phase 1
  !python kaggle_continue_pretrain.py --phase 2
"""

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path


def log(msg: str):
    print(f"\n{msg}", flush=True)


def find_resume_checkpoint() -> Path:
    """Finds the best checkpoint to resume from (checks local working dir and attached Kaggle datasets)."""
    candidates = [
        # Check active session output directory
        Path("/kaggle/working/checkpoints_100m/checkpoint-best.pt"),
        Path("/kaggle/working/checkpoints_100m/checkpoint-final.pt"),
        Path("checkpoints_100m/checkpoint-best.pt"),
    ]

    # Check attached Kaggle inputs (e.g., if user attached previous run as a dataset)
    input_matches = glob.glob("/kaggle/input/**/checkpoint-best.pt", recursive=True)
    candidates.extend(Path(p) for p in input_matches)

    for cand in candidates:
        if cand.exists() and cand.stat().st_size > 100 * 1024 * 1024:
            print(f"✓ Found checkpoint to resume from: {cand} ({cand.stat().st_size / 1024**2:.1f} MB)")
            return cand

    raise FileNotFoundError(
        "Could not find checkpoint-best.pt! Make sure previous checkpoints are in "
        "/kaggle/working/checkpoints_100m/ or attached under /kaggle/input/."
    )


def main():
    p = argparse.ArgumentParser(description="Continue Pre-Training 100M LLM")
    p.add_argument("--phase", type=int, default=1, choices=[1, 2, 3], help="Phase 1 (15870 steps) or Phase 2 (31700 steps)")
    p.add_argument("--target_steps", type=int, default=None, help="Override target steps")
    p.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps (default: 4 for effective batch 16)")
    args = p.parse_args()

    # Step targets based on Plan V2
    phase_targets = {
        1: 15_870,   # ~260M tokens (1 full epoch)
        2: 31_700,   # ~520M tokens (2 full epochs)
        3: 60_000,   # ~1B tokens
    }
    max_steps = args.target_steps if args.target_steps is not None else phase_targets[args.phase]

    log("=" * 65)
    log(f"🚀 CONTINUING 100M PRE-TRAINING | Phase {args.phase} (Target: {max_steps:,} steps)")
    log("=" * 65)

    is_kaggle = Path("/kaggle/working").exists()
    base_dir = Path("/kaggle/working") if is_kaggle else Path(".")

    corpus_path = base_dir / "data" / "large_corpus" / "fineweb_edu_subset.txt"
    if not corpus_path.exists():
        # Fallback search in input
        input_corpora = glob.glob("/kaggle/input/**/fineweb_edu_subset.txt", recursive=True)
        if input_corpora:
            corpus_path = Path(input_corpora[0])
            print(f"✓ Found corpus in input dataset: {corpus_path}")
        else:
            print("⚠️ Corpus not found locally. Re-streaming from Hugging Face FineWeb-Edu...")
            from kaggle_train_100m import prepare_dataset
            prepare_dataset(corpus_path, target_docs=250_000)

    # Locate checkpoint
    ckpt_path = find_resume_checkpoint()
    out_dir = base_dir / "checkpoints_100m"

    # Configure CUDA memory
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    cmd = [
        sys.executable, "train.py",
        "--data", str(corpus_path),
        "--resume", str(ckpt_path),
        "--template", "small",
        "--vocab_size", "32000",
        "--batch_size", "4",
        "--grad_accum", str(args.grad_accum),
        "--precision", "fp16",
        "--max_steps", str(max_steps),
        "--eval_interval", "500",
        "--save_interval", "2000",
        "--out_dir", str(out_dir),
    ]

    print(f"Running command:\n{' '.join(cmd)}\n{'-' * 65}")
    subprocess.check_call(cmd)

    # Post-training evaluation probe
    log("✨ Testing Generation from the Updated Checkpoint...")
    best_ckpt = out_dir / "checkpoint-best.pt"
    if not best_ckpt.exists():
        best_ckpt = out_dir / "checkpoint-final.pt"

    subprocess.run([
        sys.executable, "generate.py",
        "--checkpoint", str(best_ckpt),
        "--prompt", "The discovery of gravity began when",
        "--temperature", "0.75",
        "--repetition_penalty", "1.2",
        "--max_tokens", "150"
    ])

    # Zip output for quick export
    if is_kaggle:
        log("📦 Packaging checkpoint & tokenizer into zip...")
        zip_cmd = f"cd /kaggle/working && zip -r 100m_phase{args.phase}.zip checkpoints_100m/checkpoint-best.pt tokenizer/tokenizer.json"
        subprocess.run(zip_cmd, shell=True)
        print(f"✓ Created /kaggle/working/100m_phase{args.phase}.zip")
        print("\nTo download in your notebook, run:")
        print(f"  from IPython.display import FileLink; FileLink('100m_phase{args.phase}.zip')")


if __name__ == "__main__":
    main()
