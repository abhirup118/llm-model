# Standalone PyTorch GPT Language Model

A pure Python, decoder-only Transformer language model built from scratch with PyTorch, featuring custom BPE tokenization, mixed precision training, and autoregressive text generation.

## Features
- **Pure Python**: Zero JavaScript or Node.js dependencies.
- **Architectures**: Built-in size templates (`nano` ~1M, `tiny` ~3.8M, `medium` ~29.6M, `small` ~100M).
- **Fast Training**: Hardware-accelerated with PyTorch Scaled Dot-Product Attention (SDPA / FlashAttention) on CUDA GPUs and Apple Silicon (`mps`).
- **Loss Fix**: True next-token label shifting (`logits[:, :-1]` vs `labels[:, 1:]`).
- **Sampling**: Temperature scaling, top-k filtering, nucleus (top-p) sampling, and repetition penalty.
- **Self-Contained Checkpoints**: Model weights, architecture config, and the BPE tokenizer are all bundled inside each `.pt` file.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

### 1. Training from Scratch

Train a model on any text file or directory of `.txt` files:

```bash
# Train the ~29.6M Medium model
python train.py \
    --data data/raw \
    --template medium \
    --vocab_size 8000 \
    --max_steps 4000 \
    --batch_size 16 \
    --out_dir checkpoints_medium
```

### 2. Generating Text

```bash
# CLI text generation
python generate.py \
    --checkpoint checkpoints_medium/checkpoint-best.pt \
    --prompt "Once upon a time, there was a little dog named Toby." \
    --temperature 0.75 \
    --max_tokens 150
```

### 3. Interactive Shell

```bash
# Live interactive streaming playground
python interactive.py
```

## File Structure

```
.
├── model.py           # GPT-style Transformer architecture with SDPA
├── train.py           # Training loop, data pipeline, and checkpointing
├── generate.py        # Text generation with top-k / top-p sampling
├── interactive.py     # Interactive streaming terminal playground
├── requirements.txt   # Python dependencies (torch, tokenizers)
├── data/raw/          # Raw training datasets (.txt)
└── tokenizer/         # Trained Byte-Level BPE tokenizer
```
