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

### 2. Knowledge Distillation (Teacher-Student)

Distill knowledge from a pre-trained Hugging Face teacher (e.g. `gpt2` with ~124M parameters) into a compact student model (e.g. `nano` ~7M or `tiny` ~12M parameters):

```bash
# Distill using GPT-2 as teacher onto TinyStories
python distill.py \
    --teacher gpt2 \
    --student_template nano \
    --data data/raw/tinystories.txt \
    --temperature 2.0 \
    --alpha 0.5 \
    --max_steps 2000 \
    --batch_size 16 \
    --out_dir checkpoints_distill
```

### 3. Generating Text

```bash
# CLI text generation
python generate.py \
    --checkpoint checkpoints_medium/checkpoint-best.pt \
    --prompt "Once upon a time, there was a little dog named Toby." \
    --temperature 0.75 \
    --max_tokens 150
```

### 4. Interactive Shell

```bash
# Live interactive streaming playground
python interactive.py
```

## File Structure

```
.
├── model.py           # GPT-style Transformer architecture with SDPA
├── train.py           # Training loop, data pipeline, and checkpointing
├── distill.py         # Knowledge Distillation (Hugging Face teacher -> student)
├── generate.py        # Text generation with top-k / top-p sampling
├── interactive.py     # Interactive streaming terminal playground
├── requirements.txt   # Python dependencies (torch, tokenizers, transformers)
├── data/raw/          # Raw training datasets (.txt)
└── tokenizer/         # Trained Byte-Level BPE tokenizer
```
