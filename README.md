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

### 2. Pre-training 100M Model on Free Cloud GPUs (Kaggle)

Train the ~110M parameter `small` model (12 layers, 12 heads, dim 768, ctx 1024) from scratch on Kaggle Dual Nvidia T4 GPUs:

```bash
# Automated end-to-end runner (dataset download, 32k tokenizer, 4500 training steps, generation validation)
python kaggle_train_100m.py
```
*(Or upload the notebook [`notebooks/train_100m_kaggle.ipynb`](file:///Users/abhirup/Desktop/code/llm-model/notebooks/train_100m_kaggle.ipynb) directly to Kaggle and select **Save & Run All**).*

### 3. Supervised Fine-Tuning (SFT / Instruction Tuning)

Transform a pre-trained base completion model into an instruction-following conversational assistant:

```bash
# 1. Download and format Alpaca instruction pairs with prompt loss masking
python prepare_sft_data.py

# 2. Run SFT with prompt masking and conservative learning rate
python sft.py \
    --base_checkpoint checkpoints_medium/checkpoint-best.pt \
    --data_dir data/sft \
    --out_dir checkpoints_sft \
    --batch_size 8 \
    --grad_accum 2 \
    --lr 2e-5 \
    --max_steps 800
```

### 4. Knowledge Distillation (Teacher-Student)

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

### 5. Generating Text

```bash
# CLI text generation
python generate.py \
    --checkpoint checkpoints_sft/checkpoint-best.pt \
    --prompt "User: What is the opposite of hot?\nAssistant:" \
    --temperature 0.6 \
    --max_tokens 100
```

### 6. Interactive Shell

```bash
# Live interactive streaming playground (automatically detects SFT chat mode)
python interactive.py
```

## File Structure

```
.
├── model.py                     # GPT-style Transformer architecture with SDPA
├── train.py                     # Pre-training loop, data pipeline, and checkpointing
├── sft.py                       # Supervised Fine-Tuning with prompt loss masking
├── prepare_sft_data.py          # Instruction dataset downloader and formatter
├── kaggle_train_100m.py         # Automated Kaggle pre-training runner for 100M model
├── notebooks/                   # Jupyter notebooks (train_100m_kaggle.ipynb)
├── distill.py                   # Knowledge Distillation (Hugging Face teacher -> student)
├── generate.py                  # Text generation with top-k / top-p sampling
├── interactive.py               # Interactive streaming terminal playground
├── requirements.txt             # Python dependencies
├── data/raw/                    # Raw text corpora
└── tokenizer/                   # Byte-Level BPE tokenizer
```
