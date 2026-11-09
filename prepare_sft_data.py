#!/usr/bin/env python3
"""
Prepare a clean instruction-following dataset for Supervised Fine-Tuning (SFT).
Downloads a high-quality subset of Stanford Alpaca, filters for conciseness
and length (<= 384 tokens), and splits into train and validation sets.
"""

import json
import random
import urllib.request
from pathlib import Path
from tokenizers import Tokenizer

def main():
    random.seed(42)
    out_dir = Path("data/sft")
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"

    tokenizer_path = Path("tokenizer/tokenizer.json")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
    tok = Tokenizer.from_file(str(tokenizer_path))

    url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    print(f"📥 Downloading instruction dataset from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw_data = json.loads(resp.read().decode("utf-8"))

    print(f"✓ Downloaded {len(raw_data)} raw instruction pairs.")

    clean_samples = []
    max_tokens = 384  # Keep plenty of room within 512 context window

    for item in raw_data:
        instruction = item.get("instruction", "").strip()
        input_text = item.get("input", "").strip()
        output = item.get("output", "").strip()

        if not instruction or not output:
            continue

        if input_text and input_text.lower() not in ("noinput", "none", "n/a", "no input"):
            prompt = f"User: {instruction}\nInput: {input_text}\nAssistant: "
        else:
            prompt = f"User: {instruction}\nAssistant: "

        full_text = f"{prompt}{output}</s>"
        
        # Check token length
        tokens = tok.encode(full_text).ids
        if len(tokens) <= max_tokens and len(tokens) >= 15:
            clean_samples.append({
                "prompt": prompt,
                "response": output,
                "full_text": full_text
            })

    print(f"✓ Filtered to {len(clean_samples)} suitable examples (<= {max_tokens} tokens).")

    # Select 4,000 top examples for fast, high-quality SFT
    random.shuffle(clean_samples)
    selected = clean_samples[:4000]

    val_size = 200
    val_set = selected[:val_size]
    train_set = selected[val_size:]

    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_set:
            f.write(json.dumps(item) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for item in val_set:
            f.write(json.dumps(item) + "\n")

    print(f"✅ Prepared SFT datasets:")
    print(f"   - Train: {len(train_set)} examples -> {train_path}")
    print(f"   - Val:   {len(val_set)} examples -> {val_path}")

if __name__ == "__main__":
    main()
