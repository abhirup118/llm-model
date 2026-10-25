#!/usr/bin/env python3
"""
Interactive terminal playground for testing your trained LLM.
Type prompts directly and see real-time streaming text completions.
"""

import sys
from pathlib import Path
import torch
from generate import load_model, load_tokenizer, generate, decode, pick_device

def main():
    checkpoint_path = "checkpoints_medium/checkpoint-best.pt"
    if not Path(checkpoint_path).exists():
        checkpoint_path = "checkpoints_medium/checkpoint-final.pt"

    device = pick_device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n✨ Loading {checkpoint_path} on {device}...")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    tok = load_tokenizer(ckpt, None)
    model = load_model(ckpt, None, device)
    eos_id = tok.token_to_id("</s>")

    print(f"✓ Model ready! (Parameters: {model.count_parameters():,})")
    print("=" * 60)
    print("🤖 LLM Interactive Playground")
    print("Type your prompt and press Enter. (Type 'exit' or 'quit' to stop)")
    print("Tip: Adjust temperature on the fly by typing: /temp 0.7")
    print("=" * 60)

    temperature = 0.75
    top_k = 50
    top_p = 0.95
    repetition_penalty = 1.15
    max_tokens = 200

    while True:
        try:
            prompt = input(f"\n[temp={temperature}] Prompt > ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break
            if prompt.startswith("/temp "):
                try:
                    temperature = float(prompt.split()[1])
                    print(f"✓ Temperature set to {temperature}")
                    continue
                except Exception:
                    print("Usage: /temp 0.8")
                    continue

            prompt_ids = tok.encode(prompt, add_special_tokens=False).ids
            if not prompt_ids:
                prompt_ids = [tok.token_to_id("<s>") or 0]

            x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            gen_kwargs = dict(
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_id=eos_id,
            )

            print("\n" + "-" * 40)
            sys.stdout.write(prompt)
            sys.stdout.flush()

            out_tokens = []
            printed = ""
            for nt in generate(model, x, **gen_kwargs):
                out_tokens.append(int(nt[0]))
                raw_text = tok.decode(out_tokens, skip_special_tokens=True)
                if "<|endoftext|>" in raw_text:
                    break
                text = decode(tok, out_tokens, eos_id)
                if text.startswith(printed):
                    sys.stdout.write(text[len(printed):])
                    sys.stdout.flush()
                    printed = text
            print("\n" + "-" * 40)

        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
