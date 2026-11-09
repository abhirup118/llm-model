#!/usr/bin/env python3
"""
Interactive terminal playground for testing your trained LLM.
Type prompts directly and see real-time streaming text completions.
"""

import argparse
import sys
from pathlib import Path
import torch
from generate import load_model, load_tokenizer, generate, decode, pick_device

def main():
    parser = argparse.ArgumentParser(description="Interactive Playground for GPT")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint. Defaults to checkpoints_sft/checkpoint-best.pt if available.")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        for candidate in ("checkpoints_sft/checkpoint-best.pt", "checkpoints_sft/checkpoint-final.pt",
                          "checkpoints_medium/checkpoint-best.pt", "checkpoints_medium/checkpoint-final.pt"):
            if Path(candidate).exists():
                checkpoint_path = candidate
                break

    if not checkpoint_path or not Path(checkpoint_path).exists():
        print("❌ No checkpoint found. Please specify --checkpoint path/to/model.pt")
        return

    device = pick_device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n✨ Loading {checkpoint_path} on {device}...")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    tok = load_tokenizer(ckpt, None)
    model = load_model(ckpt, None, device)
    eos_id = tok.token_to_id("</s>")
    is_sft = "sft" in checkpoint_path or ckpt.get("stage") == "sft"

    print(f"✓ Model ready! (Parameters: {model.count_parameters():,}, Mode: {'SFT Chat' if is_sft else 'Base Completion'})")
    print("=" * 60)
    print("🤖 LLM Interactive Playground")
    print("Type your message and press Enter. (Type 'exit' or 'quit' to stop)")
    print("Tip: Adjust temperature on the fly by typing: /temp 0.6")
    print("=" * 60)

    temperature = 0.6 if is_sft else 0.75
    top_k = 50
    top_p = 0.95
    repetition_penalty = 1.15
    max_tokens = 150

    while True:
        try:
            user_input = input(f"\n[temp={temperature}] You > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break
            if user_input.startswith("/temp "):
                try:
                    temperature = float(user_input.split()[1])
                    print(f"✓ Temperature set to {temperature}")
                    continue
                except Exception:
                    print("Usage: /temp 0.6")
                    continue

            # In SFT mode, format input automatically with the chat template
            if is_sft and not user_input.startswith("User:"):
                formatted_prompt = f"User: {user_input}\nAssistant: "
            else:
                formatted_prompt = user_input

            prompt_ids = tok.encode(formatted_prompt, add_special_tokens=False).ids
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

            print("\nAssistant > ", end="", flush=True)

            out_tokens = []
            printed = ""
            for nt in generate(model, x, **gen_kwargs):
                token_id = int(nt[0])
                if token_id == eos_id:
                    break
                out_tokens.append(token_id)
                text = decode(tok, out_tokens, eos_id)
                if text.startswith(printed):
                    sys.stdout.write(text[len(printed):])
                    sys.stdout.flush()
                    printed = text
            print()

        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
