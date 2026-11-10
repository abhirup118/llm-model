#!/usr/bin/env python3
"""
Interactive Terminal Chat for Hugging Face Models with LoRA Adapters (SFT / DPO).

Supports running on Apple Silicon GPU (MPS) or CPU with real-time token streaming.

Usage:
  python chat_hf.py --adapter adapters/qwen_dpo
  python chat_hf.py --adapter adapters/qwen_sft
  python chat_hf.py --model Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Interactive Chat with Fine-Tuned / DPO Models")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B", help="Base model identifier on Hugging Face")
    parser.add_argument("--adapter", type=str, default=None, help="Path to local LoRA adapter folder (SFT or DPO)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--max_tokens", type=int, default=256, help="Maximum generated tokens per response")
    args = parser.parse_args()

    # Search for default adapter if not explicitly passed
    adapter_path = args.adapter
    if not adapter_path:
        for cand in ("adapters/qwen_dpo", "adapters/qwen_sft", "adapters/qwen25_05b_dpo_lora", "adapters/qwen25_05b_oasst_lora"):
            if Path(cand).exists():
                adapter_path = cand
                break

    device = pick_device()
    print("=" * 65)
    print(f"🚀 Loading Base Model: {args.model} on {device}...")
    if adapter_path:
        print(f"✓ Attaching LoRA Adapter from: {adapter_path}")
    print("=" * 65)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device.type in ("mps", "cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map={"": device} if device.type != "cpu" else None,
    )

    if adapter_path and Path(adapter_path).exists():
        model = PeftModel.from_pretrained(model, adapter_path)
        print("✓ Successfully loaded LoRA weights and merged for fast inference!")
        model = model.merge_and_unload()
    elif adapter_path:
        print(f"⚠️ Warning: Adapter path '{adapter_path}' not found. Running base model directly.")

    model.eval()
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    print("\n" + "=" * 65)
    print("🤖 Interactive Aligned Assistant Ready!")
    print("Commands:")
    print("  exit / quit : Close chat")
    print("  clear       : Reset conversation history")
    print("  /temp 0.5   : Adjust sampling temperature")
    print("=" * 65 + "\n")

    history = [
        {"role": "system", "content": "You are a helpful, respectful, and honest AI assistant."}
    ]
    temperature = args.temperature

    while True:
        try:
            user_input = input("\nYou > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting chat. Goodbye!")
                break

            if user_input.lower() == "clear":
                history = [{"role": "system", "content": "You are a helpful, respectful, and honest AI assistant."}]
                print("🧹 Conversation memory cleared.")
                continue

            if user_input.startswith("/temp"):
                parts = user_input.split()
                if len(parts) > 1:
                    try:
                        temperature = float(parts[1])
                        print(f"✓ Temperature set to {temperature}")
                    except ValueError:
                        print("Invalid temperature number.")
                continue

            # Add turn to conversation history
            history.append({"role": "user", "content": user_input})

            # Format input using model's official ChatML template
            formatted_prompt = tokenizer.apply_chat_template(
                history,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(device)

            print("\nAssistant > ", end="", flush=True)

            with torch.inference_mode():
                output_ids = model.generate(
                    formatted_prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    do_sample=temperature > 0,
                    top_p=args.top_p if temperature > 0 else 1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    streamer=streamer,
                )

            # Extract assistant reply to preserve multi-turn context
            generated_tokens = output_ids[0][formatted_prompt.shape[1]:]
            assistant_reply = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            history.append({"role": "assistant", "content": assistant_reply})

        except KeyboardInterrupt:
            print("\nExiting chat. Goodbye!")
            break


if __name__ == "__main__":
    main()
