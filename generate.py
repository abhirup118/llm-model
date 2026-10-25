#!/usr/bin/env python3
"""
Generate text from a trained GPT checkpoint.

Works with checkpoints from train.py (tokenizer + model config embedded) and with
create-llm checkpoints (pass --tokenizer and, if needed, --heads / --config).

Generated text goes to stdout; status messages go to stderr, so output can be piped.

Examples:
  python generate.py --checkpoint checkpoints/checkpoint-best.pt --prompt "ROMEO:" --max_tokens 200
  python generate.py --checkpoint checkpoints/checkpoint-final.pt --prompt "Once upon a time" \
      --temperature 0.8 --top_k 50 --top_p 0.95 --num_samples 3
  python generate.py --checkpoint checkpoints/checkpoint-final.pt --prompt "The" --temperature 0   # greedy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator, List, Optional

import torch
import torch.nn.functional as F

from model import GPTConfig, GPTModel, clean_state_dict, infer_config_from_state_dict


def eprint(*a, **k):
    print(*a, file=sys.stderr, flush=True, **k)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


# --------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------

def load_tokenizer(ckpt: dict, tokenizer_path: Optional[str]):
    from tokenizers import Tokenizer

    if tokenizer_path:
        return Tokenizer.from_file(tokenizer_path)
    if ckpt.get("tokenizer"):
        return Tokenizer.from_str(ckpt["tokenizer"])
    for candidate in (ckpt.get("tokenizer_path"), "tokenizer/tokenizer.json"):
        if candidate and Path(candidate).exists():
            return Tokenizer.from_file(candidate)
    raise FileNotFoundError("No tokenizer embedded in checkpoint; pass --tokenizer path/to/tokenizer.json")


def load_model(ckpt: dict, heads: Optional[int], device: torch.device) -> GPTModel:
    state = clean_state_dict(ckpt["model_state_dict"])
    if ckpt.get("model_config"):  # train.py checkpoint
        cfg = GPTConfig.from_dict(ckpt["model_config"])
    else:  # create-llm checkpoint: shapes from weights, heads from saved config / CLI
        saved_model_cfg = ((ckpt.get("config") or {}).get("model") or {})
        cfg = infer_config_from_state_dict(state, heads=heads or saved_model_cfg.get("heads"))
    cfg.dropout = 0.0
    model = GPTModel(cfg)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


# --------------------------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------------------------

def filter_logits(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """Apply top-k then nucleus (top-p) filtering to [B, V] logits."""
    if top_k and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        # Drop tokens once cumulative prob exceeds top_p, always keeping the most likely token.
        remove = cum_probs - sorted_logits.softmax(dim=-1) > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    return logits


def apply_repetition_penalty(logits: torch.Tensor, ids: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty == 1.0:
        return logits
    prev = torch.gather(logits, 1, ids)
    prev = torch.where(prev > 0, prev / penalty, prev * penalty)
    return logits.scatter(1, ids, prev)


@torch.no_grad()
def generate(
    model: GPTModel,
    input_ids: torch.Tensor,  # [B, T]
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float = 1.0,
    eos_id: Optional[int] = None,
) -> Iterator[torch.Tensor]:
    """Autoregressively yields the next token [B] at each step."""
    ctx = model.config.max_length
    finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)
    for _ in range(max_new_tokens):
        cond = input_ids[:, -ctx:]  # crop to the model's context window
        logits = model(cond)["logits"][:, -1, :].float()
        logits = apply_repetition_penalty(logits, cond, repetition_penalty)
        if temperature <= 0:  # greedy
            next_tok = logits.argmax(dim=-1)
        else:
            logits = filter_logits(logits / temperature, top_k, top_p)
            next_tok = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze(-1)
        if eos_id is not None:
            next_tok = torch.where(finished, torch.full_like(next_tok, eos_id), next_tok)
            finished |= next_tok == eos_id
        input_ids = torch.cat([input_ids, next_tok[:, None]], dim=1)
        yield next_tok
        if eos_id is not None and bool(finished.all()):
            break


def decode(tok, ids: List[int], eos_id: Optional[int]) -> str:
    if eos_id is not None and eos_id in ids:
        ids = ids[: ids.index(eos_id)]
    text = tok.decode(ids, skip_special_tokens=True)
    if "<|endoftext|>" in text:
        text = text.split("<|endoftext|>")[0]
    return text


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate text with a trained GPT checkpoint",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint")
    p.add_argument("--prompt", default="", help="Prompt text ('' = unconditional; '-' = read from stdin)")
    p.add_argument("--max_tokens", "--max_new_tokens", dest="max_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8, help="0 = greedy decoding")
    p.add_argument("--top_k", type=int, default=50, help="0 = disabled")
    p.add_argument("--top_p", type=float, default=0.95, help="1.0 = disabled")
    p.add_argument("--repetition_penalty", type=float, default=1.0, help="1.0 = disabled; try 1.1-1.3")
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--device", default="auto", help="auto | cuda | cuda:N | mps | cpu")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--tokenizer", default=None, help="Override tokenizer.json (required for create-llm ckpts)")
    p.add_argument("--heads", type=int, default=None, help="Attention heads, if absent from checkpoint")
    p.add_argument("--ignore_eos", action="store_true", help="Keep generating past </s>")
    p.add_argument("--no_stream", action="store_true", help="Print only when finished (default streams 1 sample)")
    p.add_argument("--no_prompt_echo", action="store_true", help="Print only the continuation")
    return p.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = pick_device(args.device)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tok = load_tokenizer(ckpt, args.tokenizer)
    model = load_model(ckpt, args.heads, device)
    eprint(f"✓ Loaded {args.checkpoint} (step {ckpt.get('step', '?')}, "
           f"{model.count_parameters():,} params, ctx {model.config.max_length}) on {device}")
    if tok.get_vocab_size() != model.config.vocab_size:
        eprint(f"⚠️  Tokenizer vocab {tok.get_vocab_size()} != model vocab {model.config.vocab_size}")

    eos_id = None if args.ignore_eos else tok.token_to_id("</s>")
    prompt = sys.stdin.read() if args.prompt == "-" else args.prompt

    prompt_ids = tok.encode(prompt, add_special_tokens=False).ids if prompt else []
    if not prompt_ids:
        # Unconditional: start from the document separator the model saw during training.
        start = tok.token_to_id("</s>")
        start = start if start is not None else (tok.token_to_id("<s>") or 0)
        prompt_ids = [start]
        prompt_has_text = False
    else:
        prompt_has_text = True
    if len(prompt_ids) > model.config.max_length:
        eprint(f"⚠️  Prompt has {len(prompt_ids)} tokens; only the last {model.config.max_length} are used as context")

    x = torch.tensor([prompt_ids] * args.num_samples, dtype=torch.long, device=device)
    stream = args.num_samples == 1 and not args.no_stream
    echo = prompt_has_text and not args.no_prompt_echo
    gen_kwargs = dict(max_new_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k,
                      top_p=args.top_p, repetition_penalty=args.repetition_penalty, eos_id=eos_id)

    out_tokens: List[List[int]] = [[] for _ in range(args.num_samples)]
    if stream:
        if echo:
            sys.stdout.write(prompt); sys.stdout.flush()
        printed = ""
        for nt in generate(model, x, **gen_kwargs):
            out_tokens[0].append(int(nt[0]))
            # Re-decode the whole continuation so multi-byte / merged tokens render correctly.
            raw_text = tok.decode(out_tokens[0], skip_special_tokens=True)
            if "<|endoftext|>" in raw_text:
                text = decode(tok, out_tokens[0], eos_id)
                if text.startswith(printed):
                    sys.stdout.write(text[len(printed):])
                break
            text = decode(tok, out_tokens[0], eos_id)
            if text.startswith(printed):
                sys.stdout.write(text[len(printed):]); sys.stdout.flush()
                printed = text
        sys.stdout.write("\n")
    else:
        for nt in generate(model, x, **gen_kwargs):
            for i, t in enumerate(nt.tolist()):
                out_tokens[i].append(t)
        for i, ids in enumerate(out_tokens):
            if args.num_samples > 1:
                print(f"===== sample {i + 1} =====")
            print((prompt if echo else "") + decode(tok, ids, eos_id))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
