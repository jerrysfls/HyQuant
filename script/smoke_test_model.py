#!/usr/bin/env python3

"""Simple smoke test for patched Llama/Qwen attention."""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


DEFAULT_PROMPTS = {
    "llama": "Briefly explain why key-value cache helps autoregressive decoding.",
    "qwen": "Briefly explain what happens during the prefill and decode stages of attention.",
}


def build_prompt(tokenizer, prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Smoke test for patched Llama/Qwen models")
    parser.add_argument("--family", required=True, choices=["llama", "qwen"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--mode",
        default="hybrid_full",
        choices=[
            "dense",
            "full_attention",
            "sage_w",
            "sage_w0",
            "vquant_vert",
            "sparse_only",
            "minference_sparse",
            "d_k4v4",
            "d_k4v4_vert",
            "d_k8v8",
            "d_k4v2",
            "d_k4v2_vert",
            "d_k2v4",
            "d_k2v4_vert",
            "hybrid_full",
            "hybrid_k4v2",
            "hybrid_k2v4",
        ],
    )
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--top-ratio", type=float, default=0.05)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--gpu", default="0", help="Value for CUDA_VISIBLE_DEVICES when unset")
    parser.add_argument("--do-sample", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    return parser.parse_args()


def maybe_patch_model(model, family: str, mode: str, window_size: int, top_ratio: float):
    from src.patch.patch_llama import patch_llama3_attention_forward_model
    from src.patch.patch_qwen import patch_qwen3_attention_forward_model

    if family == "llama":
        return patch_llama3_attention_forward_model(
            model,
            mode=mode,
            window_size=window_size,
            top_ratio=top_ratio,
        )
    return patch_qwen3_attention_forward_model(
        model,
        mode=mode,
        window_size=window_size,
        top_ratio=top_ratio,
    )


def maybe_create_cache(model, mode: str, window_size: int, top_ratio: float):
    from src.ops import create_cache_for_mode
    from src.patch._common import is_d_stage_mode

    if not is_d_stage_mode(mode):
        return None

    head_dim = model.config.hidden_size // model.config.num_attention_heads
    return create_cache_for_mode(
        mode,
        num_layers=model.config.num_hidden_layers,
        head_dim=head_dim,
        window_size=window_size,
        top_ratio=top_ratio,
    )


def main() -> None:
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke test.")

    prompt = args.prompt or DEFAULT_PROMPTS[args.family]

    print(f"Loading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model = maybe_patch_model(model, args.family, args.mode, args.window_size, args.top_ratio)
    model.eval()

    device = next(model.parameters()).device
    input_text = build_prompt(tokenizer, prompt)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.do_sample:
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = args.top_p

    cache = maybe_create_cache(model, args.mode, args.window_size, args.top_ratio)
    if cache is not None:
        gen_kwargs["past_key_values"] = cache

    print("Running generation...")
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    completion = tokenizer.decode(
        outputs[0, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()

    print("")
    print("=== Smoke Test Result ===")
    print(f"family:         {args.family}")
    print(f"mode:           {args.mode}")
    print(f"model_path:     {args.model_path}")
    print(f"window_size:    {args.window_size}")
    print(f"top_ratio:      {args.top_ratio}")
    print(f"used_d_stage:   {cache is not None}")
    print(f"prompt:         {prompt}")
    print("completion:")
    print(completion if completion else "<empty>")


if __name__ == "__main__":
    main()
