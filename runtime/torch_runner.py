from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from runtime.common import append_jsonl, get_env_snapshot, maybe_cuda_sync, now_ms


@dataclass
class Timings:
    prefill_ms: float
    first_token_ms: float
    decode_token_ms: List[float]

    @property
    def ttft_ms(self) -> float:
        return self.prefill_ms + self.first_token_ms

    @property
    def tpot_ms(self) -> float:
        if not self.decode_token_ms:
            return float("nan")
        return sum(self.decode_token_ms) / len(self.decode_token_ms)


def greedy_decode_timed(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> Tuple[torch.Tensor, Timings]:
    device = input_ids.device

    maybe_cuda_sync()
    t0 = now_ms()
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    maybe_cuda_sync()
    t1 = now_ms()

    past = out.past_key_values
    next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

    maybe_cuda_sync()
    t2_start = now_ms()
    with torch.no_grad():
        out2 = model(input_ids=next_token, past_key_values=past, use_cache=True)
    maybe_cuda_sync()
    t2_end = now_ms()

    generated = [next_token]
    past = out2.past_key_values
    next_token = torch.argmax(out2.logits[:, -1, :], dim=-1, keepdim=True)

    decode_times: List[float] = []
    for _ in range(max_new_tokens - 1):
        maybe_cuda_sync()
        td0 = now_ms()
        with torch.no_grad():
            outn = model(input_ids=next_token, past_key_values=past, use_cache=True)
        maybe_cuda_sync()
        td1 = now_ms()
        decode_times.append(td1 - td0)

        past = outn.past_key_values
        next_token = torch.argmax(outn.logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(next_token)

    gen_ids = torch.cat(generated, dim=1) if generated else torch.empty((1, 0), device=device, dtype=input_ids.dtype)
    timings = Timings(prefill_ms=t1 - t0, first_token_ms=t2_end - t2_start, decode_token_ms=decode_times)
    return gen_ids, timings


def peak_gpu_mem_mb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        return None


def run_single(
    model_id: str,
    prompt_len: int,
    out_len: int,
    dtype: str,
    device: str,
) -> Dict[str, Any]:
    torch.manual_seed(0)

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    torch_dtype = dtype_map[dtype]

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, device_map=None)
    model.eval()
    model.to(device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    text = "hello " * (prompt_len // 2)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=prompt_len)
    input_ids = enc["input_ids"].to(device)

    gen_ids, timings = greedy_decode_timed(model, input_ids, max_new_tokens=out_len)

    total_tokens = int(input_ids.shape[1] + gen_ids.shape[1])
    output_tokens = int(gen_ids.shape[1])

    total_ms = timings.prefill_ms + timings.first_token_ms + sum(timings.decode_token_ms)
    throughput_total = (total_tokens / (total_ms / 1000.0)) if total_ms > 0 else float("nan")
    throughput_out = (output_tokens / ((timings.first_token_ms + sum(timings.decode_token_ms)) / 1000.0)) if output_tokens > 0 else float("nan")

    return {
        "model": model_id,
        "dtype": dtype,
        "device": device,
        "prompt_len": prompt_len,
        "output_len": out_len,
        "prefill_ms": timings.prefill_ms,
        "first_token_ms": timings.first_token_ms,
        "ttft_ms": timings.ttft_ms,
        "tpot_ms": timings.tpot_ms,
        "total_ms": total_ms,
        "throughput_total_tok_s": throughput_total,
        "throughput_output_tok_s": throughput_out,
        "peak_gpu_mem_mb": peak_gpu_mem_mb(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt_len", type=int, required=True)
    ap.add_argument("--output_len", type=int, required=True)
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--run_id", required=True)
    args = ap.parse_args()

    record: Dict[str, Any] = {
        "run_id": args.run_id,
        "env": get_env_snapshot(),
        "result": run_single(
            model_id=args.model,
            prompt_len=args.prompt_len,
            out_len=args.output_len,
            dtype=args.dtype,
            device=args.device,
        ),
    }
    append_jsonl(args.out_jsonl, record)


if __name__ == "__main__":
    main()
