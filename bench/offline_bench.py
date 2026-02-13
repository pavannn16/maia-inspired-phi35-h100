from __future__ import annotations

import argparse
import os
import time
import uuid
from typing import Any, Dict

import pandas as pd
import yaml

from runtime.common import append_jsonl


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--config_id", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    conf = cfg["configs"][args.config_id]
    wl = cfg["workloads"]["offline_batch_decode"]

    raw_dir = os.path.join(args.out_dir, "raw")
    summary_dir = os.path.join(args.out_dir, "summary")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    run_id = f"{args.config_id}-{uuid.uuid4().hex[:10]}"
    raw_path = os.path.join(raw_dir, f"{run_id}.jsonl")

    rows = []
    for prompt_len in wl["prompt_lengths"]:
        for output_len in wl["output_lengths"]:
            for r in range(args.repeats):
                sub_id = f"{run_id}-p{prompt_len}-o{output_len}-r{r}"

                if conf.get("runtime") == "torch":
                    cmd = (
                        f"python -m runtime.torch_runner --model {args.model} "
                        f"--prompt_len {prompt_len} --output_len {output_len} "
                        f"--dtype bf16 --out_jsonl {raw_path} --run_id {sub_id}"
                    )
                else:
                    raise RuntimeError(f"Unsupported runtime for now: {conf.get('runtime')}")

                rc = os.system(cmd)
                if rc != 0:
                    raise RuntimeError(f"Command failed ({rc}): {cmd}")

    # Parse summary from JSONL.
    # (Kept simple: production version should stream-parse and validate schema.)
    import json

    with open(raw_path, "rb") as f:
        for line in f:
            rec = json.loads(line)
            res = rec["result"]
            rows.append(
                {
                    "subrun_id": rec["run_id"],
                    "config_id": args.config_id,
                    "model": res["model"],
                    "prompt_len": res["prompt_len"],
                    "output_len": res["output_len"],
                    "ttft_ms": res["ttft_ms"],
                    "tpot_ms": res["tpot_ms"],
                    "throughput_total_tok_s": res["throughput_total_tok_s"],
                    "throughput_output_tok_s": res["throughput_output_tok_s"],
                    "peak_gpu_mem_mb": res["peak_gpu_mem_mb"],
                }
            )

    df = pd.DataFrame(rows)
    summary_path = os.path.join(summary_dir, f"{run_id}.csv")
    df.to_csv(summary_path, index=False)

    print(f"Wrote: {raw_path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
