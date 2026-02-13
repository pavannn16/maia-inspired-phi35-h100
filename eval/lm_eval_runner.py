from __future__ import annotations

import argparse
import json
import os
import sys
import subprocess
from typing import Any, Dict, List

import yaml


# Allow running as `python eval/lm_eval_runner.py` (adds repo root to sys.path).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out_json", default="results/lm_eval.json")
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    tasks: List[str] = cfg["metrics"]["quality"]["tasks"]

    # This uses lm-eval-harness CLI. You can swap to Python API later.
    cmd = [
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={args.model}",
        "--tasks",
        ",".join(tasks),
        "--device",
        "cuda",
        "--batch_size",
        "auto",
        "--output_path",
        args.out_json,
    ]

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
