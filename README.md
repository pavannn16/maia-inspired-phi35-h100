# Maia-Inspired Mixed Precision for Phi-3.5 on H100 (Colab)

This repo studies **Maia-inspired mixed precision** using an **H100 proxy**: **W4A8** (4-bit weights + 8-bit activations) plus a **custom shared-scale metadata path**.

Key honesty statement: **H100 ≠ Maia-200**. H100 has strong **FP8 Tensor Core** support; **FP4 GEMM acceleration is not the same story on H100** as it is on newer architectures. This project frames W4A8 + shared-scale as an **emulation/proxy** for the Maia-style idea.

## What you get (eventually)
- Fixed experiment matrix **C0–C5** (configs + workloads + metrics)
- Reproducible artifact logging: raw JSONL + parsed CSV
- Baselines: BF16, FP8 activations, W4A16, W4A8
- Novelty: W4A8 + custom shared-scale format and CUDA dequant path

## Quick start (C0 BF16 offline benchmark)
1. Install deps:
   - `pip install -r requirements.txt`
2. Run offline sweep (prompt lengths {128,512,2048}, output lengths {64,256}):
   - `python bench/offline_bench.py --config configs/experiment_matrix.yaml --model microsoft/Phi-3.5-mini-instruct --config_id C0 --repeats 1`

Artifacts:
- `results/raw/*.jsonl` (per-request timing + env)
- `results/summary/*.csv` (one row per run)

## Repro contract
- **Do not edit** the experiment matrix after publishing results.
- Lock prompt formatting/chat template and decoding params.

See `ENVIRONMENT.md` and `configs/experiment_matrix.yaml`.
