# Project Definition — Maia-Inspired Mixed Precision for Phi-3.5 on H100

**Status:** Code complete. Awaiting Colab H100 run.
**Target audience:** Microsoft Applied Science / Research internship portfolio.
**Repo:** https://github.com/pavannn16/maia-inspired-phi35-h100

---

## 1. What This Project Is

A **reproducible systems + ML research study** that asks a single, sharp question:

> How many mantissa bits do per-group quantization scales actually need before model quality degrades — and what does the answer mean for LLM inference efficiency?

The experiment runs **Phi-3.5-mini-instruct (3.8B)** on a **Colab H100 80 GB** across six quantization configurations (C0–C5), measuring throughput, latency, memory, energy consumption, and downstream accuracy under each.

The inspiration is **Microsoft Maia 100** — a custom AI accelerator that uses hardware-native mixed-precision scale tensors (FP8 E4M3/E5M2).  This project is a **software proxy** of that idea: instead of requiring custom silicon, it implements the same scale-compression concept in pure PyTorch + CUDA on a standard H100.

---

## 2. The Novel Contribution — Scale Metadata Quantization (SMQ)

Standard W4 per-group quantization (used by AWQ, GPTQ, bitsandbytes NF4) stores one FP16 scale per 128 weights.  For a single Phi-3.5 MLP layer (`out=3072, in=3072`), that is **144 KB of scale metadata per layer**, stored in FP16 (~14 mantissa bits of precision).

**SMQ replaces each scale with an E5Mx mini-float:**

| Format | Bits per scale | Mantissa bits | Storage savings vs FP16 |
|--------|---------------|---------------|------------------------|
| FP16 (baseline) | 16 | 10 | — |
| E5M5 (C4, proposed) | 11 | 5 | −31% |
| E5M3 | 9 | 3 | −44% |
| E5M0 (binary) | 6 | 0 | −63% |
| FP32 exact (C3, control) | 32 | 23 | +100% (reference only) |

The research claim: **E5M5 scales (scale_mbits=5) are nearly lossless** — relative quantization error increases by < 0.5% vs FP32 exact — while still reducing scale metadata bandwidth by ~31%.  This mirrors the design space Maia hardware navigates in silicon.

Formally pre-registered in `quant/shared_scale_spec.md` before any experimental data is collected.

---

## 3. Experiment Matrix

All 6 configurations use the same PyTorch runtime for fair comparison.
vLLM is a separate "serving reference" — not part of the ablation.

| Config | Name | What changes | Purpose |
|--------|------|-------------|---------|
| **C0** | BF16 baseline | Nothing — full precision | Gold reference for quality and speed |
| **C1** | FP8 activations | TransformerEngine FP8 (optional) | Optional data point if TE available |
| **C2** | W4A16 NF4 | bitsandbytes 4-bit NF4 | Industry-standard W4 comparison |
| **C3** | W4 + exact scales | SMQ arch, `scale_mbits=-1` (FP32 scales) | Isolates effect of int4 weights only |
| **C4** | W4 + E5M5 scales | SMQ arch, `scale_mbits=5` | **Proposed method** |
| **C5** | SMQ sweep | `scale_mbits ∈ {0, 3, 5}` | Ablation: scale precision vs quality |

**Frozen workloads:**
- Prompt lengths: 128, 512, 2048 tokens
- Output lengths: 64, 256 tokens
- 3 repeats per config (for 95% CI via t-distribution)
- 1 warmup pass discarded before all timing

**Quality tasks** (lm-eval, zero-shot): hellaswag, arc_challenge, gsm8k, lambada_openai

---

## 4. Codebase Structure

```
maia-inspired-phi35-h100/
│
├── configs/
│   └── experiment_matrix.yaml      # FROZEN experiment contract — single source of truth
│
├── quant/
│   ├── shared_scale_spec.md        # Pre-registered formal methods (LaTeX math)
│   ├── shared_scale_quant.py       # Pure-PyTorch SMQ reference implementation
│   └── tests/
│       └── test_shared_scale.py    # 18 unit tests (all must pass before benchmarking)
│
├── cuda/
│   └── shared_scale_dequant/
│       ├── shared_scale_dequant_cuda.cu   # GPU dequant kernel (fp16/bf16, 32×8 threadblock)
│       ├── shared_scale_dequant.cpp       # PyBind11 binding
│       └── setup.py                       # torch.utils.cpp_extension build
│
├── runtime/
│   ├── common.py          # Shared utils: PowerSampler, get_gpu_state, env snapshot
│   ├── torch_runner.py    # ABLATION runtime: loads model once, sweeps all combos
│   └── vllm_runner.py     # SERVING reference only (non-comparable to torch_runner)
│
├── bench/
│   ├── offline_bench.py   # Harness: reads YAML → drives torch_runner subprocesses
│   └── online_bench.py    # Async vLLM concurrency sweep (TTFT/TPOT vs QPS)
│
├── eval/
│   └── lm_eval_runner.py  # lm-eval-harness driver (hellaswag, arc, gsm8k, lambada)
│
├── scripts/
│   ├── aggregate_results.py   # Mean ± 95% CI across repeats → paper_table.csv
│   └── colab_quickstart.md    # Step-by-step run guide for Colab
│
├── results/
│   ├── raw/       # Per-run JSONL (one file per repeat, contains env snapshot)
│   └── summary/   # Per-config CSV (one row per prompt/output combo)
│
├── report/
│   └── report.md  # Paper-style writeup template (to be filled after runs)
│
├── requirements.txt
├── ENVIRONMENT.md
└── PROJECT.md      ← this file
```

---

## 5. What Is Done

| Component | Status | Notes |
|-----------|--------|-------|
| `quant/shared_scale_quant.py` | ✅ Complete | Full E5Mx quantize/dequant, `SharedScaleLinear`, `quant_error` |
| `quant/tests/test_shared_scale.py` | ✅ Complete | 18 tests covering monotonicity, roundtrip, ablation ordering, edge cases |
| `quant/shared_scale_spec.md` | ✅ Frozen | Pre-registered LaTeX spec, ablation table, error bounds |
| `configs/experiment_matrix.yaml` | ✅ Frozen | C0–C5 fully defined, all runtime: torch, fairness constraints locked |
| `cuda/.../shared_scale_dequant_cuda.cu` | ✅ Complete | Real kernel (not stub): 32×8 threadblock, AT_DISPATCH fp16/bf16 |
| `runtime/common.py` | ✅ Complete | `PowerSampler` (pynvml, 100ms, trapezoidal joules), `get_gpu_state` |
| `runtime/torch_runner.py` | ✅ Complete | Loads model once, sweeps combos in-process, warmup, energy, quant dispatch |
| `runtime/vllm_runner.py` | ✅ Complete | vLLM throughput reference, clearly labelled non-comparable |
| `bench/offline_bench.py` | ✅ Complete | YAML → subprocess list (no shell injection), C5 auto-sweeps mbits |
| `bench/online_bench.py` | ✅ Complete | Async vLLM concurrency sweep, TTFT/TPOT p50/p95 |
| `eval/lm_eval_runner.py` | ✅ Complete | `sys.executable -m lm_eval`, `trust_remote_code=True`, gsm8k gets chat template |
| `scripts/aggregate_results.py` | ✅ Complete | t-dist CI95, falls back to 1.96×SEM if scipy missing |
| `scripts/colab_quickstart.md` | ✅ Complete | Full 5-phase copy-paste workflow |
| `report/report.md` | ⬜ Template only | Fill after runs complete |

---

## 6. What Is Remaining

### 6.1 Run experiments on Colab H100

This is the only major remaining work.  All code is ready.

| Phase | Command | Time estimate |
|-------|---------|--------------|
| Unit tests | `python -m pytest quant/tests/ -v` | 1 min |
| C0 BF16 baseline | `python -m bench.offline_bench ... --config_id C0` | ~20 min (3 repeats) |
| C2 W4 NF4 | `...--config_id C2` | ~25 min |
| C3 exact-scale | `...--config_id C3` | ~25 min |
| C4 SMQ E5M5 | `...--config_id C4` | ~25 min |
| C5 sweep | `...--config_id C5` | ~45 min (3× 3 repeats) |
| lm-eval C0 | `python -m eval.lm_eval_runner ... --config_id C0` | ~30 min |
| lm-eval C2 | `...--config_id C2` | ~30 min |
| Aggregate | `python scripts/aggregate_results.py ...` | < 1 min |
| Push results | `git add -f results/ && git commit && git push` | 1 min |

**Total: ~3.5 hours on a single Colab H100 session.**

Use `--repeats 1` for a fast 1-hour smoke run first.

### 6.2 Write the report

Fill in `report/report.md` with:
- Actual numbers from `results/aggregate/paper_table.csv`
- Quality results from `results/lm_eval/`
- Discussion: does SMQ achieve the claimed Pareto point?
- Threats to validity (Colab variability, no calibration data, H100 ≠ Maia)

### 6.3 Optional extensions (time permitting)

| Extension | Value | Effort |
|-----------|-------|--------|
| C1 FP8 via TransformerEngine | Completes the precision ladder | 1 hr if TE is on Colab image |
| Long-context stress (8K, 32K tokens) | Shows memory advantage of W4 at scale | 2 hrs |
| Per-layer error correlation | Links quant error → perplexity delta per layer | 1 day |
| vLLM serving benchmark | Throughput at concurrency 1→32 | 2 hrs |
| CUDA kernel compilation + perf test | Validates kernel vs Python path | 1 day |

---

## 7. Key Design Decisions (Locked)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Model | `microsoft/Phi-3.5-mini-instruct` | Smallest competitive instruct model; MIT license; fits in H100 BF16 with room for variants |
| Runtime for ablation | PyTorch only (no vLLM) | Removes scheduler and kernel confound; all configs strictly comparable |
| group_size | 128 | Standard for W4; consistent with AWQ/GPTQ for direct comparison |
| quant_target | MLP layers only | Attention kept BF16 for numerical stability; isolates MLP bandwidth benefit |
| Decoding | Greedy (temp=0, do_sample=False, seed=42) | Deterministic; reproducible across repeats |
| Warmup | 1 pass (128-tok prefill, 32-tok decode, discarded) | Ensures CUDA kernels and KV cache are hot before timing |
| Repeats | 3 | Minimum for meaningful CI; balances time vs statistical rigor |

---

## 8. How to Reproduce (Short Version)

```bash
# 1. Clone
git clone https://github.com/pavannn16/maia-inspired-phi35-h100.git
cd maia-inspired-phi35-h100

# 2. Install (Colab H100)
pip install -r requirements.txt

# 3. Validate code
python -m pytest quant/tests/ -v

# 4. Run ablation (fast smoke: --repeats 1)
MODEL=microsoft/Phi-3.5-mini-instruct
CFG=configs/experiment_matrix.yaml
for CID in C0 C2 C3 C4 C5; do
  python -m bench.offline_bench --config $CFG --model $MODEL --config_id $CID --repeats 1
done

# 5. Run quality eval
python -m eval.lm_eval_runner --config $CFG --model $MODEL --config_id C0

# 6. Aggregate
python scripts/aggregate_results.py --summary_dir results/summary --out_csv results/aggregate/paper_table.csv

# 7. Push results
git checkout -b "results/$(date -u +%Y%m%dT%H%M%SZ)"
git add -f results/
git commit -m "Add benchmark results"
git push -u origin HEAD
```

See `scripts/colab_quickstart.md` for the full annotated workflow.
