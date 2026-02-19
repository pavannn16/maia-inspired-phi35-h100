# Colab Step-by-Step Workflow

All commands should be run from the repo root.
Expected runtime on a Colab H100: ~60–90 min for a full C0+C4 ablation run.

---

## Phase 0 — Environment setup (run once)

```bash
# Pull latest code
git pull

# Install deps
pip install -r requirements.txt

# Optional: bitsandbytes (C2 only)
pip install bitsandbytes

# Sanity-check CUDA
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## Phase 1 — Unit tests (run before any benchmarking)

```bash
python -m pytest quant/tests/test_shared_scale.py -v
```

All tests must pass before proceeding.  If any fail, open an issue.

---

## Phase 2 — Offline latency benchmarks

Run one `config_id` at a time.  Each command:
- Loads model ONCE per repeat
- Sweeps all (prompt_len, output_len) combos in-process
- Writes raw JSONL + summary CSV to `results/`

```bash
MODEL=microsoft/Phi-3.5-mini-instruct
CFG=configs/experiment_matrix.yaml

# C0 — BF16 baseline (~5 min)
python -m bench.offline_bench --config $CFG --model $MODEL --config_id C0

# C2 — W4 NF4 via bitsandbytes (~6 min, requires bitsandbytes)
python -m bench.offline_bench --config $CFG --model $MODEL --config_id C2

# C3 — W4 SMQ exact-scale reference (~6 min)
python -m bench.offline_bench --config $CFG --model $MODEL --config_id C3

# C4 — W4 SMQ scale_mbits=5 (proposed method) (~6 min)
python -m bench.offline_bench --config $CFG --model $MODEL --config_id C4

# C5 — SMQ ablation sweep (runs scale_mbits=0,3,5 automatically) (~15 min)
python -m bench.offline_bench --config $CFG --model $MODEL --config_id C5
```

With `repeats: 3` (from YAML), each command runs the full sweep 3 times for CI.

For a quick smoke-run use `--repeats 1`.

---

## Phase 3 — Quality evaluation (lm-eval)

Run hellaswag, arc_challenge, gsm8k, lambada_openai.

```bash
# C0 BF16 baseline
python -m eval.lm_eval_runner --config $CFG --model $MODEL --config_id C0

# C2 W4-NF4 quality check
python -m eval.lm_eval_runner --config $CFG --model $MODEL --config_id C2

# C4 SMQ quality check (note: uses custom SharedScaleLinear; lm_eval invokes
# the HF model, which does not apply SMQ. Only perf ablation is meaningful
# here until a custom lm_eval ModelWrapper is implemented.)
# python -m eval.lm_eval_runner --config $CFG --model $MODEL --config_id C4
```

---

## Phase 4 — Aggregate results across repeats

```bash
python scripts/aggregate_results.py \
    --summary_dir results/summary \
    --config_ids C0 C2 C3 C4 C5_m0 C5_m3 C5_m5 \
    --out_csv results/aggregate/paper_table.csv
```

---

## Phase 5 — Push results to GitHub

```bash
# Stage everything under results/
git add results/ --force
git commit -m "Add benchmark results: $(date +%Y-%m-%d)"
git push
```

Then on your local machine:
```bash
git pull
```

---

## Interpreting results

| Config | What it measures |
|--------|------------------|
| C0     | BF16 baseline (gold reference) |
| C2     | Industry-standard W4 (bitsandbytes NF4) |
| C3     | W4 + exact per-group scales (our arch, no scale compression) |
| C4     | W4 + E5M5 scales (proposed method) |
| C5     | Ablation sweep of scale precision |

Key claims to verify:
1. C3 ≈ C4 in throughput (scale compression adds no latency)
2. C4 ≈ C3 in quality (E5M5 scale quantization is nearly lossless)
3. C5 shows quality degrades as scale_mbits decreases
4. C3/C4 ≥ C2 in throughput (SMQ kernel more cache-friendly than NF4 dequant)

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: runtime` | Run `python -m bench.offline_bench ...` not `python bench/offline_bench.py ...` |
| `RuntimeError: CUDA out of memory` | Reduce `--repeats 1`; close other processes |
| `bitsandbytes not installed` | Run `pip install bitsandbytes`; skip C2 if wheel unavailable |
| `pynvml_error` in results | Power sampling unavailable; joules_per_token will be null (non-critical) |
