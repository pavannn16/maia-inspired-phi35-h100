# Environment + Reproducibility

This project targets **Google Colab H100**.

## What to record for every run
- GPU: name + total memory
- Driver + CUDA version
- PyTorch version
- Transformers version
- vLLM version (if used)
- Model ID + revision hash
- Exact config hash (C0–C5)

The runners emit an environment snapshot into each JSONL record.

## Version pinning workflow (recommended)
1. Start with `requirements.txt` minimum versions.
2. When a config is validated end-to-end, freeze exact versions into a `requirements.lock.txt` (or paste into the report appendix).

## Colab notes
- Colab images change; expect occasional wheel incompatibilities.
- Prefer pip wheels over source builds. Keep CUDA extension builds minimal.
