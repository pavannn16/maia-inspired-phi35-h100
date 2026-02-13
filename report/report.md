# Maia-Inspired Mixed Precision for Phi-3.5 on H100 (Proxy)

## Abstract
(1 paragraph)

## Method
- Config matrix C0–C5
- Shared-scale quant format definition
- Kernel design (if applicable)

## Experimental Setup
- Hardware: Colab H100 (record GPU SKU)
- Software: pinned versions
- Measurement protocol (warmup, repeats, CI)

## Results
- Quality (lm-eval): per task + relative drop vs BF16
- Performance: throughput, TTFT/TPOT, memory
- Energy: joules/token (best-effort on Colab)

## Diagnostic Analysis
- Per-layer error vs accuracy correlation
- Long-context probes

## Threats to Validity
- H100 proxy vs Maia
- Serving scheduler confounds
- Colab variability

## Reproducibility
- Commands
- Artifact table
