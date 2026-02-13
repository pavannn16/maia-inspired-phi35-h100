# Shared-Scale W4 Quant Spec (Draft)

Goal: define a **W4 weight format** whose **scale metadata** is represented in a compact, FP8-like representation.

## Definitions (to freeze before running C4/C5)
- Group size: TBD (e.g., 64 or 128)
- Weight packing: int4 packed into uint8 (2 weights per byte)
- Scale representation: shared per-group scale `s_g`
- `scale_mbits`: mantissa bits for scale quantization (ablation: 0,3,5)

## Quantize
Given BF16 weights $w$ grouped into $g$:
1. compute group scale $s_g$ (exact definition TBD)
2. quantize scale into compact form (controlled by `scale_mbits`)
3. quantize weights: $q = \mathrm{clip}(\mathrm{round}(w / s_g), -8, 7)$

## Dequantize
$\hat{w} = q \cdot \hat{s}_g$

## Notes
- You must document rounding mode and saturation.
- Define outlier handling policy (optional fallback list).
