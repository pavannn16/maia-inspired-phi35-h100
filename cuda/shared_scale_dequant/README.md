# shared_scale_dequant (CUDA extension skeleton)

This directory will contain the CUDA extension for shared-scale W4 dequant.

Initial goal: provide a minimal `torch.utils.cpp_extension` build that exposes a function like:

- `dequant_int4_shared_scale(packed_w, scales, group_size) -> fp16/bf16 weights`

On Colab, keep this extension small to avoid long compile times.
