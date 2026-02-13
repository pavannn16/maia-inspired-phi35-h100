#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Skeleton only: implementation will be added in Week 4.
// For now, we raise a clear error if called.

torch::Tensor dequant_int4_shared_scale(torch::Tensor packed_w, torch::Tensor scales, int64_t group_size) {
  TORCH_CHECK(packed_w.is_cuda(), "packed_w must be CUDA");
  TORCH_CHECK(scales.is_cuda(), "scales must be CUDA");
  TORCH_CHECK(false, "dequant_int4_shared_scale is not implemented yet");
  return packed_w;
}
