#include <torch/extension.h>

torch::Tensor dequant_int4_shared_scale(torch::Tensor packed_w, torch::Tensor scales, int64_t group_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dequant_int4_shared_scale", &dequant_int4_shared_scale, "Shared-scale int4 dequant (CUDA)");
}
