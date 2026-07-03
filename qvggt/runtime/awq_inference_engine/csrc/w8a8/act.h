#include <torch/extension.h>
#include <cuda_fp16.h>

#include <torch/extension.h>

void gelu_and_quant(torch::Tensor &out,
                                torch::Tensor &input,
                                torch::Tensor &scale_out,
                                torch::Tensor &tmp
);

torch::Tensor silu_and_mul(torch::Tensor &input
);
