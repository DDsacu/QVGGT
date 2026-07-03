#include <torch/extension.h>
#include <cuda_fp16.h>
void rms_norm_general(torch::Tensor &out,
              torch::Tensor &input,
              torch::Tensor &weight,
              torch::Tensor &bias,
              torch::Tensor &scaling,
              float epsilon,
              bool use_per_token_quant);
