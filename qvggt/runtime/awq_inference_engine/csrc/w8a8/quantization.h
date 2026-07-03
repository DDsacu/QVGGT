#include <torch/extension.h>
void invoke_quant(torch::Tensor &out,
                  torch::Tensor &input,
                  torch::Tensor &scale);
