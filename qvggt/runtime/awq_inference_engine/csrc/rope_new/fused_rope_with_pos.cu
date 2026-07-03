#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "fused_rope_with_pos.h"

#define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)         \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

#define THREADS_PER_WARP 32

template <typename scalar_t>
__device__ void fused_rope_with_pos_block_forward(
    const scalar_t *src, const float *freqs, scalar_t *dst,
    const int offset_block, const int offset_block_dst, const int h,
    const int d, const int d2, const int stride_h, const int stride_d,
    const int o_stride_h, const int o_stride_d) {
  int s_id = blockIdx.x;
  int s = gridDim.x;
  int b_id = blockIdx.y;
#pragma unroll
  for (int d_id = threadIdx.x; d_id < d2; d_id += blockDim.x) {
    float v_cos, v_sin;
    sincosf(freqs[(b_id * s + s_id) * d2 + d_id], &v_sin, &v_cos);
#pragma unroll
    for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
      int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
      int offset_dst = offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
      float v_src = src[offset_src];
      float v_src_rotate =
          (d_id + d2 / 2 < d2)
              ? -static_cast<float>(src[offset_src + (d2 / 2) * stride_d])
              : static_cast<float>(src[offset_src + (d2 / 2 - d2) * stride_d]);
      dst[offset_dst] = v_src * v_cos + v_src_rotate * v_sin;
    }
  }

  if (d > d2) {
#pragma unroll
    for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
      int offset_head = offset_block + h_id * stride_h;
      int offset_head_dst = offset_block_dst + h_id * o_stride_h;
#pragma unroll
      for (int d_id = d2 + threadIdx.x; d_id < d; d_id += blockDim.x) {
        dst[offset_head_dst + d_id * o_stride_d] =
            src[offset_head + d_id * stride_d];
      }
    }
  }
}

template <typename scalar_t>
__global__ void fused_rope_with_pos_forward_kernel(
    const scalar_t *src, const float *freqs, scalar_t *dst, const int h,
    const int d, const int d2, const int stride_s, const int stride_b,
    const int stride_h, const int stride_d, const int o_stride_s,
    const int o_stride_b, const int o_stride_h, const int o_stride_d) {
  int s_id = blockIdx.x, b_id = blockIdx.y;
  int offset_block = s_id * stride_s + b_id * stride_b;
  int offset_block_dst = s_id * o_stride_s + b_id * o_stride_b;
  fused_rope_with_pos_block_forward<scalar_t>(
      src, freqs, dst, offset_block, offset_block_dst, h, d, d2, stride_h,
      stride_d, o_stride_h, o_stride_d);
}

template <typename scalar_t>
void fused_rope_with_pos_forward_launcher(
    const scalar_t *input, const float *freqs, scalar_t *output, const int s,
    const int b, const int h, const int d, const int d2, const int stride_s,
    const int stride_b, const int stride_h, const int stride_d,
    const int o_stride_s, const int o_stride_b, const int o_stride_h,
    const int o_stride_d, cudaStream_t stream) {
  int warps_per_block = h < 16 ? 4 : 8;
  dim3 blocks(s, b);
  dim3 threads(THREADS_PER_WARP, warps_per_block);

  fused_rope_with_pos_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
      input, freqs, output, h, d, d2, stride_s, stride_b, stride_h, stride_d,
      o_stride_s, o_stride_b, o_stride_h, o_stride_d);

}

template <typename scalar_t>
void fused_rope_with_pos_forward(const at::Tensor &input,
                                 const at::Tensor &freqs, at::Tensor &output,
                                 const int s, const int b, const int h,
                                 const int d, const int d2, const int stride_s,
                                 const int stride_b, const int stride_h,
                                 const int stride_d, const int o_stride_s,
                                 const int o_stride_b, const int o_stride_h,
                                 const int o_stride_d, cudaStream_t stream) {

  fused_rope_with_pos_forward_launcher<scalar_t>(
      reinterpret_cast<const scalar_t *>(input.data_ptr()),
      reinterpret_cast<const float *>(freqs.data_ptr()),
      reinterpret_cast<scalar_t *>(output.data_ptr()), s, b, h, d, d2, stride_s,
      stride_b, stride_h, stride_d, o_stride_s, o_stride_b, o_stride_h,
      o_stride_d, stream);

}

template <typename scalar_t>
void nvte_fused_rope_with_pos_forward(
    const at::Tensor input, const at::Tensor freqs, at::Tensor output,
    const int s, const int b, const int h, const int d, const int d2,
    const int stride_s, const int stride_b, const int stride_h,
    const int stride_d, const int o_stride_s, const int o_stride_b,
    const int o_stride_h, const int o_stride_d, cudaStream_t stream) {

  fused_rope_with_pos_forward<scalar_t>(
      input, freqs, output, s, b, h, d, d2, stride_s, stride_b, stride_h,
      stride_d, o_stride_s, o_stride_b, o_stride_h, o_stride_d, stream);
}

at::Tensor fused_rope_with_pos_forward_func(
    const at::Tensor &input, const at::Tensor &freqs,
    const bool transpose_output_memory) {

  const int s = input.size(0);
  const int b = input.size(1);
  const int h = input.size(2);
  const int d = input.size(3);

  const int stride_s = input.stride(0);
  const int stride_b = input.stride(1);
  const int stride_h = input.stride(2);
  const int stride_d = input.stride(3);

  const int d2 = freqs.size(-1);

  auto act_options = input.options().requires_grad(false);
  at::Tensor output;
  if (transpose_output_memory) {
    output = torch::empty({b, s, h, d}, act_options).transpose(0, 1);
  } else {
    output = torch::empty({s, b, h, d}, act_options);
  }

  const int o_stride_s = output.stride(0);
  const int o_stride_b = output.stride(1);
  const int o_stride_h = output.stride(2);
  const int o_stride_d = output.stride(3);

  auto input_cu = input;
  auto freqs_cu = freqs;
  auto output_cu = output;

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "nvte_fused_rope_forward", [&] {
        nvte_fused_rope_with_pos_forward<scalar_t>(
            input_cu.data(), freqs_cu.data(), output_cu.data(), s, b, h, d, d2,
            stride_s, stride_b, stride_h, stride_d, o_stride_s, o_stride_b,
            o_stride_h, o_stride_d, at::cuda::getCurrentCUDAStream());
      });

  return output;
}
