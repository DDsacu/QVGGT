#include <cuda_fp16.h>
#include <stdio.h>
#include <torch/extension.h>
#include "gemv_cuda.h"
#include "../dequantize.cuh"
#include "../dispatch_utils.cuh"
#define PACK_FACTOR 8
#define WARP_SIZE 32
#define MEM_ACCESS_SIZE 128

template <int Num, int WarpSize, typename T>
__device__ __forceinline__ static void warp_reduce(T* psum, float (*out_smem)[Num * 4])
{

      float fpsum[Num];
      #pragma unroll
      for (int i = 0; i < Num; ++i)
      {
          fpsum[i] = static_cast<float>(psum[i]);
      }

      #pragma unroll
      for (int i = 0; i < Num; ++i)
      {

          fpsum[i] += __shfl_xor_sync(~0, fpsum[i], 16);
          fpsum[i] += __shfl_xor_sync(~0, fpsum[i], 8);
          fpsum[i] += __shfl_xor_sync(~0, fpsum[i], 1);
      }
      __syncthreads();
      int warp = threadIdx.x / WarpSize, lane = threadIdx.x % WarpSize;
      if (lane == 0 || lane == 2 || lane == 4 || lane == 6)
      {
          #pragma unroll
          for (int i = 0; i < Num; ++i)
          {
              out_smem[warp][i * 4 + lane / 2] = fpsum[i];
          }
      }
      __syncthreads();
};

__device__ __forceinline__ int make_divisible(int c, int divisor){
  return (c + divisor - 1) / divisor;
}

template <int NPerBlock, int Batch, int BlockSize, int GroupSize, typename T>
__global__ void gemv_kernel(
  const T* inputs, const uint32_t* weight, const T* scales, const T* zeros, T* outputs,
  const int IC, const int OC)
{
    const int kStride = 64;
    const int kElemsPerThread = MEM_ACCESS_SIZE / 4;
    const int kThreadsNumPerTile = kStride / kElemsPerThread;

    using T2 = typename std::conditional<
        std::is_same<T, half>::value,
        half2,
        nv_bfloat162
    >::type;

    static constexpr int kShuffleSize = 32;
    static constexpr int kShuffleBasicTile = 2;
    static constexpr int kShuffleContinous = 4;
    static constexpr int kShuffleStrided = 4;

    constexpr int Num = NPerBlock * Batch;
    constexpr int kInterleave = 4;

    T local_inputs[kElemsPerThread];
    uint32_t local_qweights[MEM_ACCESS_SIZE / 32];
    T half_weight_buffer[kElemsPerThread];
    T dequantized_weight[kElemsPerThread * NPerBlock];
    T local_scale[NPerBlock];
    T local_scaled_zeros[NPerBlock];

    T psum[Num];
    for (int i = 0; i < Num; ++i)
        psum[i] = static_cast<T>(0.f);

    __shared__ float out_smem[BlockSize / WARP_SIZE * 2][Num * kInterleave];

    const int blk_row_offset = blockIdx.x * NPerBlock * kInterleave;
    const int thd_row_offset = (threadIdx.x / kThreadsNumPerTile) % kInterleave;
    const int act_k_offset = threadIdx.x / (kThreadsNumPerTile * kInterleave) * kStride
                               + (threadIdx.x % kThreadsNumPerTile) * kElemsPerThread;
    const int group_offset = act_k_offset / GroupSize;

    const uint32_t* blk_weight_ptr = weight + blk_row_offset * IC / PACK_FACTOR;
    const T* scale_ptr = scales + blk_row_offset + thd_row_offset + group_offset * OC;
    const T* zeros_ptr = zeros + blk_row_offset + thd_row_offset + group_offset * OC;
    const T* inputs_ptr = inputs + act_k_offset;

    const int act_forward_step = BlockSize * kElemsPerThread / kInterleave;
    const int scale_forward_step = act_forward_step / GroupSize * OC;

    for (int kk = threadIdx.x * kElemsPerThread; kk < IC * kInterleave; kk += BlockSize * kElemsPerThread)
    {

        #pragma unroll
        for (int idx = 0; idx < NPerBlock; ++idx)
        {

            *((float4*)(local_qweights)) =
                *((float4*)(blk_weight_ptr + (idx * kInterleave * IC + kk)/ PACK_FACTOR));
            local_scale[idx] = *(scale_ptr + idx * kInterleave);
            local_scaled_zeros[idx] = *(zeros_ptr + idx * kInterleave);

            #pragma unroll
            for (int i = 0; i < MEM_ACCESS_SIZE / 32; ++i)
            {

                dequantize_s4_to_fp16x2<T>(*reinterpret_cast<half2 *>(local_qweights + i), reinterpret_cast<uint4 *>(half_weight_buffer + i * PACK_FACTOR));
            }

            #pragma unroll
            for (int i = 0; i < kShuffleContinous; ++i)
            {
                #pragma unroll
                for (int j = 0; j < kShuffleStrided; ++j)
                {
                    T2 w =
                        *reinterpret_cast<T2*>(
                          half_weight_buffer + (i + j * kShuffleContinous)* kShuffleBasicTile
                        );
                    if constexpr (std::is_same<T, half>::value)
                    {
                      w = __hfma2(w, __half2half2(local_scale[idx]), __half2half2(local_scaled_zeros[idx]));
                    }
                    else
                    {
                      w = __hfma2(w, __bfloat162bfloat162(local_scale[idx]), __bfloat162bfloat162(local_scaled_zeros[idx]));
                    }
                    dequantized_weight[((i * kShuffleStrided + j) * kShuffleBasicTile + 0)
                          * NPerBlock + idx]
                        = w.x;
                    dequantized_weight[((i * kShuffleStrided + j) * kShuffleBasicTile + 1)
                            * NPerBlock + idx]
                        = w.y;
                }
            }
        }
        #pragma unroll
        for (int batch_idx = 0; batch_idx < Batch; ++batch_idx)
        {
            const T* local_inputs_ptr = inputs_ptr + batch_idx * IC;
            #pragma unroll
            for (int idx = 0; idx < kElemsPerThread / 8; ++idx)
            {

                *((float4*)(local_inputs + idx * 8)) = *((float4*)(local_inputs_ptr + idx * 8));
            }

            #pragma unroll
            for (int x = 0; x < NPerBlock / 2; ++x)
            {
                #pragma unroll
                for (int y = 0; y < kElemsPerThread; ++y)
                {
                    if constexpr (std::is_same<T, half>::value)
                    {
                      *reinterpret_cast<half2*>(psum + batch_idx * NPerBlock + x * 2)
                          = __hfma2(*reinterpret_cast<half2*>(dequantized_weight + y * NPerBlock + x * 2),
                              __half2half2(local_inputs[y]),
                              *reinterpret_cast<half2*>(psum + batch_idx * NPerBlock + x * 2));
                    }
                    else
                    {
                      *reinterpret_cast<nv_bfloat162*>(psum + batch_idx * NPerBlock + x * 2)
                          = __hfma2(*reinterpret_cast<nv_bfloat162*>(dequantized_weight + y * NPerBlock + x * 2),
                              __bfloat162bfloat162(local_inputs[y]),
                              *reinterpret_cast<nv_bfloat162*>(psum + batch_idx * NPerBlock + x * 2));
                    }
                }
            }
        }
        inputs_ptr += act_forward_step;
        scale_ptr += scale_forward_step;
        zeros_ptr += scale_forward_step;
    }

    warp_reduce<Num, WARP_SIZE>(psum, out_smem);

    for (int i = threadIdx.x; i < Num * kInterleave; i += BlockSize)
    {
        int batch_idx = i / (NPerBlock * kInterleave);
        int oc_idx = i % (NPerBlock * kInterleave);
        float acc = 0.f;
        for (int j = 0; j < BlockSize / WARP_SIZE; ++j)
        {
            acc += out_smem[j][i];
        }
        outputs[batch_idx * OC + blk_row_offset + oc_idx] = static_cast<T>(acc);
    }
}

torch::Tensor gemv_forward_cuda_new(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    int m,
    int n,
    int k,
    int group_size)
{

    std::vector<int64_t> output_shape = _in_feats.sizes().vec();
    output_shape.back() = n;

    auto data_type = _in_feats.scalar_type();
    TORCH_CHECK(_scaling_factors.scalar_type() == data_type);
    TORCH_CHECK(_zeros.scalar_type() == data_type);

    auto options = torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
    at::Tensor _out_feats = torch::empty(output_shape, options);

    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(data_type, ctype, {
      auto in_feats = reinterpret_cast<ctype*>(_in_feats.data_ptr());
      auto kernel = reinterpret_cast<uint32_t*>(_kernel.data_ptr());
      auto zeros = reinterpret_cast<ctype*>(_zeros.data_ptr());
      auto scaling_factors = reinterpret_cast<ctype*>(_scaling_factors.data_ptr());
      auto out_feats = reinterpret_cast<ctype*>(_out_feats.data_ptr());

      static constexpr int N_PER_BLOCK = 2;
      static constexpr int K_INTERLEAVE = 4;
      static constexpr int BLOCK_SIZE = 256;

      dim3 num_blocks(n / N_PER_BLOCK / K_INTERLEAVE);
      dim3 num_threads(BLOCK_SIZE);

      if (group_size == 128)
      {
        switch (m)
        {
        case 1:
          gemv_kernel<N_PER_BLOCK, 1, BLOCK_SIZE, 128><<<num_blocks, num_threads>>>(
            in_feats, kernel, scaling_factors, zeros, out_feats, k, n
          );
          break;
        case 2:
          gemv_kernel<N_PER_BLOCK, 2, BLOCK_SIZE, 128><<<num_blocks, num_threads>>>(
            in_feats, kernel, scaling_factors, zeros, out_feats, k, n
          );
          break;
        case 3:
          gemv_kernel<N_PER_BLOCK, 3, BLOCK_SIZE, 128><<<num_blocks, num_threads>>>(
            in_feats, kernel, scaling_factors, zeros, out_feats, k, n
          );
          break;
        case 4:
          gemv_kernel<N_PER_BLOCK, 4, BLOCK_SIZE, 128><<<num_blocks, num_threads>>>(
            in_feats, kernel, scaling_factors, zeros, out_feats, k, n
          );
          break;
        case 5:
          gemv_kernel<N_PER_BLOCK, 5, BLOCK_SIZE, 128><<<num_blocks, num_threads>>>(
            in_feats, kernel, scaling_factors, zeros, out_feats, k, n
          );
          break;
        case 6:
          gemv_kernel<N_PER_BLOCK, 6, BLOCK_SIZE, 128><<<num_blocks, num_threads>>>(
            in_feats, kernel, scaling_factors, zeros, out_feats, k, n
          );
          break;
        case 7:
          gemv_kernel<N_PER_BLOCK, 7, BLOCK_SIZE, 128><<<num_blocks, num_threads>>>(
            in_feats, kernel, scaling_factors, zeros, out_feats, k, n
          );
          break;
        default:
          throw std::runtime_error("Unsupported batch size for gemv kernel.\n");
        }
      }
      else
      {
        throw std::runtime_error("Unsupported group size for gemv kernel.\n");
      }
    });
    return _out_feats;
}
