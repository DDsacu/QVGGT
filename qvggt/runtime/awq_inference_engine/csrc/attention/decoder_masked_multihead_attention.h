#pragma once

#include "cuda_bf16_wrapper.h"
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t status_ = call;                                                                                    \
        if (status_ != cudaSuccess) {                                                                                  \
            fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status_));              \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

template<typename T>
struct Multihead_attention_params_base {

    T* out = nullptr;

    const T *q = nullptr, *q_bias = nullptr;

    const T *k = nullptr, *k_bias = nullptr;

    const T *v = nullptr, *v_bias = nullptr;

    T* k_cache = nullptr;

    T* v_cache = nullptr;

    const int* cache_indir = nullptr;

    int stride = 0;

    int batch_size = 0;

    int beam_width = 0;

    int memory_max_len = 0;

    int num_heads = 0;

    int num_kv_heads = 0;

    int hidden_size_per_head = 0;

    int  rotary_embedding_dim = 0;
    bool neox_rotary_style    = false;
    float rotary_base = 0.0f;
    float rotary_scale = 1.0f;

    int max_input_length = 0;

    int timestep = 0;

    float inv_sqrt_dh = 0.0f;

    const int* total_padding_tokens = nullptr;

    const bool* masked_tokens            = nullptr;
    const int*  prefix_prompt_lengths    = nullptr;
    int         max_prefix_prompt_length = 0;

    const T* relative_attention_bias        = nullptr;
    int      relative_attention_bias_stride = 0;

    const float* linear_bias_slopes = nullptr;

    const T*   ia3_key_weights   = nullptr;
    const T*   ia3_value_weights = nullptr;
    const int* ia3_tasks         = nullptr;

    const float* qkv_scale_out       = nullptr;
    const float* attention_out_scale = nullptr;
    int          int8_mode           = 0;
};

template<typename T, bool CROSS_ATTENTION>
struct Multihead_attention_params: public Multihead_attention_params_base<T> {

    float* cross_attention_out        = nullptr;
    int    max_decoder_seq_len        = 0;
    bool   is_return_cross_attentions = false;

    bool* finished = nullptr;

    int* memory_length_per_sample = nullptr;

    const int* length_per_sample = nullptr;
};

template<typename T>
struct Multihead_attention_params<T, true>: public Multihead_attention_params_base<T> {

    float* cross_attention_out        = nullptr;
    int    max_decoder_seq_len        = 0;
    bool   is_return_cross_attentions = false;

    bool* finished = nullptr;

    int* memory_length_per_sample = nullptr;

    const int* length_per_sample = nullptr;
};

template<class T>
using Masked_multihead_attention_params = Multihead_attention_params<T, false>;

template<class T>
using Cross_multihead_attention_params = Multihead_attention_params<T, true>;

template<typename T>
struct outputCrossAttentionParam {

    int  max_decoder_seq_len        = 0;
    T*   cross_attention_out        = nullptr;
    bool is_return_cross_attentions = false;
};

void masked_multihead_attention(const Masked_multihead_attention_params<float>& params, const cudaStream_t& stream);
void masked_multihead_attention(const Masked_multihead_attention_params<uint16_t>& params, const cudaStream_t& stream);
#ifdef ENABLE_BF16
void masked_multihead_attention(const Masked_multihead_attention_params<__nv_bfloat16>& params,
                                const cudaStream_t&                                     stream);
#endif
void cross_multihead_attention(const Cross_multihead_attention_params<float>& params, const cudaStream_t& stream);
void cross_multihead_attention(const Cross_multihead_attention_params<uint16_t>& params, const cudaStream_t& stream);
#ifdef ENABLE_BF16
void cross_multihead_attention(const Cross_multihead_attention_params<__nv_bfloat16>& params,
                               const cudaStream_t&                                    stream);
#endif
