#include <cuda_fp16.h>
#include <cuda_bf16.h>
#pragma once

template <typename T = half>
__inline__ __device__ void dequantize_s4_to_fp16x2(half2 const &source, uint4 *result);

template <>
__inline__ __device__ void dequantize_s4_to_fp16x2<half>(half2 const &source, uint4 *result)
{

    uint32_t *h = reinterpret_cast<uint32_t *>(result);
    uint32_t const i4s = reinterpret_cast<uint32_t const &>(source);

    constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
    constexpr uint32_t BOTTOM_MASK = 0x000f000f;
    constexpr uint32_t TOP_MASK = 0x00f000f0;
    constexpr uint32_t I4s_TO_F16s_MAGIC_NUM = 0x64006400;

    const uint32_t top_i4s = i4s >> 8;

    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[0])
                 : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));

    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[1])
                 : "r"(i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));

    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[2])
                 : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));

    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[3])
                 : "r"(top_i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));

    static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64006400;

    static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;

    static constexpr uint32_t NEG_64 = 0xd400d400;

    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));

    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_64));

    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(FP16_TOP_MAGIC_NUM));

    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[3]) : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_64));
}

template <>
__inline__ __device__ void dequantize_s4_to_fp16x2<nv_bfloat16>(half2 const &source, uint4 *result)
{

    uint32_t *h = reinterpret_cast<uint32_t *>(result);
    uint32_t const i4s = reinterpret_cast<uint32_t const &>(source);

    constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
    constexpr uint32_t BOTTOM_MASK = 0x000f000f;
    constexpr uint32_t I4s_TO_BF16s_MAGIC_NUM = 0x43004300;

    const uint32_t i4s1 = i4s >> 4;
    const uint32_t i4s2 = i4s >> 8;
    const uint32_t i4s3 = i4s >> 12;

    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[0])
                 : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_BF16s_MAGIC_NUM), "n"(immLut));

    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[1])
                 : "r"(i4s1), "n"(BOTTOM_MASK), "n"(I4s_TO_BF16s_MAGIC_NUM), "n"(immLut));

    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[2])
                 : "r"(i4s2), "n"(BOTTOM_MASK), "n"(I4s_TO_BF16s_MAGIC_NUM), "n"(immLut));

    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[3])
                 : "r"(i4s3), "n"(BOTTOM_MASK), "n"(I4s_TO_BF16s_MAGIC_NUM), "n"(immLut));

    static constexpr uint32_t BF16_TOP_MAGIC_NUM = 0x43004300;

    reinterpret_cast<__nv_bfloat162*>(h)[0] = __hsub2(reinterpret_cast<__nv_bfloat162*>(h)[0], reinterpret_cast<const __nv_bfloat162&>(BF16_TOP_MAGIC_NUM));
    reinterpret_cast<__nv_bfloat162*>(h)[1] = __hsub2(reinterpret_cast<__nv_bfloat162*>(h)[1], reinterpret_cast<const __nv_bfloat162&>(BF16_TOP_MAGIC_NUM));
    reinterpret_cast<__nv_bfloat162*>(h)[2] = __hsub2(reinterpret_cast<__nv_bfloat162*>(h)[2], reinterpret_cast<const __nv_bfloat162&>(BF16_TOP_MAGIC_NUM));
    reinterpret_cast<__nv_bfloat162*>(h)[3] = __hsub2(reinterpret_cast<__nv_bfloat162*>(h)[3], reinterpret_cast<const __nv_bfloat162&>(BF16_TOP_MAGIC_NUM));
}
