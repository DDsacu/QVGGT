#pragma once

class Semaphore
{
public:
  int *lock;
  bool wait_thread;
  int state;

public:

  __host__ __device__ Semaphore(int *lock_, int thread_id) : lock(lock_),
                                                             wait_thread(thread_id < 0 || thread_id == 0),
                                                             state(-1)
  {
  }

  __device__ void fetch()
  {
    if (wait_thread)
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));
#else
      asm volatile("ld.global.cg.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));
#endif
    }
  }

  __device__ int get_state() const
  {
    return state;
  }

  __device__ void wait(int status = 0)
  {
    while (__syncthreads_and(state != status))
    {
      fetch();
    }

    __syncthreads();
  }

  __device__ void release(int status = 0)
  {
    __syncthreads();

    if (wait_thread)
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
      asm volatile("st.global.release.gpu.b32 [%0], %1;\n" : : "l"(lock), "r"(status));
#else
      asm volatile("st.global.cg.b32 [%0], %1;\n" : : "l"(lock), "r"(status));
#endif
    }
  }
};
