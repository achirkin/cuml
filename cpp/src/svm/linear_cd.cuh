/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

/**
 * @file linear_svm.cuh
 * @brief Fit linear SVM.
 */

#include <iostream>
#include <random>

#include <raft/linalg/cublas_wrappers.h>
#include <raft/linalg/transpose.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <common/nvtx.hpp>
#include <label/classlabels.cuh>
#include <matrix/kernelfactory.cuh>
#include <raft/cuda_utils.cuh>
#include <raft/linalg/map.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/matrix/matrix.hpp>
#include <rmm/device_uvector.hpp>
#include "solver/shuffle.h"

#include <cuml/svm/linear.hpp>

namespace ML {
namespace SVM {

namespace SVC_CD_Impl {

#define EPS 1e-12

using RaftAllocator = std::shared_ptr<raft::mr::device::allocator>;

template <template <int> class F, int... Sizes>
struct FitBlockSize {
  template <typename... Args>
  static inline auto run(int n, Args&&... args);
};

template <template <int> class F, int Size0, int... Sizes>
struct FitBlockSize<F, Size0, Sizes...> {
  template <typename... Args>
  static inline auto run(int n, Args&&... args)
  {
    if (n <= Size0)
      return F<Size0>::run(args...);
    else
      return FitBlockSize<F, Sizes...>::run(n, args...);
  }
};

template <template <int> class F, int Size0>
struct FitBlockSize<F, Size0> {
  template <typename... Args>
  static inline auto run(int n, Args&&... args)
  {
    return F<Size0>::run(args...);
  }
};

template <int Ix, typename... Args>
struct SaveArgs {
  static void run(void** target, Args&... args);
};

template <int Ix>
struct SaveArgs<Ix> {
  static void run(void** target) {}
};

template <int Ix, typename Arg0, typename... Args>
struct SaveArgs<Ix, Arg0, Args...> {
  static void run(void** target, Arg0& arg, Args&... args)
  {
    target[Ix] = (void*)(&arg);
    SaveArgs<Ix + 1, Args...>::run(target, args...);
  }
};

/**
 * Save any stuff passed by reference to a pointer array
 *    - arguments of a CUDA launch.
 *
 * Warning: be careful with your arguments, it's not typechecked at all!
 */
template <typename... Args>
void saveArgs(void** target, Args&... args)
{
  SaveArgs<0, Args...>::run(target, args...);
}

/** Get X[i,j] (column-major) with a concatenated column of ones for the bias. */
template <typename T, bool WithBias>
struct GetX {
  inline static __device__ T
  run(const T* X, const int i, const int j, const int nRows, const int nCols);
};

template <typename T>
struct GetX<T, true> {
  inline static __device__ T
  run(const T* X, const int i, const int j, const int nRows, const int nCols)
  {
    // return j + 1 == nCols ? T(1) : X[i + j * nRows];
    return j + 1 == nCols ? T(1) : X[i * (nCols - 1) + j];
  }
};

template <typename T>
struct GetX<T, false> {
  inline static __device__ T
  run(const T* X, const int i, const int j, const int nRows, const int nCols)
  {
    // return X[i + j * nRows];
    return X[i * nCols + j];
  }
};

template <typename T, bool WithBias>
__device__ inline T getX(const T* X, const int i, const int j, const int nRows, const int nCols)
{
  return GetX<T, WithBias>::run(X, i, j, nRows, nCols);
}

struct DeviceInfo {
 private:
  static int getDevice()
  {
    int devId       = 0;
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDevice(&devId));
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (devId >= 0 && devId < deviceCount) return devId;
    ASSERT(deviceCount > 0, "LinearSVC: couldn't find any CUDA devices!");
    devId = 0;
    CUDA_CHECK(cudaSetDevice(devId));
    return devId;
  }

  static int getAttr(int devId, cudaDeviceAttr attr)
  {
    int r;
    CUDA_CHECK(cudaDeviceGetAttribute(&r, attr, devId));
    return r;
  }

 public:
  const int devId;
  const int warpSize;
  const int numSMs;
  const int maxThreadsPerBlock;
  const int maxBlockDimX;
  const int maxBlockDimY;
  const int maxBlockDim;

  DeviceInfo(int dev)
    : devId(dev),
      warpSize(getAttr(dev, cudaDevAttrWarpSize)),
      numSMs(getAttr(dev, cudaDevAttrMultiProcessorCount)),
      maxThreadsPerBlock(getAttr(dev, cudaDevAttrMaxThreadsPerBlock)),
      maxBlockDimX(getAttr(dev, cudaDevAttrMaxBlockDimX)),
      maxBlockDimY(getAttr(dev, cudaDevAttrMaxBlockDimY)),
      maxBlockDim(min(maxBlockDimX, maxBlockDimY))
  {
  }

  DeviceInfo() : DeviceInfo(getDevice()) {}
};

int calculateKernelOccupancy(const void* kernel, const int blockSize, const size_t sharedMemory)
{
  int o;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o, kernel, blockSize, sharedMemory));
  return o;
}

/*
Compute inverse values of the diagonal of Q'

1/Q'ii = 1 / (1 + yi * yi * <xi, xi> + Dii)
       = 1 / (1 + Sum_j (x_ij * x_ij) + Dii)
  where
    L1: Dii = 0
    L2: Dii = 0.5 / cWeighted

 */
template <typename T, int ColsBlock, bool WithBias>
__global__ void precomputeQii(
  T* Qii, const T* X, const T* cWeighted, const int nRows, const int nCols, const T d)
{
  typedef cub::BlockReduce<T, ColsBlock> ColsBlockReduce;
  __shared__ typename ColsBlockReduce::TempStorage shm;
  int i = blockIdx.x;

  T x, x2acc = T(0);
  for (int j = threadIdx.x; j < nCols; j += ColsBlock) {
    x = getX<T, WithBias>(X, i, j, nRows, nCols);
    x2acc += x * x;
  }
  T value = ColsBlockReduce(shm).Sum(x2acc);
  if (threadIdx.x == 0) Qii[i] = value + (d != 0 ? d / cWeighted[i] : 0);
}

template <typename T>
struct CdState {
  T pGMaxGlobal                  = std::numeric_limits<T>::infinity();
  T pGMinGlobal                  = -std::numeric_limits<T>::infinity();
  T pGMax                        = -std::numeric_limits<T>::infinity();
  T pGMin                        = std::numeric_limits<T>::infinity();
  unsigned int shrinkCounter     = 0;
  unsigned int iterShrinkCounter = 0;
  unsigned int subIterGCounter   = 0;

  inline T criterion() const
  {
    const T a = this->pGMin;
    const T b = this->pGMax;
    return raft::myMax<T>(b - a, raft::myMax<T>(abs(a), abs(b)));
  }
};

__device__ inline uint64_t gcd(uint64_t a, uint64_t b)
{
  uint64_t tmp;
  while (b != 0) {
    tmp = b;
    b   = a % b;
    a   = tmp;
  }
  return a;
}

template <typename T>
__global__ void permuteIndices(const unsigned int a,
                               const unsigned int b,
                               const unsigned int nRows,
                               const CdState<T>* state,
                               int* in,
                               int* out)
{
  uint64_t shrinkCounterPrev = state->shrinkCounter;
  uint64_t shrinkCounter     = shrinkCounterPrev + state->iterShrinkCounter;
  uint64_t n                 = nRows - shrinkCounter;
  uint64_t i                 = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= nRows || i < shrinkCounterPrev) return;

  if (i < shrinkCounter) {
    out[i] = in[i];
  } else {
    uint64_t a0 = max(uint64_t(2), a % n);
    uint64_t b0 = b % n;
    while (gcd(a0, n) != 1)
      a0++;
    out[i] = in[(a0 * i + b0) % n + shrinkCounter];
  }
}

/**
 * invariant: R < S
 *   This invariant means the WorkRatio must be smaller than blockDim.y
 *
 * @tparam R determines the max amount work done by one thread
 *     WorkRatio = 2^R               (number of entries done by one thread)
 * @tparam S - determines the work amount:
 *     maxWorkCount = 2^S - 1        (number of rows could be processed at once)
 *     blockDim.y = 2^(2S - 1 - R)   (number of threads in y dir)
 */
template <typename T, bool WithBias, int S, int R>
__global__ void __launch_bounds__(1024, 1) AggregateXw_kernel(const T* X,
                                                              const T* w,
                                                              const int* indices,
                                                              const int nRows,
                                                              const int nCols,
                                                              const int workOffset,
                                                              const int workCount,
                                                              T* xwPartial,
                                                              const int xwPartialWidth)
{
  extern __shared__ __align__(16) unsigned char shm_bytes[];
  const unsigned int mask = 0xFFFFFFFFU;
  T* shm                  = reinterpret_cast<T*>(shm_bytes);
  const int shmSize       = blockDim.x * blockDim.y;
  const int ypow          = 2 * S - 1 - R;  // blockDim.y = 2 ^ ypow
  const int workRatio     = 1 << R;
  const int ywidth        = min(warpSize, 1 << ypow);
  int shmOff              = 0;
  T xw[workRatio];

  const unsigned int threadId = threadIdx.x + threadIdx.y * blockDim.x;
  const dim3 transIdx(threadId & ((1 << ypow) - 1), threadId >> ypow, 1);
  const int i = threadIdx.y < workCount ? indices[workOffset + threadIdx.y] : -1;

  // workIdx = transIdx.x * workRatio + r = (transIdx.x << R) + r
  // l =  workIdx / (2 ^ (S-1))
  // k = (workIdx % (2 ^ (S-1)) + l) % workCount
  const unsigned int l  = transIdx.x >> (S - 1 - R);
  unsigned int k1       = l + ((transIdx.x << R) & ((1 << (S - 1)) - 1));
  const unsigned int k0 = k1 == l ? workCount : k1;
  if (++k1 >= workCount) k1 -= workCount;
  const unsigned int ti = transIdx.y + blockDim.x * (transIdx.x & (warpSize - 1));
  const int jstep       = gridDim.x * blockDim.x;

#pragma unroll
  for (int r = 0; r < workRatio; r++)
    xw[r] = 0;

  for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < nCols + threadIdx.x;
       j += jstep, shmOff ^= shmSize) {
    T t = 0;
    if (j < nCols) {
      if (threadIdx.y < workCount)
        t = getX<T, WithBias>(X, i, j, nRows, nCols);
      else if (threadIdx.y == workCount)
        t = w[j];
    }
    shm[shmOff + threadId] = t;
    __syncthreads();
    t    = shm[shmOff + ti];
    T xl = __shfl_sync(mask, t, l, ywidth);
    xw[0] += xl * __shfl_sync(mask, t, k0, ywidth);
#pragma unroll
    for (unsigned int k = k1, r = 1; r < workRatio; r++) {
      xw[r] += xl * __shfl_sync(mask, t, k, ywidth);
      if (++k == workCount) k = 0;
    }
  }

#pragma unroll
  for (int r = 0; r < workRatio; r++, shmOff ^= shmSize) {
    shm[shmOff + transIdx.x + (transIdx.y << ypow)] = xw[r];
    __syncthreads();
    T t = shm[shmOff + threadIdx.y + (threadIdx.x << ypow)];
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
      t += __shfl_down_sync(mask, t, offset);
    if (threadIdx.x == 0) xwPartial[xwPartialWidth * (r + (threadIdx.y << R)) + blockIdx.x] = t;
  }
}

// template <typename T, bool WithBias, int S>
// __global__ void __launch_bounds__(1024, 1) AggregateXw_kernel0(const T* X,
//                                                                const T* w,
//                                                                const int* indices,
//                                                                const int nRows,
//                                                                const int nCols,
//                                                                const int workOffset,
//                                                                const int workCount,
//                                                                T* xwPartial,
//                                                                const int xwPartialWidth)
// {
//   extern __shared__ __align__(16) unsigned char shm_bytes[];
//   const unsigned int mask = 0xFFFFFFFFU;
//   T* shm                  = reinterpret_cast<T*>(shm_bytes);
//   const int shmSize       = blockDim.x * blockDim.y;
//   const int ypow          = 2 * S - 1;  // blockDim.y = 2 ^ ypow
//   int shmOff              = 0;
//   T xw                    = 0;

//   const unsigned int threadId = threadIdx.x + threadIdx.y * blockDim.x;
//   const dim3 transIdx(threadId & ((1 << ypow) - 1), threadId >> ypow, 1);
//   const int i = threadIdx.y < workCount ? indices[workOffset + threadIdx.y] : -1;

//   unsigned int l = transIdx.x >> (S - 1);
//   unsigned int k = l + (transIdx.x & ((1 << (S - 1)) - 1));
//   if (k >= workCount)
//     k -= workCount;
//   else if (k == l)
//     k = workCount;
//   l               = transIdx.y + blockDim.x * l;
//   k               = transIdx.y + blockDim.x * k;
//   const int jstep = gridDim.x * blockDim.x;

//   for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < nCols + threadIdx.x;
//        j += jstep, shmOff ^= shmSize) {
//     T t = 0;
//     if (j < nCols) {
//       if (threadIdx.y < workCount)
//         t = getX<T, WithBias>(X, i, j, nRows, nCols);
//       else if (threadIdx.y == workCount)
//         t = w[j];
//     }
//     shm[shmOff + threadId] = t;
//     __syncthreads();
//     xw += shm[shmOff + k] * shm[shmOff + l];
//   }

//   shm[shmOff + transIdx.x + (transIdx.y << ypow)] = xw;
//   __syncthreads();
//   xw = shm[shmOff + threadIdx.y + (threadIdx.x << ypow)];
//   for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
//     xw += __shfl_down_sync(mask, xw, offset);
//   if (threadIdx.x == 0) xwPartial[xwPartialWidth * threadIdx.y + blockIdx.x] = xw;
// }

/**
 * Fills xwPartial with partial dot product of X and w.
 * xwPartial is row-major and has blockDim.y rows and gridDim.x columns.
 *
 * Invariants:
 *
 *   1. dims(xwPartial) = (blockDim.y, xwPartialWidth >= gridDim.x)
 *   2. Components of blockDim must be powers of two
 *   3. blockDim.x <= warpSize
 *   4. workCount <= blockDim.y
 *   5. workCount * (workCount + 1) / 2 <= blockDim.y * workRatio
 */
template <typename T, bool WithBias>
class AggregateXw {
 private:
  void* kernelArgs[9];

  struct KernelAndSR {
    const void* kernel;
    const int s;
    const int r;
    int getBlockHeight() const { return 1 << (2 * s - 1 - r); }
    int getMaxWorkCount() const { return (1 << s) - 1; }
  };

  template <int S>
  struct SelectS {
    template <int R>
    struct SelectR {
      static inline KernelAndSR run(const int r)
      {
        // if (R == 0) return KernelAndSR{(void*)(AggregateXw_kernel0<T, WithBias, S>), S, 0};
        if (r == R) return KernelAndSR{(void*)(AggregateXw_kernel<T, WithBias, S, R>), S, R};
        return SelectR<std::max(R - 1, 0)>::run(r);
      }
    };

    static inline KernelAndSR run(const int h)
    {
      return SelectR<S - 1>::run(std::max(std::min(S - 1, 2 * S - 1 - h), 0));
    }
  };

  static KernelAndSR selectKernel(const int desiredBlockHeight, const int desiredBatchSize)
  {
    int s = 0;
    int h = 0;
    while (desiredBatchSize > (1 << s))
      s++;
    while (desiredBlockHeight > (1 << h))
      h++;
    if (s > h) s = h;
    return FitBlockSize<SelectS, 2, 3, 4, 5>::run(min(s, 5), h);
  }

  static int validateOccupancy(const DeviceInfo& ctx,
                               const void* kernel,
                               const int blockWidth,
                               const int blockHeight,
                               const size_t sharedMemory)
  {
    if (blockWidth > ctx.warpSize || blockHeight > ctx.maxBlockDimY || blockWidth <= 0 ||
        blockHeight <= 0 || blockWidth * blockHeight > ctx.maxThreadsPerBlock)
      return 0;
    return calculateKernelOccupancy(kernel, blockWidth * blockHeight, sharedMemory);
  }

 public:
  const int s;
  const int maxWorkCount;
  const size_t sharedMemory;
  const void* kernel;
  const int occupancy;
  const dim3 blockDim;
  const dim3 gridDim;
  const bool isGPUFullyUsed;

  void reset(const AggregateXw& other)
  {
    std::memcpy((void*)this, (void*)(&other), sizeof(AggregateXw));
  }

  template <typename... Args>
  void setupArgs(Args&... args)
  {
    saveArgs(kernelArgs, args...);
    // This forces the types of Args at compile time, but the actual call
    // should be optimized away.
    return;
    AggregateXw_kernel<T, WithBias, 1, 0><<<gridDim, blockDim>>>(args...);
  }

  void launch(cudaStream_t stream)
  {
    CUDA_CHECK(cudaLaunchKernel(kernel, gridDim, blockDim, kernelArgs, sharedMemory, stream));
  }

  /**
   * Calculate how many rows this kernel can process at once
   * given the block size in the corresponding dimension (blockDim.y).
   *
   * Make each thread do at most `workRatio` dot products.
   * This is ensured by the following equation:
   *
   * workCount * (workCount + 1) / 2 <= blockHeight * workRatio
   */
  static int getMaxWorkCount(const DeviceInfo& ctx, const int workRatio, const int blockHeight)
  {
    auto warpLimit   = ctx.warpSize - 1;
    auto globalLimit = int(0.5 * (sqrt(1.0 + 8.0 * workRatio * blockHeight) - 1.0));
    return max(1, min(min(blockHeight - 1, warpLimit), globalLimit));
  }

  static int getSharedMemory(const DeviceInfo& ctx, const int blockWidth, const int blockHeight)
  {
    return 2 * sizeof(T) * blockWidth * blockHeight;
  }

  AggregateXw(const DeviceInfo& ctx,
              const int blockWidth,
              const KernelAndSR&& ks,
              const int maxGridDim)
    : s(ks.s),
      maxWorkCount(ks.getMaxWorkCount()),
      sharedMemory(getSharedMemory(ctx, blockWidth, ks.getBlockHeight())),
      kernel(ks.kernel),
      occupancy(validateOccupancy(ctx, ks.kernel, blockWidth, ks.getBlockHeight(), sharedMemory)),
      blockDim(blockWidth, ks.getBlockHeight(), 1),
      gridDim(min(maxGridDim, ctx.numSMs * occupancy), 1, 1),
      isGPUFullyUsed(ctx.numSMs * occupancy <= maxGridDim)
  {
  }

  AggregateXw(const DeviceInfo& ctx,
              const int blockWidth,
              const int desiredBlockHeight,
              const int desiredBatchSize,
              const int maxGridDim = std::numeric_limits<int>::max())
    : AggregateXw(ctx, blockWidth, selectKernel(desiredBlockHeight, desiredBatchSize), maxGridDim)
  {
  }
};

/**
 *
 * Invariants:
 *
 *   1. gridDim = (workCount * (workCount + 1) / 2, 1, 1)
 *   2. blockDim = (BlockLen, 1, 1)
 */
template <typename T, int BlockLen, int BlockMult>
__global__ void __launch_bounds__(BlockLen) CalculateGradAndQ_kernel(const T* xwPartial,
                                                                     const T* y,
                                                                     const T* Qii,
                                                                     const T* cWeighted,
                                                                     const T* alpha,
                                                                     const int* indices,
                                                                     const int workOffset,
                                                                     const int workCount,
                                                                     const int s,
                                                                     T* Q,
                                                                     T* g)
{
  __shared__ typename cub::BlockReduce<T, BlockLen>::TempStorage shm;
  const int l = blockIdx.x >> (s - 1);
  int k       = blockIdx.x & ((1 << (s - 1)) - 1);
  if (l >= workCount || (k + l >= workCount && k >= l)) return;
  k += l;
  if (k >= workCount) k -= workCount;

  // aggregating <Xi, w> and <Xi, Xj>
  T t = 0;
#pragma unroll
  for (int i = 0; i < BlockMult; i++)
    t += xwPartial[BlockLen * (BlockMult * blockIdx.x + i) + threadIdx.x];
  T r = cub::BlockReduce<T, BlockLen>(shm).Sum(t);

  if (threadIdx.x == 0) {
    int i = indices[workOffset + l];
    T yi  = y[i];
    if (l == k) {
      Q[(l << s) + l]     = Qii[i];
      g[l]                = r * yi - 1;
      g[(1 << s) + l]     = yi;
      g[(1 << s) * 2 + l] = cWeighted[i];
      g[(1 << s) * 3 + l] = alpha[i];
    } else {
      int j           = indices[workOffset + k];
      t               = r * yi * y[j];
      Q[(l << s) + k] = t;
      Q[(k << s) + l] = t;
    }
  }
}

// template <typename T, int BlockMult>
// __global__ void __launch_bounds__(32) CalculateGradAndQ_kernel32(const T* xwPartial,
//                                                                  const T* y,
//                                                                  const T* Qii,
//                                                                  const T* cWeighted,
//                                                                  const T* alpha,
//                                                                  const int* indices,
//                                                                  const int workOffset,
//                                                                  const int workCount,
//                                                                  const int s,
//                                                                  T* Q,
//                                                                  T* g)
// {
//   const unsigned int mask = 0xFFFFFFFFU;
//   const int l             = blockIdx.x >> (s - 1);
//   int k                   = blockIdx.x & ((1 << (s - 1)) - 1);
//   if (l >= workCount || (k + l >= workCount && k >= l)) return;
//   k += l;
//   if (k >= workCount) k -= workCount;

//   T t = 0;
// #pragma unroll
//   for (int i = 0; i < BlockMult; i++)
//     t += xwPartial[((BlockMult * blockIdx.x + i) << 5) + threadIdx.x];
//   t += __shfl_down_sync(mask, t, 16);
//   t += __shfl_down_sync(mask, t, 8);
//   t += __shfl_down_sync(mask, t, 4);
//   t += __shfl_down_sync(mask, t, 2);
//   t += __shfl_down_sync(mask, t, 1);

//   if (threadIdx.x == 0) {
//     int i = indices[workOffset + l];
//     T yi  = y[i];
//     if (l == k) {
//       Q[(l << s) + l]     = Qii[i];
//       g[l]                = t * yi - 1;
//       g[(1 << s) + l]     = yi;
//       g[(1 << s) * 2 + l] = cWeighted[i];
//       g[(1 << s) * 3 + l] = alpha[i];
//     } else {
//       int j = indices[workOffset + k];
//       t *= yi * y[j];
//       Q[(l << s) + k] = t;
//       Q[(k << s) + l] = t;
//     }
//   }
// }

template <typename T>
class CalculateGradAndQ {
 public:
  const void* kernel;
  const int occupancy;
  const int xwLen;
  const dim3 blockDim;
  dim3 gridDim;

 private:
  // Danger zone! Not type-checked:
  // position of the `workCount` argument in `CalculateGradAndQ_kernel`.
  const int workCountArgIdx = 7;
  const int sArgIdx         = 8;
  void* kernelArgs[11];

  struct KernelAndSize {
    const void* kernel;
    const int blockLen;
    const int blockMult;
  };

  template <int XwLen>
  struct SelectKernel {
    static inline KernelAndSize run(const DeviceInfo& ctx)
    {
      //   if (XwLen == 32) return KernelAndSize{(void*)(CalculateGradAndQ_kernel32<T, 1>), 32, 1};
      //   if (XwLen == 64) return KernelAndSize{(void*)(CalculateGradAndQ_kernel32<T, 2>), 32, 2};
      //   if (XwLen == 96) return KernelAndSize{(void*)(CalculateGradAndQ_kernel32<T, 3>), 32, 3};
      if (XwLen % (ctx.warpSize * 3) == 0)
        return KernelAndSize{(void*)(CalculateGradAndQ_kernel<T, XwLen / 3, 3>), XwLen / 3, 3};
      if (XwLen % (ctx.warpSize * 2) == 0)
        return KernelAndSize{(void*)(CalculateGradAndQ_kernel<T, XwLen / 2, 2>), XwLen / 2, 2};
      return KernelAndSize{(void*)(CalculateGradAndQ_kernel<T, XwLen, 1>), XwLen, 1};
    }
  };

  CalculateGradAndQ(const DeviceInfo& ctx, const KernelAndSize&& ks)
    : kernel(ks.kernel),
      blockDim(ks.blockLen, 1, 1),
      gridDim(0, 1, 1),
      xwLen(ks.blockLen * ks.blockMult),
      occupancy(ks.blockLen > 0 && ks.blockLen <= ctx.maxBlockDimX
                  ? calculateKernelOccupancy(kernel, ks.blockLen, 0)
                  : 0)
  {
  }

 public:
  static int getGridSize(const int workCount) { return workCount * (workCount + 1) / 2; }

  void reset(const CalculateGradAndQ& other)
  {
    std::memcpy((void*)this, (void*)(&other), sizeof(CalculateGradAndQ));
  }

  template <typename... Args>
  void setupArgs(Args&... args)
  {
    saveArgs(kernelArgs, args...);
    // This forces the types of Args at compile time, but the actual call
    // should be optimized away.
    return;
    CalculateGradAndQ_kernel<T, 64, 1><<<gridDim, blockDim>>>(args...);
  }

  void launch(cudaStream_t stream)
  {
    auto workCount = *reinterpret_cast<int*>(kernelArgs[workCountArgIdx]);
    auto s         = *reinterpret_cast<int*>(kernelArgs[sArgIdx]);
    gridDim.x      = workCount << (s - 1);

    CUDA_CHECK(cudaLaunchKernel(kernel, gridDim, blockDim, kernelArgs, 0, stream));
  }

  CalculateGradAndQ(const DeviceInfo& ctx, const int xwLen)
    : CalculateGradAndQ(ctx,
                        FitBlockSize<SelectKernel,
                                     // possible sizes of the grid
                                     64,
                                     96,
                                     128,
                                     160,
                                     192,
                                     224,
                                     256,
                                     320,
                                     384,
                                     512,
                                     1024>::run(xwLen, ctx))
  {
  }
};

/**
 * Calculate the step size for updating w and update alpha[i] as well.
 *
 * Invariants:
 *
 *   1. gridDim = (1, 1, 1)
 *   2. blockDim = (1 << S, 4, 1)
 *          where S in [2..5]
 */
template <typename T, int S>
__global__ void __launch_bounds__((1 << (S + 2)), 1) CalculateDw_kernel(const int* indices,
                                                                        const T* Q,
                                                                        const T* g,
                                                                        const int workOffset,
                                                                        const int workCount,
                                                                        CdState<T>* state,
                                                                        T* alpha,
                                                                        bool* shrinkStencil,
                                                                        T* dw)
{
  __shared__ T shm[(1 << S) + 2];
  const unsigned int mask = 0xFFFFFFFFU;

  auto k = threadIdx.x;
  auto i = k < workCount ? indices[workOffset + k] : 0;
  T gi   = k < workCount ? g[k] : 0;

  T yi   = g[(1 << S) + k];
  T U    = g[(1 << S) * 2 + k];  // cWeighted[i];
  T aOld = g[(1 << S) * 3 + k];  // alpha[i];

  bool shrink = false;
  if (aOld <= 0 && gi > 0) {
    shrink = gi > state->pGMaxGlobal;
    gi     = 0;
  } else if (aOld >= U && gi < 0) {
    shrink = gi < state->pGMinGlobal;
    gi     = 0;
  }

  T qij = 0;
#pragma unroll
  for (int l0 = 0; l0 < (1 << S); l0 += 4) {
    int l = l0 + threadIdx.y;
    T q   = Q[(l << S) + k];
    T gj  = __shfl_sync(mask, gi, l, 1 << S);
    qij += q * gj;
  }
  qij *= gi;

  if (k < workCount && threadIdx.y == 0) shrinkStencil[workOffset + k] = shrink;

  /* aggregate Qij along Y dim
   *
   * Since we know the blockDim.y == 4, it's always enough to do just two
   * __shfl_down_sync reductions, so that the rest fits into one column.
   */
  qij += __shfl_down_sync(mask, qij, 2);
  qij += __shfl_down_sync(mask, qij, 1);
  auto threadId = threadIdx.x + (threadIdx.y << S);
  if ((threadId & 3) == 0) shm[threadId >> 2] = qij;
  __syncthreads();
  qij = threadIdx.y == 0 ? shm[k] : 0;

  T x;
  switch (threadIdx.y) {
    case 0:  // aggregate gT Q g
      x = qij;
      break;
    case 1:  // aggregate ||g||^2
      x = gi * gi;
      break;
    case 2:  // aggregate pGMax
      x = shrink ? std::numeric_limits<T>::infinity() : -gi;
      break;
    case 3:  // aggregate pGMin
      x = shrink ? std::numeric_limits<T>::infinity() : gi;
      break;
    default:  // don't care
      x = 0;
      break;
  }

  if (threadIdx.y < 2) {
    int m = __activemask();
#pragma unroll
    for (int offset = (1 << (S - 1)); offset > 0; offset >>= 1)
      x += __shfl_down_sync(m, x, offset);
  } else {
    int m = __activemask();
#pragma unroll
    for (int offset = (1 << (S - 1)); offset > 0; offset >>= 1)
      x = raft::myMin<T>(x, __shfl_down_sync(m, x, offset));
  }

  if (threadIdx.x == 0) {
    switch (threadIdx.y) {
      case 0:  // aggregate gT Q g
        shm[1 << S] = x;
        break;
      case 1:  // aggregate ||g||^2
        shm[(1 << S) + 1] = x;
        break;
      case 2:  // aggregate pGMax
        if (state->pGMax < -x) state->pGMax = -x;
        break;
      case 3:  // aggregate pGMin
        if (state->pGMin > x) state->pGMin = x;
        break;
      default:  // don't care
        break;
    }
  }
  __syncthreads();
  if (k < workCount && threadIdx.y == 0) {
    T da   = raft::myAbs<T>(gi) > EPS ? gi * shm[(1 << S) + 1] / shm[1 << S] : 0;
    T aNew = raft::myMin<T>(raft::myMax<T>(aOld - da, 0), U);
    da     = aNew - aOld;
    if (da != 0) alpha[i] = aNew;
    dw[k] = da * yi;
  }
}

template <typename T>
class CalculateDw {
 public:
  const int s;
  const void* kernel;
  const int occupancy;
  const dim3 blockDim;
  const dim3 gridDim;

 private:
  void* kernelArgs[9];

  struct KernelAndSize {
    const void* kernel;
    const int s;
  };

  template <int S>
  struct SelectKernel {
    static inline KernelAndSize run()
    {
      return KernelAndSize{(void*)(CalculateDw_kernel<T, S>), S};
    }
  };

  CalculateDw(const DeviceInfo& ctx, const KernelAndSize&& ks)
    : kernel(ks.kernel),
      s(ks.s),
      blockDim(1 << ks.s, 4, 1),
      gridDim(1, 1, 1),
      occupancy(ks.s >= 2 && (1 << ks.s) <= ctx.warpSize
                  ? calculateKernelOccupancy(kernel, 1 << (ks.s + 2), 0)
                  : 0)
  {
  }

 public:
  void reset(const CalculateDw& other) { std::memcpy(this, &other, sizeof(CalculateDw)); }

  template <typename... Args>
  void setupArgs(Args&... args)
  {
    saveArgs(kernelArgs, args...);
    // This forces the types of Args at compile time, but the actual call
    // should be optimized away.
    return;
    CalculateDw_kernel<T, 2><<<gridDim, blockDim>>>(args...);
  }

  void launch(cudaStream_t stream)
  {
    CUDA_CHECK(cudaLaunchKernel(kernel, gridDim, blockDim, kernelArgs, 0, stream));
  }

  CalculateDw(const DeviceInfo& ctx, const int s)
    : CalculateDw(ctx, FitBlockSize<SelectKernel, 2, 3, 4, 5>::run(s))
  {
  }
};

/**
 * Updates primal coefs w using the prepared dw[j] for j in indices[..].
 * The columns of X (and vector w) are processed along `y` direction
 *  in blocks of gridDim.y * blockDim.y.
 * Values of X[i, j] for the fixed j and different i are aggregated along
 * `x` direction, must be equal to blockDim.x (which is power of two).
 *
 * Invariants:
 *
 *   1. blockDim.x == 2^k where k in {0, 1, 2 ... log2 warpSize}
 *       (i.e. blockDim.x must be power of two and not exceed warpSize).
 *   2. gridDim.y * blockDim.y >= nCols
 */
template <typename T, bool WithBias>
__global__ void __launch_bounds__(1024, 1) updateW(const T* X,
                                                   const int* indices,
                                                   const T* dw,
                                                   const int nRows,
                                                   const int nCols,
                                                   const int workOffset,
                                                   const int workCount,
                                                   T* w)
{
  const int j             = threadIdx.y + blockIdx.y * blockDim.y;
  const unsigned int mask = 0xFFFFFFFFU;
  T dwj                   = 0;
  if (j < nCols && threadIdx.x < workCount) {
    const T dwi = dw[threadIdx.x];
    if (dwi != 0) {
      const int i = indices[workOffset + threadIdx.x];
      // warning: uncoalesced access to X
      dwj = getX<T, WithBias>(X, i, j, nRows, nCols) * dwi;
    }
  }

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    dwj += __shfl_down_sync(mask, dwj, offset);

  if (threadIdx.x == 0 && j < nCols && dwj != 0) w[j] += dwj;
}

int closestPow2(const int n)
{
  if (n && !(n & (n - 1))) return n;
  int r = 1;
  for (int m = n; m > 0; m >>= 1)
    r <<= 1;
  return r;
}

// template <typename T, bool WithBias>
// inline T xdot(
//   const T* X, const int nCols, const int i, const int j, T* tmp_dev, cudaStream_t stream)
// {
//   auto f = [] __device__(const T x, const T y) { return x * y; };
//   auto n = WithBias ? nCols - 1 : nCols;
//   raft::linalg::mapThenSumReduce(tmp_dev, nCols - 1, f, stream, X + i * n, X + j * n);
//   T tmp_host;
//   raft::update_host(&tmp_host, tmp_dev, 1, stream);
//   cudaStreamSynchronize(stream);
//   return WithBias ? tmp_host + 1 : tmp_host;
// }

template <typename T, bool WithBias>
int linearSVM_solve_run(const T* X,
                        const T* y,
                        const T* cWeighted,
                        const T* Qii,
                        const int nRows,
                        const int nCols,
                        const T qThreshold,
                        const T tol,
                        const int maxIter,
                        T* alpha,
                        T* w,
                        cudaStream_t stream)
{
  // get some constants
  DeviceInfo ctx;

  // initial guess on the work size
  CalculateGradAndQ<T> calculateGradAndQ(ctx, std::min(ctx.numSMs, closestPow2(nCols >> 5)));
  AggregateXw<T, WithBias> aggregateXw(ctx, 1, 128, 15, calculateGradAndQ.xwLen);

  ASSERT(aggregateXw.occupancy > 0,
         "LinearSVC: cannot launch the kernel (SM block occupancy == 0)");

  // Select the maximum reasonable block size in the column direction,
  // because it's the most efficient way to speedup the solver.
  while (!aggregateXw.isGPUFullyUsed) {
    AggregateXw<T, WithBias> larger(ctx,
                                    aggregateXw.blockDim.x << 1,
                                    aggregateXw.blockDim.y,
                                    aggregateXw.maxWorkCount,
                                    calculateGradAndQ.xwLen);
    if (larger.occupancy == 0) break;
    aggregateXw.reset(larger);
  }

  while ((aggregateXw.blockDim.y >> 2) > aggregateXw.maxWorkCount) {
    AggregateXw<T, WithBias> larger(ctx,
                                    aggregateXw.blockDim.x << 1,
                                    aggregateXw.blockDim.y >> 1,
                                    aggregateXw.maxWorkCount,
                                    calculateGradAndQ.xwLen);
    if (larger.occupancy == 0 || larger.gridDim.x * larger.blockDim.x > nCols) break;
    aggregateXw.reset(larger);
  }

  // add a little bit of work to calculateGradAndQ
  if (aggregateXw.gridDim.x * aggregateXw.blockDim.x * 32 < nCols && aggregateXw.s >= 4 &&
      aggregateXw.occupancy == 1) {
    calculateGradAndQ.reset(CalculateGradAndQ<T>(ctx, ctx.numSMs * 4));
    aggregateXw.reset(AggregateXw<T, WithBias>(ctx,
                                               aggregateXw.blockDim.x,
                                               (aggregateXw.maxWorkCount >> 1) + 1,
                                               aggregateXw.maxWorkCount >> 1,
                                               calculateGradAndQ.xwLen));
  } else if (aggregateXw.gridDim.x * aggregateXw.blockDim.x * 8 < nCols &&
             aggregateXw.blockDim.x >= 2) {
    calculateGradAndQ.reset(CalculateGradAndQ<T>(ctx, ctx.numSMs * 2));
    aggregateXw.reset(AggregateXw<T, WithBias>(ctx,
                                               aggregateXw.blockDim.x >> 1,
                                               aggregateXw.blockDim.y,
                                               aggregateXw.maxWorkCount,
                                               calculateGradAndQ.xwLen));
  } else if (aggregateXw.gridDim.x * aggregateXw.blockDim.x >= 2 * nCols &&
             aggregateXw.blockDim.x >= 2) {
    aggregateXw.reset(AggregateXw<T, WithBias>(ctx,
                                               aggregateXw.blockDim.x >> 1,
                                               aggregateXw.blockDim.y,
                                               aggregateXw.maxWorkCount,
                                               calculateGradAndQ.xwLen));
  }

  CalculateDw<T> calculateDw(ctx, aggregateXw.s);

  auto maxWorkCount2 = closestPow2(aggregateXw.maxWorkCount);

  dim3 updWBS(maxWorkCount2, max(4, 4 * ctx.warpSize / maxWorkCount2), 1);
  dim3 updWGS(1, raft::ceildiv(nCols, (int)updWBS.y), 1);
  // dim3 updWBS(maxWorkCount2, ctx.maxThreadsPerBlock / maxWorkCount2, 1);
  // calculateKernelOccupancy(kernel, blockWidth * blockHeight, sharedMemory)

  dim3 permuteGS(raft::ceildiv(nRows, 256), 1, 1);
  dim3 permuteBS(256, 1, 1);

  printf("CD solver launch (rows = %d, cols = %d):\n", nRows, nCols);
  printf(
    "  aggregateXw   = (%d, %d, 1) * (%d, %d, 1); SM occupancy = %d; "
    "maxWorkCount = %d; workRatio = %d\n",
    aggregateXw.blockDim.x,
    aggregateXw.blockDim.y,
    aggregateXw.gridDim.x,
    aggregateXw.gridDim.y,
    aggregateXw.occupancy,
    aggregateXw.maxWorkCount,
    (1 << (2 * aggregateXw.s - 1)) / aggregateXw.blockDim.y);
  printf("  calculateGrad = (%d, %d, 1) * (%d, %d, 1); SM occupancy = %d\n",
         calculateGradAndQ.blockDim.x,
         calculateGradAndQ.blockDim.y,
         calculateGradAndQ.getGridSize(aggregateXw.maxWorkCount),
         calculateGradAndQ.gridDim.y,
         calculateGradAndQ.occupancy);
  printf("  calculateDw   = (%d, %d, 1) * (%d, %d, 1)\n",
         calculateDw.blockDim.x,
         calculateDw.blockDim.y,
         calculateDw.gridDim.x,
         calculateDw.gridDim.y);
  printf("  updateW       = (%d, %d, 1) * (%d, %d, 1)\n", updWBS.x, updWBS.y, updWGS.x, updWGS.y);

  rmm::device_uvector<CdState<T>> stateBuf(1, stream);
  rmm::device_uvector<int> indicesBuf(nRows * 2, stream);
  rmm::device_uvector<T> xwPartialBuf(calculateGradAndQ.xwLen << (2 * aggregateXw.s - 1), stream);
  rmm::device_uvector<bool> shrinkStencilBuf(nRows, stream);
  rmm::device_uvector<T> dwBuf(1 << (aggregateXw.s + 2), stream);
  rmm::device_uvector<T> QBuf(1 << (2 * aggregateXw.s), stream);
  int* indicesA      = indicesBuf.data();
  int* indicesB      = indicesA + nRows;
  auto state         = stateBuf.data();
  auto xwPartial     = xwPartialBuf.data();
  auto shrinkStencil = shrinkStencilBuf.data();
  auto dw            = dwBuf.data();
  auto Q             = QBuf.data();

  CUDA_CHECK(
    cudaMemsetAsync(Q, 0, aggregateXw.maxWorkCount * aggregateXw.maxWorkCount * sizeof(T), stream));
  CUDA_CHECK(cudaMemsetAsync(
    xwPartial, 0, (calculateGradAndQ.xwLen << (2 * aggregateXw.s - 1)) * sizeof(T), stream));

  size_t shrinkTempStorageBytes = 0;
  cub::DevicePartition::Flagged(NULL,
                                shrinkTempStorageBytes,
                                indicesA,
                                shrinkStencil,
                                indicesB,
                                &(state->iterShrinkCounter),
                                nRows,
                                stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  rmm::device_uvector<uint8_t> shrinkTempStorageBuf(shrinkTempStorageBytes, stream);
  // printf("Allocated shrinkTempStorage bytes: %zd\n", shrinkTempStorageBytes);

  auto shrinkTempStorage = reinterpret_cast<void*>(shrinkTempStorageBuf.data());

  std::mt19937 rng(rand());
  std::vector<int> indicesHost(nRows);
  ML::Solver::initShuffle(indicesHost, rng);
  ML::Solver::shuffle(indicesHost, rng);
  CUDA_CHECK(cudaMemcpyAsync(
    indicesA, indicesHost.data(), sizeof(int) * nRows, cudaMemcpyHostToDevice, stream));

  CdState<T> stateHost;
  CUDA_CHECK(
    cudaMemcpyAsync(state, &stateHost, sizeof(CdState<T>), cudaMemcpyHostToDevice, stream));

  int verbosity = 1;

  T crit, shrankPrev = 0;

  bool maybeConverged = false;
  int itersPassed     = 0;

  // // -------------- DEBUG Q
  // std::vector<T> Q_host(1 << (2 * aggregateXw.s));
  // std::vector<T> Q_check(aggregateXw.maxWorkCount * aggregateXw.maxWorkCount);
  // MLCommon::device_buffer<T> tmp_dev(allocator, stream, 1);
  // std::vector<T> y_host(nRows);
  // CUDA_CHECK(cudaMemcpyAsync(y_host.data(), y, sizeof(T) * nRows, cudaMemcpyDeviceToHost,
  // stream));
  // // ----------------

  int workOffset, workCount;
  aggregateXw.setupArgs(
    X, w, indicesA, nRows, nCols, workOffset, workCount, xwPartial, calculateGradAndQ.xwLen);
  calculateGradAndQ.setupArgs(
    xwPartial, y, Qii, cWeighted, alpha, indicesA, workOffset, workCount, aggregateXw.s, Q, dw);
  calculateDw.setupArgs(indicesA, Q, dw, workOffset, workCount, state, alpha, shrinkStencil, dw);

  for (; itersPassed < maxIter; itersPassed++) {
    ML::PUSH_RANGE("Trace::LinearSVM::solve::outer_iteration");
    stateHost.pGMax = -std::numeric_limits<T>::infinity();
    stateHost.pGMin = std::numeric_limits<T>::infinity();
    CUDA_CHECK(
      cudaMemcpyAsync(state, &stateHost, sizeof(CdState<T>), cudaMemcpyHostToDevice, stream));

    if (shrankPrev > stateHost.shrinkCounter) {
      permuteIndices<T>
        <<<permuteGS, permuteBS, 0, stream>>>(rng(), rng(), nRows, state, indicesA, indicesB);
      // CUDA_CHECK(cudaStreamSynchronize(stream));
      CUDA_CHECK(cudaPeekAtLastError());
      std::swap(indicesA, indicesB);
      // printf("Permuting after unshrink!\n");
    }
    workOffset = stateHost.shrinkCounter;
    for (; workOffset < nRows; workOffset += aggregateXw.maxWorkCount) {
      workCount = min(aggregateXw.maxWorkCount, nRows - workOffset);
      // CUDA_CHECK(cudaStreamSynchronize(stream));

      aggregateXw.launch(stream);
      // CUDA_CHECK(cudaStreamSynchronize(stream));
      calculateGradAndQ.launch(stream);
      // CUDA_CHECK(cudaStreamSynchronize(stream));

      // // -- DEBUG Q
      // CUDA_CHECK(cudaMemcpyAsync(
      //   Q_host.data(), Q, (sizeof(T)) << (2 * aggregateXw.s), cudaMemcpyDeviceToHost, stream));
      // CUDA_CHECK(cudaMemcpyAsync(
      //   indicesHost.data(), indicesA, sizeof(int) * nRows, cudaMemcpyDeviceToHost, stream));
      // CUDA_CHECK(cudaStreamSynchronize(stream));

      // bool same = true;
      // for (int i = 0; i < workCount; i++)
      //   for (int j = 0; j < workCount; j++) {
      //     int ii = indicesHost[workOffset + i];
      //     int jj = indicesHost[workOffset + j];
      //     T a =
      //       xdot<T, WithBias>(X, nCols, ii, jj, tmp_dev.data(), stream) * y_host[ii] *
      //       y_host[jj];
      //     T b                        = Q_host[(i << aggregateXw.s) + j];
      //     Q_check[i * workCount + j] = a;
      //     if (std::abs(a - b) * 2 > max(T(1), std::abs(a) + std::abs(b)) * T(0.1)) same = false;
      //   }
      // if (!same) {
      //   printf("Q incorrect (offset = %d, count = %d)!\n", workOffset, workCount);
      //   printf("Q evaluated:\n");
      //   for (int i = 0; i < workCount; i++) {
      //     for (int j = 0; j < workCount; j++) {
      //       printf(", %f", Q_host[(i << aggregateXw.s) + j]);
      //     }
      //     printf("\n");
      //   }
      //   printf("\nQ check:\n");
      //   for (int i = 0; i < workCount; i++) {
      //     for (int j = 0; j < workCount; j++) {
      //       printf(", %f", Q_check[i * workCount + j]);
      //     }
      //     printf("\n");
      //   }
      //   printf("\nQ diff:\n");
      //   for (int i = 0; i < workCount; i++) {
      //     for (int j = 0; j < workCount; j++) {
      //       printf(", %f", Q_check[i * workCount + j] - Q_host[(i << aggregateXw.s) + j]);
      //     }
      //     printf("\n");
      //   }
      //   printf("\n");
      // }
      // if (!same) return 0;

      calculateDw.launch(stream);
      // CUDA_CHECK(cudaStreamSynchronize(stream));
      // CUDA_CHECK(cudaPeekAtLastError());

      updateW<T, WithBias>
        <<<updWGS, updWBS, 0, stream>>>(X, indicesA, dw, nRows, nCols, workOffset, workCount, w);
      // CUDA_CHECK(cudaStreamSynchronize(stream));
      CUDA_CHECK(cudaPeekAtLastError());
    }

    // CUDA_CHECK(cudaStreamSynchronize(stream));
    // shrink rows
    cub::DevicePartition::Flagged(shrinkTempStorage,
                                  shrinkTempStorageBytes,
                                  indicesA + stateHost.shrinkCounter,
                                  shrinkStencil + stateHost.shrinkCounter,
                                  indicesB + stateHost.shrinkCounter,
                                  &(state->iterShrinkCounter),
                                  nRows - stateHost.shrinkCounter,
                                  stream);
    // CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaPeekAtLastError());

    permuteIndices<T>
      <<<permuteGS, permuteBS, 0, stream>>>(rng(), rng(), nRows, state, indicesB, indicesA);
    // CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaPeekAtLastError());

    shrankPrev = stateHost.shrinkCounter;
    CUDA_CHECK(
      cudaMemcpyAsync(&stateHost, state, sizeof(CdState<T>), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    stateHost.shrinkCounter += stateHost.iterShrinkCounter;
    stateHost.iterShrinkCounter = 0;

    if (shrankPrev < stateHost.shrinkCounter) {
      thrust::device_ptr<bool> ssPtr(shrinkStencil);
      thrust::fill(
        thrust::cuda::par.on(stream), ssPtr + shrankPrev, ssPtr + stateHost.shrinkCounter, true);
    }

    crit = stateHost.shrinkCounter == nRows ? T(0) : stateHost.criterion();

    if (itersPassed % verbosity == 0) {
      printf("--------- iter %d: %f %s %f; shrank = %d;\n",
             itersPassed,
             crit,
             crit > tol ? ">" : "<=",
             tol,
             stateHost.shrinkCounter);
      if (verbosity < 1000 && itersPassed >= verbosity * 10) verbosity *= 10;
    }

    if (crit <= tol) {
      if (stateHost.shrinkCounter == 0 || maybeConverged) {
        ML::POP_RANGE();
        break;
      }
      stateHost.shrinkCounter = 0;
      stateHost.pGMaxGlobal   = std::numeric_limits<T>::infinity();
      stateHost.pGMinGlobal   = -std::numeric_limits<T>::infinity();
      maybeConverged          = true;

    } else {
      stateHost.pGMaxGlobal =
        stateHost.pGMax > 0 ? stateHost.pGMax : std::numeric_limits<T>::infinity();
      stateHost.pGMinGlobal =
        stateHost.pGMin < 0 ? stateHost.pGMin : -std::numeric_limits<T>::infinity();
      maybeConverged = false;
    }
    ML::POP_RANGE();
  }
  printf("--------- iter %d (last): %f %s %f\n", itersPassed, crit, crit > tol ? ">" : "<=", tol);

  // // DEBUG INDICES ------------------------------------------------------------
  // CUDA_CHECK(cudaMemcpyAsync(indicesHost.data(), indicesA, sizeof(T) * nRows,
  //                            cudaMemcpyDeviceToHost, stream));
  // CUDA_CHECK(cudaStreamSynchronize(stream));
  // int not_found_indices = 0;
  // for (int i = 0; i < nRows; i++) {
  //   bool found = false;
  //   for (int j = 0; j < nRows; j++) {
  //     if (indicesHost[j] == i) {
  //       found = true;
  //       break;
  //     }
  //   }
  //   if (!found) {
  //     not_found_indices++;
  //     // printf("Index not found: %d!\n", i);
  //   }
  // }
  // if (not_found_indices > 0) {
  //   printf("Not found %d indices out of %d!\n", not_found_indices, nRows);
  // }

  return itersPassed;
}

/**
 * Solve linear SVM using coordinate descent in dual space.
 *
 * @param model [out] non-zero support vectors and their weights (dual coefs).
 */
template <typename T, bool WithBias>
void linearSVM_solve_mbbias(const raft::handle_t& handle,
                            const LinearSVMParams& params,
                            const T* Xt,
                            const std::size_t nRows,
                            const std::size_t nCols,
                            const T* y0,
                            const T* sampleWeight,
                            LinearSVMModel<T>& model,
                            cudaStream_t stream)
{
  rmm::device_uvector<T> Xbuf((WithBias ? nCols - 1 : nCols) * nRows, stream);
  auto X = Xbuf.data();
  raft::linalg::transpose<T>(handle, (T*)Xt, X, nRows, WithBias ? nCols - 1 : nCols, stream);

  rmm::device_uvector<T> ybuf(nRows, stream);
  auto y = ybuf.data();
  raft::linalg::unaryOp(
    y, y0, nRows, [] __device__(const T yi) -> T { return yi > 0 ? T(1) : T(-1); }, stream);

  bool isL1 = params.loss == LinearSVMParams::HINGE;
  ASSERT(params.C > 0, "Penalty term must be greater than zero");

  // dual coefs
  rmm::device_uvector<T> alphaBuf(nRows, stream);
  auto alpha = alphaBuf.data();
  CUDA_CHECK(cudaMemsetAsync(alpha, 0, nRows * sizeof(T), stream));

  // primal coefs
  rmm::device_uvector<T> wBuf(nCols, stream);
  auto w = wBuf.data();
  CUDA_CHECK(cudaMemsetAsync(w, 0, nCols * sizeof(T), stream));
  // CUDA_CHECK(cudaStreamSynchronize(stream));

  // The weighted penalty
  rmm::device_uvector<T> cWeightedBuf(nRows, stream);
  auto cWeighted = cWeightedBuf.data();
  if (sampleWeight == nullptr) {
    thrust::device_ptr<T> c_ptr(cWeighted);
    thrust::fill(thrust::cuda::par.on(stream), c_ptr, c_ptr + nRows, params.C);
  } else {
    T C = params.C;
    raft::linalg::unaryOp(
      cWeighted, sampleWeight, nRows, [C] __device__(T sw) { return C * sw; }, stream);
  }

  // The inverse values of the diagonal of the matrix Q
  // (used to compute new alpha based on projected gradient).
  T qThreshold = 1e-10;
  rmm::device_uvector<T> QiiBuf(nRows, stream);
  auto Qii = QiiBuf.data();
  {
    dim3 bs(128, 1, 1);
    dim3 gs(nRows, 1, 1);
    precomputeQii<T, 128, WithBias>
      <<<gs, bs, 0, stream>>>(Qii, X, cWeighted, nRows, nCols, T(isL1 ? 0.0 : 0.5));
    CUDA_CHECK(cudaPeekAtLastError());

    thrust::device_ptr<T> qPtr(Qii);
    T qmax = thrust::reduce(qPtr, qPtr + nRows, 0, thrust::maximum<T>());
    if (qmax >= 1) qThreshold = raft::myMax<T>(qThreshold, 1 / qmax);
    printf("qThreshold = %.20f\n", qThreshold);
  }

  int maxIter = params.max_iter;
  if (maxIter < 0) maxIter = 1000;

  auto itersPassed = linearSVM_solve_run<T, WithBias>(X,
                                                      y,
                                                      cWeighted,
                                                      Qii,
                                                      nRows,
                                                      nCols,
                                                      qThreshold,
                                                      params.grad_tol,
                                                      maxIter,
                                                      alpha,
                                                      model.w,
                                                      stream);

  CUML_LOG_DEBUG("Dual Coordinate Descent is done in %d iters out of %d.\n", itersPassed, maxIter);
}

};  // namespace SVC_CD_Impl

template <typename T>
void linearSVC_CD_fit(const raft::handle_t& handle,
                      const LinearSVMParams& params,
                      const T* X,
                      const std::size_t nRows,
                      const std::size_t nCols,
                      const T* y,
                      const T* sampleWeight,
                      LinearSVMModel<T>& model,
                      cudaStream_t stream)
{
  ML::PUSH_RANGE("Trace::LinearSVMModel::linearSVC_CD_fit");
  if (params.fit_intercept)
    SVC_CD_Impl::linearSVM_solve_mbbias<T, true>(
      handle, params, X, nRows, nCols + 1, y, sampleWeight, model, stream);
  else
    SVC_CD_Impl::linearSVM_solve_mbbias<T, false>(
      handle, params, X, nRows, nCols, y, sampleWeight, model, stream);
  ML::POP_RANGE();
}

};  // namespace SVM
};  // namespace ML
