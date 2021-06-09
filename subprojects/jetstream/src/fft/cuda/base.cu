#include "jetstream/fft/cuda.hpp"

namespace Jetstream::FFT {

__device__ inline float clamp(float x, float a, float b) {
  return max(a, min(b, x));
}

__device__ inline float scale(float x, float min, float max) {
  return (x - min) / (max - min);
}

__device__ inline float amplt(cuFloatComplex x, int n) {
  return 20 * log10(cuCabsf(x) / n);
}

__device__ inline int shift(int i, uint n) {
  return (i + (n / 2) - 1) % n;
}

__global__ void post_process(const cufftComplex* c, float* r,
    const float min, const float max, const uint n){
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
      i += blockDim.x * gridDim.x) {
    r[i] = amplt(c[shift(i, n)], n);
    r[i] = scale(r[i], min, max);
    r[i] = clamp(r[i], 0.0, 1.0);
  }
}

CUDA::CUDA(Config& c) : Generic(c) {
  fft_len = in.buf.size() * sizeof(float) * 2;
  cudaMalloc(&fft_dptr, fft_len);

  out_len = in.buf.size() * sizeof(float);
  cudaMallocManaged(&out_dptr, out_len);
  out.buf = nonstd::span<float>{out_dptr, in.buf.size()};

  cudaMemAdvise(out_dptr, out_len, cudaMemAdviseSetPreferredLocation, 0);
  cudaMemAdvise(out_dptr, out_len, cudaMemAdviseSetReadMostly, cudaCpuDeviceId);

  cufftPlan1d(&plan, in.buf.size(), CUFFT_C2C, 1);
}

CUDA::~CUDA() {
  cufftDestroy(plan);
  cudaFree(fft_dptr);
  cudaFree(out_dptr);
}

Result CUDA::underlyingCompute() {
  cudaMemcpy(fft_dptr, in.buf.data(), fft_len, cudaMemcpyHostToDevice);
  cufftExecC2C(plan, fft_dptr, fft_dptr, CUFFT_FORWARD);
  post_process<<<CB(in.buf.size()), kNumBlockThreads>>>
    (fft_dptr, out_dptr, cfg.min_db, cfg.max_db, in.buf.size());
  cudaDeviceSynchronize();
  cudaMemPrefetchAsync(out_dptr, out_len, cudaCpuDeviceId);

  return Result::SUCCESS;
}

Result CUDA::underlyingPresent() {
  return Result::SUCCESS;
}

} // namespace Jetstream::FFT
