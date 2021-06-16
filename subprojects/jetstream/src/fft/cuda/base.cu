#include "jetstream/fft/cuda.hpp"

namespace Jetstream::FFT {

static __device__ inline float clamp(float x, float a, float b) {
    return max(a, min(b, x));
}

static __device__ inline float scale(float x, float min, float max) {
    return (x - min) / (max - min);
}

static __device__ inline float amplt(cuFloatComplex x, int n) {
    return 20 * log10(cuCabsf(x) / n);
}

static __device__ inline int shift(int i, uint n) {
    return (i + (n / 2) - 1) % n;
}

static __global__ void pre(cufftComplex* c, const cufftComplex* win, const uint n){
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = threadID; i < n; i += numThreads) {
        c[i] = cuCmulf(c[i], win[i]);
    }
}

static __global__ void post(const cufftComplex* c, float* r,
    const float min, const float max, const uint n){
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = threadID; i < n; i += numThreads) {
        float tmp;

        tmp = amplt(c[shift(i, n)], n);
        tmp = scale(tmp, min, max);
        tmp = clamp(tmp, 0.0, 1.0);

        r[i] = tmp;
    }
}

CUDA::CUDA(Config& c) : Generic(c) {
    CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK_THROW(cudaHostRegister(in.buf.data(), in.buf.size() * sizeof(in.buf[0]),
            cudaHostRegisterReadOnly));

    fft_len = in.buf.size() * sizeof(in.buf[0]);
    CUDA_CHECK_THROW(cudaMalloc(&fft_dptr, fft_len));

    win_len = in.buf.size() * sizeof(in.buf[0]);
    CUDA_CHECK_THROW(cudaMalloc(&win_dptr, win_len));
    CUDA_CHECK_THROW(cudaMemcpy(win_dptr, window.data(), win_len, cudaMemcpyHostToDevice));

    out_len = in.buf.size() * sizeof(float);
    CUDA_CHECK_THROW(cudaMallocManaged(&out_dptr, out_len));
    out.buf = nonstd::span<float>{out_dptr, in.buf.size()};

    CUDA_CHECK_THROW(cufftPlan1d(&plan, in.buf.size(), CUFFT_C2C, 1));
    CUDA_CHECK_THROW(cufftSetStream(plan, stream));
}

CUDA::~CUDA() {
    CUDA_CHECK_THROW(cudaHostUnregister(in.buf.data()));
    CUDA_CHECK_THROW(cufftDestroy(plan));
    CUDA_CHECK_THROW(cudaFree(fft_dptr));
    CUDA_CHECK_THROW(cudaFree(out_dptr));
    CUDA_CHECK_THROW(cudaStreamDestroy(stream));
}

Result CUDA::underlyingCompute() {
    int N = in.buf.size();
    int threads = 32;
    int blocks = (N + threads - 1) / threads;

    CUDA_CHECK(cudaMemcpyAsync(fft_dptr, in.buf.data(), fft_len, cudaMemcpyHostToDevice, stream));
    pre<<<blocks, threads, 0, stream>>>(fft_dptr, win_dptr, N);
    CUDA_CHECK(cufftExecC2C(plan, fft_dptr, fft_dptr, CUFFT_FORWARD));
    post<<<blocks, threads, 0, stream>>>(fft_dptr, out_dptr, cfg.min_db, cfg.max_db, N);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return Result::SUCCESS;
}

Result CUDA::underlyingPresent() {
    return Result::SUCCESS;
}

} // namespace Jetstream::FFT
