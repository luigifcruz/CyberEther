#include "jetstream/fft/cuda.hpp"

namespace Jetstream {

static __device__ inline float clamp(const float x, const float a, float b) {
    return (x < a) ? a : (b < x) ? b : x;
}

static __device__ inline float scale(const float x, const float min, const float max) {
    return (x - min) / (max - min);
}

static __device__ inline float amplt(const cuFloatComplex x, const int n) {
    return 20 * log10(cuCabsf(x) / n);
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

    float tmp;
    for (int i = threadID; i < n; i += numThreads) {
        tmp = amplt(c[i], n);
        tmp = scale(tmp, min, max);
        tmp = clamp(tmp, 0.0f, 1.0f);

        r[i] = tmp;
    }
}

FFT::CUDA::CUDA(const Config & c) : FFT(c) {
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

    cufftPlan1d(&plan, in.buf.size(), CUFFT_C2C, 1);
    cufftSetStream(plan, stream);
}

FFT::CUDA::~CUDA() {
    cudaHostUnregister(in.buf.data());
    cufftDestroy(plan);
    cudaFree(fft_dptr);
    cudaFree(out_dptr);
    cudaStreamDestroy(stream);
}

Result FFT::CUDA::underlyingCompute() {
    DEBUG_PUSH("compute_fft");

    int N = in.buf.size();
    int threads = 32;
    int blocks = (N + threads - 1) / threads;
    auto [min, max] = cfg.amplitude;

    CUDA_CHECK(cudaMemcpyAsync(fft_dptr, in.buf.data(), fft_len, cudaMemcpyHostToDevice, stream));
    pre<<<blocks, threads, 0, stream>>>(fft_dptr, win_dptr, N);
    cufftExecC2C(plan, fft_dptr, fft_dptr, CUFFT_FORWARD);
    post<<<blocks, threads, 0, stream>>>(fft_dptr, out_dptr, min, max, N);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    DEBUG_POP();
    return Result::SUCCESS;
}

Result FFT::CUDA::underlyingPresent() {
    return Result::SUCCESS;
}

} // namespace Jetstream
