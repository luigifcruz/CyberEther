#include "jstcore/fft/cuda/base.hpp"

namespace Jetstream::FFT {

CUDA::CUDA(const Config& config, const Input& input) : Generic(config, input) {
    JST_CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    if ((input.in.location & Locale::CPU) == Locale::CPU &&
        (input.in.location & Locale::CUDA) != Locale::CUDA) {
        JST_CUDA_CHECK_THROW(cudaHostRegister(input.in.buf.data(), input.in.buf.size() * sizeof(input.in.buf[0]),
                cudaHostRegisterReadOnly));
    }

    fft_len = input.in.buf.size() * sizeof(input.in.buf[0]);
    JST_CUDA_CHECK_THROW(cudaMalloc(&fft_dptr, fft_len));

    win_len = input.in.buf.size() * sizeof(input.in.buf[0]);
    JST_CUDA_CHECK_THROW(cudaMalloc(&win_dptr, win_len));
    JST_CUDA_CHECK_THROW(cudaMemcpy(win_dptr, window.data(), win_len, cudaMemcpyHostToDevice));

    out_len = input.in.buf.size() * sizeof(float);
    JST_CUDA_CHECK_THROW(cudaMallocManaged(&out_dptr, out_len));
    out.location = Locale::CUDA | Locale::CPU;
    out.buf = VF32{out_dptr, input.in.buf.size()};

    cufftPlan1d(&plan, input.in.buf.size(), CUFFT_C2C, 1);
    cufftSetStream(plan, stream);
}

FFT::CUDA::~CUDA() {
    cudaHostUnregister(input.in.buf.data());
    cufftDestroy(plan);
    cudaFree(fft_dptr);
    cudaFree(out_dptr);
    cudaStreamDestroy(stream);
}

Result CUDA::underlyingCompute() {
    int N = input.in.buf.size();
    int threads = 32;
    int blocks = (N + threads - 1) / threads;
    auto [min, max] = config.amplitude;

    JST_CUDA_CHECK(cudaMemcpyAsync(fft_dptr, input.in.buf.data(), fft_len, cudaMemcpyDeviceToDevice, stream));
    Kernel::PreFFT(blocks, threads, stream, fft_dptr, win_dptr, N);
    cufftExecC2C(plan, fft_dptr, fft_dptr, CUFFT_FORWARD);
    Kernel::PostFFT(blocks, threads, stream, fft_dptr, out_dptr, min, max, N);
    JST_CUDA_CHECK(cudaStreamSynchronize(stream));

    return Result::SUCCESS;
}

} // namespace Jetstream::FFT
